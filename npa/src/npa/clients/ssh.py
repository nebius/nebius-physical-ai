"""SSH client for executing commands on the VM."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import sys
import threading
import uuid
from typing import Callable

import paramiko

from npa.clients.config import SSHConfig
from npa.clients.env import render_shell_env_file, validate_env_name


class SSHError(Exception):
    pass


NPA_DEBUG_ENV_VAR = "NPA_DEBUG"

# How many trailing stderr lines to surface by default. Install scripts emit the
# actual failure (missing token, 403, CUDA mismatch) near the end, so the tail is
# almost always the useful part.
_STDERR_TAIL_LINES = 20
_COMMAND_SUMMARY_MAX = 200


def _npa_debug_enabled() -> bool:
    return os.environ.get(NPA_DEBUG_ENV_VAR, "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


def _summarize_command(command: str) -> str:
    """Return a short, single-line description of a possibly huge remote command."""

    for raw in command.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if len(line) > _COMMAND_SUMMARY_MAX:
            return line[:_COMMAND_SUMMARY_MAX] + "…"
        return line
    stripped = command.strip()
    if len(stripped) > _COMMAND_SUMMARY_MAX:
        return stripped[:_COMMAND_SUMMARY_MAX] + "…"
    return stripped


def format_remote_failure(
    command: str,
    code: int,
    stderr: str,
    *,
    label: str | None = None,
) -> str:
    """Build a compact SSH failure message.

    By default this surfaces the step label (or a one-line command summary), the
    exit code, and the tail of stderr — never the full multi-hundred-line install
    script, which both buries the real error and can leak inlined secrets (e.g.
    docker-login tokens). Set ``NPA_DEBUG=1`` to include the full command and the
    complete stderr for deep debugging.
    """

    what = label or _summarize_command(command)
    stderr = (stderr or "").strip()
    lines = [f"Command failed (exit {code}): {what}"]

    if _npa_debug_enabled():
        lines.append(f"command:\n{command}")
        lines.append(f"stderr:\n{stderr}" if stderr else "stderr: <empty>")
        return "\n".join(lines)

    stderr_lines = stderr.splitlines()
    if stderr_lines:
        tail = stderr_lines[-_STDERR_TAIL_LINES:]
        truncated = len(stderr_lines) > len(tail)
        header = (
            f"stderr (last {len(tail)} of {len(stderr_lines)} lines):"
            if truncated
            else "stderr:"
        )
        lines.append(header + "\n" + "\n".join(tail))
    else:
        lines.append("stderr: <empty>")
    lines.append(f"Set {NPA_DEBUG_ENV_VAR}=1 for the full command and output.")
    return "\n".join(lines)


class SSHClient:
    def __init__(self, config: SSHConfig) -> None:
        self._config = config

    def _connect(self) -> paramiko.SSHClient:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        key_path = os.path.expanduser(self._config.key_path)
        try:
            client.connect(
                hostname=self._config.host,
                username=self._config.user,
                key_filename=key_path,
                timeout=15,
                look_for_keys=False,
            )
        except Exception as exc:
            raise SSHError(
                f"SSH connection to {self._config.user}@{self._config.host} failed: {exc}\n"
                f"Check NPA_SSH_HOST, NPA_SSH_USER, NPA_SSH_KEY or ~/.npa/config.yaml"
            ) from exc
        return client

    def _token_env_content(self) -> str:
        env: dict[str, str] = {}
        for key, value in sorted(self._config.tokens.items()):
            try:
                validate_env_name(key)
            except ValueError:
                raise SSHError(f"Invalid token environment variable name: {key!r}")
            env[key] = value
        return render_shell_env_file(env, export=True)

    def _write_token_env_file(self, client: paramiko.SSHClient) -> str:
        remote_path = f"/tmp/.npa-env-{uuid.uuid4().hex}"
        sftp = client.open_sftp()
        try:
            with sftp.open(remote_path, "w") as remote_file:
                remote_file.write(self._token_env_content())
            sftp.chmod(remote_path, 0o600)
        finally:
            sftp.close()
        return remote_path

    def _command_with_tokens(self, command: str, env_file: str | None = None) -> str:
        if not self._config.tokens:
            return command
        if not env_file:
            raise SSHError("Token env file was not prepared")
        env_file_q = shlex.quote(env_file)
        script = (
            "set -a\n"
            f". {env_file_q}\n"
            "set +a\n"
            f"rm -f {env_file_q}\n"
            f"{command}"
        )
        return f"bash -lc {shlex.quote(script)}"

    def run(
        self,
        command: str,
        *,
        stream: bool = False,
        on_stdout: Callable[[str], None] | None = None,
    ) -> tuple[int, str, str]:
        """Execute a command over SSH.

        Args:
            command: Shell command to run on the remote host.
            stream: If True, forward stdout to the local terminal in real time.
            on_stdout: Optional callback for each stdout line (called regardless of stream).

        Returns:
            (exit_code, stdout_text, stderr_text)
        """
        client = self._connect()
        try:
            token_env_file = self._write_token_env_file(client) if self._config.tokens else None
            transport = client.get_transport()
            if transport is None:
                raise SSHError("SSH transport is not available")
            channel = transport.open_session()
            channel.set_combine_stderr(False)
            channel.exec_command(self._command_with_tokens(command, token_env_file))

            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []

            def _read_stderr() -> None:
                while True:
                    data = channel.recv_stderr(4096)
                    if not data:
                        break
                    stderr_chunks.append(data.decode("utf-8", errors="replace"))

            stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
            stderr_thread.start()

            buf = ""
            while True:
                data = channel.recv(4096)
                if not data:
                    break
                text = data.decode("utf-8", errors="replace")
                stdout_chunks.append(text)

                if stream or on_stdout:
                    buf += text
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        if stream:
                            sys.stdout.write(line + "\n")
                            sys.stdout.flush()
                        if on_stdout:
                            on_stdout(line)

            stderr_thread.join(timeout=5)
            exit_code = channel.recv_exit_status()
            return exit_code, "".join(stdout_chunks), "".join(stderr_chunks)
        finally:
            client.close()

    def run_or_raise(
        self, command: str, *, label: str | None = None, **kwargs
    ) -> tuple[int, str, str]:
        """Run a command; raise SSHError on non-zero exit.

        On failure the error is compact by default (step ``label`` or a one-line
        command summary, the exit code, and the stderr tail). Pass ``label`` to
        describe the step for long install scripts. Set ``NPA_DEBUG=1`` to get the
        full command and complete stderr.
        """
        code, out, err = self.run(command, **kwargs)
        if code != 0:
            raise SSHError(format_remote_failure(command, code, err, label=label))
        return code, out, err

    def download_file(self, remote_path: str, local_path: str) -> str:
        """Download a single file from the VM over SFTP."""
        client = self._connect()
        sftp = None
        try:
            local = Path(local_path).expanduser()
            local.parent.mkdir(parents=True, exist_ok=True)
            sftp = client.open_sftp()
            sftp.get(remote_path, str(local))
            return str(local)
        except Exception as exc:
            raise SSHError(f"SFTP download failed: {remote_path} -> {local_path}: {exc}") from exc
        finally:
            if sftp is not None:
                sftp.close()
            client.close()

    def upload_file(self, local_path: str, remote_path: str) -> str:
        """Upload a single file to the VM over SFTP."""
        client = self._connect()
        sftp = None
        try:
            local = Path(local_path).expanduser()
            remote_parent = str(Path(remote_path).parent)
            self.run(f"mkdir -p {shlex.quote(remote_parent)}")
            sftp = client.open_sftp()
            sftp.put(str(local), remote_path)
            return remote_path
        except Exception as exc:
            raise SSHError(f"SFTP upload failed: {local_path} -> {remote_path}: {exc}") from exc
        finally:
            if sftp is not None:
                sftp.close()
            client.close()

    def upload_directory(self, local_dir: str, remote_dir: str) -> str:
        """Upload a local directory to the VM over SFTP."""
        local_root = Path(local_dir).expanduser()
        self.run(f"mkdir -p {shlex.quote(remote_dir)}")
        for path in local_root.rglob("*"):
            if path.is_file():
                rel = path.relative_to(local_root)
                self.upload_file(str(path), str(Path(remote_dir) / rel))
        return remote_dir
