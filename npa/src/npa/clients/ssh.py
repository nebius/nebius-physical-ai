"""SSH client for executing commands on the VM."""

from __future__ import annotations

import os
from pathlib import Path
import math
import shlex
import sys
import threading
import time
import uuid
from typing import Callable

import paramiko

from npa.clients.config import SSHConfig
from npa.clients.env import render_shell_env_file, validate_env_name


class SSHError(Exception):
    pass


class SSHTimeoutError(SSHError):
    """An aggregate SSH connection/command deadline expired."""


NPA_DEBUG_ENV_VAR = "NPA_DEBUG"

# How many trailing stderr lines to surface by default. Install scripts emit the
# actual failure (missing token, 403, CUDA mismatch) near the end, so the tail is
# almost always the useful part.
_STDERR_TAIL_LINES = 20


def _npa_debug_enabled() -> bool:
    return os.environ.get(NPA_DEBUG_ENV_VAR, "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


def format_remote_failure(
    code: int,
    stderr: str,
    *,
    label: str | None = None,
) -> str:
    """Build a compact SSH failure message.

    The remote command is *never* included: install scripts frequently carry
    credentials inline (S3 keys, HF/NGC/Token Factory tokens, docker-login
    registry passwords), and this exception text is surfaced to terminals,
    scrollback, CI logs, and agent transcripts. The message carries only the
    optional step ``label``, the exit code, and the stderr tail — the part that
    actually names the failure (missing token, 403, CUDA mismatch).

    ``NPA_DEBUG=1`` lifts *only* the stderr truncation (full remote stderr); it
    never reveals the command.
    """

    stderr = (stderr or "").strip()
    suffix = f": {label}" if label else ""
    lines = [f"Command failed (exit {code}){suffix}"]

    if not stderr:
        lines.append("stderr: <empty>")
        return "\n".join(lines)

    stderr_lines = stderr.splitlines()
    if _npa_debug_enabled() or len(stderr_lines) <= _STDERR_TAIL_LINES:
        lines.append("stderr:\n" + stderr)
        return "\n".join(lines)

    tail = stderr_lines[-_STDERR_TAIL_LINES:]
    lines.append(
        f"stderr (last {len(tail)} of {len(stderr_lines)} lines):\n" + "\n".join(tail)
    )
    lines.append(f"Set {NPA_DEBUG_ENV_VAR}=1 for the full remote stderr.")
    return "\n".join(lines)


class SSHClient:
    def __init__(self, config: SSHConfig) -> None:
        self._config = config

    def _connect(
        self,
        *,
        timeout_seconds: float | None = None,
        client: paramiko.SSHClient | None = None,
    ) -> paramiko.SSHClient:
        client = client or paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        key_path = os.path.expanduser(self._config.key_path)
        connect_options: dict[str, object] = {
            "hostname": self._config.host,
            "username": self._config.user,
            "key_filename": key_path,
            "timeout": 15,
            "look_for_keys": False,
        }
        if timeout_seconds is not None:
            connect_options.update(
                timeout=min(15.0, timeout_seconds),
                banner_timeout=timeout_seconds,
                auth_timeout=timeout_seconds,
                channel_timeout=timeout_seconds,
            )
        try:
            client.connect(**connect_options)
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
        script = f"set -a\n. {env_file_q}\nset +a\nrm -f {env_file_q}\n{command}"
        return f"bash -lc {shlex.quote(script)}"

    def run(
        self,
        command: str,
        *,
        stream: bool = False,
        on_stdout: Callable[[str], None] | None = None,
        timeout: float | None = None,
    ) -> tuple[int, str, str]:
        """Execute a command over SSH.

        Args:
            command: Shell command to run on the remote host.
            stream: If True, forward stdout to the local terminal in real time.
            on_stdout: Optional callback for each stdout line (called regardless of stream).
            timeout: Optional aggregate connection-and-command deadline in seconds.

        Returns:
            (exit_code, stdout_text, stderr_text)
        """
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("SSH command timeout must be finite and greater than 0")
        deadline = time.monotonic() + timeout if timeout is not None else None
        deadline_expired = threading.Event()
        client = paramiko.SSHClient() if timeout is not None else None
        watchdog: threading.Timer | None = None
        if timeout is not None and client is not None:
            watchdog_client = client

            def abort() -> None:
                deadline_expired.set()
                watchdog_client.close()

            watchdog = threading.Timer(timeout, abort)
            watchdog.daemon = True
            watchdog.start()
        try:
            client = self._connect(
                timeout_seconds=timeout,
                client=client,
            )
            if deadline_expired.is_set():
                raise SSHTimeoutError(f"SSH command timed out after {timeout:g}s")
            token_env_file = (
                self._write_token_env_file(client) if self._config.tokens else None
            )
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

            stderr_join_timeout = 5.0
            if deadline is not None:
                stderr_join_timeout = min(
                    stderr_join_timeout, max(0.0, deadline - time.monotonic())
                )
            stderr_thread.join(timeout=stderr_join_timeout)
            if deadline_expired.is_set() or (
                deadline is not None and time.monotonic() >= deadline
            ):
                raise SSHTimeoutError(f"SSH command timed out after {timeout:g}s")
            exit_code = channel.recv_exit_status()
            if deadline_expired.is_set():
                raise SSHTimeoutError(f"SSH command timed out after {timeout:g}s")
            return exit_code, "".join(stdout_chunks), "".join(stderr_chunks)
        except BaseException as exc:
            if deadline_expired.is_set():
                raise SSHTimeoutError(
                    f"SSH command timed out after {timeout:g}s"
                ) from exc
            raise
        finally:
            if watchdog is not None:
                watchdog.cancel()
            if client is not None:
                client.close()

    def run_or_raise(
        self, command: str, *, label: str | None = None, **kwargs
    ) -> tuple[int, str, str]:
        """Run a command; raise SSHError on non-zero exit.

        The full command is deliberately omitted from the error — remote install
        scripts frequently carry credentials (S3 keys, HF/NGC/Token Factory
        tokens, registry passwords) inline, and this exception text is surfaced
        to terminals, scrollback, CI logs, and agent transcripts. Only the
        optional step ``label``, the exit code, and the remote stderr are
        reported. Pass ``label`` to name the step for long install scripts; set
        ``NPA_DEBUG=1`` to get the full remote stderr (never the command).
        """
        code, out, err = self.run(command, **kwargs)
        if code != 0:
            raise SSHError(format_remote_failure(code, err, label=label))
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
            raise SSHError(
                f"SFTP download failed: {remote_path} -> {local_path}: {exc}"
            ) from exc
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
            raise SSHError(
                f"SFTP upload failed: {local_path} -> {remote_path}: {exc}"
            ) from exc
        finally:
            if sftp is not None:
                sftp.close()
            client.close()

    def upload_private_text(self, content: str, remote_path: str) -> str:
        """Create a remote owner-only file without putting content in argv."""

        client = self._connect()
        sftp = None
        try:
            sftp = client.open_sftp()
            # Paramiko requires ``w`` in addition to ``x``: unlike Python's
            # built-in open(), bare ``x`` sets CREATE|EXCL but not WRITE.
            with sftp.open(remote_path, "wx") as remote_file:
                sftp.chmod(remote_path, 0o600)
                remote_file.write(content)
                remote_file.flush()
            return remote_path
        except Exception as exc:
            raise SSHError(
                f"Private SFTP upload failed for {remote_path}: {exc}"
            ) from exc
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
