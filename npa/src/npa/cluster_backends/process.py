"""Backend-neutral subprocess helpers for cluster lifecycle implementations."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, TextIO


class BackendCommandError(RuntimeError):
    """A backend command could not be started or completed successfully."""

    def __init__(
        self, message: str, *, stdout: str | None = None, stderr: str | None = None
    ) -> None:
        self.stdout = _redact(stdout)
        self.stderr = _redact(stderr)
        super().__init__(message)


def _sensitive_values(env: dict[str, str] | None) -> tuple[str, ...]:
    if not env:
        return ()
    return tuple(
        sorted(
            {
                env[key]
                for key in (
                    "TF_VAR_iam_token",
                    "NEBIUS_IAM_TOKEN",
                    "NPA_NEBIUS_IAM_TOKEN",
                )
                if env.get(key)
            },
            key=len,
            reverse=True,
        )
    )


def _redact(
    value: str | bytes | None, *, sensitive_values: tuple[str, ...] = ()
) -> str:
    """Redact provider output before it crosses the shared process boundary."""

    if not value:
        return ""
    from npa.clients.nebius import redact_nebius_output

    safe = (
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    )
    for secret in sensitive_values:
        safe = safe.replace(secret, "<redacted>")
    return redact_nebius_output(safe)


def _command_text(args: list[str]) -> str:
    return _redact(" ".join(args))


def _stream_redactor(sensitive_values: tuple[str, ...]) -> Callable[[str], str]:
    """Return a stateful line redactor, including multiline private-key blocks."""

    in_private_key = False

    def redact_line(line: str) -> str:
        nonlocal in_private_key
        if in_private_key:
            if "-----END " in line and "PRIVATE KEY-----" in line:
                in_private_key = False
            return ""
        if "-----BEGIN " in line and "PRIVATE KEY-----" in line:
            if not ("-----END " in line and "PRIVATE KEY-----" in line):
                in_private_key = True
            return "<redacted-private-key>\n"
        return _redact(line, sensitive_values=sensitive_values)

    return redact_line


def require_bin(binary: str) -> str:
    if not binary:
        raise BackendCommandError("Required executable not found: <empty>")
    resolved = shutil.which(binary)
    if resolved:
        return resolved
    if Path(binary).is_file() and os.access(binary, os.X_OK):
        return binary
    raise BackendCommandError(f"Required executable not found: {binary}")


def run_capture(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    sensitive_values = _sensitive_values(env)
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            input=input_text,
            timeout=timeout,
            check=False,
        )
    except OSError as exc:
        raise BackendCommandError(
            f"Could not start executable {args[0]}: {exc.strerror or type(exc).__name__}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise BackendCommandError(
            f"Command timed out after {timeout} seconds: {_command_text(args)}",
            stdout=_redact(exc.stdout, sensitive_values=sensitive_values),
            stderr=_redact(exc.stderr, sensitive_values=sensitive_values),
        ) from exc
    result = subprocess.CompletedProcess(
        args=completed.args,
        returncode=completed.returncode,
        stdout=_redact(completed.stdout, sensitive_values=sensitive_values),
        stderr=_redact(completed.stderr, sensitive_values=sensitive_values),
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise BackendCommandError(
            f"Command failed ({result.returncode}): {_command_text(args)}{suffix}"
        )
    return result


def _stop_process(process: subprocess.Popen[str]) -> None:
    for signal_name, grace in (("interrupt", 30.0), ("terminate", 10.0)):
        if process.poll() is not None:
            return
        try:
            if signal_name == "interrupt":
                process.send_signal(signal.SIGINT)
            else:
                process.terminate()
        except OSError:
            return
        try:
            process.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue
    if process.poll() is None:
        process.kill()


def run_stream(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    cancel: Callable[[], str | None] | None = None,
    capture_output: bool = False,
    output_sink: TextIO | None = None,
) -> subprocess.CompletedProcess[str]:
    sensitive_values = _sensitive_values(env)
    if cancel is None:
        try:
            process = subprocess.Popen(
                args,
                cwd=cwd,
                env=env,
                text=True,
                bufsize=1,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise BackendCommandError(
                f"Could not start executable {args[0]}: "
                f"{exc.strerror or type(exc).__name__}"
            ) from exc
        captured_stdout: list[str] = []
        captured_stderr: list[str] = []
        write_lock = threading.Lock()

        def drain(pipe, captured: list[str], target) -> None:
            if pipe is None:
                return
            redact_line = _stream_redactor(sensitive_values)
            for line in iter(pipe.readline, ""):
                safe = redact_line(line)
                if capture_output:
                    captured.append(safe)
                with write_lock:
                    target.write(safe)
                    target.flush()
            pipe.close()

        readers = [
            threading.Thread(
                target=drain,
                args=(process.stdout, captured_stdout, output_sink or sys.stdout),
                daemon=True,
            ),
            threading.Thread(
                target=drain,
                args=(process.stderr, captured_stderr, output_sink or sys.stderr),
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _stop_process(process)
            raise BackendCommandError(
                f"Command timed out after {timeout} seconds: {_command_text(args)}"
            ) from exc
        finally:
            for reader in readers:
                reader.join(timeout=5)
        result = subprocess.CompletedProcess(
            args=args,
            returncode=returncode,
            stdout="".join(captured_stdout),
            stderr="".join(captured_stderr),
        )
        if result.returncode != 0:
            detail = "\n".join(
                part for part in (result.stderr or "", result.stdout or "") if part
            ).strip()
            suffix = f": {detail[-3000:]}" if detail else ""
            raise BackendCommandError(
                f"Command failed ({result.returncode}): {_command_text(args)}{suffix}"
            )
        return result

    reason = ""
    try:
        process = subprocess.Popen(
            args,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise BackendCommandError(
            f"Could not start executable {args[0]}: {exc.strerror or type(exc).__name__}"
        ) from exc
    captured_stdout: list[str] = []
    captured_stderr: list[str] = []
    write_lock = threading.Lock()

    def drain_cancel(pipe, captured: list[str], target) -> None:
        if pipe is None:
            return
        redact_line = _stream_redactor(sensitive_values)
        for line in iter(pipe.readline, ""):
            safe = redact_line(line)
            if capture_output:
                captured.append(safe)
            with write_lock:
                target.write(safe)
                target.flush()
        pipe.close()

    readers = [
        threading.Thread(
            target=drain_cancel,
            args=(process.stdout, captured_stdout, output_sink or sys.stdout),
            daemon=True,
        ),
        threading.Thread(
            target=drain_cancel,
            args=(process.stderr, captured_stderr, output_sink or sys.stderr),
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()
    deadline = None if timeout is None else time.monotonic() + timeout
    try:
        while process.poll() is None:
            if not reason:
                reason = cancel() or ""
                if reason:
                    _stop_process(process)
            if deadline is not None and time.monotonic() >= deadline:
                _stop_process(process)
                raise BackendCommandError(
                    f"Command timed out after {timeout} seconds: {_command_text(args)}"
                )
            time.sleep(1.0)
    except KeyboardInterrupt:
        _stop_process(process)
        raise
    finally:
        for reader in readers:
            reader.join(timeout=5)
    if reason:
        raise BackendCommandError(f"Cancelled `{' '.join(args[:2])}`: {reason}")
    returncode = process.returncode or 0
    if returncode != 0:
        raise BackendCommandError(
            f"Command failed ({returncode}): {_command_text(args)}"
        )
    return subprocess.CompletedProcess(
        args,
        returncode,
        "".join(captured_stdout),
        "".join(captured_stderr),
    )


def terraform_env(
    nebius_bin: str,
    *,
    profile: str = "",
    capture_runner: Callable[..., subprocess.CompletedProcess[str]] = run_capture,
    timeout: int | float | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    reuse = env.get("NPA_REUSE_IAM_TOKEN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if reuse and env.get("TF_VAR_iam_token"):
        return env
    env.pop("TF_VAR_iam_token", None)
    env.pop("NEBIUS_IAM_TOKEN", None)
    argv = [nebius_bin, *(["--profile", profile] if profile else [])]
    capture_kwargs: dict[str, object] = {"env": env}
    if timeout is not None:
        capture_kwargs["timeout"] = timeout
    token = capture_runner(
        [*argv, "iam", "get-access-token"], **capture_kwargs
    ).stdout.strip()
    env["TF_VAR_iam_token"] = token
    env["NEBIUS_IAM_TOKEN"] = token
    return env
