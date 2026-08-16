"""Backend-neutral subprocess helpers for cluster lifecycle implementations."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Callable


class BackendCommandError(RuntimeError):
    """A backend command could not be started or completed successfully."""


def require_bin(binary: str) -> str:
    resolved = shutil.which(binary)
    if resolved:
        return resolved
    if Path(binary).exists():
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
    result = subprocess.run(
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
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise BackendCommandError(
            f"Command failed ({result.returncode}): {' '.join(args)}{suffix}"
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
) -> subprocess.CompletedProcess[str]:
    if cancel is None:
        result = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            text=True,
            timeout=timeout,
            check=False,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
        )
        if result.returncode != 0:
            detail = "\n".join(
                part for part in (result.stderr or "", result.stdout or "") if part
            ).strip()
            suffix = f": {detail[-3000:]}" if detail else ""
            raise BackendCommandError(
                f"Command failed ({result.returncode}): {' '.join(args)}{suffix}"
            )
        return result

    reason = ""
    process = subprocess.Popen(args, cwd=cwd, env=env, text=True)
    deadline = None if timeout is None else time.monotonic() + timeout
    while process.poll() is None:
        if not reason:
            reason = cancel() or ""
            if reason:
                _stop_process(process)
        if deadline is not None and time.monotonic() >= deadline:
            _stop_process(process)
            raise subprocess.TimeoutExpired(args, timeout or 0)
        time.sleep(1.0)
    if reason:
        raise BackendCommandError(f"Cancelled `{' '.join(args[:2])}`: {reason}")
    returncode = process.returncode or 0
    if returncode != 0:
        raise BackendCommandError(f"Command failed ({returncode}): {' '.join(args)}")
    return subprocess.CompletedProcess(args, returncode, "", "")


def terraform_env(nebius_bin: str, *, profile: str = "") -> dict[str, str]:
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
    token = run_capture([*argv, "iam", "get-access-token"], env=env).stdout.strip()
    env["TF_VAR_iam_token"] = token
    env["NEBIUS_IAM_TOKEN"] = token
    return env
