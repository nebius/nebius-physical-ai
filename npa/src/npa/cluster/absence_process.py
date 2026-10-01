"""Bound owned read-only process groups without treating failure as absence."""

from contextlib import contextmanager
from contextvars import ContextVar
import math
import os
import signal
import subprocess
import tempfile
import threading

DEFAULT_VERIFICATION_TIMEOUT_SECONDS = 120.0
_IN_READER = ContextVar("absence_reader_group", default=False)
_TIMEOUT = ContextVar(
    "absence_read_timeout", default=DEFAULT_VERIFICATION_TIMEOUT_SECONDS
)


class VerificationReadUnavailable(RuntimeError):
    """An external read did not complete and cannot authorize reconciliation."""


@contextmanager
def verification_deadline(seconds: float):
    """Set the per-read deadline; zero disables it, cancellation remains active.

    Args:
        seconds: Finite nonnegative seconds for each external verification read.
    Yields:
        None, while this context's calls use the selected deadline.
    Raises:
        ValueError: The deadline is not a finite nonnegative number.
    """
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError("Verification deadline must be a finite nonnegative number")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Verification deadline must be a finite nonnegative number")
    token = _TIMEOUT.set(seconds)
    try:
        yield
    finally:
        _TIMEOUT.reset(token)


@contextmanager
def reader_process_group():
    """Keep nested reader children inside the parent verifier's owned group.

    Yields:
        None. The outer read controls its complete descendant group's deadline.
    """
    token = _IN_READER.set(True)
    try:
        with verification_deadline(0):
            yield
    finally:
        _IN_READER.reset(token)


@contextmanager
def _cancellation():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def cancelled(_signum, _frame):
        raise VerificationReadUnavailable("Verification read cancelled")

    signal.signal(signal.SIGTERM, cancelled)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _stop_group(process):
    try:
        if _IN_READER.get():
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        process.wait()


def run_read(
    argv: list[str], *, env: dict | None = None, text: bool = False, check: bool = False
):
    """Collect an owned process group, killing descendants on failure or cancel.

    Args:
        argv: Read-only command; never passed through a shell.
        env: Explicit authority environment where required by the caller.
        text: Decode output with the subprocess text encoding.
        check: Raise a sanitized CalledProcessError on nonzero exit.
    Returns:
        CompletedProcess with retained output, only after the child was joined.
    Raises:
        VerificationReadUnavailable: The deadline elapsed; no absence is inferred.
        subprocess.CalledProcessError: A checked read returned a failure status.
        OSError: The child could not start or its group could not be stopped.
        KeyboardInterrupt: Cancellation, after killing the owned process group.
    """
    with (
        _cancellation(),
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        process = subprocess.Popen(
            argv,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=not _IN_READER.get(),
        )
        try:
            process.wait(timeout=_TIMEOUT.get() or None)
        except subprocess.TimeoutExpired:
            raise VerificationReadUnavailable(
                "Verification read deadline elapsed"
            ) from None
        finally:
            _stop_group(process)
        stdout.seek(0)
        stderr.seek(0)
        result = subprocess.CompletedProcess(
            argv, process.returncode, stdout.read(), stderr.read()
        )
    if text:
        result.stdout, result.stderr = result.stdout.decode(), result.stderr.decode()
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, ["verification-read"])
    return result
