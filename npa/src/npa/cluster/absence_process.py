"""Bound owned read-only process groups without treating failure as absence."""

from contextlib import closing, contextmanager
from contextvars import ContextVar
import math
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time

DEFAULT_VERIFICATION_TIMEOUT_SECONDS = 120.0
_READER_PARENT = "NPA_ABSENCE_READER_PARENT_PID"
_DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
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
    """Keep nested children inside a run_read-launched reader session only.

    Yields:
        None. The outer read controls its complete descendant group's deadline.
    """
    if (
        os.environ.get(_READER_PARENT) != str(os.getppid())
        or os.getsid(0) != os.getpid()
        or os.getpgrp() != os.getpid()
    ):
        raise VerificationReadUnavailable("Reader must own a run_read child session")
    token = _IN_READER.set(True)
    try:
        with verification_deadline(0):
            yield
    finally:
        _IN_READER.reset(token)


@contextmanager
def _cancellation():
    # Read-only commands temporarily map SIGTERM to a sanitized failure; restore
    # any outer CLI handler after joining. SIGINT keeps Python's interruption.
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


def _darwin_group_only_zombies(pid):
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "pid=,pgid=,stat="],
            capture_output=True,
            text=True,
            timeout=1,
            check=True,
        )
        rows = [line.split() for line in result.stdout.splitlines()]
        owned = [
            (int(child), state) for child, group, state in rows if int(group) == pid
        ]
        return any(child == pid for child, _ in owned) and all(
            state.startswith("Z") for _, state in owned
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _stop_group(process, *, exited=False):
    # Never signal a recycled identifier after any caller has reaped the leader.
    if process.returncode is not None:
        return
    if not _child_is_reserved(process.pid):
        raise VerificationReadUnavailable("Verification child ownership unavailable")
    try:
        if _IN_READER.get():
            os.kill(process.pid, signal.SIGKILL)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Darwin refuses signalling a group containing only credential-less
        # zombies. Confirm complete native membership while PID is reserved.
        if not (
            sys.platform == "darwin"
            and exited
            and _darwin_group_only_zombies(process.pid)
        ):
            raise
    process.wait()


def _child_is_reserved(pid):
    # An external SIGCHLD handler can reap without updating Popen.returncode.
    # Recheck immediately before signalling; observation errors never grant
    # permission to signal an identifier whose reservation we cannot establish.
    if hasattr(os, "waitid"):
        try:
            os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            return True
        except ChildProcessError:
            return False
        except OSError as exc:
            raise VerificationReadUnavailable(
                "Verification child ownership unavailable"
            ) from exc
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["/bin/ps", "-p", str(pid), "-o", "ppid=,pgid="],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
            )
            parent, group = map(int, result.stdout.split())
            expected_group = os.getpgrp() if _IN_READER.get() else pid
            return result.returncode == 0 and (parent, group) == (
                os.getpid(), expected_group
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            return False
    return False


@contextmanager
def _exit_observer(pid):
    # Keep the zombie leader reserved until all owned descendants are signalled.
    if hasattr(os, "waitid"):
        flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
        yield lambda: os.waitid(os.P_PID, pid, flags) is not None
    elif hasattr(select, "kqueue"):
        with closing(select.kqueue()) as queue:
            event = select.kevent(
                pid,
                filter=select.KQ_FILTER_PROC,
                flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                fflags=select.KQ_NOTE_EXIT,
            )
            queue.control([event], 0, 0)
            yield lambda: bool(queue.control(None, 1, 0))
    else:
        raise VerificationReadUnavailable("Non-reaping process observation unavailable")


def _output_limit():
    raw = os.environ.get("NPA_ABSENCE_MAX_OUTPUT_BYTES", str(_DEFAULT_MAX_OUTPUT_BYTES))
    if not raw.isdecimal():
        raise ValueError("Verification output limit must be nonnegative bytes")
    return int(raw)


def _check_output(streams, limit):
    if limit and any(os.fstat(stream.fileno()).st_size > limit for stream in streams):
        raise VerificationReadUnavailable("Verification output exceeds byte limit")


def _wait_for_read(process, streams, limit):
    timeout = _TIMEOUT.get()
    deadline = time.monotonic() + timeout if timeout else None
    with _exit_observer(process.pid) as exited:
        while True:
            _check_output(streams, limit)
            if exited():
                return
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                raise VerificationReadUnavailable("Verification read deadline elapsed")
            time.sleep(min(0.02, remaining) if remaining is not None else 0.02)


def _capture(argv, env, streams, limit):
    child_env = dict(os.environ if env is None else env)
    if not _IN_READER.get():
        child_env[_READER_PARENT] = str(os.getpid())
    process = subprocess.Popen(
        argv,
        env=child_env,
        stdout=streams[0],
        stderr=streams[1],
        start_new_session=not _IN_READER.get(),
    )
    exited = False
    try:
        _wait_for_read(process, streams, limit)
        exited = True
    finally:
        _stop_group(process, exited=exited)
    _check_output(streams, limit)
    output = []
    for stream in streams:
        stream.seek(0)
        data = stream.read(limit + 1 if limit else -1)
        if limit and len(data) > limit:
            raise VerificationReadUnavailable("Verification output exceeds byte limit")
        output.append(data)
    return subprocess.CompletedProcess(argv, process.returncode, *output)


def run_read(
    argv: list[str], *, env: dict | None = None, text: bool = False, check: bool = False
):
    """Collect an owned read group while its leader remains reserved for cleanup.

    Args:
        argv: Read-only command; never passed through a shell.
        env: Explicit authority environment where required by the caller.
        text: Decode captured UTF-8 output.
        check: Raise a sanitized CalledProcessError on nonzero exit.
    Returns:
        CompletedProcess only after signalling owned descendants and joining.
        NPA_ABSENCE_MAX_OUTPUT_BYTES limits each stream (64 MiB default; 0 off).
    Raises:
        VerificationReadUnavailable: Deadline, SIGTERM or output limit interrupted
            verification. No failure establishes absence.
        subprocess.CalledProcessError: A checked read returned a failure status.
        OSError: The child or its owned group could not be controlled.
        KeyboardInterrupt: SIGINT, after stopping the owned process group.
        ValueError: The configured output limit is invalid.
    """
    limit = _output_limit()
    with (
        _cancellation(),
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        result = _capture(argv, env, (stdout, stderr), limit)
    if text:
        result.stdout, result.stderr = result.stdout.decode(), result.stderr.decode()
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, ["verification-read"])
    return result
