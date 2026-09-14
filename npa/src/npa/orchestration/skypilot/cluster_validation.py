"""Own the local API and recoverable smoke intent of cluster validation sessions."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import fcntl
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterator
import uuid

from npa.orchestration.skypilot import _bin, local_api


@dataclass
class _ValidationSession:
    scope: Path
    record: dict[str, Any]
    credentials_checked: bool = False

    @property
    def pending_smoke(self) -> str:
        return self.record["smoke_name"]

    def require_target(self, kubeconfig: Path, context: str) -> None:
        if self.record["kubeconfig"] != str(kubeconfig.resolve()) or self.record["context"] != context:
            raise local_api.IsolatedApiError("cluster validation session targets a different kubeconfig or context")

    def begin_smoke(self, name: str) -> None:
        if self.pending_smoke:
            raise local_api.IsolatedApiError("finish the recorded owned validation smoke before another launch")
        self.record.update(phase="smoke_pending", smoke_name=name)
        self._save()

    def smoke_removed(self) -> None:
        self.record.update(phase="checking", smoke_name="")
        self._save()

    def finish(self) -> None:
        if self.pending_smoke:
            return
        _stop_session_api(self.scope)
        self.record["phase"] = "complete"
        self._save()

    def _save(self) -> None:
        local_api._write(self.scope / "session.json", self.record)


_CURRENT: ContextVar[_ValidationSession | None] = ContextVar("cluster_validation_session", default=None)


def current_validation_session() -> _ValidationSession | None:
    """Return the validation session owned by this execution context.

    Args:
        None.
    Returns:
        The active session, or None outside validation.
    Raises:
        None.
    """
    return _CURRENT.get()


def _private_directory(path: Path) -> None:
    if path.is_symlink():
        raise local_api.IsolatedApiError("cluster validation state must not be a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def _session_base(kubeconfig: Path, context: str) -> Path:
    if not context.strip():
        raise local_api.IsolatedApiError("cluster validation requires an exact Kubernetes context")
    root = _bin.resolve_isolated_config_dir() or _bin.CONFIG_PATH.parent
    identity = hashlib.sha256(f"{kubeconfig.resolve()}\0{context}".encode()).hexdigest()[:24]
    return root.expanduser().absolute() / "cluster-validation" / identity


@contextmanager
def _session_lock(base: Path) -> Iterator[None]:
    local_api._require_linux_host()
    _private_directory(base)
    descriptor = os.open(base / "session.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as lock:
        os.fchmod(lock.fileno(), 0o600)
        # Separate open file descriptions make flock serialize threads as well
        # as processes. Nested calls reuse _CURRENT instead of taking this lock.
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def _read_session(base: Path) -> _ValidationSession | None:
    pointer = base / "current-session.json"
    if pointer.is_symlink():
        raise local_api.IsolatedApiError("cluster validation recovery state must not be a symlink")
    if not pointer.exists():
        return None
    try:
        selected = json.loads(pointer.read_text())
        name = selected["session"]
        if selected["schema_version"] != 1 or not re.fullmatch(r"session-[a-f0-9]{32}", name):
            raise ValueError
        scope = base / name
        path = scope / "session.json"
        if scope.is_symlink() or path.is_symlink():
            raise ValueError
        record = json.loads(path.read_text())
        if record["schema_version"] != 1 or record["scope"] != str(scope):
            raise ValueError
        if record["phase"] not in {"checking", "smoke_pending", "complete"}:
            raise ValueError
        if bool(record["smoke_name"]) != (record["phase"] == "smoke_pending"):
            raise ValueError
        if record["smoke_name"] and not re.fullmatch(r"[a-z0-9-]+-sky-smoke", record["smoke_name"]):
            raise ValueError
        return _ValidationSession(scope, record)
    except (OSError, ValueError, KeyError, TypeError):
        raise local_api.IsolatedApiError("cluster validation recovery state is invalid; preserve its private session records") from None


def _stop_session_api(scope: Path) -> None:
    if (scope / "local-api" / "daemon.json").exists():
        local_api.stop_isolated_api(scope)


def _open_session(base: Path, kubeconfig: Path, context: str) -> _ValidationSession:
    previous = _read_session(base)
    if previous is not None:
        previous.require_target(kubeconfig, context)
        if previous.pending_smoke:
            return previous
        # Intent precedes launch. A checking/complete session cannot own an
        # unrecorded smoke, so retire its API before accepting changed identity.
        if previous.record["phase"] != "complete":
            _finish_session(previous, None)
    scope = base / f"session-{uuid.uuid4().hex}"
    _private_directory(scope)
    session = _ValidationSession(scope, {
        "schema_version": 1, "scope": str(scope), "kubeconfig": str(kubeconfig.resolve()),
        "context": context, "phase": "checking", "smoke_name": "",
    })
    session._save()
    local_api._write(base / "current-session.json", {"schema_version": 1, "session": scope.name})
    return session


def _finish_session(session: _ValidationSession, failure: BaseException | None) -> None:
    recovery = (
        "Preserve cluster validation state and rerun the original cluster/provision "
        "command with the same NPA configuration, isolated root, kubeconfig, context, "
        f"SkyPilot binary and credentials. Private session: {session.scope / 'session.json'}"
    )
    try:
        session.finish()
    except Exception as exc:
        if failure is None:
            raise local_api.IsolatedApiError(f"Owned validation API cleanup is incomplete. {recovery}") from exc
        print(f"Owned validation API cleanup is incomplete. {recovery}", file=sys.stderr)
    if session.pending_smoke:
        message = f"Owned validation smoke removal remains unverified. {recovery}"
        if failure is None:
            raise local_api.IsolatedApiError(message)
        print(message, file=sys.stderr)


@contextmanager
def cluster_validation_session(kubeconfig: Path, context: str) -> Iterator[_ValidationSession]:
    """Serialize a durable owned API session across validation and smoke cleanup.

    Args:
        kubeconfig: Selected Kubernetes configuration file.
        context: Exact context being validated.
    Returns:
        A context manager yielding the recoverable validation session.
    Raises:
        IsolatedApiError: Ownership or pending recovery cannot be verified.
    """
    selected = Path(kubeconfig).expanduser().resolve()
    active = _CURRENT.get()
    if active is not None:
        active.require_target(selected, context)
        yield active
        return
    base = _session_base(selected, context)
    with _session_lock(base):
        session = _open_session(base, selected, context)
        token = _CURRENT.set(session)
        failure = None
        try:
            yield session
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                _finish_session(session, failure)
            finally:
                _CURRENT.reset(token)


def validation_session(function):
    """Give direct cluster check/smoke calls the same owned session boundary.

    Args:
        function: Callable whose first arguments are kubeconfig and context.
    Returns:
        The callable wrapped in a nestable validation session.
    Raises:
        IsolatedApiError: Session ownership or recovery is inconsistent.
    """
    @wraps(function)
    def wrapped(kubeconfig_path, context, *args, **kwargs):
        with cluster_validation_session(kubeconfig_path, context):
            return function(kubeconfig_path, context, *args, **kwargs)
    return wrapped
