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

import yaml

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


def _session_base(kubeconfig: Path, context: str, isolated_config_dir: Path | None = None) -> Path:
    if not context.strip():
        raise local_api.IsolatedApiError("cluster validation requires an exact Kubernetes context")
    root = _bin.resolve_isolated_config_dir(isolated_config_dir) or _bin.CONFIG_PATH.parent
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
        if not isinstance(record.get("project_alias", ""), str):
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


def _open_session(base: Path, kubeconfig: Path, context: str, project_alias: str) -> _ValidationSession:
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
        "context": context, "phase": "checking", "smoke_name": "", "project_alias": project_alias,
    })
    session._save()
    local_api._write(base / "current-session.json", {"schema_version": 1, "session": scope.name})
    return session


def _finish_session(session: _ValidationSession, failure: BaseException | None) -> None:
    recovery = (
        "Preserve cluster validation state and rerun the original validation command "
        "(cluster/provision, targeted verify or discovery) with the same NPA configuration, isolated root, kubeconfig, context, "
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


def _require_session_binding(session, kubeconfig, context, isolated_config_dir, project_alias, check_only):
    session.require_target(kubeconfig, context)
    if isolated_config_dir is not None and session.scope.parent != _session_base(kubeconfig, context, isolated_config_dir):
        raise local_api.IsolatedApiError("cluster validation session targets a different isolated state root")
    if project_alias is not None and session.record.get("project_alias", "") != project_alias:
        raise local_api.IsolatedApiError("cluster validation session targets a different selected project")
    if check_only and session.pending_smoke:
        raise local_api.IsolatedApiError(
            "Recorded validation smoke requires recovery before standalone verification/discovery. "
            "Rerun the original cluster/provision command with its unchanged configuration, "
            f"context, SkyPilot binary and credentials. Private session: {session.scope / 'session.json'}"
        )


@contextmanager
def cluster_validation_session(
    kubeconfig: Path, context: str, *, isolated_config_dir: Path | None = None,
    project_alias: str | None = None, check_only: bool = False,
) -> Iterator[_ValidationSession]:
    """Serialize a durable owned API session across validation and smoke cleanup.

    Args:
        kubeconfig: Selected Kubernetes configuration file.
        context: Exact context being validated.
        isolated_config_dir: Explicit state root, preceding environment/config.
        project_alias: Optional explicit project binding; nested bindings must agree.
        check_only: Refuse pending smoke before any standalone check or discovery.
    Returns:
        A context manager yielding the recoverable validation session.
    Raises:
        IsolatedApiError: Ownership or pending recovery cannot be verified.
    """
    selected = Path(kubeconfig).expanduser().resolve()
    active = _CURRENT.get()
    if active is not None:
        _require_session_binding(active, selected, context, isolated_config_dir, project_alias, check_only)
        yield active
        return
    base = _session_base(selected, context, isolated_config_dir)
    with _session_lock(base):
        alias = project_alias if project_alias is not None else os.environ.get("NPA_SKYPILOT_PROJECT", "")
        session = _open_session(base, selected, context, alias)
        expected_alias = alias if "project_alias" in session.record else project_alias
        _require_session_binding(session, selected, context, isolated_config_dir, expected_alias, check_only)
        with _session_lifetime(session):
            yield session


@contextmanager
def _session_lifetime(session):
    token = _CURRENT.set(session)
    failure = None
    try:
        yield
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            _finish_session(session, failure)
        finally:
            _CURRENT.reset(token)


def _named_kube_entry(document, section, name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError
    entries = document.get(section, [])
    if not isinstance(entries, list):
        raise ValueError
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("name") == name]
    if len(matches) != 1 or not isinstance(matches[0].get(section.rstrip("s")), dict):
        raise ValueError
    return matches[0][section.rstrip("s")]


def resolve_validation_target(kubeconfig: Path | str | None, context: str = "") -> tuple[Path, str]:
    """Resolve one file and validate the selected context's actual references.

    Args:
        kubeconfig: Explicit file, otherwise KUBECONFIG or the default kubeconfig.
        context: Exact selected context, otherwise that file's current-context.
    Returns:
        The resolved kubeconfig path and unique context name.
    Raises:
        IsolatedApiError: The file, context, cluster, or optional user is ambiguous or invalid.
    """
    raw = os.fspath(kubeconfig or os.environ.get("KUBECONFIG") or Path.home() / ".kube/config")
    if os.pathsep in raw:
        raise local_api.IsolatedApiError("Standalone validation requires one kubeconfig file; select a single KUBECONFIG or --kubeconfig")
    try:
        selected = Path(raw).expanduser().resolve(strict=True)
        document = yaml.safe_load(selected.read_text())
        if not isinstance(document, dict):
            raise ValueError
        exact = str(context or document.get("current-context") or "").strip()
        if not exact:
            raise ValueError
        target = _named_kube_entry(document, "contexts", exact)
        cluster = _named_kube_entry(document, "clusters", target.get("cluster"))
        if not isinstance(cluster.get("server"), str) or not cluster["server"].strip():
            raise ValueError
        if target.get("user"):
            _named_kube_entry(document, "users", target["user"])
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        raise local_api.IsolatedApiError(
            "Kubeconfig or exact context is missing, ambiguous, or invalid; select one existing file "
            "with a unique context and valid cluster/user references"
        ) from exc
    return selected, exact


def resolve_validation_project(project: str, context: str) -> str | None:
    """Validate an optional selected project against local cluster identity.

    Args:
        project: Explicit project alias, or empty for existing optional semantics.
        context: Exact selected Kubernetes context.
    Returns:
        The explicit alias, or None when no project was supplied.
    Raises:
        IsolatedApiError: Configured project and local cluster identity do not agree.
    """
    from npa.clients.config import resolve_environment
    from npa.cluster.state import load_cluster_state

    alias = str(project or "").strip()
    if not alias:
        return None
    environment = resolve_environment(alias)
    cluster = load_cluster_state(context)
    if environment is None or not environment.project_id or cluster is None:
        raise local_api.IsolatedApiError(
            "Selected project/context has no complete local identity; configure the project and "
            "adopt the exact cluster with npa cluster kubeconfig before discovery"
        )
    if cluster.project_id != environment.project_id:
        raise local_api.IsolatedApiError("Selected project does not match the exact context's local cluster identity")
    return alias


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
