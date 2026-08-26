"""Crash-safe, secret-free provisioning transactions.

Provisioning state deliberately lives outside Terraform working directories and
project configuration.  The journal is operational evidence used to resume an
interrupted apply or to destroy exactly the resources that apply owned.  Terminal
operations are retained and mirrored into the existing teardown-receipt store as
small audit receipts; ordinary cleanup does not remove either location.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
import threading
import time
from typing import Any


SCHEMA_VERSION = "npa.provisioning.operation.v2"
PROJECT_LEASE_SCHEMA_VERSION = "npa.project.lifecycle-lease.v1"
TERMINAL_PHASES = frozenset({"committed", "rolled-back", "destroyed"})
RECOVERABLE_PHASES = frozenset(
    {
        "prepared",
        "mutating",
        "resource-created",
        "state-durable",
        "recovery-required",
        "rolling-back",
        "rollback-incomplete",
    }
)
_PHASE_ORDER = {
    "prepared": 0,
    "mutating": 1,
    "resource-created": 2,
    "state-durable": 3,
    "committed": 4,
}
_SECRET_KEY_MARKERS = (
    "access_key",
    "api_key",
    "iam_token",
    "password",
    "presigned",
    "private_key",
    "secret",
    "token",
)
_SECRET_METADATA_KEYS = frozenset({"secret_fields"})
_SECRET_PATTERNS = (
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bhf_[A-Za-z0-9_=-]{8,}\b"),
    re.compile(r"\bnvapi-[A-Za-z0-9_=-]{8,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_=-]{20,}\b"),
    re.compile(
        r"(?i)\b(?:access[_-]?key|api[_-]?key|bearer|credential|password|"
        r"private[_-]?key|secret|token)\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(r"(?i)https?://[^\s\"']+[?&](?:x-amz-|signature=)[^\s\"']*"),
)


class OperationJournalError(RuntimeError):
    """The operation journal could not be used safely."""


class OperationIdentityError(OperationJournalError):
    """A retry or recovery request does not match the journal identity."""


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def operation_root() -> Path:
    override = os.environ.get("NPA_OPERATION_JOURNAL_DIR", "").strip()
    return (
        Path(override).expanduser() if override else Path.home() / ".npa" / "operations"
    )


def _slug(value: str, *, fallback: str = "operation") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return (cleaned[:48] or fallback).lower()


def deterministic_operation_id(
    *,
    command: str,
    project_id: str,
    project_alias: str,
    resource_type: str,
    requested_name: str,
) -> str:
    """Return the stable ID a retry of the same logical operation will reuse."""

    identity = "\0".join(
        str(item or "").strip()
        for item in (
            command,
            project_id or project_alias,
            resource_type,
            requested_name,
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    # Leave room for retry-generation suffixes while keeping the complete ID a
    # valid directory slug. Long provider resource names are represented by the
    # digest rather than copied verbatim into the operational path.
    return f"{_slug(resource_type)[:8]}-{_slug(requested_name)[:12]}-{digest}"


def operation_path(operation_id: str) -> Path:
    cleaned = _slug(operation_id)
    if cleaned != operation_id:
        raise OperationJournalError(f"invalid operation id: {operation_id!r}")
    return operation_root() / cleaned / "journal.json"


def _sanitize(value: object, *, key: str = "") -> object:
    normalized_key = key.lower()
    if normalized_key not in _SECRET_METADATA_KEYS and any(
        marker in normalized_key for marker in _SECRET_KEY_MARKERS
    ):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(item, key=key) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        cleaned = value
        for pattern in _SECRET_PATTERNS:
            cleaned = pattern.sub("<redacted>", cleaned)
        return cleaned
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def operation_contains_secret(payload: object) -> bool:
    """Return whether a caller attempted to journal a secret-shaped value."""

    def contains(value: object, *, key: str = "") -> bool:
        normalized_key = key.lower()
        if normalized_key not in _SECRET_METADATA_KEYS and any(
            marker in normalized_key for marker in _SECRET_KEY_MARKERS
        ):
            return True
        if isinstance(value, Mapping):
            return any(contains(item, key=str(name)) for name, item in value.items())
        if isinstance(value, (list, tuple, set)):
            return any(contains(item, key=key) for item in value)
        if isinstance(value, Path):
            value = str(value)
        return isinstance(value, str) and _sanitize(value, key=key) != value

    return contains(payload)


def _plan_differences(
    saved: Mapping[str, Any],
    requested: Mapping[str, Any],
    *,
    immutable_keys: Sequence[str],
    limit: int = 16,
) -> list[str]:
    """Return bounded, sanitized leaf differences for immutable plan identity."""

    differences: list[str] = []

    def render(value: object) -> str:
        sanitized = _sanitize(value)
        try:
            result = json.dumps(sanitized, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            result = repr(sanitized)
        return result if len(result) <= 120 else result[:117] + "..."

    def compare(path: str, before: object, after: object) -> None:
        if len(differences) >= limit:
            return
        if isinstance(before, Mapping) and isinstance(after, Mapping):
            for key in sorted(set(before) | set(after), key=str):
                compare(
                    f"{path}.{key}" if path else str(key),
                    before.get(key),
                    after.get(key),
                )
                if len(differences) >= limit:
                    return
            return
        if before != after:
            differences.append(
                f"{path}: journal={render(before)}, request={render(after)}"
            )

    for key in immutable_keys:
        compare(key, saved.get(key), requested.get(key))
        if len(differences) >= limit:
            break
    return differences


_TOPOLOGY_PROGRESS_KEYS = frozenset(
    {
        "agent_exists",
        "existing_cpu_nodes",
        "existing_gpu_nodes",
        "new_agent_instances",
        "new_cpu_nodes",
        "new_gpu_nodes",
        "required_instances",
        "required_disks",
        "required_network_ssd_bytes",
        "required_network_ssd_gib",
        "required_public_ips",
        "required_gpus",
    }
)


def _immutable_plan_identity(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return requested topology identity without provider-derived progress."""

    identity = dict(plan)
    topology = identity.get("topology")
    if isinstance(topology, Mapping):
        identity["topology"] = {
            key: value
            for key, value in topology.items()
            if key not in _TOPOLOGY_PROGRESS_KEYS
        }
    return identity


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _normalize_recovery_argv(
    command: str, argv: Sequence[str] | None
) -> tuple[list[str], str]:
    resolved = (
        [str(item) for item in argv] if argv is not None else shlex.split(command)
    )
    if operation_contains_secret(resolved):
        raise OperationJournalError("refusing to persist secret-bearing recovery argv")
    return resolved, shlex.join(resolved)


def _ensure_private_root() -> Path:
    root = operation_root()
    if root.is_symlink():
        raise OperationJournalError(f"operation journal directory {root} is a symlink")
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
    except OSError as exc:
        raise OperationJournalError(
            f"could not create private operation journal directory {root}: {exc}"
        ) from exc
    return root


@contextmanager
def _locked_operation(operation_id: str) -> Iterator[Path]:
    root = _ensure_private_root()
    directory = root / operation_id
    if directory.is_symlink():
        raise OperationJournalError(f"operation directory {directory} is a symlink")
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        lock_path = directory / ".lock"
        if lock_path.is_symlink():
            raise OperationJournalError(f"operation lock {lock_path} is a symlink")
        with lock_path.open("a+", encoding="utf-8") as lock:
            os.fchmod(lock.fileno(), 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            yield directory
    except OSError as exc:
        raise OperationJournalError(
            f"could not lock provisioning operation {operation_id}: {exc}"
        ) from exc


@contextmanager
def _locked_execution(operation_id: str) -> Iterator[None]:
    """Serialize the complete mutation window for one deterministic operation."""

    root = _ensure_private_root()
    directory = root / operation_id
    if directory.is_symlink():
        raise OperationJournalError(f"operation directory {directory} is a symlink")
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        lock_path = directory / ".execution.lock"
        if lock_path.is_symlink():
            raise OperationJournalError(f"operation lock {lock_path} is a symlink")
        with lock_path.open("a+", encoding="utf-8") as lock:
            os.fchmod(lock.fileno(), 0o600)
            announced_at = 0.0
            while True:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    now = time.monotonic()
                    if not announced_at or now - announced_at >= 10.0:
                        message = (
                            f"Provisioning operation {operation_id}: waiting for the "
                            "same operation's execution lock; no mutation is running in "
                            "this process yet"
                        )
                        print(message, file=sys.stderr, flush=True)
                        try:
                            ProvisioningOperation(operation_id).heartbeat(
                                details={"wait_reason": "execution_lock"}
                            )
                        except OperationJournalError:
                            pass
                        announced_at = now
                    time.sleep(0.25)
            yield
    except OSError as exc:
        raise OperationJournalError(
            f"could not serialize provisioning operation {operation_id}: {exc}"
        ) from exc


def _project_lease_directory(project_identity: str) -> Path:
    identity = str(project_identity or "").strip()
    if not identity:
        raise OperationIdentityError(
            "project-scoped lifecycle mutation requires an exact project id or alias"
        )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return _ensure_private_root() / ".projects" / digest


def _read_project_lease(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise OperationJournalError(f"project lifecycle lease {path} is a symlink")
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperationJournalError(
            f"invalid project lifecycle lease {path}: {exc}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PROJECT_LEASE_SCHEMA_VERSION
    ):
        raise OperationJournalError(
            f"invalid project lifecycle lease {path}: unsupported schema"
        )
    return payload


def _lease_recovery_hint(payload: Mapping[str, Any]) -> str:
    commands = payload.get("recovery_commands")
    commands = commands if isinstance(commands, Mapping) else {}
    return str(commands.get("resume") or commands.get("destroy") or "").strip()


def _project_conflict_message(
    *, project_identity: str, lease: Mapping[str, Any], active: bool
) -> str:
    operation_id = str(lease.get("operation_id") or "unknown")
    phase = str(lease.get("phase") or "unknown")
    owner_pid = int(lease.get("owner_pid") or 0)
    hint = _lease_recovery_hint(lease)
    qualifier = "active" if active else "nonterminal"
    message = (
        f"project {project_identity!r} has a conflicting {qualifier} lifecycle "
        f"operation {operation_id} (phase {phase}, owner_pid {owner_pid})"
    )
    if hint:
        message += f". Safe recovery: {hint}"
    else:
        message += (
            f". Inspect the owner journal at "
            f"{operation_path(operation_id)} before retrying"
        )
    return message


def _lease_with_current_operation(lease: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay the lease pointer with its journal's current phase/owner evidence."""

    current = dict(lease)
    operation_id = str(current.get("operation_id") or "")
    if not operation_id:
        return current
    try:
        journal = ProvisioningOperation(operation_id).read()
    except OperationJournalError:
        return current
    if journal:
        for key in (
            "phase",
            "lifecycle",
            "owner_pid",
            "recovery_commands",
            "project_id",
            "project_alias",
            "resource_type",
            "command",
        ):
            if key in journal:
                current[key] = journal[key]
    return current


@contextmanager
def _locked_project_execution(
    operation: "ProvisioningOperation",
) -> Iterator[Path]:
    """Own the durable mutation lease for the operation's whole project.

    The kernel lock prevents concurrent processes from entering mutation, while
    the atomically-written lease survives crashes and points to the operation
    journal that contains the exact recovery commands and provider identity.
    """

    operation_payload = operation.read()
    project_identity = str(
        operation_payload.get("project_id")
        or operation_payload.get("project_alias")
        or (
            "configure-bootstrap"
            if operation_payload.get("resource_type") == "configure"
            else ""
        )
        or ""
    ).strip()
    directory = _project_lease_directory(project_identity)
    if directory.is_symlink():
        raise OperationJournalError(
            f"project lifecycle directory {directory} is a symlink"
        )
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    lock_path = directory / ".lock"
    lease_path = directory / "lease.json"
    parent_operation_id = os.environ.get("NPA_PARENT_LIFECYCLE_OPERATION", "").strip()
    inherited = _read_project_lease(lease_path)
    if (
        parent_operation_id
        and inherited.get("operation_id") == parent_operation_id
        and int(inherited.get("owner_pid") or 0) == os.getppid()
        and _pid_is_alive(os.getppid())
    ):
        yield lease_path
        return
    if lock_path.is_symlink():
        raise OperationJournalError(f"project lifecycle lock {lock_path} is a symlink")
    with lock_path.open("a+", encoding="utf-8") as lock:
        os.fchmod(lock.fileno(), 0o600)
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lease = _lease_with_current_operation(_read_project_lease(lease_path))
            raise OperationJournalError(
                _project_conflict_message(
                    project_identity=project_identity,
                    lease=lease,
                    active=True,
                )
            ) from exc

        lease = _read_project_lease(lease_path)
        prior_operation_id = str(lease.get("operation_id") or "")
        if prior_operation_id and prior_operation_id != operation.operation_id:
            try:
                prior = ProvisioningOperation(prior_operation_id).reconcile_liveness()
            except OperationJournalError:
                prior = lease
            prior_phase = str(prior.get("phase") or lease.get("phase") or "")
            if prior_phase not in TERMINAL_PHASES:
                raise OperationJournalError(
                    _project_conflict_message(
                        project_identity=project_identity,
                        lease=prior,
                        active=(
                            str(prior.get("lifecycle") or "") == "running"
                            and _pid_is_alive(int(prior.get("owner_pid") or 0))
                        ),
                    )
                )

        current = dict(operation.read())
        _write_atomic(
            lease_path,
            {
                "schema_version": PROJECT_LEASE_SCHEMA_VERSION,
                "project_identity": project_identity,
                "project_id": str(current.get("project_id") or ""),
                "project_alias": str(current.get("project_alias") or ""),
                "operation_id": operation.operation_id,
                "command": str(current.get("command") or ""),
                "resource_type": str(current.get("resource_type") or ""),
                "phase": str(current.get("phase") or "prepared"),
                "lifecycle": str(current.get("lifecycle") or "running"),
                "owner_pid": os.getpid(),
                "acquired_at": utc_now(),
                "recovery_commands": dict(current.get("recovery_commands") or {}),
            },
        )
        try:
            yield lease_path
        finally:
            final = operation.read()
            finished = _read_project_lease(lease_path)
            finished.update(
                {
                    "phase": str(final.get("phase") or ""),
                    "lifecycle": str(final.get("lifecycle") or ""),
                    "owner_pid": int(final.get("owner_pid") or os.getpid()),
                    "released_at": utc_now(),
                    "recovery_commands": dict(final.get("recovery_commands") or {}),
                }
            )
            _write_atomic(lease_path, finished)


def _read_unlocked(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise OperationJournalError(f"operation journal {path} is a symlink")
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperationJournalError(f"invalid operation journal {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OperationJournalError(
            f"invalid operation journal {path}: expected object"
        )
    schema = payload.get("schema_version")
    if schema in {1, "1"}:
        # The pre-release representation used an integer.  Normalize in memory;
        # the next atomic mutation upgrades it without losing fields.
        payload["schema_version"] = SCHEMA_VERSION
    elif schema != SCHEMA_VERSION:
        raise OperationJournalError(
            f"invalid operation journal {path}: unsupported schema {schema!r}"
        )
    if not isinstance(payload.get("resources", []), list):
        raise OperationJournalError(
            f"invalid operation journal {path}: resources must be a list"
        )
    return payload


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    if operation_contains_secret(payload):
        raise OperationJournalError("refusing to persist secret-bearing operation data")
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_path)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_sanitize(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class ProvisioningOperation:
    """Handle for one stable, lock-protected provisioning transaction."""

    operation_id: str

    @property
    def path(self) -> Path:
        return operation_path(self.operation_id)

    @property
    def state_dir(self) -> Path:
        return self.path.parent / "state"

    @classmethod
    def prepare(
        cls,
        *,
        command: str,
        project_alias: str = "",
        project_id: str = "",
        tenant_id: str = "",
        region: str = "",
        backend: Mapping[str, Any] | None = None,
        resource_type: str,
        requested_name: str,
        ownership_source: str = "npa-command",
        resume_command: str,
        destroy_command: str = "",
        resume_argv: Sequence[str] | None = None,
        destroy_argv: Sequence[str] | None = None,
    ) -> ProvisioningOperation:
        base_operation_id = deterministic_operation_id(
            command=command,
            project_id=project_id,
            project_alias=project_alias,
            resource_type=resource_type,
            requested_name=requested_name,
        )
        now = utc_now()
        normalized_resume_argv, normalized_resume = _normalize_recovery_argv(
            resume_command, resume_argv
        )
        normalized_destroy_argv, normalized_destroy = _normalize_recovery_argv(
            destroy_command, destroy_argv
        )
        generation = 0
        while True:
            operation_id = (
                base_operation_id
                if generation == 0
                else f"{base_operation_id}-r{generation}"
            )
            operation = cls(operation_id)
            with _locked_operation(operation_id):
                candidate = _read_unlocked(operation.path)
            if not candidate or candidate.get("phase") not in TERMINAL_PHASES:
                break
            generation += 1
        with _locked_operation(operation_id):
            existing = _read_unlocked(operation.path)
            expected = {
                "operation_id": operation_id,
                "project_alias": str(project_alias or ""),
                "project_id": str(project_id or ""),
                "resource_type": str(resource_type or ""),
                "requested_name": str(requested_name or ""),
            }
            if existing:
                for key, value in expected.items():
                    saved = str(existing.get(key) or "")
                    if value and saved and saved != value:
                        raise OperationIdentityError(
                            f"operation {operation_id} identity mismatch for {key}: "
                            f"journal={saved!r}, request={value!r}"
                        )
                payload = existing
                prior_pid = int(payload.get("owner_pid") or 0)
                if (
                    str(payload.get("phase") or "") in RECOVERABLE_PHASES
                    and str(payload.get("lifecycle") or "") == "running"
                    and prior_pid != os.getpid()
                    and not _pid_is_alive(prior_pid)
                ):
                    payload["lifecycle"] = "interrupted"
                    payload.setdefault("events", []).append(
                        {
                            "phase": str(payload.get("phase") or "prepared"),
                            "lifecycle": "interrupted",
                            "error_type": "ProcessExited",
                            "recorded_at": now,
                        }
                    )
                payload["resume_count"] = int(payload.get("resume_count") or 0) + 1
                payload["updated_at"] = now
                payload["owner_pid"] = os.getpid()
                payload["heartbeat_at"] = now
                payload["lifecycle"] = "running"
                payload["recovery_commands"] = {
                    "resume": normalized_resume,
                    "destroy": normalized_destroy,
                    "resume_argv": normalized_resume_argv,
                    "destroy_argv": normalized_destroy_argv,
                }
            else:
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "operation_id": operation_id,
                    "command": str(command or ""),
                    "project_alias": str(project_alias or ""),
                    "project_id": str(project_id or ""),
                    "tenant_id": str(tenant_id or ""),
                    "region": str(region or ""),
                    "backend": dict(backend or {}),
                    "resource_type": str(resource_type or ""),
                    "requested_name": str(requested_name or ""),
                    "ownership_source": str(ownership_source or ""),
                    "phase": "prepared",
                    "created_at": now,
                    "updated_at": now,
                    "owner_pid": os.getpid(),
                    "heartbeat_at": now,
                    "lifecycle": "running",
                    "resume_count": 0,
                    "resources": [],
                    "config_mutations": [],
                    "events": [{"phase": "prepared", "recorded_at": now}],
                    "recovery_commands": {
                        "resume": normalized_resume,
                        "destroy": normalized_destroy,
                        "resume_argv": normalized_resume_argv,
                        "destroy_argv": normalized_destroy_argv,
                    },
                    "last_error": "",
                    "last_error_type": "",
                }
            _write_atomic(operation.path, payload)
        return operation

    def read(self) -> dict[str, Any]:
        with _locked_operation(self.operation_id):
            return deepcopy(_read_unlocked(self.path))

    def reconcile_liveness(self) -> dict[str, Any]:
        """Atomically classify a recoverable journal whose owner process exited."""

        with _locked_operation(self.operation_id):
            payload = _read_unlocked(self.path)
            if not payload:
                return {}
            phase = str(payload.get("phase") or "")
            lifecycle = str(payload.get("lifecycle") or "")
            owner_pid = int(payload.get("owner_pid") or 0)
            if (
                phase in RECOVERABLE_PHASES
                and lifecycle == "running"
                and owner_pid != os.getpid()
                and not _pid_is_alive(owner_pid)
            ):
                now = utc_now()
                payload["lifecycle"] = "interrupted"
                payload["last_error_type"] = "ProcessExited"
                payload["last_error"] = (
                    "operation owner process exited before convergence"
                )
                payload["updated_at"] = now
                payload.setdefault("events", []).append(
                    {
                        "phase": phase,
                        "lifecycle": "interrupted",
                        "error_type": "ProcessExited",
                        "recorded_at": now,
                    }
                )
                _write_atomic(self.path, payload)
            return deepcopy(payload)

    def set_recovery_commands(
        self,
        *,
        resume_argv: Sequence[str] | None = None,
        destroy_argv: Sequence[str] | None = None,
    ) -> None:
        """Atomically replace structured non-secret recovery commands."""

        with _locked_operation(self.operation_id):
            payload = _read_unlocked(self.path)
            commands = payload.get("recovery_commands")
            commands = dict(commands) if isinstance(commands, dict) else {}
            if resume_argv is not None:
                normalized, rendered = _normalize_recovery_argv("", resume_argv)
                commands["resume_argv"] = normalized
                commands["resume"] = rendered
            if destroy_argv is not None:
                normalized, rendered = _normalize_recovery_argv("", destroy_argv)
                commands["destroy_argv"] = normalized
                commands["destroy"] = rendered
            payload["recovery_commands"] = commands
            payload["updated_at"] = utc_now()
            _write_atomic(self.path, payload)

    def transition(
        self,
        phase: str,
        *,
        error: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        cleaned = str(phase or "").strip().lower()
        allowed = set(_PHASE_ORDER) | RECOVERABLE_PHASES | TERMINAL_PHASES
        if cleaned not in allowed:
            raise ValueError(f"unsupported provisioning phase: {phase!r}")
        now = utc_now()
        with _locked_operation(self.operation_id):
            payload = _read_unlocked(self.path)
            if not payload:
                raise OperationJournalError(f"operation {self.operation_id} is missing")
            current = str(payload.get("phase") or "prepared")
            if current in TERMINAL_PHASES and cleaned != current:
                raise OperationJournalError(
                    f"operation {self.operation_id} is terminal in phase {current}"
                )
            if cleaned in _PHASE_ORDER and current in _PHASE_ORDER:
                if _PHASE_ORDER[cleaned] < _PHASE_ORDER[current]:
                    raise OperationJournalError(
                        f"operation {self.operation_id} cannot move backward from "
                        f"{current} to {cleaned}"
                    )
            event: dict[str, Any] = {"phase": cleaned, "recorded_at": now}
            if error:
                event["error"] = str(_sanitize(error))
                payload["last_error"] = str(_sanitize(error))
            if details:
                sanitized_details = _sanitize(dict(details))
                event["details"] = sanitized_details
                if isinstance(sanitized_details, dict) and sanitized_details.get(
                    "error_type"
                ):
                    payload["last_error_type"] = str(sanitized_details["error_type"])
            payload["phase"] = cleaned
            payload["updated_at"] = now
            payload["heartbeat_at"] = now
            payload["owner_pid"] = os.getpid()
            if cleaned == "committed":
                payload["lifecycle"] = "succeeded"
                payload["result"] = "succeeded"
            elif cleaned == "destroyed":
                payload["lifecycle"] = "succeeded"
                payload["result"] = "destroyed"
            elif cleaned == "rolled-back":
                payload["lifecycle"] = "failed"
                payload["result"] = "rolled_back"
            elif cleaned in {"recovery-required", "rollback-incomplete"}:
                payload["lifecycle"] = "failed"
                payload["result"] = "failed"
            else:
                payload["lifecycle"] = (
                    "failed" if payload.get("last_error") else "running"
                )
            payload.setdefault("events", []).append(event)
            _write_atomic(self.path, payload)

    def record_rollback(
        self,
        *,
        attempted: bool,
        completed: bool,
        removed: Sequence[Mapping[str, Any]],
        preserved: Sequence[Mapping[str, Any]],
        outcomes: Sequence[Mapping[str, Any]] = (),
        error: str = "",
    ) -> None:
        """Persist rollback outcome separately from command success/failure."""

        def resource_identity(item: Mapping[str, Any]) -> dict[str, str]:
            return {
                key: str(item.get(key) or "")
                for key in (
                    "resource_type",
                    "provider_id",
                    "requested_name",
                    "project_id",
                    "ownership",
                    "ownership_source",
                )
                if str(item.get(key) or "")
            }

        with _locked_operation(self.operation_id):
            payload = _read_unlocked(self.path)
            payload["rollback"] = {
                "attempted": bool(attempted),
                "completed": bool(completed),
                "resources_removed": [resource_identity(item) for item in removed],
                "resources_preserved": [resource_identity(item) for item in preserved],
                "resource_outcomes": [
                    {
                        **resource_identity(item),
                        "outcome": str(item.get("outcome") or "unknown"),
                        **(
                            {"error": str(_sanitize(item.get("error")))}
                            if item.get("error")
                            else {}
                        ),
                    }
                    for item in outcomes
                ],
                "error": str(_sanitize(error)) if error else "",
                "updated_at": utc_now(),
            }
            payload["updated_at"] = utc_now()
            _write_atomic(self.path, payload)

    def record_failure(
        self, error: BaseException | str, *, error_type: str = ""
    ) -> None:
        """Persist the primary failure without changing or reopening its phase."""

        message = str(error or "provisioning failed")
        kind = error_type or (
            type(error).__name__
            if isinstance(error, BaseException)
            else "ProvisioningError"
        )
        now = utc_now()
        with _locked_operation(self.operation_id):
            payload = _read_unlocked(self.path)
            payload["last_error"] = str(_sanitize(message))
            payload["last_error_type"] = str(_sanitize(kind))
            payload["lifecycle"] = "failed"
            payload["result"] = (
                "rolled_back"
                if str(payload.get("phase") or "") == "rolled-back"
                else "failed"
            )
            payload["updated_at"] = now
            payload.setdefault("events", []).append(
                {
                    "phase": str(payload.get("phase") or "prepared"),
                    "lifecycle": "failed",
                    "error_type": str(_sanitize(kind)),
                    "error": str(_sanitize(message)),
                    "recorded_at": now,
                }
            )
            _write_atomic(self.path, payload)

    def heartbeat(self, *, details: Mapping[str, Any] | None = None) -> None:
        """Refresh liveness without changing the operation phase."""

        now = utc_now()
        with _locked_operation(self.operation_id):
            payload = _read_unlocked(self.path)
            if not payload:
                raise OperationJournalError(f"operation {self.operation_id} is missing")
            if str(payload.get("phase") or "") in TERMINAL_PHASES:
                return
            payload["owner_pid"] = os.getpid()
            payload["heartbeat_at"] = now
            payload["lifecycle"] = "running"
            if details:
                payload.setdefault("events", []).append(
                    {
                        "phase": str(payload.get("phase") or "prepared"),
                        "lifecycle": "heartbeat",
                        "details": _sanitize(dict(details)),
                        "recorded_at": now,
                    }
                )
            payload["updated_at"] = now
            _write_atomic(self.path, payload)

    def checkpoint(self, name: str, evidence: Mapping[str, Any]) -> None:
        """Persist one named, secret-free lifecycle checkpoint.

        Checkpoints complement the coarse provisioning phases.  They are used by
        multi-step resources (notably the agent VM) to resume after a client-side
        transport loss without pretending that a subprocess return code is the
        remote system's state.
        """

        checkpoint_name = _slug(name, fallback="checkpoint")
        now = utc_now()
        with _locked_operation(self.operation_id):
            payload = _read_unlocked(self.path)
            if not payload:
                raise OperationJournalError(f"operation {self.operation_id} is missing")
            sanitized = _sanitize(dict(evidence))
            payload.setdefault("checkpoints", {})[checkpoint_name] = {
                **(sanitized if isinstance(sanitized, dict) else {}),
                "recorded_at": now,
            }
            payload["heartbeat_at"] = now
            payload["updated_at"] = now
            payload.setdefault("events", []).append(
                {
                    "phase": str(payload.get("phase") or "prepared"),
                    "lifecycle": "checkpoint",
                    "checkpoint": checkpoint_name,
                    "recorded_at": now,
                }
            )
            _write_atomic(self.path, payload)

    def interrupt(self, error: BaseException | str = "operation interrupted") -> None:
        """Atomically retain the current recoverable phase as interrupted."""

        now = utc_now()
        error_type = (
            type(error).__name__ if isinstance(error, BaseException) else "Interrupted"
        )
        message = str(error or "operation interrupted")
        with _locked_operation(self.operation_id):
            payload = _read_unlocked(self.path)
            if not payload or str(payload.get("phase") or "") in TERMINAL_PHASES:
                return
            payload["lifecycle"] = "interrupted"
            payload["last_error"] = str(_sanitize(message))
            payload["last_error_type"] = error_type
            payload["heartbeat_at"] = now
            payload["updated_at"] = now
            payload.setdefault("events", []).append(
                {
                    "phase": str(payload.get("phase") or "prepared"),
                    "lifecycle": "interrupted",
                    "error_type": error_type,
                    "error": str(_sanitize(message)),
                    "recorded_at": now,
                }
            )
            _write_atomic(self.path, payload)

    def update_identity(
        self,
        *,
        project_alias: str = "",
        project_id: str = "",
        tenant_id: str = "",
        region: str = "",
        backend: Mapping[str, Any] | None = None,
        allow_region_correction: bool = False,
    ) -> None:
        """Fill initially unknown identity fields, never replace known identity."""

        incoming = {
            "project_alias": project_alias,
            "project_id": project_id,
            "tenant_id": tenant_id,
            "region": region,
        }
        with _locked_operation(self.operation_id):
            payload = _read_unlocked(self.path)
            for key, value in incoming.items():
                cleaned = str(value or "").strip()
                saved = str(payload.get(key) or "").strip()
                if cleaned and saved and cleaned != saved:
                    if (
                        key == "region"
                        and allow_region_correction
                        and not payload.get("resources")
                        and payload.get("phase") in {"prepared", "mutating"}
                    ):
                        payload.setdefault("events", []).append(
                            {
                                "phase": str(payload.get("phase") or "prepared"),
                                "recorded_at": utc_now(),
                                "details": {
                                    "identity_correction": "provider-project-region",
                                    "previous_region": saved,
                                    "region": cleaned,
                                },
                            }
                        )
                        payload[key] = cleaned
                        continue
                    raise OperationIdentityError(
                        f"operation {self.operation_id} identity mismatch for {key}"
                    )
                if cleaned:
                    payload[key] = cleaned
            if backend:
                saved_backend = payload.get("backend")
                saved_backend = saved_backend if isinstance(saved_backend, dict) else {}
                for key, value in backend.items():
                    cleaned = str(value or "").strip()
                    saved = str(saved_backend.get(key) or "").strip()
                    if cleaned and saved and cleaned != saved:
                        raise OperationIdentityError(
                            f"operation {self.operation_id} backend mismatch for {key}"
                        )
                    if cleaned:
                        saved_backend[str(key)] = cleaned
                payload["backend"] = saved_backend
            payload["updated_at"] = utc_now()
            _write_atomic(self.path, payload)

    def record_resource(
        self,
        *,
        resource_type: str,
        requested_name: str,
        provider_id: str = "",
        ownership: str,
        ownership_source: str,
        project_id: str = "",
        labels: Mapping[str, str] | None = None,
        creation_window_start: str = "",
        creation_window_end: str = "",
    ) -> None:
        if ownership not in {
            "created_by_this_operation",
            "adopted",
            "pre_existing",
        }:
            raise ValueError(f"unsupported resource ownership: {ownership!r}")
        entry: dict[str, Any] = {
            "resource_type": str(resource_type or ""),
            "provider_id": str(provider_id or ""),
            "requested_name": str(requested_name or ""),
            "project_id": str(project_id or ""),
            "ownership": ownership,
            "ownership_source": str(ownership_source or ""),
            "labels": dict(labels or {}),
            "creation_window_start": str(creation_window_start or ""),
            "creation_window_end": str(creation_window_end or ""),
            "recorded_at": utc_now(),
        }
        if not entry["resource_type"] or not entry["requested_name"]:
            raise ValueError("resource type and requested name are required")
        with _locked_operation(self.operation_id):
            payload = _read_unlocked(self.path)
            if ownership == "created_by_this_operation":
                entry["creation_window_start"] = entry["creation_window_start"] or str(
                    payload.get("created_at") or ""
                )
                entry["creation_window_end"] = entry["creation_window_end"] or utc_now()
            resources = list(payload.get("resources") or [])
            match_index = next(
                (
                    index
                    for index, saved in enumerate(resources)
                    if isinstance(saved, dict)
                    and saved.get("resource_type") == entry["resource_type"]
                    and saved.get("requested_name") == entry["requested_name"]
                ),
                None,
            )
            if match_index is None:
                resources.append(entry)
            else:
                saved = dict(resources[match_index])
                saved_id = str(saved.get("provider_id") or "")
                if (
                    saved_id
                    and entry["provider_id"]
                    and saved_id != entry["provider_id"]
                ):
                    saved_project = str(saved.get("project_id") or "")
                    same_owned_selector = (
                        saved.get("ownership") == "created_by_this_operation"
                        and entry["ownership"] == "created_by_this_operation"
                        and (
                            not saved_project
                            or not entry["project_id"]
                            or saved_project == entry["project_id"]
                        )
                    )
                    if not same_owned_selector:
                        raise OperationIdentityError(
                            f"operation resource identity changed for {entry['resource_type']} "
                            f"{entry['requested_name']!r}"
                        )
                    # A retry may recreate an operation-owned logical resource
                    # after the provider deleted the first immutable identity.
                    # Preserve both exact IDs as cleanup generations; neither is
                    # allowed to replace or select the other as authoritative.
                    siblings = [
                        item
                        for item in resources
                        if isinstance(item, dict)
                        and item.get("resource_type") == entry["resource_type"]
                        and item.get("requested_name") == entry["requested_name"]
                    ]
                    if "generation" not in saved:
                        saved["generation"] = 1
                        resources[match_index] = saved
                    entry["generation"] = (
                        max(
                            (int(item.get("generation") or 1) for item in siblings),
                            default=0,
                        )
                        + 1
                    )
                    resources.append(entry)
                    payload["resources"] = resources
                    payload["updated_at"] = utc_now()
                    _write_atomic(self.path, payload)
                    return
                saved.update(
                    {
                        key: value
                        for key, value in entry.items()
                        if value not in ("", {})
                    }
                )
                resources[match_index] = saved
            payload["resources"] = resources
            payload["updated_at"] = utc_now()
            _write_atomic(self.path, payload)

    def record_config_mutation(
        self,
        *,
        store: str,
        fields: Sequence[str],
        before: Mapping[str, Any] | None = None,
        secret_fields: Sequence[str] = (),
    ) -> None:
        """Record exact field ownership without copying credential values."""

        secret_set = {str(item) for item in secret_fields}
        snapshot = {
            str(key): value
            for key, value in dict(before or {}).items()
            if str(key) not in secret_set
        }
        mutation = {
            "store": str(store or ""),
            "fields": [str(item) for item in fields],
            "secret_fields": sorted(secret_set),
            "before": snapshot,
            "recorded_at": utc_now(),
        }
        with _locked_operation(self.operation_id):
            payload = _read_unlocked(self.path)
            payload.setdefault("config_mutations", []).append(mutation)
            payload["updated_at"] = utc_now()
            _write_atomic(self.path, payload)

    def record_preflight_plan(self, plan: Mapping[str, Any]) -> None:
        """Persist one immutable, non-secret resolved plan for deterministic resume."""

        candidate = dict(plan)
        if operation_contains_secret(candidate):
            raise OperationJournalError(
                "refusing to persist a secret-bearing preflight plan"
            )
        with _locked_operation(self.operation_id):
            payload = _read_unlocked(self.path)
            existing = payload.get("preflight_plan")
            immutable_keys = (
                "project_alias",
                "project_id",
                "tenant_id",
                "region",
                "topology",
                "source_action",
                "input_action",
            )
            differences = (
                _plan_differences(
                    _immutable_plan_identity(existing),
                    _immutable_plan_identity(candidate),
                    immutable_keys=immutable_keys,
                )
                if isinstance(existing, dict)
                else []
            )
            if differences:
                raise OperationIdentityError(
                    f"operation {self.operation_id} resolved topology changed; "
                    "resume must keep the original resource shape; changed fields: "
                    + "; ".join(differences)
                )
            payload.setdefault("preflight_evaluations", []).append(
                {
                    "recorded_at": utc_now(),
                    "decision": candidate.get("decision"),
                    "quotas": candidate.get("quotas", []),
                    "reasons": candidate.get("reasons", []),
                }
            )
            payload["preflight_plan"] = candidate
            payload["updated_at"] = utc_now()
            _write_atomic(self.path, payload)

    def preserve_state_bytes(self, data: bytes, *, name: str) -> Path:
        """Atomically preserve a Terraform state copy with owner-only permissions."""

        if not data:
            raise ValueError("refusing to preserve an empty Terraform state")
        safe_name = _slug(name, fallback="terraform") + ".tfstate"
        with _locked_operation(self.operation_id):
            state_dir = self.state_dir
            if state_dir.is_symlink():
                raise OperationJournalError(
                    f"operation state directory {state_dir} is a symlink"
                )
            state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            state_dir.chmod(0o700)
            path = state_dir / safe_name
            descriptor, raw = tempfile.mkstemp(prefix=f".{safe_name}.", dir=state_dir)
            temporary = Path(raw)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                path.chmod(0o600)
                directory_fd = os.open(state_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                temporary.unlink(missing_ok=True)
                raise
            payload = _read_unlocked(self.path)
            copies = payload.setdefault("local_state_copies", [])
            relative = str(path.relative_to(self.path.parent))
            copies[:] = [
                item
                for item in copies
                if not isinstance(item, Mapping) or item.get("path") != relative
            ]
            backend = payload.get("backend")
            backend = dict(backend) if isinstance(backend, Mapping) else {}
            copies.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                    "operation_id": self.operation_id,
                    "backend_identity": {
                        key: str(backend.get(key) or "")
                        for key in ("bucket", "endpoint", "key", "region")
                        if str(backend.get(key) or "")
                    },
                    "preserved_at": utc_now(),
                }
            )
            payload["updated_at"] = utc_now()
            _write_atomic(self.path, payload)
            return path

    def preserve_state_file(self, source: Path, *, name: str = "terraform") -> Path:
        try:
            data = source.read_bytes()
        except OSError as exc:
            raise OperationJournalError(
                f"could not read Terraform state {source}: {exc}"
            ) from exc
        return self.preserve_state_bytes(data, name=name)

    def state_copies(self) -> list[Path]:
        payload = self.read()
        paths: list[Path] = []
        backend = payload.get("backend")
        backend = dict(backend) if isinstance(backend, Mapping) else {}
        expected_backend = {
            key: str(backend.get(key) or "")
            for key in ("bucket", "endpoint", "key", "region")
            if str(backend.get(key) or "")
        }
        for item in reversed(payload.get("local_state_copies") or []):
            if not isinstance(item, dict):
                continue
            raw = str(item.get("path") or "")
            candidate = self.path.parent / raw
            try:
                inside = candidate.resolve().is_relative_to(self.path.parent.resolve())
            except OSError as exc:
                raise OperationJournalError(
                    f"could not verify preserved Terraform state {candidate}: {exc}; "
                    "no state was adopted"
                ) from exc
            if not inside:
                raise OperationIdentityError(
                    f"preserved Terraform state path escapes operation "
                    f"{self.operation_id}; no state was adopted"
                )
            if str(item.get("operation_id") or self.operation_id) != self.operation_id:
                raise OperationIdentityError(
                    "preserved Terraform state belongs to another operation; "
                    "no state was adopted"
                )
            recorded_backend = item.get("backend_identity")
            if (
                isinstance(recorded_backend, Mapping)
                and dict(recorded_backend) != expected_backend
            ):
                raise OperationIdentityError(
                    "preserved Terraform state backend identity no longer matches "
                    f"operation {self.operation_id}; no state was adopted"
                )
            if not candidate.is_file():
                raise OperationJournalError(
                    f"preserved Terraform state {candidate} is missing; no state was "
                    f"adopted. Safe recovery: {_lease_recovery_hint(payload)}"
                )
            expected_hash = str(item.get("sha256") or "")
            if expected_hash:
                actual_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
                if actual_hash != expected_hash:
                    raise OperationJournalError(
                        f"preserved Terraform state {candidate} failed SHA-256 "
                        f"validation for operation {self.operation_id}; no state was "
                        f"adopted. Safe recovery: {_lease_recovery_hint(payload)}"
                    )
            paths.append(candidate)
        return list(dict.fromkeys(paths))

    def recovery_summary(self) -> dict[str, Any]:
        payload = self.reconcile_liveness()
        commands = payload.get("recovery_commands")
        commands = commands if isinstance(commands, dict) else {}
        return {
            "operation_id": self.operation_id,
            "project_alias": payload.get("project_alias", ""),
            "project_id": payload.get("project_id", ""),
            "tenant_id": payload.get("tenant_id", ""),
            "region": payload.get("region", ""),
            "phase": payload.get("phase", ""),
            "lifecycle": payload.get("lifecycle", "unknown"),
            "heartbeat_at": payload.get("heartbeat_at", ""),
            "journal": str(self.path),
            "local_state": [str(path) for path in self.state_copies()],
            "backend": payload.get("backend", {}),
            "resume_command": commands.get("resume", ""),
            "destroy_command": commands.get("destroy", ""),
            "resume_argv": list(commands.get("resume_argv") or []),
            "destroy_argv": list(commands.get("destroy_argv") or []),
            "last_error_type": payload.get("last_error_type", ""),
            "last_error": payload.get("last_error", ""),
            "result": payload.get("result", ""),
            "rollback": dict(payload.get("rollback") or {}),
            "resources": list(payload.get("resources") or []),
        }

    def commit(self) -> None:
        self.transition("committed")
        payload = self.read()
        try:
            from npa.cleanup_identity import provisioning_operation_cleanup_identity
            from npa.teardown_receipts import record_teardown_event

            operation_identity = provisioning_operation_cleanup_identity(
                payload,
                state_paths=[str(path) for path in self.state_copies()],
            )

            record_teardown_event(
                phase=f"provision-{payload.get('resource_type', 'resource')}",
                resource=str(payload.get("requested_name") or self.operation_id),
                terminal_state="completed",
                project_alias=str(payload.get("project_alias") or ""),
                project_id=str(payload.get("project_id") or ""),
                context=str(payload.get("requested_name") or ""),
                precheck={"operation_id": self.operation_id},
                action={"kind": "provisioning_commit"},
                verification={"operation_journal": str(self.path)},
                identity={
                    "project_alias": str(payload.get("project_alias") or ""),
                    "project_id": str(payload.get("project_id") or ""),
                    "parent_id": str(payload.get("project_id") or ""),
                    "tenant_id": str(payload.get("tenant_id") or ""),
                    "region": str(payload.get("region") or ""),
                    "operations": [operation_identity],
                },
            )
        except (OSError, RuntimeError, ValueError) as exc:
            # The operational journal is already committed.  A receipt failure is
            # retained in that journal instead of making successful cloud work look
            # failed or attempting destructive rollback.
            with _locked_operation(self.operation_id):
                current = _read_unlocked(self.path)
                current.setdefault("audit_warnings", []).append(str(_sanitize(exc)))
                current["updated_at"] = utc_now()
                _write_atomic(self.path, current)


def load_operation(operation_id: str) -> ProvisioningOperation:
    operation = ProvisioningOperation(operation_id)
    if not operation.read():
        raise OperationJournalError(f"operation {operation_id!r} was not found")
    return operation


def list_operations(
    *,
    project_alias: str = "",
    project_id: str = "",
    resource_type: str = "",
    requested_name: str = "",
) -> list[ProvisioningOperation]:
    root = operation_root()
    if not root.is_dir() or root.is_symlink():
        return []
    matches: list[tuple[str, ProvisioningOperation]] = []
    for path in sorted(root.glob("*/journal.json")):
        operation = ProvisioningOperation(path.parent.name)
        try:
            payload = operation.reconcile_liveness()
        except OperationJournalError:
            continue
        if project_alias and payload.get("project_alias") != project_alias:
            continue
        if project_id and payload.get("project_id") != project_id:
            continue
        if resource_type and payload.get("resource_type") != resource_type:
            continue
        if requested_name and payload.get("requested_name") != requested_name:
            continue
        matches.append((str(payload.get("updated_at") or ""), operation))
    return [
        item[1]
        for item in sorted(
            matches,
            key=lambda item: (item[0], item[1].operation_id),
            reverse=True,
        )
    ]


_CURRENT_OPERATION: ContextVar[ProvisioningOperation | None] = ContextVar(
    "npa_current_provisioning_operation", default=None
)


def current_operation() -> ProvisioningOperation | None:
    return _CURRENT_OPERATION.get()


@contextmanager
def operation_heartbeats(
    operation: ProvisioningOperation | None,
    *,
    phase: str,
    interval_seconds: float = 15.0,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[None]:
    """Emit structured liveness while a blocking local transport call runs.

    The worker thread only writes redacted journal metadata and emits a stable
    phase/count payload; it never receives argv, environment, provider bodies, or
    credentials.  The caller remains in the main thread, so interrupts retain
    their normal semantics.
    """

    if operation is None:
        yield
        return
    interval = max(0.005, float(interval_seconds))
    stop = threading.Event()

    def beat() -> None:
        count = 0
        while not stop.wait(interval):
            count += 1
            details = {"operation_phase": str(phase), "heartbeat_sequence": count}
            operation.heartbeat(details=details)
            if emit is not None:
                emit(dict(details))

    thread = threading.Thread(
        target=beat,
        name=f"npa-operation-heartbeat-{operation.operation_id}",
        daemon=True,
    )
    initial = {"operation_phase": str(phase), "heartbeat_sequence": 0}
    operation.heartbeat(details=initial)
    if emit is not None:
        emit(dict(initial))
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=min(interval, 1.0))


@contextmanager
def operation_context(
    operation: ProvisioningOperation,
) -> Iterator[ProvisioningOperation]:
    current = current_operation()
    if current is not None:
        current_payload = current.read()
        candidate = operation.read()
        if current.operation_id != operation.operation_id and (
            current_payload.get("project_id") != candidate.get("project_id")
            or current_payload.get("resource_type") != candidate.get("resource_type")
        ):
            raise OperationIdentityError(
                "nested provisioning attempted to replace its parent operation context"
            )
        yield current
        return
    with (
        _locked_project_execution(operation),
        _locked_execution(operation.operation_id),
    ):
        # A concurrent retry may have completed while this caller waited for the
        # execution lock.  Never replay a terminal transaction with stale state.
        phase = str(operation.read().get("phase") or "")
        if phase in TERMINAL_PHASES:
            raise OperationJournalError(
                f"operation {operation.operation_id} completed while this retry waited "
                f"for its execution lock (phase {phase}); rerun to start a new operation"
            )
        token = _CURRENT_OPERATION.set(operation)
        try:
            yield operation
        except (KeyboardInterrupt, SystemExit) as exc:
            operation.interrupt(exc)
            raise
        finally:
            _CURRENT_OPERATION.reset(token)


def emit_recovery_summary(
    operation: ProvisioningOperation, *, output_json: bool = False
) -> str:
    """Return and optionally print the deterministic recovery information."""

    summary = operation.recovery_summary()
    if output_json:
        rendered = json.dumps(summary, indent=2, sort_keys=True)
    else:
        states = ", ".join(summary["local_state"]) or "none yet"
        rendered = (
            f"Provisioning operation: {summary['operation_id']}\n"
            f"Durable journal: {summary['journal']}\n"
            f"Preserved local Terraform state: {states}\n"
            f"Resume: {summary['resume_command']}"
        )
        if summary.get("last_error"):
            rendered += (
                "\nPrimary failure: "
                f"{summary.get('last_error_type') or 'ProvisioningError'}: "
                f"{summary['last_error']}"
            )
        rollback = summary.get("rollback") or {}
        if rollback:
            rendered += (
                "\nRollback: attempted="
                f"{bool(rollback.get('attempted'))}, completed="
                f"{bool(rollback.get('completed'))}, resources_removed="
                f"{len(rollback.get('resources_removed') or [])}"
            )
            outcomes = rollback.get("resource_outcomes") or []
            if outcomes:
                rendered += "\nRollback outcomes: " + ", ".join(
                    f"{item.get('resource_type', 'resource')}:"
                    f"{item.get('provider_id', item.get('requested_name', '?'))}="
                    f"{item.get('outcome', 'unknown')}"
                    for item in outcomes
                )
        if summary.get("destroy_command"):
            rendered += f"\nDestroy: {summary['destroy_command']}"
    return rendered


__all__ = [
    "OperationIdentityError",
    "OperationJournalError",
    "ProvisioningOperation",
    "SCHEMA_VERSION",
    "TERMINAL_PHASES",
    "current_operation",
    "deterministic_operation_id",
    "emit_recovery_summary",
    "list_operations",
    "load_operation",
    "operation_contains_secret",
    "operation_context",
    "operation_path",
    "operation_root",
    "utc_now",
]
