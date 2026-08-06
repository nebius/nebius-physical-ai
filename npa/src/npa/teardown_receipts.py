"""Durable, non-secret teardown evidence shared by NPA cleanup commands.

Receipts deliberately live outside per-project configuration.  They are audit
records, not operational state: successful cleanup may remove config, caches,
credentials, and provider state while these small JSON files remain available
to explain what was verified.  They are retained until an explicit prune.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator


SCHEMA_VERSION = "npa.teardown.receipt.v1"
TERMINAL_STATES = frozenset(
    {
        "already_absent",
        "cancelled",
        "completed",
        "deleted",
        "terminal",
        "verified_absent",
        "verified_deleted",
    }
)
_UNSAFE_KEYS = (
    "access_key",
    "api_key",
    "credential",
    "endpoint",
    "environment",
    "password",
    "private_key",
    "secret",
    "token",
)
_SECRET_PATTERNS = (
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bhf_[A-Za-z0-9_=-]{8,}\b"),
    re.compile(r"\bnvapi-[A-Za-z0-9_=-]{8,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_=-]{20,}\b"),
    re.compile(
        r"(?i)\b(?:access[_-]?key|api[_-]?key|bearer|credential|password|"
        r"private[_-]?key|secret|token)\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(r"(?i)https?://[^\s\"']+"),
)


class TeardownReceiptError(RuntimeError):
    """A durable receipt could not be read or written safely."""


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def receipt_root() -> Path:
    override = os.environ.get("NPA_TEARDOWN_RECEIPT_DIR", "").strip()
    return (
        Path(override).expanduser()
        if override
        else Path.home() / ".npa" / "teardown-receipts"
    )


def _journal_key(project_alias: str, project_id: str) -> str:
    identity = str(project_id or project_alias or "global").strip()
    label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", project_alias.strip()).strip("-._")
    label = label[:48] or "global"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{label}-{digest}"


def receipt_path(*, project_alias: str = "", project_id: str = "") -> Path:
    return receipt_root() / f"{_journal_key(project_alias, project_id)}.json"


def _sanitize(value: object, *, key: str = "") -> object:
    lowered = key.lower()
    if any(marker in lowered for marker in _UNSAFE_KEYS):
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


def receipt_contains_secret(payload: object) -> bool:
    """Return whether a caller tried to persist a known secret-shaped value."""

    def contains(value: object, *, key: str = "") -> bool:
        if any(marker in key.lower() for marker in _UNSAFE_KEYS):
            return True
        if isinstance(value, Mapping):
            return any(contains(item, key=str(name)) for name, item in value.items())
        if isinstance(value, (list, tuple, set)):
            return any(contains(item, key=key) for item in value)
        if isinstance(value, Path):
            value = str(value)
        return isinstance(value, str) and _sanitize(value, key=key) != value

    return contains(payload)


@contextmanager
def _locked_root() -> Iterator[Path]:
    root = receipt_root()
    try:
        if root.is_symlink():
            raise TeardownReceiptError(
                f"teardown receipt directory {root} is a symlink"
            )
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
        lock_path = root / ".lock"
        if lock_path.is_symlink():
            raise TeardownReceiptError(
                f"teardown receipt lock {lock_path} is a symlink"
            )
        with lock_path.open("a+", encoding="utf-8") as lock:
            os.fchmod(lock.fileno(), 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            yield root
    except OSError as exc:
        raise TeardownReceiptError(
            f"could not access teardown receipt directory {root}: {exc}"
        ) from exc


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise TeardownReceiptError(f"teardown receipt {path} is a symlink")
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TeardownReceiptError(f"invalid teardown receipt {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise TeardownReceiptError(
            f"invalid teardown receipt {path}: unsupported schema"
        )
    events = payload.get("events")
    if not isinstance(events, list):
        raise TeardownReceiptError(
            f"invalid teardown receipt {path}: events must be a list"
        )
    return payload


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise TeardownReceiptError(
            f"could not write teardown receipt {path}: {exc}"
        ) from exc


def record_teardown_event(
    *,
    phase: str,
    resource: str,
    terminal_state: str,
    project_alias: str = "",
    project_id: str = "",
    context: str = "",
    precheck: Mapping[str, Any] | None = None,
    action: Mapping[str, Any] | str | None = None,
    verification: Mapping[str, Any] | None = None,
    errors: Sequence[str] = (),
) -> Path:
    """Append one crash-safe teardown transaction event.

    Callers provide only evidence, never credentials or subprocess environments.
    A defense-in-depth sanitizer removes secret-bearing keys and recognizable
    token forms before the atomic write.
    """

    cleaned_phase = str(phase or "").strip()
    cleaned_resource = str(resource or "").strip()
    cleaned_state = str(terminal_state or "unknown").strip().lower()
    if not cleaned_phase or not cleaned_resource:
        raise ValueError("teardown receipt phase and resource are required")
    now = utc_now()
    event = _sanitize(
        {
            "phase": cleaned_phase,
            "resource": cleaned_resource,
            "project_alias": str(project_alias or ""),
            "project_id": str(project_id or ""),
            "context": str(context or ""),
            "precheck": dict(precheck or {}),
            "action": action if action is not None else {},
            "verification": dict(verification or {}),
            "terminal_state": cleaned_state,
            "errors": [str(item) for item in errors],
            "recorded_at": now,
        }
    )
    path = receipt_path(project_alias=project_alias, project_id=project_id)
    with _locked_root():
        payload = _read(path)
        if not payload:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "receipt_id": path.stem,
                "created_at": now,
                "updated_at": now,
                "project_alias": str(project_alias or ""),
                "project_id": str(project_id or ""),
                "events": [],
            }
        if str(payload.get("project_id") or "") not in {"", str(project_id or "")}:
            raise TeardownReceiptError(
                f"receipt identity mismatch at {path}; refusing to mix projects"
            )
        events = list(payload.get("events") or [])
        events.append(event)
        payload["events"] = events
        payload["updated_at"] = now
        if project_alias and not payload.get("project_alias"):
            payload["project_alias"] = project_alias
        if project_id and not payload.get("project_id"):
            payload["project_id"] = project_id
        _write_atomic(path, payload)
    return path


def list_teardown_receipts() -> list[dict[str, Any]]:
    root = receipt_root()
    if not root.is_dir():
        return []
    receipts: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        payload = _read(path)
        payload["path"] = str(path)
        receipts.append(payload)
    return sorted(
        receipts, key=lambda item: str(item.get("updated_at") or ""), reverse=True
    )


def latest_phase_states(
    *, project_alias: str = "", project_id: str = ""
) -> dict[str, dict[str, Any]]:
    """Return latest evidence per phase, including receipts surviving config removal."""

    per_receipt: dict[tuple[str, str], dict[str, Any]] = {}
    for receipt in list_teardown_receipts():
        receipt_project_id = str(receipt.get("project_id") or "")
        receipt_alias = str(receipt.get("project_alias") or "")
        if project_id and receipt_project_id != project_id:
            continue
        if project_alias and receipt_alias != project_alias:
            continue
        for event in receipt.get("events") or []:
            if not isinstance(event, dict):
                continue
            phase = str(event.get("phase") or "")
            key = (str(receipt.get("receipt_id") or ""), phase)
            previous = per_receipt.get(key) or {}
            if phase and str(event.get("recorded_at") or "") >= str(
                previous.get("recorded_at") or ""
            ):
                per_receipt[key] = event
    grouped: dict[str, list[dict[str, Any]]] = {}
    for (_receipt_id, phase), event in per_receipt.items():
        grouped.setdefault(phase, []).append(event)
    states: dict[str, dict[str, Any]] = {}
    for phase, events in grouped.items():
        unresolved = [
            event
            for event in events
            if str(event.get("terminal_state") or "").lower() not in TERMINAL_STATES
        ]
        # Across projects, one unresolved identity must remain actionable; a
        # newer completed project may not hide it. If every identity is terminal,
        # report the newest durable convergence evidence.
        states[phase] = max(
            unresolved or events, key=lambda event: str(event.get("recorded_at") or "")
        )
    return states


def receipt_is_terminal(receipt: Mapping[str, Any]) -> bool:
    latest: dict[str, str] = {}
    for event in receipt.get("events") or []:
        if isinstance(event, Mapping):
            latest[str(event.get("phase") or "")] = str(
                event.get("terminal_state") or "unknown"
            ).lower()
    return bool(latest) and all(state in TERMINAL_STATES for state in latest.values())


def prune_teardown_receipts(*, older_than_days: int) -> tuple[list[Path], list[str]]:
    """Delete only terminal receipts older than an explicit retention threshold."""

    if older_than_days < 0:
        raise ValueError("receipt retention days must be non-negative")
    now = datetime.now(timezone.utc)
    removed: list[Path] = []
    retained: list[str] = []
    with _locked_root() as root:
        for path in sorted(root.glob("*.json")):
            payload = _read(path)
            if not receipt_is_terminal(payload):
                retained.append(f"{path.name}: unresolved/uncertain phase remains")
                continue
            try:
                updated = datetime.fromisoformat(
                    str(payload.get("updated_at") or "").replace("Z", "+00:00")
                )
            except ValueError:
                retained.append(f"{path.name}: invalid updated_at")
                continue
            age_days = max(0, (now - updated).days)
            if age_days < older_than_days:
                retained.append(
                    f"{path.name}: {age_days} day(s) old; retention is {older_than_days}"
                )
                continue
            try:
                path.unlink()
            except OSError as exc:
                retained.append(f"{path.name}: could not prune: {exc}")
            else:
                removed.append(path)
    return removed, retained
