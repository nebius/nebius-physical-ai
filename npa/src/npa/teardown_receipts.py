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
from urllib.parse import urlparse


SCHEMA_VERSION = "npa.teardown.receipt.v2"
LEGACY_SCHEMA_VERSIONS = frozenset({"npa.teardown.receipt.v1"})
TERMINAL_STATES = frozenset(
    {
        "already_absent",
        "cancelled",
        "completed",
        "deleted",
        "not_submitted",
        "terminal",
        "verified_absent",
        "verified_deleted",
    }
)
_RECEIPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
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
    if override:
        return Path(override).expanduser()
    config_dir = os.environ.get("NPA_CONFIG_DIR", "").strip()
    root = Path(config_dir).expanduser() if config_dir else Path.home() / ".npa"
    return root / "teardown-receipts"


def _journal_key(project_alias: str, project_id: str) -> str:
    identity = str(project_id or project_alias or "global").strip()
    label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", project_alias.strip()).strip("-._")
    label = label[:48] or "global"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{label}-{digest}"


def receipt_path(*, project_alias: str = "", project_id: str = "") -> Path:
    return receipt_root() / f"{_journal_key(project_alias, project_id)}.json"


def receipt_path_for_id(receipt_id: str) -> Path:
    """Resolve one opaque receipt ID without permitting arbitrary file reads."""

    cleaned = str(receipt_id or "").strip()
    if not _RECEIPT_ID_RE.fullmatch(cleaned) or cleaned in {".", ".."}:
        raise TeardownReceiptError(
            "receipt selector must be the opaque ID printed by `npa cleanup "
            "--list-receipts`, not a filesystem path"
        )
    root = receipt_root()
    candidate = root / f"{cleaned}.json"
    try:
        if candidate.parent.resolve() != root.resolve():
            raise TeardownReceiptError("receipt selector escapes the receipt root")
    except OSError as exc:
        raise TeardownReceiptError(
            f"could not resolve teardown receipt root: {exc}"
        ) from exc
    return candidate


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
    schema = payload.get("schema_version") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or schema not in {
        SCHEMA_VERSION,
        *LEGACY_SCHEMA_VERSIONS,
    }:
        raise TeardownReceiptError(
            f"invalid teardown receipt {path}: unsupported schema"
        )
    events = payload.get("events")
    if not isinstance(events, list):
        raise TeardownReceiptError(
            f"invalid teardown receipt {path}: events must be a list"
        )
    def require_timestamp(value: object, field: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise TeardownReceiptError(
                f"invalid teardown receipt {path}: {field} must be a timestamp"
            )
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise TeardownReceiptError(
                f"invalid teardown receipt {path}: {field} is not ISO-8601"
            ) from exc
        if parsed.tzinfo is None:
            raise TeardownReceiptError(
                f"invalid teardown receipt {path}: {field} has no timezone"
            )

    receipt_id = payload.get("receipt_id")
    if not isinstance(receipt_id, str) or receipt_id != path.stem:
        raise TeardownReceiptError(
            f"invalid teardown receipt {path}: embedded receipt_id mismatch"
        )
    for field in ("created_at", "updated_at"):
        require_timestamp(payload.get(field), field)
    for field in ("project_alias", "project_id"):
        if not isinstance(payload.get(field, ""), str):
            raise TeardownReceiptError(
                f"invalid teardown receipt {path}: {field} must be a string"
            )
    if not isinstance(payload.get("identity", {}), Mapping):
        raise TeardownReceiptError(
            f"invalid teardown receipt {path}: identity must be an object"
        )

    identity_fields = frozenset(
        {
            "project_alias",
            "project_id",
            "tenant_id",
            "parent_id",
            "account_id",
            "region",
            "profile",
            "context",
            "kubeconfig_path",
            "cluster_id",
            "cluster_name",
            "agent_name",
            "instance_id",
            "operation_id",
            "service_account_id",
            "run_id",
            "workflow_s3_uri",
            "sky_job_id",
            "controller_context",
            "resource_type",
            "requested_name",
        }
    )

    def validate_identity(value: object, location: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str) or not key:
                    raise TeardownReceiptError(
                        f"invalid teardown receipt {path}: {location} has invalid key"
                    )
                if key in identity_fields and not isinstance(item, str):
                    raise TeardownReceiptError(
                        f"invalid teardown receipt {path}: {location}.{key} "
                        "must be a string"
                    )
                collection_keys = {
                    "agents": ("agent_name", "instance_id"),
                    "clusters": (
                        "cluster_id",
                        "context",
                        "controller_context",
                        "operation_id",
                    ),
                    "workflows": ("run_id",),
                    "operations": ("operation_id",),
                    "generations": ("service_account_id",),
                }
                if key in collection_keys:
                    if not isinstance(item, list):
                        raise TeardownReceiptError(
                            f"invalid teardown receipt {path}: {location}.{key} "
                            "must be a list"
                        )
                    for index, member in enumerate(item, start=1):
                        if not isinstance(member, Mapping) or not any(
                            isinstance(member.get(candidate), str)
                            and bool(str(member.get(candidate) or "").strip())
                            for candidate in collection_keys[key]
                        ):
                            raise TeardownReceiptError(
                                f"invalid teardown receipt {path}: {location}.{key}"
                                f"[{index}] has no complete immutable identity"
                            )
                        if key == "agents" and not all(
                            isinstance(member.get(candidate), str)
                            and bool(str(member.get(candidate) or "").strip())
                            for candidate in ("agent_name", "instance_id")
                        ):
                            raise TeardownReceiptError(
                                f"invalid teardown receipt {path}: {location}.agents"
                                f"[{index}] has no complete agent identity"
                            )
                validate_identity(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value, start=1):
                validate_identity(item, f"{location}[{index}]")

    def reject_duplicate_identities(identity: Mapping[str, Any], location: str) -> None:
        for collection, key in (
            ("agents", "instance_id"),
            ("clusters", "cluster_id"),
            ("workflows", "run_id"),
            ("operations", "operation_id"),
            ("generations", "service_account_id"),
        ):
            values = identity.get(collection)
            if not isinstance(values, list):
                continue
            identifiers = [
                item.get(key)
                for item in values
                if isinstance(item, Mapping) and item.get(key) not in (None, "")
            ]
            if len(identifiers) != len(set(identifiers)):
                raise TeardownReceiptError(
                    f"invalid teardown receipt {path}: {location}.{collection} "
                    f"contains duplicate {key} values"
                )
            for index, item in enumerate(values, start=1):
                if isinstance(item, Mapping):
                    reject_duplicate_identities(
                        item, f"{location}.{collection}[{index}]"
                    )

    root_identity = payload.get("identity", {})
    validate_identity(root_identity, "identity")
    reject_duplicate_identities(root_identity, "identity")
    previous_sequence = 0
    for index, event in enumerate(events, start=1):
        if not isinstance(event, Mapping):
            raise TeardownReceiptError(
                f"invalid teardown receipt {path}: event {index} is not an object"
            )
        for field in ("phase", "resource", "terminal_state"):
            if not isinstance(event.get(field), str) or not str(event[field]).strip():
                raise TeardownReceiptError(
                    f"invalid teardown receipt {path}: event {index} has invalid {field}"
                )
        for field in ("project_alias", "project_id", "context"):
            if not isinstance(event.get(field, ""), str):
                raise TeardownReceiptError(
                    f"invalid teardown receipt {path}: event {index} has invalid {field}"
                )
        for field in ("precheck", "verification", "identity"):
            if not isinstance(event.get(field, {}), Mapping):
                raise TeardownReceiptError(
                    f"invalid teardown receipt {path}: event {index} has invalid {field}"
                )
        action = event.get("action", {})
        if not isinstance(action, (Mapping, str)):
            raise TeardownReceiptError(
                f"invalid teardown receipt {path}: event {index} has invalid action"
            )
        if isinstance(action, Mapping) and "kind" in action and (
            not isinstance(action.get("kind"), str) or not action.get("kind")
        ):
            raise TeardownReceiptError(
                f"invalid teardown receipt {path}: event {index} has invalid action kind"
            )
        errors = event.get("errors", [])
        if not isinstance(errors, list) or not all(
            isinstance(item, str) for item in errors
        ):
            raise TeardownReceiptError(
                f"invalid teardown receipt {path}: event {index} has invalid errors"
            )
        require_timestamp(event.get("recorded_at"), f"event {index} recorded_at")
        sequence = event.get("sequence")
        if schema == SCHEMA_VERSION:
            if type(sequence) is not int or sequence != previous_sequence + 1:
                raise TeardownReceiptError(
                    f"invalid teardown receipt {path}: event sequence is missing, "
                    "duplicated, or non-monotonic"
                )
            previous_sequence = sequence
        verification = event.get("verification", {})
        event_identity = event.get("identity", {})
        validate_identity(event_identity, f"event {index} identity")
        reject_duplicate_identities(event_identity, f"event {index} identity")
        for field in ("exact_instance_absent", "terraform_destroy_completed"):
            if field in verification and type(verification[field]) is not bool:
                raise TeardownReceiptError(
                    f"invalid teardown receipt {path}: event {index} verification "
                    f"field {field} must be boolean"
                )
        graph = verification.get("terraform_dependency_graph")
        if graph is not None and (
            not isinstance(graph, list)
            or not all(isinstance(item, str) and item for item in graph)
            or len(set(graph)) != len(graph)
        ):
            raise TeardownReceiptError(
                f"invalid teardown receipt {path}: event {index} has invalid "
                "terraform dependency graph"
            )
    if schema in LEGACY_SCHEMA_VERSIONS:
        for index, event in enumerate(events, start=1):
            event["sequence"] = index
    # Normalize additively in memory. The next append durably migrates v1 to v2.
    payload["schema_version"] = SCHEMA_VERSION
    payload.setdefault("identity", {})
    return payload


def load_teardown_receipt(receipt_id: str) -> dict[str, Any]:
    """Load one receipt by opaque ID, rejecting paths and symlinks."""

    path = receipt_path_for_id(receipt_id)
    payload = _read(path)
    if not payload:
        raise TeardownReceiptError(f"teardown receipt {receipt_id!r} was not found")
    if str(payload.get("receipt_id") or "") != str(receipt_id).strip():
        raise TeardownReceiptError(
            f"teardown receipt {receipt_id!r} has a mismatched embedded ID"
        )
    return payload


_IDENTITY_SECRET_KEYS = (
    "access_key",
    "api_key",
    "credential",
    "iam_token",
    "password",
    "presigned",
    "private_key",
    "secret",
    "token",
)


def _clean_identity(value: object, *, key: str = "") -> object:
    """Validate recovery identity instead of silently redacting it."""

    lowered = key.lower()
    if any(marker in lowered for marker in _IDENTITY_SECRET_KEYS):
        raise TeardownReceiptError(
            f"refusing to persist secret-shaped cleanup identity field {key!r}"
        )
    if isinstance(value, Mapping):
        return {
            str(name): _clean_identity(item, key=str(name))
            for name, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_clean_identity(item, key=key) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        for pattern in _SECRET_PATTERNS[:-1]:
            if pattern.search(value):
                raise TeardownReceiptError(
                    f"refusing to persist secret-shaped cleanup identity field {key!r}"
                )
        if "://" in value:
            parsed = urlparse(value)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise TeardownReceiptError(
                    "cleanup identity URLs must not contain credentials, query strings, "
                    "or fragments"
                )
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _merge_identity(
    existing: object, incoming: object, *, path: str = "identity"
) -> object:
    if existing in (None, "", {}, []):
        return incoming
    if incoming in (None, "", {}, []):
        return existing
    if isinstance(existing, Mapping) and isinstance(incoming, Mapping):
        mapping_merged = dict(existing)
        for key, value in incoming.items():
            if (
                path == "identity.storage_iam"
                and key != "generations"
                and mapping_merged.get(str(key)) not in (None, "", {}, [])
                and mapping_merged.get(str(key)) != value
            ):
                # Compatibility scalar fields describe the first generation;
                # the authoritative complete set lives in generations.
                continue
            mapping_merged[str(key)] = _merge_identity(
                mapping_merged.get(str(key)), value, path=f"{path}.{key}"
            )
        return mapping_merged
    if isinstance(existing, list) and isinstance(incoming, list):
        list_merged = list(existing)
        seen = {json.dumps(item, sort_keys=True, default=str) for item in list_merged}
        for item in incoming:
            marker = json.dumps(item, sort_keys=True, default=str)
            if marker not in seen:
                if isinstance(item, Mapping):
                    identity_keys = (
                        ("instance_id", "agent_name")
                        if path == "identity.agents"
                        else (
                            "agent_name",
                            "cluster_id",
                            "context",
                            "run_id",
                            "operation_id",
                            "service_account_id",
                        )
                    )
                    identity_key = next(
                        (
                            key
                            for key in identity_keys
                            if item.get(key) not in (None, "")
                        ),
                        "",
                    )
                    if identity_key:
                        matches = [
                            index
                            for index, saved in enumerate(list_merged)
                            if isinstance(saved, Mapping)
                            and saved.get(identity_key) == item.get(identity_key)
                        ]
                        if len(matches) > 1:
                            raise TeardownReceiptError(
                                f"ambiguous cleanup identity at {path} for "
                                f"{identity_key}={item.get(identity_key)!r}"
                            )
                        if matches:
                            index = matches[0]
                            list_merged[index] = _merge_identity(
                                list_merged[index],
                                item,
                                path=f"{path}[{identity_key}]",
                            )
                            seen.add(
                                json.dumps(
                                    list_merged[index], sort_keys=True, default=str
                                )
                            )
                            continue
                list_merged.append(item)
                seen.add(marker)
        return list_merged
    if existing != incoming:
        raise TeardownReceiptError(
            f"immutable cleanup identity conflict at {path}: {existing!r} != {incoming!r}"
        )
    return existing


def _root_identity(
    identity: Mapping[str, Any], *, phase: str, resource: str
) -> dict[str, Any]:
    """Canonicalize resource identity so one project receipt can hold many resources."""

    root = dict(identity)
    resource_fields: tuple[str, ...] = ()
    collection = ""
    if phase == "agent":
        collection = "agents"
        resource_fields = (
            "agent_name",
            "instance_id",
            "operation_id",
            "service_account_id",
        )
    elif phase in {"cluster", "controller"}:
        collection = "clusters"
        resource_fields = (
            "context",
            "controller_context",
            "cluster_id",
            "cluster_name",
            "kubeconfig_path",
            "operation_id",
        )
    elif phase == "workflow":
        collection = "workflows"
        resource_fields = (
            "run_id",
            "workflow_s3_uri",
            "sky_job_id",
            "submission_status",
        )
    elif phase == "storage_iam":
        collection = "storage_iam"
        resource_fields = (
            "service_account_id",
            "service_account_name",
            "ownership",
            "iam_key_ids",
        )
    # ``cluster_absent`` is observed state, not immutable identity.  Keeping it
    # at the project root made a successful cleanup receipt for an old cluster
    # prevent cleanup of a later cluster created with the same context.
    if phase in {"cluster", "controller"}:
        root.pop("cluster_absent", None)
    scoped = {
        key: root.pop(key)
        for key in resource_fields
        if root.get(key) not in (None, "", [], {})
    }
    if collection in {"agents", "clusters", "workflows"} and scoped:
        identity_key = {
            "agents": "agent_name",
            # One context may have several immutable IDs across failed/retried
            # attempts.  Preserve each ID independently instead of treating
            # audit history as an immutable-context conflict.
            "clusters": "cluster_id",
            "workflows": "run_id",
        }[collection]
        scoped.setdefault(identity_key, resource)
        root[collection] = _merge_identity(root.get(collection, []), [scoped])
    elif collection == "storage_iam" and scoped:
        existing = root.get(collection, {})
        if isinstance(existing, Mapping) and "generations" in existing:
            generations = existing.get("generations")
            generations = list(generations) if isinstance(generations, list) else []
        elif isinstance(existing, Mapping) and existing:
            # Additive v2 migration from the original single-generation shape.
            generations = [dict(existing)]
        else:
            generations = []
        account_id = str(scoped.get("service_account_id") or "")
        matching = [
            item
            for item in generations
            if isinstance(item, Mapping)
            and str(item.get("service_account_id") or "") == account_id
        ]
        if matching:
            saved = matching[0]
            incoming = dict(scoped)
            # Ownership proof is monotonic. A later provider observation marked
            # "unverified" must not erase durable NPA creation provenance.
            if saved.get("ownership") == "npa" and incoming.get("ownership") != "npa":
                incoming.pop("ownership", None)
            merged_generation = _merge_identity(
                saved, incoming, path="identity.storage_iam.generation"
            )
            generations[generations.index(saved)] = merged_generation
        else:
            generations.append(dict(scoped))
        compatibility = dict(existing) if isinstance(existing, Mapping) else {}
        compatibility.pop("generations", None)
        if not compatibility or compatibility.get("service_account_id") == account_id:
            for key, value in scoped.items():
                if key == "ownership" and compatibility.get(key) == "npa":
                    continue
                compatibility[key] = value
        compatibility["generations"] = generations
        root[collection] = compatibility
    return root


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
    identity: Mapping[str, Any] | None = None,
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
    cleaned_identity_value = _clean_identity(dict(identity or {}))
    if not isinstance(cleaned_identity_value, Mapping):  # pragma: no cover - defensive
        raise TeardownReceiptError("cleanup identity must be a mapping")
    cleaned_identity = dict(cleaned_identity_value)
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
    if isinstance(event, dict):
        event["identity"] = cleaned_identity
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
                "identity": {},
                "events": [],
            }
        if str(payload.get("project_id") or "") not in {"", str(project_id or "")}:
            raise TeardownReceiptError(
                f"receipt identity mismatch at {path}; refusing to mix projects"
            )
        events = list(payload.get("events") or [])
        event["sequence"] = len(events) + 1
        events.append(event)
        payload["events"] = events
        payload["updated_at"] = now
        payload["schema_version"] = SCHEMA_VERSION
        existing_identity = dict(payload.get("identity") or {})
        if cleaned_phase in {"cluster", "controller"}:
            # Migrate receipts written before controller observations and
            # operation IDs were scoped to an immutable cluster entry.
            existing_identity.pop("cluster_absent", None)
            legacy_operation_id = existing_identity.pop("operation_id", None)
            clusters = existing_identity.get("clusters")
            if (
                legacy_operation_id
                and isinstance(clusters, list)
                and len(clusters) == 1
                and isinstance(clusters[0], Mapping)
                and not clusters[0].get("operation_id")
            ):
                clusters[0] = {**clusters[0], "operation_id": legacy_operation_id}
        durable_identity = _root_identity(
            cleaned_identity, phase=cleaned_phase, resource=cleaned_resource
        )
        payload["identity"] = _merge_identity(existing_identity, durable_identity)
        if project_alias and not payload.get("project_alias"):
            payload["project_alias"] = project_alias
        if project_id and not payload.get("project_id"):
            payload["project_id"] = project_id
        _write_atomic(path, payload)
    return path


def list_teardown_receipts(
    *,
    project_alias: str = "",
    project_id: str = "",
    legacy: str = "include",
    strict: bool = False,
) -> list[dict[str, Any]]:
    """List receipts with explicit project and legacy-identity semantics.

    ``legacy`` is ``include`` (backward-compatible all-receipts view), ``exclude``
    (safe project-scoped view), or ``only`` (operator audit of unscoped history).
    """

    if legacy not in {"include", "exclude", "only"}:
        raise ValueError("legacy receipt selector must be include, exclude, or only")
    root = receipt_root()
    if not root.is_dir():
        return []
    receipts: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = _read(path)
        except TeardownReceiptError:
            if strict:
                raise
            if project_alias or project_id or legacy == "exclude":
                continue
            receipts.append(
                {
                    "schema_version": "unreadable",
                    "receipt_id": path.stem,
                    "path": str(path),
                    "project_alias": "",
                    "project_id": "",
                    "operational_status": "unresolved",
                    "audit_only": True,
                    "unresolved_actions": ["inspect unreadable receipt"],
                    "updated_at": "",
                }
            )
            continue
        receipt_alias = str(payload.get("project_alias") or "")
        receipt_project_id = str(payload.get("project_id") or "")
        identity = payload.get("identity")

        def collect(mapping: object, key: str) -> set[str]:
            found: set[str] = set()
            if isinstance(mapping, Mapping):
                for item_key, value in mapping.items():
                    if str(item_key) == key and isinstance(value, str) and value:
                        found.add(value)
                    elif isinstance(value, (Mapping, list)):
                        found.update(collect(value, key))
            elif isinstance(mapping, list):
                for value in mapping:
                    found.update(collect(value, key))
            return found

        subject_project_ids = collect(identity, "project_id") | (
            {receipt_project_id} if receipt_project_id else set()
        )
        subject_aliases = collect(identity, "project_alias") | (
            {receipt_alias} if receipt_alias else set()
        )
        is_legacy = not subject_aliases and not subject_project_ids
        if legacy == "only" and not is_legacy:
            continue
        if legacy == "exclude" and is_legacy:
            continue
        if project_id and project_id not in subject_project_ids:
            continue
        if project_alias and not project_id and project_alias not in subject_aliases:
            continue
        unresolved = [
            f"{event.get('phase', 'unknown')}:{event.get('resource', 'unknown')}"
            for event in payload.get("events") or []
            if isinstance(event, dict)
            and str(event.get("terminal_state") or "") not in TERMINAL_STATES
        ]
        terminal = receipt_is_terminal(payload)
        if not terminal and not unresolved:
            unresolved.append("inspect incomplete or conflicting terminal evidence")
        payload["subject"] = {
            "project_ids": sorted(subject_project_ids),
            "project_aliases": sorted(subject_aliases),
        }
        payload["audit_only"] = True
        payload["operational_status"] = "terminal" if terminal else "unresolved"
        payload["unresolved_actions"] = unresolved
        payload["path"] = str(path)
        receipts.append(payload)
    return sorted(
        receipts, key=lambda item: str(item.get("updated_at") or ""), reverse=True
    )


_AGENT_TERRAFORM_GRAPH = frozenset(
    {
        "compute_instance",
        "boot_disk",
        "network",
        "subnet",
        "security_group",
        "public_ip",
    }
)


def teardown_event_authorizes_convergence(event: Mapping[str, Any]) -> bool:
    """Validate terminal evidence against the action that produced it.

    A terminal word is never enough for a destructive consumer.  This function
    is deliberately shared by cleanup, status, receipt aggregation, and pruning
    so those surfaces cannot assign different meanings to the same evidence.
    Unknown phases retain the historical terminal-state contract; cloud phases
    use action-specific schemas and exact verification fields.
    """

    phase = str(event.get("phase") or "")
    state = str(event.get("terminal_state") or "").lower()
    action = event.get("action")
    action = action if isinstance(action, Mapping) else {}
    verification = event.get("verification")
    verification = verification if isinstance(verification, Mapping) else {}
    identity = event.get("identity")
    identity = identity if isinstance(identity, Mapping) else {}
    errors = event.get("errors")
    if not isinstance(errors, list) or errors:
        return False
    kind = str(action.get("kind") or "")
    if phase.startswith("project_destroy_"):
        return bool(
            state == "completed"
            and kind == "npa_guarded_phase"
            and verification.get("converged") is True
        )
    if phase == "workflow_audit":
        return bool(
            kind == "read_only_managed_job_audit"
            and verification.get("nonterminal_job_ids") == []
            and (
                (
                    state in {"not_submitted", "verified_absent"}
                    and not verification.get("detail")
                )
                or (
                    state == "operator_attested"
                    and verification.get("operator_attestation") is True
                    and verification.get("queue_state") == "SKIPPED_BY_OPERATOR"
                )
            )
        )
    if phase == "workflow":
        outcome = str(verification.get("outcome") or "")
        return bool(
            (state == "not_submitted" and kind == "none" and outcome == "not_submitted")
            or (
                state == "verified_absent"
                and kind == "none"
                and outcome == "already_absent"
            )
            or (
                state == "cancelled"
                and kind == "managed_job_cancel"
                and outcome == "cancelled"
            )
            or (
                state == "terminal"
                and kind == "none"
                and outcome in {"terminal", "no_cancellation_needed"}
            )
        )
    if phase == "agent":
        graph = verification.get("terraform_dependency_graph")
        iam_authoritative = bool(
            verification.get("iam_cleanup_complete") is True
            and verification.get("iam_disposition") in {"absent", "deleted"}
        )
        return bool(
            state in {"verified_absent", "verified_deleted"}
            and kind == "terraform_agent_destroy"
            and bool(str(identity.get("project_id") or "").strip())
            and bool(str(identity.get("instance_id") or "").strip())
            and str(identity.get("agent_name") or event.get("resource") or "")
            == str(event.get("resource") or "")
            and verification.get("exact_instance_absent") is True
            and verification.get("terraform_destroy_completed") is True
            and isinstance(graph, list)
            and _AGENT_TERRAFORM_GRAPH.issubset(graph)
            and iam_authoritative
        )
    if phase == "bucket":
        return bool(
            state in {"verified_absent", "verified_deleted"}
            and kind in {"none", "bucket_delete", "scheduled_bucket_purge"}
            and verification.get("bucket_absent") is True
        )
    if phase == "storage_iam":
        provider_outcome = str(verification.get("provider_outcome") or "")
        if not str(identity.get("service_account_id") or "").strip():
            return False
        if kind in {"none", "exact_provider_check", "remove_verified_absent_local_evidence"}:
            return bool(
                state == "verified_absent"
                and provider_outcome == "verified_absent"
                and verification.get("exact_service_account_absent") is True
            )
        if kind == "delete_npa_owned_service_account":
            return bool(
                state == "verified_deleted"
                and provider_outcome == "verified_absent"
                and verification.get("exact_service_account_absent") is True
                and verification.get("access_keys_absent") is True
                and verification.get("local_recovery_retired") is True
            )
        if kind == "preserve_unowned_account_remove_npa_access":
            return bool(
                state == "completed"
                and provider_outcome == "retained_account_access_resolved"
                and verification.get("service_account_preserved") is True
                and verification.get("access_state_absent") is True
                and verification.get("retained_account_dependents_verified") is True
            )
        return False
    if phase == "cluster":
        return bool(
            (
                state == "verified_absent"
                and kind == "exact_provider_check"
                and verification.get("provider_absence") == "verified"
            )
            or (
                state == "verified_deleted"
                and kind == "terraform_full_cluster_destroy"
                and verification.get("terraform_destroy") == "completed"
            )
            or (
                state == "verified_deleted"
                and kind == "delete_exact_operation_inventory"
                and verification.get("state_consumers_absent") is True
                and verification.get("errors") == []
            )
        )
    if phase == "controller":
        return bool(
            state in {"verified_absent", "verified_deleted"}
            and kind == "controller_cleanup"
            and verification.get("local_state_removed_after_remote_absence") is True
            and verification.get("remote_controller_pods") == []
        )
    return bool(state in TERMINAL_STATES)


def _event_generation_key(
    event: Mapping[str, Any], *, receipt_id: str
) -> tuple[str, ...]:
    """Return one immutable resource-generation key for receipt aggregation."""

    phase = str(event.get("phase") or "")
    resource = str(event.get("resource") or "")
    identity = event.get("identity")
    identity = identity if isinstance(identity, Mapping) else {}
    project_id = str(event.get("project_id") or identity.get("project_id") or "")
    selectors: tuple[str, ...]
    if phase == "agent":
        selectors = (
            str(identity.get("agent_name") or resource),
            str(identity.get("instance_id") or ""),
        )
    elif phase == "storage_iam":
        selectors = (str(identity.get("service_account_id") or resource),)
    elif phase in {"cluster", "controller"}:
        selectors = (
            str(
                identity.get("cluster_id")
                or identity.get("controller_context")
                or identity.get("context")
                or event.get("context")
                or resource
            ),
        )
    elif phase in {"workflow", "workflow_audit"}:
        selectors = (str(identity.get("run_id") or resource),)
    else:
        selectors = (resource,)
    if not all(selectors):
        # Incomplete identity must not coalesce with a complete sibling.  Keep it
        # independently actionable and let the action schema classify it unsafe.
        selectors = (*selectors, f"incomplete@{receipt_id}")
    return (phase, project_id, *selectors)


def _parse_event_time(event: Mapping[str, Any]) -> datetime | None:
    value = event.get("recorded_at")
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _latest_generation_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Select newest evidence without treating cross-file sequence as a clock."""

    per_receipt: dict[str, dict[str, Any]] = {}
    for event in events:
        receipt_id = str(event.get("_receipt_id") or "")
        previous = per_receipt.get(receipt_id)
        if previous is None or int(event.get("sequence") or 0) > int(
            previous.get("sequence") or 0
        ):
            per_receipt[receipt_id] = event
    candidates = list(per_receipt.values())
    if len(candidates) == 1:
        return candidates[0]
    dated = [(event, _parse_event_time(event)) for event in candidates]
    if any(parsed is None for _event, parsed in dated):
        return {
            "phase": str(candidates[0].get("phase") or ""),
            "resource": str(candidates[0].get("resource") or ""),
            "terminal_state": "evidence_conflict",
            "action": {"kind": "receipt_generation_conflict"},
            "verification": {"cross_receipt_order": "unavailable"},
            "errors": ["same resource generation has unordered cross-receipt evidence"],
            "recorded_at": "",
            "sequence": 0,
        }
    newest_time = max(parsed for _event, parsed in dated if parsed is not None)
    newest = [event for event, parsed in dated if parsed == newest_time]
    if len(newest) != 1:
        return {
            "phase": str(newest[0].get("phase") or ""),
            "resource": str(newest[0].get("resource") or ""),
            "terminal_state": "evidence_conflict",
            "action": {"kind": "receipt_generation_conflict"},
            "verification": {"cross_receipt_order": "conflicting"},
            "errors": ["same resource generation has conflicting newest receipt evidence"],
            "recorded_at": str(newest[0].get("recorded_at") or ""),
            "sequence": 0,
        }
    return newest[0]


def latest_resource_generation_events(
    *,
    project_alias: str = "",
    project_id: str = "",
    phase: str = "",
    resource: str = "",
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Return newest evidence for every matching immutable resource generation."""

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for receipt in list_teardown_receipts(strict=strict):
        receipt_project_id = str(receipt.get("project_id") or "")
        receipt_alias = str(receipt.get("project_alias") or "")
        if project_id and receipt_project_id != project_id:
            continue
        if project_alias and receipt_alias != project_alias:
            continue
        receipt_id = str(receipt.get("receipt_id") or "")
        for event in receipt.get("events") or []:
            if not isinstance(event, dict):
                continue
            if project_id and str(event.get("project_id") or "") != project_id:
                continue
            if phase and str(event.get("phase") or "") != phase:
                continue
            if resource and str(event.get("resource") or "") != resource:
                continue
            annotated = dict(event)
            annotated["_receipt_id"] = receipt_id
            key = _event_generation_key(annotated, receipt_id=receipt_id)
            grouped.setdefault(key, []).append(annotated)
    return [_latest_generation_event(events) for events in grouped.values()]


def latest_phase_states(
    *, project_alias: str = "", project_id: str = "", strict: bool = False
) -> dict[str, dict[str, Any]]:
    """Return phase convergence while preserving every immutable generation.

    The returned event remains backward-compatible for display callers.  It is
    the first unresolved/conflicting generation when any exist; otherwise it is
    a representative terminal event after *all* sibling generations validate.
    Receipt-local sequence is compared only within its own file.
    """

    grouped: dict[str, dict[tuple[str, ...], list[dict[str, Any]]]] = {}
    for receipt in list_teardown_receipts(strict=strict):
        receipt_project_id = str(receipt.get("project_id") or "")
        receipt_alias = str(receipt.get("project_alias") or "")
        if project_id and receipt_project_id != project_id:
            continue
        if project_alias and receipt_alias != project_alias:
            continue
        for event in receipt.get("events") or []:
            if not isinstance(event, dict):
                continue
            if project_id and str(event.get("project_id") or "") != project_id:
                continue
            phase = str(event.get("phase") or "")
            if not phase:
                continue
            annotated = dict(event)
            annotated["_receipt_id"] = str(receipt.get("receipt_id") or "")
            key = _event_generation_key(
                annotated, receipt_id=str(receipt.get("receipt_id") or "")
            )
            grouped.setdefault(phase, {}).setdefault(key, []).append(annotated)
    states: dict[str, dict[str, Any]] = {}
    for phase, generations in grouped.items():
        current = [_latest_generation_event(events) for events in generations.values()]
        unresolved = [
            event for event in current if not teardown_event_authorizes_convergence(event)
        ]
        selected = sorted(
            unresolved or current,
            key=lambda event: (
                str(event.get("recorded_at") or ""),
                str(event.get("_receipt_id") or ""),
                str(event.get("resource") or ""),
            ),
            reverse=True,
        )[0]
        selected = dict(selected)
        selected["resource_generation_count"] = len(current)
        selected["unresolved_resource_generation_count"] = len(unresolved)
        states[phase] = selected
    return states


def receipt_is_terminal(receipt: Mapping[str, Any]) -> bool:
    generations: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    receipt_id = str(receipt.get("receipt_id") or "")
    for event in receipt.get("events") or []:
        if isinstance(event, Mapping):
            copied = dict(event)
            copied["_receipt_id"] = receipt_id
            key = _event_generation_key(copied, receipt_id=receipt_id)
            generations.setdefault(key, []).append(copied)
    latest = [_latest_generation_event(events) for events in generations.values()]
    return bool(latest) and all(
        teardown_event_authorizes_convergence(event) for event in latest
    )


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
