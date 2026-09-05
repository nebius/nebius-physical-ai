"""Goal-level NPA agent trajectory collection.

This module is shipped to the agent VM, so it intentionally depends only on the
standard library plus the runtime's boto3 installation. Product work must never
fail merely because telemetry cannot be uploaded: upload failures enter an
owner-only outbox and are retried at the next episode boundary.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import ipaddress
import json
import logging
import os
import re
import stat
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

SCHEMA_VERSION = "npa.agent.trajectory.v1"
LOGGER = logging.getLogger(__name__)
_OUTBOX_LOCK = threading.RLock()
_SECRET_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|(?:secret|access|private)[_-]?(?:access[_-]?)?key|"
    r"secret|password|passwd|authorization|credential|cookie|"
    r"(?:access|refresh|identity|id|bearer)[_-]?token|"
    r"(^|[_-])token($|[_-])|environment|environ|(^|[_-])env($|[_-])|"
    r"image[_-]?(?:data|bytes)|raw[_-]?(?:data|image)|base64)"
)
_SECRET_TEXT_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.DOTALL),
    re.compile(r"(?i)data:[^\s,]*;base64,[^\s\"']*"),
    re.compile(
        r"(?i)(?:[a-z0-9_]*)(api[_-]?key|(?:secret|access|private)[_-]?(?:access[_-]?)?key|"
        r"secret|password|passwd|token|authorization|credential)"
        r"[a-z0-9_]*\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)"
    ),
    re.compile(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)(?:AKIA|ASIA)[0-9A-Z]{16}"),
    re.compile(r"\b(?:hf_|gh[pousr]_|github_pat_|sk-)[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*"),
    re.compile(r"\b(?:xox[baprs]-|xapp-)[A-Za-z0-9-]{8,}\b"),
    re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{12,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bnvapi-[A-Za-z0-9_-]{12,}\b"),
)
_URI_RE = re.compile(r"(?:s3|https?|gs|file|ssh)://[^\s\"'<>]+")
_INFRA_RE = re.compile(
    r"(?i)\b(?:tenant|project|capacityblockgroup|computeinstance|mk8scluster|"
    r"mk8snodegroup|storagebucket|registry|network|serviceaccount|cluster)-[a-z0-9]{8,}\b"
    r"|(?<![a-z0-9])(?:e00(?!(?:[0-9a-f]{37}|[0-9a-f]{61})(?![a-z0-9]))|u00)"
    r"[a-z0-9]{12,}(?![a-z0-9])"
    r"|\bcr\.[a-z0-9-]+\.nebius\.cloud/[^\s\"'<>]+"
)
_IP_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])|(?<![\w:])[0-9a-fA-F:]*:[0-9a-fA-F:]+(?![\w:])")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_REDACTION_MARKER_RE = re.compile(
    r"(<(?:redacted|uri-ref|infra-ref|address-ref|inline-data-ref|tenant-ref|bucket-ref|private-ref)>"
    r")"
)


class AgentRunDataError(RuntimeError):
    """Trajectory collection configuration or integrity error."""


class AgentRunDataConflict(AgentRunDataError):
    """Different finalized bytes were supplied for one episode."""


class CollectionStatus:
    COLLECTED = "collected"
    PENDING = "pending"
    DISABLED = "disabled"


@dataclass(frozen=True)
class DatasetConfig:
    tenant_id: str
    dataset_uri: str
    bucket: str
    prefix: str


def _outbox_dir() -> Path:
    return Path(
        os.environ.get("NPA_AGENT_DATASET_OUTBOX", "~/.npa/agent-dataset-outbox")
    ).expanduser()


def resolve_dataset_config(
    *, active_tenant_id: str = "", active_bucket: str = ""
) -> DatasetConfig | None:
    """Resolve and scope-check owner-provided collection configuration."""
    tenant_id = os.environ.get("NPA_AGENT_DATASET_TENANT_ID", "").strip()
    dataset_uri = os.environ.get("NPA_AGENT_DATASET_URI", "").strip()
    if not tenant_id and not dataset_uri:
        return None
    if not tenant_id or not dataset_uri:
        raise AgentRunDataError(
            "NPA_AGENT_DATASET_TENANT_ID and NPA_AGENT_DATASET_URI must be set together"
        )
    parsed = urlparse(dataset_uri)
    if (
        parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment
        or parsed.username or parsed.password or ":" in parsed.netloc
    ):
        raise AgentRunDataError("NPA_AGENT_DATASET_URI must be an unsigned s3:// URI")
    if not active_tenant_id.strip() or not active_bucket.strip():
        raise AgentRunDataError("verified active deployment tenant and bucket are required")
    if tenant_id != active_tenant_id.strip():
        raise AgentRunDataError("agent dataset tenant does not match the active deployment")
    if parsed.netloc != active_bucket.strip():
        raise AgentRunDataError("agent dataset bucket does not match the active deployment")
    return DatasetConfig(
        tenant_id=tenant_id,
        dataset_uri=dataset_uri,
        bucket=parsed.netloc,
        prefix=parsed.path.lstrip("/").rstrip("/"),
    )


def _storage_client() -> Any:
    try:
        import boto3
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise AgentRunDataError("boto3 is unavailable for agent run data") from exc
    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get(
        "NEBIUS_S3_ENDPOINT"
    )
    kwargs = {"endpoint_url": endpoint or None}
    return boto3.client("s3", **kwargs)


def _s3(storage: Any | None) -> Any:
    client = storage if storage is not None else _storage_client()
    return getattr(client, "s3", client)


class _UnavailableStorage:
    """Keep a failed injected factory from falling through to live credentials."""

    def head_bucket(self, **kwargs: Any) -> None:
        raise AgentRunDataError("trajectory storage factory is unavailable")

    def delete_object(self, **kwargs: Any) -> None:
        return None


def verify_destination(config: DatasetConfig, *, storage: Any | None = None) -> None:
    """Prove bucket access and write/read/delete behavior before enabling."""
    s3 = _s3(storage)
    probe_key = "/".join(
        item
        for item in (config.prefix, ".npa-probes", uuid.uuid4().hex)
        if item
    )
    body = b"npa-agent-dataset-probe"
    try:
        s3.head_bucket(Bucket=config.bucket)
        s3.put_object(Bucket=config.bucket, Key=probe_key, Body=body)
        fetched = s3.get_object(Bucket=config.bucket, Key=probe_key)["Body"].read()
        if fetched != body:
            raise AgentRunDataError("dataset probe read-after-write mismatch")
    except AgentRunDataError:
        raise
    except Exception as exc:
        raise AgentRunDataError("agent dataset destination is not writable") from exc
    finally:
        try:
            s3.delete_object(Bucket=config.bucket, Key=probe_key)
        except Exception:
            LOGGER.debug("agent dataset probe cleanup failed")


def _redact_string(value: str) -> str:
    # A data URI may contain wrapped base64 or whitespace. Removing its entire
    # string prevents fragments of the inline image from surviving a regex stop.
    if re.search(r"(?i)data:[^\s,]*;base64,", value):
        return "<inline-data-ref>"
    text = value
    for pattern in _SECRET_TEXT_PATTERNS:
        text = pattern.sub("<redacted>", text)

    text = _URI_RE.sub("<uri-ref>", text)
    text = _INFRA_RE.sub("<infra-ref>", text)

    def redact_ip(match: re.Match[str]) -> str:
        try:
            ipaddress.ip_address(match.group(0))
        except ValueError:
            return match.group(0)
        return "<address-ref>"

    return _IP_RE.sub(redact_ip, text)


def redact(value: Any) -> Any:
    """Remove secret payloads and private references before JSON serialization.

    Unstructured customer names require an operator-private literal denylist;
    recognizable infrastructure and credential forms are always removed.
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise AgentRunDataError("trajectory mapping keys must be strings")
            safe_key = _redact_string(key)
            if safe_key != key:
                safe_key += "-" + hashlib.sha256(key.encode()).hexdigest()[:12]
            result[safe_key] = "<redacted>" if _SECRET_KEY_RE.search(key) else redact(item)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise AgentRunDataError("trajectory values must be JSON data, never raw objects or bytes")


def _redact_identifiers(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            safe_key = _redact_identifiers(key, replacements)
            if safe_key != key:
                safe_key += "-" + hashlib.sha256(key.encode()).hexdigest()[:12]
            result[safe_key] = _redact_identifiers(item, replacements)
        return result
    if isinstance(value, list):
        return [_redact_identifiers(item, replacements) for item in value]
    if isinstance(value, str):
        # Preserve only exact internal markers. Scan surrounding text and key
        # digest suffixes too; caller-supplied text can imitate a suffix.
        parts = _REDACTION_MARKER_RE.split(value)
        for index in range(0, len(parts), 2):
            for identifier, replacement in replacements.items():
                if identifier:
                    parts[index] = parts[index].replace(identifier, replacement)
        return "".join(parts)
    return value


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _private_replacements(config: DatasetConfig) -> dict[str, str]:
    replacements = {
        config.tenant_id: "<tenant-ref>",
        config.bucket: "<bucket-ref>",
        config.dataset_uri: "<uri-ref>",
    }
    prefix = config.prefix.strip("/")
    if prefix:
        # The configured prefix is private even when a tool reports it without
        # its bucket, for example as part of a local configuration directory.
        replacements[prefix] = "<private-ref>"
    configured = os.environ.get("NPA_AGENT_DATASET_REDACTION_FILE", "").strip()
    if configured:
        path = Path(configured).expanduser()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode) or mode & 0o077:
            raise AgentRunDataError("trajectory redaction configuration must be an owner-only file")
        data = json.loads(path.read_text(encoding="utf-8"))
        literals = data.get("literals") if isinstance(data, dict) else None
        if not isinstance(literals, list) or any(not isinstance(item, str) or not item for item in literals):
            raise AgentRunDataError("trajectory redaction configuration requires nonempty string literals")
        replacements.update({literal: "<private-ref>" for literal in literals})
    return dict(sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True))


def _sanitize_payload(payload: dict[str, Any], config: DatasetConfig) -> dict[str, Any]:
    sanitized = _redact_identifiers(redact(payload), _private_replacements(config))
    # This is the sole permitted concrete routing identifier in an episode.
    sanitized["scope"]["tenant_id"] = config.tenant_id
    return sanitized


def _destination_digest(config: DatasetConfig) -> str:
    return hashlib.sha256(_canonical_json({
        "tenant_id": config.tenant_id, "bucket": config.bucket, "prefix": config.prefix,
    }).encode()).hexdigest()


def _content_sha256(payload: dict[str, Any]) -> str:
    candidate = json.loads(_canonical_json(payload))
    candidate["collection"]["content_sha256"] = ""
    return hashlib.sha256(_canonical_json(candidate).encode()).hexdigest()


def _object_key(config: DatasetConfig, payload: dict[str, Any]) -> str:
    started = datetime.fromisoformat(
        str(payload["timing"]["started_at"]).replace("Z", "+00:00")
    )
    stem = f"{payload['episode_id']}-{payload['collection']['content_sha256']}.json"
    return "/".join(
        item
        for item in (
            config.prefix,
            "episodes",
            f"{started:%Y}",
            f"{started:%m}",
            f"{started:%d}",
            stem,
        )
        if item
    )


def _secure_outbox() -> Path:
    outbox = Path(os.path.abspath(_outbox_dir()))
    with _OUTBOX_LOCK:
        # Reject symlinks before mkdir, including parents. Checking only the
        # final directory would already create or write through a parent link.
        for component in (outbox, *outbox.parents):
            try:
                mode = component.lstat().st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(mode):
                raise AgentRunDataError("agent dataset outbox path must not contain symlinks")
        outbox.mkdir(mode=0o700, parents=True, exist_ok=True)
        if stat.S_ISLNK(outbox.lstat().st_mode):
            raise AgentRunDataError("agent dataset outbox must not be a symlink")
        outbox.chmod(0o700)
    return outbox


def _validated_body(config: DatasetConfig, payload: dict[str, Any]) -> bytes:
    """Check finalized integrity and run the sanitizer again before persistence."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise AgentRunDataError("unsupported trajectory schema")
    if not all(_SAFE_ID_RE.fullmatch(str(payload.get(field, ""))) for field in ("episode_id", "session_id")):
        raise AgentRunDataError("trajectory episode and session IDs must be safe stable identifiers")
    if payload.get("scope", {}).get("tenant_id") != config.tenant_id:
        raise AgentRunDataError("pending trajectory tenant does not match its destination")
    collection = payload.get("collection", {})
    if collection.get("status") != CollectionStatus.PENDING:
        raise AgentRunDataError("immutable raw trajectory must retain pending delivery status")
    if collection.get("content_sha256") != _content_sha256(payload):
        raise AgentRunDataError("finalized trajectory content hash mismatch")
    if _sanitize_payload(payload, config) != payload:
        raise AgentRunDataError("finalized trajectory failed pre-write privacy verification")
    return _canonical_json(payload).encode()


def _write_outbox(config: DatasetConfig, payload: dict[str, Any], *, conflict: bool = False) -> Path:
    """Preserve exact finalized bytes in a destination-bound private envelope."""
    body = _validated_body(config, payload)
    envelope = {
        "schema_version": "npa.agent.trajectory-outbox.v1",
        "destination_sha256": _destination_digest(config),
        "payload_sha256": hashlib.sha256(body).hexdigest(),
        "payload": payload,
        "failure": "episode_conflict" if conflict else "delivery_pending",
    }
    encoded = _canonical_json(envelope).encode()
    outbox = _secure_outbox()
    path = outbox / (
        f"{payload['episode_id']}-{payload['collection']['content_sha256']}.json"
    )
    with _OUTBOX_LOCK:
        if path.exists():
            if not stat.S_ISREG(path.lstat().st_mode):
                raise AgentRunDataError("agent dataset outbox record must be a regular file")
            existing = json.loads(path.read_bytes())
            # Delivery diagnostics may differ, but routing and finalized bytes may not.
            if any(existing.get(key) != envelope[key] for key in (
                "schema_version", "destination_sha256", "payload_sha256", "payload",
            )):
                raise AgentRunDataError("outbox key conflicts with different bytes")
            path.chmod(0o600)
            return path
        temporary = outbox / f".{path.name}.{uuid.uuid4().hex}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            # A process-local lock cannot serialize another agent process.
            # Hard-link publication is atomic and cannot replace its record.
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                if not stat.S_ISREG(path.lstat().st_mode):
                    raise AgentRunDataError("agent dataset outbox record must be a regular file") from None
                existing = json.loads(path.read_bytes())
                if any(existing.get(key) != envelope[key] for key in (
                    "schema_version", "destination_sha256", "payload_sha256", "payload",
                )):
                    raise AgentRunDataError("concurrent outbox key conflicts with different bytes") from None
            path.chmod(0o600)
            directory_fd = os.open(outbox, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()
    return path


def _missing_object(error: Exception) -> bool:
    if isinstance(error, KeyError):  # in-memory injected stores
        return True
    response = getattr(error, "response", {})
    return isinstance(response, dict) and str(response.get("Error", {}).get("Code")) in {
        "NoSuchKey", "404", "NotFound",
    }


def _put_immutable(config: DatasetConfig, key: str, body: bytes, s3: Any) -> None:
    try:
        existing = s3.get_object(Bucket=config.bucket, Key=key)["Body"].read()
    except Exception as exc:
        if not _missing_object(exc):
            raise
        existing = None
    if existing is not None:
        if existing != body:
            raise AgentRunDataConflict("immutable trajectory key contains different bytes")
        return
    try:
        s3.put_object(
            Bucket=config.bucket,
            Key=key,
            Body=body,
            IfNoneMatch="*",
        )
    except Exception:
        # A concurrent identical writer may have won the conditional put.
        LOGGER.debug("conditional trajectory write did not create a new object")
    fetched = s3.get_object(Bucket=config.bucket, Key=key)["Body"].read()
    if fetched != body:
        raise AgentRunDataConflict("trajectory read-after-write mismatch")


def _put_and_verify(config: DatasetConfig, payload: dict[str, Any], s3: Any) -> str:
    body = _validated_body(config, payload)
    key = _object_key(config, payload)
    _put_immutable(config, key, body, s3)
    # Preserve every divergent payload, but make one atomic claim per episode.
    # A competing digest remains visible and is reported as a data-quality conflict.
    episode_digest = hashlib.sha256(payload["episode_id"].encode()).hexdigest()
    claim_key = "/".join(filter(None, (config.prefix, "episode-index", episode_digest + ".json")))
    claim = {
        "schema_version": "npa.agent.trajectory-claim.v1",
        "episode_id": payload["episode_id"],
        "content_sha256": payload["collection"]["content_sha256"],
    }
    _put_immutable(config, claim_key, _canonical_json(claim).encode(), s3)
    # Raw bytes never assert unverified delivery. Publish a separate immutable
    # receipt only after their exact read-after-write check has succeeded.
    receipt_key = "/".join(filter(None, (
        config.prefix, "receipts", episode_digest,
        payload["collection"]["content_sha256"] + ".json",
    )))
    receipt = {
        "schema_version": "npa.agent.trajectory-receipt.v1",
        "episode_id": payload["episode_id"],
        "content_sha256": payload["collection"]["content_sha256"],
        "payload_sha256": hashlib.sha256(body).hexdigest(),
        "status": CollectionStatus.COLLECTED,
    }
    _put_immutable(config, receipt_key, _canonical_json(receipt).encode(), s3)
    return key


def emit_trajectory(
    *,
    episode_id: str,
    session_id: str,
    request_content: str,
    intent: str,
    trajectory: list[dict[str, Any]],
    outcome: dict[str, Any],
    routing: dict[str, Any],
    versions: dict[str, Any],
    initial_state: dict[str, Any] | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    storage: Any | None = None,
    active_tenant_id: str = "",
    active_bucket: str = "",
) -> tuple[str, str]:
    """Finalize exactly one episode and return collection status plus id."""
    config = resolve_dataset_config(
        active_tenant_id=active_tenant_id, active_bucket=active_bucket
    )
    if config is None:
        return CollectionStatus.DISABLED, episode_id
    now = datetime.now(timezone.utc)
    started = started_at or now.isoformat()
    ended = ended_at or now.isoformat()
    try:
        latency_ms = max(
            0,
            int(
                (
                    datetime.fromisoformat(ended.replace("Z", "+00:00"))
                    - datetime.fromisoformat(started.replace("Z", "+00:00"))
                ).total_seconds()
                * 1000
            ),
        )
    except ValueError:
        latency_ms = 0
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "session_id": session_id,
        "scope": {
            "tenant_id": config.tenant_id,
            "dataset_role": "agent-finetuning-raw",
        },
        "timing": {
            "started_at": started,
            "ended_at": ended,
            "latency_ms": latency_ms,
        },
        "request": {"content": request_content, "intent": intent},
        "initial_state": initial_state or {},
        "trajectory": trajectory,
        "outcome": outcome,
        "routing": routing,
        "versions": versions,
        "redaction": {
            "applied": True,
            "fields_removed": [
                "credential-and-environment-payloads", "private-uris-and-addresses",
                "infra-identifiers", "operator-private-literals", "inline-data",
            ],
        },
        "collection": {"status": CollectionStatus.PENDING, "content_sha256": ""},
    }
    payload = _sanitize_payload(payload, config)
    payload["collection"]["content_sha256"] = _content_sha256(payload)
    _validated_body(config, payload)
    try:
        verify_destination(config, storage=storage)
        _put_and_verify(config, payload, _s3(storage))
        return CollectionStatus.COLLECTED, episode_id
    except Exception as exc:
        conflict = isinstance(exc, AgentRunDataConflict)
        if conflict:
            LOGGER.warning("trajectory content conflict; finalized record retained pending")
        _write_outbox(config, payload, conflict=conflict)
        return CollectionStatus.PENDING, episode_id


def flush_outbox(
    *,
    storage: Any | None = None,
    active_tenant_id: str = "",
    active_bucket: str = "",
) -> list[str]:
    """Flush pending records only after exact read-after-write proof."""
    config = resolve_dataset_config(
        active_tenant_id=active_tenant_id, active_bucket=active_bucket
    )
    if config is None:
        return []
    outbox = _secure_outbox()
    if not outbox.exists():
        return []
    s3 = _s3(storage)
    flushed: list[str] = []
    for path in sorted(outbox.glob("*.json")):
        try:
            if not stat.S_ISREG(path.lstat().st_mode):
                continue
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope.get("schema_version") != "npa.agent.trajectory-outbox.v1":
                raise AgentRunDataError("legacy outbox requires explicit destination reconciliation")
            if envelope.get("destination_sha256") != _destination_digest(config):
                raise AgentRunDataError("pending trajectory belongs to a different destination")
            payload = envelope["payload"]
            body = _validated_body(config, payload)
            expected_name = f"{payload['episode_id']}-{payload['collection']['content_sha256']}.json"
            if path.name != expected_name or envelope.get("payload_sha256") != hashlib.sha256(body).hexdigest():
                raise AgentRunDataError("pending trajectory envelope integrity mismatch")
            verify_destination(config, storage=s3)
            _put_and_verify(config, payload, s3)
            path.unlink()
            flushed.append(str(payload["episode_id"]))
        except Exception:
            LOGGER.debug("pending trajectory retained after scope, integrity or delivery failure")
            continue
    return flushed


def _request_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") if isinstance(payload, dict) else []
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def _episode_record(
    *,
    payload: dict[str, Any],
    response: dict[str, Any] | None,
    error: BaseException | None,
    episode_id: str,
    session_id: str,
    started_at: str,
) -> dict[str, Any]:
    ended_at = datetime.now(timezone.utc).isoformat()
    steps = response.get("steps", []) if isinstance(response, dict) else []
    events = [
        {
            "sequence": 0,
            "phase": "plan",
            "tool": "",
            "arguments": {},
            "observation": {"accepted": True},
            "status": "ok",
        }
    ]
    if isinstance(steps, list):
        for step in steps:
            events.append(
                {
                    "sequence": len(events),
                    "phase": "tool",
                    "tool": str(step.get("tool") or "") if isinstance(step, dict) else "",
                    "arguments": step.get("arguments", {}) if isinstance(step, dict) else {},
                    "observation": step.get("observation", {}) if isinstance(step, dict) else {},
                    "status": "ok" if isinstance(step, dict) and step.get("ok", True) else "error",
                }
            )
    apis_used = response.get("apis_used", []) if isinstance(response, dict) else []
    if isinstance(apis_used, list):
        for api in apis_used:
            events.append(
                {
                    "sequence": len(events),
                    "phase": "tool",
                    "tool": str(api),
                    "arguments": {},
                    "observation": {"reported_by_endpoint": True},
                    "status": "ok",
                }
            )
    stopped = str(response.get("stopped_reason") or "") if isinstance(response, dict) else ""
    refused = bool(response and (response.get("refused") or response.get("needs_confirmation")))
    cancelled = isinstance(error, asyncio.CancelledError) or stopped in {"cancelled", "canceled"}
    failed = error is not None or (isinstance(response, dict) and response.get("ok") is False)
    outcome_status = "cancelled" if cancelled else "refused" if refused else "failed" if failed else "succeeded"
    if refused and isinstance(response, dict):
        events.append(
            {
                "sequence": len(events),
                "phase": "confirm",
                "tool": "",
                "arguments": response.get("proposed_action") or {},
                "observation": {"needs_confirmation": True},
                "status": "rejected",
            }
        )
    events.append(
        {
            "sequence": len(events),
            "phase": "final",
            "tool": "",
            "arguments": {},
            "observation": {
                "response_ok": response.get("ok") if isinstance(response, dict) else False,
                "error_type": type(error).__name__ if error else "",
            },
            "status": "cancelled" if cancelled else "rejected" if refused else "error" if failed else "ok",
        }
    )
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    grounded = bool(response.get("grounded")) if isinstance(response, dict) else False
    routing: dict[str, Any] = {
        "grounded": grounded,
        "tier": str(response.get("tier") or "") if isinstance(response, dict) else "",
        "model": str(response.get("model") or "") if isinstance(response, dict) else "",
    }
    if grounded:
        routing.update({"input_tokens": 0, "output_tokens": 0})
    elif isinstance(usage, dict):
        for source, target in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
            if source in usage:
                try:
                    routing[target] = int(usage[source])
                except (TypeError, ValueError, OverflowError):
                    # Unknown usage must not become fabricated zero or a failure
                    # of the product operation that already completed.
                    continue
    # An endpoint returning normally proves response delivery, not the user's
    # goal. Only the exception observed here is objective outcome evidence;
    # explicit emit_trajectory callers can supply actual workload/test proof.
    verified = error is not None
    return {
        "episode_id": episode_id,
        "session_id": session_id,
        "request_content": _request_text(payload),
        "intent": str(response.get("semantic_intent") or response.get("mode") or "chat") if isinstance(response, dict) else "chat",
        "trajectory": events,
        "outcome": {
            "status": outcome_status,
            "verified": verified,
            "verified_by": ["agent endpoint exception"] if verified else [],
            "artifact_uris": (
                response.get("artifact_uris", [])
                if isinstance(response, dict)
                and isinstance(response.get("artifact_uris"), list)
                else []
            ),
            "operator_interventions": [],
            "preference_pairs": [],
        },
        "routing": routing,
        "versions": {"agent": os.environ.get("NPA_AGENT_UI_VERSION", ""), "tools": {}},
        "initial_state": {},
        "started_at": started_at,
        "ended_at": ended_at,
    }


def goal_episode_boundary(
    *,
    active_tenant_id: Callable[[], str] = lambda: "",
    active_bucket: Callable[[], str] = lambda: "",
    storage_factory: Callable[[], Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a synchronous goal endpoint with one terminal trajectory."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapped(payload: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            safe_payload = payload if isinstance(payload, dict) else {}
            episode_id = f"episode-{uuid.uuid4().hex}"
            session_id = str(safe_payload.get("session_id") or "default")
            started_at = datetime.now(timezone.utc).isoformat()
            try:
                storage = storage_factory() if storage_factory else None
            except Exception:
                storage = _UnavailableStorage()
                LOGGER.debug("trajectory storage factory unavailable; preserving product execution")
            try:
                flush_outbox(
                    storage=storage,
                    active_tenant_id=active_tenant_id(),
                    active_bucket=active_bucket(),
                )
            except Exception:
                LOGGER.debug("pending trajectory flush failed at episode start")
            response: dict[str, Any] | None = None
            error: BaseException | None = None
            try:
                result = function(payload, *args, **kwargs)
                response = result if isinstance(result, dict) else {"ok": True}
                return result
            except BaseException as exc:
                error = exc
                raise
            finally:
                try:
                    record = _episode_record(
                        payload=safe_payload,
                        response=response,
                        error=error,
                        episode_id=episode_id,
                        session_id=session_id,
                        started_at=started_at,
                    )
                    emit_trajectory(
                        **record,
                        storage=storage,
                        active_tenant_id=active_tenant_id(),
                        active_bucket=active_bucket(),
                    )
                except Exception:
                    LOGGER.debug("terminal trajectory emission failed")

        return wrapped

    return decorate


__all__ = [
    "AgentRunDataError",
    "CollectionStatus",
    "DatasetConfig",
    "emit_trajectory",
    "flush_outbox",
    "goal_episode_boundary",
    "redact",
    "resolve_dataset_config",
    "verify_destination",
]
