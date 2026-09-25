"""Thin S3 garbage collection for workbench run artifacts.

Scans run prefixes under an S3 root, classifies each run from its
``npa-workflow/manifest.json``, and deletes only runs that are provably
terminal *and* older than the retention window. Everything ambiguous is kept:

- missing or unreadable manifests (fail closed: treated as live),
- non-terminal or unknown run statuses,
- unknown artifact ages,
- runs carrying the pin marker (explicitly retained).

The S3-touching helpers take a plain boto3-style client so unit tests can
substitute a fake; nothing here imports typer or touches the CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from botocore.exceptions import ClientError

#: Default per-run retention: artifacts of terminal runs older than this are
#: eligible for deletion. Mirrors the policy table in
#: ``docs/workbench/artifact-retention.md``.
DEFAULT_RETENTION_DAYS = 90

#: A run prefix containing this marker file is never deleted, regardless of
#: age or status. This is the explicit pinning mechanism from issue #525.
PIN_MARKER = ".npa-retain"

#: Manifest location relative to a run prefix; its presence is what makes an
#: S3 prefix a *run* the collector understands.
MANIFEST_SUFFIX = "npa-workflow/manifest.json"

#: S3 DeleteObjects accepts at most 1000 keys per call.
DELETE_BATCH_SIZE = 1000

#: Manifest statuses that prove a run will never write again. Anything else
#: (including a missing or unreadable manifest) is treated as live.
TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED", "FAILED_STARTUP"})

#: S3 error codes that mean "this key is simply not there" as opposed to a
#: real failure. Absence feeds the fail-closed path (keep the run).
_MISSING_CODES = {"404", "NoSuchKey", "NotFound", "NoSuchBucket"}


class GcError(Exception):
    """The collector hit a real failure (not a fail-closed keep)."""


class S3Client(Protocol):
    """Minimal boto3 surface the collector needs (faked in unit tests)."""

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]: ...
    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]: ...
    def get_paginator(self, name: str) -> Any: ...
    def delete_objects(self, Bucket: str, Delete: dict[str, Any]) -> dict[str, Any]: ...


def is_terminal_status(status: object) -> bool:
    """Whether a manifest status proves the run will never write again."""
    normalized = str(status or "").strip().upper()
    return bool(normalized) and (
        normalized in TERMINAL_STATUSES or normalized.startswith("FAILED")
    )


def parse_manifest_status(payload: bytes) -> str | None:
    """Extract the run status from a manifest body; None if unusable."""
    try:
        data = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    status = str(data.get("status") or "").strip()
    return status or None


def parse_manifest_time(value: object) -> datetime | None:
    """Parse a manifest timestamp; None when absent or unparseable."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        # fromisoformat on 3.10 does not accept the trailing "Z".
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass
class RetentionPolicy:
    """What the collector is allowed to delete."""

    retention_days: int = DEFAULT_RETENTION_DAYS
    pin_marker: str = PIN_MARKER


@dataclass
class RunInfo:
    """Everything the collector knows about one candidate run prefix."""

    prefix: str
    manifest_status: str | None  # None: missing or unreadable manifest
    updated_at: datetime | None
    newest_object_mtime: datetime | None
    size_bytes: int = 0
    object_count: int = 0
    pinned: bool = False


@dataclass
class GcDecision:
    """The collector's verdict for one run."""

    run: RunInfo
    delete: bool
    reason: str


def run_age_days(run: RunInfo, now: datetime) -> float | None:
    """Age of a run in days, anchored on the manifest then newest object."""
    anchor = run.updated_at or run.newest_object_mtime
    if anchor is None:
        return None
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    return (now - anchor).total_seconds() / 86400.0


def decide_run(run: RunInfo, policy: RetentionPolicy, now: datetime) -> GcDecision:
    """Classify one run. Ambiguity always resolves to *keep* (fail closed)."""
    if run.manifest_status is None:
        return GcDecision(run, False, "no-manifest")
    if not is_terminal_status(run.manifest_status):
        return GcDecision(run, False, "live")
    if run.pinned:
        return GcDecision(run, False, "pinned")
    age_days = run_age_days(run, now)
    if age_days is None:
        return GcDecision(run, False, "unknown-age")
    if age_days < policy.retention_days:
        return GcDecision(run, False, "young")
    return GcDecision(run, True, "expired")


def plan_gc(
    runs: list[RunInfo],
    policy: RetentionPolicy,
    now: datetime | None = None,
) -> list[GcDecision]:
    """Build the deletion plan; performs no S3 writes."""
    moment = now or datetime.now(timezone.utc)
    return [decide_run(run, policy, moment) for run in runs]


def plan_summary(decisions: list[GcDecision]) -> dict[str, Any]:
    """Aggregate counts for display and journaling."""
    doomed = [d for d in decisions if d.delete]
    kept: dict[str, int] = {}
    for decision in decisions:
        if not decision.delete:
            kept[decision.reason] = kept.get(decision.reason, 0) + 1
    return {
        "runs_scanned": len(decisions),
        "runs_to_delete": len(doomed),
        "bytes_to_delete": sum(d.run.size_bytes for d in doomed),
        "objects_to_delete": sum(d.run.object_count for d in doomed),
        "kept_by_reason": kept,
    }


def _is_missing(exc: ClientError) -> bool:
    code = str(exc.response.get("Error", {}).get("Code", ""))
    return code in _MISSING_CODES


def _key_exists(client: S3Client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if _is_missing(exc):
            return False
        raise
    return True


def _common_prefixes(client: S3Client, bucket: str, prefix: str) -> list[str]:
    kwargs: dict[str, Any] = {"Bucket": bucket, "Delimiter": "/"}
    if prefix:
        kwargs["Prefix"] = prefix.rstrip("/") + "/"
    found: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(**kwargs):
        for entry in page.get("CommonPrefixes", []) or []:
            child = str(entry.get("Prefix", "")).rstrip("/")
            if child:
                found.append(child)
    return found


def discover_run_prefixes(
    client: S3Client,
    bucket: str,
    root_prefix: str = "",
    max_depth: int = 3,
) -> list[str]:
    """Find run prefixes: S3 prefixes containing ``npa-workflow/manifest.json``.

    Breadth-first over common prefixes so nested layouts are found without
    listing every object. Prefixes with no manifest are never returned (the
    collector cannot prove their lifecycle, so it will not touch them).
    """
    root = root_prefix.strip("/")
    runs: list[str] = []
    if root and _key_exists(client, bucket, f"{root}/{MANIFEST_SUFFIX}"):
        return [root]
    frontier = [root]
    for _ in range(max_depth):
        next_frontier: list[str] = []
        for prefix in frontier:
            for child in _common_prefixes(client, bucket, prefix):
                if _key_exists(client, bucket, f"{child}/{MANIFEST_SUFFIX}"):
                    runs.append(child)
                else:
                    next_frontier.append(child)
        frontier = next_frontier
        if not frontier:
            break
    return sorted(runs)


def describe_run(
    client: S3Client,
    bucket: str,
    run_prefix: str,
    pin_marker: str = PIN_MARKER,
) -> RunInfo:
    """Read one run's manifest, pin marker, and object inventory."""
    manifest_key = f"{run_prefix}/{MANIFEST_SUFFIX}"
    status: str | None = None
    updated_at: datetime | None = None
    try:
        body = client.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
        status = parse_manifest_status(body)
        try:
            data = json.loads(body)
            updated_at = (
                parse_manifest_time(data.get("updated_at"))
                if isinstance(data, dict)
                else None
            )
        except (ValueError, UnicodeDecodeError):
            updated_at = None
    except ClientError as exc:
        if not _is_missing(exc):
            raise
        # Missing/unreadable manifest: status stays None -> fail closed downstream.

    pinned = _key_exists(client, bucket, f"{run_prefix}/{pin_marker}")

    size_bytes = 0
    object_count = 0
    newest: datetime | None = None
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=run_prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []) or []:
            object_count += 1
            size_bytes += int(item.get("Size") or 0)
            modified = item.get("LastModified")
            if isinstance(modified, datetime) and (newest is None or modified > newest):
                newest = modified

    return RunInfo(
        prefix=run_prefix,
        manifest_status=status,
        updated_at=updated_at,
        newest_object_mtime=newest,
        size_bytes=size_bytes,
        object_count=object_count,
        pinned=pinned,
    )


def delete_run_prefix(client: S3Client, bucket: str, run_prefix: str) -> int:
    """Delete every object under a run prefix. Returns the deleted count.

    Refuses to run unless every key is provably under the prefix; S3 delete
    errors raise :class:`GcError` instead of silently continuing.
    """
    root = run_prefix.rstrip("/") + "/"
    deleted = 0
    batch: list[str] = []

    def flush() -> None:
        nonlocal deleted
        for key in batch:
            if not key.startswith(root):
                raise GcError(f"refusing to delete key outside run prefix: {key!r}")
        response = client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
        )
        deleted += len(response.get("Deleted", []) or [])
        errors = response.get("Errors", []) or []
        if errors:
            raise GcError(f"delete failed for {len(errors)} object(s): {errors!r}")

    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=root):
        for item in page.get("Contents", []) or []:
            key = str(item.get("Key", ""))
            if key:
                batch.append(key)
            if len(batch) >= DELETE_BATCH_SIZE:
                flush()
                batch = []
    if batch:
        flush()
    return deleted


@dataclass
class ApplyResult:
    """Outcome of executing a plan against S3."""

    deleted_runs: list[str] = field(default_factory=list)
    skipped_runs: list[tuple[str, str]] = field(default_factory=list)
    deleted_objects: int = 0


def apply_plan(
    client: S3Client,
    bucket: str,
    decisions: list[GcDecision],
    pin_marker: str = PIN_MARKER,
) -> ApplyResult:
    """Execute delete decisions, re-verifying each run's manifest first.

    The re-check closes the plan/apply race: a run that left the terminal
    state after planning is skipped instead of deleted.
    """
    result = ApplyResult()
    for decision in decisions:
        if not decision.delete:
            continue
        prefix = decision.run.prefix
        try:
            body = client.get_object(Bucket=bucket, Key=f"{prefix}/{MANIFEST_SUFFIX}")[
                "Body"
            ].read()
            status = parse_manifest_status(body)
        except ClientError as exc:
            if _is_missing(exc):
                status = None
            else:
                raise
        if not is_terminal_status(status):
            result.skipped_runs.append((prefix, "became-live"))
            continue
        result.deleted_objects += delete_run_prefix(client, bucket, prefix)
        result.deleted_runs.append(prefix)
    return result
