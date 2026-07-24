"""Artifact-first S3 discovery helpers for agent browsing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

try:
    from npa.workflows.rerun_serve import validate_run_id
except Exception:  # pragma: no cover - embedded backend fallback
    import re

    _PLACEHOLDER_RUN_ID_RE = re.compile(
        r"yyyymmdd|hhmmss|your-run-id|<run-id>|placeholder|example-run|tbd|xxxx",
        re.IGNORECASE,
    )
    _SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._:-]+$")

    def validate_run_id(run_id: str) -> str:
        value = run_id.strip()
        if not value:
            raise ArtifactDiscoveryError("run-id is required")
        if _PLACEHOLDER_RUN_ID_RE.search(value):
            raise ArtifactDiscoveryError("run-id looks like a placeholder")
        if value.startswith("/") or value.endswith("/"):
            raise ArtifactDiscoveryError("run-id must not start or end with '/'")
        segments = value.split("/")
        for segment in segments:
            if segment in {"", ".", ".."}:
                raise ArtifactDiscoveryError("run-id traversal segments are not allowed")
            if not _SAFE_SEGMENT_RE.fullmatch(segment):
                raise ArtifactDiscoveryError("run-id contains unsupported characters")
        return value

_RERUN_EXTENSIONS = {".rrd"}
_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov"}
# Browser-native image formats an <img> tag can render directly.
_WEB_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
# Image formats a browser CANNOT render natively (e.g. sim-rollout camera frames
# saved as Netpbm .ppm). They are still images — classified as "image" so they
# appear as viewable — and are transcoded to PNG on the way out (see
# needs_image_transcode / the agent's /api/artifacts/file endpoint).
_NON_WEB_IMAGE_EXTENSIONS = {".ppm", ".pgm", ".pbm", ".pnm", ".bmp", ".tif", ".tiff"}
_IMAGE_EXTENSIONS = _WEB_IMAGE_EXTENSIONS | _NON_WEB_IMAGE_EXTENSIONS


def needs_image_transcode(name: str) -> bool:
    """True when ``name`` is an image a browser cannot render natively (→ PNG)."""
    return Path(str(name or "")).suffix.lower() in _NON_WEB_IMAGE_EXTENSIONS
_JSON_EXTENSIONS = {".json"}
_TEXT_EXTENSIONS = {".txt", ".log", ".csv", ".yaml", ".yml", ".md"}
_RENDER_ORDER = {
    "rerun": 0,
    "video": 1,
    "image": 2,
    "json": 3,
    "text": 4,
    "download": 5,
}


class ArtifactDiscoveryError(RuntimeError):
    """Raised when artifact discovery or retrieval fails."""


@dataclass(frozen=True)
class Artifact:
    run_id: str
    key: str
    s3_uri: str
    size: int
    last_modified: str
    render: str
    inline: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "key": self.key,
            "s3_uri": self.s3_uri,
            "size": self.size,
            "last_modified": self.last_modified,
            "render": self.render,
            "inline": self.inline,
        }


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    last_modified: str
    artifact_count: int
    has_viewable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "last_modified": self.last_modified,
            "artifact_count": self.artifact_count,
            "has_viewable": self.has_viewable,
        }


@dataclass(frozen=True)
class RunListPage:
    runs: list[RunSummary]
    truncated: bool
    total_runs: int
    limit: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": [item.to_dict() for item in self.runs],
            "truncated": self.truncated,
            "total_runs": self.total_runs,
            "limit": self.limit,
        }


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ArtifactDiscoveryError(f"expected s3:// URI, got {uri!r}")
    key = parsed.path.lstrip("/")
    if not key:
        raise ArtifactDiscoveryError(f"S3 URI missing object key: {uri!r}")
    return parsed.netloc, key


def build_s3_client(
    *,
    endpoint_url: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    region_name: str = "eu-north1",
):
    import boto3

    kwargs: dict[str, Any] = {
        "aws_access_key_id": aws_access_key_id or None,
        "aws_secret_access_key": aws_secret_access_key or None,
        "region_name": region_name,
        "config": BotoConfig(signature_version="s3v4"),
    }
    if endpoint_url.strip():
        kwargs["endpoint_url"] = endpoint_url.strip()
    return boto3.client("s3", **kwargs)


def render_hint_for_object(*, key: str, content_type: str = "") -> str:
    ext = Path(key).suffix.lower()
    if ext in _RERUN_EXTENSIONS:
        return "rerun"
    if ext in _VIDEO_EXTENSIONS:
        return "video"
    if ext in _IMAGE_EXTENSIONS:
        return "image"
    if ext in _JSON_EXTENSIONS:
        return "json"
    if ext in _TEXT_EXTENSIONS:
        return "text"
    lowered_type = content_type.lower().strip()
    if lowered_type.startswith("video/"):
        return "video"
    if lowered_type.startswith("image/"):
        return "image"
    if lowered_type in {"application/json", "application/ld+json"}:
        return "json"
    if lowered_type.startswith("text/"):
        return "text"
    guessed_type, _ = mimetypes.guess_type(key)
    guessed = str(guessed_type or "").lower()
    if guessed.startswith("video/"):
        return "video"
    if guessed.startswith("image/"):
        return "image"
    if guessed == "application/json":
        return "json"
    if guessed.startswith("text/"):
        return "text"
    return "download"


def is_inline_render(render: str) -> bool:
    return render in {"rerun", "video", "image", "json", "text"}


def artifact_media_type(filename: str) -> str:
    """Return a browser-playable Content-Type for an artifact filename.

    Used by the agent ``/api/artifacts/file/...`` endpoint so ``<video>`` /
    ``<img>`` previews (and authenticated blob fetches) receive a real media
    type instead of ``application/octet-stream``.
    """
    name = str(filename or "").strip()
    suffix = Path(name).suffix.lower()
    explicit = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".json": "application/json",
        ".txt": "text/plain; charset=utf-8",
        ".log": "text/plain; charset=utf-8",
        ".md": "text/plain; charset=utf-8",
        ".csv": "text/plain; charset=utf-8",
        ".yaml": "text/plain; charset=utf-8",
        ".yml": "text/plain; charset=utf-8",
    }
    if suffix in explicit:
        return explicit[suffix]
    guessed, _ = mimetypes.guess_type(name)
    if guessed:
        return str(guessed)
    return "application/octet-stream"


def list_runs(
    bucket: str,
    *,
    prefix: str = "",
    limit: int = 50,
    s3=None,
) -> RunListPage:
    if limit <= 0:
        raise ArtifactDiscoveryError("limit must be > 0")
    client = s3
    if client is None:
        raise ArtifactDiscoveryError("s3 client is required")
    normalized_prefix = _normalize_prefix(prefix)
    summary: dict[str, dict[str, Any]] = {}
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
            for item in page.get("Contents", []) or []:
                key = str(item.get("Key") or "")
                run_id = _run_id_for_key(key, normalized_prefix)
                if not run_id:
                    continue
                # A run is a directory (``<run_id>/<stage>/...``). Skip bare files
                # sitting directly under the prefix (e.g. ``<cat>/records.json``),
                # which are not runs — this keeps generic root-level discovery clean.
                remainder = key[len(normalized_prefix):] if normalized_prefix else key
                if "/" not in remainder.lstrip("/"):
                    continue
                render = render_hint_for_object(key=key)
                current = summary.setdefault(
                    run_id,
                    {"artifact_count": 0, "last_modified": "", "has_viewable": False},
                )
                current["artifact_count"] = int(current["artifact_count"]) + 1
                current["has_viewable"] = bool(current["has_viewable"] or render != "download")
                ts = _to_iso8601(item.get("LastModified"))
                if ts and ts > str(current["last_modified"]):
                    current["last_modified"] = ts
    except (ClientError, BotoCoreError) as exc:
        raise ArtifactDiscoveryError(f"failed to list runs from s3://{bucket}/{normalized_prefix}: {exc}") from exc

    runs = [
        RunSummary(
            run_id=run_id,
            last_modified=str(payload["last_modified"]),
            artifact_count=int(payload["artifact_count"]),
            has_viewable=bool(payload["has_viewable"]),
        )
        for run_id, payload in summary.items()
    ]
    runs.sort(key=lambda item: (item.last_modified, item.run_id), reverse=True)
    total = len(runs)
    truncated = total > limit
    if truncated:
        runs = runs[:limit]
    return RunListPage(runs=runs, truncated=truncated, total_runs=total, limit=limit)


def list_run_categories(bucket: str, *, base_prefix: str = "", s3=None) -> list[str]:
    """Return the immediate sub-directory prefixes under ``base_prefix``.

    Runs are stored as ``<root>/<category>/<run_id>/...`` (e.g.
    ``checkpoints/sim2real-b/...``, ``checkpoints/physical-ai-data-factory/...``).
    This enumerates the ``<category>`` folders dynamically from S3 so discovery
    never hardcodes specific workflow paths. Returns category prefixes WITHOUT a
    trailing slash. If the root has no sub-folders, returns ``[]``.
    """
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    root = _normalize_prefix(base_prefix)
    categories: list[str] = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=root, Delimiter="/"):
            for common in page.get("CommonPrefixes", []) or []:
                pfx = str(common.get("Prefix") or "").rstrip("/")
                if pfx:
                    categories.append(pfx)
    except (ClientError, BotoCoreError) as exc:
        raise ArtifactDiscoveryError(
            f"failed to list run categories under s3://{bucket}/{root}: {exc}"
        ) from exc
    return categories


def discovery_categories(
    bucket: str, *, base_prefix: str = "", exclude: "set[str] | None" = None, s3=None
) -> list[str]:
    """Return every candidate *run-parent* prefix in the bucket, generically.

    Runs live one level below a category prefix (``<category>/<run_id>/...``).
    Categories themselves may sit either under a configured base root
    (``<base>/<category>/<run_id>/...``, e.g. ``checkpoints/sim2real-b/...``) or
    directly at the bucket root (``<category>/<run_id>/...``, e.g.
    ``scenario-gen-smoke/...``, ``physical-ai-data-factory/...``). Different
    workflows write to different roots, so discovery must span both.

    This merges, in order (newest-workflow-agnostic, no hardcoded paths):

    1. categories under the configured ``base_prefix`` (``<base>/<category>``);
    2. categories at the bucket root (``<category>``), excluding ``base_prefix``
       itself — its children are categories, not runs, and are covered by (1).

    ``exclude`` drops categories whose first path segment matches (e.g. the
    agent's own state/memory root), so infra prefixes never masquerade as runs.

    Prefixes are returned without a trailing slash, de-duplicated, base-first.
    """
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    base = str(base_prefix or "").strip().strip("/")
    excluded = {str(x).strip().strip("/") for x in (exclude or set()) if str(x).strip().strip("/")}
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(prefix: str) -> None:
        value = str(prefix or "").strip().strip("/")
        if not value or value in seen:
            return
        if value in excluded or value.split("/", 1)[0] in excluded:
            return
        seen.add(value)
        ordered.append(value)

    if base:
        for category in list_run_categories(bucket, base_prefix=base, s3=s3):
            _add(category)
    for category in list_run_categories(bucket, base_prefix="", s3=s3):
        # The base root's children are categories (handled above), not runs.
        if base and category.strip("/") == base:
            continue
        _add(category)
    return ordered


def list_all_runs(
    bucket: str,
    *,
    base_prefix: str = "",
    limit: int = 50,
    exclude: "set[str] | None" = None,
    s3=None,
) -> RunListPage:
    """Discover runs across every category in the bucket generically.

    Enumerates category folders under the configured base root AND at the bucket
    root (see :func:`discovery_categories`) and merges each category's runs
    (dedup by run_id, keep newest), latest-first. No workflow path is hardcoded;
    a new workflow folder — under any root — shows up automatically. ``exclude``
    drops infra roots (e.g. the agent's own state prefix) from the listing.
    """
    if limit <= 0:
        raise ArtifactDiscoveryError("limit must be > 0")
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    categories = discovery_categories(bucket, base_prefix=base_prefix, exclude=exclude, s3=s3)
    if not categories:
        # Flat layout: run_ids sit directly under the root.
        return list_runs(bucket, prefix=base_prefix, limit=limit, s3=s3)
    best: dict[str, RunSummary] = {}
    total = 0
    for category in categories:
        try:
            page = list_runs(bucket, prefix=category, limit=limit, s3=s3)
        except ArtifactDiscoveryError:
            continue
        for run in page.runs:
            current = best.get(run.run_id)
            if current is None or run.last_modified > current.last_modified:
                best[run.run_id] = run
    total = len(best)
    runs = sorted(best.values(), key=lambda item: (item.last_modified, item.run_id), reverse=True)
    truncated = len(runs) > limit
    if len(runs) > limit:
        runs = runs[:limit]
    return RunListPage(runs=runs, truncated=truncated, total_runs=total, limit=limit)


def find_run_artifacts(bucket: str, *, base_prefix: str, run_id: str, s3=None) -> "list[Artifact]":
    """Locate a run's artifacts anywhere in the bucket without a hardcoded path.

    Probes each candidate parent prefix (categories under the base root and at
    the bucket root — see :func:`discovery_categories`) as ``<parent>/<run_id>/``
    and returns the first non-empty match, then falls back to the flat layouts
    (``<base>/<run_id>/`` and ``<run_id>/``). Run ids are unique, so the first
    hit is authoritative — a run stored under any workflow root resolves.
    """
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    categories = discovery_categories(bucket, base_prefix=base_prefix, s3=s3)
    for category in categories:
        artifacts = list_artifacts(bucket, run_id, prefix=category, s3=s3)
        if artifacts:
            return artifacts
    # Flat fallbacks: run directly under the base root, or at the bucket root.
    for flat_prefix in (base_prefix, ""):
        artifacts = list_artifacts(bucket, run_id, prefix=flat_prefix, s3=s3)
        if artifacts:
            return artifacts
    return []


def list_artifacts(
    bucket: str,
    run_id: str,
    *,
    prefix: str = "",
    s3=None,
) -> list[Artifact]:
    client = s3
    if client is None:
        raise ArtifactDiscoveryError("s3 client is required")
    normalized_prefix = _normalize_prefix(prefix)
    run_prefix = _normalize_prefix(validate_run_id(run_id))
    scope = f"{normalized_prefix}{run_prefix}"
    artifacts: list[Artifact] = []
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=scope):
            for item in page.get("Contents", []) or []:
                key = str(item.get("Key") or "")
                if not key:
                    continue
                render = render_hint_for_object(key=key)
                artifacts.append(
                    Artifact(
                        run_id=run_id,
                        key=key,
                        s3_uri=f"s3://{bucket}/{key}",
                        size=int(item.get("Size") or 0),
                        last_modified=_to_iso8601(item.get("LastModified")),
                        render=render,
                        inline=is_inline_render(render),
                    )
                )
    except (ClientError, BotoCoreError) as exc:
        raise ArtifactDiscoveryError(f"failed to list artifacts under s3://{bucket}/{scope}: {exc}") from exc
    artifacts.sort(key=lambda item: (item.last_modified, item.key), reverse=True)
    return artifacts


def select_preferred_artifact(artifacts: list[Artifact]) -> Artifact | None:
    if not artifacts:
        return None
    def _score(item: Artifact) -> tuple[int, int, str, str]:
        key = item.key.lower()
        if key.endswith("/reports/sim2real.rrd"):
            specificity = 0
        elif key.endswith(".rrd"):
            specificity = 1
        elif key.endswith("/reports/sim2real-report.json"):
            specificity = 2
        elif "/reports/" in key:
            specificity = 3
        elif "/component-io/" in key:
            specificity = 20
        else:
            specificity = 10
        return (_RENDER_ORDER.get(item.render, 99), specificity, item.last_modified, item.key)
    return sorted(
        artifacts,
        key=_score,
    )[0]


def download_s3_uri(s3_uri: str, destination: Path, *, s3) -> Path:
    bucket, key = parse_s3_uri(s3_uri)
    return download_object(bucket=bucket, key=key, destination=destination, s3=s3)


def download_object(*, bucket: str, key: str, destination: Path, s3) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(bucket, key, str(destination))
    except (ClientError, BotoCoreError) as exc:
        raise ArtifactDiscoveryError(f"failed to download s3://{bucket}/{key}: {exc}") from exc
    return destination


def _normalize_prefix(prefix: str) -> str:
    value = str(prefix or "").strip().strip("/")
    if not value:
        return ""
    return value + "/"


def _run_id_for_key(key: str, normalized_prefix: str) -> str:
    if normalized_prefix:
        if not key.startswith(normalized_prefix):
            return ""
        remainder = key[len(normalized_prefix) :]
    else:
        remainder = key
    remainder = remainder.lstrip("/")
    if not remainder:
        return ""
    first_segment = remainder.split("/", 1)[0].strip()
    return first_segment


def _to_iso8601(value: Any) -> str:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        return value
    else:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()
