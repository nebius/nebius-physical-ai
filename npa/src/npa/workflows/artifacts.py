"""Artifact-first S3 discovery helpers for agent browsing."""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from datetime import datetime, timezone
import mimetypes
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable
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
# Recording formats the embedded MCAP viewers open directly. MCAP is the
# canonical one; Lichtblick/Foxglove also read ROS 1 bags, ROS 2 db3 and PX4 ulog.
_MCAP_EXTENSIONS = {".mcap", ".bag", ".db3", ".ulg", ".ulog"}
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
# 3D scene/asset artifacts. Deliberately DOWNLOAD-ONLY: no browser renders USDZ
# or PLY, and the agent ships no 3D-asset viewer, so offering them as an inline
# preview would produce a broken pane. Stating the set explicitly (rather than
# letting them fall through the mimetypes guesses below) pins that decision and
# keeps a future `mimetypes` addition -- e.g. a `model/vnd.usdz+zip` entry -- from
# silently reclassifying a NuRec reconstruction. A run stays viewable through its
# `.rrd` / `.png` / `.mp4` / `.json` artifacts.
_MODEL_EXTENSIONS = {".usdz", ".usd", ".usda", ".usdc", ".ply", ".obj", ".glb", ".gltf"}


def is_model_artifact(name: str) -> bool:
    """True when ``name`` is a 3D scene/asset artifact offered as a download."""
    return Path(str(name or "")).suffix.lower() in _MODEL_EXTENSIONS


_JSON_EXTENSIONS = {".json"}
_TEXT_EXTENSIONS = {".txt", ".log", ".csv", ".yaml", ".yml", ".md"}
_RENDER_ORDER = {
    "rerun": 0,
    "mcap": 1,
    "video": 2,
    "image": 3,
    "json": 4,
    "text": 5,
    "download": 6,
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
    bucket: str = ""
    # When the run STARTED (its id-encoded submit time, or the earliest artifact
    # write as a fallback) — distinct from last_modified (newest artifact write).
    started_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "last_modified": self.last_modified,
            "started_at": self.started_at,
            "artifact_count": self.artifact_count,
            "has_viewable": self.has_viewable,
            "bucket": self.bucket,
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
    if ext in _MCAP_EXTENSIONS:
        return "mcap"
    if ext in _VIDEO_EXTENSIONS:
        return "video"
    if ext in _IMAGE_EXTENSIONS:
        return "image"
    if ext in _MODEL_EXTENSIONS:
        return "download"
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
    return render in {"rerun", "mcap", "video", "image", "json", "text"}


def artifact_media_type(filename: str) -> str:
    """Return a browser-playable Content-Type for an artifact filename.

    Used by the agent ``/api/artifacts/file/...`` endpoint so ``<video>`` /
    ``<img>`` previews (and authenticated blob fetches) receive a real media
    type instead of ``application/octet-stream``.
    """
    name = str(filename or "").strip()
    suffix = Path(name).suffix.lower()
    explicit = {
        # MCAP has no registered IANA type; Foxglove selects its reader from the
        # URL extension, and octet-stream keeps byte-range streaming intact.
        ".mcap": "application/octet-stream",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        # PIL-only image types are transcoded to PNG by the agent before serving.
        ".ppm": "image/png",
        ".pgm": "image/png",
        ".pnm": "image/png",
        ".bmp": "image/png",
        ".tif": "image/png",
        ".tiff": "image/png",
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
    contains: str = "",
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
                    {"artifact_count": 0, "last_modified": "", "earliest": "", "has_viewable": False},
                )
                current["artifact_count"] = int(current["artifact_count"]) + 1
                current["has_viewable"] = bool(current["has_viewable"] or render != "download")
                ts = _to_iso8601(item.get("LastModified"))
                if ts:
                    if ts > str(current["last_modified"]):
                        current["last_modified"] = ts
                    if not current["earliest"] or ts < str(current["earliest"]):
                        current["earliest"] = ts
    except (ClientError, BotoCoreError) as exc:
        raise ArtifactDiscoveryError(f"failed to list runs from s3://{bucket}/{normalized_prefix}: {exc}") from exc

    runs = [
        RunSummary(
            run_id=run_id,
            last_modified=str(payload["last_modified"]),
            started_at=_run_started_at(run_id, str(payload.get("earliest") or "")),
            artifact_count=int(payload["artifact_count"]),
            has_viewable=bool(payload["has_viewable"]),
        )
        for run_id, payload in summary.items()
    ]
    # Substring search (case-insensitive) applied BEFORE truncation so a matching
    # run is found even when it is far older than the newest `limit` runs.
    needle = str(contains or "").strip().lower()
    if needle:
        runs = [item for item in runs if needle in item.run_id.lower()]
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
    contains: str = "",
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
        return list_runs(bucket, prefix=base_prefix, limit=limit, contains=contains, s3=s3)
    # Each category's run listing is an independent, I/O-bound S3 pagination, so
    # scanning them concurrently turns the whole discovery from O(sum of categories)
    # sequential round-trips into ~O(slowest category) wall time. This is the main
    # latency lever for the default (no-prefix) run list, which previously took
    # several seconds while every category was walked one after another. boto3
    # clients are safe to share across threads for API calls.
    from concurrent.futures import ThreadPoolExecutor

    def _runs_for_category(category: str) -> list[RunSummary]:
        try:
            return list_runs(bucket, prefix=category, limit=limit, contains=contains, s3=s3).runs
        except ArtifactDiscoveryError:
            return []

    best: dict[str, RunSummary] = {}
    max_workers = min(len(categories), 16)
    if max_workers <= 1:
        results = [_runs_for_category(categories[0])] if categories else []
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_runs_for_category, categories))
    for category_runs in results:
        for run in category_runs:
            current = best.get(run.run_id)
            if current is None or run.last_modified > current.last_modified:
                best[run.run_id] = run
    total = len(best)
    runs = sorted(best.values(), key=lambda item: (item.last_modified, item.run_id), reverse=True)
    truncated = len(runs) > limit
    if len(runs) > limit:
        runs = runs[:limit]
    return RunListPage(runs=runs, truncated=truncated, total_runs=total, limit=limit)


# --- Multi-bucket discovery ---------------------------------------------------
# The agent may have access to several buckets (different workflows/projects write
# to different buckets). Discovery must span every bucket the credentials can see
# so no run is invisible just because it landed in another bucket — never rely on
# copying runs into one bucket.


def list_accessible_buckets(
    s3,
    *,
    primary: str = "",
    extra: "list[str] | tuple[str, ...] | None" = None,
    exclude: "set[str] | None" = None,
) -> list[str]:
    """Return the bucket names the agent should search, primary-first.

    Order: the configured primary bucket, any explicitly-configured extras, then
    every bucket ``ListBuckets`` returns (best-effort — on failure just the
    primary/extras). ``exclude`` drops names that must never be scanned.
    """
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    excluded = {str(x).strip() for x in (exclude or set()) if str(x).strip()}
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        value = str(name or "").strip()
        if value and value not in seen and value not in excluded:
            seen.add(value)
            ordered.append(value)

    _add(primary)
    for name in extra or ():
        _add(name)
    try:
        resp = s3.list_buckets()
        for entry in resp.get("Buckets", []) or []:
            _add(str(entry.get("Name") or ""))
    except (ClientError, BotoCoreError):
        pass  # No ListBuckets permission — fall back to primary/extras.
    return ordered


def list_all_runs_across_buckets(
    buckets: "list[str] | tuple[str, ...]",
    *,
    base_prefix: str = "",
    limit: int = 50,
    exclude: "set[str] | None" = None,
    contains: str = "",
    s3=None,
) -> RunListPage:
    """Discover runs across every accessible bucket, latest-first.

    Each bucket is scanned with :func:`list_all_runs` (concurrently), every run
    tagged with its bucket, then merged and de-duped by ``(bucket, run_id)``.
    """
    if limit <= 0:
        raise ArtifactDiscoveryError("limit must be > 0")
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    bucket_list = [str(b).strip() for b in buckets if str(b).strip()]
    if not bucket_list:
        return RunListPage(runs=[], truncated=False, total_runs=0, limit=limit)

    def _runs_for_bucket(bucket: str) -> "tuple[str, list[RunSummary], int]":
        try:
            page = list_all_runs(
                bucket, base_prefix=base_prefix, limit=limit, exclude=exclude, contains=contains, s3=s3
            )
        except (ArtifactDiscoveryError, ClientError, BotoCoreError):
            return bucket, [], 0
        tagged = [dataclass_replace(run, bucket=bucket) for run in page.runs]
        return bucket, tagged, page.total_runs

    from concurrent.futures import ThreadPoolExecutor

    max_workers = min(len(bucket_list), 8)
    if max_workers <= 1:
        results = [_runs_for_bucket(bucket_list[0])]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_runs_for_bucket, bucket_list))

    merged: dict[tuple, RunSummary] = {}
    total = 0
    for _bucket, runs, bucket_total in results:
        total += int(bucket_total or 0)
        for run in runs:
            merged[(run.bucket, run.run_id)] = run
    ordered = sorted(merged.values(), key=lambda item: (item.last_modified, item.run_id), reverse=True)
    truncated = len(ordered) > limit
    if len(ordered) > limit:
        ordered = ordered[:limit]
    return RunListPage(runs=ordered, truncated=truncated, total_runs=total, limit=limit)


def _s3_cache_identity(s3) -> str:
    """Best-effort stable identity for an S3 client so the process-global run
    cache never cross-serves pages across distinct credential/endpoint scopes.

    Uses only the public endpoint plus the access-key id (guarded); mock clients
    in unit tests yield a stable empty identity, so cache behavior is unchanged.
    """
    if s3 is None:
        return ""
    endpoint = ""
    access_key = ""
    try:
        endpoint = str(getattr(getattr(s3, "meta", None), "endpoint_url", "") or "")
    except Exception:  # noqa: BLE001
        endpoint = ""
    try:
        creds = s3._request_signer._credentials  # boto3 internal; best-effort only
        access_key = str(getattr(creds, "access_key", "") or "")
    except Exception:  # noqa: BLE001
        access_key = ""
    return f"{endpoint}|{access_key}"


def list_runs_cached_multi(
    buckets: "list[str] | tuple[str, ...]",
    *,
    base_prefix: str = "",
    limit: int = 50,
    exclude: "set[str] | None" = None,
    contains: str = "",
    s3=None,
    ttl: "float | None" = None,
    refresh_sync: bool = False,
) -> RunListPage:
    """TTL + stale-while-revalidate wrapper over :func:`list_all_runs_across_buckets`."""
    # DEFAULT_RUN_LIST_TTL is defined below this block; resolve at call time.
    if ttl is None:
        ttl = DEFAULT_RUN_LIST_TTL
    bucket_list = tuple(str(b).strip() for b in buckets if str(b).strip())
    key = (
        "__multi__",
        _s3_cache_identity(s3),
        bucket_list,
        base_prefix,
        int(limit),
        tuple(sorted(exclude or ())),
        str(contains or ""),
    )

    def _compute() -> RunListPage:
        return list_all_runs_across_buckets(
            list(bucket_list), base_prefix=base_prefix, limit=limit, exclude=exclude, contains=contains, s3=s3
        )

    now = time.monotonic()
    with _RUN_LIST_LOCK:
        entry = _RUN_LIST_CACHE.get(key)
    if entry is not None:
        ts, page = entry
        if now - ts < ttl:
            return page
        _schedule_run_list_refresh(key, _compute, sync=refresh_sync)
        if refresh_sync:
            with _RUN_LIST_LOCK:
                entry = _RUN_LIST_CACHE.get(key)
            return entry[1] if entry else page
        return page
    page = _compute()
    with _RUN_LIST_LOCK:
        _RUN_LIST_CACHE[key] = (time.monotonic(), page)
    return page


def find_run_artifacts_across_buckets(
    buckets: "list[str] | tuple[str, ...]",
    *,
    base_prefix: str = "",
    run_id: str,
    s3=None,
) -> "tuple[str, list[Artifact]]":
    """Locate a run in the first accessible bucket that contains it.

    Returns ``(bucket, artifacts)``; ``("", [])`` when no bucket has the run.
    Artifacts carry bucket-qualified ``s3_uri`` so downstream reads/downloads
    resolve the correct bucket without a copy.
    """
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    for bucket in buckets:
        name = str(bucket).strip()
        if not name:
            continue
        try:
            artifacts = find_run_artifacts(name, base_prefix=base_prefix, run_id=run_id, s3=s3)
        except (ArtifactDiscoveryError, ClientError, BotoCoreError):
            continue
        if artifacts:
            return name, artifacts
    return "", []


# --- Run-list cache (TTL + stale-while-revalidate) ----------------------------
# Discovering the full run list walks every object under every category
# (O(objects in bucket)); with many workflows and frame-heavy runs (e.g. .ppm
# rollouts) that costs several seconds even parallelized, because wall time is
# floored by the single slowest category. The default agent run-list view is
# polled on every page load, so an uncached call means the UI shows "no runs"
# for seconds each time. This cache serves a warm result instantly and refreshes
# in the background once stale, so only the very first (cold) load pays the walk;
# new runs surface within one TTL. Results are exact — this only reuses recent
# work, it does not approximate counts or ordering.
DEFAULT_RUN_LIST_TTL = 30.0
_RUN_LIST_CACHE: "dict[tuple, tuple[float, RunListPage]]" = {}
_RUN_LIST_INFLIGHT: "set[tuple]" = set()
_RUN_LIST_LOCK = threading.Lock()


def _run_list_cache_clear() -> None:
    """Drop all cached run-list pages (test/maintenance helper)."""
    with _RUN_LIST_LOCK:
        _RUN_LIST_CACHE.clear()
        _RUN_LIST_INFLIGHT.clear()


def _schedule_run_list_refresh(
    key: tuple, compute: "Callable[[], RunListPage]", *, sync: bool = False
) -> "threading.Thread | None":
    """Refresh a stale cache entry. Runs in a daemon thread unless ``sync``."""

    def _run() -> None:
        try:
            page = compute()
        except Exception:  # noqa: BLE001 - background refresh must never raise
            page = None
        finally:
            with _RUN_LIST_LOCK:
                if page is not None:
                    _RUN_LIST_CACHE[key] = (time.monotonic(), page)
                _RUN_LIST_INFLIGHT.discard(key)

    with _RUN_LIST_LOCK:
        if key in _RUN_LIST_INFLIGHT:
            return None
        _RUN_LIST_INFLIGHT.add(key)
    if sync:
        _run()
        return None
    thread = threading.Thread(target=_run, name="npa-runlist-refresh", daemon=True)
    thread.start()
    return thread


def list_runs_cached(
    bucket: str,
    *,
    prefix: str = "",
    base_prefix: str = "",
    limit: int = 50,
    exclude: "set[str] | None" = None,
    contains: str = "",
    s3=None,
    all_categories: bool = False,
    ttl: float = DEFAULT_RUN_LIST_TTL,
    refresh_sync: bool = False,
) -> RunListPage:
    """TTL + stale-while-revalidate wrapper over :func:`list_runs` /
    :func:`list_all_runs`.

    - Fresh cache hit (age < ``ttl``): returned immediately, no S3 calls.
    - Stale cache hit: the cached page is returned immediately AND a single
      background refresh is scheduled (``refresh_sync`` forces it inline, for
      tests) so the next caller sees fresh data.
    - Cold miss: computed synchronously, cached, returned.

    ``all_categories=True`` discovers across every category (no user prefix);
    otherwise it lists a single ``prefix``.
    """
    key = (
        bucket,
        _s3_cache_identity(s3),
        base_prefix,
        prefix,
        int(limit),
        tuple(sorted(exclude or ())),
        str(contains or ""),
        bool(all_categories),
    )

    def _compute() -> RunListPage:
        if all_categories:
            return list_all_runs(
                bucket, base_prefix=base_prefix, limit=limit, exclude=exclude, contains=contains, s3=s3
            )
        return list_runs(bucket, prefix=prefix, limit=limit, contains=contains, s3=s3)

    now = time.monotonic()
    with _RUN_LIST_LOCK:
        entry = _RUN_LIST_CACHE.get(key)
    if entry is not None:
        ts, page = entry
        if now - ts < ttl:
            return page
        _schedule_run_list_refresh(key, _compute, sync=refresh_sync)
        if refresh_sync:
            with _RUN_LIST_LOCK:
                entry = _RUN_LIST_CACHE.get(key)
            return entry[1] if entry else page
        return page
    page = _compute()
    with _RUN_LIST_LOCK:
        _RUN_LIST_CACHE[key] = (time.monotonic(), page)
    return page


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


# Run ids commonly embed the run's start time, e.g. ``s2r-real-0725t222636z``
# (MMDD + t + HHMMSS + z, year omitted) or ``...20260725T222636Z`` (full date).
# Match a date (8-digit YYYYMMDD or 4-digit MMDD) + 6-digit HHMMSS, with an
# optional ``t``/``-``/``_`` separator, not glued to surrounding digits.
_RUN_ID_TS_RE = re.compile(r"(?<![0-9])(\d{8}|\d{4})[tT_-]?(\d{6})(?![0-9])")


def _parse_run_id_timestamps(run_id: str, *, year_hint: int | None = None) -> list[str]:
    """Best-effort extract every start time encoded in a run id.

    Returns ISO-8601 UTC strings (possibly several — a run id may contain more
    than one timestamp-like token; :func:`_run_started_at` picks the right one).
    For the year-less ``MMDD`` form the year comes from ``year_hint`` (typically
    the earliest artifact's year) or the current UTC year; the prior year is also
    offered so a run that started late in December with its first artifact in
    January (or a hint one year off) still resolves.
    """
    out: list[str] = []
    for match in _RUN_ID_TS_RE.finditer(str(run_id or "")):
        date_part, time_part = match.group(1), match.group(2)
        try:
            hour, minute, second = int(time_part[0:2]), int(time_part[2:4]), int(time_part[4:6])
            if len(date_part) == 8:
                candidate_years = [int(date_part[0:4])]
                month, day = int(date_part[4:6]), int(date_part[6:8])
            else:
                base_year = int(year_hint or datetime.now(timezone.utc).year)
                candidate_years = [base_year, base_year - 1]
                month, day = int(date_part[0:2]), int(date_part[2:4])
        except ValueError:
            continue
        for year in candidate_years:
            try:
                dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
            except ValueError:
                continue
            out.append(dt.isoformat())
    return out


def _run_started_at(run_id: str, earliest_iso: str) -> str:
    """Resolve when a run started: the id-encoded submit time when trustworthy,
    else the earliest artifact write.

    A run's start precedes its first artifact write, so an id-encoded time is
    accepted only when it is at/just-before the earliest object (within a few
    days). Among several id-encoded candidates, the latest one satisfying that
    constraint is chosen — this ignores red-herring timestamps elsewhere in the
    id and unrelated digit runs that would parse to a bogus far-off date, while
    still yielding an exact start for delayed-upload runs.
    """
    earliest = str(earliest_iso or "")
    year_hint = int(earliest[0:4]) if earliest[0:4].isdigit() else None
    candidates = _parse_run_id_timestamps(run_id, year_hint=year_hint)
    if not candidates:
        return earliest
    if not earliest:
        # No artifact time to corroborate against; use the first-encoded time.
        return candidates[0]
    window_seconds = 3 * 24 * 3600
    best = ""
    for candidate in candidates:
        if candidate > earliest:
            continue
        try:
            gap = (datetime.fromisoformat(earliest) - datetime.fromisoformat(candidate)).total_seconds()
        except ValueError:
            gap = 0.0
        if gap <= window_seconds and candidate > best:
            best = candidate
    return best or earliest


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


def _run_relative_key(key: str, run_id: str) -> str:
    """Return an object key relative to its run root (drops the run prefix)."""
    k = str(key or "")
    marker = "/" + str(run_id) + "/"
    if marker in k:
        return k.split(marker, 1)[1]
    if run_id and k.startswith(str(run_id) + "/"):
        return k[len(str(run_id)) + 1 :]
    return k


def build_fiftyone_dataset(
    keys: list[str],
    *,
    run_id: str,
    read_json: Any,
    bucket: str = "",
) -> dict[str, Any]:
    """Assemble a FiftyOne/Voxel51-style sample dataset for a data-factory run.

    Pure logic (no I/O of its own): ``keys`` are the run's object keys and
    ``read_json(key)`` returns parsed JSON (or ``None``). Groups the augmented
    scenario variants (thumbnail + appearance-variable tags + augmented caption +
    video) and the original input frames, then summarizes the grade + curation so
    the agent's Voxel51 tab can render "the relevant components of this workflow"
    as a sample grid. Mirrors the ``build_run_provenance`` callback pattern so it
    unit-tests without S3.
    """
    by_clip: dict[str, dict[str, Any]] = {}
    input_frames: list[str] = []
    input_videos: list[str] = []
    json_rel: dict[str, str] = {}
    for key in keys:
        rel = _run_relative_key(key, run_id)
        low = rel.lower()
        if low.endswith(".json"):
            json_rel[rel] = key
        if rel.startswith("cosmos_augmented/"):
            parts = rel.split("/")
            if len(parts) >= 3:
                clip = parts[1]
                fname = parts[-1]
                entry = by_clip.setdefault(clip, {"frames": [], "video": "", "meta": ""})
                if low.endswith(".png"):
                    entry["frames"].append(key)
                elif low.endswith(".mp4"):
                    entry["video"] = key
                elif fname == "metadata.json":
                    entry["meta"] = key
        elif rel.startswith("input/") and low.endswith(".png"):
            input_frames.append(key)
        elif rel.startswith("input/") and low.endswith(".mp4"):
            input_videos.append(key)

    # Resolve the grade report by the vlm_eval tool's own RESULT_FILENAME so this
    # stays in sync if the tool renames it, instead of hardcoding a magic string
    # (mirrors data_factory_stages.grade_gate). Fall back to the non-stub name.
    try:
        from npa.workbench.vlm_eval import RESULT_FILENAME as _VLM_RESULT_FILENAME
    except Exception:  # noqa: BLE001
        _VLM_RESULT_FILENAME = "vlm_eval_stub.json"
    grade = {}
    for _grade_name in (_VLM_RESULT_FILENAME, "vlm_eval.json"):
        grade = read_json(json_rel.get(f"grade/{_grade_name}", "")) or {}
        if grade:
            break
    decision = read_json(json_rel.get("grade/decision.json", "")) or {}
    curation = read_json(json_rel.get("curation/report.json", "")) or {}
    aug_caps = read_json(json_rel.get("labeled_augmented/captions.json", "")) or {}
    cap_items = aug_caps.get("captions", []) if isinstance(aug_caps, dict) else []
    # Captions of the SOURCE frames from the annotate-original stage, so input
    # cards carry their VLM label too (not just the augmented variants).
    orig_caps = read_json(json_rel.get("labeled_original/captions.json", "")) or {}
    orig_cap_items = orig_caps.get("captions", []) if isinstance(orig_caps, dict) else []

    def _orig_caption_for(name: str, idx: int) -> str:
        for item in orig_cap_items:
            if isinstance(item, dict) and name and name in str(item.get("image") or ""):
                return str(item.get("caption") or "")
        if 0 <= idx < len(orig_cap_items) and isinstance(orig_cap_items[idx], dict):
            return str(orig_cap_items[idx].get("caption") or "")
        return ""

    def _caption_for(clip: str, idx: int) -> str:
        for item in cap_items:
            if isinstance(item, dict) and clip and clip in str(item.get("image") or ""):
                return str(item.get("caption") or "")
        if 0 <= idx < len(cap_items) and isinstance(cap_items[idx], dict):
            return str(cap_items[idx].get("caption") or "")
        return ""

    # Real FiftyOne Brain curation (when the curate stage ran inside the
    # npa-fiftyone image): per-clip uniqueness + keep/drop decisions.
    fo_curation = curation.get("fiftyone", {}) if isinstance(curation, dict) else {}
    if not isinstance(fo_curation, dict):
        fo_curation = {}
    fo_samples = fo_curation.get("samples", {}) if isinstance(fo_curation.get("samples"), dict) else {}
    # FiftyOne Brain 2D visualization (PCA) points, keyed by clip id.
    viz_points: dict[str, list[Any]] = {}
    fo_viz = fo_curation.get("visualization", [])
    if isinstance(fo_viz, list):
        for entry in fo_viz:
            if isinstance(entry, dict) and entry.get("id") is not None:
                viz_points[str(entry.get("id"))] = entry.get("point") or []

    bkt = str(bucket or "").strip()

    def _uri(key: str) -> str:
        k = str(key or "")
        return f"s3://{bkt}/{k}" if (bkt and k) else ""

    samples: list[dict[str, Any]] = []
    tag_keys: set[str] = set()
    for idx, clip in enumerate(sorted(by_clip)):
        entry = by_clip[clip]
        variables: dict[str, Any] = {}
        if entry["meta"]:
            meta = read_json(entry["meta"]) or {}
            if isinstance(meta, dict) and isinstance(meta.get("variables"), dict):
                variables = meta["variables"]
        tags = {str(k): v for k, v in variables.items() if k != "prompt"}
        for tk in tags:
            tag_keys.add(tk)
        thumbnail = sorted(entry["frames"])[0] if entry["frames"] else ""
        fo_sample = fo_samples.get(clip, {}) if isinstance(fo_samples.get(clip), dict) else {}
        curation_flags = []
        if fo_sample.get("redundant"):
            curation_flags.append("redundant")
        samples.append(
            {
                "group": "augmented",
                "id": clip,
                "label": clip,
                "thumbnail_key": thumbnail,
                "thumbnail_uri": _uri(thumbnail),
                "video_key": entry["video"],
                "video_uri": _uri(entry["video"]),
                "tags": tags,
                "prompt": str(variables.get("prompt") or ""),
                "caption": _caption_for(clip, idx),
                "uniqueness": fo_sample.get("uniqueness"),
                "curated": fo_sample.get("kept") if "kept" in fo_sample else None,
                "curation_flags": curation_flags,
                "point": viz_points.get(clip) or None,
            }
        )
    # Source clip video(s) from the input stage, so the "input data" includes the
    # original footage the pipeline augments (not just the extracted frames).
    for vkey in sorted(input_videos):
        vname = vkey.rsplit("/", 1)[-1]
        poster = sorted(input_frames)[0] if input_frames else ""
        samples.append(
            {
                "group": "input",
                "id": vname,
                "label": vname,
                "thumbnail_key": poster,
                "thumbnail_uri": _uri(poster),
                "video_key": vkey,
                "video_uri": _uri(vkey),
                "tags": {},
                "prompt": "",
                "caption": "",
            }
        )
    for idx, key in enumerate(sorted(input_frames)[:12]):
        name = key.rsplit("/", 1)[-1]
        samples.append(
            {
                "group": "input",
                "id": name,
                "label": name,
                "caption": _orig_caption_for(name, idx),
                "thumbnail_key": key,
                "thumbnail_uri": _uri(key),
                "video_key": "",
                "video_uri": "",
                "tags": {},
                "prompt": "",
            }
        )

    multiply = curation.get("multiply", {}) if isinstance(curation, dict) else {}
    if not isinstance(multiply, dict):
        multiply = {}
    variant_count = multiply.get("variant_count") or curation.get("variant_count") or len(by_clip)
    fo_brain = fo_curation.get("brain", {}) if isinstance(fo_curation.get("brain"), dict) else {}
    fo_selection = fo_curation.get("selection", {}) if isinstance(fo_curation.get("selection"), dict) else {}
    summary = {
        "augmented_count": len(by_clip),
        "input_count": len(input_frames),
        "variant_count": int(variant_count or 0),
        "multiply_mode": str(multiply.get("mode") or curation.get("multiply_mode") or ""),
        "grade_score": grade.get("score") if isinstance(grade, dict) else None,
        "grade_decision": str(decision.get("decision") or "") if isinstance(decision, dict) else "",
        # Real FiftyOne curation surface (empty when the curate stage ran
        # report-only, i.e. outside the npa-fiftyone image).
        "curation_engine": str(curation.get("curation_engine") or "") if isinstance(curation, dict) else "",
        "curated_kept": curation.get("curated_kept") if isinstance(curation, dict) else None,
        "curated_dropped": curation.get("curated_dropped") if isinstance(curation, dict) else None,
        "near_duplicate_count": (
            fo_brain.get("near_duplicate_count")
            if fo_brain.get("near_duplicate_count") is not None
            else fo_selection.get("near_duplicate_count")
        ),
        "uniqueness": fo_brain.get("uniqueness", {}),
    }
    visualization = fo_curation.get("visualization", []) if isinstance(fo_curation.get("visualization"), list) else []
    return {
        "fields": sorted(tag_keys),
        "summary": summary,
        "samples": samples,
        "visualization": visualization,
    }
