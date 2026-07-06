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
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
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
