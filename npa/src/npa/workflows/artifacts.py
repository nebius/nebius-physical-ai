"""Artifact-first S3 discovery helpers for agent browsing."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace as dataclass_replace
from datetime import datetime, timezone
from functools import partial
import json
import math
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
                raise ArtifactDiscoveryError(
                    "run-id traversal segments are not allowed"
                )
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
_TEXT_EXTENSIONS = {".txt", ".log", ".csv", ".jsonl", ".yaml", ".yml", ".md"}
INLINE_TEXT_MAX_BYTES = 256 * 1024
_INPUT_ARTIFACT_ROOTS = {
    "data",
    "dataset",
    "datasets",
    "input",
    "inputs",
    "source",
    "sources",
}
_RENDER_ORDER = {
    "rerun": 0,
    "mcap": 1,
    "video": 2,
    "image": 3,
    "json": 4,
    "text": 5,
    "download": 6,
}
ARTIFACT_DISCOVERY_CONTRACT = "s3-source-qualified-v1"

# This is the authoritative artifact/semantic contract for the shipped GR00T
# offline workflow.  Backend responses serialize this contract for the browser;
# the UI must not maintain a second collection of filename heuristics.
GROOT_ARTIFACT_CONTRACT_SCHEMA = "npa.groot.artifacts/v1"
GROOT_ARTIFACT_PATHS: dict[str, tuple[str, ...]] = {
    "split_manifest": ("reports/split/manifest.json",),
    "report": (
        "reports/two-gpu-pipeline-report.json",
        "reports/learning-rigor-report.json",
        "reports/learning-report.json",
    ),
    "rrd": (
        "reports/groot-offline-evaluation.rrd",
        "reports/groot-learning.rrd",
    ),
    "mcap": (
        "reports/groot-offline-evaluation.mcap",
        "reports/groot-learning.mcap",
    ),
    "baseline_evaluation": (
        "offline/baseline/evaluation.json",
        "eval/baseline/evaluation.json",
    ),
    "trained_evaluation": (
        "offline/trained/evaluation.json",
        "eval/posttrain/evaluation.json",
    ),
    "training_manifest": (
        "checkpoints/candidate/npa_groot_finetune_manifest.json",
        "checkpoints/npa_groot_finetune_manifest.json",
    ),
    "checkpoint_reference": ("reports/trained-checkpoint.json",),
    "comparison_video": ("reports/offline-heldout-comparison.mp4",),
    "publish_manifest": ("reports/publish-manifest.json",),
    "workflow_spec": ("workflow.yaml",),
}
GROOT_ARTIFACT_STAGES: tuple[dict[str, Any], ...] = (
    {
        "id": "split",
        "label": "Prepare leakage-free split",
        "semantics": ("split_manifest",),
        "description": "Episode-disjoint train/held-out split with train-only statistics.",
    },
    {
        "id": "baseline",
        "label": "Offline baseline",
        "semantics": ("baseline_evaluation",),
        "description": "Deterministic held-out action prediction before training.",
    },
    {
        "id": "train",
        "label": "Multi-GPU policy training",
        "semantics": ("training_manifest", "checkpoint_reference"),
        "description": "Optimizer loss and selected candidate-checkpoint provenance.",
    },
    {
        "id": "posttrain",
        "label": "Offline post-training evaluation",
        "semantics": ("trained_evaluation", "report"),
        "description": "Held-out action prediction after training; not a rollout.",
    },
    {
        "id": "compare",
        "label": "Classify learning outcome",
        "semantics": ("report", "comparison_video"),
        "description": "Before/after action error and outcome, separate from pipeline status.",
    },
    {
        "id": "diagnostics",
        "label": "Synchronized diagnostics",
        "semantics": ("rrd", "mcap", "comparison_video"),
        "description": "RRD/MCAP diagnostics aligned to the held-out dataset timeline.",
    },
    {
        "id": "publish",
        "label": "Validate and publish",
        "semantics": ("publish_manifest", "workflow_spec"),
        "description": "Independent artifact inspection, hashes, and submitted workflow provenance.",
    },
)

# Bucket roots owned by infrastructure tooling are never user workflow runs.
# Keep this list deliberately narrow and structural: these are canonical state
# roots, not customer-selected bucket/run names or a deployment allowlist.
_INFRASTRUCTURE_ROOTS = frozenset(
    {".terraform", "terraform", "terraform-state", "tfstate"}
)
# Structural storage trees which contain platform state or source snapshots, not
# workflow/artifact runs. These are data contracts rather than customer names:
# ignoring them prevents a tenant/category root (notably ``tenants``) or a source
# cache whose children happen to look like run ids from becoming a fake run.
_STRUCTURAL_NON_RUN_PREFIXES = frozenset(
    {
        ".terraform",
        "terraform",
        "terraform-state",
        "terraform_state",
        "tfstate",
        "npa/terraform-state",
        "npa/terraform_state",
        "npa-src",
        "detached-source",
        "npa-agent/session-state",
        "npa-agent/tenants",
        "tenants",
    }
)
LIGHTWEIGHT_RUN_SUMMARY_PAGE_SIZE = 1000
RUN_DISCOVERY_MAX_WORKERS = 8
MAX_RUN_PARENT_CANDIDATES = 64
# Discovery follows every native S3 continuation page, but still has an explicit
# security/resource boundary. Hitting either cap is reported as incomplete; it
# is never presented as an exhaustive not-found result.
MAX_RUN_DISCOVERY_OBJECTS = 250_000
MAX_RUN_DISCOVERY_PAGES = 1_000


class ArtifactDiscoveryError(RuntimeError):
    """Raised when artifact discovery or retrieval fails."""


class AmbiguousRunSourceError(ArtifactDiscoveryError):
    """Raised when an unqualified run id resolves to multiple artifact sources."""


class AmbiguousRunError(AmbiguousRunSourceError):
    """Raised when a basename identifies more than one S3 run root."""

    def __init__(self, run_id: str, references: list[str]) -> None:
        self.run_id = run_id
        self.references = list(references)
        super().__init__(
            f"run_id {run_id!r} is ambiguous across {len(references)} artifact roots; "
            "use the source-qualified run_ref returned by discovery"
        )


@dataclass(frozen=True)
class Artifact:
    run_id: str
    key: str
    s3_uri: str
    size: int
    last_modified: str
    render: str
    inline: bool
    role: str = "output"
    namespace: str = ""
    relative_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        data_role = artifact_data_role(self.key, self.run_id)
        category = artifact_category_for_relative_key(self.relative_key, role=self.role)
        return {
            "run_id": self.run_id,
            "key": self.key,
            "s3_uri": self.s3_uri,
            "size": self.size,
            "last_modified": self.last_modified,
            "render": self.render,
            "inline": self.inline,
            "data_role": data_role["role"],
            "data_role_label": data_role["label"],
            "data_role_detail": data_role["detail"],
            "role": self.role,
            "namespace": self.namespace,
            "relative_key": self.relative_key,
            "category": category,
            "content_type": artifact_media_type(self.key),
            "download_only": not self.inline,
            "is_output": self.role == "output",
        }


@dataclass(frozen=True)
class ArtifactListPage:
    artifacts: list[Artifact]
    truncated: bool
    next_cursor: str
    page_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": [item.to_dict() for item in self.artifacts],
            "count": len(self.artifacts),
            "truncated": self.truncated,
            "next_cursor": self.next_cursor,
            "page_size": self.page_size,
        }


@dataclass(frozen=True)
class ArtifactSource:
    """One authorized direct run-parent in durable artifact storage.

    ``resolved_prefix`` is the exact directory *above* run ids.  Keeping the
    owning project alongside the bucket matters even when a provider happens to
    expose equal bucket names through one credential scope: authorization and
    ambiguity are defined by the complete source tuple, not by bucket name.

    Args:
        project_id: Project whose credentials authorize this source.
        bucket: Object-storage bucket containing the source.
        resolved_prefix: Exact object prefix directly above run ids.

    Returns:
        None.

    Raises:
        ArtifactDiscoveryError: If any source identity field is invalid.
    """

    project_id: str
    bucket: str
    resolved_prefix: str = ""

    def __post_init__(self) -> None:
        project_id = str(self.project_id or "").strip()
        bucket = str(self.bucket or "").strip()
        resolved_prefix = _validate_source_prefix(self.resolved_prefix)
        if (
            not project_id
            or len(project_id) > 255
            or "/" in project_id
            or "\\" in project_id
            or any(ord(char) < 33 for char in project_id)
        ):
            raise ArtifactDiscoveryError("artifact source project_id is invalid")
        if not _SAFE_BUCKET_RE.fullmatch(bucket):
            raise ArtifactDiscoveryError("artifact source bucket is invalid")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(self, "resolved_prefix", resolved_prefix)

    @property
    def identity(self) -> tuple[str, str, str]:
        """Return the complete tuple used for authorization and de-duplication.

        Args:
            None.

        Returns:
            The project, bucket, and resolved-prefix tuple.

        Raises:
            None.
        """
        return (self.project_id, self.bucket, self.resolved_prefix)


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    last_modified: str
    artifact_count: int
    has_viewable: bool | None
    bucket: str = ""
    project_id: str = ""
    summary_complete: bool = True
    # When the run STARTED (its id-encoded submit time, or the earliest artifact
    # write as a fallback) — distinct from last_modified (newest artifact write).
    started_at: str = ""
    # Exact directory *above* run_id. This is part of source identity because
    # the same run id may legitimately exist under multiple prefixes/buckets.
    resolved_prefix: str = ""
    output_artifact_count: int = 0
    input_artifact_count: int = 0
    metadata_artifact_count: int = 0
    namespaces: tuple[str, ...] = ()
    canonical_score: int = 0

    @property
    def run_ref(self) -> str:
        if not self.bucket:
            return ""
        return encode_run_ref(self.bucket, self.resolved_prefix, self.run_id)

    @property
    def source_prefix(self) -> str:
        """Compatibility alias for the exact directory above ``run_id``."""
        return self.resolved_prefix

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "last_modified": self.last_modified,
            "started_at": self.started_at,
            "artifact_count": self.artifact_count,
            "has_viewable": self.has_viewable,
            "bucket": self.bucket,
            "project_id": self.project_id,
            "resolved_prefix": self.resolved_prefix,
            "source_prefix": self.resolved_prefix,
            "run_ref": self.run_ref,
            "summary_complete": self.summary_complete,
            "output_artifact_count": self.output_artifact_count,
            "input_artifact_count": self.input_artifact_count,
            "metadata_artifact_count": self.metadata_artifact_count,
            "namespaces": list(self.namespaces),
            "canonical_score": self.canonical_score,
            "canonical_candidate": bool(
                self.input_artifact_count > 0
                and self.output_artifact_count > 1
                and self.canonical_score > 0
            ),
            "source_type": "artifact_storage",
            "source_label": "S3 artifacts",
        }


def _canonical_root(value: str) -> str:
    return (
        str(value or "").strip().strip("/").split("/", 1)[0].lower().replace("_", "-")
    )


def _normalized_discovery_exclusions(exclude: "set[str] | None") -> set[str]:
    return {
        str(value).strip().strip("/").lower()
        for value in (exclude or set())
        if str(value).strip().strip("/")
    }


def _is_excluded_prefix(value: str, excluded: set[str]) -> bool:
    normalized = str(value or "").strip().strip("/").lower()
    return any(
        normalized == candidate or normalized.startswith(candidate + "/")
        for candidate in excluded
    )


def is_infrastructure_root(value: str) -> bool:
    """Return whether a bucket-root prefix belongs to infrastructure state."""
    return _canonical_root(value) in _INFRASTRUCTURE_ROOTS


def _looks_like_flat_run_prefix(prefix: str) -> bool:
    """Best-effort distinguish a root-level run from a workflow category.

    S3 has no directory metadata, so ``<run>/<stage>/file`` and
    ``<category>/<run>/file`` are structurally identical. Real run identifiers
    conventionally carry an encoded timestamp (or an explicit ``run`` token),
    which is the useful signal available without walking every object.
    """
    leaf = str(prefix or "").strip().strip("/").rsplit("/", 1)[-1]
    if not leaf or is_infrastructure_root(leaf):
        return False
    if _parse_run_id_timestamps(leaf):
        return True
    return bool(re.search(r"(?:^|[-_.])run(?:$|[-_.])", leaf, re.IGNORECASE))


@dataclass(frozen=True)
class RunListPage:
    runs: list[RunSummary]
    truncated: bool
    total_runs: int
    limit: int
    discovery_complete: bool = True
    source_errors: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": [item.to_dict() for item in self.runs],
            "count": len(self.runs),
            "truncated": self.truncated,
            "total_runs": self.total_runs,
            "limit": self.limit,
            "pagination_complete": self.discovery_complete,
            "source_errors": [dict(item) for item in self.source_errors],
        }


def _run_storage_path(run: RunSummary) -> str:
    prefix = str(run.resolved_prefix or "").strip("/")
    run_id = str(run.run_id or "").strip("/")
    return "/".join(part for part in (prefix, run_id) if part)


def _artifact_source_covers_run(source: ArtifactSource, run: RunSummary) -> bool:
    if source.bucket != run.bucket:
        return False
    if source.project_id and run.project_id != source.project_id:
        return False
    prefix = str(source.resolved_prefix or "").strip("/")
    run_path = _run_storage_path(run)
    return not prefix or run_path == prefix or run_path.startswith(prefix + "/")


def exclude_run_source_subtrees(
    page: RunListPage,
    sources: "list[ArtifactSource] | tuple[ArtifactSource, ...]",
) -> RunListPage:
    """Remove generic rows covered by exact sources while preserving S3 case.

    Args:
        page: Generic-discovery page to filter.
        sources: Authorized exact sources whose subtrees are covered elsewhere.

    Returns:
        A page with covered rows removed and its observed total adjusted.

    Raises:
        None.
    """
    exact_sources = tuple(sources)
    runs = [
        run
        for run in page.runs
        if not any(_artifact_source_covers_run(source, run) for source in exact_sources)
    ]

    removed = len(page.runs) - len(runs)
    return RunListPage(
        runs=runs,
        truncated=page.truncated,
        total_runs=max(0, int(page.total_runs) - removed),
        limit=page.limit,
        discovery_complete=page.discovery_complete,
        source_errors=page.source_errors,
    )


def _without_infrastructure_roots(page: RunListPage) -> RunListPage:
    """Remove infrastructure-only root rows from a fallback run page."""
    runs = [item for item in page.runs if not is_infrastructure_root(item.run_id)]
    removed = len(page.runs) - len(runs)
    return RunListPage(
        runs=runs,
        truncated=page.truncated,
        total_runs=max(0, page.total_runs - removed),
        limit=page.limit,
        discovery_complete=page.discovery_complete,
        source_errors=page.source_errors,
    )


@dataclass(frozen=True)
class RunResolution:
    """An exact, source-qualified S3 run and its listed artifacts."""

    run_id: str
    bucket: str
    source_prefix: str
    artifacts: list[Artifact]

    @property
    def run_ref(self) -> str:
        return encode_run_ref(self.bucket, self.source_prefix, self.run_id)


def _merge_staging_resolutions(
    matches: list[RunResolution],
) -> list[RunResolution]:
    """De-duplicate repeated observations without crossing source tuples.

    Input/output roles are not proof that two prefixes share one authorization
    identity. Distinct source tuples therefore remain distinct and force a
    caller using a plain run id to select the server-issued tuple explicitly.
    """

    grouped: dict[tuple[str, str], list[RunResolution]] = {}
    for match in matches:
        grouped.setdefault((match.bucket, match.run_id), []).append(match)
    merged: list[RunResolution] = []
    for same_run in grouped.values():
        merged.extend(_merge_staging_resolution_group(same_run))
    return merged


def _merge_staging_resolution_group(
    matches: list[RunResolution],
) -> list[RunResolution]:
    """Merge duplicate observations of the same exact source only."""
    by_source: dict[tuple[str, str], RunResolution] = {}
    for match in matches:
        source_key = (match.source_prefix, match.run_id)
        existing = by_source.get(source_key)
        if existing is None:
            by_source[source_key] = match
            continue
        artifacts = {artifact.key: artifact for artifact in existing.artifacts}
        for artifact in match.artifacts:
            previous = artifacts.get(artifact.key)
            if previous is not None and previous != artifact:
                return matches
            artifacts[artifact.key] = artifact
        by_source[source_key] = RunResolution(
            run_id=match.run_id,
            bucket=match.bucket,
            source_prefix=match.source_prefix,
            artifacts=sorted(artifacts.values(), key=lambda item: item.key),
        )
    return list(by_source.values())


def _merge_staging_summaries(runs: list[RunSummary]) -> list[RunSummary]:
    """De-duplicate repeated summaries without collapsing distinct sources."""
    merged: dict[tuple[str, str, str, str], RunSummary] = {}
    for run in runs:
        identity = (
            str(run.project_id or ""),
            str(run.bucket or ""),
            str(run.resolved_prefix or "").strip("/"),
            str(run.run_id or ""),
        )
        existing = merged.get(identity)
        merged[identity] = (
            _merge_same_run_summary(existing, run) if existing is not None else run
        )
    return list(merged.values())


_RUN_REF_PREFIX = "npa1_"
_SAFE_BUCKET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,62}$")


def _validate_run_basename(run_id: str) -> str:
    try:
        value = validate_run_id(str(run_id or "").strip())
    except Exception as exc:
        raise ArtifactDiscoveryError(str(exc)) from exc
    if "/" in value or "\\" in value:
        raise ArtifactDiscoveryError("run_id must be a single path segment")
    return value


def _validate_source_prefix(prefix: str) -> str:
    value = str(prefix or "").strip().strip("/")
    if not value:
        return ""
    if "\\" in value or any(ord(ch) < 32 for ch in value):
        raise ArtifactDiscoveryError("source prefix contains unsupported characters")
    if any(segment in {"", ".", ".."} for segment in value.split("/")):
        raise ArtifactDiscoveryError("source prefix contains traversal segments")
    return value


def encode_run_ref(bucket: str, source_prefix: str, run_id: str) -> str:
    """Return a stable URL-safe identity for one exact S3 run root."""
    safe_bucket = str(bucket or "").strip()
    if not _SAFE_BUCKET_RE.fullmatch(safe_bucket):
        raise ArtifactDiscoveryError("invalid S3 bucket in run reference")
    payload = [
        safe_bucket,
        _validate_source_prefix(source_prefix),
        _validate_run_basename(run_id),
    ]
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return _RUN_REF_PREFIX + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_run_ref(run_ref: str) -> tuple[str, str, str]:
    """Decode and validate a source-qualified run reference, failing closed."""
    value = str(run_ref or "").strip()
    if not value.startswith(_RUN_REF_PREFIX) or len(value) > 4096:
        raise ArtifactDiscoveryError("invalid run_ref")
    token = value[len(_RUN_REF_PREFIX) :]
    if not token or not re.fullmatch(r"[A-Za-z0-9_-]+", token):
        raise ArtifactDiscoveryError("invalid run_ref")
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ArtifactDiscoveryError("invalid run_ref") from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 3
        or not all(isinstance(item, str) for item in payload)
    ):
        raise ArtifactDiscoveryError("invalid run_ref")
    bucket, source_prefix, run_id = payload
    if not _SAFE_BUCKET_RE.fullmatch(bucket):
        raise ArtifactDiscoveryError("invalid S3 bucket in run_ref")
    return (
        bucket,
        _validate_source_prefix(source_prefix),
        _validate_run_basename(run_id),
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ArtifactDiscoveryError(f"expected s3:// URI, got {uri!r}")
    key = parsed.path.lstrip("/")
    if not key:
        raise ArtifactDiscoveryError(f"S3 URI missing object key: {uri!r}")
    return parsed.netloc, key


def authorize_artifact_inventory_key(
    run_id: str,
    key: str,
    inventory_keys: "list[str] | tuple[str, ...] | set[str]",
) -> str:
    """Authorize one S3 key against the already-resolved inventory for a run.

    The inventory must come from S3-first discovery for ``run_id``. Requiring an
    exact match means S3 credentials can never turn this endpoint into an
    arbitrary object reader, even when the credentials can access other runs.
    """
    normalized_run = validate_run_id(run_id)
    normalized_key = str(key or "").strip().lstrip("/")
    if not normalized_key:
        raise ArtifactDiscoveryError("artifact key is required")
    if "\\" in normalized_key or any(
        segment in {"", ".", ".."} for segment in normalized_key.split("/")
    ):
        raise ArtifactDiscoveryError("artifact key traversal is not allowed")
    if normalized_key not in {str(item) for item in inventory_keys}:
        raise ArtifactDiscoveryError(
            f"artifact key is not part of validated run {normalized_run!r}"
        )
    return normalized_key


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


def artifact_role_for_relative_key(relative_key: str) -> str:
    """Classify an object relative to its run root without workflow allowlists."""
    relative = str(relative_key or "").strip().lstrip("/")
    first = relative.split("/", 1)[0].lower() if relative else ""
    return "input" if first in _INPUT_ARTIFACT_ROOTS else "output"


def artifact_category_for_relative_key(
    relative_key: str, *, role: str = "output"
) -> str:
    """Return a user-facing artifact category without inspecting object bytes."""
    relative = str(relative_key or "").strip().lstrip("/")
    lowered = relative.lower()
    leaf = Path(lowered).name
    first = lowered.split("/", 1)[0] if lowered else ""
    suffix = Path(leaf).suffix
    if leaf == "manifest.json" or leaf.endswith("_manifest.json"):
        return "manifest"
    if first in {"checkpoint", "checkpoints"} or suffix in {
        ".bin",
        ".ckpt",
        ".pth",
        ".pt",
        ".safetensors",
    }:
        return "checkpoint"
    if suffix == ".log" or first in {"log", "logs", "evidence"}:
        return "log"
    if (
        suffix in {".yaml", ".yml"}
        or "config" in leaf
        or first in {"config", "configs"}
    ):
        return "config"
    normalized_role = str(role or "output").lower()
    if normalized_role == "input":
        return "input"
    if normalized_role == "metadata":
        return "staged"
    return "output"


def artifact_inventory_counts(artifacts: "list[Artifact]") -> dict[str, int]:
    counts = {"output": 0, "input": 0, "metadata": 0}
    for item in artifacts:
        role = str(item.role or "output").lower()
        if role not in counts:
            role = "metadata"
        counts[role] += 1
    return counts


def _artifact_output_signal_score(relative_key: str) -> int:
    """Score evidence that a namespace is the authoritative output root."""
    relative = str(relative_key or "").strip().lstrip("/").lower()
    first = relative.split("/", 1)[0] if relative else ""
    if relative == "manifest.json":
        return 10_000
    if first in {"checkpoint", "checkpoints"}:
        return 1_000
    if first in {"artifacts", "evidence", "output", "outputs", "reports", "results"}:
        return 100
    return 0


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
        ".jsonl": "application/x-ndjson; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
        ".log": "text/plain; charset=utf-8",
        ".md": "text/plain; charset=utf-8",
        ".csv": "text/plain; charset=utf-8",
        ".yaml": "text/plain; charset=utf-8",
        ".yml": "text/plain; charset=utf-8",
        ".bin": "application/octet-stream",
        ".ckpt": "application/octet-stream",
        ".pt": "application/octet-stream",
        ".pth": "application/octet-stream",
        ".safetensors": "application/octet-stream",
    }
    if suffix in explicit:
        return explicit[suffix]
    guessed, _ = mimetypes.guess_type(name)
    if guessed:
        return str(guessed)
    return "application/octet-stream"


_SECRET_KEY_RE = re.compile(
    r"(?i)(authorization|cookie|password|passwd|private[_-]?key|secret|token|"
    r"access[_-]?key|api[_-]?key|client[_-]?secret)"
)
_SECRET_LINE_RE = re.compile(
    r"(?im)^(?P<prefix>\s*[\"']?[A-Za-z0-9_.-]*"
    r"(?:authorization|cookie|password|passwd|private[_-]?key|secret|token|"
    r"access[_-]?key|api[_-]?key|client[_-]?secret)[\"']?\s*[:=]\s*)"
    r"(?P<value>[^\r\n,}]+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{8,}=*")
_AWS_ACCESS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_PEM_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?-----END [^-\r\n]*PRIVATE KEY-----",
    re.DOTALL,
)


def _redact_json_value(value: Any) -> tuple[Any, bool]:
    redacted = False
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            if _SECRET_KEY_RE.search(str(key)):
                out[str(key)] = "[REDACTED]"
                redacted = True
            else:
                clean, changed = _redact_json_value(child)
                out[str(key)] = clean
                redacted = redacted or changed
        return out, redacted
    if isinstance(value, list):
        out_list = []
        for child in value:
            clean, changed = _redact_json_value(child)
            out_list.append(clean)
            redacted = redacted or changed
        return out_list, redacted
    if isinstance(value, str):
        clean, changed = redact_artifact_text(value)
        return clean, changed
    return value, False


def redact_artifact_text(text: str) -> tuple[str, bool]:
    """Redact common credential forms from an inline text preview."""
    value = str(text or "")
    original = value
    value = _PEM_PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", value)
    value = _BEARER_RE.sub("Bearer [REDACTED]", value)
    value = _AWS_ACCESS_KEY_RE.sub("[REDACTED ACCESS KEY]", value)
    value = _SECRET_LINE_RE.sub(
        lambda match: match.group("prefix") + "[REDACTED]", value
    )
    return value, value != original


def build_text_preview(
    data: bytes,
    *,
    total_bytes: int,
    render: str,
    max_bytes: int = INLINE_TEXT_MAX_BYTES,
) -> dict[str, Any]:
    """Decode, redact, and format a bounded UTF-8 artifact preview."""
    if max_bytes <= 0:
        raise ArtifactDiscoveryError("text preview max_bytes must be > 0")
    raw = bytes(data or b"")
    bounded = raw[:max_bytes]
    text = bounded.decode("utf-8", errors="replace")
    redacted = False
    normalized_render = str(render or "text").lower()
    if normalized_render == "json":
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            text, redacted = redact_artifact_text(text)
        else:
            parsed, redacted = _redact_json_value(parsed)
            text = json.dumps(parsed, indent=2, sort_keys=True, ensure_ascii=False)
    else:
        text, redacted = redact_artifact_text(text)
    total = max(0, int(total_bytes or 0))
    summary = {
        "text": text,
        "truncated": total > len(bounded) or len(raw) > len(bounded),
        "bytes_read": len(bounded),
        "total_bytes": total,
        "max_bytes": int(max_bytes),
        "encoding": "utf-8",
        "invalid_utf8_replaced": "\ufffd" in text,
        "redacted": redacted,
        "render": normalized_render,
    }
    return summary


def parse_http_byte_range(value: str, total_bytes: int) -> tuple[int, int] | None:
    """Parse one RFC 7233 byte range; reject multiple or unsatisfiable ranges."""
    raw = str(value or "").strip()
    if not raw:
        return None
    total = int(total_bytes)
    if total < 0 or not raw.lower().startswith("bytes=") or "," in raw:
        raise ArtifactDiscoveryError("invalid byte range")
    spec = raw.split("=", 1)[1].strip()
    if "-" not in spec:
        raise ArtifactDiscoveryError("invalid byte range")
    start_raw, end_raw = (part.strip() for part in spec.split("-", 1))
    if not start_raw:
        if not end_raw.isdigit() or int(end_raw) <= 0 or total <= 0:
            raise ArtifactDiscoveryError("unsatisfiable byte range")
        length = min(int(end_raw), total)
        return total - length, total - 1
    if not start_raw.isdigit():
        raise ArtifactDiscoveryError("invalid byte range")
    start = int(start_raw)
    if start >= total:
        raise ArtifactDiscoveryError("unsatisfiable byte range")
    if end_raw:
        if not end_raw.isdigit():
            raise ArtifactDiscoveryError("invalid byte range")
        end = min(int(end_raw), total - 1)
        if end < start:
            raise ArtifactDiscoveryError("unsatisfiable byte range")
    else:
        end = total - 1
    return start, end


def safe_artifact_filename(key: str) -> str:
    """Return a conservative ASCII download filename for Content-Disposition."""
    leaf = Path(str(key or "").replace("\\", "/")).name or "artifact.bin"
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf).strip("._")
    return clean[:180] or "artifact.bin"


def safe_content_disposition(key: str, *, attachment: bool) -> str:
    disposition = "attachment" if attachment else "inline"
    return f'{disposition}; filename="{safe_artifact_filename(key)}"'


def _relative_artifact_key(artifact: Artifact) -> str:
    relative = str(artifact.relative_key or "").strip().strip("/").lower()
    if relative:
        return relative
    key = str(artifact.key or "").strip().strip("/").lower()
    for aliases in GROOT_ARTIFACT_PATHS.values():
        for alias in aliases:
            if key == alias or key.endswith("/" + alias):
                return alias
    return key


def groot_artifact_contract(
    artifacts: list[Artifact],
    learning_report: dict[str, Any],
) -> dict[str, Any]:
    """Serialize path matches only after report metadata establishes semantics.

    A matching filename is useful for locating an object, but it is not enough
    to classify an arbitrary run as GR00T.  The report schema, evaluation kind,
    and provenance establish that meaning; retained aliases are then matched
    through this one Python-owned table.
    """
    report: dict[str, Any] = (
        learning_report if isinstance(learning_report, dict) else {}
    )
    authoritative = (
        report.get("schema") == "npa.groot.learning.v1"
        and str(report.get("evaluation_kind") or "").strip()
        == "offline held-out policy evaluation"
        and report.get("closed_loop") is False
    )
    inventory = {_relative_artifact_key(item): item for item in artifacts}
    matches: dict[str, list[str]] = {}
    for semantic, aliases in GROOT_ARTIFACT_PATHS.items():
        matches[semantic] = [alias for alias in aliases if alias.lower() in inventory]

    dataset_value = report.get("dataset")
    dataset: dict[str, Any] = dataset_value if isinstance(dataset_value, dict) else {}
    provenance_value = report.get("provenance")
    provenance: dict[str, Any] = (
        provenance_value if isinstance(provenance_value, dict) else {}
    )
    camera_names = [
        str(value).strip()
        for value in dataset.get("camera_names") or []
        if str(value).strip()
    ]
    primary_camera = str(provenance.get("primary_camera") or "").strip()
    camera_valid = bool(primary_camera and primary_camera in camera_names)
    if authoritative and not camera_valid:
        # A semantically valid report with contradictory modality provenance is
        # not safe input for a viewer camera/topic selection.
        authoritative = False

    return {
        "schema": GROOT_ARTIFACT_CONTRACT_SCHEMA,
        "authoritative": authoritative,
        "evaluation_kind": str(report.get("evaluation_kind") or ""),
        "closed_loop": report.get("closed_loop") is True,
        "primary_camera": primary_camera if camera_valid else "",
        "camera_names": camera_names,
        "paths": {key: list(value) for key, value in GROOT_ARTIFACT_PATHS.items()},
        "matches": matches if authoritative else {},
        "stages": [
            {**stage, "semantics": list(stage["semantics"])}
            for stage in GROOT_ARTIFACT_STAGES
        ],
    }


def _finite_loss_history(
    value: Any,
    *,
    step_keys: tuple[str, ...] = ("optimizer_step", "step"),
) -> list[dict[str, Any]] | None:
    """Return validated optimizer-step loss points, or ``None`` if malformed."""
    if not isinstance(value, list) or not value:
        return None
    points: list[dict[str, Any]] = []
    previous = 0
    for raw in value:
        if not isinstance(raw, dict):
            return None
        step_value = next(
            (raw.get(key) for key in step_keys if raw.get(key) is not None), None
        )
        try:
            step = int(str(step_value))
            loss = float(str(raw.get("loss")))
        except (TypeError, ValueError):
            return None
        if step <= previous or step <= 0 or not math.isfinite(loss):
            return None
        points.append({"optimizer_step": step, "loss": loss})
        previous = step
    return points


def build_run_summary(
    run_id: str,
    artifacts: list[Artifact],
    documents: dict[str, Any],
) -> dict[str, Any]:
    """Build a concise run summary from allowlisted manifest/evidence fields."""
    docs = documents if isinstance(documents, dict) else {}

    def _doc(suffix: str) -> dict[str, Any]:
        normalized = suffix.lower().strip("/")
        for key, value in docs.items():
            if str(key).lower().strip("/") == normalized and isinstance(value, dict):
                return value
        for key, value in docs.items():
            if str(key).lower().endswith(suffix.lower()) and isinstance(value, dict):
                return value
        return {}

    def _semantic_doc(semantic: str) -> tuple[str, dict[str, Any]]:
        for path in GROOT_ARTIFACT_PATHS[semantic]:
            document = _doc(path)
            if document:
                return path, document
        return "", {}

    root_manifest = _doc("manifest.json")
    workflow_manifest = _doc("npa-workflow/manifest.json")
    training = _doc("evidence/training.json")
    capacity = _doc("evidence/capacity.json")
    collective = _doc("evidence/collective.json")
    checkpoint_path, checkpoint = _semantic_doc("training_manifest")
    learning_path, learning_doc = _semantic_doc("report")
    learning: dict[str, Any] = {}
    if learning_doc.get("schema") == "npa.groot.learning.v1":
        learning_dataset_value = learning_doc.get("dataset")
        learning_dataset: dict[str, Any] = (
            learning_dataset_value if isinstance(learning_dataset_value, dict) else {}
        )
        learning_training_value = learning_doc.get("training")
        learning_training: dict[str, Any] = (
            learning_training_value if isinstance(learning_training_value, dict) else {}
        )
        learning_eval_value = learning_doc.get("evaluation")
        learning_eval: dict[str, Any] = (
            learning_eval_value if isinstance(learning_eval_value, dict) else {}
        )
        learning_viz_value = learning_doc.get("visualizations")
        learning_viz: dict[str, Any] = (
            learning_viz_value if isinstance(learning_viz_value, dict) else {}
        )
        learning = {
            "badge": str(
                learning_doc.get("badge") or "Offline held-out policy evaluation"
            ),
            "pipeline_status": str(learning_doc.get("pipeline_status") or "unknown"),
            "learning_outcome": str(
                learning_doc.get("learning_outcome") or "inconclusive"
            ),
            "candidate_promoted": learning_doc.get("candidate_promoted") is True,
            "evaluation_kind": str(learning_doc.get("evaluation_kind") or ""),
            "closed_loop": learning_doc.get("closed_loop") is True,
            "embodiment": str(learning_dataset.get("embodiment") or ""),
            "camera_names": [
                str(value) for value in learning_dataset.get("camera_names") or []
            ],
            "source_resolution": str(
                learning_dataset.get("source_resolution") or "unknown"
            ),
            "train_episodes": int(learning_dataset.get("train_episodes") or 0),
            "heldout_episodes": int(learning_dataset.get("heldout_episodes") or 0),
            "heldout_samples": int(learning_dataset.get("heldout_samples") or 0),
            "split_hash": str(learning_dataset.get("split_hash") or ""),
            "leakage_free": learning_dataset.get("leakage_free") is True,
            "gpu_count": int(learning_training.get("gpu_count") or 0),
            "optimizer_steps": int(learning_training.get("optimizer_steps") or 0),
            "training_examples": int(learning_training.get("training_examples") or 0),
            "epoch_equivalent": float(learning_training.get("epoch_equivalent") or 0.0),
            "checkpoint_uri": str(learning_training.get("checkpoint_uri") or ""),
            "metric_name": str(learning_eval.get("metric_name") or "action_mse"),
            "baseline_value": float(learning_eval.get("baseline_value") or 0.0),
            "posttrain_value": float(learning_eval.get("posttrain_value") or 0.0),
            "absolute_improvement": float(
                learning_eval.get("absolute_improvement") or 0.0
            ),
            "relative_improvement_percent": float(
                learning_eval.get("relative_improvement_percent") or 0.0
            ),
            "improved": learning_eval.get("improved") is True,
            "gate_passed": learning_eval.get("gate_passed") is True,
            "zero_predictor_mse": float(learning_eval.get("zero_predictor_mse") or 0.0),
            "train_mean_predictor_mse": float(
                learning_eval.get("train_mean_predictor_mse") or 0.0
            ),
            "baseline_skill_score": float(
                learning_eval.get("baseline_skill_score") or 0.0
            ),
            "posttrain_skill_score": float(
                learning_eval.get("posttrain_skill_score") or 0.0
            ),
            "repeat_noise_multiple": float(
                (learning_eval.get("gate") or {}).get("repeat_noise_multiple") or 0.0
            ),
            "per_horizon_mse": learning_eval.get("per_horizon_mse") or {},
            "loss_trend": learning_training.get("loss_trend") or {},
            "checkpoint_selection_uri": str(
                learning_eval.get("checkpoint_selection_uri") or ""
            ),
            "per_dimension": learning_eval.get("per_dimension") or [],
            "mcap_uri": str(learning_viz.get("mcap_uri") or ""),
            "rrd_uri": str(learning_viz.get("rrd_uri") or ""),
            "comparison_video_uri": str(
                learning_viz.get("comparison_video_uri")
                or (learning_viz.get("comparison_video") or {}).get("uri")
                or ""
            ),
            "native_resolution_preserved": learning_viz.get(
                "native_resolution_preserved"
            )
            is True,
            "semantic_phases": [
                str(value) for value in learning_doc.get("semantic_phases") or []
            ],
        }
        learning["artifact_contract"] = groot_artifact_contract(artifacts, learning_doc)
    workflow = str(
        root_manifest.get("workflow_name")
        or workflow_manifest.get("workflow")
        or workflow_manifest.get("name")
        or ""
    )
    steps_value = workflow_manifest.get("steps")
    steps: list[Any] = steps_value if isinstance(steps_value, list) else []
    tool_ref = ""
    for step in steps:
        if isinstance(step, dict) and step.get("tool_ref"):
            tool_ref = str(step["tool_ref"])
            break
    status = str(
        learning_doc.get("status")
        or training.get("status")
        or training.get("terminal_status")
        or checkpoint.get("status")
        or root_manifest.get("status")
        or "unknown"
    )
    accelerator_count = int(
        (learning.get("gpu_count") if learning else 0)
        or training.get("gpu_count")
        or training.get("distinct_gpu_count")
        or checkpoint.get("num_gpus")
        or capacity.get("total_allocatable")
        or 0
    )
    accelerator_type = str(
        training.get("accelerator")
        or collective.get("accelerator")
        or capacity.get("accelerator")
        or ""
    )
    world_size = int(training.get("world_size") or collective.get("world_size") or 0)
    training_steps = int(
        (learning.get("optimizer_steps") if learning else 0)
        or training.get("optimizer_steps")
        or checkpoint.get("training_step")
        or checkpoint.get("max_steps")
        or 0
    )
    report_training_value = learning_doc.get("training")
    report_training: dict[str, Any] = (
        report_training_value if isinstance(report_training_value, dict) else {}
    )
    loss_sources = (
        (learning_path, report_training, "final_loss", "report"),
        (checkpoint_path, checkpoint, "train_loss", "training_manifest"),
        ("evidence/training.json", training, "train_loss", "generic_training"),
    )
    loss: float | None = None
    loss_history: list[dict[str, Any]] = []
    loss_source = ""
    loss_error = "no_loss_evidence"
    for source, payload, final_key, source_kind in loss_sources:
        if not payload:
            continue
        raw_history = payload.get("loss_history")
        if raw_history is None:
            raw_history = payload.get("losses")
        final_value = payload.get(final_key)
        if final_value is None and source_kind == "training_manifest":
            final_value = payload.get("final_step_loss")
        if final_value is None and source.endswith("manifest.json"):
            final_value = payload.get("final_loss")
        # Presence at a higher-precedence source is intentional: malformed or
        # non-finite evidence must remain visible as invalid, never be hidden by
        # a weaker fallback document.
        if raw_history is None and final_value is None:
            continue
        loss_source = source
        parsed_history = _finite_loss_history(raw_history)
        try:
            candidate_loss = float(final_value) if final_value is not None else None
        except (TypeError, ValueError):
            candidate_loss = None
        if (
            parsed_history
            and candidate_loss is not None
            and math.isfinite(candidate_loss)
            and math.isclose(
                candidate_loss, parsed_history[-1]["loss"], rel_tol=1e-9, abs_tol=1e-12
            )
        ):
            loss = candidate_loss
            loss_history = parsed_history
            loss_error = ""
        else:
            loss_error = "malformed_or_nonfinite_loss_evidence"
        break
    # The generic training evidence contract predates per-step trajectories and
    # may honestly contain only a finite terminal loss.  Preserve that signal
    # for non-GR00T summaries, while reports/manifests claiming a trajectory
    # remain subject to the stricter history/final-value consistency check.
    terminal_only_loss = (
        loss_source == "evidence/training.json"
        and not loss_history
        and training.get("loss_finite") is True
    )
    if terminal_only_loss:
        try:
            terminal = float(str(training.get("train_loss")))
        except (TypeError, ValueError):
            terminal = math.nan
        if math.isfinite(terminal):
            loss = terminal
            loss_error = ""
    finite_loss = loss is not None and (bool(loss_history) or terminal_only_loss)
    counts = artifact_inventory_counts(artifacts)
    has_recording = any(item.render in {"rerun", "mcap"} for item in artifacts)
    summary = {
        "run_id": str(run_id),
        "completion_status": status,
        "workflow": workflow,
        "tool": tool_ref or workflow,
        "accelerator_count": accelerator_count,
        "accelerator_type": accelerator_type,
        "world_size": world_size,
        "training_steps": training_steps,
        "loss": loss if finite_loss else None,
        "finite_loss": finite_loss,
        "loss_history": loss_history if finite_loss else [],
        "loss_point_count": len(loss_history) if finite_loss else 0,
        "loss_source": loss_source,
        "loss_validation_error": loss_error,
        "artifact_count": len(artifacts),
        "output_artifact_count": counts["output"],
        "input_artifact_count": counts["input"],
        "metadata_artifact_count": counts["metadata"],
        "total_bytes": sum(max(0, int(item.size or 0)) for item in artifacts),
        "has_recording": has_recording,
        "recording_state": (
            "Recording available"
            if has_recording
            else "No RRD/MCAP recording; use the artifacts below"
        ),
    }
    if learning:
        summary["learning"] = learning
    return summary


def list_runs(
    bucket: str,
    *,
    prefix: str = "",
    limit: int = 50,
    contains: str = "",
    exclude: "set[str] | None" = None,
    s3=None,
) -> RunListPage:
    if limit <= 0:
        raise ArtifactDiscoveryError("limit must be > 0")
    client = s3
    if client is None:
        raise ArtifactDiscoveryError("s3 client is required")
    normalized_prefix = _normalize_prefix(prefix)
    excluded = _normalized_discovery_exclusions(exclude)
    summary: dict[str, dict[str, Any]] = {}
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
            for item in page.get("Contents", []) or []:
                key = str(item.get("Key") or "")
                run_id = _run_id_for_key(key, normalized_prefix)
                if not run_id or is_infrastructure_root(run_id):
                    continue
                discovered_path = "/".join(
                    part for part in (normalized_prefix.strip("/"), run_id) if part
                )
                if _is_excluded_prefix(discovered_path, excluded):
                    continue
                try:
                    run_id = _validate_run_basename(run_id)
                except ArtifactDiscoveryError:
                    continue
                # A run is a directory (``<run_id>/<stage>/...``). Skip bare files
                # sitting directly under the prefix (e.g. ``<cat>/records.json``),
                # which are not runs — this keeps generic root-level discovery clean.
                remainder = key[len(normalized_prefix) :] if normalized_prefix else key
                if "/" not in remainder.lstrip("/"):
                    continue
                render = render_hint_for_object(key=key)
                current = summary.setdefault(
                    run_id,
                    {
                        "artifact_count": 0,
                        "last_modified": "",
                        "earliest": "",
                        "has_viewable": False,
                    },
                )
                current["artifact_count"] = int(current["artifact_count"]) + 1
                current["has_viewable"] = bool(
                    current["has_viewable"] or render != "download"
                )
                ts = _to_iso8601(item.get("LastModified"))
                if ts:
                    if ts > str(current["last_modified"]):
                        current["last_modified"] = ts
                    if not current["earliest"] or ts < str(current["earliest"]):
                        current["earliest"] = ts
    except (ClientError, BotoCoreError) as exc:
        raise ArtifactDiscoveryError(
            f"failed to list runs from s3://{bucket}/{normalized_prefix}: {exc}"
        ) from exc

    runs = [
        RunSummary(
            run_id=run_id,
            last_modified=str(payload["last_modified"]),
            started_at=_run_started_at(run_id, str(payload.get("earliest") or "")),
            artifact_count=int(payload["artifact_count"]),
            has_viewable=bool(payload["has_viewable"]),
            bucket=bucket,
            resolved_prefix=normalized_prefix.rstrip("/"),
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


def list_run_categories(
    bucket: str,
    *,
    base_prefix: str = "",
    max_results: int | None = None,
    s3=None,
) -> list[str]:
    """Return the immediate sub-directory prefixes under ``base_prefix``.

    Runs are stored as ``<root>/<category>/<run_id>/...`` (e.g.
    ``checkpoints/sim2real-b/...``, ``checkpoints/physical-ai-data-factory/...``).
    This enumerates the ``<category>`` folders dynamically from S3 so discovery
    never hardcodes specific workflow paths. Returns category prefixes WITHOUT a
    trailing slash. If the root has no sub-folders, returns ``[]``.
    """
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    if max_results is not None and int(max_results) <= 0:
        return []
    root = _normalize_prefix(base_prefix)
    categories: list[str] = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=root, Delimiter="/"):
            for common in page.get("CommonPrefixes", []) or []:
                pfx = str(common.get("Prefix") or "").rstrip("/")
                if pfx:
                    categories.append(pfx)
                    if max_results is not None and len(categories) >= max(
                        0, max_results
                    ):
                        return categories
    except (ClientError, BotoCoreError) as exc:
        raise ArtifactDiscoveryError(
            f"failed to list run categories under s3://{bucket}/{root}: {exc}"
        ) from exc
    return categories


def discovery_categories(
    bucket: str,
    *,
    base_prefix: str = "",
    exclude: "set[str] | None" = None,
    max_categories: int | None = None,
    s3=None,
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

    ``exclude`` drops the named prefix subtree exactly. A nested exclusion such
    as ``npa-agent/session-state`` does not hide unrelated siblings under
    ``npa-agent``.

    Prefixes are returned without a trailing slash, de-duplicated, base-first.
    """
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    base = str(base_prefix or "").strip().strip("/")
    excluded = _normalized_discovery_exclusions(exclude)
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(prefix: str) -> None:
        value = str(prefix or "").strip().strip("/")
        if not value or value in seen:
            return
        if is_infrastructure_root(value) or _is_excluded_prefix(value, excluded):
            return
        seen.add(value)
        ordered.append(value)

    category_cap = None if max_categories is None else max(0, int(max_categories))
    if category_cap == 0:
        return []
    if base:
        for category in list_run_categories(
            bucket,
            base_prefix=base,
            max_results=category_cap,
            s3=s3,
        ):
            _add(category)
    remaining = None if category_cap is None else max(0, category_cap - len(ordered))
    if remaining == 0:
        return ordered
    for category in list_run_categories(
        bucket,
        base_prefix="",
        max_results=remaining,
        s3=s3,
    ):
        # The base root's children are categories (handled above), not runs.
        normalized_category = category.strip("/")
        if base and (
            normalized_category == base or base.startswith(normalized_category + "/")
        ):
            continue
        _add(category)
    return ordered


def _path_in_non_run_tree(path: str, excluded: set[str]) -> bool:
    normalized = str(path or "").strip().strip("/").lower()
    blocked = set(_STRUCTURAL_NON_RUN_PREFIXES)
    blocked.update(excluded)
    return any(
        normalized == candidate or normalized.startswith(candidate + "/")
        for candidate in blocked
    )


def _run_identity_for_object_key(
    key: str,
    *,
    base_prefix: str,
    excluded: set[str],
) -> tuple[str, str] | None:
    """Return ``(parent_prefix, run_id)`` for one artifact object.

    Strong run-id evidence (an encoded timestamp or explicit run token) wins at
    any nesting depth. Otherwise the established artifact layout contract is
    used: ``<category>/<run>/...`` or ``<base>/<category>/<run>/...``. Selecting
    the earliest strong segment is important because stages/variants can repeat
    the run id deeper in the path.
    """
    normalized = str(key or "").strip().strip("/")
    if not normalized or _path_in_non_run_tree(normalized, excluded):
        return None
    parts = [part for part in normalized.split("/") if part]
    directories = parts[:-1]
    if not directories:
        return None
    for index, segment in enumerate(directories):
        candidate_path = "/".join(directories[: index + 1])
        if _path_in_non_run_tree(candidate_path, excluded):
            return None
        if _looks_like_flat_run_prefix(segment):
            return "/".join(directories[:index]), segment

    base_parts = [
        part for part in str(base_prefix or "").strip().strip("/").split("/") if part
    ]
    if base_parts and directories[: len(base_parts)] == base_parts:
        run_index = len(base_parts) + 1
    else:
        run_index = 1
    if run_index >= len(directories):
        return None
    parent = "/".join(directories[:run_index])
    run_id = directories[run_index]
    if not run_id or is_infrastructure_root(run_id):
        return None
    return parent, run_id


def _run_index_observation(
    item: dict[str, Any], *, base_prefix: str, excluded: set[str]
) -> tuple[str, str, str] | None:
    key = str(item.get("Key") or "")
    identity = _run_identity_for_object_key(
        key,
        base_prefix=base_prefix,
        excluded=excluded,
    )
    if identity is None:
        return None
    parent, untrusted_run_id = identity
    try:
        run_id = _validate_run_basename(untrusted_run_id)
    except ArtifactDiscoveryError:
        return None
    discovered_path = "/".join(part for part in (parent, run_id) if part)
    if _is_excluded_prefix(discovered_path, excluded):
        return None
    return parent, run_id, key


def _new_run_index_summary() -> dict[str, Any]:
    return {
        "artifact_count": 0,
        "last_modified": "",
        "earliest": "",
        "has_viewable": False,
        "output_artifact_count": 0,
        "input_artifact_count": 0,
        "canonical_score": 0,
    }


def _update_run_index_times(summary: dict[str, Any], modified: Any) -> None:
    timestamp = _to_iso8601(modified)
    if not timestamp:
        return
    summary["last_modified"] = max(str(summary["last_modified"]), timestamp)
    if not summary["earliest"] or timestamp < str(summary["earliest"]):
        summary["earliest"] = timestamp


def _record_run_index_observation(
    summaries: dict[tuple[str, str], dict[str, Any]],
    observation: tuple[str, str, str],
    modified: Any,
) -> None:
    parent, run_id, key = observation
    summary = summaries.setdefault((parent, run_id), _new_run_index_summary())
    summary["artifact_count"] = int(summary["artifact_count"]) + 1
    scope = "/".join(part for part in (parent, run_id) if part)
    relative_key = key[len(scope) :].lstrip("/") if key.startswith(scope) else key
    role = artifact_role_for_relative_key(relative_key)
    count_key = "input_artifact_count" if role == "input" else "output_artifact_count"
    summary[count_key] = int(summary[count_key]) + 1
    summary["canonical_score"] = int(
        summary["canonical_score"]
    ) + _artifact_output_signal_score(relative_key)
    summary["has_viewable"] = bool(
        summary["has_viewable"] or render_hint_for_object(key=key) != "download"
    )
    _update_run_index_times(summary, modified)


def _scan_run_index_page(
    contents: list[dict[str, Any]],
    *,
    object_count: int,
    summaries: dict[tuple[str, str], dict[str, Any]],
    base_prefix: str,
    excluded: set[str],
) -> tuple[int, bool]:
    for item in contents:
        object_count += 1
        if object_count > MAX_RUN_DISCOVERY_OBJECTS:
            return object_count, False
        observation = _run_index_observation(
            item,
            base_prefix=base_prefix,
            excluded=excluded,
        )
        if observation is not None:
            _record_run_index_observation(
                summaries,
                observation,
                item.get("LastModified"),
            )
    return object_count, True


def _scan_artifact_run_index(
    bucket: str, *, base_prefix: str, excluded: set[str], s3
) -> tuple[dict[tuple[str, str], dict[str, Any]], bool]:
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    object_count = 0
    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix="")
        for page_count, page in enumerate(pages, start=1):
            if page_count > MAX_RUN_DISCOVERY_PAGES:
                return summaries, False
            contents = page.get("Contents", []) or []
            object_count, complete = _scan_run_index_page(
                contents,
                object_count=object_count,
                summaries=summaries,
                base_prefix=base_prefix,
                excluded=excluded,
            )
            if not complete:
                return summaries, False
    except (ClientError, BotoCoreError) as exc:
        raise ArtifactDiscoveryError(
            f"failed to build run index for s3://{bucket}: {exc}"
        ) from exc
    return summaries, True


def _run_index_summary(
    bucket: str,
    identity: tuple[str, str],
    payload: dict[str, Any],
    *,
    complete: bool,
) -> RunSummary:
    parent, run_id = identity
    return RunSummary(
        run_id=run_id,
        last_modified=str(payload["last_modified"]),
        started_at=_run_started_at(run_id, str(payload["earliest"])),
        artifact_count=int(payload["artifact_count"]),
        has_viewable=bool(payload["has_viewable"]),
        bucket=bucket,
        summary_complete=complete,
        resolved_prefix=parent,
        output_artifact_count=int(payload["output_artifact_count"]),
        input_artifact_count=int(payload["input_artifact_count"]),
        namespaces=(parent,),
        canonical_score=int(payload["canonical_score"]),
    )


def _artifact_run_index_page(
    bucket: str,
    summaries: dict[tuple[str, str], dict[str, Any]],
    *,
    complete: bool,
    contains: str,
    limit: int,
) -> RunListPage:
    needle = str(contains or "").strip().lower()
    runs = [
        _run_index_summary(bucket, identity, payload, complete=complete)
        for identity, payload in summaries.items()
        if not needle or needle in identity[1].lower()
    ]
    runs = _merge_staging_summaries(runs)
    runs.sort(
        key=lambda item: (
            item.last_modified,
            item.run_id.lower(),
            item.resolved_prefix,
        ),
        reverse=True,
    )
    total = len(runs)
    return RunListPage(
        runs=runs[:limit],
        truncated=total > limit or not complete,
        total_runs=total,
        limit=limit,
        discovery_complete=complete,
    )


def _list_artifact_run_index(
    bucket: str,
    *,
    base_prefix: str = "",
    limit: int = 50,
    exclude: "set[str] | None" = None,
    contains: str = "",
    s3=None,
) -> RunListPage:
    """Build a deterministic artifact-first run index from bounded S3 pages."""
    if limit <= 0:
        raise ArtifactDiscoveryError("limit must be > 0")
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    excluded = _normalized_discovery_exclusions(exclude)
    summaries, complete = _scan_artifact_run_index(
        bucket,
        base_prefix=base_prefix,
        excluded=excluded,
        s3=s3,
    )
    return _artifact_run_index_page(
        bucket,
        summaries,
        complete=complete,
        contains=contains,
        limit=limit,
    )


def list_all_runs(
    bucket: str,
    *,
    base_prefix: str = "",
    limit: int = 50,
    exclude: "set[str] | None" = None,
    contains: str = "",
    s3=None,
) -> RunListPage:
    """Discover artifact-backed runs across every bounded native S3 page."""
    return _list_artifact_run_index(
        bucket,
        base_prefix=base_prefix,
        limit=limit,
        exclude=exclude,
        contains=contains,
        s3=s3,
    )


def _direct_run_candidates(
    bucket: str,
    *,
    parent: str,
    limit: int,
    contains: str,
    excluded: set[str],
    s3,
) -> tuple[list[tuple[str, str]], bool]:
    candidates = list_run_categories(
        bucket,
        base_prefix=parent,
        max_results=limit + 1,
        s3=s3,
    )
    children: list[tuple[str, str]] = []
    needle = str(contains or "").strip().lower()
    for child in candidates[:limit]:
        try:
            run_id = _validate_run_basename(child.rsplit("/", 1)[-1].strip())
        except ArtifactDiscoveryError:
            continue
        if is_infrastructure_root(run_id) or _is_excluded_prefix(child, excluded):
            continue
        if needle and needle not in run_id.lower():
            continue
        children.append((child, run_id))
    return children, len(candidates) > limit


def _lightweight_run_observations(
    bucket: str, child: str, *, s3
) -> tuple[str, int, bool, bool]:
    last_modified = ""
    observed_count = 0
    has_viewable = False
    complete = False
    try:
        page = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=_normalize_prefix(child),
            MaxKeys=LIGHTWEIGHT_RUN_SUMMARY_PAGE_SIZE,
        )
        contents = page.get("Contents", []) if isinstance(page, dict) else []
        observed_count = len(contents)
        complete = not bool(page.get("IsTruncated"))
        for artifact in contents:
            if not isinstance(artifact, dict):
                continue
            timestamp = _to_iso8601(artifact.get("LastModified"))
            if timestamp > last_modified:
                last_modified = timestamp
            key = str(artifact.get("Key") or "")
            if key and render_hint_for_object(key=key) != "download":
                has_viewable = True
    except (ClientError, BotoCoreError):
        pass
    return last_modified, observed_count, has_viewable, complete


def _summarize_direct_run_prefix(
    item: tuple[str, str], *, bucket: str, parent: str, s3
) -> RunSummary:
    child, run_id = item
    last_modified, observed_count, has_viewable, complete = (
        _lightweight_run_observations(bucket, child, s3=s3)
    )
    started_at = _run_started_at(run_id, "")
    return RunSummary(
        run_id=run_id,
        last_modified=last_modified or started_at,
        started_at=started_at,
        artifact_count=observed_count if complete else 0,
        has_viewable=True if has_viewable else False if complete else None,
        bucket=bucket,
        resolved_prefix=parent,
        summary_complete=complete,
    )


def _summarize_direct_run_prefixes(
    children: list[tuple[str, str]],
    summarize: "Callable[[tuple[str, str]], RunSummary]",
) -> list[RunSummary]:
    worker_count = min(len(children), RUN_DISCOVERY_MAX_WORKERS)
    if worker_count <= 1:
        return [summarize(children[0])] if children else []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        return list(pool.map(summarize, children))


def _list_direct_run_prefixes(
    bucket: str,
    *,
    prefix: str,
    limit: int,
    contains: str,
    exclude: "set[str] | None",
    s3,
) -> RunListPage:
    if limit <= 0:
        raise ArtifactDiscoveryError("limit must be > 0")
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    parent = str(prefix or "").strip().strip("/")
    children, source_truncated = _direct_run_candidates(
        bucket,
        parent=parent,
        limit=limit,
        contains=contains,
        excluded=_normalized_discovery_exclusions(exclude),
        s3=s3,
    )
    summarize = partial(
        _summarize_direct_run_prefix, bucket=bucket, parent=parent, s3=s3
    )
    summaries = _summarize_direct_run_prefixes(children, summarize)
    summaries.sort(
        key=lambda item: (item.last_modified, item.started_at, item.run_id),
        reverse=True,
    )
    total = len(summaries)
    return RunListPage(
        runs=summaries[:limit],
        truncated=source_truncated or total > limit,
        total_runs=total,
        limit=limit,
        discovery_complete=not source_truncated,
    )


def list_run_prefixes(
    bucket: str,
    *,
    prefix: str = "",
    limit: int = 50,
    contains: str = "",
    exclude: "set[str] | None" = None,
    s3=None,
) -> RunListPage:
    """Discover immediate run directories with bounded S3 summary probes.

    Args:
        bucket: Authorized object-storage bucket to inspect.
        prefix: Exact directory directly above run ids.
        limit: Maximum number of candidate runs to summarize.
        contains: Optional case-insensitive run-id substring.
        exclude: Structural prefix subtrees that discovery must skip.
        s3: Authorized S3-compatible client used for object listing.

    Returns:
        A bounded run page with explicit completeness and truncation state.

    Raises:
        ArtifactDiscoveryError: If inputs are invalid or S3 discovery fails.
    """
    return _list_direct_run_prefixes(
        bucket,
        prefix=prefix,
        limit=limit,
        contains=contains,
        exclude=exclude,
        s3=s3,
    )


def list_all_run_prefixes(
    bucket: str,
    *,
    base_prefix: str = "",
    limit: int = 50,
    exclude: "set[str] | None" = None,
    contains: str = "",
    s3=None,
) -> RunListPage:
    """Discover artifact-backed run prefixes with complete bounded summaries."""
    return _list_artifact_run_index(
        bucket,
        base_prefix=base_prefix,
        limit=limit,
        exclude=exclude,
        contains=contains,
        s3=s3,
    )


def _run_source_identity(run: RunSummary) -> tuple[str, str, str, str]:
    return (run.project_id, run.bucket, run.resolved_prefix, run.run_id)


def _preferred_run_summary(first: RunSummary, second: RunSummary) -> RunSummary:
    return max(
        (first, second),
        key=lambda item: (
            bool(item.summary_complete),
            item.artifact_count,
            item.canonical_score,
            item.last_modified,
        ),
    )


def _merged_run_viewability(first: RunSummary, second: RunSummary) -> bool | None:
    if first.has_viewable is True or second.has_viewable is True:
        return True
    summaries = (first, second)
    if any(item.summary_complete and item.has_viewable is False for item in summaries):
        return False
    if first.has_viewable is None or second.has_viewable is None:
        return None
    return False


def _merge_same_run_summary(first: RunSummary, second: RunSummary) -> RunSummary:
    """Merge repeated observations of one exact source without double-counting."""
    if _run_source_identity(first) != _run_source_identity(second):
        raise ArtifactDiscoveryError("cannot merge different artifact run sources")
    preferred = _preferred_run_summary(first, second)
    started = [value for value in (first.started_at, second.started_at) if value]
    return dataclass_replace(
        preferred,
        last_modified=max(first.last_modified, second.last_modified),
        started_at=min(started) if started else "",
        artifact_count=max(first.artifact_count, second.artifact_count),
        has_viewable=_merged_run_viewability(first, second),
        summary_complete=first.summary_complete or second.summary_complete,
        output_artifact_count=max(
            first.output_artifact_count, second.output_artifact_count
        ),
        input_artifact_count=max(
            first.input_artifact_count, second.input_artifact_count
        ),
        metadata_artifact_count=max(
            first.metadata_artifact_count, second.metadata_artifact_count
        ),
        namespaces=tuple(dict.fromkeys((*first.namespaces, *second.namespaces))),
        canonical_score=max(first.canonical_score, second.canonical_score),
    )


def _run_summary_sort_key(run: RunSummary) -> tuple[Any, ...]:
    return (
        run.started_at or run.last_modified,
        run.run_id.lower(),
        run.canonical_score,
        run.artifact_count,
        run.last_modified,
        run.project_id,
        run.bucket,
        run.resolved_prefix,
    )


def _merge_run_page_summaries(pages: list[RunListPage]) -> list[RunSummary]:
    merged: dict[tuple[str, str, str, str], RunSummary] = {}
    for page in pages:
        for run in page.runs:
            identity = _run_source_identity(run)
            existing = merged.get(identity)
            merged[identity] = (
                _merge_same_run_summary(existing, run) if existing is not None else run
            )
    return sorted(merged.values(), key=_run_summary_sort_key, reverse=True)


def _merge_run_page_errors(pages: list[RunListPage]) -> tuple[dict[str, str], ...]:
    source_errors: list[dict[str, str]] = []
    seen_errors: set[tuple[tuple[str, str], ...]] = set()
    for page in pages:
        for error in page.source_errors:
            normalized = {str(key): str(value) for key, value in error.items()}
            identity = tuple(sorted(normalized.items()))
            if identity in seen_errors:
                continue
            seen_errors.add(identity)
            source_errors.append(normalized)
    return tuple(source_errors)


def merge_run_list_pages(
    pages: "list[RunListPage] | tuple[RunListPage, ...]",
    *,
    limit: int,
) -> RunListPage:
    """Merge bounded pages by complete source identity.

    When every input is exhaustive, ``total_runs`` is an exact de-duplicated
    total.  Otherwise it is only the number of distinct rows actually observed;
    ``truncated``/``discovery_complete`` keep callers from presenting that count
    as a global total.

    Args:
        pages: Source pages whose rows and errors should be combined.
        limit: Maximum number of sorted rows to return.

    Returns:
        A de-duplicated page preserving incomplete and truncation semantics.

    Raises:
        ArtifactDiscoveryError: If ``limit`` is not positive.
    """
    if limit <= 0:
        raise ArtifactDiscoveryError("limit must be > 0")
    page_list = list(pages)
    ordered = _merge_run_page_summaries(page_list)
    total = len(ordered)
    incomplete = any(
        page.truncated or not page.discovery_complete for page in page_list
    )
    return RunListPage(
        runs=ordered[:limit],
        truncated=incomplete or total > limit,
        total_runs=total,
        limit=limit,
        discovery_complete=all(page.discovery_complete for page in page_list),
        source_errors=_merge_run_page_errors(page_list),
    )


# --- Multi-bucket discovery ---------------------------------------------------
# The agent may be configured with several buckets. Discovery spans only the
# primary and explicitly configured extras; it never enumerates unrelated buckets
# merely because the credentials happen to be able to see them.


def _unique_artifact_sources(
    sources: "list[ArtifactSource] | tuple[ArtifactSource, ...]",
) -> list[ArtifactSource]:
    unique: list[ArtifactSource] = []
    seen: set[tuple[str, str, str]] = set()
    for source in sources:
        if not isinstance(source, ArtifactSource):
            raise ArtifactDiscoveryError(
                "direct artifact discovery requires ArtifactSource values"
            )
        if source.identity in seen:
            continue
        seen.add(source.identity)
        unique.append(source)
    return unique


def _source_discovery_failure(
    source: ArtifactSource, *, limit: int, code: str, message: str
) -> RunListPage:
    error = {
        "project_id": source.project_id,
        "bucket": source.bucket,
        "resolved_prefix": source.resolved_prefix,
        "code": code,
        "message": message,
    }
    return RunListPage(
        runs=[],
        truncated=True,
        total_runs=0,
        limit=limit,
        discovery_complete=False,
        source_errors=(error,),
    )


def _qualify_source_run_page(page: RunListPage, source: ArtifactSource) -> RunListPage:
    runs = [
        dataclass_replace(
            run,
            project_id=source.project_id,
            bucket=source.bucket,
            resolved_prefix=source.resolved_prefix,
            namespaces=(source.resolved_prefix,),
        )
        for run in page.runs
    ]
    return dataclass_replace(page, runs=runs)


def _discover_runs_for_source(
    source: ArtifactSource,
    *,
    limit: int,
    contains: str,
    exclude: "set[str] | None",
    excluded: set[str],
    s3,
) -> RunListPage:
    if _path_in_non_run_tree(source.resolved_prefix, excluded):
        return _source_discovery_failure(
            source,
            limit=limit,
            code="artifact_source_not_searchable",
            message="The configured artifact source is not a searchable run parent.",
        )
    try:
        page = list_run_prefixes(
            source.bucket,
            prefix=source.resolved_prefix,
            limit=limit,
            contains=contains,
            exclude=exclude,
            s3=s3,
        )
    except (ArtifactDiscoveryError, ClientError, BotoCoreError):
        return _source_discovery_failure(
            source,
            limit=limit,
            code="artifact_discovery_unavailable",
            message="Run discovery is unavailable for this artifact source.",
        )
    return _qualify_source_run_page(page, source)


def _discover_source_run_pages(
    sources: list[ArtifactSource], discover: "Callable[[ArtifactSource], RunListPage]"
) -> list[RunListPage]:
    worker_count = min(len(sources), RUN_DISCOVERY_MAX_WORKERS)
    if worker_count <= 1:
        return [discover(sources[0])]
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        return list(pool.map(discover, sources))


def _list_direct_source_runs(
    sources: "list[ArtifactSource] | tuple[ArtifactSource, ...]",
    *,
    limit: int,
    contains: str,
    exclude: "set[str] | None",
    s3,
) -> RunListPage:
    if limit <= 0:
        raise ArtifactDiscoveryError("limit must be > 0")
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    unique_sources = _unique_artifact_sources(sources)
    if not unique_sources:
        return RunListPage(runs=[], truncated=False, total_runs=0, limit=limit)
    discover = partial(
        _discover_runs_for_source,
        limit=limit,
        contains=contains,
        exclude=exclude,
        excluded=_normalized_discovery_exclusions(exclude),
        s3=s3,
    )
    pages = _discover_source_run_pages(unique_sources, discover)
    merged = merge_run_list_pages(pages, limit=limit)
    observed_total = sum(page.total_runs for page in pages)
    return dataclass_replace(
        merged,
        total_runs=observed_total,
        truncated=merged.truncated or observed_total > limit,
    )


def list_runs_across_sources(
    sources: "list[ArtifactSource] | tuple[ArtifactSource, ...]",
    *,
    limit: int = 50,
    contains: str = "",
    exclude: "set[str] | None" = None,
    s3=None,
) -> RunListPage:
    """Discover immediate runs below authorized direct-parent sources.

    Unlike generic bucket discovery, an ``ArtifactSource`` explicitly says its
    prefix is already the directory above run ids.  That makes timestamp-less
    ids discoverable without guessing whether the prefix is a category
    container.  Every returned row retains the full project/bucket/prefix tuple.

    Args:
        sources: Authorized source tuples whose prefixes are direct run parents.
        limit: Maximum rows returned after source pages are merged.
        contains: Optional case-insensitive run-id substring.
        exclude: Structural prefix subtrees that discovery must skip.
        s3: Authorized S3-compatible client used for object listing.

    Returns:
        A merged page that preserves source tuples and incomplete-source errors.

    Raises:
        ArtifactDiscoveryError: If inputs are invalid or no S3 client is supplied.
    """
    return _list_direct_source_runs(
        sources,
        limit=limit,
        contains=contains,
        exclude=exclude,
        s3=s3,
    )


def list_accessible_buckets(
    s3,
    *,
    primary: str = "",
    extra: "list[str] | tuple[str, ...] | None" = None,
    exclude: "set[str] | None" = None,
) -> list[str]:
    """Return the bucket names the agent should search, primary-first.

    Order: the configured primary bucket, then explicitly configured extras.
    ``exclude`` drops names that must never be scanned. The ``s3`` argument is
    retained for API compatibility but no ListBuckets request is made.
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
    return ordered


@dataclass(frozen=True)
class _BucketRunQuery:
    base_prefix: str
    prefix: str
    limit: int
    exclude: set[str] | None
    contains: str
    bucket_projects: dict[str, str] | None
    lightweight: bool


@dataclass(frozen=True)
class _BucketRunResult:
    runs: list[RunSummary]
    total: int
    complete: bool
    truncated: bool
    source_error: dict[str, str] | None


def _bucket_run_page(
    bucket: str, query: _BucketRunQuery, *, all_categories: bool, s3
) -> RunListPage:
    if all_categories:
        discover = list_all_run_prefixes if query.lightweight else list_all_runs
        return discover(
            bucket,
            base_prefix=query.base_prefix,
            limit=query.limit,
            exclude=query.exclude,
            contains=query.contains,
            s3=s3,
        )
    discover = list_run_prefixes if query.lightweight else list_runs
    return discover(
        bucket,
        prefix=query.prefix,
        limit=query.limit,
        contains=query.contains,
        s3=s3,
    )


def _bucket_discovery_error(
    bucket: str, projects: "dict[str, str] | None"
) -> dict[str, str]:
    return {
        "bucket": bucket,
        "project_id": str((projects or {}).get(bucket) or ""),
        "code": "artifact_discovery_unavailable",
        "message": "Run discovery is unavailable for this object storage resource.",
    }


def _qualify_bucket_run(
    run: RunSummary,
    *,
    bucket: str,
    project_id: str,
    resolved_prefix: str | None,
) -> RunSummary:
    changes = {"bucket": bucket, "project_id": project_id}
    if resolved_prefix is not None:
        changes["resolved_prefix"] = resolved_prefix
    return dataclass_replace(run, **changes)


def _discover_bucket_runs(
    bucket: str, *, query: _BucketRunQuery, all_categories: bool, s3
) -> _BucketRunResult:
    try:
        page = _bucket_run_page(
            bucket,
            query,
            all_categories=all_categories,
            s3=s3,
        )
    except (ArtifactDiscoveryError, ClientError, BotoCoreError):
        return _BucketRunResult(
            [],
            0,
            False,
            True,
            _bucket_discovery_error(bucket, query.bucket_projects),
        )
    resolved_prefix = (
        None if all_categories else str(query.prefix or "").strip().strip("/")
    )
    project_id = str((query.bucket_projects or {}).get(bucket) or "")
    runs = [
        _qualify_bucket_run(
            run,
            bucket=bucket,
            project_id=project_id,
            resolved_prefix=resolved_prefix,
        )
        for run in page.runs
    ]
    return _BucketRunResult(
        runs, page.total_runs, page.discovery_complete, page.truncated, None
    )


def _discover_bucket_run_pages(
    buckets: list[str], discover: "Callable[[str], _BucketRunResult]"
) -> list[_BucketRunResult]:
    worker_count = min(len(buckets), RUN_DISCOVERY_MAX_WORKERS)
    if worker_count <= 1:
        return [discover(buckets[0])]
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        return list(pool.map(discover, buckets))


def _bucket_run_sort_key(run: RunSummary, *, all_categories: bool) -> tuple[Any, ...]:
    if all_categories:
        return _run_summary_sort_key(run)
    return (
        run.started_at or run.last_modified,
        run.run_id,
        run.canonical_score,
        run.artifact_count,
        run.last_modified,
    )


def _merge_bucket_run_results(
    results: list[_BucketRunResult],
    *,
    limit: int,
    all_categories: bool,
) -> RunListPage:
    merged: dict[tuple[str, str, str, str], RunSummary] = {}
    for result in results:
        for run in result.runs:
            merged[_run_source_identity(run)] = run
    ordered = sorted(
        merged.values(),
        key=partial(_bucket_run_sort_key, all_categories=all_categories),
        reverse=True,
    )
    total = sum(int(result.total or 0) for result in results)
    complete = all(result.complete for result in results)
    any_truncated = any(result.truncated for result in results)
    observed_over_limit = total > limit if all_categories else len(ordered) > limit
    return RunListPage(
        runs=ordered[:limit],
        truncated=any_truncated or observed_over_limit or not complete,
        total_runs=total,
        limit=limit,
        discovery_complete=complete,
        source_errors=tuple(
            result.source_error for result in results if result.source_error is not None
        ),
    )


def _list_bucket_runs(
    buckets: "list[str] | tuple[str, ...]",
    *,
    query: _BucketRunQuery,
    all_categories: bool,
    s3,
) -> RunListPage:
    if query.limit <= 0:
        raise ArtifactDiscoveryError("limit must be > 0")
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    bucket_list = [str(bucket).strip() for bucket in buckets if str(bucket).strip()]
    if not bucket_list:
        return RunListPage(runs=[], truncated=False, total_runs=0, limit=query.limit)
    discover = partial(
        _discover_bucket_runs,
        query=query,
        all_categories=all_categories,
        s3=s3,
    )
    results = _discover_bucket_run_pages(bucket_list, discover)
    return _merge_bucket_run_results(
        results,
        limit=query.limit,
        all_categories=all_categories,
    )


def list_all_runs_across_buckets(
    buckets: "list[str] | tuple[str, ...]",
    *,
    base_prefix: str = "",
    limit: int = 50,
    exclude: "set[str] | None" = None,
    contains: str = "",
    bucket_projects: "dict[str, str] | None" = None,
    lightweight: bool = False,
    s3=None,
) -> RunListPage:
    """Discover runs across every accessible bucket, latest-first.

    Args:
        buckets: Authorized buckets to search.
        base_prefix: Optional storage root used to interpret generic layouts.
        limit: Maximum number of merged rows to return.
        exclude: Structural prefix subtrees that discovery must skip.
        contains: Optional case-insensitive run-id substring.
        bucket_projects: Owning project id for each bucket.
        lightweight: Whether to use direct-parent summary probes.
        s3: Authorized S3-compatible client used for object listing.

    Returns:
        A source-qualified page spanning every supplied bucket.

    Raises:
        ArtifactDiscoveryError: If inputs are invalid or no client is supplied.
    """
    query = _BucketRunQuery(
        base_prefix, "", limit, exclude, contains, bucket_projects, lightweight
    )
    return _list_bucket_runs(
        buckets,
        query=query,
        all_categories=True,
        s3=s3,
    )


def list_runs_at_prefix_across_buckets(
    buckets: "list[str] | tuple[str, ...]",
    *,
    prefix: str,
    limit: int = 50,
    contains: str = "",
    bucket_projects: "dict[str, str] | None" = None,
    lightweight: bool = False,
    s3=None,
) -> RunListPage:
    """List one run-parent prefix across every accessible bucket.

    Args:
        buckets: Authorized buckets to search.
        prefix: Exact directory directly above run ids.
        limit: Maximum number of merged rows to return.
        contains: Optional case-insensitive run-id substring.
        bucket_projects: Owning project id for each bucket.
        lightweight: Whether to use direct-parent summary probes.
        s3: Authorized S3-compatible client used for object listing.

    Returns:
        A source-qualified page for the prefix across supplied buckets.

    Raises:
        ArtifactDiscoveryError: If inputs are invalid or no client is supplied.
    """
    query = _BucketRunQuery(
        "", prefix, limit, None, contains, bucket_projects, lightweight
    )
    return _list_bucket_runs(
        buckets,
        query=query,
        all_categories=False,
        s3=s3,
    )


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


def _multi_run_cache_key(
    buckets: tuple[str, ...], query: _BucketRunQuery, *, s3
) -> tuple:
    return (
        "__multi__",
        _s3_cache_identity(s3),
        buckets,
        query.base_prefix,
        query.prefix,
        int(query.limit),
        tuple(sorted(query.exclude or ())),
        str(query.contains or ""),
        tuple(sorted((query.bucket_projects or {}).items())),
        bool(query.lightweight),
    )


def _discover_multi_bucket_run_page(
    buckets: tuple[str, ...], query: _BucketRunQuery, *, s3
) -> RunListPage:
    if query.prefix:
        return list_runs_at_prefix_across_buckets(
            list(buckets),
            prefix=query.prefix,
            limit=query.limit,
            contains=query.contains,
            bucket_projects=query.bucket_projects,
            lightweight=query.lightweight,
            s3=s3,
        )
    return list_all_runs_across_buckets(
        list(buckets),
        base_prefix=query.base_prefix,
        limit=query.limit,
        exclude=query.exclude,
        contains=query.contains,
        bucket_projects=query.bucket_projects,
        lightweight=query.lightweight,
        s3=s3,
    )


def _cached_multi_bucket_run_page(
    buckets: "list[str] | tuple[str, ...]",
    query: _BucketRunQuery,
    s3,
    ttl: "float | None",
    refresh_sync: bool,
) -> RunListPage:
    effective_ttl = DEFAULT_RUN_LIST_TTL if ttl is None else ttl
    bucket_list = tuple(
        str(bucket).strip() for bucket in buckets if str(bucket).strip()
    )
    compute = partial(_discover_multi_bucket_run_page, bucket_list, query, s3=s3)
    return _cached_run_list_page(
        _multi_run_cache_key(bucket_list, query, s3=s3),
        compute,
        ttl=float(effective_ttl),
        refresh_sync=refresh_sync,
    )


def list_runs_cached_multi(
    buckets: "list[str] | tuple[str, ...]",
    *,
    base_prefix: str = "",
    prefix: str = "",
    limit: int = 50,
    exclude: "set[str] | None" = None,
    contains: str = "",
    bucket_projects: "dict[str, str] | None" = None,
    lightweight: bool = False,
    s3=None,
    ttl: "float | None" = None,
    refresh_sync: bool = False,
) -> RunListPage:
    """Cache generic or explicit-prefix discovery across accessible buckets.
    Args:
        buckets: Authorized buckets to search.
        base_prefix: Optional storage root for generic discovery.
        prefix: Exact run-parent prefix, when one is selected.
        limit: Maximum number of merged rows to return.
        exclude: Structural prefix subtrees that discovery must skip.
        contains: Optional case-insensitive run-id substring.
        bucket_projects: Owning project id for each bucket.
        lightweight: Whether to use direct-parent summary probes.
        s3: Authorized S3-compatible client used for object listing.
        ttl: Cache lifetime in seconds, or the default when omitted.
        refresh_sync: Whether stale data must be refreshed before returning.
    Returns:
        The credential-scoped cached or freshly discovered run page.
    Raises:
        ArtifactDiscoveryError: If inputs are invalid or discovery fails.
    """
    query = _BucketRunQuery(
        base_prefix, prefix, limit, exclude, contains, bucket_projects, lightweight
    )
    return _cached_multi_bucket_run_page(buckets, query, s3, ttl, refresh_sync)


def _source_run_cache_key(
    sources: tuple[ArtifactSource, ...],
    *,
    s3,
    limit: int,
    exclude: "set[str] | None",
    contains: str,
) -> tuple:
    return (
        "__sources__",
        _s3_cache_identity(s3),
        tuple(sorted(source.identity for source in sources)),
        int(limit),
        tuple(sorted(exclude or ())),
        str(contains or ""),
    )


def _cached_direct_source_runs(
    sources: "list[ArtifactSource] | tuple[ArtifactSource, ...]",
    *,
    limit: int,
    contains: str,
    exclude: "set[str] | None",
    s3,
    ttl: "float | None",
    refresh_sync: bool,
) -> RunListPage:
    effective_ttl = DEFAULT_RUN_LIST_TTL if ttl is None else ttl
    source_list = tuple(sources)
    if any(not isinstance(source, ArtifactSource) for source in source_list):
        raise ArtifactDiscoveryError(
            "direct artifact discovery requires ArtifactSource values"
        )
    key = _source_run_cache_key(
        source_list,
        s3=s3,
        limit=limit,
        exclude=exclude,
        contains=contains,
    )
    compute = partial(
        list_runs_across_sources,
        source_list,
        limit=limit,
        contains=contains,
        exclude=exclude,
        s3=s3,
    )
    return _cached_run_list_page(
        key,
        compute,
        ttl=float(effective_ttl),
        refresh_sync=refresh_sync,
    )


def list_runs_cached_sources(
    sources: "list[ArtifactSource] | tuple[ArtifactSource, ...]",
    *,
    limit: int = 50,
    contains: str = "",
    exclude: "set[str] | None" = None,
    s3=None,
    ttl: "float | None" = None,
    refresh_sync: bool = False,
) -> RunListPage:
    """Cache discovery for explicit direct-parent source tuples.

    Args:
        sources: Authorized source tuples whose prefixes are direct run parents.
        limit: Maximum number of merged rows to return.
        contains: Optional case-insensitive run-id substring.
        exclude: Structural prefix subtrees that discovery must skip.
        s3: Authorized S3-compatible client used for object listing.
        ttl: Cache lifetime in seconds, or the default when omitted.
        refresh_sync: Whether stale data must be refreshed before returning.

    Returns:
        The credential-scoped cached or freshly discovered source page.

    Raises:
        ArtifactDiscoveryError: If a source value is invalid or discovery fails.
    """
    return _cached_direct_source_runs(
        sources,
        limit=limit,
        contains=contains,
        exclude=exclude,
        s3=s3,
        ttl=ttl,
        refresh_sync=refresh_sync,
    )


def _find_run_matches_across_buckets(
    buckets: "list[str] | tuple[str, ...]",
    *,
    base_prefix: str,
    run_id: str,
    s3,
) -> tuple[list[RunResolution], bool]:
    matches: list[RunResolution] = []
    incomplete = False
    for bucket in buckets:
        name = str(bucket).strip()
        if not name:
            continue
        try:
            found = find_run_artifact_matches(
                name,
                base_prefix=base_prefix,
                run_id=run_id,
                s3=s3,
            )
        except (ArtifactDiscoveryError, ClientError, BotoCoreError):
            incomplete = True
            continue
        matches.extend(found)
    return matches, incomplete


def _unique_run_resolution(
    run_id: str, matches: list[RunResolution]
) -> RunResolution | None:
    unique = _merge_staging_resolutions(matches)
    if not unique:
        return None
    if len(unique) > 1:
        raise AmbiguousRunError(run_id, [item.run_ref for item in unique])
    return unique[0]


def find_run_artifacts_across_buckets(
    buckets: "list[str] | tuple[str, ...]",
    *,
    base_prefix: str = "",
    run_id: str,
    s3=None,
) -> "tuple[str, list[Artifact]]":
    """Locate a unique run across configured buckets or fail on ambiguity.
    Args:
        buckets: Authorized buckets to search.
        base_prefix: Optional storage root used to interpret generic layouts.
        run_id: Validated run basename to resolve.
        s3: Authorized S3-compatible client used for object listing.
    Returns:
        The unique bucket and artifact list, or an empty pair when absent.
    Raises:
        ArtifactDiscoveryError: If discovery is invalid or incomplete.
        AmbiguousRunError: If the basename matches multiple source tuples.
    """
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    matches, incomplete = _find_run_matches_across_buckets(
        buckets,
        base_prefix=base_prefix,
        run_id=run_id,
        s3=s3,
    )
    if incomplete:
        raise ArtifactDiscoveryError(
            "run source discovery was incomplete; select an exact source"
        )
    resolution = _unique_run_resolution(run_id, matches)
    if resolution is None:
        return "", []
    return resolution.bucket, resolution.artifacts


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
_RUN_LIST_INFLIGHT: "dict[tuple, tuple[int, threading.Event]]" = {}
_RUN_LIST_LOCK = threading.Lock()
_RUN_LIST_GENERATION = 0


@dataclass(frozen=True)
class _RunListRefreshLease:
    generation: int
    event: threading.Event
    token: tuple[int, threading.Event]
    owner: bool


def _run_list_cache_clear() -> None:
    """Invalidate cached pages, including results from older in-flight work."""
    global _RUN_LIST_GENERATION
    with _RUN_LIST_LOCK:
        _RUN_LIST_GENERATION += 1
        _RUN_LIST_CACHE.clear()


def _acquire_run_list_refresh_lease(key: tuple) -> _RunListRefreshLease:
    with _RUN_LIST_LOCK:
        generation = _RUN_LIST_GENERATION
        existing = _RUN_LIST_INFLIGHT.get(key)
        if existing is not None and existing[0] == generation:
            return _RunListRefreshLease(generation, existing[1], existing, False)
        event = threading.Event()
        token = (generation, event)
        _RUN_LIST_INFLIGHT[key] = token
        return _RunListRefreshLease(generation, event, token, True)


def _release_run_list_refresh_lease(
    key: tuple, lease: _RunListRefreshLease, page: RunListPage | None
) -> None:
    with _RUN_LIST_LOCK:
        if page is not None and lease.generation == _RUN_LIST_GENERATION:
            _RUN_LIST_CACHE[key] = (time.monotonic(), page)
        if _RUN_LIST_INFLIGHT.get(key) == lease.token:
            _RUN_LIST_INFLIGHT.pop(key, None)
    lease.event.set()


def _execute_run_list_refresh(
    key: tuple,
    compute: "Callable[[], RunListPage]",
    lease: _RunListRefreshLease,
    *,
    sync: bool,
) -> None:
    page: RunListPage | None = None
    failure: Exception | None = None
    try:
        page = compute()
    except Exception as exc:  # noqa: BLE001 - async refresh degrades to stale
        failure = exc
    finally:
        _release_run_list_refresh_lease(key, lease, page)
    if failure is not None and sync:
        raise failure


def _schedule_run_list_refresh(
    key: tuple, compute: "Callable[[], RunListPage]", *, sync: bool = False
) -> "threading.Thread | None":
    """Refresh one entry, waiting for equivalent in-flight work when requested."""
    lease = _acquire_run_list_refresh_lease(key)
    if not lease.owner:
        if sync:
            lease.event.wait()
        return None
    refresh = partial(_execute_run_list_refresh, key, compute, lease, sync=sync)
    if sync:
        refresh()
        return None
    thread = threading.Thread(target=refresh, name="npa-runlist-refresh", daemon=True)
    thread.start()
    return thread


def _run_list_cache_snapshot(
    key: tuple,
) -> tuple[tuple[float, RunListPage] | None, int]:
    with _RUN_LIST_LOCK:
        return _RUN_LIST_CACHE.get(key), _RUN_LIST_GENERATION


def _store_cold_run_list_page(
    key: tuple, compute: "Callable[[], RunListPage]", generation: int
) -> RunListPage:
    page = compute()
    with _RUN_LIST_LOCK:
        if generation == _RUN_LIST_GENERATION:
            _RUN_LIST_CACHE[key] = (time.monotonic(), page)
    return page


def _newer_cached_run_list_page(
    key: tuple, previous_timestamp: float
) -> RunListPage | None:
    with _RUN_LIST_LOCK:
        refreshed = _RUN_LIST_CACHE.get(key)
    if refreshed is None or refreshed[0] <= previous_timestamp:
        return None
    return refreshed[1]


def _cached_run_list_page(
    key: tuple,
    compute: "Callable[[], RunListPage]",
    *,
    ttl: float,
    refresh_sync: bool,
) -> RunListPage:
    """Serve one cached page, with an optionally authoritative sync refresh."""
    now = time.monotonic()
    entry, generation = _run_list_cache_snapshot(key)
    if entry is None:
        return _store_cold_run_list_page(key, compute, generation)

    timestamp, stale_page = entry
    if not refresh_sync and now - timestamp < ttl:
        return stale_page

    _schedule_run_list_refresh(key, compute, sync=refresh_sync)
    if not refresh_sync:
        return stale_page

    refreshed_page = _newer_cached_run_list_page(key, timestamp)
    if refreshed_page is not None:
        return refreshed_page

    # An already-running asynchronous refresh may have failed or been invalidated
    # while this caller waited. A synchronous request must not silently return the
    # stale (notably empty) page, so retry once as the foreground owner.
    _schedule_run_list_refresh(key, compute, sync=True)
    return _newer_cached_run_list_page(key, timestamp) or stale_page


def _single_run_cache_key(
    bucket: str, query: _BucketRunQuery, *, all_categories: bool, s3
) -> tuple:
    return (
        bucket,
        _s3_cache_identity(s3),
        query.base_prefix,
        query.prefix,
        int(query.limit),
        tuple(sorted(query.exclude or ())),
        str(query.contains or ""),
        bool(all_categories),
    )


def _discover_single_bucket_run_page(
    bucket: str, query: _BucketRunQuery, *, all_categories: bool, s3
) -> RunListPage:
    if all_categories:
        return list_all_runs(
            bucket,
            base_prefix=query.base_prefix,
            limit=query.limit,
            exclude=query.exclude,
            contains=query.contains,
            s3=s3,
        )
    return list_runs(
        bucket,
        prefix=query.prefix,
        limit=query.limit,
        contains=query.contains,
        s3=s3,
    )


def _cached_single_bucket_run_page(
    bucket: str,
    query: _BucketRunQuery,
    s3,
    all_categories: bool,
    ttl: float,
    refresh_sync: bool,
) -> RunListPage:
    compute = partial(
        _discover_single_bucket_run_page,
        bucket,
        query,
        all_categories=all_categories,
        s3=s3,
    )
    key = _single_run_cache_key(
        bucket,
        query,
        all_categories=all_categories,
        s3=s3,
    )
    return _cached_run_list_page(
        key,
        compute,
        ttl=float(ttl),
        refresh_sync=refresh_sync,
    )


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
    """Cache single-bucket discovery with stale-while-revalidate behavior.
    Args:
        bucket: Authorized object-storage bucket to inspect.
        prefix: Exact run-parent prefix for single-prefix discovery.
        base_prefix: Optional storage root for all-category discovery.
        limit: Maximum number of rows to return.
        exclude: Structural prefix subtrees that discovery must skip.
        contains: Optional case-insensitive run-id substring.
        s3: Authorized S3-compatible client used for object listing.
        all_categories: Whether to discover generic layouts across the bucket.
        ttl: Cache lifetime in seconds.
        refresh_sync: Whether stale data must be refreshed before returning.
    Returns:
        The credential-scoped cached or freshly discovered run page.
    Raises:
        ArtifactDiscoveryError: If inputs are invalid or discovery fails.
    """
    query = _BucketRunQuery(base_prefix, prefix, limit, exclude, contains, None, False)
    return _cached_single_bucket_run_page(
        bucket, query, s3, all_categories, ttl, refresh_sync
    )


def _cached_server_discovered_run_summary(
    *,
    bucket: str,
    source_prefix: str,
    run_id: str,
    s3,
) -> RunSummary | None:
    """Return an exact tuple previously emitted by this credential-scoped server.

    A run reference is not itself an authorization capability. The run-list
    cache, however, contains tuples the server positively observed through its
    bounded discovery path. Reusing that observation avoids repeating a full
    bucket walk for every idempotent export while absent/caller-invented tuples
    continue through the fail-closed discovery path below.
    """
    identity = _s3_cache_identity(s3)
    with _RUN_LIST_LOCK:
        pages = [
            page
            for key, (_timestamp, page) in _RUN_LIST_CACHE.items()
            if len(key) > 1 and key[1] == identity
        ]
    for page in pages:
        for item in page.runs:
            if (
                item.bucket == bucket
                and item.resolved_prefix == source_prefix
                and item.run_id == run_id
            ):
                return item
    return None


def find_run_artifact_matches(
    bucket: str,
    *,
    base_prefix: str,
    run_id: str,
    exact_source_prefix: "str | None" = None,
    s3=None,
) -> list[RunResolution]:
    """Return every exact source match for a run basename in one bucket."""
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    normalized_run = _validate_run_basename(run_id)
    index = _list_artifact_run_index(
        bucket,
        base_prefix=base_prefix,
        limit=MAX_RUN_PARENT_CANDIDATES,
        contains=normalized_run,
        s3=s3,
    )
    candidates = [
        item
        for item in index.runs
        if item.run_id == normalized_run
        and (exact_source_prefix is None or item.resolved_prefix == exact_source_prefix)
    ]
    # An incomplete bounded scan cannot prove that an absent source does not
    # exist or that a plain run basename is unique. A positive exact-prefix
    # match is different: the server itself observed that tuple in its bounded
    # index, so it is safe to resolve without letting an unrelated high-volume
    # source disable the source-qualified selector.
    if (index.truncated or not index.discovery_complete) and (
        exact_source_prefix is None or not candidates
    ):
        raise ArtifactDiscoveryError(
            "run source discovery was incomplete; select an exact source"
        )
    return [
        RunResolution(
            run_id=normalized_run,
            bucket=bucket,
            source_prefix=item.resolved_prefix,
            artifacts=[
                artifact
                for namespace in (item.namespaces or (item.resolved_prefix,))
                for artifact in list_artifacts(
                    bucket,
                    normalized_run,
                    prefix=namespace,
                    s3=s3,
                )
            ],
        )
        for item in candidates
    ]


def _cached_summary_resolution(
    observed: RunSummary,
    *,
    bucket: str,
    source_prefix: str,
    run_id: str,
    s3,
) -> RunResolution:
    artifacts = [
        artifact
        for namespace in (observed.namespaces or (source_prefix,))
        for artifact in list_artifacts(
            bucket,
            run_id,
            prefix=namespace,
            s3=s3,
        )
    ]
    return RunResolution(
        run_id=run_id,
        bucket=bucket,
        source_prefix=source_prefix,
        artifacts=artifacts,
    )


def _resolve_exact_run_reference(
    configured: list[str], *, base_prefix: str, requested: str, s3
) -> RunResolution | None:
    bucket, source_prefix, run_id = decode_run_ref(requested)
    if bucket not in configured:
        raise ArtifactDiscoveryError("run_ref bucket is not configured for this agent")
    observed = _cached_server_discovered_run_summary(
        bucket=bucket,
        source_prefix=source_prefix,
        run_id=run_id,
        s3=s3,
    )
    if observed is not None:
        return _cached_summary_resolution(
            observed,
            bucket=bucket,
            source_prefix=source_prefix,
            run_id=run_id,
            s3=s3,
        )
    matches = find_run_artifact_matches(
        bucket,
        base_prefix=base_prefix,
        run_id=run_id,
        exact_source_prefix=source_prefix,
        s3=s3,
    )
    exact = [
        item
        for item in _merge_staging_resolutions(matches)
        if item.source_prefix == source_prefix
    ]
    return _unique_run_resolution(run_id, exact)


def _resolve_plain_run_id(
    configured: list[str], *, base_prefix: str, requested: str, s3
) -> RunResolution | None:
    run_id = _validate_run_basename(requested)
    matches, incomplete = _find_run_matches_across_buckets(
        configured,
        base_prefix=base_prefix,
        run_id=run_id,
        s3=s3,
    )
    if incomplete:
        raise ArtifactDiscoveryError(
            "run source discovery was incomplete; select a server-discovered exact source"
        )
    return _unique_run_resolution(run_id, matches)


def resolve_run_artifacts(
    buckets: "list[str] | tuple[str, ...]",
    *,
    base_prefix: str,
    run_ref_or_id: str,
    s3=None,
) -> RunResolution | None:
    """Resolve an exact run reference or a unique plain run basename.
    Args:
        buckets: Authorized buckets that may contain the run.
        base_prefix: Optional storage root used to interpret generic layouts.
        run_ref_or_id: Server-issued run reference or validated run basename.
        s3: Authorized S3-compatible client used for object listing.
    Returns:
        The unique source-qualified resolution, or ``None`` when absent.
    Raises:
        ArtifactDiscoveryError: If selection is unauthorized or incomplete.
        AmbiguousRunError: If a basename matches multiple source tuples.
    """
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    configured = [str(bucket).strip() for bucket in buckets if str(bucket).strip()]
    requested = str(run_ref_or_id or "").strip()
    if requested.startswith(_RUN_REF_PREFIX):
        return _resolve_exact_run_reference(
            configured,
            base_prefix=base_prefix,
            requested=requested,
            s3=s3,
        )
    return _resolve_plain_run_id(
        configured,
        base_prefix=base_prefix,
        requested=requested,
        s3=s3,
    )


def find_run_artifacts(
    bucket: str, *, base_prefix: str, run_id: str, s3=None
) -> "list[Artifact]":
    """Locate a run's artifacts anywhere in the bucket without a hardcoded path.

    Probes every candidate parent prefix and returns the unique match. Duplicate
    basenames fail closed rather than silently selecting the first category.

    Args:
        bucket: Authorized object-storage bucket to search.
        base_prefix: Optional storage root used to interpret generic layouts.
        run_id: Validated run basename to resolve.
        s3: Authorized S3-compatible client used for object listing.

    Returns:
        The unique run's artifacts, or an empty list when absent.

    Raises:
        ArtifactDiscoveryError: If discovery is invalid or incomplete.
        AmbiguousRunError: If the basename matches multiple source tuples.
    """
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    matches = find_run_artifact_matches(
        bucket, base_prefix=base_prefix, run_id=run_id, s3=s3
    )
    matches = _merge_staging_resolutions(matches)
    if not matches:
        return []
    if len(matches) > 1:
        raise AmbiguousRunError(run_id, [item.run_ref for item in matches])
    return matches[0].artifacts


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
    run_prefix = _normalize_prefix(_validate_run_basename(run_id))
    scope = f"{normalized_prefix}{run_prefix}"
    namespace = normalized_prefix.rstrip("/") or "<bucket-root>"
    artifacts: list[Artifact] = []
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=scope):
            for item in page.get("Contents", []) or []:
                key = str(item.get("Key") or "")
                if not key:
                    continue
                render = render_hint_for_object(key=key)
                relative_key = key[len(scope) :] if key.startswith(scope) else key
                relative_key = relative_key.lstrip("/")
                artifacts.append(
                    Artifact(
                        run_id=run_id,
                        key=key,
                        s3_uri=f"s3://{bucket}/{key}",
                        size=int(item.get("Size") or 0),
                        last_modified=_to_iso8601(item.get("LastModified")),
                        render=render,
                        inline=is_inline_render(render),
                        role=artifact_role_for_relative_key(relative_key),
                        namespace=namespace,
                        relative_key=relative_key,
                    )
                )
    except (ClientError, BotoCoreError) as exc:
        raise ArtifactDiscoveryError(
            f"failed to list artifacts under s3://{bucket}/{scope}: {exc}"
        ) from exc
    artifacts.sort(key=lambda item: (item.last_modified, item.key), reverse=True)
    return artifacts


def list_artifacts_page(
    bucket: str,
    run_id: str,
    *,
    prefix: str = "",
    cursor: str = "",
    page_size: int = 1000,
    s3=None,
) -> ArtifactListPage:
    """Return one native S3 page for a run without materializing the whole run."""
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    normalized_prefix = _normalize_prefix(prefix)
    run_prefix = _normalize_prefix(validate_run_id(run_id))
    scope = f"{normalized_prefix}{run_prefix}"
    namespace = normalized_prefix.rstrip("/") or "<bucket-root>"
    size = max(1, min(int(page_size), 1000))
    kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": scope, "MaxKeys": size}
    if str(cursor or "").strip():
        kwargs["ContinuationToken"] = str(cursor).strip()
    try:
        response = s3.list_objects_v2(**kwargs)
    except (ClientError, BotoCoreError) as exc:
        raise ArtifactDiscoveryError(
            f"failed to list artifact page under s3://{bucket}/{scope}: {exc}"
        ) from exc
    artifacts: list[Artifact] = []
    for item in response.get("Contents", []) or []:
        key = str(item.get("Key") or "")
        if not key:
            continue
        render = render_hint_for_object(key=key)
        relative_key = key[len(scope) :] if key.startswith(scope) else key
        relative_key = relative_key.lstrip("/")
        artifacts.append(
            Artifact(
                run_id=run_id,
                key=key,
                s3_uri=f"s3://{bucket}/{key}",
                size=int(item.get("Size") or 0),
                last_modified=_to_iso8601(item.get("LastModified")),
                render=render,
                inline=is_inline_render(render),
                role=artifact_role_for_relative_key(relative_key),
                namespace=namespace,
                relative_key=relative_key,
            )
        )
    artifacts.sort(key=lambda item: (item.last_modified, item.key), reverse=True)
    return ArtifactListPage(
        artifacts=artifacts,
        truncated=bool(response.get("IsTruncated")),
        next_cursor=str(response.get("NextContinuationToken") or ""),
        page_size=size,
    )


def find_run_artifact_page(
    bucket: str,
    *,
    base_prefix: str,
    run_id: str,
    page_size: int = 1000,
    s3=None,
) -> tuple[str, ArtifactListPage]:
    """Locate one unambiguous run source and return its first artifact page."""
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    index = _list_artifact_run_index(
        bucket,
        base_prefix=base_prefix,
        limit=MAX_RUN_PARENT_CANDIDATES,
        contains=run_id,
        s3=s3,
    )
    if index.truncated or not index.discovery_complete:
        raise ArtifactDiscoveryError(
            "run source discovery was incomplete; select an exact source"
        )
    candidates = sorted(
        (item for item in index.runs if item.run_id == run_id),
        key=lambda item: item.resolved_prefix,
    )
    if len(candidates) > 1:
        raise AmbiguousRunSourceError("run id is ambiguous within this bucket")
    if candidates:
        prefix = candidates[0].resolved_prefix
        return prefix, list_artifacts_page(
            bucket,
            run_id,
            prefix=prefix,
            page_size=page_size,
            s3=s3,
        )
    return "", ArtifactListPage([], False, "", page_size)


def find_run_sources_across_buckets(
    buckets: "list[str] | tuple[str, ...]",
    *,
    base_prefix: str,
    run_id: str,
    exact_prefix: "str | None" = None,
    exclude: "set[str] | None" = None,
    bucket_projects: "dict[str, str] | None" = None,
    s3=None,
) -> tuple[list[RunSummary], tuple[dict[str, str], ...], bool]:
    """Return every exact source for ``run_id`` without silently choosing one."""
    if s3 is None:
        raise ArtifactDiscoveryError("s3 client is required")
    bucket_list = [str(value).strip() for value in buckets if str(value).strip()]
    normalized_run = _validate_run_basename(run_id)
    # A source-qualified request already carries the exact directory above the
    # run id. Re-authorize that tuple with one bounded server-side prefix probe
    # instead of walking every category in the bucket again. The probe remains
    # fail-closed: the bucket came from effective access, excluded structural
    # roots are rejected, and a caller-supplied path is accepted only when S3
    # currently returns an object beneath the exact ``prefix/run_id/`` scope.
    if exact_prefix is not None and len(bucket_list) == 1:
        bucket = bucket_list[0]
        parent = _validate_source_prefix(str(exact_prefix or ""))
        excluded = _normalized_discovery_exclusions(exclude)
        if _path_in_non_run_tree(parent, excluded) or is_infrastructure_root(
            normalized_run
        ):
            return [], (), True
        scope = "/".join(part for part in (parent, normalized_run) if part) + "/"
        try:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=scope, MaxKeys=1)
        except (ArtifactDiscoveryError, ClientError, BotoCoreError):
            return (
                [],
                (
                    {
                        "bucket": bucket,
                        "project_id": str((bucket_projects or {}).get(bucket) or ""),
                        "code": "artifact_discovery_unavailable",
                        "message": "Run discovery is unavailable for this object storage resource.",
                    },
                ),
                False,
            )
        objects = [
            item
            for item in response.get("Contents", []) or []
            if str(item.get("Key") or "").startswith(scope)
        ]
        if not objects:
            return [], (), True
        first = objects[0]
        last_modified = _to_iso8601(first.get("LastModified"))
        source = RunSummary(
            run_id=normalized_run,
            last_modified=last_modified,
            artifact_count=1,
            has_viewable=None,
            bucket=bucket,
            project_id=str((bucket_projects or {}).get(bucket) or ""),
            summary_complete=False,
            started_at=_run_started_at(normalized_run, last_modified),
            resolved_prefix=parent,
        )
        return [source], (), True
    page = list_all_runs_across_buckets(
        bucket_list,
        base_prefix=base_prefix,
        limit=MAX_RUN_PARENT_CANDIDATES,
        exclude=exclude,
        contains=normalized_run,
        bucket_projects=bucket_projects,
        lightweight=True,
        s3=s3,
    )
    matches = sorted(
        (item for item in page.runs if item.run_id == normalized_run),
        key=lambda item: (item.project_id, item.bucket, item.resolved_prefix),
    )
    # A caller-selected source is an authorization boundary. If the bounded
    # candidate list was truncated, absence from ``matches`` is not proof that
    # the requested source does not exist and must not become a false 404.
    return matches, page.source_errors, page.discovery_complete and not page.truncated


def find_run_artifact_page_across_buckets(
    buckets: "list[str] | tuple[str, ...]",
    *,
    base_prefix: str,
    run_id: str,
    page_size: int = 1000,
    s3=None,
) -> tuple[str, str, ArtifactListPage]:
    """Locate a run's first artifact page across accessible buckets."""
    for bucket in buckets:
        name = str(bucket or "").strip()
        if not name:
            continue
        try:
            prefix, page = find_run_artifact_page(
                name,
                base_prefix=base_prefix,
                run_id=run_id,
                page_size=page_size,
                s3=s3,
            )
        except (ArtifactDiscoveryError, ClientError, BotoCoreError):
            continue
        if page.artifacts:
            return name, prefix, page
    return "", "", ArtifactListPage([], False, "", page_size)


def select_preferred_artifact(artifacts: list[Artifact]) -> Artifact | None:
    if not artifacts:
        return None

    def _score(item: Artifact) -> tuple[int, int, int, str, str]:
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
        role_rank = {"output": 0, "metadata": 1, "input": 2}.get(
            str(item.role or ""), 3
        )
        return (
            role_rank,
            _RENDER_ORDER.get(item.render, 99),
            specificity,
            item.last_modified,
            item.key,
        )

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
        raise ArtifactDiscoveryError(
            f"failed to download s3://{bucket}/{key}: {exc}"
        ) from exc
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
            hour, minute, second = (
                int(time_part[0:2]),
                int(time_part[2:4]),
                int(time_part[4:6]),
            )
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
                dt = datetime(
                    year, month, day, hour, minute, second, tzinfo=timezone.utc
                )
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
            gap = (
                datetime.fromisoformat(earliest) - datetime.fromisoformat(candidate)
            ).total_seconds()
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


_PIPELINE_STAGE_SEGMENTS = frozenset(
    {
        "input",
        "configs",
        "labeled_original",
        "cosmos_augmented",
        "grade",
        "labeled_augmented",
        "curation",
        "reports",
        "npa-workflow",
        "eval",
        "actions",
        "rollouts",
        "training_signal",
        "envs",
        "outer_loop",
    }
)


def infer_run_id_from_artifact_key(key: str) -> str:
    """Infer a full run id from a pipeline artifact key when possible."""

    parts = [part for part in str(key or "").strip("/").split("/") if part]
    for index, part in enumerate(parts):
        if part in _PIPELINE_STAGE_SEGMENTS and index > 0:
            return parts[index - 1]
    return ""


def resolve_run_artifact(
    artifacts: list[Artifact],
    *,
    run_id: str,
    requested_key: str,
) -> Artifact | None:
    """Resolve a full or run-relative key against artifacts actually discovered.

    This deliberately does not concatenate guessed S3 prefixes.  A run can live
    below any workflow/category prefix, so the listed object keys are the source
    of truth.  Ambiguous suffixes fail instead of loading the wrong run's data.
    """

    wanted = str(requested_key or "").strip().lstrip("/")
    if not wanted:
        return None
    exact = [item for item in artifacts if str(item.key or "").strip("/") == wanted]
    if len(exact) == 1:
        return exact[0]
    relative = [
        item
        for item in artifacts
        if _run_relative_key(str(item.key or ""), run_id).strip("/") == wanted
    ]
    if len(relative) == 1:
        return relative[0]
    if len(relative) > 1 or len(exact) > 1:
        raise ArtifactDiscoveryError(
            f"artifact key is ambiguous for run {run_id!r}: {requested_key!r}"
        )
    return None


def artifact_data_role(key: str, run_id: str = "") -> dict[str, str]:
    """Classify an artifact without making an unsupported authenticity claim."""

    rel = _run_relative_key(str(key or ""), run_id).strip("/")
    stage, _, leaf = rel.partition("/")
    lower_leaf = leaf.lower()
    if stage == "input":
        if lower_leaf.endswith("conditioning.mp4") or lower_leaf.startswith(
            "conditioning-frame-"
        ):
            return {
                "role": "derived_conditioning",
                "label": "Derived conditioning clip/frame",
                "detail": "Normalized or extracted from the run source for model conditioning; not an augmentation output.",
            }
        if lower_leaf.endswith("provenance.json"):
            return {
                "role": "input_provenance",
                "label": "Input provenance",
                "detail": "Authenticity, license, integrity, staged URI, and source-to-conditioning lineage.",
            }
        return {
            "role": "source_input",
            "label": "Source input",
            "detail": "Source supplied to this run; consult provenance for real/user/fixture classification.",
        }
    if stage in {"cosmos_augmented", "labeled_augmented"}:
        return {
            "role": "synthetic_augmented",
            "label": "Synthetic / augmented data",
            "detail": "Output derived from the run input by the augmentation pipeline.",
        }
    if stage == "labeled_original":
        return {
            "role": "conditioning_metadata",
            "label": "Input-derived labels",
            "detail": "Labels derived from conditioning frames; not source media or augmentation output.",
        }
    return {
        "role": "pipeline_metadata",
        "label": "Pipeline metadata / report",
        "detail": "Configuration, evaluation, curation, visualization, or workflow state.",
    }


def build_fiftyone_dataset(
    keys: list[str],
    *,
    run_id: str,
    read_json: Any,
    bucket: str = "",
) -> dict[str, Any]:
    """Assemble an artifact-backed dataset summary for a data-factory run.

    Pure logic (no I/O of its own): ``keys`` are the run's object keys and
    ``read_json(key)`` returns parsed JSON (or ``None``). Groups the augmented
    scenario variants (thumbnail + appearance-variable tags + augmented caption +
    video), source media, and derived conditioning data, then summarizes the grade
    and curation so the agent can clearly separate source, conditioning, and
    synthetic/augmented output. The payload also states whether real FiftyOne
    Brain curation ran; this summary is never presented as FiftyOne when it did not.
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
                entry = by_clip.setdefault(
                    clip, {"frames": [], "video": "", "meta": ""}
                )
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
    grade: dict[str, Any] = {}
    for _grade_name in (_VLM_RESULT_FILENAME, "vlm_eval.json"):
        grade = read_json(json_rel.get(f"grade/{_grade_name}", "")) or {}
        if grade:
            break
    decision = read_json(json_rel.get("grade/decision.json", "")) or {}
    curation = read_json(json_rel.get("curation/report.json", "")) or {}
    config_manifest = read_json(json_rel.get("configs/manifest.json", "")) or {}
    input_provenance = read_json(json_rel.get("input/provenance.json", "")) or {}
    aug_caps = read_json(json_rel.get("labeled_augmented/captions.json", "")) or {}
    cap_items = aug_caps.get("captions", []) if isinstance(aug_caps, dict) else []
    # Captions of the SOURCE frames from the annotate-original stage, so input
    # cards carry their VLM label too (not just the augmented variants).
    orig_caps = read_json(json_rel.get("labeled_original/captions.json", "")) or {}
    orig_cap_items = (
        orig_caps.get("captions", []) if isinstance(orig_caps, dict) else []
    )

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
    fo_samples = (
        fo_curation.get("samples", {})
        if isinstance(fo_curation.get("samples"), dict)
        else {}
    )
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

    input_source = input_provenance if isinstance(input_provenance, dict) else {}
    if not input_source:
        input_source = (
            config_manifest.get("input_source", {})
            if isinstance(config_manifest, dict)
            else {}
        )
    if not isinstance(input_source, dict):
        input_source = {}
    if not input_source:
        seeded_count = int(config_manifest.get("seeded_default_input_frames") or 0)
        input_source = {
            "source_kind": "synthetic_fixture" if seeded_count else "user_supplied",
            "input_origin_label": (
                "Synthetic seeded fixture" if seeded_count else "User-supplied input"
            ),
            "staged_canonical_s3_uri": "",
            "frame_count": seeded_count or len(input_frames),
            "description": (
                "NPA-generated seeded fixture used as this run's input"
                if seeded_count
                else "Input artifacts stored under this run's input/ prefix"
            ),
        }

    augmented_samples: list[dict[str, Any]] = []
    source_samples: list[dict[str, Any]] = []
    conditioning_samples: list[dict[str, Any]] = []
    source_kind = str(input_source.get("source_kind") or input_source.get("kind") or "")
    source_label = str(input_source.get("input_origin_label") or "")
    if not source_label:
        source_label = {
            "upstream_sample": "Upstream real sample",
            "user_supplied": "User-supplied input",
            "synthetic_fixture": "Synthetic seeded fixture",
            "npa_seeded_fixture": "Synthetic seeded fixture",
        }.get(source_kind, "Source input")
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
        fo_sample = (
            fo_samples.get(clip, {}) if isinstance(fo_samples.get(clip), dict) else {}
        )
        curation_flags = []
        if fo_sample.get("redundant"):
            curation_flags.append("redundant")
        augmented_samples.append(
            {
                "group": "augmented",
                "data_role": "synthetic_augmented",
                "data_role_label": "Synthetic / augmented output",
                "source": {
                    "kind": "cosmos_transfer_output",
                    "derived_from": input_source,
                },
                "lineage": {
                    "source_sha256": input_source.get("sha256"),
                    "conditioning_uri": (
                        (input_source.get("cosmos_conditioning") or {}).get(
                            "staged_uri"
                        )
                        if isinstance(input_source.get("cosmos_conditioning"), dict)
                        else ""
                    ),
                    "relation": "cosmos_transfer_variant_of_conditioned_source",
                },
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
        is_conditioning = vname.lower().endswith("conditioning.mp4")
        sample = {
            "group": "conditioning" if is_conditioning else "source",
            "data_role": (
                "derived_conditioning" if is_conditioning else "source_input"
            ),
            "data_role_label": (
                "Derived conditioning clip" if is_conditioning else source_label
            ),
            "source": input_source,
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
        (conditioning_samples if is_conditioning else source_samples).append(sample)
    for idx, key in enumerate(sorted(input_frames)[:12]):
        name = key.rsplit("/", 1)[-1]
        is_conditioning = name.lower().startswith("conditioning-frame-")
        is_fixture = (
            source_kind in {"synthetic_fixture", "npa_seeded_fixture"}
            and not is_conditioning
        )
        sample = {
            "group": "conditioning" if is_conditioning else "source",
            "data_role": (
                "derived_conditioning"
                if is_conditioning
                else ("synthetic_fixture" if is_fixture else "source_input")
            ),
            "data_role_label": (
                "Derived conditioning frame" if is_conditioning else source_label
            ),
            "source": input_source,
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
        (conditioning_samples if is_conditioning else source_samples).append(sample)

    multiply = curation.get("multiply", {}) if isinstance(curation, dict) else {}
    if not isinstance(multiply, dict):
        multiply = {}
    variant_count = (
        multiply.get("variant_count") or curation.get("variant_count") or len(by_clip)
    )
    fo_brain = (
        fo_curation.get("brain", {})
        if isinstance(fo_curation.get("brain"), dict)
        else {}
    )
    fo_selection = (
        fo_curation.get("selection", {})
        if isinstance(fo_curation.get("selection"), dict)
        else {}
    )
    curation_engine = (
        str(curation.get("curation_engine") or "") if isinstance(curation, dict) else ""
    )
    review = {
        "engine": curation_engine,
        "real_fiftyone": curation_engine == "fiftyone-brain",
        "label": (
            "Real FiftyOne Brain review"
            if curation_engine == "fiftyone-brain"
            else "Artifact summary only — FiftyOne did not run"
        ),
        "limitation": (
            ""
            if curation_engine == "fiftyone-brain"
            else "No FiftyOne Brain uniqueness or duplicate review is available for this run."
        ),
    }
    summary = {
        "augmented_count": len(by_clip),
        "input_count": len(input_frames),
        "source_input_count": len(source_samples),
        # A synthetic fixture is a source input, but never "original" real data.
        "original_input_count": sum(
            1
            for sample in source_samples
            if sample.get("data_role") != "synthetic_fixture"
        ),
        "conditioning_count": sum(
            1
            for sample in conditioning_samples
            if sample.get("data_role") == "derived_conditioning"
        ),
        "fixture_count": sum(
            1
            for sample in source_samples
            if sample.get("data_role") == "synthetic_fixture"
        ),
        "synthetic_augmented_count": len(augmented_samples),
        "variant_count": int(variant_count or 0),
        "multiply_mode": str(
            multiply.get("mode") or curation.get("multiply_mode") or ""
        ),
        "grade_score": grade.get("score") if isinstance(grade, dict) else None,
        "grade_decision": str(decision.get("decision") or "")
        if isinstance(decision, dict)
        else "",
        # Real FiftyOne curation surface (empty when the curate stage ran
        # report-only, i.e. outside the npa-fiftyone image).
        "curation_engine": curation_engine,
        "curated_kept": curation.get("curated_kept")
        if isinstance(curation, dict)
        else None,
        "curated_dropped": curation.get("curated_dropped")
        if isinstance(curation, dict)
        else None,
        "near_duplicate_count": (
            fo_brain.get("near_duplicate_count")
            if fo_brain.get("near_duplicate_count") is not None
            else fo_selection.get("near_duplicate_count")
        ),
        "uniqueness": fo_brain.get("uniqueness", {}),
    }
    visualization = (
        fo_curation.get("visualization", [])
        if isinstance(fo_curation.get("visualization"), list)
        else []
    )
    return {
        "fields": sorted(tag_keys),
        "summary": summary,
        # Source, derived conditioning, and generated variants are separate groups.
        "samples": source_samples + conditioning_samples + augmented_samples,
        "visualization": visualization,
        "source": input_source,
        "review": review,
    }
