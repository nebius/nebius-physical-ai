"""Curate a Physical AI Data Factory run's augmented variants with Cosmos Curator.

:func:`curate_augmented` is the stage entry point. It stages the run's augmented
variants down from S3 (one video per variant, named after the variant so the
curator's ``source_video`` field maps straight back), hands them to the real
upstream curator stages in :mod:`.pipeline`, publishes the curator's own output
tree, and summarizes it as an ``npa.cosmos_curate.curation.v1`` report.

:func:`ingest_output` reads that tree — ``clips/``, ``metas/v0/*.json``,
``processed_videos/`` — and works for output written by either supported path:
the in-process stages or upstream's full ``video-pipeline split`` in the curator
container.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Container
from urllib.parse import urlparse

from npa.workbench.cosmos_curate.pipeline import (
    DEFAULT_CLIP_LEN_S,
    DEFAULT_MIN_CLIP_LEN_S,
    curate_videos,
)
from npa.workbench.cosmos_curate.upstream import (
    UPSTREAM_LICENSE,
    UPSTREAM_REPO,
    CosmosCurateError,
    CosmosCurateUnavailable,
    probe_availability,
)

_log = logging.getLogger(__name__)

RESULT_FILENAME = "cosmos_curator.json"
REPORT_SCHEMA = "npa.cosmos_curate.curation.v1"
CLIPS_DIR = "clips"
METAS_DIR = "metas/v0"
PROCESSED_VIDEOS_DIR = "processed_videos"
VARIANT_VIDEO_NAME = "augmented_video.mp4"

ENGINE_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class CuratedClip:
    """One clip the curator produced, as recorded in its own metadata."""

    clip_id: str
    source: str
    span: list[float] = field(default_factory=list)
    duration_s: float = 0.0
    num_frames: int = 0
    width: int = 0
    height: int = 0
    framerate: float = 0.0
    num_bytes: int = 0
    motion_score_global_mean: float | None = None
    motion_score_per_patch_min_256: float | None = None
    caption: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CurationReport:
    """Run-level Cosmos Curator report."""

    status: str
    engine: str
    augment_uri: str
    curated_uri: str
    result_uri: str
    variant_count: int
    clip_count: int
    filtered_count: int
    total_duration_s: float
    motion_filter: str
    generated_at: str
    encoder: str = ""
    source: str = ""
    clips: list[CuratedClip] = field(default_factory=list)
    per_variant: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["clips"] = [clip.to_dict() for clip in self.clips]
        payload["schema"] = REPORT_SCHEMA
        payload["upstream"] = {"repo": UPSTREAM_REPO, "license": UPSTREAM_LICENSE}
        return payload


def result_uri_for(output_uri: str) -> str:
    if output_uri.endswith(".json"):
        return output_uri
    return output_uri.rstrip("/") + f"/{RESULT_FILENAME}"


def curate_augmented(
    *,
    augment_uri: str,
    curated_uri: str,
    report_uri: str = "",
    clip_len_s: float = DEFAULT_CLIP_LEN_S,
    min_clip_length_s: float = DEFAULT_MIN_CLIP_LEN_S,
    motion_filter: str = "score-only",
    limit_clips: int = 0,
    max_variants: int = 0,
    require_curator: bool = False,
    verbose: bool = False,
    storage: Any | None = None,
) -> CurationReport:
    """Run Cosmos Curator over a run's augmented variants and report the result.

    When the upstream curator cannot run here the report records
    ``engine: unavailable`` plus the reason, so the blueprint's FiftyOne review
    stage still has something to merge. Pass ``require_curator=True`` to make
    that case an error instead.
    """

    if not augment_uri:
        raise CosmosCurateError("--augment-uri is required")
    if not curated_uri:
        raise CosmosCurateError("--curated-uri is required")

    store = storage if storage is not None else _storage()
    target_report_uri = result_uri_for(report_uri or curated_uri)
    warnings: list[str] = []

    availability = probe_availability()
    if not availability.can_run_in_process:
        reason = availability.reason()
        if require_curator:
            raise CosmosCurateUnavailable(reason)
        warnings.append(reason)
        return CurationReport(
            status="skipped",
            engine=ENGINE_UNAVAILABLE,
            augment_uri=augment_uri,
            curated_uri=curated_uri,
            result_uri=target_report_uri,
            variant_count=0,
            clip_count=0,
            filtered_count=0,
            total_duration_s=0.0,
            motion_filter=motion_filter,
            generated_at=_now(),
            warnings=warnings,
        )

    with tempfile.TemporaryDirectory(prefix="npa-cosmos-curate-") as tmp:
        workdir = Path(tmp)
        staged = workdir / "input"
        produced = workdir / "output"
        variants = _stage_variants(
            augment_uri,
            staged,
            store=store,
            max_variants=max_variants,
            warnings=warnings,
        )
        if not variants:
            raise CosmosCurateError(f"no augmented variant videos found under {augment_uri}")

        run = curate_videos(
            input_dir=staged,
            output_dir=produced,
            clip_len_s=clip_len_s,
            min_clip_length_s=min_clip_length_s,
            limit_clips=limit_clips,
            motion_filter=motion_filter,
            verbose=verbose,
        )
        ingested = ingest_output(produced)
        published = _publish(produced, curated_uri, store=store, warnings=warnings)

    clips = [
        _rewrite_source(clip, variants=variants, curated_uri=curated_uri)
        for clip in ingested["clips"]
    ]
    per_variant: dict[str, int] = {}
    for clip in clips:
        per_variant[clip.source] = per_variant.get(clip.source, 0) + 1
    if not published and require_curator:
        raise CosmosCurateError(
            f"curated {len(clips)} clips but could not publish them to {curated_uri}; "
            f"{'; '.join(warnings[-2:]) or 'see warnings'}"
        )
    return CurationReport(
        # The clips exist only in a temporary directory that this function is about to
        # drop, so "completed" would point the next stage at an empty prefix.
        status="completed" if published else "degraded",
        engine=run.engine,
        augment_uri=augment_uri,
        curated_uri=curated_uri,
        result_uri=target_report_uri,
        variant_count=len(variants),
        clip_count=len(clips),
        filtered_count=run.clips_filtered,
        total_duration_s=round(sum(clip.duration_s for clip in clips), 3),
        motion_filter=motion_filter,
        generated_at=_now(),
        encoder=run.encoder,
        source=run.source,
        clips=clips,
        per_variant=per_variant,
        warnings=warnings,
        errors=run.errors,
    )


def ingest_output(output_dir: str | Path) -> dict[str, Any]:
    """Parse a Cosmos Curator output tree into clip records."""

    root = Path(output_dir)
    metas_dir = root / METAS_DIR
    clips: list[CuratedClip] = []
    if metas_dir.is_dir():
        for meta_path in sorted(metas_dir.glob("*.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(meta, dict):
                clips.append(_clip_from_meta(meta, clip_id=meta_path.stem))
    processed = sorted((root / PROCESSED_VIDEOS_DIR).glob("*.json")) if (root / PROCESSED_VIDEOS_DIR).is_dir() else []
    clip_files = sorted((root / CLIPS_DIR).glob("*.mp4")) if (root / CLIPS_DIR).is_dir() else []
    return {
        "clips": clips,
        "clip_files": [str(path) for path in clip_files],
        "processed_videos": [path.name for path in processed],
    }


def _clip_from_meta(meta: dict[str, Any], *, clip_id: str) -> CuratedClip:
    span_raw = meta.get("duration_span") or []
    span = [float(value) for value in span_raw if isinstance(value, (int, float))]
    duration = round(span[1] - span[0], 3) if len(span) == 2 else 0.0
    motion = meta.get("motion_score") if isinstance(meta.get("motion_score"), dict) else {}
    errors = meta.get("errors")
    caption = ""
    windows = meta.get("windows")
    if isinstance(windows, list):
        for window in windows:
            if not isinstance(window, dict):
                continue
            for key, value in window.items():
                if key.endswith("_caption") and isinstance(value, str) and value.strip():
                    caption = value.strip()
                    break
            if caption:
                break
    return CuratedClip(
        clip_id=str(meta.get("span_uuid") or clip_id),
        source=str(meta.get("source_video") or ""),
        span=span,
        duration_s=duration,
        num_frames=int(meta.get("num_frames") or 0),
        width=int(meta.get("width") or 0),
        height=int(meta.get("height") or 0),
        framerate=float(meta.get("framerate") or 0.0),
        num_bytes=int(meta.get("num_bytes") or 0),
        motion_score_global_mean=_maybe_float(motion.get("global_mean")),
        motion_score_per_patch_min_256=_maybe_float(motion.get("per_patch_min_256")),
        caption=caption,
        errors=[str(entry) for entry in errors] if isinstance(errors, list) else [],
    )


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rewrite_source(clip: CuratedClip, *, variants: dict[str, str], curated_uri: str) -> CuratedClip:
    """Map a clip's local staging path back to the variant it came from."""

    stem = Path(clip.source).stem
    variant = variants.get(stem, stem)
    return CuratedClip(
        clip_id=clip.clip_id,
        source=variant,
        span=clip.span,
        duration_s=clip.duration_s,
        num_frames=clip.num_frames,
        width=clip.width,
        height=clip.height,
        framerate=clip.framerate,
        num_bytes=clip.num_bytes,
        motion_score_global_mean=clip.motion_score_global_mean,
        motion_score_per_patch_min_256=clip.motion_score_per_patch_min_256,
        caption=clip.caption,
        errors=clip.errors,
    )


# ---------------------------------------------------------------------------
# Artifact plumbing
# ---------------------------------------------------------------------------


def _storage() -> Any:
    # Deferred: a run over local paths must not need object-storage credentials.
    from npa.clients.storage import LazyStorageClient

    return LazyStorageClient()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _is_remote(uri: str) -> bool:
    return uri.startswith("s3://")


def _local_path(uri: str) -> str:
    return uri[len("file://") :] if uri.startswith("file://") else uri


def _list_keys(uri: str, *, store: Any) -> list[str]:
    prefixed = uri if uri.endswith("/") else uri + "/"
    bucket, prefix = _split(prefixed)
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = store.s3.list_objects_v2(**kwargs)
        keys.extend(entry["Key"] for entry in page.get("Contents", []) if entry.get("Key"))
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return keys


def _stage_variants(
    augment_uri: str,
    staged: Path,
    *,
    store: Any,
    max_variants: int,
    warnings: list[str],
) -> dict[str, str]:
    """Download one video per augmented variant, named ``<variant>.mp4``.

    Returns a ``stem -> variant id`` map so curator metadata (which records the
    staged path) can be reported against the variant it came from.
    """

    staged.mkdir(parents=True, exist_ok=True)
    prefixed = augment_uri if augment_uri.endswith("/") else augment_uri + "/"
    variants: dict[str, str] = {}

    if not _is_remote(prefixed):
        root = Path(_local_path(prefixed))
        if not root.is_dir():
            return {}
        for child in sorted(child for child in root.iterdir() if child.is_dir()):
            video = child / VARIANT_VIDEO_NAME
            if not video.is_file():
                candidates = sorted(child.glob("*.mp4"))
                video = candidates[0] if candidates else video
            if not video.is_file():
                warnings.append(f"variant {child.name} has no video to curate")
                continue
            target = staged / f"{_safe_stem(child.name, taken=variants)}.mp4"
            target.write_bytes(video.read_bytes())
            variants[target.stem] = child.name
            if max_variants and len(variants) >= max_variants:
                break
        return variants

    bucket, prefix = _split(prefixed)
    by_variant: dict[str, list[str]] = {}
    for key in _list_keys(prefixed, store=store):
        if not key.startswith(prefix) or not key.lower().endswith(".mp4"):
            continue
        rest = key[len(prefix) :]
        head, _, tail = rest.partition("/")
        if not tail or not head:
            continue
        by_variant.setdefault(head, []).append(key)

    for variant in sorted(by_variant):
        keys = by_variant[variant]
        preferred = [key for key in keys if key.endswith(VARIANT_VIDEO_NAME)] or sorted(keys)
        target = staged / f"{_safe_stem(variant, taken=variants)}.mp4"
        try:
            local = store.download_path(f"s3://{bucket}/{preferred[0]}", str(target))
        except Exception as exc:  # noqa: BLE001 - skip a variant we cannot read
            warnings.append(f"could not download variant {variant}: {exc}"[:300])
            continue
        path = Path(local)
        if path.is_dir():
            found = sorted(path.rglob("*.mp4"))
            if not found:
                warnings.append(f"variant {variant} download produced no mp4")
                continue
            found[0].replace(target)
        elif path != target:
            path.replace(target)
        variants[target.stem] = variant
        if max_variants and len(variants) >= max_variants:
            break
    return variants


def _safe_stem(variant: str, *, taken: Container[str] = frozenset()) -> str:
    """A filesystem-safe stem that still round-trips to the variant name.

    Every unsafe character maps to ``_``, so ``cam a``, ``cam/b`` and ``cam_b`` all
    collapse to the same stem. Staged under one name, the second download overwrites
    the first and its clips are attributed to the wrong variant, so a collision gets
    a suffix instead.
    """

    stem = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_" for char in variant
    )
    if stem not in taken:
        return stem
    for index in range(2, 1000):
        candidate = f"{stem}-{index}"
        if candidate not in taken:
            return candidate
    raise CosmosCurateError(f"could not find a free staging name for variant {variant!r}")


def _publish(produced: Path, curated_uri: str, *, store: Any, warnings: list[str]) -> bool:
    """Publish the curator's output tree to ``curated_uri``.

    Returns whether the clips are actually readable at ``curated_uri``. The caller
    needs that: a report claiming ``completed`` with a real clip count, whose clips
    never left the temporary directory, sends the next stage to an empty prefix.
    """

    if not produced.is_dir():
        warnings.append("curator produced no output directory")
        return False
    if not _is_remote(curated_uri):
        target = Path(_local_path(curated_uri))
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(produced.rglob("*")):
            if not path.is_file():
                continue
            dest = target / path.relative_to(produced)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(path.read_bytes())
        return True
    try:
        store.upload_directory(str(produced), curated_uri.rstrip("/") + "/")
    except Exception as exc:  # noqa: BLE001 - report but keep the run's findings
        _log.warning("could not publish curator output to %s", curated_uri, exc_info=True)
        warnings.append(f"could not publish curator output to {curated_uri}: {exc}"[:300])
        return False
    return True


def write_report(payload: dict[str, Any], *, result_uri: str, storage: Any | None = None) -> str:
    """Write the curation report to S3 or a local path."""

    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if _is_remote(result_uri):
        store = storage if storage is not None else _storage()
        with tempfile.TemporaryDirectory(prefix="npa-cosmos-curate-out-") as tmp:
            local = Path(tmp) / RESULT_FILENAME
            local.write_text(body, encoding="utf-8")
            return store.upload_file(str(local), result_uri)
    path = Path(_local_path(result_uri))
    if path.suffix != ".json":
        path = path / RESULT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)
