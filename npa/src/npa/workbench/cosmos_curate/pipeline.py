"""Run NVIDIA Cosmos Curator's real split-annotate stages over a clip set.

Upstream ships its video curation as Ray pipeline stages
(``cosmos_curator.pipelines.video.*`` in
https://github.com/nvidia-cosmos/cosmos-curate, Apache-2.0, Copyright (c)
2025-2026 NVIDIA CORPORATION & AFFILIATES). Each stage is a ``CuratorStage`` with
a ``process_data(tasks)`` method, and the stages this module uses need no GPU and
no Ray scheduler, so they can be driven directly:

``VideoDownloader`` (probe + read) → ``FixedStrideExtractorStage`` (segment into
clips) → ``ClipTranscodingStage`` (ffmpeg transcode per clip) → optional
``MotionVectorDecodeStage`` + ``MotionFilterStage`` (motion scoring / filtering)
→ ``ClipWriterStage`` (canonical curator output layout).

The output is upstream's own layout, unchanged::

    <output>/clips/<clip-uuid>.mp4
    <output>/metas/v0/<clip-uuid>.json        # spans, geometry, motion scores
    <output>/processed_videos/<video>.json
    <output>/processed_clip_chunks/<video>_<chunk>.json

:func:`split_pipeline_argv` covers the other supported way to run the curator —
upstream's ``video-pipeline split`` entry point inside the curator container,
which adds the GPU stages (TransNetV2 shot detection, aesthetic filtering,
InternVideo2 / Cosmos-Embed1 embeddings, VLM captioning). Both paths write the
same layout, so :mod:`.report` reads either.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from npa.workbench.cosmos_curate.upstream import (
    PINNED_REVISION,
    CPU_ENCODER,
    UPSTREAM_LICENSE,
    UPSTREAM_REPO,
    CosmosCurateError,
    CosmosCurateUnavailable,
    ensure_upstream_importable,
    probe_availability,
)

_log = logging.getLogger(__name__)

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".webm")

DEFAULT_CLIP_LEN_S = 10.0
DEFAULT_MIN_CLIP_LEN_S = 2.0
# Upstream's motion thresholds (MotionFilterConfig defaults).
DEFAULT_MOTION_GLOBAL_MEAN_THRESHOLD = 0.00098
DEFAULT_MOTION_PER_PATCH_THRESHOLD = 0.000001

MOTION_MODES = ("disable", "score-only", "enable")

ENGINE_IN_PROCESS = "cosmos-curator-stages"
ENGINE_PIPELINE_CLI = "cosmos-curator-video-pipeline"


@dataclass(frozen=True)
class CuratorRunResult:
    """What one in-process curator run did."""

    status: str
    engine: str
    source: str
    encoder: str
    input_dir: str
    output_dir: str
    input_videos: int
    clips_written: int
    clips_filtered: int
    motion_filter: str
    stages: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["upstream"] = {"repo": UPSTREAM_REPO, "license": UPSTREAM_LICENSE}
        return payload


def discover_videos(input_dir: Path) -> list[Path]:
    """Video files under ``input_dir``, deepest-path-stable ordering."""

    if input_dir.is_file():
        return [input_dir] if input_dir.suffix.lower() in VIDEO_SUFFIXES else []
    if not input_dir.is_dir():
        return []
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def curate_videos(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    clip_len_s: float = DEFAULT_CLIP_LEN_S,
    clip_stride_s: float = 0.0,
    min_clip_length_s: float = DEFAULT_MIN_CLIP_LEN_S,
    limit_clips: int = 0,
    motion_filter: str = "score-only",
    encoder: str = "",
    verbose: bool = False,
) -> CuratorRunResult:
    """Curate every video under ``input_dir`` with upstream's own stages.

    Both paths are local: the caller stages S3 inputs down and the outputs back
    up, so upstream's storage client never needs credentials of its own.
    """

    if motion_filter not in MOTION_MODES:
        raise CosmosCurateError(f"--motion-filter must be one of {', '.join(MOTION_MODES)}")
    source_dir = Path(input_dir)
    target_dir = Path(output_dir)
    videos = discover_videos(source_dir)
    if not videos:
        raise CosmosCurateError(f"no video files found under {source_dir}")

    availability = probe_availability()
    if not availability.can_run_in_process:
        raise CosmosCurateUnavailable(availability.reason())
    chosen_encoder = encoder or availability.encoder
    root = ensure_upstream_importable()
    target_dir.mkdir(parents=True, exist_ok=True)

    stages = _load_stages()
    stage_names = ["VideoDownloader", "FixedStrideExtractorStage", "ClipTranscodingStage"]
    if motion_filter != "disable":
        stage_names.extend(["MotionVectorDecodeStage", "MotionFilterStage"])
    stage_names.append("ClipWriterStage")

    downloader = _construct(
        stages,
        "VideoDownloader",
        input_path=str(source_dir),
        input_s3_profile_name="default",
        verbose=verbose,
    )
    downloader.stage_setup()
    splitter = _construct(
        stages,
        "FixedStrideExtractorStage",
        clip_len_s=clip_len_s,
        clip_stride_s=clip_stride_s or clip_len_s,
        min_clip_length_s=min_clip_length_s,
        limit_clips=limit_clips,
        verbose=verbose,
    )
    transcoder = _construct(
        stages,
        "ClipTranscodingStage",
        encoder=chosen_encoder,
        encode_batch_size=8,
        verbose=verbose,
    )
    writer = _construct(
        stages,
        "ClipWriterStage",
        output_path=str(target_dir),
        input_path=str(source_dir),
        output_s3_profile_name="default",
        upload_clips=True,
        upload_clip_info_in_chunks=False,
        upload_clip_info_in_lance=False,
        upload_cds_parquet=False,
        dry_run=False,
        generate_embeddings=False,
        embedding_algorithm="internvideo2",
        embedding_model_version="",
        generate_previews=False,
        caption_models=[],
        verbose=verbose,
    )
    writer.stage_setup()

    errors: dict[str, str] = {}
    clips_written = 0
    clips_filtered = 0
    for video_path in videos:
        task = _construct(
            stages,
            "SplitPipeTask",
            session_id=str(video_path),
            video=_construct(stages, "Video", input_video=video_path),
        )
        tasks: list[Any] = [task]
        tasks = downloader.process_data(tasks) or tasks
        tasks = splitter.process_data(tasks) or tasks
        tasks = transcoder.process_data(tasks) or []
        if not tasks:
            errors[video_path.name] = "transcoding produced no clip chunks"
            continue
        if motion_filter != "disable":
            tasks = _run_motion_stages(
                stages,
                tasks,
                score_only=motion_filter == "score-only",
                verbose=verbose,
            )
        for chunk in tasks:
            clips_written += len(chunk.video.clips)
            clips_filtered += len(chunk.video.filtered_clips)
            for key, message in (chunk.video.errors or {}).items():
                errors[f"{video_path.name}:{key}"] = str(message)[:300]
            for key, message in (chunk.errors or {}).items():
                errors[f"{video_path.name}:{key}"] = str(message)[:300]
        writer.process_data(tasks)

    return CuratorRunResult(
        status="completed",
        engine=ENGINE_IN_PROCESS,
        source=str(root),
        encoder=chosen_encoder,
        input_dir=str(source_dir),
        output_dir=str(target_dir),
        input_videos=len(videos),
        clips_written=clips_written,
        clips_filtered=clips_filtered,
        motion_filter=motion_filter,
        stages=stage_names,
        errors=errors,
    )


def _run_motion_stages(
    stages: dict[str, Any],
    tasks: list[Any],
    *,
    score_only: bool,
    verbose: bool,
) -> list[Any]:
    """Score (and optionally drop) low-motion clips with upstream's stages.

    ``num_gpus_per_worker=0`` selects upstream's CPU code path in
    ``check_if_small_motion``, so this runs on the CPU tier.
    """

    decode = _construct(stages, "MotionVectorDecodeStage", num_cpus_per_worker=1.0, verbose=verbose)
    scorer = _construct(
        stages,
        "MotionFilterStage",
        score_only=score_only,
        global_mean_threshold=DEFAULT_MOTION_GLOBAL_MEAN_THRESHOLD,
        per_patch_min_256_threshold=DEFAULT_MOTION_PER_PATCH_THRESHOLD,
        num_gpus_per_worker=0,
        verbose=verbose,
    )
    tasks = decode.process_data(tasks) or tasks
    return scorer.process_data(tasks) or tasks


def _construct(stages: dict[str, Any], name: str, /, **kwargs: Any) -> Any:
    """Build an upstream stage, turning a signature mismatch into "cannot run here".

    Upstream's constructor keyword arguments are not a published API, so a checkout
    whose signature moved raises ``TypeError`` from inside its own code. Uncaught,
    that reaches the operator as a traceback with no indication that the fix is a
    version mismatch rather than a bug in this call.
    """

    try:
        return stages[name](**kwargs)
    except TypeError as exc:
        raise CosmosCurateUnavailable(
            f"upstream's {name} does not accept the arguments this code passes "
            f"({exc}); the checkout is likely not at {PINNED_REVISION}"
        ) from exc


def _load_stages() -> dict[str, Any]:
    """Import the upstream stage classes and task types."""

    try:
        from cosmos_curator.pipelines.video.clipping.clip_extraction_stages import (  # type: ignore
            ClipTranscodingStage,
            FixedStrideExtractorStage,
        )
        from cosmos_curator.pipelines.video.filtering.motion.motion_filter_stages import (  # type: ignore
            MotionFilterStage,
            MotionVectorDecodeStage,
        )
        from cosmos_curator.pipelines.video.read_write.download_stages import VideoDownloader  # type: ignore
        from cosmos_curator.pipelines.video.read_write.metadata_writer_stage import ClipWriterStage  # type: ignore
        from cosmos_curator.pipelines.video.utils.data_model import SplitPipeTask, Video  # type: ignore
    except Exception as exc:  # noqa: BLE001 - surfaced as "cannot run here"
        raise CosmosCurateUnavailable(f"Cosmos Curator stages are not importable: {exc}") from exc
    return {
        "ClipTranscodingStage": ClipTranscodingStage,
        "ClipWriterStage": ClipWriterStage,
        "FixedStrideExtractorStage": FixedStrideExtractorStage,
        "MotionFilterStage": MotionFilterStage,
        "MotionVectorDecodeStage": MotionVectorDecodeStage,
        "SplitPipeTask": SplitPipeTask,
        "Video": Video,
        "VideoDownloader": VideoDownloader,
    }


def split_pipeline_argv(
    *,
    input_video_path: str,
    output_clip_path: str,
    splitting_algorithm: str = "fixed-stride",
    fixed_stride_split_duration: int = 10,
    fixed_stride_min_clip_length_s: float = DEFAULT_MIN_CLIP_LEN_S,
    captioning_algorithm: str = "",
    embedding_algorithm: str = "",
    generate_embeddings: bool = False,
    limit: int = 0,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build upstream's ``video-pipeline split`` argv for the container path.

    This is the documented entry point of the curator container (the ``pixi run
    --as-is video-pipeline split ...`` command in upstream's end-user guide). The
    flags here are upstream's own; the in-process path above covers the GPU-free
    subset of the same pipeline.
    """

    if not input_video_path or not output_clip_path:
        raise CosmosCurateError("split_pipeline_argv requires input and output paths")
    if splitting_algorithm not in {"fixed-stride", "transnetv2"}:
        raise CosmosCurateError("--splitting-algorithm must be fixed-stride or transnetv2")
    argv = [
        "video-pipeline",
        "split",
        "--input-video-path",
        input_video_path,
        "--output-clip-path",
        output_clip_path,
        "--splitting-algorithm",
        splitting_algorithm,
    ]
    if splitting_algorithm == "fixed-stride":
        argv += [
            "--fixed-stride-split-duration",
            str(int(fixed_stride_split_duration)),
            "--fixed-stride-min-clip-length-s",
            str(fixed_stride_min_clip_length_s),
        ]
    if generate_embeddings:
        if embedding_algorithm:
            argv += ["--embedding-algorithm", embedding_algorithm]
    else:
        argv.append("--no-generate-embeddings")
    if captioning_algorithm:
        argv += ["--captioning-algorithm", captioning_algorithm]
    if limit:
        argv += ["--limit", str(int(limit))]
    argv.extend(extra_args)
    return argv


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def default_encoder(*, environ: dict[str, str] | None = None) -> str:
    """Encoder the transcoding stage will use here, or ``""`` when none fits."""

    env = os.environ if environ is None else environ
    override = str(env.get("NPA_COSMOS_CURATE_ENCODER", "") or "").strip()
    if override:
        return override
    return probe_availability(environ=env).encoder or CPU_ENCODER
