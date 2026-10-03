"""Run LingBot World Base (Cam) continuation stages with durable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping
from urllib.parse import urlparse

from npa.clients.storage import StorageClient
from npa.solutions.lingbot_camera import (
    FRAME_COUNT,
    MODEL_ID,
    MODEL_REF,
    SOURCE_REF,
    SOURCE_REPO,
    TEXT_ENCODER_REF,
    create_camera_controls,
    generate_camera_video,
)
from npa.solutions.media_input import download_input


SCHEMA = "npa.workbench.lingbot_world.controlled_continuation.v1"
MODEL_CARD_URL = "https://huggingface.co/robbyant/lingbot-world-base-cam"
PAPER_URL = "https://arxiv.org/abs/2601.20540"
UPSTREAM_CITATION = (
    "Robbyant Team, Advancing Open-source World Models, arXiv:2601.20540, 2026"
)


class LingBotWorldStageError(RuntimeError):
    """Raised when a LingBot World continuation stage cannot prove its contract."""


def file_sha256(path: Path) -> str:
    """Return the SHA-256 identity of one regular file.

    Args:
        path: Existing regular file to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Raises:
        OSError: The file cannot be read.
    """

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_s3(uri: str) -> bool:
    """Return whether a location is an S3 object URI."""

    return str(uri).startswith("s3://")


def _uri_parent(uri: str) -> str:
    """Return the directory-like parent of an artifact URI."""

    stripped = str(uri).rstrip("/")
    if "/" not in stripped:
        raise LingBotWorldStageError(f"Artifact URI has no parent: {uri!r}")
    return stripped.rsplit("/", 1)[0]


def _uri_filename(uri: str) -> str:
    """Return an artifact filename while rejecting directory-only destinations."""

    name = Path(urlparse(uri).path).name
    if not name:
        raise LingBotWorldStageError(f"Artifact URI has no filename: {uri!r}")
    return name


def _require_s3_uri(uri: str, label: str) -> None:
    """Require a safe, explicit S3 URI for a durable workflow artifact."""

    parsed = urlparse(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.lstrip("/")
        or parsed.query
        or parsed.fragment
    ):
        raise LingBotWorldStageError(f"{label} must be an explicit s3:// URI")


def _storage() -> StorageClient:
    """Resolve the stage's configured object-storage client."""

    return StorageClient.from_environment()


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Write a finite, stable JSON artifact."""

    path.write_text(
        json.dumps(dict(document), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(uri: str, destination: Path, storage: Any) -> dict[str, Any]:
    """Download and validate one JSON object from a stage artifact."""

    _download_file(uri, destination, storage)
    try:
        document = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LingBotWorldStageError(f"Invalid JSON artifact: {uri}") from exc
    if not isinstance(document, dict):
        raise LingBotWorldStageError(f"JSON artifact must be an object: {uri}")
    return document


def _download_file(uri: str, destination: Path, storage: Any) -> Path:
    """Copy one local or S3 artifact to a new local path."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if _is_s3(uri):
        storage.download_file(uri, str(destination))
    else:
        shutil.copy2(uri, destination)
    if not destination.is_file():
        raise LingBotWorldStageError(f"Artifact download did not create {destination}")
    return destination


def _publish_directory(root: Path, manifest_uri: str, storage: Any) -> None:
    """Publish one new stage directory without reusing another run's prefix."""

    target = _uri_parent(manifest_uri)
    if _is_s3(manifest_uri):
        storage.upload_directory(str(root), target + "/", require_empty=True)
        return
    target_path = Path(target)
    if target_path.exists():
        raise LingBotWorldStageError(f"Output directory must be new: {target_path}")
    shutil.copytree(root, target_path)


def _upstream_provenance() -> dict[str, Any]:
    """Return immutable source, checkpoint, licence, and credit facts."""

    return {
        "source": {
            "repository": SOURCE_REPO,
            "revision": SOURCE_REF,
            "license": "Apache-2.0",
            "authors": "Robbyant Team",
        },
        "checkpoint": {
            "model_id": MODEL_ID,
            "revision": MODEL_REF,
            "model_card": MODEL_CARD_URL,
            "license": "Apache-2.0",
            "baked_into_image": False,
        },
        "tokenizer": {
            "model_id": "google/umt5-xxl",
            "revision": TEXT_ENCODER_REF,
            "license": "Apache-2.0",
            "baked_into_image": False,
        },
        "citation": UPSTREAM_CITATION,
        "paper": PAPER_URL,
        "upstream_notice": "LingBot source acknowledges the Wan2.2 team.",
        "npa_modification": "Durable camera-control artifact wiring and evaluation; upstream inference remains generate.py.",
    }


def _validate_image(path: Path) -> None:
    """Decode an input image before it becomes a model context frame."""

    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.convert("RGB").save(path.with_suffix(".normalized.png"))


def _normalized_context(source: Path, destination: Path) -> None:
    """Write the verified RGB context image used by both continuations."""

    normalized = source.with_suffix(".normalized.png")
    _validate_image(source)
    normalized.replace(destination)


def _prepared_document(
    *,
    manifest_uri: str,
    input_uri: str,
    input_sha256: str,
    context: Path,
    prescribed: dict[str, str],
    alternative: dict[str, str],
    run_id: str,
) -> dict[str, Any]:
    """Build the provenance manifest that binds one matched control pair."""

    root = _uri_parent(manifest_uri)
    return {
        "schema": SCHEMA,
        "stage": "prepare",
        "run_id": run_id,
        "upstream": _upstream_provenance(),
        "input": {
            "source_uri": input_uri,
            "source_sha256": input_sha256,
            "context_uri": root + "/context.png",
            "context_sha256": file_sha256(context),
        },
        "controls": {
            "prescribed": {
                "uri": root + "/prescribed-controls",
                "files": prescribed,
                "description": "Authored forward/right camera translation with positive yaw.",
            },
            "alternative": {
                "uri": root + "/alternative-controls",
                "files": alternative,
                "description": "Matched authored reverse/left camera translation with negative yaw.",
            },
        },
        "not_claimed": [
            "robot_action_conditioning",
            "robot_action_dynamics",
            "camera_calibration_accuracy",
            "training",
        ],
    }


def prepare_context(
    input_uri: str,
    input_sha256: str,
    manifest_uri: str,
    run_id: str,
) -> dict[str, Any]:
    """Stage one verified image and paired upstream camera-control trajectories.

    Args:
        input_uri: S3 URI of the user-selected context image.
        input_sha256: Required SHA-256 of the source image bytes.
        manifest_uri: Run-scoped S3 URI for the prepared manifest.
        run_id: Workflow run identity included in provenance.

    Returns:
        The published preparation manifest.

    Raises:
        LingBotWorldStageError: Inputs or durable output locations are invalid.
        RuntimeError: Input download or image decoding fails.
    """

    _require_s3_uri(input_uri, "input_uri")
    _require_s3_uri(manifest_uri, "manifest_uri")
    with tempfile.TemporaryDirectory(prefix="npa-lingbot-prepare-") as temporary:
        root = Path(temporary) / "prepared"
        root.mkdir()
        source = download_input(input_uri, input_sha256, root / "source-image")
        context = root / "context.png"
        _normalized_context(source, context)
        prescribed = create_camera_controls(
            root / "prescribed-controls",
            translation_x=0.7,
            translation_z=1.5,
            yaw_radians=0.1,
        )
        alternative = create_camera_controls(
            root / "alternative-controls",
            translation_x=-0.7,
            translation_z=1.5,
            yaw_radians=-0.1,
        )
        document = _prepared_document(
            manifest_uri=manifest_uri,
            input_uri=input_uri,
            input_sha256=input_sha256,
            context=context,
            prescribed=prescribed,
            alternative=alternative,
            run_id=run_id,
        )
        _write_json(root / _uri_filename(manifest_uri), document)
        _publish_directory(root, manifest_uri, _storage())
    return document


def _require_prepared(document: Mapping[str, Any]) -> None:
    """Fail closed unless a manifest is a LingBot preparation artifact."""

    if document.get("schema") != SCHEMA or document.get("stage") != "prepare":
        raise LingBotWorldStageError("Expected a LingBot preparation manifest")
    controls = document.get("controls")
    if not isinstance(controls, dict) or set(controls) != {"prescribed", "alternative"}:
        raise LingBotWorldStageError("Preparation manifest lacks the matched controls")


def _materialize_controls(
    document: Mapping[str, Any], name: str, root: Path, storage: Any
) -> Path:
    """Download and checksum one named trajectory from preparation output."""

    controls = document["controls"]
    entry = controls.get(name) if isinstance(controls, dict) else None
    if not isinstance(entry, dict) or not isinstance(entry.get("uri"), str):
        raise LingBotWorldStageError(f"Preparation manifest lacks {name} controls")
    expected = entry.get("files")
    if not isinstance(expected, dict):
        raise LingBotWorldStageError(f"Preparation manifest lacks {name} hashes")
    target = root / name
    target.mkdir()
    for filename in ("poses.npy", "intrinsics.npy"):
        _download_file(
            entry["uri"].rstrip("/") + "/" + filename, target / filename, storage
        )
        if file_sha256(target / filename) != expected.get(filename):
            raise LingBotWorldStageError(
                f"Control checksum mismatch for {name}/{filename}"
            )
    return target


def _generation_document(
    *,
    prepared: Mapping[str, Any],
    prepared_manifest: Path,
    output_manifest_uri: str,
    role: str,
    controls: str,
    result: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Build one generation manifest that identifies its exact inputs and video."""

    root = _uri_parent(output_manifest_uri)
    return {
        "schema": SCHEMA,
        "stage": "generate",
        "role": role,
        "run_id": run_id,
        "prepared_manifest_sha256": file_sha256(prepared_manifest),
        "input": dict(prepared["input"]),
        "control": controls,
        "control_uri": root + "/controls",
        "video_uri": root + "/video.mp4",
        "upstream": _upstream_provenance(),
        "native_generation": dict(result),
        "not_claimed": ["robot_action_conditioning", "training", "real_time"],
    }


def _validate_generation(document: Mapping[str, Any], role: str) -> None:
    """Require an expected, fully identified native generation manifest."""

    if document.get("schema") != SCHEMA or document.get("stage") != "generate":
        raise LingBotWorldStageError("Expected a LingBot generation manifest")
    if document.get("role") != role or not isinstance(document.get("video_uri"), str):
        raise LingBotWorldStageError(f"Expected a {role} generation artifact")


def _verify_previous_video(
    document: Mapping[str, Any], root: Path, storage: Any
) -> None:
    """Consume and fully decode the earlier continuation before comparison."""

    video = _download_file(document["video_uri"], root / "previous.mp4", storage)
    metrics = _video_metrics(video)
    if metrics["frame_count"] != FRAME_COUNT:
        raise LingBotWorldStageError(
            "Earlier continuation has an unexpected frame count"
        )


def generate_continuation(
    prepared_manifest_uri: str,
    output_manifest_uri: str,
    prompt: str,
    seed: int,
    degree: int,
    controls: str,
    role: str,
    run_id: str,
    previous_manifest_uri: str = "",
) -> dict[str, Any]:
    """Run pinned native inference for one prepared camera trajectory.

    Args:
        prepared_manifest_uri: S3 preparation manifest from the first stage.
        output_manifest_uri: Run-scoped S3 generation-manifest destination.
        prompt: Text prompt passed directly to upstream ``generate.py``.
        seed: Nonnegative native generation seed.
        degree: Required FSDP/Ulysses GPU count (two or four).
        controls: ``prescribed`` or ``alternative`` trajectory selector.
        role: Artifact role used to bind later matched-control evaluation.
        run_id: Workflow run identity included in provenance.
        previous_manifest_uri: Earlier generation required by the alternative path.

    Returns:
        The published native-generation manifest.

    Raises:
        LingBotWorldStageError: Stage artifacts are incomplete or mismatched.
        RuntimeError: Native multi-GPU inference fails.
    """

    if controls not in {"prescribed", "alternative"} or role != controls:
        raise LingBotWorldStageError("Generation role must match a named control")
    _require_s3_uri(prepared_manifest_uri, "prepared_manifest_uri")
    _require_s3_uri(output_manifest_uri, "output_manifest_uri")
    if controls == "alternative" and not previous_manifest_uri:
        raise LingBotWorldStageError(
            "Alternative generation requires prescribed output"
        )
    if previous_manifest_uri:
        _require_s3_uri(previous_manifest_uri, "previous_manifest_uri")
    with tempfile.TemporaryDirectory(prefix="npa-lingbot-generate-") as temporary:
        root = Path(temporary)
        storage = _storage()
        prepared_path = root / "prepared.json"
        prepared = _read_json(prepared_manifest_uri, prepared_path, storage)
        _require_prepared(prepared)
        if previous_manifest_uri:
            previous = _read_json(
                previous_manifest_uri, root / "previous.json", storage
            )
            _validate_generation(previous, "prescribed")
            if previous.get("input") != prepared.get("input"):
                raise LingBotWorldStageError(
                    "Alternative generation context differs from prescribed run"
                )
            _verify_previous_video(previous, root, storage)
        context = _download_file(
            prepared["input"]["context_uri"], root / "context.png", storage
        )
        if file_sha256(context) != prepared["input"].get("context_sha256"):
            raise LingBotWorldStageError("Prepared context checksum mismatch")
        control_dir = _materialize_controls(prepared, controls, root, storage)
        output = root / "generation"
        result = generate_camera_video(
            context, prompt, seed, output, degree, control_dir
        )
        document = _generation_document(
            prepared=prepared,
            prepared_manifest=prepared_path,
            output_manifest_uri=output_manifest_uri,
            role=role,
            controls=controls,
            result=result,
            run_id=run_id,
        )
        _write_json(output / _uri_filename(output_manifest_uri), document)
        _publish_directory(output, output_manifest_uri, storage)
    return document


def _video_metrics(path: Path) -> dict[str, Any]:
    """Decode every video frame and report actual frame and pixel statistics."""

    import av
    import numpy as np

    frame_count, spatial, delta, previous, shape = 0, 0.0, 0.0, None, None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate else 0.0
        for frame in container.decode(stream):
            pixels = frame.to_ndarray(format="rgb24")
            if shape is not None and pixels.shape != shape:
                raise LingBotWorldStageError("Video dimensions changed across frames")
            sample = pixels[::8, ::8].astype(np.float32)
            spatial = max(spatial, float(sample.std()))
            if previous is not None:
                delta += float(np.abs(sample - previous).mean())
            previous, shape = sample, pixels.shape
            frame_count += 1
    if frame_count != FRAME_COUNT or shape is None or spatial < 1 or delta <= 0.001:
        raise LingBotWorldStageError("Video is incomplete, blank, or stationary")
    return {
        "frame_count": frame_count,
        "height": int(shape[0]),
        "width": int(shape[1]),
        "fps": rate,
        "max_spatial_std": spatial,
        "mean_temporal_delta": delta / max(1, frame_count - 1),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _paired_video_delta(left: Path, right: Path) -> dict[str, float]:
    """Decode matched video pairs and calculate their actual RGB difference."""

    import av
    import numpy as np

    frames, absolute_total, maximum = 0, 0.0, 0.0
    with av.open(str(left)) as left_container, av.open(str(right)) as right_container:
        left_stream = left_container.streams.video[0]
        right_stream = right_container.streams.video[0]
        for left_frame, right_frame in zip(
            left_container.decode(left_stream),
            right_container.decode(right_stream),
            strict=True,
        ):
            left_pixels = left_frame.to_ndarray(format="rgb24")
            right_pixels = right_frame.to_ndarray(format="rgb24")
            if left_pixels.shape != right_pixels.shape:
                raise LingBotWorldStageError(
                    "Matched continuations have different dimensions"
                )
            difference = np.abs(
                left_pixels.astype(np.float32) - right_pixels.astype(np.float32)
            )
            absolute_total += float(difference.sum())
            maximum = max(maximum, float(difference.max()))
            frames += 1
    if frames != FRAME_COUNT:
        raise LingBotWorldStageError(
            "Matched continuation pair has an unexpected frame count"
        )
    pixels = FRAME_COUNT * left_pixels.shape[0] * left_pixels.shape[1] * 3
    return {
        "mean_absolute_rgb_delta": absolute_total / pixels,
        "max_absolute_rgb_delta": maximum,
    }


def _control_metrics(prescribed: Path, alternative: Path) -> dict[str, float]:
    """Measure actual divergence between the two camera-pose trajectories."""

    import numpy as np

    left = np.load(prescribed / "poses.npy", allow_pickle=False)
    right = np.load(alternative / "poses.npy", allow_pickle=False)
    if left.shape != (FRAME_COUNT, 4, 4) or right.shape != left.shape:
        raise LingBotWorldStageError("Camera controls do not match the native shape")
    translation = np.linalg.norm(left[:, :3, 3] - right[:, :3, 3], axis=1)
    relative = np.matmul(np.transpose(left[:, :3, :3], (0, 2, 1)), right[:, :3, :3])
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
    rotation = np.degrees(np.arccos(cosine))
    return {
        "translation_rms": float(np.sqrt(np.mean(np.square(translation)))),
        "translation_max": float(translation.max()),
        "rotation_degrees_mean": float(rotation.mean()),
        "rotation_degrees_max": float(rotation.max()),
    }


def evaluate_control_response(
    prescribed_manifest_uri: str,
    alternative_manifest_uri: str,
    report_uri: str,
    run_id: str,
) -> dict[str, Any]:
    """Decode both continuations and measure visual response to camera controls.

    Args:
        prescribed_manifest_uri: S3 manifest for the prescribed continuation.
        alternative_manifest_uri: S3 manifest for the matched alternative.
        report_uri: Run-scoped S3 destination for the measured report.
        run_id: Workflow run identity included in the report.

    Returns:
        The published decode and control-response report.

    Raises:
        LingBotWorldStageError: Inputs are unmatched, corrupt, or visually identical.
    """

    _require_s3_uri(prescribed_manifest_uri, "prescribed_manifest_uri")
    _require_s3_uri(alternative_manifest_uri, "alternative_manifest_uri")
    _require_s3_uri(report_uri, "report_uri")
    with tempfile.TemporaryDirectory(prefix="npa-lingbot-evaluate-") as temporary:
        root = Path(temporary)
        storage = _storage()
        prescribed = _read_json(
            prescribed_manifest_uri, root / "prescribed.json", storage
        )
        alternative = _read_json(
            alternative_manifest_uri, root / "alternative.json", storage
        )
        _validate_generation(prescribed, "prescribed")
        _validate_generation(alternative, "alternative")
        _require_matched_generations(prescribed, alternative)
        prescribed_video = _download_file(
            prescribed["video_uri"], root / "prescribed.mp4", storage
        )
        alternative_video = _download_file(
            alternative["video_uri"], root / "alternative.mp4", storage
        )
        prescribed_controls = _materialize_generation_controls(
            prescribed, root, storage
        )
        alternative_controls = _materialize_generation_controls(
            alternative, root, storage
        )
        response = _paired_video_delta(prescribed_video, alternative_video)
        if response["mean_absolute_rgb_delta"] <= 0.001:
            raise LingBotWorldStageError(
                "Different camera controls produced no measured visual response"
            )
        report = _evaluation_document(
            prescribed,
            alternative,
            root / "prescribed.json",
            root / "alternative.json",
            prescribed_video,
            alternative_video,
            prescribed_controls,
            alternative_controls,
            response,
            run_id,
        )
        _write_json(root / _uri_filename(report_uri), report)
        _publish_directory(root, report_uri, storage)
    return report


def _require_matched_generations(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> None:
    """Reject continuations that do not share the prepared context and model."""

    for key in ("prepared_manifest_sha256", "input", "upstream"):
        if left.get(key) != right.get(key):
            raise LingBotWorldStageError(f"Matched continuation field differs: {key}")


def _materialize_generation_controls(
    document: Mapping[str, Any], root: Path, storage: Any
) -> Path:
    """Download a generated continuation's exact camera-control files."""

    role = str(document["role"])
    target = root / f"{role}-controls"
    target.mkdir()
    for filename in ("poses.npy", "intrinsics.npy"):
        _download_file(
            document["control_uri"] + "/" + filename, target / filename, storage
        )
    return target


def _evaluation_document(
    prescribed: Mapping[str, Any],
    alternative: Mapping[str, Any],
    prescribed_manifest: Path,
    alternative_manifest: Path,
    prescribed_video: Path,
    alternative_video: Path,
    prescribed_controls: Path,
    alternative_controls: Path,
    response: Mapping[str, float],
    run_id: str,
) -> dict[str, Any]:
    """Build the factual paired-control report consumed by visualization."""

    return {
        "schema": SCHEMA,
        "stage": "evaluate",
        "run_id": run_id,
        "upstream": _upstream_provenance(),
        "prescribed_manifest_sha256": file_sha256(prescribed_manifest),
        "alternative_manifest_sha256": file_sha256(alternative_manifest),
        "baseline_video_uri": prescribed["video_uri"],
        "alternative_video_uri": alternative["video_uri"],
        "decode": {
            "prescribed": _video_metrics(prescribed_video),
            "alternative": _video_metrics(alternative_video),
        },
        "control_difference": _control_metrics(
            prescribed_controls, alternative_controls
        ),
        "visual_response": dict(response),
        "interpretation": "Measured pixel response to different authored camera poses; not calibrated camera-pose accuracy or robot-action dynamics.",
        "not_claimed": [
            "robot_action_conditioning",
            "camera_calibration_accuracy",
            "training",
        ],
    }


def _comparison_frames(left: Path, right: Path):
    """Yield synchronized RGB frame pairs from two already-validated MP4 files."""

    import av

    with av.open(str(left)) as left_container, av.open(str(right)) as right_container:
        left_stream = left_container.streams.video[0]
        right_stream = right_container.streams.video[0]
        for left_frame, right_frame in zip(
            left_container.decode(left_stream),
            right_container.decode(right_stream),
            strict=True,
        ):
            yield (
                left_frame.to_ndarray(format="rgb24"),
                right_frame.to_ndarray(format="rgb24"),
            )


def _write_comparison_mp4(
    left: Path, right: Path, destination: Path, fps: float
) -> dict[str, Any]:
    """Encode every synchronized frame pair into a side-by-side MP4."""

    import av
    import numpy as np

    first_left, first_right = next(_comparison_frames(left, right))
    if first_left.shape != first_right.shape:
        raise LingBotWorldStageError(
            "Cannot visualize videos with different dimensions"
        )
    height, width, _ = first_left.shape
    output = av.open(str(destination), "w")
    stream = output.add_stream("libx264", rate=max(1, round(fps)))
    stream.width, stream.height, stream.pix_fmt = width * 2, height, "yuv420p"
    frames = 0
    try:
        for prescribed, alternative in _comparison_frames(left, right):
            combined = np.concatenate((prescribed, alternative), axis=1)
            encoded = stream.encode(
                av.VideoFrame.from_ndarray(combined, format="rgb24")
            )
            for packet in encoded:
                output.mux(packet)
            frames += 1
        for packet in stream.encode():
            output.mux(packet)
    finally:
        output.close()
    if frames != FRAME_COUNT:
        raise LingBotWorldStageError("Comparison MP4 did not contain every frame")
    return _video_metrics(destination)


def _review_image(image: Any) -> Any:
    """Downsample an RGB frame for a complete but practical Rerun recording."""

    from PIL import Image

    height, width = image.shape[:2]
    target_width = min(width, 320)
    target_height = round(height * target_width / width)
    return __import__("numpy").asarray(
        Image.fromarray(image).resize(
            (target_width, target_height), Image.Resampling.BILINEAR
        )
    )


def _set_rerun_time(rr: Any, recording: Any, seconds: float) -> None:
    """Set a Rerun time using the installed SDK's compatible method."""

    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("video_time", seconds, recording=recording)
    else:
        rr.set_time("video_time", duration=seconds, recording=recording)


def _write_rrd(
    left: Path, right: Path, report: Mapping[str, Any], destination: Path, run_id: str
) -> dict[str, Any]:
    """Write and independently inspect a synchronized camera-comparison RRD."""

    import rerun as rr

    recording = rr.RecordingStream(
        "npa_lingbot_world_controlled_continuation", recording_id=run_id
    )
    recording.save(str(destination))
    recording.log(
        "provenance", rr.TextDocument(json.dumps(report, indent=2)), static=True
    )
    fps = float(report["decode"]["prescribed"]["fps"] or 16.0)
    for index, (prescribed, alternative) in enumerate(_comparison_frames(left, right)):
        _set_rerun_time(rr, recording, index / fps)
        recording.log(
            "camera/prescribed", rr.Image(_review_image(prescribed), color_model="RGB")
        )
        recording.log(
            "camera/alternative",
            rr.Image(_review_image(alternative), color_model="RGB"),
        )
        recording.log(
            "metrics/mean_absolute_rgb_delta",
            rr.Scalars(report["visual_response"]["mean_absolute_rgb_delta"]),
        )
    recording.flush()
    recording.disconnect()
    _inspect_rrd(destination, run_id)
    return {
        "sha256": file_sha256(destination),
        "size_bytes": destination.stat().st_size,
        "frames": FRAME_COUNT,
    }


def _inspect_rrd(path: Path, run_id: str) -> None:
    """Require the installed Rerun reader to decode the emitted recording."""

    binary = Path(sys.executable).parent / "rerun"
    subprocess.run(
        [str(binary), "rrd", "verify", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    printed = subprocess.run(
        [str(binary), "rrd", "print", "-vv", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    required = (
        "npa_lingbot_world_controlled_continuation",
        run_id,
        "camera/prescribed",
        "camera/alternative",
    )
    if not all(value in printed.stdout for value in required):
        raise LingBotWorldStageError(
            "Rerun inspection did not find required synchronized entities"
        )


def visualize_comparison(
    prescribed_manifest_uri: str,
    alternative_manifest_uri: str,
    report_uri: str,
    visualization_manifest_uri: str,
    run_id: str,
) -> dict[str, Any]:
    """Emit synchronized comparison MP4, RRD, and provenance manifest.

    Args:
        prescribed_manifest_uri: S3 manifest for the prescribed continuation.
        alternative_manifest_uri: S3 manifest for the alternative continuation.
        report_uri: S3 decode/control-response report from evaluation.
        visualization_manifest_uri: Run-scoped S3 destination for visualization metadata.
        run_id: Workflow run identity included in the Rerun recording.

    Returns:
        Visualization manifest with independently verified MP4 and RRD identities.

    Raises:
        LingBotWorldStageError: Earlier artifacts are absent or visualization fails.
    """

    _require_s3_uri(prescribed_manifest_uri, "prescribed_manifest_uri")
    _require_s3_uri(alternative_manifest_uri, "alternative_manifest_uri")
    _require_s3_uri(report_uri, "report_uri")
    _require_s3_uri(visualization_manifest_uri, "visualization_manifest_uri")
    with tempfile.TemporaryDirectory(prefix="npa-lingbot-visualize-") as temporary:
        root = Path(temporary) / "reports"
        root.mkdir()
        storage = _storage()
        prescribed = _read_json(
            prescribed_manifest_uri, root / "prescribed.json", storage
        )
        alternative = _read_json(
            alternative_manifest_uri, root / "alternative.json", storage
        )
        report = _read_json(report_uri, root / "control-response.json", storage)
        _validate_generation(prescribed, "prescribed")
        _validate_generation(alternative, "alternative")
        if report.get("stage") != "evaluate":
            raise LingBotWorldStageError(
                "Visualization requires a measured evaluation report"
            )
        left = _download_file(prescribed["video_uri"], root / "prescribed.mp4", storage)
        right = _download_file(
            alternative["video_uri"], root / "alternative.mp4", storage
        )
        mp4 = _write_comparison_mp4(
            left, right, root / "comparison.mp4", report["decode"]["prescribed"]["fps"]
        )
        rrd = _write_rrd(left, right, report, root / "comparison.rrd", run_id)
        manifest = _visualization_document(
            visualization_manifest_uri, report, mp4, rrd, run_id
        )
        _write_json(root / _uri_filename(visualization_manifest_uri), manifest)
        _publish_directory(root, visualization_manifest_uri, storage)
    return manifest


def _visualization_document(
    manifest_uri: str,
    report: Mapping[str, Any],
    mp4: Mapping[str, Any],
    rrd: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Build durable visualization provenance without inventing performance claims."""

    root = _uri_parent(manifest_uri)
    return {
        "schema": SCHEMA,
        "stage": "visualize",
        "run_id": run_id,
        "upstream": _upstream_provenance(),
        "evaluation": dict(report),
        "comparison_mp4": {"uri": root + "/comparison.mp4", **dict(mp4)},
        "comparison_rrd": {"uri": root + "/comparison.rrd", **dict(rrd)},
        "synchronization": "Frame-index and generated-video time; not physical camera timestamps.",
        "not_claimed": [
            "robot_action_conditioning",
            "camera_calibration_accuracy",
            "training",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    """Create the stage-command parser used by the workflow's real shell steps."""

    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare")
    prepare.add_argument("--input-uri", required=True)
    prepare.add_argument("--input-sha256", required=True)
    prepare.add_argument("--manifest-uri", required=True)
    prepare.add_argument("--run-id", required=True)
    generate = subcommands.add_parser("generate")
    for name in ("prepared-manifest-uri", "manifest-uri", "prompt", "run-id"):
        generate.add_argument("--" + name, required=True)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--degree", type=int, required=True)
    generate.add_argument(
        "--controls", choices=("prescribed", "alternative"), required=True
    )
    generate.add_argument("--previous-manifest-uri", default="")
    evaluate = subcommands.add_parser("evaluate")
    evaluate.add_argument("--prescribed-manifest-uri", required=True)
    evaluate.add_argument("--alternative-manifest-uri", required=True)
    evaluate.add_argument("--report-uri", required=True)
    evaluate.add_argument("--run-id", required=True)
    visualize = subcommands.add_parser("visualize")
    visualize.add_argument("--prescribed-manifest-uri", required=True)
    visualize.add_argument("--alternative-manifest-uri", required=True)
    visualize.add_argument("--report-uri", required=True)
    visualize.add_argument("--manifest-uri", required=True)
    visualize.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one durable LingBot workflow stage from an explicit argument list.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Zero when the selected stage published its declared artifact.

    Raises:
        LingBotWorldStageError: A selected stage's artifact contract is violated.
    """

    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_context(
            args.input_uri, args.input_sha256, args.manifest_uri, args.run_id
        )
    elif args.command == "generate":
        result = generate_continuation(
            args.prepared_manifest_uri,
            args.manifest_uri,
            args.prompt,
            args.seed,
            args.degree,
            args.controls,
            args.controls,
            args.run_id,
            args.previous_manifest_uri,
        )
    elif args.command == "evaluate":
        result = evaluate_control_response(
            args.prescribed_manifest_uri,
            args.alternative_manifest_uri,
            args.report_uri,
            args.run_id,
        )
    else:
        result = visualize_comparison(
            args.prescribed_manifest_uri,
            args.alternative_manifest_uri,
            args.report_uri,
            args.manifest_uri,
            args.run_id,
        )
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "run_id": result["run_id"],
                "status": "published",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
