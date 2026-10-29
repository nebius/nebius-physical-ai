"""Canonical Stage 3 Cosmos3-Nano structural video-transfer adapter.

The adapter intentionally owns only Sim2Real's source-frame selection and
publication boundary.  Model execution remains in
``npa.workbench.cosmos.generate`` so this workflow cannot degrade into a
descriptor generator or an unrelated text-to-video invocation.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.models import Sim2RealLoopError


_FRAME_NAME = re.compile(r"^(?:camera|frame)-(?P<index>[0-9]+)\.png$")
_FPS = 24
_FRAME_GLOB = "frame-*.png"


def _frame_index(path: Path) -> int | None:
    """Return the strict primary-frame ordinal embedded in ``path``."""

    match = _FRAME_NAME.fullmatch(path.name)
    return int(match.group("index")) if match else None


def select_source_frames(root: Path) -> list[Path]:
    """Select one complete primary-camera sequence without truncation.

    The workflow consumes the ordinary ``camera-N.png`` naming first.  The
    seed-dataset ``frame-N.png`` form is accepted only when camera frames are
    absent.  Other views deliberately remain outside this single-video transfer
    rather than being interleaved into an incoherent temporal sequence.
    """

    candidates = [
        path for path in root.rglob("*.png") if _frame_index(path) is not None
    ]
    camera = [path for path in candidates if path.name.startswith("camera-")]
    frames = camera or [path for path in candidates if path.name.startswith("frame-")]
    ordered = sorted(frames, key=lambda path: (_frame_index(path), str(path)))
    indices = [_frame_index(path) for path in ordered]
    if len(ordered) < 6 or len(set(indices)) != len(indices):
        raise Sim2RealLoopError(
            "Stage 3 requires one complete, uniquely indexed primary-camera sequence "
            "with at least six frames"
        )
    if any(
        path.is_symlink() or not path.is_file() or path.stat().st_size <= 0
        for path in ordered
    ):
        raise Sim2RealLoopError("Stage 3 source sequence contains an invalid frame")
    return ordered


def _sha256(path: Path) -> str:
    """Return the byte digest for a regular local artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_digest(frames: list[Path]) -> str:
    """Bind ordering and bytes of the complete selected source sequence."""

    material = _source_frame_records(frames)
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def _source_frame_records(frames: list[Path]) -> list[dict[str, Any]]:
    """Return the ordered original-frame identity retained with the transfer."""

    return [
        {"ordinal": ordinal, "name": path.name, "sha256": _sha256(path)}
        for ordinal, path in enumerate(frames)
    ]


def encode_source_video(frames: list[Path], destination: Path) -> None:
    """Encode every selected primary frame into an exact ordered source video."""

    sequence = destination.parent / "sequence"
    sequence.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        shutil.copy2(frame, sequence / f"frame-{index:05d}.png")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-xerror",
            "-framerate",
            str(_FPS),
            "-i",
            str(sequence / "frame-%05d.png"),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        check=True,
    )


def decode_generated_frames(
    video: Path, destination: Path, expected: int
) -> list[Path]:
    """Fully decode the generated video into the complete EnvGen frame handoff."""

    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(video),
            "-vsync",
            "0",
            str(destination / "frame-%05d.png"),
        ],
        check=True,
    )
    frames = sorted(destination.glob(_FRAME_GLOB))
    if len(frames) != expected or any(path.stat().st_size <= 0 for path in frames):
        raise Sim2RealLoopError("Stage 3 generated-frame decode is incomplete")
    return frames


def _transfer_prompt() -> str:
    """Return the explicit invariant-preservation contract for Cosmos3-Nano."""

    return (
        "Transfer the complete robot manipulation video with physically plausible "
        "appearance variation. Preserve source motion, robot and object identity, "
        "contact geometry, camera geometry, topology, and temporal ordering in every "
        "frame. Do not add or remove objects, contacts, hands, or actions."
    )


def _assert_guarded_transfer(result: dict[str, Any]) -> dict[str, Any]:
    """Accept only an effective guarded structural-transfer execution receipt."""

    state = dict(result.get("guardrail_state") or {})
    transfer = dict(result.get("structural_transfer") or {})
    required = (
        "text_guardrail_passed",
        "video_guardrail_passed",
        "guardrail_postprocessing_applied",
    )
    if state.get("status") != "passed" or state.get("effective") is not True:
        raise Sim2RealLoopError("Stage 3 Cosmos3 guardrails were not effective")
    if not all(transfer.get(key) is True for key in required):
        raise Sim2RealLoopError("Stage 3 lacks guarded structural-transfer evidence")
    if not Path(str(transfer.get("control_path") or "")).is_file():
        raise Sim2RealLoopError("Stage 3 lacks its native edge-control artifact")
    return transfer


def _upload_media(client: Any, local: Path, uri: str) -> dict[str, Any]:
    """Publish a local media artifact with immutable byte evidence."""

    client.upload_file(str(local), uri)
    return {"uri": uri, "sha256": _sha256(local), "bytes": local.stat().st_size}


def _upload_frames(client: Any, frames: list[Path], root: str) -> list[dict[str, str]]:
    """Publish every decoded generated frame and return ordered downstream URIs."""

    records = []
    for index, frame in enumerate(frames):
        uri = f"{root}frame-{index:05d}.png"
        client.upload_file(str(frame), uri)
        records.append({"frame_id": f"frame-{index:05d}", "uri": uri})
    return records


def _manifest(
    *,
    input_uri: str,
    source_count: int,
    sequence_sha256: str,
    source_frames: list[dict[str, Any]],
    source: dict[str, Any],
    prepared: dict[str, Any],
    generated: dict[str, Any],
    control: dict[str, Any],
    alignment: dict[str, Any],
    result: dict[str, Any],
    frames: list[dict[str, str]],
) -> dict[str, Any]:
    """Build the explicit Stage 3 provenance and EnvGen-consumption contract."""

    return {
        "schema": "npa.cosmos3.structural-transfer.v1",
        "status": "executed",
        "mode": "cosmos3_nano_video2video_edge_gpu",
        "model": "Cosmos3-Nano",
        "input_conditioned": True,
        "input_conditioning": "complete_primary_camera_video_edge",
        "source_dataset_uri": input_uri,
        "source_sequence": {
            "selected_frame_count": source_count,
            "ordered_frame_sha256": sequence_sha256,
            "frames": source_frames,
            "video": source,
        },
        "prepared_reference": prepared,
        "augmented_video": generated,
        "edge_control": control,
        "guardrail_state": result["guardrail_state"],
        "structural_transfer": {
            key: value
            for key, value in result["structural_transfer"].items()
            if key != "control_path"
        },
        "temporal_alignment": alignment,
        "full_source_coverage": {
            "selected_source_frames": source_count,
            "prepared_frames": alignment["decoded_frames"],
            "generated_frames": alignment["decoded_frames"],
            "decoded_envgen_frames": len(frames),
            "complete": True,
        },
        "frame_count": len(frames),
        "frames": frames,
        "downstream_consumers": [
            "stage_04_envgen_raw",
            "stage_05_split_scenario_lineage",
            "stage_08_rollout_context",
        ],
        "state_policy_pixel_contract": "lineage_and_auxiliary_vlm_only; PPO observations are simulator state",
        "weights_baked": False,
        "native_model_selection": result.get("native_model_selection"),
        "input_spec": result["input_spec"],
    }


def run_cosmos3_structural_augmentation(
    *, input_uri: str, root_uri: str, run_id: str, client: Any | None = None
) -> dict[str, Any]:
    """Execute real complete-source Cosmos3-Nano edge transfer and publish it.

    Args:
        input_uri: Exact task seed S3 prefix containing one primary frame sequence.
        root_uri: Canonical run S3 root.
        run_id: Workflow run identity forwarded to model provenance.
        client: Optional storage seam for focused tests.
    Returns:
        Published Stage 3 result containing the canonical manifest.
    Raises:
        Sim2RealLoopError: Source, guardrails, decoded coverage, or publication fails.
    """

    from npa.workbench.cosmos.generate import run_cosmos3_generate
    from npa.workbench.cosmos.structural_transfer import TransferSettings
    from npa.workflows.paidf_cosmos3_media import (
        prepare_reference,
        probe_video,
        verify_pair,
    )
    from npa.workflows.sim2real.workflow_io import storage, write_json

    store = client or storage()
    root = root_uri.rstrip("/")
    with tempfile.TemporaryDirectory(prefix="npa-s2r-cosmos3-stage-03-") as directory:
        work = Path(directory)
        incoming = work / "source"
        store.download_directory(input_uri.rstrip("/") + "/", str(incoming))
        source_frames = select_source_frames(incoming)
        source = work / "source.mp4"
        encode_source_video(source_frames, source)
        if probe_video(source)["decoded_frames"] != len(source_frames):
            raise Sim2RealLoopError(
                "Stage 3 source video omitted a selected input frame"
            )
        prepared = work / "prepared.mp4"
        normalization = prepare_reference(source, prepared, _FPS)
        result = run_cosmos3_generate(
            mode="video2video",
            checkpoint="Cosmos3-Nano",
            name="sim2real-stage-03",
            prompt=_transfer_prompt(),
            vision_path=str(prepared),
            output_dir=work / "generated",
            seed=42,
            guidance=3.0,
            num_steps=35,
            no_guardrails=False,
            transfer=TransferSettings(fps=_FPS, chunk_frames=93, control_guidance=1.5),
        )
        transfer = _assert_guarded_transfer(result)
        generated_path = Path(str(result["output_path"]))
        alignment = verify_pair(prepared, generated_path, _FPS)
        frames = decode_generated_frames(
            generated_path, work / "frames", alignment["decoded_frames"]
        )
        prefix = f"{root}/augment/"
        source_record = _upload_media(store, source, prefix + "source.mp4")
        prepared_record = _upload_media(store, prepared, prefix + "prepared-source.mp4")
        generated_record = _upload_media(
            store, generated_path, prefix + "augmented-video.mp4"
        )
        control_record = _upload_media(
            store, Path(transfer["control_path"]), prefix + "source-edges.mkv"
        )
        frame_records = _upload_frames(store, frames, prefix + "frames/")
        manifest = _manifest(
            input_uri=input_uri,
            source_count=len(source_frames),
            sequence_sha256=_sequence_digest(source_frames),
            source_frames=_source_frame_records(source_frames),
            source=source_record,
            prepared={**normalization, "uri": prepared_record["uri"]},
            generated=generated_record,
            control=control_record,
            alignment=alignment,
            result=result,
            frames=frame_records,
        )
        write_json(
            prefix + "frames/index.json",
            {
                "schema": "npa.sim2real.augmented_frames.v1",
                "frame_count": len(frame_records),
                "frames": frame_records,
            },
            directory=work / "index",
        )
        write_json(prefix + "manifest.json", manifest, directory=work / "manifest")
        result_uri = prefix + "cosmos3-transfer-result.json"
        write_json(
            result_uri,
            {
                "schema": "npa.sim2real.cosmos3_stage_result.v1",
                "run_id": run_id,
                "manifest": manifest,
            },
            directory=work / "result",
        )
        return {
            "manifest": manifest,
            "result_uri": result_uri,
            "frames_uri": prefix + "frames/",
        }
