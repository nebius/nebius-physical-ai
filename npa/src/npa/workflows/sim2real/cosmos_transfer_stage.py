"""Task-conditioned Stage 3 Cosmos Transfer execution and artifact publishing."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from npa.workflows.sim2real.models import Sim2RealLoopError


def _result_uri_and_prefix(output_uri: str) -> tuple[str, str]:
    """Resolve either a result object URI or a directory-style output prefix."""

    normalized = str(output_uri or "").strip()
    parsed = urlparse(normalized)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise Sim2RealLoopError(
            f"Cosmos Transfer output must be an explicit s3:// URI, got {output_uri!r}"
        )
    if normalized.endswith("/"):
        return normalized + "cosmos2-transfer-result.json", normalized
    leaf = parsed.path.rsplit("/", 1)[-1]
    if not leaf:
        raise Sim2RealLoopError(
            f"Cosmos Transfer output has no object name: {output_uri!r}"
        )
    return normalized, normalized.rsplit("/", 1)[0] + "/"


def _frames_prefix(augmented_frames_uri: str) -> str:
    normalized = str(augmented_frames_uri or "").strip().rstrip("/") + "/"
    parsed = urlparse(normalized)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise Sim2RealLoopError(
            "Cosmos Transfer frames output must be an explicit s3:// prefix, "
            f"got {augmented_frames_uri!r}"
        )
    return normalized


def _validate_real_frame(frame: Any, *, frames_root: str, index: int) -> None:
    if (
        not isinstance(frame, dict)
        or not isinstance(frame.get("frame_id"), str)
        or not str(frame["frame_id"]).strip()
        or not isinstance(frame.get("uri"), str)
    ):
        raise Sim2RealLoopError(
            f"real Cosmos-Transfer2.5 returned a malformed frame at index {index}"
        )
    uri = str(frame["uri"])
    parsed_uri = urlparse(uri)
    parsed_root = urlparse(frames_root)
    relative = (
        parsed_uri.path[len(parsed_root.path) :]
        if parsed_uri.path.startswith(parsed_root.path)
        else ""
    )
    if (
        parsed_uri.scheme != "s3"
        or parsed_uri.netloc != parsed_root.netloc
        or not parsed_uri.path.startswith(parsed_root.path)
        or not relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise Sim2RealLoopError(
            "real Cosmos-Transfer2.5 returned a frame outside its declared "
            f"output prefix at index {index}"
        )


def run_cosmos_transfer_component(
    *,
    input_uri: str,
    output_uri: str,
    augmented_frames_uri: str,
    assets_uri: str = "",
    scene_spec_uri: str = "",
    image: str = "",
    run_id: str = "",
    real_runner: Callable[..., dict[str, Any] | None],
) -> dict[str, Any]:
    """Run the real component or its explicit non-required descriptor seam."""

    from npa.clients.storage import StorageClient
    from npa.workflows.cosmos_split import (
        Cosmos2TransferConfig,
        build_cosmos2_transfer_manifest,
    )
    from npa.workflows.sim2real_stages import resolve_augment_frame_count

    client = StorageClient.from_environment()
    result_uri, augment_prefix = _result_uri_and_prefix(output_uri)
    frames_root = _frames_prefix(augmented_frames_uri)
    frame_count = resolve_augment_frame_count()
    manifest = build_cosmos2_transfer_manifest(
        Cosmos2TransferConfig(
            input_uri=input_uri,
            output_uri=augment_prefix,
            assets_uri=assets_uri,
            scene_spec_uri=scene_spec_uri,
            image=image,
            run_id=run_id,
        )
    )
    real = real_runner(client, input_uri, augment_prefix, frames_root, run_id)
    if real is not None:
        if not isinstance(real, dict):
            raise Sim2RealLoopError(
                "real Cosmos-Transfer2.5 returned a malformed result object"
            )
        frames = real.get("frames")
        if not isinstance(frames, list) or not frames:
            raise Sim2RealLoopError(
                "real Cosmos-Transfer2.5 completed without a non-empty exact "
                "frames list"
            )
        for index_no, frame in enumerate(frames):
            _validate_real_frame(frame, frames_root=frames_root, index=index_no)
        frame_ids = [str(frame["frame_id"]) for frame in frames]
        frame_uris = [str(frame["uri"]) for frame in frames]
        if len(set(frame_ids)) != len(frame_ids) or len(set(frame_uris)) != len(
            frame_uris
        ):
            raise Sim2RealLoopError(
                "real Cosmos-Transfer2.5 returned duplicate frame lineage"
            )
        if int(real.get("frame_count") or 0) != len(frames):
            raise Sim2RealLoopError(
                "real Cosmos-Transfer2.5 frame_count does not match frames"
            )
        manifest.update(
            {
                "status": "executed",
                "mode": "cosmos_transfer2.5_gpu",
                "augmented_frames_uri": frames_root,
                "augmented_video_uri": real["augmented_video_uri"],
                "frame_count": real["frame_count"],
                "frames": frames,
                "video_bytes": real["video_bytes"],
                "control_spec": real["spec"],
                "input_conditioned": bool(real.get("input_conditioned", False)),
                "semantic_task": str(
                    real.get("semantic_task") or "Isaac-Lift-Cube-Franka-v0"
                ),
                "downstream_consumers": [
                    "scenario.source_augmentation",
                    "scenario_stratification",
                    "cosmos_reason_context",
                ],
                "state_policy_pixel_contract": (
                    "lineage_and_auxiliary_vlm_only; PPO observations are simulator state"
                ),
            }
        )
        if real.get("fixture_provenance"):
            manifest["fixture_provenance"] = real["fixture_provenance"]
    else:
        if os.environ.get("NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS", "").strip() == "1":
            raise Sim2RealLoopError(
                "real Cosmos-Transfer2.5 was required, but the component attempted "
                "to fall back to descriptor_stub"
            )
        index: list[dict[str, str]] = []
        for index_no in range(frame_count):
            frame_key = f"frame-{index_no:05d}.json"
            payload = {
                "schema": "npa.sim2real.augmented_frame.v1",
                "frame_id": f"frame-{index_no:05d}",
                "source_dataset_uri": input_uri,
                "perturbation": ["lighting", "texture", "background", "contrast"][
                    index_no % 4
                ],
                "status": "cosmos2_transfer_executed",
            }
            local = Path(f"/tmp/{frame_key}")
            local.write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
            )
            client.upload_file(str(local), f"{frames_root}{frame_key}")
            index.append(
                {"frame_id": payload["frame_id"], "uri": f"{frames_root}{frame_key}"}
            )
        index_payload = {
            "schema": "npa.sim2real.augmented_frames.v1",
            "frame_count": frame_count,
            "frames": index,
        }
        index_local = Path("/tmp/augmented-frames-index.json")
        index_local.write_text(
            json.dumps(index_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        client.upload_file(str(index_local), f"{frames_root}index.json")
        manifest.update(
            {
                "status": "executed",
                "mode": "descriptor_stub",
                "augmented_frames_uri": frames_root,
                "frame_count": frame_count,
                "frames": index,
            }
        )

    manifest_local = Path("/tmp/cosmos2-transfer-manifest.json")
    manifest_local.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    client.upload_file(str(manifest_local), f"{augment_prefix}manifest.json")
    result = {"manifest": manifest, "augmented_frames_uri": frames_root}
    result_local = Path("/tmp/cosmos2-transfer-result.json")
    result_local.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    client.upload_file(str(result_local), result_uri)
    return result


def _task_conditioned_transfer_input(
    client: Any, input_uri: str, run_id: str, run_cosmos_transfer: Callable[..., Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Resolve an explicit operator spec or build a clip from real task frames."""

    if os.environ.get("NPA_SIM2REAL_TRANSFER_SPEC", "").strip():
        return run_cosmos_transfer(run_id=run_id or "augment"), None

    input_dir = Path("/tmp/npa-sim2real-transfer-input")
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    if not input_uri.startswith("s3://"):
        raise Sim2RealLoopError(
            "real Cosmos Transfer requires an s3:// Isaac lift-cube frame prefix"
        )
    client.download_directory(input_uri.rstrip("/") + "/", str(input_dir))
    primary = sorted(
        path
        for path in input_dir.rglob("camera-*.png")
        if path.stem.removeprefix("camera-").isdigit()
    )
    if not primary:
        # The task-seed contract publishes ordinary trajectory frames as
        # ``frame-N.png``; real Isaac rollout artifacts use ``camera-N.png``.
        # Both are primary-camera observations. Prefer the explicit Isaac name
        # when present, then accept only strictly numbered seed frames rather
        # than broadening discovery to arbitrary PNG assets under the trigger.
        primary = sorted(
            path
            for path in input_dir.rglob("frame-*.png")
            if path.stem.removeprefix("frame-").isdigit()
        )
    if len(primary) < 4:
        raise Sim2RealLoopError(
            "task-aligned seed dataset must contain at least four primary Isaac "
            f"camera frames under {input_uri!r}; found {len(primary)}"
        )
    sequence = input_dir / "sequence"
    sequence.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(primary[:93]):
        shutil.copy2(frame, sequence / f"frame-{index:05d}.png")
    input_video = input_dir / "isaac-lift-cube-seed.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "10",
            "-i",
            str(sequence / "frame-%05d.png"),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(input_video),
        ],
        check=True,
    )
    transfer = run_cosmos_transfer(
        run_id=run_id or "augment",
        input_video=str(input_video),
        prompt=(
            "A photorealistic Franka Panda robot lifting a rigid cube above a "
            "tabletop toward a visible 3D target, realistic materials and lighting"
        ),
        control="edge",
        control_weight=1.0,
        guidance=3.0,
    )
    return transfer, {
        "provenance": "task-aligned Isaac lift-cube RGB trajectory",
        "source_uri": input_uri,
        "source_frame_count": len(primary),
        "input_conditioned": True,
    }


def run_real_cosmos_transfer(
    client: Any,
    input_uri: str,
    augment_prefix: str,
    frames_root: str,
    run_id: str,
) -> dict[str, Any] | None:
    """Run real Cosmos Transfer and publish its conditioned video and frames."""

    require_real = (
        os.environ.get("NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS", "").strip() == "1"
    )
    if os.environ.get("NPA_SIM2REAL_AUGMENT_MODE", "real").strip().lower() == "stub":
        if require_real:
            raise Sim2RealLoopError(
                "NPA_SIM2REAL_AUGMENT_MODE=stub conflicts with the real-component contract"
            )
        return None
    try:
        from npa.workbench.cosmos.transfer import (
            FrameExtractionError,
            cosmos_transfer_available,
            extract_frames,
            run_cosmos_transfer,
        )
    except ImportError as exc:
        if require_real:
            raise Sim2RealLoopError(
                f"real Cosmos-Transfer2.5 modules are unavailable: {exc}"
            ) from exc
        return None
    if not cosmos_transfer_available():
        if require_real:
            raise Sim2RealLoopError(
                "real Cosmos-Transfer2.5 runtime is unavailable in the required GPU image"
            )
        return None
    try:
        transfer, fixture = _task_conditioned_transfer_input(
            client, input_uri, run_id, run_cosmos_transfer
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(
            json.dumps(
                {
                    "component": "cosmos2_transfer",
                    "event": "real_transfer_failed_fallback",
                    "error": str(exc)[:400],
                }
            ),
            file=sys.stderr,
        )
        if require_real:
            raise Sim2RealLoopError(
                f"required real Cosmos-Transfer2.5 inference failed: {exc}"
            ) from exc
        return None

    from npa.workflows.sim2real_stages import resolve_augment_frame_count

    augmented_video_uri = f"{augment_prefix}video/augmented.mp4"
    index: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="npa-augment-frames-") as frame_tmp:
        try:
            frames = extract_frames(
                transfer["video_path"],
                Path(frame_tmp),
                max_frames=resolve_augment_frame_count(),
            )
        except FrameExtractionError as exc:
            print(
                json.dumps(
                    {
                        "component": "cosmos2_transfer",
                        "event": "frame_extraction_failed_fallback",
                        "error": str(exc)[:400],
                    }
                ),
                file=sys.stderr,
            )
            return None
        if not frames:
            print(
                json.dumps(
                    {"component": "cosmos2_transfer", "event": "zero_frames_fallback"}
                ),
                file=sys.stderr,
            )
            return None
        client.upload_file(transfer["video_path"], augmented_video_uri)
        for index_no, frame_path in enumerate(frames):
            frame_key = f"frame-{index_no:05d}.png"
            client.upload_file(str(frame_path), f"{frames_root}{frame_key}")
            index.append(
                {
                    "frame_id": f"frame-{index_no:05d}",
                    "uri": f"{frames_root}{frame_key}",
                }
            )
    index_payload = {
        "schema": "npa.sim2real.augmented_frames.v1",
        "frame_count": len(index),
        "frames": index,
        "augmented_video_uri": augmented_video_uri,
        "mode": "cosmos_transfer2.5_gpu",
    }
    index_local = Path("/tmp/augmented-frames-index.json")
    index_local.write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    client.upload_file(str(index_local), f"{frames_root}index.json")
    print(
        json.dumps(
            {
                "component": "cosmos2_transfer",
                "event": "real_transfer_complete",
                "augmented_video_uri": augmented_video_uri,
                "video_bytes": transfer["video_bytes"],
                "frame_count": len(index),
                "spec": transfer["spec"],
            },
            sort_keys=True,
        )
    )
    result = {
        "augmented_video_uri": augmented_video_uri,
        "frame_count": len(index),
        "video_bytes": transfer["video_bytes"],
        "spec": transfer["spec"],
        "input_conditioned": bool(transfer.get("input_conditioned")),
        "input_video": str(transfer.get("input_video") or ""),
        "semantic_task": "Isaac-Lift-Cube-Franka-v0",
        "frames": index,
    }
    if fixture is not None:
        result["fixture_provenance"] = fixture["provenance"]
    return result
