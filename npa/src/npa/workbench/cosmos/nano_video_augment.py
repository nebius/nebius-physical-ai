"""Source-aligned Cosmos3-Nano edge transfer on the pinned vLLM-Omni service.

The CPU request/report contract imports neither Torch nor Ray. The serving-only
preprocessor calls the installed upstream edge helper, retaining lossless source
controls for every interval. RGB overlap supplies continuity, not source motion.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from .nano_video import (
    FPS,
    HEIGHT,
    MODEL_REVISION,
    WIDTH,
    DeviceMemorySampler,
    NanoVideoError,
    _positive_header,
    artifact,
    utc_now,
    write_json,
)

SCHEMA = "npa.cosmos3.nano-video.augmentation.v1"
OVERLAP_FRAMES = 5
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates images or videos following the "
    "user instructions and control signals. Follow the provided edge structure "
    "and motion while rendering the requested appearance."
)
DEFAULTS: dict[str, Any] = {
    "negative_prompt": "",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "num_inference_steps": 35,
    "guidance_scale": 3.0,
    "flow_shift": 10.0,
    "control_guidance": 1.5,
    "edge_threshold": "medium",
    "chunk_frames": 121,
    "max_sequence_length": 4096,
}
REQUIRED = {"mode", "request_id", "prompt", "seed", "source_sha256", "source_bytes"}


class AugmentationInputError(NanoVideoError, ValueError):
    """A source or supported parameter failed validation before GPU admission."""


def validate_request(payload: Any) -> dict[str, Any]:
    """Normalize only supported controls; reject dropped extras and coercions."""
    if not isinstance(payload, dict) or not REQUIRED <= payload.keys():
        raise AugmentationInputError("Missing required augmentation request fields")
    if payload.keys() - (REQUIRED | DEFAULTS.keys()):
        raise AugmentationInputError("Unknown augmentation request fields")
    request = {**DEFAULTS, **payload}
    if request["mode"] != "augmentation":
        raise AugmentationInputError("mode must be augmentation")
    if not isinstance(request["request_id"], str) or not SAFE_NAME.fullmatch(request["request_id"]):
        raise AugmentationInputError("Invalid augmentation request_id")
    for key in ("prompt", "negative_prompt", "system_prompt"):
        value = request[key]
        if not isinstance(value, str) or (key != "negative_prompt" and not value.strip()):
            raise AugmentationInputError(f"{key} must be text and required prompts must be nonempty")
        if key != "system_prompt":
            # The pinned pipeline strips these before tokenization even when
            # metadata templates are disabled; bind the canonical request to it.
            request[key] = value.strip()
    if not isinstance(request["source_sha256"], str) or not SHA256.fullmatch(request["source_sha256"]):
        raise AugmentationInputError("source_sha256 must be a lowercase SHA256 digest")
    for key in ("seed", "source_bytes", "num_inference_steps", "chunk_frames", "max_sequence_length"):
        if type(request[key]) is not int:
            raise AugmentationInputError(f"{key} must be an integer")
    if not 0 <= request["seed"] < 2**63:
        raise AugmentationInputError("seed must be a nonnegative signed 64-bit integer")
    if min(request[key] for key in ("source_bytes", "num_inference_steps", "max_sequence_length")) <= 0:
        raise AugmentationInputError("Byte count, steps and sequence length must be positive")
    if request["num_inference_steps"] > 200:
        raise AugmentationInputError("The installed video API supports at most 200 inference steps")
    frames = request["chunk_frames"]
    if not 9 <= frames <= 297 or (frames - 1) % 4:
        raise AugmentationInputError("chunk_frames must be 4k+1 between 9 and 297")
    for key in ("guidance_scale", "flow_shift", "control_guidance"):
        value = request[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise AugmentationInputError(f"{key} must be a finite positive number")
        request[key] = float(value)
    if request["guidance_scale"] > 20:
        raise AugmentationInputError("The installed video API supports guidance_scale at most 20")
    if request["edge_threshold"] not in ("low", "medium", "high"):
        raise AugmentationInputError("edge_threshold must be low, medium or high")
    return request


def request_sha256(request: dict[str, Any]) -> str:
    canonical = json.dumps(validate_request(request), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def chunk_plan(frames: int, chunk_frames: int = 121) -> list[dict[str, int]]:
    """Cover every source frame; each later interval includes five old frames."""
    if type(frames) is not int or frames < 6:
        raise AugmentationInputError("Source must contain at least six frames")
    if type(chunk_frames) is not int or not 9 <= chunk_frames <= 297 or (chunk_frames - 1) % 4:
        raise AugmentationInputError("Invalid transfer chunk length")
    result = []
    start = 0
    while True:
        length = min(chunk_frames, frames - start)
        result.append({
            "index": len(result), "source_start": start, "source_frames": length,
            "model_chunk_frames": math.ceil((length - 1) / 4) * 4 + 1,
            "drop_prefix_frames": OVERLAP_FRAMES if result else 0,
        })
        if start + length == frames:
            return result
        start += length - OVERLAP_FRAMES


def _command(argv: list[str], *, data: bytes | None = None) -> bytes:
    result = subprocess.run(argv, input=data, capture_output=True, check=False)
    if result.returncode:
        raise NanoVideoError(f"{Path(argv[0]).name} failed with exit code {result.returncode}")
    return result.stdout


def validate_media(path: Path, *, frames: int | None = None, width: int = WIDTH) -> dict[str, Any]:
    """Decode all frames and prove the presentation timeline, including MKV."""
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise NanoVideoError("Video must be a nonempty regular file")
    raw = _command([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,duration:format=duration:frame=best_effort_timestamp_time,width,height",
        "-of", "json", str(path),
    ])
    try:
        evidence = json.loads(raw)
        stream = evidence["streams"][0]
        decoded = evidence["frames"]
        count = len(decoded)
        if count < 1 or (frames is not None and count != frames):
            raise ValueError("frame count")
        if Fraction(stream["avg_frame_rate"]) != FPS or (stream["width"], stream["height"]) != (width, HEIGHT):
            raise ValueError("shape or fps")
        stamps = [float(frame["best_effort_timestamp_time"]) for frame in decoded]
        # Matroska's millisecond time base rounds 24fps timestamps by <=0.334ms.
        error = max(abs(stamp - index / FPS) for index, stamp in enumerate(stamps))
        if error > 0.00051 or any(not math.isfinite(t) for t in stamps):
            raise ValueError("timestamps")
        if any(right <= left for left, right in zip(stamps, stamps[1:])):
            raise ValueError("nonmonotonic timestamps")
        if any((frame["width"], frame["height"]) != (width, HEIGHT) for frame in decoded):
            raise ValueError("changing dimensions")
        duration = float(stream.get("duration", evidence.get("format", {}).get("duration", "nan")))
        if not math.isfinite(duration) or abs(duration - count / FPS) > 0.002:
            raise ValueError("container duration")
    except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError) as exc:
        raise NanoVideoError("Video frame/rate/shape/timestamp contract failed") from exc
    _command(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"])
    return {
        "valid": True, "full_decode_passed": True, "decoded_frames": count,
        "fps": float(FPS), "width": width, "height": HEIGHT,
        "duration_seconds": count / FPS, "timestamps_verified": True,
        "first_timestamp_seconds": stamps[0], "last_timestamp_seconds": stamps[-1],
        "max_timestamp_error_seconds": error,
        "container_duration_seconds": duration,
    }


def validate_source(path: Path, request: dict[str, Any]) -> dict[str, Any]:
    try:
        identity = artifact(path)
        if (identity["sha256"], identity["bytes"]) != (request["source_sha256"], request["source_bytes"]):
            raise NanoVideoError("Source bytes do not match the declared identity")
        video = validate_media(path)
        plan = chunk_plan(video["decoded_frames"], request["chunk_frames"])
        if request["seed"] + len(plan) - 1 >= 2**63:
            raise NanoVideoError("Per-chunk seeds would exceed the supported range")
        return video
    except (NanoVideoError, OSError) as exc:
        raise AugmentationInputError("Source hash, decode, timeline or 480p/24fps contract failed") from exc


def transfer_fields(request: dict[str, Any], chunk: dict[str, Any], control_path: Path) -> dict[str, str]:
    """Build actual multipart fields for the installed /v1/videos/sync parser."""
    extra = {
        "edge": {"control_path": str(control_path), "preset_edge_threshold": request["edge_threshold"]},
        "resolution": "480", "control_guidance": request["control_guidance"],
        "num_video_frames_per_chunk": chunk["model_chunk_frames"],
        "num_conditional_frames": OVERLAP_FRAMES,
        "num_first_chunk_conditional_frames": chunk["drop_prefix_frames"],
        "max_frames": chunk["source_frames"], "share_vision_temporal_positions": True,
        "show_input": False, "show_control_condition": False,
        "max_sequence_length": request["max_sequence_length"],
        "use_system_prompt": True, "system_prompt": request["system_prompt"],
        "use_duration_template": False, "use_resolution_template": False,
        "guardrails": False,
    }
    return {
        "prompt": request["prompt"], "negative_prompt": request["negative_prompt"],
        "size": f"{WIDTH}x{HEIGHT}", "fps": str(FPS),
        "num_frames": str(chunk["source_frames"]),
        "num_inference_steps": str(request["num_inference_steps"]),
        "guidance_scale": str(request["guidance_scale"]), "flow_shift": str(request["flow_shift"]),
        "seed": str(request["seed"] + chunk["index"]),
        "extra_params": json.dumps(extra, separators=(",", ":")),
    }


def _effective_transfer(fields: dict[str, str], chunk: dict[str, Any]) -> dict[str, Any]:
    # Import the very pipeline the image serves, not a second implementation of
    # its formatter or transfer defaults. No CUDA allocation occurs here.
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import _format_json_object_prompt
    from vllm_omni.diffusion.models.cosmos3.transfer import resolve_transfer_config

    extra = json.loads(fields["extra_params"])
    config = resolve_transfer_config(SimpleNamespace(
        extra_args=extra, guidance_scale=float(fields["guidance_scale"]),
        flow_shift=float(fields["flow_shift"]), fps=FPS, num_frames=int(fields["num_frames"]),
    ))
    if config is None or set(config.hints) != {"edge"}:
        raise NanoVideoError("Installed pipeline did not resolve structural edge transfer")
    resolved = asdict(config)
    expected = {
        "num_video_frames_per_chunk": chunk["model_chunk_frames"],
        "num_conditional_frames": OVERLAP_FRAMES,
        "num_first_chunk_conditional_frames": chunk["drop_prefix_frames"],
        "max_frames": chunk["source_frames"], "share_vision_temporal_positions": True,
        "show_input": False, "show_control_condition": False,
    }
    if any(resolved[key] != value for key, value in expected.items()):
        raise NanoVideoError("Installed transfer parser changed the requested controls")
    prompt = _format_json_object_prompt(fields["prompt"], num_frames=chunk["model_chunk_frames"],
        frame_rate=FPS, height=HEIGHT, width=WIDTH, aspect_ratio=None)
    return {"transfer_config": resolved, "positive_prompt": prompt if prompt is not None else fields["prompt"],
        "negative_prompt": fields["negative_prompt"], "system_prompt": extra["system_prompt"],
        "sampling": {"num_inference_steps": int(fields["num_inference_steps"]),
            "max_sequence_length": extra["max_sequence_length"], "resolution": extra["resolution"],
            "fps": int(fields["fps"]), "use_system_prompt": extra["use_system_prompt"],
            "use_duration_template": extra["use_duration_template"],
            "use_resolution_template": extra["use_resolution_template"]},
        "metadata_note": "Upstream rewrites JSON-object duration to floor(model frames/fps), even with templates off."}


def prepare_control(source: Path, target: Path, chunk: dict[str, Any], preset: str) -> dict[str, Any]:
    """Call official Canny preprocessing on this exact ORIGINAL source interval."""
    import numpy as np
    import torch
    from vllm_omni.diffusion.models.cosmos3 import transfer

    start, frames = chunk["source_start"], chunk["source_frames"]
    raw = _command([
        "ffmpeg", "-v", "error", "-xerror", "-i", str(source), "-vf",
        f"trim=start_frame={start}:end_frame={start + frames},setpts=PTS-STARTPTS",
        "-an", "-vsync", "0", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ])
    if len(raw) != frames * WIDTH * HEIGHT * 3:
        raise NanoVideoError("Source interval did not decode to its declared frame count")
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(frames, HEIGHT, WIDTH, 3).copy()
    source_tensor = torch.from_numpy(pixels).permute(3, 0, 1, 2).contiguous()
    controls = transfer.make_edge_control(source_tensor, preset)
    control_rgb = controls.permute(1, 2, 3, 0).contiguous().numpy().tobytes()
    _command([
        "ffmpeg", "-v", "error", "-xerror", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an",
        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", str(target),
    ], data=control_rgb)
    # Read through the installed transfer loader, proving its actual codec and
    # pixel interpretation match the arrays produced by the official helper.
    loaded = transfer.media_to_uint8_cthw(target, height=HEIGHT, width=WIDTH, max_frames=frames)
    if not torch.equal(controls, loaded):
        raise NanoVideoError("Lossless edge control readback changed pixels")
    return {
        "engine": "vllm-omni.cosmos3.transfer.make_edge_control",
        "upstream_module_sha256": artifact(Path(transfer.__file__))["sha256"],
        "source_start": start, "source_frames": frames,
        "source_rgb_sha256": hashlib.sha256(raw).hexdigest(),
        "original_source_sha256": artifact(source)["sha256"],
        "control_rgb_sha256": hashlib.sha256(control_rgb).hexdigest(),
        "preset": preset, "canny_thresholds": list(transfer.EDGE_PRESETS[preset]),
        "control_video": validate_media(target, frames=frames),
        "lossless_upstream_readback_equal": True, "source": "original input.mp4 only",
    }


def extract_tail(source: Path, target: Path, frames: int) -> dict[str, Any]:
    _command([
        "ffmpeg", "-v", "error", "-xerror", "-y", "-i", str(source), "-vf",
        f"trim=start_frame={frames - OVERLAP_FRAMES}:end_frame={frames},setpts=PTS-STARTPTS",
        "-an", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", "-r", str(FPS), str(target),
    ])
    return validate_media(target, frames=OVERLAP_FRAMES)


def stitch(chunks: list[dict[str, Any]], directory: Path, target: Path) -> None:
    argv = ["ffmpeg", "-v", "error", "-xerror", "-y"]
    filters = []
    for chunk in chunks:
        index = chunk["index"]
        argv.extend(["-i", str(directory / chunk["output_path"])])
        filters.append(f"[{index}:v]trim=start_frame={chunk['drop_prefix_frames']}:end_frame={chunk['source_frames']},setpts=PTS-STARTPTS[v{index}]")
    filters.append("".join(f"[v{i}]" for i in range(len(chunks))) + f"concat=n={len(chunks)}:v=1:a=0[out]")
    argv.extend(["-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-c:v", "libx264",
        "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart", str(target)])
    _command(argv)


def comparison(source: Path, output: Path, target: Path) -> None:
    _command([
        "ffmpeg", "-v", "error", "-xerror", "-y", "-i", str(source), "-i", str(output),
        "-filter_complex",
        "[0:v]setpts=PTS-STARTPTS,drawtext=text='SOURCE':fontcolor=white:fontsize=24:x=12:y=12:box=1:boxcolor=black@0.6[l];"
        "[1:v]setpts=PTS-STARTPTS,drawtext=text='AUGMENTED':fontcolor=white:fontsize=24:x=12:y=12:box=1:boxcolor=black@0.6[r];"
        "[l][r]hstack=inputs=2[out]",
        "-map", "[out]", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-movflags", "+faststart", str(target),
    ])


def run_augmentation(*, endpoint: str, output_dir: Path, input_video: Path,
                     request: dict[str, Any], replica_id: str) -> dict[str, Any]:
    """One admitted workload, all source intervals, no implicit generation retry."""
    request = validate_request(request)
    if output_dir.exists():
        raise FileExistsError("Request already exists")
    source_video = validate_source(input_video, request)  # Before any GPU access.
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)  # Cross-replica immutable admission.
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": SCHEMA, "status": "running", "request_id": request["request_id"],
        "request": request, "request_sha256": request_sha256(request),
        "model": "nvidia/Cosmos3-Nano", "model_revision": MODEL_REVISION,
        "replica_id": replica_id, "started_at": utc_now(), "chunks": [], "artifacts": [],
        "effective_parameters": {"dtype": "bfloat16", "tensor_parallel_size": 1, "guardrails": False,
            "sound_gen": False, "structural_control": "edge", "resolution": "480", "fps": FPS,
            "overlap_frames": OVERLAP_FRAMES, "source_motion_conditioning": "every original source interval",
            "stitch": "drop duplicate RGB prefixes, then concatenate; no blending or interpolation"},
        "quality_evaluation": "pending; technical validation is not visual quality approval",
        "measurement_limits": [
            "HTTP wall time includes reference upload/decode; server handler time includes generation and encoding.",
            "The installed profiler does not instrument diffuse_transfer; missing stage timing is not zero GPU time.",
            "NVML samples total assigned-device memory every 0.5s and may miss short peaks; engine peak is separate.",
            "Separate requests restart RNG and reconstruct RGB overlap; not bit-identical to native multi-chunk inference.",
        ],
    }
    write_json(output_dir / "request.json", request)
    write_json(output_dir / "report.json", report)
    sampler = DeviceMemorySampler()
    try:
        source = output_dir / "input.mp4"
        shutil.copyfile(input_video, source)
        source.chmod(0o400)
        report["source"] = {**artifact(source), "video": source_video}
        plan = chunk_plan(source_video["decoded_frames"], request["chunk_frames"])
        sampler.start()
        with httpx.Client(timeout=None, trust_env=False, follow_redirects=False) as client:
            for planned in plan:
                chunk: dict[str, Any] = {**planned, "status": "running", "started_at": utc_now(),
                    "seed": request["seed"] + planned["index"]}
                index = chunk["index"]
                chunk.update(control_path=f"control-{index:03d}.mkv", output_path=f"chunk-{index:03d}.mp4",
                    request_path=f"request-{index:03d}.json", reference_path=None)
                report["chunks"].append(chunk)
                write_json(output_dir / "report.json", report)
                preparation = time.monotonic()
                control = output_dir / chunk["control_path"]
                chunk["control_provenance"] = prepare_control(source, control, chunk, request["edge_threshold"])
                reference = None
                if index:
                    previous = report["chunks"][index - 1]
                    reference = output_dir / f"reference-{index:03d}.mp4"
                    chunk["reference_path"] = reference.name
                    chunk["reference_video"] = extract_tail(output_dir / previous["output_path"], reference, previous["source_frames"])
                    chunk["reference_from_output"] = previous["output_path"]
                fields = transfer_fields(request, chunk, control)
                chunk["effective"] = _effective_transfer(fields, chunk)
                write_json(output_dir / chunk["request_path"], {"fields": fields, "effective": chunk["effective"],
                    "source_sha256": request["source_sha256"], "source_interval": planned,
                    "reference_path": chunk["reference_path"], "reference_role": "previous augmented tail only" if index else "none"})
                chunk["preparation_seconds"] = time.monotonic() - preparation
                samples_start = len(sampler.samples)
                chunk_started = time.monotonic()
                chunk["generation_started_at"] = utc_now()
                files: dict[str, Any] = {key: (None, value) for key, value in fields.items()}
                stream = reference.open("rb") if reference is not None else None
                try:
                    if stream is not None:
                        files["input_reference"] = (reference.name, stream, "video/mp4")
                    response = client.post(endpoint.rstrip("/") + "/v1/videos/sync", files=files,
                        headers={"Accept": "video/mp4"})
                finally:
                    if stream is not None:
                        stream.close()
                chunk["wall_seconds"] = time.monotonic() - chunk_started
                chunk["generation_finished_at"] = utc_now()
                chunk["http_status"] = response.status_code
                if response.status_code != 200 or response.headers.get("content-type", "").split(";")[0] != "video/mp4":
                    write_json(output_dir / f"error-{index:03d}.json", {"http_status": response.status_code,
                        "response": response.text[:16000]})
                    raise NanoVideoError(f"Transfer did not return MP4 (HTTP {response.status_code})")
                output = output_dir / chunk["output_path"]
                output.write_bytes(response.content)
                chunk["server_handler_seconds"] = _positive_header(response.headers, "X-Inference-Time-S")
                chunk["engine_peak_memory_mb"] = _positive_header(response.headers, "X-Peak-Memory-MB")
                raw_stages = response.headers.get("X-Stage-Durations", "{}")
                try:
                    stages = json.loads(raw_stages)
                    _stage_evidence(stages)
                except (ValueError, TypeError) as exc:
                    write_json(output_dir / f"error-{index:03d}.json", {"invalid_stage_durations_header": raw_stages})
                    raise NanoVideoError("Malformed server stage-duration evidence") from exc
                chunk["stage_durations"] = stages
                samples = sampler.samples[samples_start:]
                if not samples:
                    raise NanoVideoError("No device-memory samples captured for transfer interval")
                chunk["device_peak_used_mib"] = max(row["used_mib"] for row in samples)
                chunk["device_memory_sample_count"] = len(samples)
                chunk["validation"] = validate_media(output, frames=chunk["source_frames"])
                chunk["finished_at"] = utc_now()
                chunk["status"] = "succeeded"
                write_json(output_dir / "report.json", report)
        target = output_dir / "augmented.mp4"
        stitching = time.monotonic()
        stitch(report["chunks"], output_dir, target)
        report["stitch_seconds"] = time.monotonic() - stitching
        report["output"] = {**artifact(target), "video": validate_media(target, frames=source_video["decoded_frames"])}
        if report["output"]["sha256"] == report["source"]["sha256"]:
            raise NanoVideoError("Augmented video is identical to source bytes")
        comparison_path = output_dir / "comparison.mp4"
        comparison(source, target, comparison_path)
        report["comparison"] = {**artifact(comparison_path), "video": validate_media(comparison_path,
            frames=source_video["decoded_frames"], width=2 * WIDTH), "layout": "actual source left; augmentation right"}
        report["status"] = "succeeded"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        if report["chunks"] and report["chunks"][-1]["status"] == "running":
            report["chunks"][-1].update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        memory = sampler.stop()
        write_json(output_dir / "gpu-memory.json", memory)
        report["device_peak_used_mib"] = memory["peak_used_mib"]
        if memory["error"]:
            report.update(status="failed", memory_measurement_error=memory["error"])
        report["finished_at"] = utc_now()
        report["total_wall_seconds"] = time.monotonic() - started
        report["artifacts"] = [artifact(path) for path in sorted(output_dir.iterdir())
            if path.is_file() and not path.is_symlink() and path.name != "report.json"]
        validation_error = None
        if report["status"] == "succeeded":
            try:
                validate_report(report, request)
            except NanoVideoError as exc:
                report.update(status="failed", error_type=type(exc).__name__, evidence_validation_failed=True)
                validation_error = exc
        write_json(output_dir / "report.json", report)
        for path in output_dir.iterdir():
            if path.is_file() and not path.is_symlink():
                path.chmod(0o400)
        if validation_error is not None:
            raise validation_error
    if report["status"] != "succeeded":
        raise NanoVideoError("Transfer completed but required memory measurement failed")
    return report


def artifact_manifest(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = report.get("artifacts")
    if not isinstance(items, list) or not items:
        raise NanoVideoError("Augmentation artifact manifest is missing")
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise NanoVideoError("Invalid artifact manifest entry")
        name = item.get("path")
        if (not isinstance(name, str) or not SAFE_NAME.fullmatch(name) or name in result
                or Path(name).suffix not in {".mp4", ".mkv", ".json", ".png"}
                or type(item.get("bytes")) is not int or item["bytes"] <= 0
                or not isinstance(item.get("sha256"), str) or not SHA256.fullmatch(item["sha256"])):
            raise NanoVideoError("Unsafe, duplicate or incomplete artifact manifest entry")
        result[name] = item
    if not {"input.mp4", "augmented.mp4", "comparison.mp4", "request.json", "gpu-memory.json"} <= result.keys():
        raise NanoVideoError("Required augmentation artifacts are missing")
    return result


def validate_report(report: Any, request: dict[str, Any]) -> None:
    """Reject unrelated T2V, missing intervals and unsupported parameter evidence."""
    request = validate_request(request)
    if not isinstance(report, dict):
        raise NanoVideoError("Augmentation report must be an object")
    expected = {"schema_version": SCHEMA, "status": "succeeded", "request_id": request["request_id"],
        "request": request, "request_sha256": request_sha256(request), "model_revision": MODEL_REVISION}
    if any(report.get(key) != value for key, value in expected.items()):
        raise NanoVideoError("Augmentation request/result identity mismatch")
    manifest = artifact_manifest(report)
    try:
        frames = report["source"]["video"]["decoded_frames"]
        plan = chunk_plan(frames, request["chunk_frames"])
        if report["source"]["sha256"] != request["source_sha256"] or report["source"]["bytes"] != request["source_bytes"]:
            raise ValueError("source identity")
        if report["source"]["sha256"] == report["output"]["sha256"]:
            raise ValueError("source/output distinction")
        for role, name, width in (("source", "input.mp4", WIDTH), ("output", "augmented.mp4", WIDTH),
                                  ("comparison", "comparison.mp4", WIDTH * 2)):
            item = report[role]
            if any(item[key] != manifest[name][key] for key in ("path", "sha256", "bytes")):
                raise ValueError("artifact identity")
            _video_evidence(item["video"], frames, width)
        chunks = report["chunks"]
        if len(chunks) != len(plan):
            raise ValueError("source coverage")
        for chunk, planned in zip(chunks, plan, strict=True):
            if any(chunk[key] != value for key, value in planned.items()) or chunk["status"] != "succeeded":
                raise ValueError("source interval")
            index = chunk["index"]
            if chunk["seed"] != request["seed"] + index or chunk["http_status"] != 200:
                raise ValueError("chunk seed/status")
            for key, name in (("control_path", f"control-{index:03d}.mkv"), ("output_path", f"chunk-{index:03d}.mp4"),
                              ("request_path", f"request-{index:03d}.json")):
                if chunk[key] != name or name not in manifest:
                    raise ValueError("chunk artifact")
            if index:
                if chunk["reference_path"] != f"reference-{index:03d}.mp4" or chunk["reference_path"] not in manifest:
                    raise ValueError("RGB tail")
                if chunk["reference_from_output"] != chunks[index - 1]["output_path"]:
                    raise ValueError("RGB tail source")
                _video_evidence(chunk["reference_video"], OVERLAP_FRAMES, WIDTH)
            elif chunk["reference_path"] is not None:
                raise ValueError("unexpected original RGB conditioning")
            provenance = chunk["control_provenance"]
            if (provenance["source_start"], provenance["source_frames"]) != (planned["source_start"], planned["source_frames"]):
                raise ValueError("control source interval")
            if provenance["lossless_upstream_readback_equal"] is not True or provenance["source"] != "original input.mp4 only":
                raise ValueError("control provenance")
            if provenance["original_source_sha256"] != request["source_sha256"]:
                raise ValueError("control original source identity")
            thresholds = {"low": [50, 100], "medium": [100, 200], "high": [200, 300]}
            if (provenance["engine"] != "vllm-omni.cosmos3.transfer.make_edge_control"
                    or provenance["preset"] != request["edge_threshold"]
                    or provenance["canny_thresholds"] != thresholds[request["edge_threshold"]]):
                raise ValueError("control preprocessing identity")
            for key in ("source_rgb_sha256", "control_rgb_sha256", "upstream_module_sha256"):
                if not isinstance(provenance[key], str) or not SHA256.fullmatch(provenance[key]):
                    raise ValueError("control content identity")
            _video_evidence(provenance["control_video"], planned["source_frames"], WIDTH)
            _video_evidence(chunk["validation"], planned["source_frames"], WIDTH)
            config = chunk["effective"]["transfer_config"]
            for key, value in (("num_video_frames_per_chunk", planned["model_chunk_frames"]),
                               ("max_frames", planned["source_frames"]), ("num_conditional_frames", OVERLAP_FRAMES),
                               ("num_first_chunk_conditional_frames", planned["drop_prefix_frames"]),
                               ("control_guidance", request["control_guidance"]), ("guidance_scale", request["guidance_scale"]),
                               ("flow_shift", request["flow_shift"]), ("share_vision_temporal_positions", True)):
                if config[key] != value:
                    raise ValueError("effective transfer parameter")
            for key, value in (("control_guidance_interval", None), ("fps", float(FPS)),
                               ("num_frames", planned["source_frames"]), ("show_input", False),
                               ("show_control_condition", False)):
                if config[key] != value:
                    raise ValueError("unexpected transfer override")
            if set(config) != {"hints", "guidance_scale", "control_guidance", "control_guidance_interval", "flow_shift",
                    "num_video_frames_per_chunk", "num_conditional_frames", "max_frames", "show_control_condition",
                    "show_input", "num_first_chunk_conditional_frames", "share_vision_temporal_positions", "num_frames", "fps"}:
                raise ValueError("unknown transfer resolver fields")
            if set(config["hints"]) != {"edge"} or Path(config["hints"]["edge"]["control_path"]).name != chunk["control_path"]:
                raise ValueError("missing edge control")
            hint = config["hints"]["edge"]
            if hint["preset_edge_threshold"] != request["edge_threshold"]:
                raise ValueError("edge preset")
            if (hint["key"] != "edge" or hint["control"] is not None or hint["preset_blur_strength"] != "medium"
                    or not Path(hint["control_path"]).is_absolute()
                    or set(hint) != {"key", "control", "control_path", "preset_edge_threshold", "preset_blur_strength"}):
                raise ValueError("unrequested or ambiguous structural control")
            effective = chunk["effective"]
            if (effective["positive_prompt"] != _expected_prompt(request["prompt"], planned["model_chunk_frames"])
                    or effective["negative_prompt"] != request["negative_prompt"]
                    or effective["system_prompt"] != request["system_prompt"]):
                raise ValueError("effective prompt identity")
            expected_sampling = {"num_inference_steps": request["num_inference_steps"],
                "max_sequence_length": request["max_sequence_length"], "resolution": "480", "fps": FPS,
                "use_system_prompt": True, "use_duration_template": False, "use_resolution_template": False}
            if effective["sampling"] != expected_sampling:
                raise ValueError("effective sampling identity")
            _stage_evidence(chunk["stage_durations"])
            for key in ("wall_seconds", "server_handler_seconds", "engine_peak_memory_mb", "device_peak_used_mib"):
                _measurement(chunk[key])
        for key in ("total_wall_seconds", "device_peak_used_mib"):
            _measurement(report[key])
        if not isinstance(report["replica_id"], str) or not report["replica_id"]:
            raise ValueError("replica identity")
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        raise NanoVideoError("Augmentation coverage, artifact or measurement evidence failed") from exc


def _measurement(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid measurement")


def _stage_evidence(stages: Any) -> None:
    if not isinstance(stages, dict):
        raise ValueError("Stage durations must be an object")
    for key, value in stages.items():
        if (not isinstance(key, str) or not key or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError("Invalid stage-duration measurement")


def _expected_prompt(prompt: str, model_frames: int) -> str:
    """Independent CPU assertion of the pinned upstream JSON metadata contract."""
    try:
        value = json.loads(prompt)
    except ValueError:
        return prompt
    if not isinstance(value, dict):
        return prompt
    value.update(duration=f"{int(model_frames / FPS)}s", fps=float(FPS), resolution={"H": HEIGHT, "W": WIDTH})
    return json.dumps(value)


def _video_evidence(video: dict[str, Any], frames: int, width: int) -> None:
    expected = {"valid": True, "full_decode_passed": True, "decoded_frames": frames, "fps": float(FPS),
        "width": width, "height": HEIGHT, "timestamps_verified": True, "duration_seconds": frames / FPS}
    if any(video.get(key) != value for key, value in expected.items()):
        raise ValueError("Invalid decoded video evidence")
    if any(video[key] is not True for key in ("valid", "full_decode_passed", "timestamps_verified")):
        raise ValueError("Decode assertions must be boolean evidence")
