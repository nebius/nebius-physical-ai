"""Measured 30-second Cosmos3-Nano diffusion rollout and CPU batch client.

The serving image can import this module without installing NPA's full CPU
dependency set. Neither Ray nor Torch is imported by the rollout/client code.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import httpx

MODEL_REVISION = "7a312c868bcce8e40b3eb40861300a9d0ba3fde1"
CHUNK_FRAMES = (297, 297, 137)
PREFIX_FRAMES = 5
FPS = 24
WIDTH, HEIGHT = 832, 480
FINAL_FRAMES = 720
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")
SCHEMA = "npa.cosmos3.nano-video.rollout.v1"
DEFAULT_PROMPT = (
    "A continuous realistic documentary shot of a small orange warehouse robot "
    "rolling slowly along a clean factory aisle. The camera tracks smoothly "
    "beside the robot at a constant speed, keeping the whole robot in view. "
    "Its wheels turn naturally, overhead lights cast stable soft shadows, and "
    "orderly blue shelving passes steadily in the background. The robot moves "
    "forward throughout the shot. Consistent robot appearance, lighting and "
    "camera direction. No scene changes, cuts, titles, or people."
)


class NanoVideoError(RuntimeError):
    """The requested generation or its evidence did not satisfy the contract."""


class DeviceMemorySampler:
    """Observe total residency on the single GPU assigned to the worker pod."""

    def __init__(self) -> None:
        self.samples: list[dict[str, float]] = []
        self.error: str | None = None
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = time.monotonic()

    def _sample(self) -> None:
        argv = [
            "nvidia-smi", "--query-gpu=name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            if "," in visible or not re.fullmatch(r"[A-Za-z0-9-]+", visible):
                raise NanoVideoError("Ray must assign exactly one GPU to the replica")
        # Each Ray worker pod requests exactly one GPU. CUDA may call that
        # device 0 while NVML retains its physical index, so CUDA ordinals must
        # never be passed to nvidia-smi --id. Query the pod-visible set instead
        # and fail closed if the one-GPU deployment contract is not satisfied.
        rows = _command(argv).decode().strip().splitlines()
        if len(rows) != 1:
            raise NanoVideoError("VRAM measurement requires exactly one assigned GPU")
        name, used, total = (part.strip() for part in rows[0].split(","))
        if "B200" not in name:
            raise NanoVideoError("requested B200 device was not observed")
        sample = {"elapsed_seconds": time.monotonic() - self._started,
                  "used_mib": float(used), "total_mib": float(total)}
        if not 0 < sample["used_mib"] <= sample["total_mib"]:
            raise NanoVideoError("invalid device VRAM measurement")
        self.samples.append(sample)

    def start(self) -> None:
        self._sample()  # Fail before generation when the target is wrong.

        def sample_until_done() -> None:
            while not self._done.wait(0.5):
                try:
                    self._sample()
                except Exception as exc:
                    self.error = type(exc).__name__
                    return

        self._thread = threading.Thread(target=sample_until_done, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._done.set()
        if self._thread:
            self._thread.join()
        return {"source": "nvidia-smi Ray-assigned B200 device memory.used",
                "sampling_interval_seconds": 0.5, "samples": self.samples,
                "peak_used_mib": max((sample["used_mib"] for sample in self.samples), default=None),
                "error": self.error}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    """Atomically publish owner-only JSON for concurrent shared-storage readers."""
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        # NamedTemporaryFile creates mode0600. Keep the temporary file beside
        # the destination so replacement is atomic on the shared filesystem.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # Cleanup must not replace the original publication failure.
                pass


def artifact(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def request_fields(prompt: str, seed: int, frames: int, *, continuation: bool) -> dict[str, str]:
    if frames < 5 or frames > 300 or (frames - 1) % 4:
        raise ValueError("chunk frames must be 4k+1, at least 5 and at most 300")
    extra: dict[str, Any] = {
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": False,
        "max_sequence_length": 4096,
    }
    if continuation:
        extra.update(condition_frame_indexes_vision=[0, 1], condition_video_keep="last")
    return {
        "prompt": prompt,
        "negative_prompt": "cuts, abrupt camera motion, scene changes, flicker, frozen motion, text, watermark",
        "size": f"{WIDTH}x{HEIGHT}",
        "fps": str(FPS),
        "num_frames": str(frames),
        "num_inference_steps": "35",
        "guidance_scale": "6.0",
        "flow_shift": "10.0",
        "seed": str(seed),
        "extra_params": json.dumps(extra, separators=(",", ":")),
    }


def _command(argv: list[str]) -> bytes:
    result = subprocess.run(argv, capture_output=True, check=False)
    if result.returncode:
        # Do not copy paths, URLs or private runtime diagnostics into API errors.
        raise NanoVideoError(f"{Path(argv[0]).name} failed with exit code {result.returncode}")
    return result.stdout


def validate_video(path: Path, frames: int) -> dict[str, Any]:
    """Count decoded frames, verify shape/rate/duration and decode every frame."""
    raw = _command([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,duration",
        "-of", "json", str(path),
    ])
    try:
        stream = json.loads(raw)["streams"][0]
        observed = int(stream["nb_read_frames"])
        rate = Fraction(stream["avg_frame_rate"])
        duration = float(stream["duration"])
        valid = (
            observed == frames and rate == FPS
            and stream["width"] == WIDTH and stream["height"] == HEIGHT
            and math.isfinite(duration) and abs(duration - frames / FPS) < 0.002
        )
    except (KeyError, ValueError, IndexError, ZeroDivisionError) as exc:
        raise NanoVideoError("video has incomplete frame/rate/shape evidence") from exc
    if not valid:
        raise NanoVideoError("decoded video does not match requested frames, rate, duration or 480p shape")
    _command(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"])
    return {"valid": True, "decoded_frames": observed, "fps": float(rate),
            "width": WIDTH, "height": HEIGHT, "duration_seconds": duration,
            "full_decode_passed": True}


def stitch_chunks(chunks: list[Path], target: Path) -> None:
    if len(chunks) != len(CHUNK_FRAMES):
        raise ValueError("the complete rollout requires exactly three chunks")
    argv = ["ffmpeg", "-v", "error", "-xerror", "-y"]
    filters = []
    for index, (chunk, frames) in enumerate(zip(chunks, CHUNK_FRAMES, strict=True)):
        argv.extend(["-i", str(chunk)])
        start = PREFIX_FRAMES if index else 0
        filters.append(f"[{index}:v]trim=start_frame={start}:end_frame={frames},setpts=PTS-STARTPTS[v{index}]")
    filters.append("[v0][v1][v2]concat=n=3:v=1:a=0,trim=end_frame=720,setpts=PTS-STARTPTS[out]")
    argv.extend([
        "-filter_complex", ";".join(filters), "-map", "[out]", "-an",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-movflags", "+faststart", str(target),
    ])
    _command(argv)


def seam_evidence(video: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Keep unblended boundary frames and factual adjacent-frame differences."""
    seams = []
    for index, boundary in enumerate((297, 589), 1):
        selection = f"select=between(n\\,{boundary - 4}\\,{boundary + 3})"
        pixels = _command([
            "ffmpeg", "-v", "error", "-i", str(video), "-vf", selection + ",scale=128:72,format=gray",
            "-vsync", "0", "-f", "rawvideo", "-",
        ])
        size = 128 * 72
        if len(pixels) != 8 * size:
            raise NanoVideoError("could not decode all stitch boundary frames")
        frames = [pixels[i * size:(i + 1) * size] for i in range(8)]
        diffs = [statistics.mean(abs(a - b) for a, b in zip(left, right, strict=True))
                 for left, right in zip(frames[:-1], frames[1:], strict=True)]
        contact = output_dir / f"seam-{index}.png"
        _command([
            "ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf",
            selection + ",scale=416:240,tile=4x2", "-frames:v", "1", str(contact),
        ])
        seams.append({
            "first_new_frame": boundary, "time_seconds": boundary / FPS,
            "contact_sheet": contact.name, "contact_sheet_frames": list(range(boundary - 4, boundary + 4)),
            "boundary_mean_absolute_gray_difference": diffs[3],
            "neighbor_median_mean_absolute_gray_difference": statistics.median(diffs[:3] + diffs[4:]),
            "adjacent_mean_absolute_gray_differences": diffs,
            "visual_review": "pending", "transition": "direct concatenation; no blending or interpolation",
        })
    return seams


def _positive_header(headers: httpx.Headers, name: str) -> float:
    try:
        value = float(headers[name])
    except (KeyError, ValueError) as exc:
        raise NanoVideoError(f"missing or malformed measurement header {name}") from exc
    if not math.isfinite(value) or value <= 0:
        raise NanoVideoError(f"invalid measurement header {name}")
    return value


def run_rollout(*, endpoint: str, output_dir: Path, prompt: str, seed: int, replica_id: str) -> dict[str, Any]:
    """Generate all chunks on one warmed TP=1 service; never retry GPU requests."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": SCHEMA, "status": "running", "started_at": utc_now(),
        "model": "nvidia/Cosmos3-Nano", "model_revision": MODEL_REVISION,
        "pipeline": "Cosmos3OmniDiffusersPipeline", "guardrails": False,
        "dtype": "bfloat16", "tensor_parallel_size": 1, "replica_id": replica_id,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "seed": seed,
        "chunks": [], "artifacts": [],
    }
    write_json(output_dir / "request.json", {"prompt": prompt, "seed": seed, "chunk_frames": CHUNK_FRAMES})
    paths: list[Path] = []
    sampler = DeviceMemorySampler()
    try:
        sampler.start()
        # No request deadline: server initialization has its separate 1800s setting.
        with httpx.Client(timeout=None, trust_env=False) as client:
            for index, frames in enumerate(CHUNK_FRAMES):
                fields = request_fields(prompt, seed + index, frames, continuation=index > 0)
                write_json(output_dir / f"chunk-{index + 1}-request.json", fields)
                chunk: dict[str, Any] = {"index": index + 1, "requested_frames": frames,
                    "seed": seed + index, "started_at": utc_now(), "status": "running"}
                report["chunks"].append(chunk)
                write_json(output_dir / "report.json", report)
                chunk_started = time.monotonic()
                previous = paths[-1].open("rb") if paths else None
                try:
                    # Always multipart, including the initial T2V request.
                    files: dict[str, Any] = {key: (None, value) for key, value in fields.items()}
                    if previous is not None:
                        files["input_reference"] = ("previous.mp4", previous, "video/mp4")
                        chunk["conditioned_on"] = artifact(paths[-1])
                    response = client.post(endpoint.rstrip("/") + "/v1/videos/sync",
                        files=files, headers={"Accept": "video/mp4"})
                finally:
                    if previous is not None:
                        previous.close()
                chunk["wall_seconds"] = time.monotonic() - chunk_started
                chunk["finished_at"] = utc_now()
                chunk["http_status"] = response.status_code
                if response.status_code != 200:
                    raise NanoVideoError(f"diffusion generation returned HTTP {response.status_code}")
                if response.headers.get("content-type", "").split(";")[0] != "video/mp4":
                    raise NanoVideoError("diffusion generation did not return video/mp4")
                path = output_dir / f"chunk-{index + 1}.mp4"
                path.write_bytes(response.content)
                paths.append(path)
                chunk["artifact"] = artifact(path)
                chunk["inference_seconds"] = _positive_header(response.headers, "X-Inference-Time-S")
                chunk["peak_memory_mb"] = _positive_header(response.headers, "X-Peak-Memory-MB")
                chunk["stage_durations"] = json.loads(response.headers["X-Stage-Durations"])
                chunk["validation"] = validate_video(path, frames)
                chunk["status"] = "succeeded"
                write_json(output_dir / "report.json", report)
        target = output_dir / "video-30s.mp4"
        stitch_started = time.monotonic()
        stitch_chunks(paths, target)
        report["stitch_seconds"] = time.monotonic() - stitch_started
        report["validation"] = validate_video(target, FINAL_FRAMES)
        report["seams"] = seam_evidence(target, output_dir)
        report["peak_memory_mb"] = max(chunk["peak_memory_mb"] for chunk in report["chunks"])
        report["memory_measurement"] = "vLLM-Omni X-Peak-Memory-MB; engine-reported CUDA peak, not total device residency"
        report["stitch"] = {"duplicate_prefix_frames_removed": [0, 5, 5], "final_tail_frames_trimmed": 1}
        report["status"] = "succeeded"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        if report["chunks"] and report["chunks"][-1]["status"] == "running":
            report["chunks"][-1].update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        memory = sampler.stop()
        write_json(output_dir / "gpu-memory.json", memory)
        report["device_peak_used_mib"] = memory["peak_used_mib"]
        if memory["error"]:
            report["status"] = "failed"
            report["memory_measurement_error"] = memory["error"]
        report["finished_at"] = utc_now()
        report["total_wall_seconds"] = time.monotonic() - started
        report["artifacts"] = [artifact(path) for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "report.json"]
        write_json(output_dir / "report.json", report)
    if report["status"] != "succeeded":
        raise NanoVideoError("generation completed but required device measurement failed")
    return report


def _measurement(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NanoVideoError(f"missing or malformed measurement {name}")
    if not math.isfinite(value) or value <= 0:
        raise NanoVideoError(f"invalid measurement {name}")
    return float(value)


def _interval(item: dict[str, Any]) -> tuple[datetime, datetime]:
    try:
        started = datetime.fromisoformat(item["started_at"])
        finished = datetime.fromisoformat(item["finished_at"])
        if started.utcoffset() is None or finished.utcoffset() is None or finished <= started:
            raise ValueError("invalid interval")
    except (KeyError, TypeError, ValueError) as exc:
        raise NanoVideoError("missing or invalid generation time interval") from exc
    return started, finished


def _video_evidence(value: Any, frames: int) -> None:
    if not isinstance(value, dict) or (
        value.get("valid") is not True or value.get("full_decode_passed") is not True
        or value.get("decoded_frames") != frames or value.get("fps") != FPS
        or value.get("width") != WIDTH or value.get("height") != HEIGHT
    ):
        raise NanoVideoError("incomplete decoded video evidence in service report")
    if abs(_measurement(value.get("duration_seconds"), "duration_seconds") - frames / FPS) >= 0.002:
        raise NanoVideoError("service video duration differs from the requested frame count")


def _artifact_manifest(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = report.get("artifacts")
    if not isinstance(values, list):
        raise NanoVideoError("service report has no artifact manifest")
    manifest = {}
    for item in values:
        name = item.get("path") if isinstance(item, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
            raise NanoVideoError("unsafe artifact path in service response")
        if name in manifest:
            raise NanoVideoError("duplicate artifact path in service response")
        size, digest = item.get("bytes"), item.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise NanoVideoError("missing or invalid artifact byte count")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise NanoVideoError("missing or invalid artifact SHA256")
        manifest[name] = item
    required = {"video-30s.mp4", *(f"chunk-{index}.mp4" for index in range(1, 4))}
    if not required <= manifest.keys():
        raise NanoVideoError("service report omits a required generated MP4 artifact")
    return manifest


def _validate_rollout_report(report: Any, request_id: str, prompt: str, seed: int) -> None:
    if not isinstance(report, dict):
        raise NanoVideoError("service did not return a rollout report")
    expected = {
        "schema_version": SCHEMA, "status": "succeeded", "request_id": request_id,
        "model": "nvidia/Cosmos3-Nano", "model_revision": MODEL_REVISION,
        "pipeline": "Cosmos3OmniDiffusersPipeline", "dtype": "bfloat16",
        "tensor_parallel_size": 1, "guardrails": False, "seed": seed,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise NanoVideoError("service report does not match the requested diffusion rollout")
    if report.get("guardrails") is not False or type(report.get("tensor_parallel_size")) is not int:
        raise NanoVideoError("service report has invalid guardrail or tensor parallel evidence")
    replica = report.get("replica_id")
    if not isinstance(replica, str) or not replica:
        raise NanoVideoError("service report has no replica identity")
    started, finished = _interval(report)
    _measurement(report.get("total_wall_seconds"), "total_wall_seconds")
    _measurement(report.get("device_peak_used_mib"), "device_peak_used_mib")
    manifest = _artifact_manifest(report)
    chunks = report.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != len(CHUNK_FRAMES):
        raise NanoVideoError("service did not complete the full three-chunk rollout")
    previous_finished = started
    for index, (chunk, frames) in enumerate(zip(chunks, CHUNK_FRAMES, strict=True), 1):
        if not isinstance(chunk, dict) or (
            chunk.get("status") != "succeeded" or chunk.get("index") != index
            or chunk.get("requested_frames") != frames or chunk.get("seed") != seed + index - 1
            or chunk.get("http_status") != 200
        ):
            raise NanoVideoError("service report has incomplete chunk generation evidence")
        chunk_start, chunk_end = _interval(chunk)
        if chunk_start < previous_finished or chunk_end > finished:
            raise NanoVideoError("chunk timing does not describe a sequential complete rollout")
        previous_finished = chunk_end
        for name in ("wall_seconds", "inference_seconds", "peak_memory_mb"):
            _measurement(chunk.get(name), name)
        _video_evidence(chunk.get("validation"), frames)
        item = chunk.get("artifact")
        expected_artifact = manifest[f"chunk-{index}.mp4"]
        if not isinstance(item, dict) or any(item.get(key) != expected_artifact[key] for key in ("path", "bytes", "sha256")):
            raise NanoVideoError("chunk artifact differs from the published manifest")
    peak = _measurement(report.get("peak_memory_mb"), "peak_memory_mb")
    if peak != max(chunk["peak_memory_mb"] for chunk in chunks):
        raise NanoVideoError("rollout peak memory differs from observed chunk peaks")
    _video_evidence(report.get("validation"), FINAL_FRAMES)


def _download_result(client: httpx.Client, endpoint: str, request_id: str, root: Path, report: dict[str, Any]) -> None:
    for name, item in _artifact_manifest(report).items():
        response = client.get(f"{endpoint}/artifacts/{request_id}/{name}")
        response.raise_for_status()
        data = response.content
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise NanoVideoError("artifact hash or length mismatch")
        (root / name).write_bytes(data)
    for index, frames in enumerate(CHUNK_FRAMES, 1):
        validate_video(root / f"chunk-{index}.mp4", frames)
    validate_video(root / "video-30s.mp4", FINAL_FRAMES)


def _peak_overlap(intervals: list[tuple[datetime, datetime]]) -> int:
    boundaries = [(started, 1) for started, _ in intervals]
    boundaries.extend((finished, -1) for _, finished in intervals)
    running = peak = 0
    for _, delta in sorted(boundaries):
        running += delta
        peak = max(peak, running)
    return peak


def run_batch(*, endpoint: str, output_dir: Path, concurrency: int, token: str, prompt: str = DEFAULT_PROMPT) -> dict[str, Any]:
    """Barrier-start complete requests and verify downloaded generation evidence."""
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if not token:
        raise ValueError("serving API token is required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    barrier = threading.Barrier(concurrency)
    batch_id = "batch-" + os.urandom(8).hex()
    batch_started = time.monotonic()

    def one(index: int) -> dict[str, Any]:
        request_id = f"{batch_id}-{index}"
        root = output_dir / request_id
        started_at = utc_now()
        admitted = False
        try:
            root.mkdir()
            with httpx.Client(timeout=None, trust_env=False, headers={"Authorization": f"Bearer {token}"}) as client:
                barrier.wait()
                admitted = True
                started_at = utc_now()
                started = time.monotonic()
                response = client.post(endpoint.rstrip("/") + "/run", json={
                    "request_id": request_id, "prompt": prompt, "seed": 1000 + index * 100,
                })
                response.raise_for_status()
                report = response.json()
                _validate_rollout_report(report, request_id, prompt, 1000 + index * 100)
                report["client_started_at"] = started_at
                report["client_generation_wall_seconds"] = time.monotonic() - started
                _download_result(client, endpoint.rstrip("/"), request_id, root, report)
                report["client_total_wall_seconds"] = time.monotonic() - started
                write_json(root / "report.json", report)
                return {"request_id": request_id, "status": "succeeded", "report": report}
        except Exception as exc:
            # Setup failure must release peers before any failure-report I/O.
            # BrokenBarrierError from that abort follows this same sanitized
            # failure path. No admitted generation request is retried.
            if not admitted:
                barrier.abort()
            result = {"request_id": request_id, "status": "failed", "error_type": type(exc).__name__,
                      "started_at": started_at, "finished_at": utc_now()}
            evidence = (
                root / "client-failure.json" if root.is_dir()
                else output_dir / f"{request_id}-client-failure.json"
            )
            try:
                write_json(evidence, result)
            except OSError as evidence_error:
                # The batch manifest still retains this per-request failure if
                # its individual recovery file cannot be published.
                result["evidence_error_type"] = type(evidence_error).__name__
            return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(one, range(concurrency)))
    good = [item["report"] for item in results if item["status"] == "succeeded"]
    peak = _peak_overlap([_interval(report) for report in good])
    chunk_peak = _peak_overlap([_interval(chunk) for report in good for chunk in report["chunks"]])
    distinct = len({report["replica_id"] for report in good})
    passed = len(good) == concurrency and peak == concurrency and chunk_peak == concurrency and distinct == concurrency
    batch = {"schema_version": "npa.cosmos3.nano-video.batch.v1", "status": "succeeded" if passed else "failed",
             "concurrency": concurrency, "completed": len(good), "distinct_replicas": distinct,
             "peak_overlapping_rollouts": peak, "peak_overlapping_chunk_requests": chunk_peak,
             "total_wall_seconds": time.monotonic() - batch_started,
             "requests": results, "fanout_verified": passed}
    write_json(output_dir / "batch.json", batch)
    return batch


def submit_batch(*, output_path: str, concurrency: int, endpoint: str = "", input_path: str = "",
                 token_env: str = "NPA_COSMOS3_VIDEO_TOKEN", storage_client: Any = None) -> dict[str, Any]:
    """Shared CLI/SDK client with verified S3 publication and local recovery."""
    endpoint = endpoint or os.environ.get("NPA_COSMOS3_VIDEO_ENDPOINT", "")
    if not endpoint:
        raise ValueError("serving endpoint is required")
    if not output_path.startswith("s3://"):
        raise ValueError("public output-path must be an s3:// prefix")
    from urllib.parse import urlsplit

    destination = urlsplit(output_path)
    if not destination.netloc or not destination.path.strip("/") or destination.query or destination.fragment:
        raise ValueError("output-path must name a bucket and non-empty S3 prefix")
    if input_path:
        source = urlsplit(input_path)
        if source.scheme != "s3" or not source.netloc or not source.path.strip("/") or source.query or source.fragment:
            raise ValueError("input-path must name a bucket and exact S3 object")
    from npa.clients.storage import StorageClient

    storage = storage_client or StorageClient.from_environment()
    bucket, prefix = destination.netloc, destination.path.strip("/")
    existing = storage.s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/", MaxKeys=1)
    if existing.get("KeyCount", 0) or existing.get("Contents"):
        raise NanoVideoError("output prefix already contains artifacts; recover the existing batch instead of regenerating")
    # Persist the local recovery copy outside the repository. Upload failures
    # must never cause an otherwise completed GPU workload to run again.
    recovery = Path(os.environ.get("NPA_COSMOS3_VIDEO_RECOVERY_DIR", tempfile.gettempdir()))
    recovery.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="npa-nano-video-", dir=recovery))
    prompt = DEFAULT_PROMPT
    if input_path:
        storage.download_file(input_path, str(root / "input.json"))
        prompt = json.loads((root / "input.json").read_text())["prompt"]
    # A retained conditional reservation verifies write/read access and prevents
    # simultaneous clients from generating into the same immutable batch prefix.
    reservation = json.dumps({"schema_version": "npa.cosmos3.nano-video.reservation.v1",
                              "created_at": utc_now(), "id": os.urandom(16).hex()}).encode()
    reservation_key = prefix + "/reservation.json"
    storage.s3.put_object(Bucket=bucket, Key=reservation_key, Body=reservation, IfNoneMatch="*")
    proof = storage.s3.get_object(Bucket=bucket, Key=reservation_key)
    try:
        verified = proof["Body"].read() == reservation
    finally:
        proof["Body"].close()
    if not verified:
        raise NanoVideoError("S3 reservation read-after-write failed before GPU generation")
    batch = run_batch(endpoint=endpoint, output_dir=root / "batch", concurrency=concurrency,
                      token=os.environ.get(token_env, ""), prompt=prompt)
    objects = []
    try:
        for path in sorted((root / "batch").rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root / "batch").as_posix()
            key = f"{prefix}/{relative}"
            data = path.read_bytes()
            storage.s3.put_object(Bucket=bucket, Key=key, Body=data, IfNoneMatch="*")
            response = storage.s3.get_object(Bucket=bucket, Key=key)
            try:
                actual = response["Body"].read()
            finally:
                response["Body"].close()
            if hashlib.sha256(actual).digest() != hashlib.sha256(data).digest():
                raise NanoVideoError("S3 read-after-write hash mismatch; retain local recovery evidence")
            objects.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    except Exception as exc:
        write_json(root / "publication.json", {"verified": False, "status": "pending",
                   "error_type": type(exc).__name__, "verified_objects": objects})
        raise
    write_json(root / "publication.json", {"verified": True, "objects": objects})
    return {"status": batch["status"], "concurrency": concurrency, "completed": batch["completed"],
            "distinct_replicas": batch["distinct_replicas"], "peak_overlapping_rollouts": batch["peak_overlapping_rollouts"],
            "total_wall_seconds": batch["total_wall_seconds"], "published_objects": len(objects),
            "publication_verified": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get("NPA_COSMOS3_VIDEO_ENDPOINT", ""))
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--input-path", type=Path, help="Optional JSON containing a prompt; default uses a synthetic robot scene")
    args = parser.parse_args()
    prompt = json.loads(args.input_path.read_text())["prompt"] if args.input_path else DEFAULT_PROMPT
    result = run_batch(endpoint=args.endpoint, output_dir=args.output_path, concurrency=args.concurrency,
                       token=os.environ.get("NPA_COSMOS3_VIDEO_TOKEN", ""), prompt=prompt)
    print(json.dumps({key: result[key] for key in ("status", "completed", "distinct_replicas", "peak_overlapping_rollouts", "total_wall_seconds")}))
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
