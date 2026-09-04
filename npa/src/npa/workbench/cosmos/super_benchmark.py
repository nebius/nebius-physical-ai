"""Reproducible single-node B200/H200 serving benchmark for Cosmos3-Super.

The benchmark runs inside the immutable public vLLM-Omni Cosmos3 image.  It
starts independent loopback services on disjoint GPU sets, validates one warmup
per service, and then measures a fixed production cell.  Only the prompt hashes
are recorded; prompt text remains in the operator's runtime model cache.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from npa.clients.storage import StorageClient, StoragePreconditionFailed

SCHEMA_VERSION = "npa.cosmos3-super.b200-benchmark.v1"
ATTEMPT_SCHEMA_VERSION = "npa.cosmos3-super.b200-attempt.v1"
SUPPORTED_GPU_FAMILIES = ("B200", "H200")
PRIMARY_SUITE = "primary"
B200_FULL_SUITE = "b200-full"
H200_SINGLE_GPU_SUITE = "h200-single-gpu"
SUITE_CHOICES = (PRIMARY_SUITE, B200_FULL_SUITE, H200_SINGLE_GPU_SUITE)
UPSTREAM_METHOD_REVISION = "532bffd4c2b2ec08909a92d5bc0b3bab4e911b2b"
UPSTREAM_B200_RECORD_SHA256 = (
    "18cf5ae1d118e07f3f2111b56a3e02c76eb9282d847a78005c6ca060f8106221"
)
MODEL_ID = "nvidia/Cosmos3-Super"
MODEL_REVISION = "e0262be9d8f7586bc24c069a2aed2b665bdff266"
IMAGE = (
    "docker.io/vllm/vllm-omni:cosmos3@"
    "sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587"
)
PROMPT_ASSET = "assets/example_t2v_prompt.json"
NEGATIVE_PROMPT_ASSET = "assets/negative_prompt.json"
PUBLIC_PROMPT_SHA256 = "61c9c4b46b6787d967cc509a2bf323766e70bf5ecf40e09a739362beac135677"
PUBLIC_NEGATIVE_PROMPT_SHA256 = (
    "007a1bdfe1ec3edf3b9a71789ca1999a47ad565560f269a3d78bf9a8dfef9cfd"
)
SEEDS = (17, 23, 41)
VIDEO_SECONDS = 189 / 24
SYNC_TIMEOUT_SECONDS = 5400
TOPOLOGY_ORDER = ("1x8", "2x4", "4x2", "8x1")
SINGLE_GPU_TOPOLOGY_ORDER = ("1x1",)
WORKLOAD = {
    "precision": "bf16",
    "size": "1280x720",
    "num_frames": 189,
    "fps": 24,
    "num_inference_steps": 35,
    "guidance_scale": 6.0,
    "flow_shift": 10.0,
    "max_sequence_length": 4096,
    "guardrails": False,
}


class Cosmos3SuperBenchmarkError(RuntimeError):
    """Raised when the fixed benchmark contract cannot be completed safely."""


@dataclass(frozen=True)
class Topology:
    name: str
    services: int
    gpus_per_service: int
    server_args: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkCell:
    """One measured cell, keeping request concurrency separate from services."""

    name: str
    topology: str
    request_concurrency_per_service: int = 1
    repeat_of: str = ""


TOPOLOGIES: dict[str, Topology] = {
    "1x1": Topology("1x1", 1, 1, ("--tensor-parallel-size", "1")),
    "1x8": Topology(
        "1x8",
        1,
        8,
        (
            "--cfg-parallel-size",
            "2",
            "--ulysses-degree",
            "4",
            "--use-hsdp",
            "--hsdp-shard-size",
            "8",
        ),
    ),
    "2x4": Topology("2x4", 2, 4, ("--tensor-parallel-size", "4")),
    "4x2": Topology("4x2", 4, 2, ("--tensor-parallel-size", "2")),
    "8x1": Topology("8x1", 8, 1, ("--tensor-parallel-size", "1")),
}

PRIMARY_CELLS = tuple(BenchmarkCell(name, name) for name in TOPOLOGY_ORDER)
B200_FULL_CELLS = (
    BenchmarkCell("T1_1x8", "1x8"),
    BenchmarkCell("T2_2x4", "2x4"),
    BenchmarkCell("T3_4x2", "4x2"),
    BenchmarkCell("T4_8x1", "8x1"),
    BenchmarkCell("T1C2", "1x8", 2),
    BenchmarkCell("T2C2", "2x4", 2),
    BenchmarkCell("T3C2", "4x2", 2),
    BenchmarkCell("T4C2", "8x1", 2),
    BenchmarkCell("T1R", "1x8", 1, "T1_1x8"),
    BenchmarkCell("T1C2R", "1x8", 2, "T1C2"),
)
H200_SINGLE_GPU_CELLS = (BenchmarkCell("H200_TP1_1GPU", "1x1"),)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256(value.encode("utf-8"))


def _public_prompt_asset(value: str, *, expected_sha256: str, name: str) -> str:
    """Return the exact public-record bytes, tolerating only terminal newlines."""
    if _sha256_text(value) == expected_sha256:
        return value
    normalized = value.rstrip("\r\n")
    if _sha256_text(normalized) == expected_sha256:
        return normalized
    raise Cosmos3SuperBenchmarkError(
        f"pinned {name} does not match the public benchmark SHA-256"
    )


def parse_topologies(value: str | Sequence[str]) -> tuple[str, ...]:
    raw = value.split(",") if isinstance(value, str) else value
    selected = tuple(str(item).strip() for item in raw if str(item).strip())
    if not selected:
        raise Cosmos3SuperBenchmarkError("at least one topology is required")
    unknown = [item for item in selected if item not in TOPOLOGIES]
    if unknown:
        raise Cosmos3SuperBenchmarkError(
            f"unknown topology {unknown[0]!r}; choose from "
            f"{', '.join((*TOPOLOGY_ORDER, *SINGLE_GPU_TOPOLOGY_ORDER))}"
        )
    if len(set(selected)) != len(selected):
        raise Cosmos3SuperBenchmarkError("topologies must not contain duplicates")
    return selected


def parse_suite(value: str) -> str:
    suite = value.strip().lower()
    if suite not in SUITE_CHOICES:
        raise Cosmos3SuperBenchmarkError(
            f"unknown suite {value!r}; choose from {', '.join(SUITE_CHOICES)}"
        )
    return suite


def benchmark_cells(
    *, suite: str, topologies: str | Sequence[str], attempts: int, gpu_family: str
) -> tuple[BenchmarkCell, ...]:
    selected_suite = parse_suite(suite)
    selected_topologies = parse_topologies(topologies)
    family = _normalize_gpu_family(gpu_family)
    if selected_suite == B200_FULL_SUITE:
        if family != "B200":
            raise Cosmos3SuperBenchmarkError(
                "the b200-full suite is the exact public B200 ten-cell record; "
                "use primary for H200"
            )
        if selected_topologies != TOPOLOGY_ORDER:
            raise Cosmos3SuperBenchmarkError(
                "the b200-full suite fixes topologies to 1x8,2x4,4x2,8x1"
            )
        if attempts != 24:
            raise Cosmos3SuperBenchmarkError(
                "the b200-full suite fixes exactly 24 measured attempts per cell"
            )
        return B200_FULL_CELLS
    if selected_suite == H200_SINGLE_GPU_SUITE:
        if family != "H200":
            raise Cosmos3SuperBenchmarkError(
                "the h200-single-gpu suite requires the H200 GPU family"
            )
        if selected_topologies != SINGLE_GPU_TOPOLOGY_ORDER:
            raise Cosmos3SuperBenchmarkError(
                "the h200-single-gpu suite fixes topologies to 1x1"
            )
        if attempts != 24:
            raise Cosmos3SuperBenchmarkError(
                "the h200-single-gpu suite fixes exactly 24 measured attempts"
            )
        return H200_SINGLE_GPU_CELLS
    return tuple(BenchmarkCell(name, name) for name in selected_topologies)


def _normalize_gpu_family(value: str) -> str:
    family = value.strip().upper()
    if family not in SUPPORTED_GPU_FAMILIES:
        raise Cosmos3SuperBenchmarkError(
            f"unsupported GPU family {value!r}; choose from "
            f"{', '.join(SUPPORTED_GPU_FAMILIES)}"
        )
    return family


def _schema_version(
    gpu_family: str, *, attempt: bool = False, suite: str = PRIMARY_SUITE
) -> str:
    if suite == H200_SINGLE_GPU_SUITE and not attempt:
        return "npa.cosmos3-super.h200-single-gpu-validation.v1"
    suffix = "attempt" if attempt else "benchmark"
    return f"npa.cosmos3-super.{gpu_family.lower()}-{suffix}.v1"


def benchmark_plan(
    *,
    output_path: str,
    topologies: str | Sequence[str],
    attempts: int = 24,
    gpu_family: str = "B200",
    suite: str = PRIMARY_SUITE,
) -> dict[str, Any]:
    family = _normalize_gpu_family(gpu_family)
    if attempts < 1:
        raise Cosmos3SuperBenchmarkError("attempts must be positive")
    cells = benchmark_cells(
        suite=suite,
        topologies=topologies,
        attempts=attempts,
        gpu_family=family,
    )
    for cell in cells:
        topology = TOPOLOGIES[cell.topology]
        if attempts % topology.services:
            raise Cosmos3SuperBenchmarkError(
                f"attempts={attempts} must divide evenly across {cell.topology}'s "
                f"{topology.services} services"
            )
    if not output_path.startswith("s3://") and not Path(output_path).is_absolute():
        raise Cosmos3SuperBenchmarkError(
            "output_path must be an s3:// URI or absolute local directory"
        )
    cell_plans = [
        {
            "name": cell.name,
            "topology": cell.topology,
            "services": TOPOLOGIES[cell.topology].services,
            "gpus_per_service": TOPOLOGIES[cell.topology].gpus_per_service,
            "server_parallelism": list(TOPOLOGIES[cell.topology].server_args),
            "request_concurrency_per_service": cell.request_concurrency_per_service,
            "warmups_per_service": 1,
            "measured_attempts": attempts,
            "repeat_of": cell.repeat_of or None,
        }
        for cell in cells
    ]
    return {
        "schema_version": _schema_version(family, suite=parse_suite(suite)),
        "status": "planned",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "runtime_image": IMAGE,
        "gpu": {
            "family": family,
            "node_gpu_count": max(
                TOPOLOGIES[cell.topology].services
                * TOPOLOGIES[cell.topology].gpus_per_service
                for cell in cells
            ),
        },
        "suite": parse_suite(suite),
        "upstream": {
            "method_revision": UPSTREAM_METHOD_REVISION,
            "b200_record_sha256": UPSTREAM_B200_RECORD_SHA256,
        },
        "planned_cells": cell_plans,
        "cells": cell_plans,
        # Kept for consumers of the original primary-only plan schema.
        "topologies": [
            {
                "name": cell["topology"],
                "services": cell["services"],
                "gpus_per_service": cell["gpus_per_service"],
                "server_parallelism": cell["server_parallelism"],
                "request_concurrency_per_service": cell[
                    "request_concurrency_per_service"
                ],
                "warmups_per_service": cell["warmups_per_service"],
                "measured_attempts": cell["measured_attempts"],
            }
            for cell in cell_plans
        ],
        "workload": dict(WORKLOAD),
        "seeds": list(SEEDS),
        "sync_timeout_seconds": SYNC_TIMEOUT_SECONDS,
        "output_path": output_path,
        "validation_scope": (
            {
                "kind": "single-gpu-functional-performance",
                "paper_reproduction": False,
                "paper_cell": None,
                "claim": (
                    "one H200, one TP-1 service, sequential requests; not the "
                    "paper's eight-replica 8x1 node cell"
                ),
            }
            if parse_suite(suite) == H200_SINGLE_GPU_SUITE
            else {
                "kind": "eight-gpu-node-benchmark",
                "paper_reproduction": True,
            }
        ),
    }


def _visible_gpu_count() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise Cosmos3SuperBenchmarkError(
            "nvidia-smi failed; the benchmark requires one visible eight-GPU B200 node"
        )
    return len([line for line in result.stdout.splitlines() if line.strip()])


def _gpu_name() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    names = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return next(iter(names)) if result.returncode == 0 and len(names) == 1 else ""


def _require_gpu_family(gpu_family: str, *, expected_count: int = 8) -> dict[str, Any]:
    family = _normalize_gpu_family(gpu_family)
    count = _visible_gpu_count()
    name = _gpu_name()
    if count != expected_count:
        raise Cosmos3SuperBenchmarkError(
            f"the selected suite requires exactly {expected_count} visible GPUs; "
            f"found {count}"
        )
    if family not in name.upper():
        raise Cosmos3SuperBenchmarkError(
            f"the benchmark requires {family} GPUs; "
            f"nvidia-smi reported {name or 'unknown'}"
        )
    return {"family": family, "node_gpu_count": count}


def _load_anchor_prompts() -> tuple[str, str, dict[str, str]]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise Cosmos3SuperBenchmarkError(
            "huggingface_hub is required to resolve the pinned model prompt assets"
        ) from exc
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    root = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            token=token,
            allow_patterns=[PROMPT_ASSET, NEGATIVE_PROMPT_ASSET],
        )
    )
    prompt = _public_prompt_asset(
        (root / PROMPT_ASSET).read_text(encoding="utf-8"),
        expected_sha256=PUBLIC_PROMPT_SHA256,
        name="prompt asset",
    )
    negative = _public_prompt_asset(
        (root / NEGATIVE_PROMPT_ASSET).read_text(encoding="utf-8"),
        expected_sha256=PUBLIC_NEGATIVE_PROMPT_SHA256,
        name="negative prompt asset",
    )
    if not prompt.strip() or not negative.strip():
        raise Cosmos3SuperBenchmarkError("pinned model prompt assets must be non-empty")
    return prompt, negative, {
        "prompt_sha256": _sha256_text(prompt),
        "negative_prompt_sha256": _sha256_text(negative),
    }


def service_command(topology: Topology, *, port: int) -> list[str]:
    return [
        "vllm",
        "serve",
        MODEL_ID,
        "--revision",
        MODEL_REVISION,
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--init-timeout",
        "1800",
        *topology.server_args,
        "--no-guardrails",
    ]


def _gpu_set(topology: Topology, replica: int) -> str:
    first = replica * topology.gpus_per_service
    return ",".join(str(index) for index in range(first, first + topology.gpus_per_service))


def _ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


@contextmanager
def running_services(
    topology: Topology, *, base_port: int, work_dir: Path, gpu_family: str = "B200"
) -> Iterator[list[str]]:
    processes: list[subprocess.Popen[bytes]] = []
    logs: list[Any] = []
    urls: list[str] = []
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        for replica in range(topology.services):
            port = base_port + replica
            log = (work_dir / f"service-r{replica}.log").open("wb")
            logs.append(log)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = _gpu_set(topology, replica)
            env["VLLM_OMNI_VIDEO_SYNC_TIMEOUT"] = str(SYNC_TIMEOUT_SECONDS)
            if _normalize_gpu_family(gpu_family) == "H200":
                env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            process = subprocess.Popen(
                service_command(topology, port=port),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            ready_url = f"http://127.0.0.1:{port}/v1/models"
            while not _ready(ready_url):
                if process.poll() is not None:
                    raise Cosmos3SuperBenchmarkError(
                        f"{topology.name} replica {replica} exited before readiness; "
                        "inspect the access-controlled workflow logs"
                    )
                time.sleep(5)
            urls.append(f"http://127.0.0.1:{port}/v1/videos/sync")
        yield urls
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()


def _run(command: list[str], *, binary: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=False, capture_output=True, text=not binary)


def validate_video(path: Path) -> dict[str, Any]:
    """Apply the full decode, shape, blank-frame, and basic-motion gate."""

    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        return {"valid": False, "errors": [{"check": "dependency", "detail": missing}]}
    errors: list[dict[str, Any]] = []
    decoded = _run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"]
    )
    if decoded.returncode != 0 or decoded.stderr.strip():
        errors.append(
            {"check": "decode", "detail": decoded.stderr.strip() or "nonzero exit"}
        )
    probed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    stream: dict[str, Any] = {}
    if probed.returncode:
        errors.append({"check": "ffprobe", "detail": probed.stderr.strip()})
    else:
        try:
            stream = (json.loads(probed.stdout).get("streams") or [{}])[0]
        except (json.JSONDecodeError, IndexError) as exc:
            errors.append({"check": "ffprobe", "detail": str(exc)})
    if stream:
        if (stream.get("width"), stream.get("height")) != (1280, 720):
            errors.append(
                {"check": "geometry", "detail": f"{stream.get('width')}x{stream.get('height')}"}
            )
        try:
            fps = float(Fraction(stream["avg_frame_rate"]))
            if not math.isclose(fps, 24.0, abs_tol=0.001):
                errors.append({"check": "fps", "detail": fps})
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            errors.append({"check": "fps", "detail": str(exc)})
        try:
            frames = int(stream["nb_read_frames"])
            if frames != 189:
                errors.append({"check": "frames", "detail": frames})
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({"check": "frames", "detail": str(exc)})
        try:
            duration = float(stream["duration"])
            if not 7.80 <= duration <= 7.95:
                errors.append({"check": "duration", "detail": duration})
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({"check": "duration", "detail": str(exc)})
    else:
        errors.append({"check": "stream", "detail": "no video stream"})

    sampled = _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "select='eq(n,0)+eq(n,47)+eq(n,94)+eq(n,141)+eq(n,188)',scale=32:32,format=gray",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-",
        ],
        binary=True,
    )
    sample_size = 32 * 32
    samples: list[dict[str, float]] = []
    if sampled.returncode or len(sampled.stdout) != 5 * sample_size:
        errors.append(
            {
                "check": "sampling",
                "detail": f"expected {5 * sample_size} bytes, got {len(sampled.stdout)}",
            }
        )
    else:
        frames = [
            sampled.stdout[offset : offset + sample_size]
            for offset in range(0, len(sampled.stdout), sample_size)
        ]
        samples = [
            {
                "mean_luma": round(statistics.mean(frame), 3),
                "spatial_stdev": round(statistics.pstdev(frame), 3),
            }
            for frame in frames
        ]
        if max(sample["spatial_stdev"] for sample in samples) < 1.0:
            errors.append({"check": "blank", "detail": "sampled frames lack detail"})
        diffs = [
            statistics.mean(abs(left - right) for left, right in zip(frames[0], frame))
            for frame in frames[1:]
        ]
        if max(diffs, default=0.0) < 0.5:
            errors.append({"check": "frozen", "detail": "sampled frames do not change"})
    return {
        "valid": not errors,
        "errors": errors,
        "stream": stream,
        "sampled_frames": samples,
    }


def _multipart(fields: Mapping[str, str]) -> tuple[bytes, str]:
    boundary = f"npa-cosmos3-{os.urandom(12).hex()}"
    body = b"".join(
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
            f"{value}\r\n"
        ).encode()
        for key, value in fields.items()
    ) + f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _request_fields(prompt: str, negative: str, seed: int) -> dict[str, str]:
    return {
        "prompt": prompt,
        "negative_prompt": negative,
        "size": "1280x720",
        "num_frames": "189",
        "fps": "24",
        "num_inference_steps": "35",
        "guidance_scale": "6.0",
        "flow_shift": "10.0",
        "max_sequence_length": "4096",
        "seed": str(seed),
        "extra_params": json.dumps(
            {
                "use_resolution_template": False,
                "use_duration_template": False,
                "guardrails": False,
            },
            separators=(",", ":"),
        ),
    }


def one_attempt(
    *,
    url: str,
    prompt: str,
    negative_prompt: str,
    prompt_hashes: Mapping[str, str],
    seed: int,
    attempt_id: str,
    replica: int,
    clip_path: Path,
    kind: str,
    gpu_family: str = "B200",
) -> dict[str, Any]:
    body, content_type = _multipart(_request_fields(prompt, negative_prompt, seed))
    started_at = _utc_now()
    started = time.monotonic()
    record: dict[str, Any] = {
        "schema_version": _schema_version(
            _normalize_gpu_family(gpu_family), attempt=True
        ),
        "attempt_id": attempt_id,
        "kind": kind,
        "replica": replica,
        "seed": seed,
        **prompt_hashes,
        "started_at": started_at,
        "finished_at": "",
        "client_wall_seconds": 0.0,
        "http_status": None,
        "output_bytes": 0,
        "output_sha256": "",
        "technical_valid": False,
        "validation": None,
        "failure_reason": None,
    }
    try:
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Accept": "video/mp4", "Content-Type": content_type},
        )
        with urllib.request.urlopen(request, timeout=SYNC_TIMEOUT_SECONDS) as response:
            payload = response.read()
            record["http_status"] = response.status
        record["output_bytes"] = len(payload)
        record["output_sha256"] = _sha256(payload)
        if response.status != 200:
            record["failure_reason"] = f"http_{response.status}"
        elif not payload:
            record["failure_reason"] = "empty_response"
        else:
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            clip_path.write_bytes(payload)
            validation = validate_video(clip_path)
            record["validation"] = validation
            record["technical_valid"] = bool(validation["valid"])
            if not validation["valid"]:
                checks = ",".join(item["check"] for item in validation["errors"])
                record["failure_reason"] = f"video_invalid:{checks}"
    except urllib.error.HTTPError as exc:
        record["http_status"] = exc.code
        record["failure_reason"] = f"http_{exc.code}"
    except Exception as exc:  # noqa: BLE001 - every attempt remains in the denominator
        record["failure_reason"] = f"{type(exc).__name__}:{exc}"
    record["client_wall_seconds"] = round(time.monotonic() - started, 6)
    record["finished_at"] = _utc_now()
    return record


def _strict_valid(record: Mapping[str, Any]) -> bool:
    return bool(
        record.get("http_status") == 200
        and int(record.get("output_bytes") or 0) > 0
        and record.get("technical_valid") is True
        and record.get("failure_reason") is None
        and isinstance(record.get("validation"), Mapping)
        and record["validation"].get("valid") is True
    )


def derive_cell(records: Sequence[Mapping[str, Any]], window_seconds: float) -> dict[str, Any]:
    if window_seconds <= 0:
        raise Cosmos3SuperBenchmarkError("measurement window must be positive")
    valid = [record for record in records if _strict_valid(record)]
    latencies = [float(record["client_wall_seconds"]) for record in valid]
    video_seconds = len(valid) * VIDEO_SECONDS
    return {
        "attempts": len(records),
        "valid_attempts": len(valid),
        "failed_attempts": len(records) - len(valid),
        "technical_validity_yield": round(len(valid) / len(records), 6)
        if records
        else None,
        "mean_request_latency_seconds": round(statistics.mean(latencies), 3)
        if latencies
        else None,
        "median_request_latency_seconds": round(statistics.median(latencies), 3)
        if latencies
        else None,
        "window_seconds": round(window_seconds, 6),
        "credited_valid_video_seconds": round(video_seconds, 3),
        "valid_video_seconds_per_node_hour": round(video_seconds * 3600 / window_seconds, 1),
        "failed_attempt_video_seconds_credit": 0,
    }


def _add_resource_normalized_metrics(
    derived: dict[str, Any], *, gpu_count: int, service_count: int
) -> None:
    """Add explicit resource-hour rates without changing node-rate semantics."""

    credited = float(derived["credited_valid_video_seconds"])
    window = float(derived["window_seconds"])
    derived["valid_video_seconds_per_gpu_hour"] = round(
        credited * 3600 / (window * gpu_count), 1
    )
    derived["valid_video_seconds_per_service_hour"] = round(
        credited * 3600 / (window * service_count), 1
    )


def derive_suite_comparisons(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Derive the public record's primary, concurrency, and repeat comparisons."""

    required = {cell.name for cell in B200_FULL_CELLS}
    if not required.issubset(cells):
        return {}

    def metric(name: str, key: str) -> float:
        return float(cells[name]["derived"][key])

    def delta(value: float, baseline: float) -> float:
        return round((value / baseline - 1.0) * 100.0, 2)

    primary_names = ("T1_1x8", "T2_2x4", "T3_4x2", "T4_8x1")
    concurrency_pairs = (
        ("T1C2", "T1_1x8"),
        ("T2C2", "T2_2x4"),
        ("T3C2", "T3_4x2"),
        ("T4C2", "T4_8x1"),
    )
    repeat_pairs = (("T1R", "T1_1x8"), ("T1C2R", "T1C2"))
    latency_key = "mean_request_latency_seconds"
    throughput_key = "valid_video_seconds_per_node_hour"
    return {
        "primary_frontier": {
            name: {
                "mean_request_latency_seconds": metric(name, latency_key),
                "valid_video_seconds_per_node_hour": metric(name, throughput_key),
                "latency_delta_vs_T1_pct": delta(
                    metric(name, latency_key), metric("T1_1x8", latency_key)
                ),
                "throughput_delta_vs_T1_pct": delta(
                    metric(name, throughput_key), metric("T1_1x8", throughput_key)
                ),
            }
            for name in primary_names
        },
        "concurrency_two_delta_pct": {
            current: {
                "mean_request_latency": delta(
                    metric(current, latency_key), metric(baseline, latency_key)
                ),
                "valid_video_seconds_per_node_hour": delta(
                    metric(current, throughput_key), metric(baseline, throughput_key)
                ),
            }
            for current, baseline in concurrency_pairs
        },
        "repeat_variation_pct": {
            current: {
                "mean_request_latency": delta(
                    metric(current, latency_key), metric(baseline, latency_key)
                ),
                "valid_video_seconds_per_node_hour": delta(
                    metric(current, throughput_key), metric(baseline, throughput_key)
                ),
            }
            for current, baseline in repeat_pairs
        },
    }


AttemptFn = Callable[..., dict[str, Any]]


def dispatch_cell(
    *,
    topology: Topology,
    urls: Sequence[str],
    attempts: int,
    prompt: str,
    negative_prompt: str,
    prompt_hashes: Mapping[str, str],
    clips_dir: Path,
    kind: str,
    gpu_family: str = "B200",
    request_concurrency_per_service: int = 1,
    cell_name: str = "",
    attempt_fn: AttemptFn = one_attempt,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(urls) != topology.services:
        raise Cosmos3SuperBenchmarkError("URL count does not match service topology")
    if attempts % topology.services:
        raise Cosmos3SuperBenchmarkError("attempts must divide evenly across services")
    if request_concurrency_per_service < 1:
        raise Cosmos3SuperBenchmarkError("request concurrency must be positive")
    per_service = attempts // topology.services
    workers_per_service = min(request_concurrency_per_service, per_service)
    barrier = threading.Barrier(topology.services * workers_per_service)
    boundary_lock = threading.Lock()
    first_dispatch: float | None = None
    final_completion: float | None = None

    def attempt(replica: int, index: int) -> dict[str, Any]:
        nonlocal first_dispatch, final_completion
        if index < workers_per_service:
            barrier.wait()
        identity = cell_name or topology.name
        attempt_id = f"{kind}-{identity}-r{replica}-a{index:03d}"
        dispatched = time.monotonic()
        with boundary_lock:
            if first_dispatch is None or dispatched < first_dispatch:
                first_dispatch = dispatched
        row = attempt_fn(
            url=urls[replica],
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_hashes=prompt_hashes,
            seed=SEEDS[index % len(SEEDS)],
            attempt_id=attempt_id,
            replica=replica,
            clip_path=clips_dir / f"{attempt_id}.mp4",
            kind=kind,
            gpu_family=gpu_family,
        )
        completed = time.monotonic()
        with boundary_lock:
            if final_completion is None or completed > final_completion:
                final_completion = completed
        row["cell"] = identity
        row["topology"] = topology.name
        row["request_concurrency_per_service"] = request_concurrency_per_service
        row["service_attempt_index"] = index
        return row

    def worker(replica: int) -> list[dict[str, Any]]:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers_per_service
        ) as request_pool:
            futures = [
                request_pool.submit(attempt, replica, index)
                for index in range(per_service)
            ]
            return [future.result() for future in futures]

    with concurrent.futures.ThreadPoolExecutor(max_workers=topology.services) as pool:
        groups = list(pool.map(worker, range(topology.services)))
    if first_dispatch is None or final_completion is None:
        raise Cosmos3SuperBenchmarkError("measurement produced no request boundaries")
    rows = [row for group in groups for row in group]
    rows.sort(key=lambda row: str(row["attempt_id"]))
    window = {
        "started_at": min(str(row["started_at"]) for row in rows),
        "finished_at": max(str(row["finished_at"]) for row in rows),
        "seconds": round(final_completion - first_dispatch, 6),
        "boundary": (
            "shared first-dispatch-to-final-completion client window; includes routing, "
            "generation, MP4 encoding, uneven completion, failures, and tail idle time; "
            "excludes startup, model load, and warmup"
        ),
        "request_concurrency_per_service": request_concurrency_per_service,
    }
    return rows, window


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _cell_contract_sha256(
    *,
    plan: Mapping[str, Any],
    cell: BenchmarkCell,
    prompt_hashes: Mapping[str, str],
    run_id: str,
) -> str:
    planned = next(
        item for item in plan["planned_cells"] if item["name"] == cell.name
    )
    return _sha256(
        _canonical_json_bytes(
            {
                "schema_version": plan["schema_version"],
                "suite": plan["suite"],
                "run_id": run_id,
                "model": plan["model"],
                "runtime_image": plan["runtime_image"],
                "gpu": plan["gpu"],
                "workload": plan["workload"],
                "seeds": plan["seeds"],
                "sync_timeout_seconds": plan["sync_timeout_seconds"],
                "cell": planned,
                "prompt_hashes": dict(prompt_hashes),
            }
        )
    )


def _read_json_object(storage: Any, uri: str) -> tuple[dict[str, Any], bytes] | None:
    found = storage.read_bytes_with_etag(uri)
    if found is None:
        return None
    payload, _etag = found
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise Cosmos3SuperBenchmarkError(
            "durable resume object is not valid JSON; refusing to overwrite it"
        ) from exc
    if not isinstance(parsed, dict):
        raise Cosmos3SuperBenchmarkError(
            "durable resume object must contain a JSON object"
        )
    return parsed, payload


def _load_completed_cell(
    *, storage: Any, output_path: str, cell: BenchmarkCell, contract_sha256: str
) -> dict[str, Any] | None:
    cell_uri = f"{output_path.rstrip('/')}/cells/{cell.name}"
    marker_result = _read_json_object(storage, f"{cell_uri}/complete.json")
    if marker_result is None:
        return None
    marker, _marker_bytes = marker_result
    if marker.get("contract_sha256") != contract_sha256:
        raise Cosmos3SuperBenchmarkError(
            f"completed cell {cell.name} has a different immutable contract; "
            "choose a new output path"
        )
    cell_result = _read_json_object(storage, f"{cell_uri}/cell.json")
    if cell_result is None:
        raise Cosmos3SuperBenchmarkError(
            f"completed cell {cell.name} is missing its durable cell record"
        )
    payload, payload_bytes = cell_result
    if _sha256(payload_bytes) != marker.get("cell_json_sha256"):
        raise Cosmos3SuperBenchmarkError(
            f"completed cell {cell.name} failed its durable hash check"
        )
    attempts = payload.get("attempts")
    derived = payload.get("derived")
    if (
        not isinstance(attempts, list)
        or not isinstance(derived, Mapping)
        or len(attempts) != int(derived.get("attempts") or -1)
    ):
        raise Cosmos3SuperBenchmarkError(
            f"completed cell {cell.name} failed its attempt-count audit"
        )
    return payload


def _publish_completed_cell(
    *,
    storage: Any,
    output_path: str,
    cell: BenchmarkCell,
    cell_dir: Path,
    payload: Mapping[str, Any],
    contract_sha256: str,
) -> None:
    cell_uri = f"{output_path.rstrip('/')}/cells/{cell.name}"
    cell_json = _canonical_json_bytes(payload)
    (cell_dir / "cell.json").write_bytes(cell_json)
    storage.upload_directory(str(cell_dir), cell_uri + "/")
    marker = {
        "schema_version": "npa.cosmos3-super.cell-completion.v1",
        "cell": cell.name,
        "contract_sha256": contract_sha256,
        "cell_json_sha256": _sha256(cell_json),
        "attempts": len(payload["attempts"]),
        "valid_attempts": int(payload["derived"]["valid_attempts"]),
        "failed_attempts": int(payload["derived"]["failed_attempts"]),
        "published_at": _utc_now(),
    }
    marker_bytes = _canonical_json_bytes(marker)
    marker_uri = f"{cell_uri}/complete.json"
    try:
        storage.put_bytes_conditional(
            marker_bytes,
            marker_uri,
            if_none_match=True,
            content_type="application/json",
        )
    except StoragePreconditionFailed:
        existing = _read_json_object(storage, marker_uri)
        if existing is None or existing[0].get("contract_sha256") != contract_sha256:
            raise Cosmos3SuperBenchmarkError(
                f"completed cell {cell.name} was concurrently replaced by a different run"
            )
    verified = _read_json_object(storage, marker_uri)
    if verified is None or verified[1] != marker_bytes:
        raise Cosmos3SuperBenchmarkError(
            f"completed cell {cell.name} failed read-after-write verification"
        )


def _publish(local_dir: Path, output_path: str, storage_client: Any = None) -> str:
    if output_path.startswith("s3://"):
        return str(
            (storage_client or StorageClient.from_environment()).upload_directory(
                str(local_dir), output_path.rstrip("/") + "/"
            )
        )
    target = Path(output_path)
    if target.exists():
        raise Cosmos3SuperBenchmarkError(f"local output already exists: {target}")
    shutil.copytree(local_dir, target)
    return str(target)


def run_benchmark(
    *,
    output_path: str,
    topologies: str | Sequence[str] = TOPOLOGY_ORDER,
    attempts: int = 24,
    base_port: int = 8100,
    run_id: str = "",
    dry_run: bool = False,
    storage_client: Any = None,
    gpu_family: str = "B200",
    suite: str = PRIMARY_SUITE,
) -> dict[str, Any]:
    """Run and publish a fixed primary sweep or the exact ten-cell B200 suite."""

    plan = benchmark_plan(
        output_path=output_path,
        topologies=topologies,
        attempts=attempts,
        gpu_family=gpu_family,
        suite=suite,
    )
    plan["run_id"] = run_id
    if dry_run:
        return plan
    if os.environ.get("NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE") != "YES":
        raise Cosmos3SuperBenchmarkError(
            "set NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE=YES for this run after "
            "reviewing the vLLM-Omni container's NVIDIA runtime terms"
        )
    family = _normalize_gpu_family(gpu_family)
    gpu = _require_gpu_family(family, expected_count=int(plan["gpu"]["node_gpu_count"]))
    prompt, negative, prompt_hashes = _load_anchor_prompts()
    selected_cells = benchmark_cells(
        suite=suite,
        topologies=topologies,
        attempts=attempts,
        gpu_family=family,
    )
    remote = output_path.startswith("s3://")
    storage = (
        storage_client or StorageClient.from_environment()
    ) if remote else None
    with tempfile.TemporaryDirectory(
        prefix=f"npa-cosmos3-super-{family.lower()}-"
    ) as tmp:
        root = Path(tmp)
        cells: dict[str, Any] = {}
        execution: dict[str, str] = {}
        for cell in selected_cells:
            topology = TOPOLOGIES[cell.topology]
            cell_dir = root / "cells" / cell.name
            log_dir = root / "private-service-logs" / cell.name
            contract_sha256 = _cell_contract_sha256(
                plan=plan,
                cell=cell,
                prompt_hashes=prompt_hashes,
                run_id=run_id,
            )
            if remote:
                completed = _load_completed_cell(
                    storage=storage,
                    output_path=output_path,
                    cell=cell,
                    contract_sha256=contract_sha256,
                )
                if completed is not None:
                    cells[cell.name] = completed
                    execution[cell.name] = "resumed"
                    continue
            with running_services(
                topology,
                base_port=base_port,
                work_dir=log_dir,
                gpu_family=family,
            ) as urls:
                warmups, _ = dispatch_cell(
                    topology=topology,
                    urls=urls,
                    attempts=topology.services,
                    prompt=prompt,
                    negative_prompt=negative,
                    prompt_hashes=prompt_hashes,
                    clips_dir=cell_dir / "warmup-clips",
                    kind="warmup",
                    gpu_family=family,
                    request_concurrency_per_service=1,
                    cell_name=cell.name,
                )
                if not all(_strict_valid(row) for row in warmups):
                    _write_json(cell_dir / "warmups.json", warmups)
                    raise Cosmos3SuperBenchmarkError(
                        f"{cell.name} warmup validation failed; "
                        "measurement window was not opened"
                    )
                shutil.rmtree(cell_dir / "warmup-clips", ignore_errors=True)
                records, window = dispatch_cell(
                    topology=topology,
                    urls=urls,
                    attempts=attempts,
                    prompt=prompt,
                    negative_prompt=negative,
                    prompt_hashes=prompt_hashes,
                    clips_dir=cell_dir / "clips",
                    kind="production",
                    gpu_family=family,
                    request_concurrency_per_service=(
                        cell.request_concurrency_per_service
                    ),
                    cell_name=cell.name,
                )
            derived = derive_cell(records, float(window["seconds"]))
            _add_resource_normalized_metrics(
                derived,
                gpu_count=topology.services * topology.gpus_per_service,
                service_count=topology.services,
            )
            _write_json(cell_dir / "attempts.json", records)
            _write_json(cell_dir / "window.json", window)
            _write_json(cell_dir / "derived.json", derived)
            cell_payload = {
                "cell": cell.name,
                "topology": {
                    "name": topology.name,
                    "services": topology.services,
                    "gpus_per_service": topology.gpus_per_service,
                    "server_parallelism": list(topology.server_args),
                    "request_concurrency_per_service": (
                        cell.request_concurrency_per_service
                    ),
                    "warmups_per_service": 1,
                },
                "repeat_of": cell.repeat_of or None,
                "warmups": warmups,
                "attempts": records,
                "window": window,
                "derived": derived,
                "contract_sha256": contract_sha256,
            }
            cells[cell.name] = cell_payload
            if remote:
                _publish_completed_cell(
                    storage=storage,
                    output_path=output_path,
                    cell=cell,
                    cell_dir=cell_dir,
                    payload=cell_payload,
                    contract_sha256=contract_sha256,
                )
            execution[cell.name] = "executed"
        shutil.rmtree(root / "private-service-logs", ignore_errors=True)
        artifact_uri = output_path.rstrip("/") + "/" if remote else output_path
        total_attempts = sum(
            int(cell["derived"]["attempts"]) for cell in cells.values()
        )
        valid_attempts = sum(
            int(cell["derived"]["valid_attempts"]) for cell in cells.values()
        )
        failed_attempts = total_attempts - valid_attempts
        report = {
            **plan,
            "status": "succeeded"
            if failed_attempts == 0
            else "completed_with_invalid_attempts",
            "run_id": run_id,
            "gpu": gpu,
            "prompt_hashes": prompt_hashes,
            "cells": cells,
            "execution": execution,
            "audit": {
                "expected_cells": len(selected_cells),
                "completed_cells": len(cells),
                "expected_attempts": len(selected_cells) * attempts,
                "attempts": total_attempts,
                "valid_attempts": valid_attempts,
                "failed_attempts": failed_attempts,
                "all_attempt_records_present": (
                    total_attempts == len(selected_cells) * attempts
                ),
            },
            "comparisons": derive_suite_comparisons(cells),
            "completed_at": _utc_now(),
            "measurement_claim": (
                "single-GPU H200 TP-1 functional/performance validation; technical "
                "validity only; not eight-GPU node throughput or the paper's 8x1 cell"
                if parse_suite(suite) == H200_SINGLE_GPU_SUITE
                else "technical validity only; semantic quality was not measured"
            ),
            "artifact_uri": artifact_uri,
        }
        _write_json(root / "benchmark.json", report)
        if remote:
            published = storage.upload_file(
                str(root / "benchmark.json"),
                f"{output_path.rstrip('/')}/benchmark.json",
            )
            verified = _read_json_object(
                storage, f"{output_path.rstrip('/')}/benchmark.json"
            )
            expected_bytes = (root / "benchmark.json").read_bytes()
            if verified is None or verified[1] != expected_bytes:
                raise Cosmos3SuperBenchmarkError(
                    "final benchmark report failed read-after-write verification"
                )
            published = output_path.rstrip("/") + "/"
        else:
            published = _publish(root, output_path)
        report["artifact_uri"] = published
        if report["status"] != "succeeded":
            raise Cosmos3SuperBenchmarkError(
                f"benchmark completed with invalid attempts; evidence retained at {published}"
            )
        return report


__all__ = [
    "ATTEMPT_SCHEMA_VERSION",
    "B200_FULL_CELLS",
    "B200_FULL_SUITE",
    "H200_SINGLE_GPU_CELLS",
    "H200_SINGLE_GPU_SUITE",
    "BenchmarkCell",
    "Cosmos3SuperBenchmarkError",
    "IMAGE",
    "MODEL_ID",
    "MODEL_REVISION",
    "PRIMARY_SUITE",
    "SCHEMA_VERSION",
    "SUPPORTED_GPU_FAMILIES",
    "TOPOLOGIES",
    "TOPOLOGY_ORDER",
    "SINGLE_GPU_TOPOLOGY_ORDER",
    "WORKLOAD",
    "benchmark_cells",
    "benchmark_plan",
    "derive_cell",
    "derive_suite_comparisons",
    "dispatch_cell",
    "parse_suite",
    "parse_topologies",
    "run_benchmark",
    "service_command",
    "validate_video",
]
