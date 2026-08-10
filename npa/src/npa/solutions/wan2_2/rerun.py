"""Wan 2.2 solution-scoped Rerun recording and verification.

The recording is an evidence view over artifacts that a Wan BYOF smoke already
produced.  It embeds the generated MP4 byte-for-byte and logs only static facts
that are present in the run JSONs; it does not manufacture rollout telemetry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from npa.clients.project_credentials import s3_client_for_project
from npa.deploy.images import wan_accepted_image_manifest

APPLICATION_ID = "npa_wan2_2"
ENTITY_ROOT = "wan2_2"
VIDEO_ASSET_ENTITY = f"{ENTITY_ROOT}/video/asset"
VIDEO_FRAME_ENTITY = f"{ENTITY_ROOT}/video/frame"
VIDEO_TIMELINE = "video_time"
RRD_CAPABILITY = "wan2.2_verified_rerun_recording"
RRD_MANIFEST_SCHEMA = "npa.workbench.wan2_2.rerun_manifest.v1"
RRD_SUMMARY_SCHEMA = "npa.workbench.wan2_2.rerun_summary.v1"
SOURCE_REPO = "https://github.com/Wan-Video/Wan2.2.git"
SOURCE_REF = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
MODEL_REF = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
TOKENIZER_ID = "google/umt5-xxl"
TOKENIZER_REF = (
    "66cb9e7e85526fe440a945569e42c72fb6cbc0ad"  # gitleaks:allow; public revision
)


class WanRrdError(RuntimeError):
    """Raised when source evidence or the generated recording fails closed."""


@dataclass(frozen=True)
class WanRunLayout:
    variant: str
    primary_filenames: tuple[str, ...]
    primary_schema: str
    video_filename: str
    inventory_filename: str
    topology_filename: str | None
    rank_filenames: tuple[str, ...]
    additional_evidence_filenames: tuple[str, ...]
    rrd_filename: str
    manifest_filename: str


SINGLE_GPU_LAYOUT = WanRunLayout(
    variant="single-gpu",
    primary_filenames=(
        "wan2_2_ti2v_5b_text_to_video.json",
        "wan2_2_ti2v_5b_image_to_video.json",
    ),
    primary_schema="npa.workbench.byof.wan2_2_ti2v_5b.v1",
    video_filename="wan2_2_ti2v_5b.mp4",
    inventory_filename="wan2_2_runtime_inventory.json",
    topology_filename=None,
    rank_filenames=(),
    additional_evidence_filenames=(),
    rrd_filename="wan2_2_ti2v_5b.rrd",
    manifest_filename="wan2_2_ti2v_5b_rrd_manifest.json",
)
MULTI_GPU_LAYOUT = WanRunLayout(
    variant="multigpu",
    primary_filenames=("wan2_2_ti2v_5b_multigpu.json",),
    primary_schema="npa.workbench.byof.wan2_2_ti2v_5b_multigpu.v1",
    video_filename="wan2_2_ti2v_5b_multigpu.mp4",
    inventory_filename="wan2_2_multigpu_runtime_inventory.json",
    topology_filename="wan2_2_multigpu_topology.json",
    rank_filenames=tuple(f"wan2_2_multigpu_rank_{rank}.json" for rank in range(4)),
    additional_evidence_filenames=(
        *(f"wan2_2_multigpu_progress_rank_{rank}.json" for rank in range(4)),
        "wan2_2_multigpu_nccl_summary.json",
        *(f"wan2_2_multigpu_nccl_rank_{rank}.log" for rank in range(4)),
    ),
    rrd_filename="wan2_2_ti2v_5b_multigpu.rrd",
    manifest_filename="wan2_2_ti2v_5b_multigpu_rrd_manifest.json",
)

WAN_SOLUTION_LAYOUTS = {
    "wan2.2": SINGLE_GPU_LAYOUT,
    "wan2.2-multigpu": MULTI_GPU_LAYOUT,
}


def layout_for_solution(solution_name: str) -> WanRunLayout | None:
    """Return the automatic postprocessor layout for a BYOF solution name."""

    return WAN_SOLUTION_LAYOUTS.get(solution_name.strip().lower())


def layout_for_variant(variant: str) -> WanRunLayout:
    normalized = variant.strip().lower().replace("_", "-")
    if normalized in {"single", "single-gpu", "rtxpro"}:
        return SINGLE_GPU_LAYOUT
    if normalized in {"multi", "multigpu", "multi-gpu", "b200"}:
        return MULTI_GPU_LAYOUT
    raise WanRrdError(f"unknown Wan RRD variant: {variant!r}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WanRrdError(message)


def _require_mapping(value: Any, message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WanRrdError(message)
    return value


def _require_list(value: Any, message: str) -> list[Any]:
    if not isinstance(value, list):
        raise WanRrdError(message)
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WanRrdError(f"cannot read JSON evidence {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WanRrdError(f"JSON evidence must be an object: {path.name}")
    return payload


def _validate_container_image(summary: dict[str, Any]) -> dict[str, str]:
    """Bind run/RRD evidence to the repository's accepted immutable image."""

    reference = str(summary.get("image") or "").strip()
    match = re.fullmatch(r"([^\s@]+)@(sha256:[0-9a-f]{64})", reference)
    if match is None:
        raise WanRrdError("BYOF summary image must be digest-pinned")
    try:
        accepted = wan_accepted_image_manifest()
    except (OSError, RuntimeError, ValueError) as exc:
        raise WanRrdError(
            f"cannot load the accepted Wan image manifest: {exc}"
        ) from exc
    accepted_digest = str(accepted.get("oci_digest") or "")
    accepted_tag = str(accepted.get("tag") or "")
    _require(
        bool(re.fullmatch(r"sha256:[0-9a-f]{64}", accepted_digest)),
        "accepted Wan image digest is invalid",
    )
    _require(
        match.group(2) == accepted_digest, "BYOF summary image digest is not accepted"
    )
    _require(bool(accepted_tag), "accepted Wan image tag is invalid")
    return {
        "reference": reference,
        "oci_digest": accepted_digest,
        "accepted_tag": accepted_tag,
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _primary_path(run_dir: Path, layout: WanRunLayout, summary: dict[str, Any]) -> Path:
    declared = str(summary.get("smoke_artifact_name") or "").strip()
    if declared:
        _require(
            Path(declared).name == declared, "smoke artifact name must be a basename"
        )
        _require(
            declared in layout.primary_filenames,
            f"unexpected Wan primary artifact for {layout.variant}: {declared}",
        )
        candidate = run_dir / declared
        _require(candidate.is_file(), f"missing Wan primary artifact: {declared}")
        return candidate
    matches = [
        run_dir / name
        for name in layout.primary_filenames
        if (run_dir / name).is_file()
    ]
    _require(len(matches) == 1, "expected exactly one Wan primary capability artifact")
    return matches[0]


def _decode_video_metrics(video_path: Path) -> dict[str, Any]:
    """Independently decode the downloaded MP4 rather than trusting run JSON."""

    try:
        import av
    except ModuleNotFoundError as exc:
        raise WanRrdError(
            "Wan MP4 verification requires PyAV; install the npa[viz] extra"
        ) from exc

    try:
        with av.open(str(video_path), mode="r") as container:
            stream = container.streams.video[0]
            codec = str(stream.codec_context.name or "").lower()
            width = int(stream.codec_context.width)
            height = int(stream.codec_context.height)
            fps = float(stream.average_rate or 0)
            decoded_frames = [
                frame.to_ndarray(format="rgb24") for frame in container.decode(stream)
            ]
    except (IndexError, OSError, TypeError, ValueError, av.FFmpegError) as exc:
        raise WanRrdError(
            f"downloaded MP4 probe failed or decode failed: {exc}"
        ) from exc

    _require(codec == "h264", "downloaded source video is not H.264")
    _require(width > 0 and height > 0 and fps > 0, "downloaded MP4 metadata is invalid")
    frame_count = len(decoded_frames)
    _require(frame_count > 0, "downloaded MP4 has no decodable frames")
    _require(
        all(frame.shape == (height, width, 3) for frame in decoded_frames),
        "downloaded MP4 decoded frame geometry is inconsistent",
    )
    frames = np.stack(decoded_frames)
    max_spatial_std = max(float(np.std(frame)) for frame in frames)
    pixel_range = int(frames.max()) - int(frames.min())
    mean_temporal_abs_delta = (
        float(
            np.mean(
                np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32))
            )
        )
        if frame_count > 1
        else 0.0
    )
    return {
        "codec": codec,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": fps,
        "max_spatial_std": max_spatial_std,
        "pixel_range": pixel_range,
        "mean_temporal_abs_delta": mean_temporal_abs_delta,
    }


def _validate_output(
    primary: dict[str, Any], video_path: Path, layout: WanRunLayout
) -> dict[str, Any]:
    _require(video_path.is_file(), f"missing generated MP4: {video_path.name}")
    video_size = video_path.stat().st_size
    video_sha256 = _sha256_file(video_path)
    observed = _require_mapping(
        primary.get("observed"), "primary artifact has no observed video metrics"
    )
    _require(int(observed.get("width") or 0) > 0, "observed video width is invalid")
    _require(int(observed.get("height") or 0) > 0, "observed video height is invalid")
    _require(
        int(observed.get("frame_count") or 0) > 0, "observed frame count is invalid"
    )
    _require(float(observed.get("fps") or 0.0) > 0.0, "observed FPS is invalid")
    _require(
        float(observed.get("max_spatial_std") or 0.0) >= 1.0,
        "source output lacks the required spatial variation",
    )
    _require(
        int(observed.get("pixel_range") or 0) >= 4,
        "source output lacks the required pixel range",
    )
    _require(
        float(observed.get("mean_temporal_abs_delta") or 0.0) > 0.001,
        "source output lacks the required temporal variation",
    )
    capabilities = set(primary.get("capabilities_exercised") or [])
    _require(
        "wan2.2_decoded_mp4_validation" in capabilities,
        "decoded MP4 hard gate is absent",
    )
    _require(
        not primary.get("deferred"),
        "source capability artifact has hard-gate deferrals",
    )
    decoded = _decode_video_metrics(video_path)
    _require(video_size >= 4096, "downloaded MP4 is implausibly small")
    _require(
        decoded["width"] == 1280
        and decoded["height"] == 704
        and decoded["frame_count"] == 17
        and abs(float(decoded["fps"]) - 24.0) <= 0.01,
        "downloaded MP4 violates the accepted 1280x704/17-frame/24-fps contract",
    )
    for key in ("width", "height", "frame_count", "pixel_range"):
        _require(
            int(observed.get(key) or -1) == int(decoded[key]),
            f"downloaded MP4 disagrees with reported {key}",
        )
    _require(
        abs(float(observed.get("fps") or 0.0) - float(decoded["fps"])) <= 0.01,
        "downloaded MP4 disagrees with reported FPS",
    )
    for key in ("max_spatial_std", "mean_temporal_abs_delta"):
        reported = float(observed.get(key) or 0.0)
        actual = float(decoded[key])
        _require(
            abs(reported - actual) <= max(0.1, actual * 0.02),
            f"downloaded MP4 disagrees with reported {key}",
        )
    _require(
        decoded["max_spatial_std"] >= 1.0
        and decoded["pixel_range"] >= 4
        and decoded["mean_temporal_abs_delta"] > 0.001,
        "downloaded MP4 fails independent spatial/temporal variation checks",
    )

    if layout is MULTI_GPU_LAYOUT:
        output = _require_mapping(
            primary.get("output"), "multi-GPU artifact has no output object"
        )
        _require(
            output.get("filename") == video_path.name,
            "multi-GPU output filename mismatch",
        )
        _require(
            int(output.get("size_bytes") or 0) == video_size,
            "multi-GPU output size mismatch",
        )
        _require(
            output.get("sha256") == video_sha256, "multi-GPU output SHA-256 mismatch"
        )
        _require(
            str(observed.get("codec") or "").lower() == "h264",
            "source video is not H.264",
        )
    else:
        _require(
            primary.get("output_filename") == video_path.name,
            "single-GPU output filename mismatch",
        )
        _require(
            int(primary.get("output_size_bytes") or 0) == video_size,
            "single-GPU output size mismatch",
        )
        _require(
            primary.get("output_sha256") == video_sha256,
            "single-GPU output SHA-256 mismatch",
        )

    return {
        "path": video_path,
        "size_bytes": video_size,
        "sha256": video_sha256,
        "observed": {**observed, **decoded},
        "capabilities": sorted(capabilities),
    }


def _validate_multigpu(
    primary: dict[str, Any],
    topology: dict[str, Any],
    rank_documents: list[dict[str, Any]],
    progress_documents: list[dict[str, Any]],
    nccl_summary: dict[str, Any],
    nccl_log_paths: list[Path],
) -> dict[str, Any]:
    distributed = _require_mapping(
        primary.get("distributed"),
        "multi-GPU artifact has no distributed evidence",
    )
    _require(distributed.get("world_size") == 4, "distributed world size must be four")
    _require(distributed.get("local_world_size") == 4, "local world size must be four")
    _require(distributed.get("node_count") == 1, "distributed run must use one node")
    _require(
        str(distributed.get("backend") or "").lower() == "nccl", "backend must be NCCL"
    )
    fsdp = _require_mapping(
        distributed.get("fsdp"), "distributed FSDP evidence is absent"
    )
    _require(
        fsdp.get("enabled") is True
        and fsdp.get("t5") is True
        and fsdp.get("dit") is True
        and fsdp.get("sharding_strategy") == "FULL_SHARD",
        "T5 and DiT FULL_SHARD FSDP evidence is incomplete",
    )
    ulysses = _require_mapping(
        distributed.get("ulysses"), "Ulysses size-four evidence is incomplete"
    )
    _require(
        ulysses.get("enabled") is True and ulysses.get("size") == 4,
        "Ulysses size-four evidence is incomplete",
    )

    topology_ranks = _require_list(
        topology.get("rank_evidence"), "topology must contain four ranks"
    )
    _require(
        len(topology_ranks) == 4,
        "topology must contain four ranks",
    )
    _require(len(rank_documents) == 4, "four standalone rank artifacts are required")
    topology_by_rank = {
        int(item.get("rank", -1)): item
        for item in topology_ranks
        if isinstance(item, dict)
    }
    rank_by_rank = {int(item.get("rank", -1)): item for item in rank_documents}
    _require(
        set(topology_by_rank) == {0, 1, 2, 3}, "topology ranks must be 0 through 3"
    )
    _require(set(rank_by_rank) == {0, 1, 2, 3}, "standalone ranks must be 0 through 3")
    _require(
        topology_by_rank == rank_by_rank,
        "standalone rank evidence disagrees with topology",
    )

    gpu_hashes: set[str] = set()
    host_hashes: set[str] = set()
    for rank in range(4):
        item = rank_by_rank[rank]
        _require(
            item.get("local_rank") == rank, f"rank {rank} local-rank mapping is invalid"
        )
        _require(item.get("world_size") == 4, f"rank {rank} world size is invalid")
        _require(
            item.get("process_group_initialized") is True,
            f"rank {rank} lacks process-group evidence",
        )
        _require(
            str(item.get("backend") or "").lower() == "nccl",
            f"rank {rank} did not use NCCL",
        )
        all_reduce = _require_mapping(
            item.get("nccl_all_reduce"),
            f"rank {rank} NCCL all-reduce evidence is invalid",
        )
        _require(
            all_reduce.get("finite") is True
            and float(all_reduce.get("observed_sum") or 0.0) == 10.0
            and float(all_reduce.get("expected_sum") or 0.0) == 10.0,
            f"rank {rank} NCCL all-reduce evidence is invalid",
        )
        device = _require_mapping(
            item.get("device"), f"rank {rank} device evidence is absent"
        )
        _require(
            device.get("compute_capability") == [10, 0],
            f"rank {rank} is not compute capability 10.0",
        )
        _require(
            str(device.get("name") or "") == "NVIDIA B200", f"rank {rank} is not a B200"
        )
        _require(
            item.get("current_cuda_device") == rank,
            f"rank {rank} used the wrong CUDA device",
        )
        _require(
            "sm_100" in (item.get("torch_cuda_arch_list") or []),
            f"rank {rank} lacks sm_100",
        )
        gpu_hashes.add(str(device.get("uuid_sha256") or ""))
        host_hashes.add(str(item.get("hostname_sha256") or ""))

        wrappers = _require_list(
            item.get("fsdp_wrappers"), f"rank {rank} needs two FSDP wrappers"
        )
        _require(
            len(wrappers) == 2,
            f"rank {rank} needs two FSDP wrappers",
        )
        sources = {
            str(wrapper.get("source_class") or "")
            for wrapper in wrappers
            if isinstance(wrapper, dict)
        }
        _require(
            sources == {"T5Encoder", "WanModel"},
            f"rank {rank} FSDP sources are invalid",
        )
        _require(
            all(
                str(wrapper.get("sharding_strategy") or "").endswith("FULL_SHARD")
                for wrapper in wrappers
            ),
            f"rank {rank} did not use FULL_SHARD",
        )
        _require(
            any(
                wrapper.get("source_class") == "WanModel"
                and wrapper.get("sequence_parallel_active_before_wrap") is True
                for wrapper in wrappers
            ),
            f"rank {rank} lacks active sequence parallelism",
        )
        _require(
            int(item.get("ulysses_distributed_attention_calls") or 0) > 0,
            f"rank {rank} lacks Ulysses attention calls",
        )
        _require(
            int(item.get("ulysses_all_to_all_calls") or 0) > 0,
            f"rank {rank} lacks Ulysses all-to-all calls",
        )
        _require(
            int(item.get("barrier_calls") or 0) > 0,
            f"rank {rank} lacks upstream barriers",
        )
        _require(
            item.get("observer_final_barrier") is True,
            f"rank {rank} lacks terminal synchronization",
        )
        loaded_nccl = _require_mapping(
            item.get("loaded_nccl"), f"rank {rank} lacks loaded NCCL evidence"
        )
        _require(
            loaded_nccl.get("version") == "2.27.7"
            and loaded_nccl.get("version_code") == 22707
            and str(loaded_nccl.get("library_basename") or "").startswith(
                "libnccl.so.2"
            ),
            f"rank {rank} did not load the accepted NCCL 2.27.7 runtime",
        )

    _require(
        len(gpu_hashes) == 4 and "" not in gpu_hashes,
        "four unique hashed GPU participants are required",
    )
    _require(
        len(host_hashes) == 1 and "" not in host_hashes,
        "all ranks must share one hashed hostname",
    )
    _require(len(progress_documents) == 4, "four post-destroy markers are required")
    progress_by_rank = {int(item.get("rank", -1)): item for item in progress_documents}
    _require(
        set(progress_by_rank) == {0, 1, 2, 3}, "post-destroy ranks must be 0 through 3"
    )
    for rank, item in progress_by_rank.items():
        _require(
            item.get("local_rank") == rank
            and item.get("world_size") == 4
            and item.get("stage") == "process_group_destroyed"
            and int(item.get("time_ns") or 0) > 0,
            f"rank {rank} lacks a valid post-destroy marker",
        )

    _require(
        nccl_summary.get("schema")
        == "npa.workbench.byof.wan2_2_multigpu_nccl_summary.v1",
        "NCCL summary schema is invalid",
    )
    _require(
        nccl_summary.get("loaded_version") == "2.27.7",
        "NCCL summary does not prove loaded version 2.27.7",
    )
    log_entries = _require_list(
        nccl_summary.get("rank_logs"), "NCCL summary must contain four rank logs"
    )
    _require(len(log_entries) == 4, "NCCL summary must contain four rank logs")
    logs_by_name = {path.name: path for path in nccl_log_paths}
    _require(len(logs_by_name) == 4, "four deterministic NCCL logs are required")
    seen_log_ranks: set[int] = set()
    for entry in log_entries:
        _require(isinstance(entry, dict), "NCCL log summary entry is invalid")
        rank = int(entry.get("rank", -1))
        filename = str(entry.get("filename") or "")
        _require(rank in {0, 1, 2, 3}, "NCCL log rank is invalid")
        _require(rank not in seen_log_ranks, "NCCL log ranks are duplicated")
        seen_log_ranks.add(rank)
        expected_name = f"wan2_2_multigpu_nccl_rank_{rank}.log"
        _require(filename == expected_name, f"rank {rank} NCCL log filename is invalid")
        log_path = logs_by_name.get(filename)
        if log_path is None or not log_path.is_file():
            raise WanRrdError(f"missing rank {rank} NCCL log")
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise WanRrdError(f"cannot read rank {rank} NCCL log: {exc}") from exc
        _require(
            int(entry.get("size_bytes") or 0) == log_path.stat().st_size
            and entry.get("sha256") == _sha256_file(log_path),
            f"rank {rank} NCCL log hash/size is invalid",
        )
        _require(
            re.search(r"\bNCCL version 2\.27\.7(?:\+cuda[0-9.]+)?\b", log_text)
            is not None
            and re.search(r"\bInit COMPLETE\b", log_text) is not None
            and entry.get("version_line_observed") is True
            and entry.get("init_complete_observed") is True,
            f"rank {rank} NCCL log lacks initialization/version evidence",
        )
    _require(seen_log_ranks == {0, 1, 2, 3}, "NCCL log ranks must be 0 through 3")
    return {
        "distributed": distributed,
        "topology": topology,
        "ranks": rank_documents,
        "post_destroy": progress_documents,
        "nccl_summary": nccl_summary,
    }


def _validate_single_gpu(
    primary: dict[str, Any], inventory: dict[str, Any]
) -> dict[str, Any]:
    topology = _require_mapping(
        primary.get("device_topology"), "single-GPU device topology is absent"
    )
    devices = _require_list(topology.get("devices"), "single-GPU devices are absent")
    _require(
        topology.get("cuda_device_count") == 1 and len(devices) == 1,
        "single-GPU evidence must contain exactly one CUDA device",
    )
    device = _require_mapping(devices[0], "single-GPU device evidence is invalid")
    _require(
        device.get("name") == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        and device.get("compute_capability") == [12, 0],
        "single-GPU evidence is not the accepted RTX PRO 6000 sm_120 device",
    )
    _require(
        str(topology.get("torch") or "").startswith("2.7.1+cu128")
        and topology.get("torch_cuda") == "12.8"
        and "sm_120" in (topology.get("torch_cuda_arch_list") or []),
        "single-GPU CUDA runtime is not the accepted sm_120 closure",
    )
    driver_versions = _require_list(
        topology.get("driver_versions"),
        "single-GPU driver version evidence is absent",
    )
    _require(
        bool(driver_versions)
        and all(
            isinstance(version, str) and version.strip() for version in driver_versions
        ),
        "single-GPU driver version evidence is invalid",
    )
    probe = _require_mapping(
        topology.get("sdpa_probe"), "single-GPU SDPA probe evidence is absent"
    )
    _require(
        topology.get("attention_backend")
        == "torch.nn.functional.scaled_dot_product_attention"
        and topology.get("flash_attention_installed") is False
        and topology.get("sdpa_source_binding") is True
        and probe.get("dtype") == "torch.bfloat16"
        and probe.get("shape") == [1, 4, 32, 64]
        and probe.get("finite") is True,
        "single-GPU native BF16 SDPA gate is invalid",
    )
    runtime_stack = _require_mapping(
        inventory.get("runtime_stack"),
        "single-GPU runtime stack inventory is absent",
    )
    for key in (
        "devices",
        "torch",
        "torch_cuda",
        "torch_cuda_arch_list",
        "driver_versions",
        "attention_backend",
        "flash_attention_installed",
        "sdpa_source_binding",
        "sdpa_probe",
    ):
        _require(
            runtime_stack.get(key) == topology.get(key),
            f"single-GPU runtime inventory disagrees on {key}",
        )
    return {"distributed": None, "topology": topology, "ranks": []}


def validate_wan_run(run_dir: Path, layout: WanRunLayout) -> dict[str, Any]:
    """Validate immutable Wan output artifacts and return sanitized evidence."""

    summary = _load_json(run_dir / "npa_byof_summary.json")
    container_image = _validate_container_image(summary)
    primary_path = _primary_path(run_dir, layout, summary)
    primary = _load_json(primary_path)
    inventory = _load_json(run_dir / layout.inventory_filename)
    source_metadata = _load_json(run_dir / "npa_source_metadata.json")
    video = _validate_output(primary, run_dir / layout.video_filename, layout)

    _require(summary.get("status") == "success", "BYOF summary is not successful")
    _require(
        int(summary.get("smoke_exit_code", -1)) == 0, "Wan smoke exit code is not zero"
    )
    run_id = str(summary.get("run_id") or "").strip()
    _require(bool(run_id), "BYOF summary has no run id")
    _require(
        primary.get("schema") == layout.primary_schema, "primary Wan schema is invalid"
    )
    _require(
        primary.get("solution") == "wan2.2", "primary artifact is not a Wan 2.2 result"
    )
    _require(
        source_metadata.get("repo") == SOURCE_REPO,
        "source metadata repo is not official Wan",
    )
    _require(
        source_metadata.get("ref") == SOURCE_REF,
        "source metadata revision is not pinned",
    )

    if layout is MULTI_GPU_LAYOUT:
        upstream = _require_mapping(
            primary.get("upstream"), "official upstream repo is absent"
        )
        _require(
            upstream.get("repo") == SOURCE_REPO,
            "official upstream repo is absent",
        )
        _require(
            upstream.get("ref") == SOURCE_REF, "official source revision is invalid"
        )
        _require(
            upstream.get("entrypoint")
            == "python -m torch.distributed.run --standalone --nnodes=1 "
            "--nproc_per_node=4 wan22_distributed_wrapper.py"
            and upstream.get("wrapper_execution")
            == "runpy.run_path('/opt/byof/generate.py', run_name='__main__')",
            "official distributed wrapper execution identity is invalid",
        )
        model = _require_mapping(primary.get("model"), "official model id is invalid")
        _require(model.get("id") == MODEL_ID, "official model id is invalid")
        _require(model.get("ref") == MODEL_REF, "official model revision is invalid")
        tokenizer = _require_mapping(
            primary.get("tokenizer"), "official tokenizer identity is invalid"
        )
        _require(
            tokenizer.get("id") == TOKENIZER_ID
            and tokenizer.get("ref") == TOKENIZER_REF,
            "official tokenizer identity is invalid",
        )
        _require_mapping(
            primary.get("generation"), "multi-GPU generation object is absent"
        )
        _require_mapping(primary.get("runtime"), "multi-GPU runtime object is absent")
        _require(
            model.get("weights_baked") is False,
            "model weights must remain runtime-only",
        )
        topology_filename = layout.topology_filename
        if not isinstance(topology_filename, str) or not topology_filename:
            raise WanRrdError("multi-GPU layout has no topology filename")
        topology = _load_json(run_dir / topology_filename)
        ranks = [_load_json(run_dir / name) for name in layout.rank_filenames]
        progress = [
            _load_json(run_dir / f"wan2_2_multigpu_progress_rank_{rank}.json")
            for rank in range(4)
        ]
        nccl_summary = _load_json(run_dir / "wan2_2_multigpu_nccl_summary.json")
        nccl_logs = [
            run_dir / f"wan2_2_multigpu_nccl_rank_{rank}.log" for rank in range(4)
        ]
        execution = _validate_multigpu(
            primary, topology, ranks, progress, nccl_summary, nccl_logs
        )
        _require(
            inventory.get("source_ref") == SOURCE_REF,
            "runtime inventory source revision is invalid",
        )
        _require(
            inventory.get("non_root") is True, "multi-GPU runtime was not non-root"
        )
        _require(
            inventory.get("weights_baked") is False,
            "runtime inventory reports baked weights",
        )
    else:
        _require(
            primary.get("upstream_repo") == SOURCE_REPO,
            "official upstream repo is absent",
        )
        _require(
            primary.get("upstream_ref") == SOURCE_REF,
            "official source revision is invalid",
        )
        _require(primary.get("model_id") == MODEL_ID, "official model id is invalid")
        _require(
            primary.get("model_ref") == MODEL_REF, "official model revision is invalid"
        )
        _require(
            primary.get("tokenizer_id") == TOKENIZER_ID
            and primary.get("tokenizer_ref") == TOKENIZER_REF,
            "official tokenizer identity is invalid",
        )
        _require(
            isinstance(primary.get("requested"), dict),
            "single-GPU requested generation object is absent",
        )
        _require(
            primary.get("weights_baked") is False,
            "model weights must remain runtime-only",
        )
        baked_runtime = _require_mapping(
            inventory.get("baked_runtime"), "single-GPU runtime was not non-root"
        )
        _require(
            baked_runtime.get("non_root") is True,
            "single-GPU runtime was not non-root",
        )
        _require(
            baked_runtime.get("venv_readable") is True,
            "single-GPU accepted runtime venv was not readable",
        )
        _require(
            not baked_runtime.get("large_checkpoint_shaped_files"),
            "runtime inventory found baked weights",
        )
        execution = _validate_single_gpu(primary, inventory)

    return {
        "run_id": run_id,
        "layout": layout,
        "summary": {
            key: summary.get(key)
            for key in (
                "status",
                "tool",
                "workload",
                "run_id",
                "solution_name",
                "capability_name",
                "smoke_artifact_name",
                "smoke_exit_code",
                "created_unix",
            )
        },
        "primary": primary,
        "primary_filename": primary_path.name,
        "inventory": inventory,
        "source_metadata": source_metadata,
        "container_image": container_image,
        "video": video,
        **execution,
    }


def _machine_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    primary = evidence["primary"]
    layout: WanRunLayout = evidence["layout"]
    if layout is MULTI_GPU_LAYOUT:
        source = primary["upstream"]
        model = primary["model"]
        tokenizer = primary["tokenizer"]
        generation = primary["generation"]
        runtime = primary["runtime"]
    else:
        source = {"repo": primary["upstream_repo"], "ref": primary["upstream_ref"]}
        model = {"id": primary["model_id"], "ref": primary["model_ref"]}
        tokenizer = {"id": primary["tokenizer_id"], "ref": primary["tokenizer_ref"]}
        generation = {
            "task": primary.get("task"),
            "prompt": primary.get("prompt"),
            "seed": primary.get("seed"),
            **(primary.get("requested") or {}),
        }
        runtime = primary.get("device_topology") or {}
    return {
        "schema": RRD_SUMMARY_SCHEMA,
        "run_id": evidence["run_id"],
        "variant": layout.variant,
        "official_source": source,
        "official_model": model,
        "tokenizer": tokenizer,
        "container_image": evidence["container_image"],
        "generation": generation,
        "observed_output": {
            **evidence["video"]["observed"],
            "filename": layout.video_filename,
            "size_bytes": evidence["video"]["size_bytes"],
            "sha256": evidence["video"]["sha256"],
            "decode_status": "passed",
        },
        "execution": (
            {
                **evidence["distributed"],
                "loaded_nccl": evidence["nccl_summary"],
                "process_group_destroyed_on_all_ranks": True,
            }
            if layout is MULTI_GPU_LAYOUT
            else runtime
        ),
        "capabilities_exercised": evidence["video"]["capabilities"],
        "weights_baked": False,
    }


def _overview_markdown(evidence: dict[str, Any]) -> str:
    summary = _machine_summary(evidence)
    output = summary["observed_output"]
    execution = summary["execution"]
    if evidence["layout"] is MULTI_GPU_LAYOUT:
        runtime_path = (
            "one node, 4× NVIDIA B200, world size 4, NCCL, T5/DiT FULL_SHARD "
            "FSDP, Ulysses size 4"
        )
    else:
        devices = execution.get("devices") or []
        device_name = (
            devices[0].get("name") if devices else "one RTX PRO 6000 Blackwell"
        )
        runtime_path = f"single GPU ({device_name}), native PyTorch SDPA"
    return (
        "# Official Wan 2.2 TI2V-5B evidence\n\n"
        f"- **Run:** `{summary['run_id']}`\n"
        f"- **Source:** `{SOURCE_REPO}@{SOURCE_REF}`\n"
        f"- **Model:** `{MODEL_ID}@{MODEL_REF}` (fetched at run time)\n"
        f"- **Container image digest:** `{summary['container_image']['oci_digest']}`\n"
        f"- **Runtime path:** {runtime_path}\n"
        f"- **Validated output:** `{output['filename']}` — "
        f"{output['frame_count']} H.264 frames, {output['width']}×{output['height']} "
        f"at {float(output['fps']):g} fps, {output['size_bytes']:,} bytes\n"
        f"- **MP4 SHA-256:** `{output['sha256']}`\n\n"
        "The video bytes are embedded in this recording. The surrounding evidence is "
        "static because the source run produced static JSON facts, not telemetry."
    )


def _validation_markdown(evidence: dict[str, Any]) -> str:
    observed = evidence["video"]["observed"]
    return (
        "# Output validation\n\n"
        "- Decode status: **passed**\n"
        f"- Codec: `{observed.get('codec', 'h264')}`\n"
        f"- Frames: `{observed['frame_count']}`\n"
        f"- Resolution / FPS: `{observed['width']}×{observed['height']}` / `{observed['fps']}`\n"
        f"- Maximum spatial standard deviation: `{observed['max_spatial_std']}`\n"
        f"- Pixel range: `{observed['pixel_range']}`\n"
        f"- Mean temporal absolute delta: `{observed['mean_temporal_abs_delta']}`\n"
        f"- SHA-256: `{evidence['video']['sha256']}`"
    )


def _execution_markdown(evidence: dict[str, Any]) -> str:
    if evidence["layout"] is SINGLE_GPU_LAYOUT:
        topology = evidence["primary"].get("device_topology") or {}
        return (
            "# Runtime evidence\n\n"
            f"- CUDA devices: `{topology.get('cuda_device_count')}`\n"
            f"- Devices: `{json.dumps(topology.get('devices') or [], sort_keys=True)}`\n"
            f"- PyTorch / CUDA: `{topology.get('torch')}` / `{topology.get('torch_cuda')}`\n"
            f"- Driver: `{', '.join(topology.get('driver_versions') or [])}`\n"
            f"- Attention: `{topology.get('attention_backend')}`\n"
            f"- SDPA source binding: `{topology.get('sdpa_source_binding')}`"
        )
    distributed = evidence["distributed"]
    ranks = evidence["ranks"]
    attention_calls = sorted(
        {int(rank["ulysses_distributed_attention_calls"]) for rank in ranks}
    )
    all_to_all_calls = sorted({int(rank["ulysses_all_to_all_calls"]) for rank in ranks})
    barrier_calls = sorted({int(rank["barrier_calls"]) for rank in ranks})
    return (
        "# Distributed execution evidence\n\n"
        f"- World / local world / nodes: `{distributed['world_size']}` / "
        f"`{distributed['local_world_size']}` / `{distributed['node_count']}`\n"
        f"- Backend: `{distributed['backend']}`; all-reduce observed/expected: `10 / 10` on every rank\n"
        "- Participants: **four unique hashed NVIDIA B200 GPUs**\n"
        "- FSDP: `T5Encoder` and `WanModel`, `FULL_SHARD`, on every rank\n"
        f"- Ulysses: size `{distributed['ulysses']['size']}`, distributed-attention calls "
        f"`{attention_calls}`, all-to-all calls `{all_to_all_calls}` per rank\n"
        f"- Upstream barriers: `{barrier_calls}` per rank; observer terminal barrier: `true` on every rank\n"
        "- Loaded NCCL: `2.27.7`, independently confirmed in every rank log\n"
        "- Process-group teardown: `process_group_destroyed` on every rank"
    )


def _runtime_markdown(evidence: dict[str, Any]) -> str:
    inventory = evidence["inventory"]
    if evidence["layout"] is MULTI_GPU_LAYOUT:
        runtime = inventory["runtime"]
        packages = inventory.get("package_versions") or {}
        return (
            "# Runtime inventory\n\n"
            f"- Non-root: `{inventory.get('non_root')}`\n"
            f"- Weights baked: `{inventory.get('weights_baked')}`\n"
            f"- PyTorch / CUDA / NCCL: `{runtime.get('torch')}` / "
            f"`{runtime.get('torch_cuda')}` / `{runtime.get('nccl_loaded_version')}`\n"
            f"- Driver: `{', '.join(runtime.get('driver_versions') or [])}`\n"
            f"- Package versions: `{json.dumps(packages, sort_keys=True)}`"
        )
    baked = inventory["baked_runtime"]
    runtime = inventory["runtime_stack"]
    return (
        "# Runtime inventory\n\n"
        f"- Non-root: `{baked.get('non_root')}`\n"
        f"- Accessible venv: `{baked.get('venv_readable')}`\n"
        f"- Large checkpoint-shaped files baked into source tree: "
        f"`{len(baked.get('large_checkpoint_shaped_files') or [])}`\n"
        f"- PyTorch / CUDA: `{runtime.get('torch')}` / `{runtime.get('torch_cuda')}`\n"
        f"- Driver: `{', '.join(runtime.get('driver_versions') or [])}`"
    )


def _build_blueprint(rrb: Any, fps: float) -> Any:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(
                origin=f"{ENTITY_ROOT}/video",
                contents=f"{ENTITY_ROOT}/video/**",
                name="Official Wan 2.2 output",
            ),
            rrb.Tabs(
                rrb.TextDocumentView(
                    origin=f"{ENTITY_ROOT}/summary/overview", name="Overview"
                ),
                rrb.TextDocumentView(
                    origin=f"{ENTITY_ROOT}/evidence/validation",
                    name="Output validation",
                ),
                rrb.TextDocumentView(
                    origin=f"{ENTITY_ROOT}/evidence/execution", name="Execution"
                ),
                rrb.TextDocumentView(
                    origin=f"{ENTITY_ROOT}/evidence/runtime", name="Runtime"
                ),
                rrb.TextDocumentView(
                    origin=f"{ENTITY_ROOT}/summary/machine_readable",
                    name="Machine summary",
                ),
                active_tab=0,
                name="Run evidence",
            ),
            column_shares=[2.4, 1.4],
        ),
        rrb.BlueprintPanel(state=rrb.PanelState.Hidden),
        rrb.SelectionPanel(state=rrb.PanelState.Hidden),
        rrb.TimePanel(state=rrb.PanelState.Expanded, timeline=VIDEO_TIMELINE, fps=fps),
        auto_layout=False,
        collapse_panels=True,
    )


def _log_recording(output_path: Path, evidence: dict[str, Any]) -> None:
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:  # pragma: no cover - exercised by runtime failure path
        raise WanRrdError("Wan RRD generation requires the npa[viz] extra") from exc

    observed = evidence["video"]["observed"]
    fps = float(observed["fps"])
    frame_count = int(observed["frame_count"])
    blueprint = _build_blueprint(rrb, fps)
    recording = rr.RecordingStream(APPLICATION_ID, recording_id=evidence["run_id"])
    recording.save(output_path, default_blueprint=blueprint)
    try:
        recording.log(
            VIDEO_ASSET_ENTITY,
            rr.AssetVideo(path=evidence["video"]["path"]),
            static=True,
        )
        recording.log(
            f"{ENTITY_ROOT}/summary/overview",
            rr.TextDocument(_overview_markdown(evidence), media_type="text/markdown"),
            static=True,
        )
        recording.log(
            f"{ENTITY_ROOT}/summary/machine_readable",
            rr.TextDocument(
                json.dumps(_machine_summary(evidence), indent=2, sort_keys=True),
                media_type="text/plain",
            ),
            static=True,
        )
        recording.log(
            f"{ENTITY_ROOT}/evidence/validation",
            rr.TextDocument(_validation_markdown(evidence), media_type="text/markdown"),
            static=True,
        )
        recording.log(
            f"{ENTITY_ROOT}/evidence/execution",
            rr.TextDocument(_execution_markdown(evidence), media_type="text/markdown"),
            static=True,
        )
        recording.log(
            f"{ENTITY_ROOT}/evidence/runtime",
            rr.TextDocument(_runtime_markdown(evidence), media_type="text/markdown"),
            static=True,
        )
        for rank in evidence["ranks"]:
            recording.log(
                f"{ENTITY_ROOT}/evidence/ranks/rank_{rank['rank']}",
                rr.TextDocument(
                    json.dumps(rank, indent=2, sort_keys=True), media_type="text/plain"
                ),
                static=True,
            )
        if evidence["layout"] is MULTI_GPU_LAYOUT:
            recording.log(
                f"{ENTITY_ROOT}/evidence/teardown",
                rr.TextDocument(
                    json.dumps(
                        {
                            "post_destroy": evidence["post_destroy"],
                            "nccl_summary": evidence["nccl_summary"],
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    media_type="text/plain",
                ),
                static=True,
            )

        scalar_values = {
            "frame_count": frame_count,
            "width": int(observed["width"]),
            "height": int(observed["height"]),
            "fps": fps,
            "max_spatial_std": float(observed["max_spatial_std"]),
            "pixel_range": int(observed["pixel_range"]),
            "mean_temporal_abs_delta": float(observed["mean_temporal_abs_delta"]),
            "output_size_bytes": evidence["video"]["size_bytes"],
            "world_size": int(
                (evidence.get("distributed") or {}).get("world_size") or 1
            ),
        }
        for name, value in scalar_values.items():
            recording.log(
                f"{ENTITY_ROOT}/metrics/{name}", rr.Scalars(float(value)), static=True
            )

        for frame_index in range(frame_count):
            timestamp = frame_index / fps
            rr.set_time(VIDEO_TIMELINE, duration=timestamp, recording=recording)
            recording.log(
                VIDEO_FRAME_ENTITY,
                rr.VideoFrameReference(
                    seconds=timestamp,
                    video_reference=VIDEO_ASSET_ENTITY,
                ),
            )
        recording.flush()
    finally:
        recording.disconnect()


def _run_rerun_cli(path: Path, command: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "rerun", "rrd", command, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise WanRrdError(
            f"rerun rrd {command} failed ({proc.returncode}): "
            f"{(proc.stdout + proc.stderr).strip()}"
        )
    return (proc.stdout + proc.stderr).strip()


def _stats_fields(text: str) -> dict[str, int]:
    fields: dict[str, int] = {}
    for name in ("num_chunks", "num_entity_paths", "num_rows", "num_static"):
        match = re.search(rf"^{name}\s*=\s*([0-9]+)", text, flags=re.MULTILINE)
        if match is None:
            raise WanRrdError(f"rerun stats missing required field: {name}")
        fields[name] = int(match.group(1))
    _require(fields["num_chunks"] > 0, "rerun stats reports no chunks")
    _require(fields["num_entity_paths"] > 0, "rerun stats reports no entity paths")
    _require(fields["num_rows"] > 0, "rerun stats reports no rows")
    return fields


def verify_wan_rrd(
    rrd_path: Path,
    *,
    source_video_path: Path,
    expected_frame_count: int,
    expected_fps: float,
    expected_rank_count: int,
) -> dict[str, Any]:
    """Parse an RRD, verify entity rows, and hash its embedded MP4 bytes."""

    try:
        from rerun.recording import load_recording
    except ImportError as exc:  # pragma: no cover - exercised by runtime failure path
        raise WanRrdError("Wan RRD verification requires the npa[viz] extra") from exc

    _require(
        rrd_path.is_file() and rrd_path.stat().st_size > 0, "Rerun recording is empty"
    )
    chunks = list(load_recording(rrd_path).chunks())
    counts: dict[str, int] = {}
    for chunk in chunks:
        entity = str(chunk.entity_path)
        counts[entity] = counts.get(entity, 0) + int(chunk.num_rows)

    required = {
        f"/{VIDEO_ASSET_ENTITY}": 1,
        f"/{VIDEO_FRAME_ENTITY}": expected_frame_count,
        f"/{ENTITY_ROOT}/summary/overview": 1,
        f"/{ENTITY_ROOT}/summary/machine_readable": 1,
        f"/{ENTITY_ROOT}/evidence/validation": 1,
        f"/{ENTITY_ROOT}/evidence/execution": 1,
        f"/{ENTITY_ROOT}/evidence/runtime": 1,
    }
    for rank in range(expected_rank_count):
        required[f"/{ENTITY_ROOT}/evidence/ranks/rank_{rank}"] = 1
    if expected_rank_count:
        required[f"/{ENTITY_ROOT}/evidence/teardown"] = 1
    for entity, minimum in required.items():
        _require(
            counts.get(entity, 0) >= minimum, f"RRD entity {entity} has too few rows"
        )

    embedded_blobs: list[bytes] = []
    timestamps_ns: list[int] = []
    video_references: list[str] = []
    for chunk in chunks:
        batch = chunk.to_record_batch()
        if str(chunk.entity_path) == f"/{VIDEO_ASSET_ENTITY}":
            values = batch.column("AssetVideo:blob").to_pylist()
            embedded_blobs.extend(bytes(row[0]) for row in values if row)
        elif str(chunk.entity_path) == f"/{VIDEO_FRAME_ENTITY}":
            timestamps_ns.extend(
                int(row[0])
                for row in batch.column("VideoFrameReference:timestamp").to_pylist()
                if row
            )
            video_references.extend(
                str(row[0])
                for row in batch.column(
                    "VideoFrameReference:video_reference"
                ).to_pylist()
                if row
            )

    _require(len(embedded_blobs) == 1, "RRD must contain exactly one embedded MP4 blob")
    source_sha256 = _sha256_file(source_video_path)
    embedded_sha256 = _sha256_bytes(embedded_blobs[0])
    _require(
        embedded_sha256 == source_sha256, "embedded MP4 does not match the source MP4"
    )
    _require(
        len(timestamps_ns) == expected_frame_count,
        "video frame-reference count is invalid",
    )
    expected_timestamps = [
        round(frame / expected_fps * 1_000_000_000)
        for frame in range(expected_frame_count)
    ]
    _require(
        timestamps_ns == expected_timestamps,
        "video frame-reference timestamps are invalid",
    )
    _require(
        video_references == [VIDEO_ASSET_ENTITY] * expected_frame_count,
        "video frame references do not target the embedded MP4 entity",
    )

    verify_output = _run_rerun_cli(rrd_path, "verify")
    stats_output = _run_rerun_cli(rrd_path, "stats")
    return {
        "rrd_parse": "passed",
        "rerun_cli_verify": "passed",
        "embedded_mp4_sha256_match": True,
        "embedded_mp4_sha256": embedded_sha256,
        "video_frame_reference_count": len(timestamps_ns),
        "video_timestamps_valid": True,
        "entity_row_counts": dict(sorted(counts.items())),
        "required_entity_paths": sorted(required),
        "rerun_stats": _stats_fields(stats_output),
        "rerun_verify_output": verify_output,
    }


def build_wan_rrd(
    run_dir: Path, output_path: Path, *, layout: WanRunLayout
) -> dict[str, Any]:
    """Build and fully verify a local RRD from a materialized Wan run directory."""

    evidence = validate_wan_run(run_dir, layout)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _log_recording(output_path, evidence)
    verification = verify_wan_rrd(
        output_path,
        source_video_path=evidence["video"]["path"],
        expected_frame_count=int(evidence["video"]["observed"]["frame_count"]),
        expected_fps=float(evidence["video"]["observed"]["fps"]),
        expected_rank_count=len(evidence["ranks"]),
    )
    return {
        "run_id": evidence["run_id"],
        "variant": layout.variant,
        "layout": layout,
        "rrd_path": output_path,
        "rrd_size_bytes": output_path.stat().st_size,
        "rrd_sha256": _sha256_file(output_path),
        "video_size_bytes": evidence["video"]["size_bytes"],
        "video_sha256": evidence["video"]["sha256"],
        "video_observed": evidence["video"]["observed"],
        "container_image": evidence["container_image"],
        "primary_filename": evidence["primary_filename"],
        "verification": verification,
    }


def _parse_s3_prefix(uri: str) -> tuple[str, str, str]:
    parsed = urlparse(uri.rstrip("/"))
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise WanRrdError("run prefix must be an s3:// bucket/prefix URI")
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    normalized = f"s3://{parsed.netloc}/{prefix}"
    return parsed.netloc, prefix, normalized


def _source_role(name: str, layout: WanRunLayout) -> str:
    if name == "npa_byof_summary.json":
        return "byof_summary"
    if name == "npa_source_metadata.json":
        return "source_metadata"
    if name in layout.primary_filenames:
        return "capability_evidence"
    if name == layout.video_filename:
        return "generated_video"
    if name == layout.inventory_filename:
        return "runtime_inventory"
    if name == layout.topology_filename:
        return "distributed_topology"
    if name in layout.rank_filenames:
        return "distributed_rank_evidence"
    if name.startswith("wan2_2_multigpu_progress_rank_"):
        return "distributed_post_destroy_evidence"
    if name == "wan2_2_multigpu_nccl_summary.json":
        return "distributed_loaded_nccl_summary"
    if name.startswith("wan2_2_multigpu_nccl_rank_"):
        return "distributed_nccl_log"
    return "source_evidence"


def _source_filenames(layout: WanRunLayout, primary_filename: str) -> list[str]:
    names = [
        "npa_byof_summary.json",
        "npa_source_metadata.json",
        primary_filename,
        layout.inventory_filename,
    ]
    if layout.topology_filename:
        names.append(layout.topology_filename)
    names.extend(layout.rank_filenames)
    names.extend(layout.additional_evidence_filenames)
    names.append(layout.video_filename)
    return names


def _manifest_payload(
    build: dict[str, Any],
    *,
    source_prefix_uri: str,
    source_objects: list[dict[str, Any]],
    rrd_uri: str,
    remote_verification: dict[str, Any],
) -> dict[str, Any]:
    try:
        import rerun as rr
    except ImportError as exc:  # pragma: no cover
        raise WanRrdError("Wan RRD manifest requires the npa[viz] extra") from exc

    observed = build["video_observed"]
    return {
        "schema": RRD_MANIFEST_SCHEMA,
        "status": "verified",
        "capability": RRD_CAPABILITY,
        "run_id": build["run_id"],
        "variant": build["variant"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_prefix_uri": source_prefix_uri,
        "source_objects": source_objects,
        "container_image": build["container_image"],
        "rrd": {
            "uri": rrd_uri,
            "filename": build["layout"].rrd_filename,
            "size_bytes": build["rrd_size_bytes"],
            "sha256": build["rrd_sha256"],
            "rerun_sdk_version": rr.__version__,
            "application_id": APPLICATION_ID,
            "recording_id": build["run_id"],
            "entity_paths": remote_verification["required_entity_paths"],
            "entity_row_counts": remote_verification["entity_row_counts"],
        },
        "video": {
            "source_filename": build["layout"].video_filename,
            "asset_entity_path": f"/{VIDEO_ASSET_ENTITY}",
            "frame_entity_path": f"/{VIDEO_FRAME_ENTITY}",
            "timeline": VIDEO_TIMELINE,
            "frame_reference_count": remote_verification["video_frame_reference_count"],
            "width": int(observed["width"]),
            "height": int(observed["height"]),
            "fps": float(observed["fps"]),
            "source_size_bytes": build["video_size_bytes"],
            "source_sha256": build["video_sha256"],
            "embedded_sha256": remote_verification["embedded_mp4_sha256"],
        },
        "verification": {
            "source_contract": "passed",
            "source_reported_decode_status": "passed",
            "local_rrd_parse": build["verification"]["rrd_parse"],
            "local_rerun_cli_verify": build["verification"]["rerun_cli_verify"],
            "remote_rrd_parse": remote_verification["rrd_parse"],
            "remote_rerun_cli_verify": remote_verification["rerun_cli_verify"],
            "remote_size_and_sha256_match": True,
            "embedded_mp4_sha256_match": remote_verification[
                "embedded_mp4_sha256_match"
            ],
            "video_timestamps_valid": remote_verification["video_timestamps_valid"],
            "rerun_stats": remote_verification["rerun_stats"],
        },
    }


def publish_wan_rrd_from_s3(
    run_prefix_uri: str,
    *,
    variant: str,
    project: str | None = None,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Materialize, verify, upload, download, and re-verify a Wan RRD + manifest."""

    layout = layout_for_variant(variant)
    bucket, prefix, normalized_prefix = _parse_s3_prefix(run_prefix_uri)
    s3 = s3_client or s3_client_for_project(project, allow_host_creds=True)

    with tempfile.TemporaryDirectory(prefix="npa-wan-rrd-") as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        summary_name = "npa_byof_summary.json"
        summary_response = s3.get_object(Bucket=bucket, Key=prefix + summary_name)
        summary_bytes = summary_response["Body"].read()
        (run_dir / summary_name).write_bytes(summary_bytes)
        summary = _load_json(run_dir / summary_name)
        primary_name = str(summary.get("smoke_artifact_name") or "").strip()
        _require(
            primary_name in layout.primary_filenames,
            "S3 summary declares an unexpected primary artifact",
        )

        source_objects: list[dict[str, Any]] = []
        for name in _source_filenames(layout, primary_name):
            key = prefix + name
            if name == summary_name:
                payload = summary_bytes
                response = summary_response
            else:
                response = s3.get_object(Bucket=bucket, Key=key)
                payload = response["Body"].read()
                (run_dir / name).write_bytes(payload)
            _require(
                int(response.get("ContentLength") or -1) == len(payload),
                f"S3 size mismatch for {name}",
            )
            etag = str(response.get("ETag") or "").strip('"')
            _require(bool(etag), f"S3 ETag is absent for {name}")
            source_object = {
                "role": _source_role(name, layout),
                "uri": normalized_prefix + name,
                "etag": etag,
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
            if response.get("VersionId"):
                source_object["version_id"] = str(response["VersionId"])
            source_objects.append(source_object)

        local_rrd = Path(tmp) / layout.rrd_filename
        build = build_wan_rrd(run_dir, local_rrd, layout=layout)
        rrd_key = prefix + layout.rrd_filename
        rrd_uri = normalized_prefix + layout.rrd_filename
        rrd_bytes = local_rrd.read_bytes()
        s3.put_object(
            Bucket=bucket,
            Key=rrd_key,
            Body=rrd_bytes,
            ContentType="application/octet-stream",
            IfNoneMatch="*",
        )
        remote_head = s3.head_object(Bucket=bucket, Key=rrd_key)
        remote_bytes = s3.get_object(Bucket=bucket, Key=rrd_key)["Body"].read()
        _require(
            int(remote_head.get("ContentLength") or -1) == len(rrd_bytes),
            "uploaded RRD size mismatch",
        )
        _require(
            remote_bytes == rrd_bytes, "uploaded RRD bytes do not match local bytes"
        )
        _require(
            _sha256_bytes(remote_bytes) == build["rrd_sha256"],
            "uploaded RRD SHA-256 mismatch",
        )
        remote_rrd = Path(tmp) / f"remote-{layout.rrd_filename}"
        remote_rrd.write_bytes(remote_bytes)
        remote_verification = verify_wan_rrd(
            remote_rrd,
            source_video_path=run_dir / layout.video_filename,
            expected_frame_count=int(build["video_observed"]["frame_count"]),
            expected_fps=float(build["video_observed"]["fps"]),
            expected_rank_count=len(layout.rank_filenames),
        )

        manifest = _manifest_payload(
            build,
            source_prefix_uri=normalized_prefix,
            source_objects=source_objects,
            rrd_uri=rrd_uri,
            remote_verification=remote_verification,
        )
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode()
        manifest_key = prefix + layout.manifest_filename
        manifest_uri = normalized_prefix + layout.manifest_filename
        s3.put_object(
            Bucket=bucket,
            Key=manifest_key,
            Body=manifest_bytes,
            ContentType="application/json",
            IfNoneMatch="*",
        )
        manifest_head = s3.head_object(Bucket=bucket, Key=manifest_key)
        remote_manifest = s3.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
        _require(
            int(manifest_head.get("ContentLength") or -1) == len(manifest_bytes),
            "uploaded RRD manifest size mismatch",
        )
        _require(
            remote_manifest == manifest_bytes,
            "uploaded RRD manifest bytes do not match",
        )
        parsed_manifest = json.loads(remote_manifest)
        _require(
            parsed_manifest.get("status") == "verified",
            "uploaded RRD manifest is not verified",
        )
        _require(
            parsed_manifest.get("capability") == RRD_CAPABILITY,
            "RRD manifest capability is invalid",
        )

        return {
            "status": "verified",
            "capability": RRD_CAPABILITY,
            "run_id": build["run_id"],
            "variant": layout.variant,
            "rrd_uri": rrd_uri,
            "rrd_size_bytes": len(rrd_bytes),
            "rrd_sha256": build["rrd_sha256"],
            "manifest_uri": manifest_uri,
            "manifest_size_bytes": len(manifest_bytes),
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "entity_paths": remote_verification["required_entity_paths"],
            "video_sha256": build["video_sha256"],
            "container_image": build["container_image"],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix-uri", required=True)
    parser.add_argument("--variant", choices=("single-gpu", "multigpu"), required=True)
    parser.add_argument("--project", default="")
    args = parser.parse_args(argv)
    result = publish_wan_rrd_from_s3(
        args.run_prefix_uri,
        variant=args.variant,
        project=args.project or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
