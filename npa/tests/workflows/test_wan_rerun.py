from __future__ import annotations

import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from npa.deploy.images import wan_accepted_image_manifest
from npa.workflows.wan_rerun import (
    MULTI_GPU_LAYOUT,
    SINGLE_GPU_LAYOUT,
    RRD_CAPABILITY,
    RRD_MANIFEST_SCHEMA,
    WanRrdError,
    build_wan_rrd,
    publish_wan_rrd_from_s3,
)
from npa.solutions.wan2_2 import rerun as wan_rerun


def _accepted_image_ref() -> str:
    accepted = wan_accepted_image_manifest()
    return f"registry.example/project/npa-wan2-2@{accepted['oci_digest']}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@lru_cache(maxsize=1)
def _valid_video_bytes() -> bytes:
    output = io.BytesIO()
    x = np.arange(1280, dtype=np.uint16)[None, :]
    y = np.arange(704, dtype=np.uint16)[:, None]
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width = 1280
        stream.height = 704
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast"}
        for index in range(17):
            pixels = np.empty((704, 1280, 3), dtype=np.uint8)
            pixels[:, :, 0] = (x + index * 13) % 256
            pixels[:, :, 1] = (y * 2 + index * 7) % 256
            pixels[:, :, 2] = (x // 4 + y // 4 + index * 17) % 256
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def _rank(rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "local_rank": rank,
        "world_size": 4,
        "hostname_sha256": "1" * 64,
        "process_group_initialized": True,
        "backend": "nccl",
        "nccl_all_reduce": {
            "finite": True,
            "observed_sum": 10.0,
            "expected_sum": 10.0,
        },
        "fsdp_wrappers": [
            {
                "source_class": "T5Encoder",
                "wrapper_class": "FullyShardedDataParallel",
                "sharding_strategy": "ShardingStrategy.FULL_SHARD",
                "sequence_parallel_active_before_wrap": False,
            },
            {
                "source_class": "WanModel",
                "wrapper_class": "FullyShardedDataParallel",
                "sharding_strategy": "ShardingStrategy.FULL_SHARD",
                "sequence_parallel_active_before_wrap": True,
            },
        ],
        "ulysses_distributed_attention_calls": 480,
        "ulysses_all_to_all_calls": 1920,
        "barrier_calls": 3,
        "observer_final_barrier": True,
        "device": {
            "cuda_index": rank,
            "name": "NVIDIA B200",
            "compute_capability": [10, 0],
            "total_memory_bytes": 191503007744,
            "uuid_sha256": hashlib.sha256(f"gpu-{rank}".encode()).hexdigest(),
        },
        "current_cuda_device": rank,
        "torch": "2.13.0+cu130",
        "torch_cuda": "13.0",
        "torch_cuda_arch_list": ["sm_100", "sm_120", "compute_120"],
        "nccl_build_api_version": [2, 29, 7],
        "loaded_nccl": {
            "version": "2.29.7",
            "version_code": 22907,
            "library_basename": "libnccl.so.2",
            "mapped_path_sha256": hashlib.sha256(
                f"/runtime/rank-{rank}/libnccl.so.2".encode()
            ).hexdigest(),
        },
    }


def _materialize_multigpu_run(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    video = _valid_video_bytes()
    video_path = root / MULTI_GPU_LAYOUT.video_filename
    video_path.write_bytes(video)
    decoded = wan_rerun._decode_video_metrics(video_path)
    video_sha256 = hashlib.sha256(video).hexdigest()
    ranks = [_rank(rank) for rank in range(4)]

    _write_json(
        root / "npa_byof_summary.json",
        {
            "status": "success",
            "tool": "byof",
            "workload": "solution-smoke-wan22-b200-4gpu",
            "run_id": "wan-test-run",
            "image": _accepted_image_ref(),
            "solution_name": "wan2.2-multigpu",
            "capability_name": "wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses",
            "smoke_artifact_name": "wan2_2_ti2v_5b_multigpu.json",
            "smoke_exit_code": 0,
            "created_unix": 1.0,
        },
    )

    _write_json(
        root / "npa_source_metadata.json",
        {
            "source": "oss-byof",
            "repo": "https://github.com/Wan-Video/Wan2.2.git",
            "ref": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",
        },
    )
    _write_json(
        root / "wan2_2_ti2v_5b_multigpu.json",
        {
            "schema": "npa.workbench.byof.wan2_2_ti2v_5b_multigpu.v1",
            "solution": "wan2.2",
            "capability": "wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses",
            "upstream": {
                "repo": "https://github.com/Wan-Video/Wan2.2.git",
                "ref": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",
                "entrypoint": "python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 wan22_distributed_wrapper.py",
                "wrapper_execution": "runpy.run_path('/opt/byof/generate.py', run_name='__main__')",
            },
            "model": {
                "id": "Wan-AI/Wan2.2-TI2V-5B",
                "ref": "921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
                "weights_baked": False,
            },
            "tokenizer": {
                "id": "google/umt5-xxl",
                "ref": "66cb9e7e85526fe440a945569e42c72fb6cbc0ad",
            },
            "distributed": {
                "world_size": 4,
                "local_world_size": 4,
                "node_count": 1,
                "backend": "nccl",
                "fsdp": {
                    "enabled": True,
                    "sharding_strategy": "FULL_SHARD",
                    "t5": True,
                    "dit": True,
                },
                "ulysses": {"enabled": True, "size": 4, "num_attention_heads": 24},
                "topology_filename": "wan2_2_multigpu_topology.json",
            },
            "generation": {
                "prompt": "An abstract color study.",
                "seed": 42,
                "width": 1280,
                "height": 704,
                "frames": 17,
                "fps": 24.0,
                "steps": 8,
            },
            "observed": {
                "width": decoded["width"],
                "height": decoded["height"],
                "channels": 3,
                "frame_count": decoded["frame_count"],
                "fps": decoded["fps"],
                "codec": decoded["codec"],
                "max_spatial_std": decoded["max_spatial_std"],
                "pixel_range": decoded["pixel_range"],
                "mean_temporal_abs_delta": decoded["mean_temporal_abs_delta"],
            },
            "output": {
                "filename": MULTI_GPU_LAYOUT.video_filename,
                "size_bytes": len(video),
                "sha256": video_sha256,
            },
            "runtime": {
                "torch": "2.13.0+cu130",
                "torch_cuda": "13.0",
                "torch_cuda_arch_list": ["sm_100", "sm_120"],
                "driver_versions": ["580.159.04"],
                "nccl_build_api_version": [2, 29, 7],
                "nccl_loaded_version": "2.29.7",
            },
            "capabilities_exercised": [
                "wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses",
                "wan2.2_distributed_rank_topology_validation",
                "wan2.2_decoded_mp4_validation",
            ],
            "deferred": [],
        },
    )
    _write_json(
        root / "wan2_2_multigpu_topology.json",
        {
            "schema": "npa.workbench.byof.wan2_2_multigpu_topology.v1",
            "world_size": 4,
            "local_world_size": 4,
            "node_count": 1,
            "distributed_backend": "nccl",
            "fsdp": {
                "enabled": True,
                "sharding_strategy": "FULL_SHARD",
                "t5": True,
                "dit": True,
            },
            "ulysses": {"enabled": True, "size": 4, "num_attention_heads": 24},
            "rank_evidence": ranks,
        },
    )
    for rank, payload in enumerate(ranks):
        _write_json(root / f"wan2_2_multigpu_rank_{rank}.json", payload)
        _write_json(
            root / f"wan2_2_multigpu_progress_rank_{rank}.json",
            {
                "rank": rank,
                "local_rank": rank,
                "world_size": 4,
                "stage": "process_group_destroyed",
                "time_ns": rank + 1,
            },
        )
    rank_logs = []
    for rank in range(4):
        log_path = root / f"wan2_2_multigpu_nccl_rank_{rank}.log"
        log_path.write_text(
            f"rank {rank}: NCCL version 2.29.7+cuda13.0\nrank {rank}: Init COMPLETE\n",
            encoding="utf-8",
        )
        rank_logs.append(
            {
                "rank": rank,
                "filename": log_path.name,
                "size_bytes": log_path.stat().st_size,
                "sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
                "version_line_observed": True,
                "init_complete_observed": True,
            }
        )
    _write_json(
        root / "wan2_2_multigpu_nccl_summary.json",
        {
            "schema": "npa.workbench.byof.wan2_2_multigpu_nccl_summary.v1",
            "loaded_version": "2.29.7",
            "process_group_destroyed_on_all_ranks": True,
            "rank_logs": rank_logs,
        },
    )
    _write_json(
        root / "wan2_2_multigpu_runtime_inventory.json",
        {
            "schema": "npa.workbench.byof.wan2_2_multigpu_runtime_inventory.v1",
            "non_root": True,
            "source_ref": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",
            "weights_baked": False,
            "runtime": {
                "torch": "2.13.0+cu130",
                "torch_cuda": "13.0",
                "torch_cuda_arch_list": ["sm_100", "sm_120"],
                "driver_versions": ["580.159.04"],
                "nccl_build_api_version": [2, 29, 7],
                "nccl_loaded_version": "2.29.7",
            },
            "package_versions": {
                "torch": "2.13.0+cu130",
                "torchvision": "0.28.0",
                "nvidia-nccl-cu13": "2.29.7",
            },
            "devices": [rank["device"] for rank in ranks],
        },
    )


def _materialize_single_gpu_run(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    video = _valid_video_bytes()
    video_path = root / SINGLE_GPU_LAYOUT.video_filename
    video_path.write_bytes(video)
    decoded = wan_rerun._decode_video_metrics(video_path)
    _write_json(
        root / "npa_byof_summary.json",
        {
            "status": "success",
            "tool": "byof",
            "workload": "solution-smoke-wan22-rtxpro-gpu",
            "run_id": "wan-single-run",
            "image": _accepted_image_ref(),
            "solution_name": "wan2.2",
            "capability_name": "wan2.2_ti2v_5b_text_to_video",
            "smoke_artifact_name": "wan2_2_ti2v_5b_text_to_video.json",
            "smoke_exit_code": 0,
            "created_unix": 1.0,
        },
    )
    _write_json(
        root / "npa_source_metadata.json",
        {
            "source": "oss-byof",
            "repo": "https://github.com/Wan-Video/Wan2.2.git",
            "ref": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",
        },
    )
    _write_json(
        root / "wan2_2_ti2v_5b_text_to_video.json",
        {
            "schema": "npa.workbench.byof.wan2_2_ti2v_5b.v1",
            "solution": "wan2.2",
            "upstream_repo": "https://github.com/Wan-Video/Wan2.2.git",
            "upstream_ref": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",
            "model_id": "Wan-AI/Wan2.2-TI2V-5B",
            "model_ref": "921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
            "tokenizer_id": "google/umt5-xxl",
            "tokenizer_ref": "66cb9e7e85526fe440a945569e42c72fb6cbc0ad",
            "weights_baked": False,
            "task": "text-to-video",
            "prompt": "An abstract color study.",
            "seed": 42,
            "requested": {"width": 1280, "height": 704, "frames": 17, "fps": 24.0},
            "observed": {
                "width": decoded["width"],
                "height": decoded["height"],
                "frame_count": decoded["frame_count"],
                "fps": decoded["fps"],
                "codec": decoded["codec"],
                "max_spatial_std": decoded["max_spatial_std"],
                "pixel_range": decoded["pixel_range"],
                "mean_temporal_abs_delta": decoded["mean_temporal_abs_delta"],
            },
            "output_filename": SINGLE_GPU_LAYOUT.video_filename,
            "output_size_bytes": len(video),
            "output_sha256": hashlib.sha256(video).hexdigest(),
            "device_topology": {
                "cuda_device_count": 1,
                "devices": [
                    {
                        "index": 0,
                        "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                        "compute_capability": [12, 0],
                        "total_memory_bytes": 1024,
                    }
                ],
                "torch": "2.13.0+cu130",
                "torch_cuda": "13.0",
                "torch_cuda_arch_list": ["sm_100", "sm_120", "compute_120"],
                "driver_versions": ["580.159.04"],
                "attention_backend": "torch.nn.functional.scaled_dot_product_attention",
                "flash_attention_installed": False,
                "sdpa_source_binding": True,
                "sdpa_probe": {
                    "dtype": "torch.bfloat16",
                    "shape": [1, 4, 32, 64],
                    "finite": True,
                },
            },
            "capabilities_exercised": [
                "wan2.2_ti2v_5b_text_to_video",
                "wan2.2_decoded_mp4_validation",
            ],
            "deferred": [],
        },
    )
    _write_json(
        root / SINGLE_GPU_LAYOUT.inventory_filename,
        {
            "schema": "npa.workbench.byof.wan2_2_runtime_inventory.v1",
            "baked_runtime": {
                "non_root": True,
                "venv_readable": True,
                "large_checkpoint_shaped_files": [],
            },
            "runtime_stack": {
                "devices": [
                    {
                        "index": 0,
                        "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                        "compute_capability": [12, 0],
                        "total_memory_bytes": 1024,
                    }
                ],
                "torch": "2.13.0+cu130",
                "torch_cuda": "13.0",
                "torch_cuda_arch_list": ["sm_100", "sm_120", "compute_120"],
                "driver_versions": ["580.159.04"],
                "attention_backend": "torch.nn.functional.scaled_dot_product_attention",
                "flash_attention_installed": False,
                "sdpa_source_binding": True,
                "sdpa_probe": {
                    "dtype": "torch.bfloat16",
                    "shape": [1, 4, 32, 64],
                    "finite": True,
                },
            },
        },
    )


class _MemoryS3:
    def __init__(self, bucket: str, prefix: str, run_dir: Path) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        for path in run_dir.iterdir():
            if path.is_file():
                self.objects[(bucket, prefix + path.name)] = path.read_bytes()
        self.content_types: dict[tuple[str, str], str] = {}
        self.head_calls: list[tuple[str, str]] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        payload = self.objects[(Bucket, Key)]
        return {
            "Body": io.BytesIO(payload),
            "ContentLength": len(payload),
            "ETag": f'"{hashlib.md5(payload, usedforsecurity=False).hexdigest()}"',
            "VersionId": "version-1",
        }

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.head_calls.append((Bucket, Key))
        payload = self.objects[(Bucket, Key)]
        return {
            "ContentLength": len(payload),
            "ETag": f'"{hashlib.md5(payload, usedforsecurity=False).hexdigest()}"',
            "ContentType": self.content_types.get((Bucket, Key), "binary/octet-stream"),
        }

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
        IfNoneMatch: str,
    ) -> dict[str, Any]:
        assert IfNoneMatch == "*"
        if (Bucket, Key) in self.objects:
            raise RuntimeError("precondition failed: object already exists")
        self.objects[(Bucket, Key)] = bytes(Body)
        self.content_types[(Bucket, Key)] = ContentType
        return {"ETag": f'"{hashlib.md5(Body, usedforsecurity=False).hexdigest()}"'}


def test_build_wan_rrd_embeds_exact_video_and_timestamped_frames(
    tmp_path: Path,
) -> None:
    from rerun.recording import load_recording

    run_dir = tmp_path / "run"
    _materialize_multigpu_run(run_dir)
    output = tmp_path / MULTI_GPU_LAYOUT.rrd_filename

    result = build_wan_rrd(run_dir, output, layout=MULTI_GPU_LAYOUT)

    assert result["verification"]["rrd_parse"] == "passed"
    assert result["verification"]["rerun_cli_verify"] == "passed"
    assert result["verification"]["embedded_mp4_sha256_match"] is True
    assert result["verification"]["video_frame_reference_count"] == 17
    assert result["verification"]["video_timestamps_valid"] is True
    assert result["verification"]["entity_row_counts"]["/wan2_2/video/frame"] == 17
    entities = {str(chunk.entity_path) for chunk in load_recording(output).chunks()}
    assert "/wan2_2/video/asset" in entities
    assert "/wan2_2/evidence/ranks/rank_3" in entities
    assert "/wan2_2/evidence/execution" in entities
    assert "/wan2_2/evidence/distributed" not in entities


def test_single_gpu_layout_builds_and_uses_accurate_execution_entity(
    tmp_path: Path,
) -> None:
    from rerun.recording import load_recording

    run_dir = tmp_path / "single"
    _materialize_single_gpu_run(run_dir)
    output = tmp_path / SINGLE_GPU_LAYOUT.rrd_filename
    result = build_wan_rrd(run_dir, output, layout=SINGLE_GPU_LAYOUT)

    assert result["variant"] == "single-gpu"
    assert result["verification"]["video_frame_reference_count"] == 17
    entities = {str(chunk.entity_path) for chunk in load_recording(output).chunks()}
    assert "/wan2_2/evidence/execution" in entities
    assert not any("/distributed" in entity for entity in entities)


def test_single_gpu_rejects_stale_torch_cuda_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "single"
    _materialize_single_gpu_run(run_dir)
    artifact_path = run_dir / "wan2_2_ti2v_5b_text_to_video.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["device_topology"]["torch"] = "2.7.1+cu128"
    artifact["device_topology"]["torch_cuda"] = "12.8"
    _write_json(artifact_path, artifact)
    inventory_path = run_dir / SINGLE_GPU_LAYOUT.inventory_filename
    inventory = json.loads(inventory_path.read_text())
    inventory["runtime_stack"] = artifact["device_topology"]
    _write_json(inventory_path, inventory)

    with pytest.raises(WanRrdError, match="Torch 2.13.0 / CUDA 13.0"):
        build_wan_rrd(run_dir, tmp_path / "broken.rrd", layout=SINGLE_GPU_LAYOUT)


def test_single_gpu_schema_drift_fails_with_structured_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "single"
    _materialize_single_gpu_run(run_dir)
    artifact = run_dir / "wan2_2_ti2v_5b_text_to_video.json"
    payload = json.loads(artifact.read_text())
    payload.pop("tokenizer_id")
    _write_json(artifact, payload)

    with pytest.raises(WanRrdError, match="tokenizer identity"):
        build_wan_rrd(run_dir, tmp_path / "broken.rrd", layout=SINGLE_GPU_LAYOUT)


@pytest.mark.parametrize(
    "image",
    ["", "registry.example/wan:wrong", "registry.example/wan@sha256:" + "0" * 64],
)
def test_run_image_must_match_the_accepted_digest(tmp_path: Path, image: str) -> None:
    run_dir = tmp_path / "single"
    _materialize_single_gpu_run(run_dir)
    summary_path = run_dir / "npa_byof_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["image"] = image
    _write_json(summary_path, summary)

    with pytest.raises(WanRrdError, match="image"):
        build_wan_rrd(run_dir, tmp_path / "broken.rrd", layout=SINGLE_GPU_LAYOUT)


def test_host_postprocess_rejects_a_forged_nondecodable_mp4(tmp_path: Path) -> None:
    run_dir = tmp_path / "single"
    _materialize_single_gpu_run(run_dir)
    video_path = run_dir / SINGLE_GPU_LAYOUT.video_filename
    forged = b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 64
    video_path.write_bytes(forged)
    artifact = run_dir / "wan2_2_ti2v_5b_text_to_video.json"
    payload = json.loads(artifact.read_text())
    payload["output_size_bytes"] = len(forged)
    payload["output_sha256"] = hashlib.sha256(forged).hexdigest()
    _write_json(artifact, payload)

    with pytest.raises(WanRrdError, match="MP4 probe failed"):
        build_wan_rrd(run_dir, tmp_path / "broken.rrd", layout=SINGLE_GPU_LAYOUT)


def test_single_gpu_runtime_gate_drift_fails_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "single"
    _materialize_single_gpu_run(run_dir)
    artifact = run_dir / "wan2_2_ti2v_5b_text_to_video.json"
    payload = json.loads(artifact.read_text())
    payload["device_topology"]["sdpa_probe"]["finite"] = False
    _write_json(artifact, payload)

    with pytest.raises(WanRrdError, match="BF16 SDPA gate"):
        build_wan_rrd(run_dir, tmp_path / "broken.rrd", layout=SINGLE_GPU_LAYOUT)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("driver_mismatch", "disagrees on driver_versions"),
        ("venv_unreadable", "runtime venv was not readable"),
    ],
)
def test_single_gpu_runtime_identity_drift_fails_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    run_dir = tmp_path / "single"
    _materialize_single_gpu_run(run_dir)
    inventory_path = run_dir / SINGLE_GPU_LAYOUT.inventory_filename
    inventory = json.loads(inventory_path.read_text())
    if mutation == "driver_mismatch":
        inventory["runtime_stack"]["driver_versions"] = ["different"]
    else:
        inventory["baked_runtime"]["venv_readable"] = False
    _write_json(inventory_path, inventory)

    with pytest.raises(WanRrdError, match=message):
        build_wan_rrd(run_dir, tmp_path / "broken.rrd", layout=SINGLE_GPU_LAYOUT)


def test_single_gpu_publish_manifest_and_source_roles(tmp_path: Path) -> None:
    run_dir = tmp_path / "single"
    _materialize_single_gpu_run(run_dir)
    bucket = "test-bucket"
    prefix = "oss-solutions/wan2.2/wan-single-run/"
    s3 = _MemoryS3(bucket, prefix, run_dir)

    result = publish_wan_rrd_from_s3(
        f"s3://{bucket}/{prefix}", variant="single", s3_client=s3
    )
    manifest = json.loads(
        s3.objects[(bucket, prefix + SINGLE_GPU_LAYOUT.manifest_filename)]
    )
    assert result["status"] == "verified"
    assert manifest["variant"] == "single-gpu"
    assert (
        manifest["container_image"]["oci_digest"]
        == wan_accepted_image_manifest()["oci_digest"]
    )
    assert len(manifest["source_objects"]) == 5
    assert not any(
        item["role"].startswith("distributed") for item in manifest["source_objects"]
    )


@pytest.mark.parametrize(
    "stats",
    [
        "num_chunks = 3\nnum_entity_paths = 4\nnum_rows = 9\n",
        "chunks: 3; entities: 4; rows: 9; static: 2",
        "",
    ],
)
def test_rerun_stats_format_drift_fails_closed(stats: str) -> None:
    with pytest.raises(WanRrdError, match="missing required field"):
        wan_rerun._stats_fields(stats)


def test_blueprint_is_supplied_exactly_once(tmp_path: Path, monkeypatch) -> None:
    import rerun as rr

    run_dir = tmp_path / "single"
    _materialize_single_gpu_run(run_dir)
    evidence = wan_rerun.validate_wan_run(run_dir, SINGLE_GPU_LAYOUT)
    calls = {"save": 0, "send": 0}

    class Recording:
        def save(self, _path, *, default_blueprint=None):
            assert default_blueprint is not None
            calls["save"] += 1

        def send_blueprint(self, _blueprint):
            calls["send"] += 1

        def log(self, *_args, **_kwargs):
            return None

        def flush(self):
            return None

        def disconnect(self):
            return None

    monkeypatch.setattr(rr, "RecordingStream", lambda *_args, **_kwargs: Recording())
    monkeypatch.setattr(rr, "set_time", lambda *_args, **_kwargs: None)
    wan_rerun._log_recording(tmp_path / "once.rrd", evidence)
    assert calls == {"save": 1, "send": 0}


def test_build_wan_rrd_rejects_rank_evidence_disagreement(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _materialize_multigpu_run(run_dir)
    rank = _rank(0)
    rank["observer_final_barrier"] = False
    _write_json(run_dir / "wan2_2_multigpu_rank_0.json", rank)

    with pytest.raises(WanRrdError, match="disagrees with topology"):
        build_wan_rrd(
            run_dir,
            tmp_path / MULTI_GPU_LAYOUT.rrd_filename,
            layout=MULTI_GPU_LAYOUT,
        )


def test_multigpu_rejects_stale_primary_torch_cuda_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _materialize_multigpu_run(run_dir)
    artifact_path = run_dir / "wan2_2_ti2v_5b_multigpu.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["runtime"]["torch"] = "2.7.1+cu128"
    artifact["runtime"]["torch_cuda"] = "12.8"
    _write_json(artifact_path, artifact)

    with pytest.raises(WanRrdError, match="Torch 2.13.0 / CUDA 13.0"):
        build_wan_rrd(
            run_dir,
            tmp_path / MULTI_GPU_LAYOUT.rrd_filename,
            layout=MULTI_GPU_LAYOUT,
        )


def test_multigpu_rejects_stale_loaded_nccl_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _materialize_multigpu_run(run_dir)
    topology_path = run_dir / "wan2_2_multigpu_topology.json"
    topology = json.loads(topology_path.read_text())
    stale_rank = topology["rank_evidence"][0]
    stale_rank["loaded_nccl"]["version"] = "2.27.7"
    stale_rank["loaded_nccl"]["version_code"] = 22707
    _write_json(topology_path, topology)
    _write_json(run_dir / "wan2_2_multigpu_rank_0.json", stale_rank)

    with pytest.raises(WanRrdError, match="accepted NCCL 2.29.7"):
        build_wan_rrd(
            run_dir,
            tmp_path / MULTI_GPU_LAYOUT.rrd_filename,
            layout=MULTI_GPU_LAYOUT,
        )


def test_multigpu_rejects_stale_nccl_build_api_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _materialize_multigpu_run(run_dir)
    topology_path = run_dir / "wan2_2_multigpu_topology.json"
    topology = json.loads(topology_path.read_text())
    stale_rank = topology["rank_evidence"][0]
    stale_rank["nccl_build_api_version"] = [2, 26, 2]
    _write_json(topology_path, topology)
    _write_json(run_dir / "wan2_2_multigpu_rank_0.json", stale_rank)

    with pytest.raises(WanRrdError, match="wrong NCCL build API version"):
        build_wan_rrd(
            run_dir,
            tmp_path / MULTI_GPU_LAYOUT.rrd_filename,
            layout=MULTI_GPU_LAYOUT,
        )


def test_build_wan_rrd_rejects_distributed_wrapper_identity_drift(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _materialize_multigpu_run(run_dir)
    artifact_path = run_dir / "wan2_2_ti2v_5b_multigpu.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["upstream"]["wrapper_execution"] = "untrusted.py"
    _write_json(artifact_path, artifact)

    with pytest.raises(WanRrdError, match="wrapper execution identity"):
        build_wan_rrd(
            run_dir,
            tmp_path / MULTI_GPU_LAYOUT.rrd_filename,
            layout=MULTI_GPU_LAYOUT,
        )


def test_build_wan_rrd_requires_post_destroy_and_loaded_nccl_evidence(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _materialize_multigpu_run(run_dir)
    progress = run_dir / "wan2_2_multigpu_progress_rank_2.json"
    payload = json.loads(progress.read_text())
    payload["stage"] = "process_group_initialized"
    _write_json(progress, payload)

    with pytest.raises(WanRrdError, match="post-destroy marker"):
        build_wan_rrd(
            run_dir,
            tmp_path / MULTI_GPU_LAYOUT.rrd_filename,
            layout=MULTI_GPU_LAYOUT,
        )


def test_build_wan_rrd_rejects_loaded_nccl_summary_corruption(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _materialize_multigpu_run(run_dir)
    summary_path = run_dir / "wan2_2_multigpu_nccl_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["loaded_version"] = "0.0.0"
    _write_json(summary_path, summary)

    with pytest.raises(WanRrdError, match="loaded version"):
        build_wan_rrd(
            run_dir,
            tmp_path / MULTI_GPU_LAYOUT.rrd_filename,
            layout=MULTI_GPU_LAYOUT,
        )


def test_build_wan_rrd_reparses_nccl_log_content(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _materialize_multigpu_run(run_dir)
    log_path = run_dir / "wan2_2_multigpu_nccl_rank_1.log"
    log_path.write_text("rank 1: forged log\n", encoding="utf-8")
    summary_path = run_dir / "wan2_2_multigpu_nccl_summary.json"
    summary = json.loads(summary_path.read_text())
    entry = summary["rank_logs"][1]
    entry["size_bytes"] = log_path.stat().st_size
    entry["sha256"] = hashlib.sha256(log_path.read_bytes()).hexdigest()
    entry["version_line_observed"] = True
    entry["init_complete_observed"] = True
    _write_json(summary_path, summary)

    with pytest.raises(WanRrdError, match="initialization/version evidence"):
        build_wan_rrd(
            run_dir,
            tmp_path / MULTI_GPU_LAYOUT.rrd_filename,
            layout=MULTI_GPU_LAYOUT,
        )


def test_publish_wan_rrd_rechecks_remote_bytes_and_writes_manifest(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _materialize_multigpu_run(run_dir)
    bucket = "test-bucket"
    prefix = "oss-solutions/wan2.2-multigpu/wan-test-run/"
    s3 = _MemoryS3(bucket, prefix, run_dir)

    result = publish_wan_rrd_from_s3(
        f"s3://{bucket}/{prefix}",
        variant="multigpu",
        s3_client=s3,
    )

    assert result["status"] == "verified"
    assert result["capability"] == RRD_CAPABILITY
    assert result["rrd_uri"].endswith(MULTI_GPU_LAYOUT.rrd_filename)
    assert result["manifest_uri"].endswith(MULTI_GPU_LAYOUT.manifest_filename)
    assert (
        result["video_sha256"]
        == hashlib.sha256(
            (run_dir / MULTI_GPU_LAYOUT.video_filename).read_bytes()
        ).hexdigest()
    )
    manifest = json.loads(
        s3.objects[(bucket, prefix + MULTI_GPU_LAYOUT.manifest_filename)]
    )
    assert manifest["schema"] == RRD_MANIFEST_SCHEMA
    assert manifest["status"] == "verified"
    assert manifest["capability"] == RRD_CAPABILITY
    assert manifest["container_image"] == result["container_image"]
    assert manifest["rrd"]["sha256"] == result["rrd_sha256"]
    assert manifest["video"]["embedded_sha256"] == result["video_sha256"]
    assert manifest["verification"]["remote_rerun_cli_verify"] == "passed"
    assert len(manifest["source_objects"]) == 19
    roles = {item["role"] for item in manifest["source_objects"]}
    assert "distributed_post_destroy_evidence" in roles
    assert "distributed_loaded_nccl_summary" in roles
    assert "distributed_nccl_log" in roles
    source_keys = {
        (bucket, prefix + Path(item["uri"]).name) for item in manifest["source_objects"]
    }
    assert source_keys.isdisjoint(s3.head_calls)
    assert s3.content_types[(bucket, prefix + MULTI_GPU_LAYOUT.manifest_filename)] == (
        "application/json"
    )


def test_publish_wan_rrd_refuses_to_overwrite_an_existing_recording(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _materialize_single_gpu_run(run_dir)
    bucket = "test-bucket"
    prefix = "oss-solutions/wan2.2/immutable-run/"
    s3 = _MemoryS3(bucket, prefix, run_dir)

    publish_wan_rrd_from_s3(
        f"s3://{bucket}/{prefix}", variant="single-gpu", s3_client=s3
    )
    with pytest.raises(RuntimeError, match="object already exists"):
        publish_wan_rrd_from_s3(
            f"s3://{bucket}/{prefix}", variant="single-gpu", s3_client=s3
        )
