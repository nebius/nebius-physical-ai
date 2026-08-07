from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from npa.workflows.wan_rerun import (
    MULTI_GPU_LAYOUT,
    RRD_CAPABILITY,
    RRD_MANIFEST_SCHEMA,
    WanRrdError,
    build_wan_rrd,
    publish_wan_rrd_from_s3,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
        "torch": "2.7.1+cu128",
        "torch_cuda": "12.8",
        "torch_cuda_arch_list": ["sm_100", "sm_120", "compute_120"],
        "nccl_version": [2, 26, 2],
    }


def _materialize_multigpu_run(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    video = b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 64
    video_path = root / MULTI_GPU_LAYOUT.video_filename
    video_path.write_bytes(video)
    video_sha256 = hashlib.sha256(video).hexdigest()
    ranks = [_rank(rank) for rank in range(4)]

    _write_json(
        root / "npa_byof_summary.json",
        {
            "status": "success",
            "tool": "byof",
            "workload": "solution-smoke-wan22-b200-4gpu",
            "run_id": "wan-test-run",
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
                "entrypoint": "torchrun --standalone --nnodes=1 --nproc_per_node=4 generate.py",
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
                "width": 1280,
                "height": 704,
                "channels": 3,
                "frame_count": 17,
                "fps": 24.0,
                "codec": "h264",
                "max_spatial_std": 46.9,
                "pixel_range": 255,
                "mean_temporal_abs_delta": 0.73,
            },
            "output": {
                "filename": MULTI_GPU_LAYOUT.video_filename,
                "size_bytes": len(video),
                "sha256": video_sha256,
            },
            "runtime": {
                "torch": "2.7.1+cu128",
                "torch_cuda": "12.8",
                "torch_cuda_arch_list": ["sm_100", "sm_120"],
                "driver_versions": ["580.159.04"],
                "nccl_version": [2, 26, 2],
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
        root / "wan2_2_multigpu_runtime_inventory.json",
        {
            "schema": "npa.workbench.byof.wan2_2_multigpu_runtime_inventory.v1",
            "non_root": True,
            "source_ref": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",
            "weights_baked": False,
            "runtime": {
                "torch": "2.7.1+cu128",
                "torch_cuda": "12.8",
                "driver_versions": ["580.159.04"],
                "nccl_version": [2, 26, 2],
            },
            "package_versions": {"torch": "2.7.1+cu128"},
            "devices": [rank["device"] for rank in ranks],
        },
    )


class _MemoryS3:
    def __init__(self, bucket: str, prefix: str, run_dir: Path) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        for path in run_dir.iterdir():
            if path.is_file():
                self.objects[(bucket, prefix + path.name)] = path.read_bytes()
        self.content_types: dict[tuple[str, str], str] = {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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
    assert manifest["rrd"]["sha256"] == result["rrd_sha256"]
    assert manifest["video"]["embedded_sha256"] == result["video_sha256"]
    assert manifest["verification"]["remote_rerun_cli_verify"] == "passed"
    assert len(manifest["source_objects"]) == 10
    assert s3.content_types[(bucket, prefix + MULTI_GPU_LAYOUT.manifest_filename)] == (
        "application/json"
    )
