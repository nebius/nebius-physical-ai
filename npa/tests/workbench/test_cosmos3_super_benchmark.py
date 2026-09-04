from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from npa.workbench.cosmos import super_benchmark as benchmark


def _valid_record(name: str, latency: float = 2.0) -> dict:
    return {
        "attempt_id": name,
        "http_status": 200,
        "output_bytes": 10,
        "technical_valid": True,
        "failure_reason": None,
        "client_wall_seconds": latency,
        "validation": {"valid": True},
    }


def test_plan_pins_primary_contract() -> None:
    plan = benchmark.benchmark_plan(
        output_path="s3://example-bucket/run/",
        topologies="1x8,2x4,4x2,8x1",
        attempts=24,
    )
    assert plan["runtime_image"] == benchmark.IMAGE
    assert plan["model"]["revision"] == benchmark.MODEL_REVISION
    assert plan["workload"] == benchmark.WORKLOAD
    assert [cell["services"] for cell in plan["topologies"]] == [1, 2, 4, 8]
    assert all(cell["request_concurrency_per_service"] == 1 for cell in plan["topologies"])
    assert all(cell["warmups_per_service"] == 1 for cell in plan["topologies"])
    assert plan["sync_timeout_seconds"] == 5400
    assert plan["gpu"] == {"family": "B200", "node_gpu_count": 8}
    assert plan["schema_version"] == benchmark.SCHEMA_VERSION


def test_h200_plan_changes_only_hardware_identity() -> None:
    b200 = benchmark.benchmark_plan(
        output_path="s3://example-bucket/b200/", topologies="1x8,2x4,4x2,8x1"
    )
    h200 = benchmark.benchmark_plan(
        output_path="s3://example-bucket/h200/",
        topologies="1x8,2x4,4x2,8x1",
        gpu_family="h200",
    )
    assert h200["gpu"] == {"family": "H200", "node_gpu_count": 8}
    assert h200["schema_version"] == "npa.cosmos3-super.h200-benchmark.v1"
    for key in ("model", "runtime_image", "topologies", "workload", "seeds"):
        assert h200[key] == b200[key]


def test_h200_single_gpu_plan_is_tp1_and_explicitly_not_a_paper_cell() -> None:
    plan = benchmark.benchmark_plan(
        output_path="s3://example-bucket/h200-single/",
        topologies="1x1",
        attempts=24,
        gpu_family="H200",
        suite="h200-single-gpu",
    )
    assert plan["schema_version"] == (
        "npa.cosmos3-super.h200-single-gpu-validation.v1"
    )
    assert plan["gpu"] == {"family": "H200", "node_gpu_count": 1}
    assert plan["validation_scope"] == {
        "kind": "single-gpu-functional-performance",
        "paper_reproduction": False,
        "paper_cell": None,
        "claim": (
            "one H200, one TP-1 service, sequential requests; not the paper's "
            "eight-replica 8x1 node cell"
        ),
    }
    assert plan["cells"] == [
        {
            "name": "H200_TP1_1GPU",
            "topology": "1x1",
            "services": 1,
            "gpus_per_service": 1,
            "server_parallelism": ["--tensor-parallel-size", "1"],
            "request_concurrency_per_service": 1,
            "warmups_per_service": 1,
            "measured_attempts": 24,
            "repeat_of": None,
        }
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gpu_family": "B200"}, "requires the H200"),
        ({"topologies": "8x1"}, "fixes topologies to 1x1"),
        ({"attempts": 23}, "exactly 24"),
    ],
)
def test_h200_single_gpu_suite_rejects_scope_drift(
    kwargs: dict, message: str
) -> None:
    options = {
        "output_path": "/tmp/results",
        "topologies": "1x1",
        "attempts": 24,
        "gpu_family": "H200",
        "suite": "h200-single-gpu",
    }
    options.update(kwargs)
    with pytest.raises(benchmark.Cosmos3SuperBenchmarkError, match=message):
        benchmark.benchmark_plan(**options)


def test_full_suite_matches_machine_readable_public_record() -> None:
    plan = benchmark.benchmark_plan(
        output_path="s3://example-bucket/full/",
        topologies="1x8,2x4,4x2,8x1",
        attempts=24,
        suite="b200-full",
    )
    assert [cell["name"] for cell in plan["cells"]] == [
        "T1_1x8",
        "T2_2x4",
        "T3_4x2",
        "T4_8x1",
        "T1C2",
        "T2C2",
        "T3C2",
        "T4C2",
        "T1R",
        "T1C2R",
    ]
    assert [cell["request_concurrency_per_service"] for cell in plan["cells"]] == [
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        1,
        2,
    ]
    assert all(cell["measured_attempts"] == 24 for cell in plan["cells"])
    assert plan["cells"][-2]["repeat_of"] == "T1_1x8"
    assert plan["cells"][-1]["repeat_of"] == "T1C2"
    assert plan["upstream"]["method_revision"] == benchmark.UPSTREAM_METHOD_REVISION
    assert plan["upstream"]["b200_record_sha256"] == (
        benchmark.UPSTREAM_B200_RECORD_SHA256
    )


def test_public_prompt_asset_normalizes_only_terminal_newline() -> None:
    value = '{"negative": true}'
    digest = benchmark._sha256_text(value)
    assert (
        benchmark._public_prompt_asset(
            value + "\n", expected_sha256=digest, name="negative prompt asset"
        )
        == value
    )
    with pytest.raises(benchmark.Cosmos3SuperBenchmarkError, match="SHA-256"):
        benchmark._public_prompt_asset(
            value + " ", expected_sha256=digest, name="negative prompt asset"
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gpu_family": "H200"}, "exact public B200"),
        ({"attempts": 16}, "exactly 24"),
        ({"topologies": "1x8,8x1"}, "fixes topologies"),
    ],
)
def test_full_suite_rejects_contract_drift(kwargs: dict, message: str) -> None:
    options = {
        "output_path": "/tmp/results",
        "topologies": "1x8,2x4,4x2,8x1",
        "attempts": 24,
        "suite": "b200-full",
    }
    options.update(kwargs)
    with pytest.raises(benchmark.Cosmos3SuperBenchmarkError, match=message):
        benchmark.benchmark_plan(**options)


def test_plan_rejects_unqualified_gpu_family() -> None:
    with pytest.raises(benchmark.Cosmos3SuperBenchmarkError, match="choose from"):
        benchmark.benchmark_plan(
            output_path="/tmp/results", topologies="1x8", gpu_family="H100"
        )


def test_plan_rejects_uneven_service_distribution() -> None:
    with pytest.raises(benchmark.Cosmos3SuperBenchmarkError, match="divide evenly"):
        benchmark.benchmark_plan(
            output_path="/tmp/results", topologies="8x1", attempts=23
        )


def test_service_commands_fill_node_with_expected_parallelism() -> None:
    hybrid = benchmark.service_command(benchmark.TOPOLOGIES["1x8"], port=8100)
    assert hybrid[-1] == "--no-guardrails"
    assert hybrid[hybrid.index("--revision") + 1] == benchmark.MODEL_REVISION
    assert "--cfg-parallel-size" in hybrid
    assert "--ulysses-degree" in hybrid
    assert "--use-hsdp" in hybrid
    assert "--hsdp-shard-size" in hybrid
    for name, size in (("2x4", "4"), ("4x2", "2"), ("8x1", "1")):
        command = benchmark.service_command(benchmark.TOPOLOGIES[name], port=8100)
        assert command[command.index("--tensor-parallel-size") + 1] == size
    single = benchmark.service_command(benchmark.TOPOLOGIES["1x1"], port=8100)
    assert single[single.index("--tensor-parallel-size") + 1] == "1"


def test_failed_attempt_keeps_window_time_and_gets_zero_credit() -> None:
    failed = _valid_record("failed", latency=9.0)
    failed.update(
        {
            "http_status": 500,
            "output_bytes": 0,
            "technical_valid": False,
            "failure_reason": "http_500",
            "validation": None,
        }
    )
    derived = benchmark.derive_cell([_valid_record("ok", 3.0), failed], 12.0)
    assert derived["attempts"] == 2
    assert derived["valid_attempts"] == 1
    assert derived["failed_attempts"] == 1
    assert derived["credited_valid_video_seconds"] == 7.875
    assert derived["valid_video_seconds_per_node_hour"] == 2362.5
    assert derived["window_seconds"] == 12.0


def test_resource_normalized_metrics_are_explicit() -> None:
    derived = benchmark.derive_cell([_valid_record("ok")], 90.0)
    benchmark._add_resource_normalized_metrics(
        derived, gpu_count=1, service_count=1
    )
    assert derived["valid_video_seconds_per_gpu_hour"] == 315.0
    assert derived["valid_video_seconds_per_service_hour"] == 315.0


def test_dispatch_runs_one_request_at_a_time_per_service(tmp_path: Path) -> None:
    active: dict[int, int] = {}
    peaks: dict[int, int] = {}
    seeds: dict[int, list[int]] = {}
    lock = threading.Lock()

    def fake_attempt(**kwargs):
        replica = kwargs["replica"]
        with lock:
            active[replica] = active.get(replica, 0) + 1
            peaks[replica] = max(peaks.get(replica, 0), active[replica])
            seeds.setdefault(replica, []).append(kwargs["seed"])
        time.sleep(0.005)
        with lock:
            active[replica] -= 1
        row = _valid_record(kwargs["attempt_id"], latency=0.005)
        row.update(
            {
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:00:01Z",
                "replica": replica,
            }
        )
        return row

    topology = benchmark.TOPOLOGIES["4x2"]
    records, window = benchmark.dispatch_cell(
        topology=topology,
        urls=[f"http://127.0.0.1:{8100 + index}" for index in range(4)],
        attempts=24,
        prompt="prompt",
        negative_prompt="negative",
        prompt_hashes={"prompt_sha256": "a", "negative_prompt_sha256": "b"},
        clips_dir=tmp_path,
        kind="production",
        attempt_fn=fake_attempt,
    )
    assert len(records) == 24
    assert set(peaks.values()) == {1}
    assert all(values == [17, 23, 41, 17, 23, 41] for values in seeds.values())
    assert window["seconds"] > 0
    assert "tail idle time" in window["boundary"]


def test_dispatch_concurrency_two_is_per_service_not_replica_count(
    tmp_path: Path,
) -> None:
    active: dict[int, int] = {}
    peaks: dict[int, int] = {}
    lock = threading.Lock()

    def fake_attempt(**kwargs):
        replica = kwargs["replica"]
        with lock:
            active[replica] = active.get(replica, 0) + 1
            peaks[replica] = max(peaks.get(replica, 0), active[replica])
        time.sleep(0.01)
        with lock:
            active[replica] -= 1
        row = _valid_record(kwargs["attempt_id"], latency=0.01)
        row.update(
            {
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:00:01Z",
                "replica": replica,
            }
        )
        return row

    topology = benchmark.TOPOLOGIES["2x4"]
    records, window = benchmark.dispatch_cell(
        topology=topology,
        urls=["http://127.0.0.1:8100", "http://127.0.0.1:8101"],
        attempts=24,
        prompt="prompt",
        negative_prompt="negative",
        prompt_hashes={"prompt_sha256": "a", "negative_prompt_sha256": "b"},
        clips_dir=tmp_path,
        kind="production",
        request_concurrency_per_service=2,
        cell_name="T2C2",
        attempt_fn=fake_attempt,
    )
    assert len(records) == 24
    assert peaks == {0: 2, 1: 2}
    assert {record["cell"] for record in records} == {"T2C2"}
    assert {record["request_concurrency_per_service"] for record in records} == {2}
    assert window["request_concurrency_per_service"] == 2


class _MemoryStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_directory(self, local_dir: str, bucket_uri: str) -> str:
        for root, _dirs, files in os.walk(local_dir):
            for name in files:
                path = Path(root) / name
                relative = path.relative_to(local_dir).as_posix()
                self.objects[f"{bucket_uri.rstrip('/')}/{relative}"] = path.read_bytes()
        return bucket_uri

    def put_bytes_conditional(self, payload: bytes, uri: str, **_kwargs) -> str:
        if uri in self.objects:
            raise benchmark.StoragePreconditionFailed("exists")
        self.objects[uri] = payload
        return "etag"

    def read_bytes_with_etag(self, uri: str):
        payload = self.objects.get(uri)
        return None if payload is None else (payload, "etag")


def test_completed_cell_marker_supports_hash_verified_resume(tmp_path: Path) -> None:
    storage = _MemoryStorage()
    cell = benchmark.B200_FULL_CELLS[0]
    payload = {
        "cell": cell.name,
        "attempts": [_valid_record(f"a{index}") for index in range(24)],
        "derived": {"attempts": 24, "valid_attempts": 24, "failed_attempts": 0},
    }
    benchmark._publish_completed_cell(
        storage=storage,
        output_path="s3://example-bucket/run",
        cell=cell,
        cell_dir=tmp_path,
        payload=payload,
        contract_sha256="contract",
    )
    resumed = benchmark._load_completed_cell(
        storage=storage,
        output_path="s3://example-bucket/run",
        cell=cell,
        contract_sha256="contract",
    )
    assert resumed == payload
    storage.objects["s3://example-bucket/run/cells/T1_1x8/cell.json"] += b" "
    with pytest.raises(benchmark.Cosmos3SuperBenchmarkError, match="hash check"):
        benchmark._load_completed_cell(
            storage=storage,
            output_path="s3://example-bucket/run",
            cell=cell,
            contract_sha256="contract",
        )


def test_full_suite_comparisons_cover_primary_concurrency_and_repeats() -> None:
    cells = {}
    for index, cell in enumerate(benchmark.B200_FULL_CELLS, start=1):
        cells[cell.name] = {
            "derived": {
                "mean_request_latency_seconds": float(index),
                "valid_video_seconds_per_node_hour": float(index * 10),
            }
        }
    comparisons = benchmark.derive_suite_comparisons(cells)
    assert set(comparisons) == {
        "primary_frontier",
        "concurrency_two_delta_pct",
        "repeat_variation_pct",
    }
    assert set(comparisons["concurrency_two_delta_pct"]) == {
        "T1C2",
        "T2C2",
        "T3C2",
        "T4C2",
    }


def test_video_gate_checks_shape_blank_and_motion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(benchmark.shutil, "which", lambda _name: "/usr/bin/tool")
    stream = {
        "streams": [
            {
                "width": 1280,
                "height": 720,
                "avg_frame_rate": "24/1",
                "nb_read_frames": "189",
                "duration": "7.875",
            }
        ]
    }
    frames = b"".join(bytes((value + index) % 256 for value in range(256)) * 4 for index in range(5))

    def fake_run(command, *, binary=False):
        if "ffprobe" in command[0]:
            return subprocess.CompletedProcess(command, 0, json.dumps(stream), "")
        if binary:
            return subprocess.CompletedProcess(command, 0, frames, b"")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(benchmark, "_run", fake_run)
    result = benchmark.validate_video(tmp_path / "clip.mp4")
    assert result["valid"] is True
    assert result["stream"]["nb_read_frames"] == "189"
