"""cuRobo contracts: no fake success, complete denominators, private scoped I/O."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from npa.workbench.curobo import runtime
from npa.workbench.curobo.artifacts import (
    CuroboError,
    _expected_rrd_chunks,
    _scan_decoded_chunks,
    build_rrd,
    canonical,
    decode_rrd,
    summarize,
    validate_report,
)
from npa.workbench.curobo.schemas import (
    BenchmarkManifest,
    PlanManifest,
    Pose,
    PrepareRequest,
    RunRequest,
    SOURCE_REVISION,
)
from npa.workbench.curobo.service import create_app


def row():
    return {
        "mode": "kinematic",
        "dataset": "synthetic",
        "problem_id": "case",
        "status": "success",
        "query": {
            "start": [0.0],
            "goal_pose": {
                "position_xyz": [0.1, 0.0, 0.0],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        },
        "metrics": {"wall_plan_seconds": 0.01},
        "trajectory": {
            "joint_names": ["joint"],
            "dt": 0.1,
            "position": [[0.0], [0.2]],
            "velocity": [[0.0], [0.1]],
            "acceleration": [[0.0], [0.0]],
            "jerk": [[0.0], [0.0]],
            "tool_position": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            "tool_quaternion": [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
        },
    }


def plan_row():
    result = row()
    result["dataset"] = "operator"
    result["metrics"].update(
        planner_total_seconds=0.008,
        solver_seconds=0.006,
        position_error_m=0.001,
        rotation_error_rad=0.002,
        joint_path_length_rad=0.2,
        tool_path_length_m=0.1,
        trajectory_duration_seconds=0.1,
        max_abs_jerk_rad_s3=0.0,
    )
    return result


def request(**kwargs):
    return RunRequest(
        input_path="s3://example-bucket/input",
        output_path="s3://example-bucket/output",
        run_id="unit-run",
        **kwargs,
    )


def test_denominators_preserve_failures_and_invalid_inputs():
    rows = [
        row(),
        {
            "mode": "kinematic",
            "dataset": "synthetic",
            "problem_id": "failed",
            "status": "failed",
        },
        {
            "mode": "kinematic",
            "dataset": "synthetic",
            "problem_id": "invalid",
            "status": "invalid",
        },
    ]
    group = summarize(rows)["kinematic"]
    assert group["input_count"] == 3
    assert group["eligible_count"] == 2
    assert group["success_fraction_all"] == 1 / 3
    assert group["success_fraction_eligible"] == 0.5
    assert group["metrics"]["wall_plan_seconds"]["count"] == 1


@pytest.mark.parametrize(
    "mutation",
    ["nan", "shape", "dt", "tool", "quaternion", "duplicate", "false_solution"],
)
def test_malformed_journal_never_passes(mutation):
    rows = [row()]
    if mutation == "nan":
        rows[0]["metrics"]["wall_plan_seconds"] = float("nan")
    if mutation == "shape":
        rows[0]["trajectory"]["velocity"] = [[0.0]]
    if mutation == "dt":
        rows[0]["trajectory"]["dt"] = -1
    if mutation == "tool":
        rows[0]["trajectory"]["tool_position"] = [[0, 0, 0]]
    if mutation == "quaternion":
        rows[0]["trajectory"]["tool_quaternion"][0] = [2, 0, 0, 0]
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    if mutation == "false_solution":
        rows[0]["status"] = "failed"
    with pytest.raises(CuroboError):
        summarize(rows)


def test_strict_manifest_rejects_executable_configs_and_nonunit_pose():
    with pytest.raises(ValidationError):
        Pose(position_xyz=[0, 0, 0], quaternion_wxyz=[2, 0, 0, 0])
    with pytest.raises(ValidationError):
        PlanManifest(robot="/tmp/robot.yml", problems=[])
    with pytest.raises(ValidationError):
        BenchmarkManifest(modes=["kinematic", "kinematic"])
    with pytest.raises(ValidationError):
        BenchmarkManifest(limit=1)


def test_public_operations_refuse_local_paths_before_io(monkeypatch):
    monkeypatch.setattr(
        runtime, "read_bytes_uri", lambda *a: pytest.fail("unexpected read")
    )
    with pytest.raises(ValueError):
        runtime.plan(
            RunRequest(
                input_path="/tmp/input",
                output_path="s3://example-bucket/output",
                run_id="unit",
            )
        )


def test_prepare_full_recipe_readback_and_hash_mismatch(monkeypatch):
    objects = {}
    monkeypatch.setattr(
        runtime,
        "write_bytes_uri",
        lambda uri, payload: objects.__setitem__(uri, payload),
    )
    monkeypatch.setattr(runtime, "read_bytes_uri", lambda uri: objects[uri])
    runtime.prepare(PrepareRequest(output_path="s3://example-bucket/recipe.json"))
    assert json.loads(next(iter(objects.values())))["modes"] == [
        "kinematic",
        "dynamics",
    ]
    monkeypatch.setattr(runtime, "read_bytes_uri", lambda uri: b"different")
    with pytest.raises(CuroboError, match="digest mismatch"):
        runtime.prepare(PrepareRequest(output_path="s3://example-bucket/recipe.json"))


def test_gpu_subprocess_failure_preserves_interrupted_journal_and_receipt(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("NPA_CUROBO_WORK_DIR", str(tmp_path))
    partial_journal = canonical(row()) + b"\n\n" + b"42\n" + b'{"status":"truncated"'
    objects = {
        request().input_path: canonical(BenchmarkManifest().model_dump(mode="json"))
    }
    events = []
    calls = []

    def read(uri):
        events.append(("read", uri))
        return objects[uri]

    def write(uri, payload):
        events.append(("write", uri))
        objects[uri] = payload

    def run(argv, **kwargs):
        calls.append(argv)
        kwargs["stdout"].write(b"upstream traceback\n")
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir()
        (output / "problems.jsonl").write_bytes(partial_journal)
        (output / "result.json").write_bytes(b'{"status":"untrusted-partial"}')
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(runtime, "read_bytes_uri", read)
    monkeypatch.setattr(runtime, "write_bytes_uri", write)
    monkeypatch.setattr(runtime.subprocess, "run", run)
    with pytest.raises(
        CuroboError,
        match=r"exit code 7; durable failure receipt: _failures/unit-run/failure.json",
    ):
        runtime.benchmark(request())

    namespace = request().output_path + "/_failures/unit-run"
    log_uri = namespace + "/runtime.log"
    partial_uri = namespace + "/partial-problems.jsonl"
    receipt_uri = namespace + "/failure.json"
    expected_log = b"upstream traceback\n"
    receipt = json.loads(objects[receipt_uri])
    assert receipt == {
        "schema_version": "npa.curobo.failure.v1",
        "status": "failed",
        "failure_type": "subprocess_exit",
        "kind": "benchmark",
        "run_id": "unit-run",
        "subprocess_exit_code": 7,
        "artifacts": [
            {
                "role": "runtime_log",
                "path": "runtime.log",
                "bytes": len(expected_log),
                "sha256": hashlib.sha256(expected_log).hexdigest(),
            },
            {
                "role": "partial_journal",
                "path": "partial-problems.jsonl",
                "bytes": len(partial_journal),
                "sha256": hashlib.sha256(partial_journal).hexdigest(),
                "partial": True,
                "physical_line_count": 4,
                "complete_record_count": 1,
            },
        ],
    }
    assert objects[log_uri] == expected_log
    assert objects[partial_uri] == partial_journal
    assert events == [
        ("read", request().input_path),
        ("write", log_uri),
        ("read", log_uri),
        ("write", partial_uri),
        ("read", partial_uri),
        ("write", receipt_uri),
        ("read", receipt_uri),
    ]
    assert not any(uri.endswith("/result.json") for uri in objects)
    assert len(calls) == 1
    logs = list(tmp_path.glob("*/runtime.log"))
    assert len(logs) == 1 and logs[0].read_bytes() == expected_log
    assert (logs[0].parent / "output/result.json").is_file()
    assert logs[0].parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize(
    ("failure", "secondary_type"),
    [("write", "OSError"), ("readback", "CuroboError")],
)
def test_failure_publication_error_preserves_subprocess_failure(
    monkeypatch, tmp_path, failure, secondary_type
):
    monkeypatch.setenv("NPA_CUROBO_WORK_DIR", str(tmp_path))
    objects = {
        request().input_path: canonical(BenchmarkManifest().model_dump(mode="json"))
    }

    def read(uri):
        payload = objects[uri]
        if failure == "readback" and uri.endswith("/failure.json"):
            return b"changed remote receipt"
        return payload

    def write(uri, payload):
        if failure == "write" and uri.endswith("/failure.json"):
            raise OSError("credential-shaped private storage diagnostic")
        objects[uri] = payload

    def run(argv, **kwargs):
        kwargs["stdout"].write(b"primary runner failure\n")
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir()
        (output / "problems.jsonl").write_bytes(canonical(row()) + b"\n")
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(runtime, "read_bytes_uri", read)
    monkeypatch.setattr(runtime, "write_bytes_uri", write)
    monkeypatch.setattr(runtime.subprocess, "run", run)
    with pytest.raises(CuroboError) as raised:
        runtime.benchmark(request())
    message = str(raised.value)
    assert message == (
        "upstream cuRobo benchmark failed with exit code 9; "
        f"failure evidence publication failed ({secondary_type})"
    )
    assert "credential-shaped" not in message
    assert request().output_path + "/_failures/unit-run/runtime.log" in objects
    assert (
        request().output_path + "/_failures/unit-run/partial-problems.jsonl" in objects
    )
    assert not any(uri.endswith("/result.json") for uri in objects)


@pytest.mark.parametrize("operation", [runtime.validate, runtime.visualize])
def test_failure_namespace_is_never_an_accepted_result(operation, monkeypatch):
    monkeypatch.setattr(
        runtime, "read_bytes_uri", lambda *_: pytest.fail("read failure namespace")
    )
    failed = RunRequest(
        input_path="s3://example-bucket/output/_failures/unit-run",
        output_path="s3://example-bucket/review",
        run_id="unit-run",
    )
    with pytest.raises(CuroboError, match="not an accepted cuRobo result"):
        operation(failed)


@pytest.fixture
def completed_plan(monkeypatch, tmp_path):
    """Real artifact validation/publication, with only GPU and storage mocked."""
    monkeypatch.setenv("NPA_CUROBO_WORK_DIR", str(tmp_path))
    manifest = PlanManifest(
        problems=[
            {
                "id": "case",
                "start": [0, 0, 0, 0, 0, 0, 0],
                "goal_pose": {
                    "position_xyz": [0.5, 0, 0.3],
                    "quaternion_wxyz": [1, 0, 0, 0],
                },
            }
        ]
    ).model_dump(mode="json")
    objects = {request().input_path: canonical(manifest)}
    events = []
    calls = []

    def read(uri):
        events.append(("read", uri))
        return objects[uri]

    def write(uri, payload):
        events.append(("write", uri))
        objects[uri] = payload

    def run(argv, **kwargs):
        calls.append(argv)
        kwargs["stdout"].write(b"completed solver fixture\n")
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir()
        (output / "problems.jsonl").write_bytes(canonical(plan_row()) + b"\n")
        (output / "result.json").write_bytes(
            canonical(
                {
                    "schema_version": "npa.curobo.result.v1",
                    "engine": "nvidia-curobo-v2",
                    "source_revision": SOURCE_REVISION,
                    "run_id": "unit-run",
                    "kind": "plan",
                    "requested_modes": ["kinematic"],
                    "summary": summarize([plan_row()]),
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime, "read_bytes_uri", read)
    monkeypatch.setattr(runtime, "write_bytes_uri", write)
    monkeypatch.setattr(runtime.subprocess, "run", run)
    return SimpleNamespace(root=tmp_path, objects=objects, events=events, calls=calls)


def test_success_cleans_only_owned_directory_after_both_readbacks(
    completed_plan, monkeypatch
):
    state = completed_plan
    unrelated = state.root / "another-operation"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("unrelated evidence")
    removed = []
    original = runtime.shutil.rmtree

    def cleanup(directory):
        assert state.events == [
            ("read", request().input_path),
            ("write", request().output_path + "/problems.jsonl"),
            ("read", request().output_path + "/problems.jsonl"),
            ("write", request().output_path + "/result.json"),
            ("read", request().output_path + "/result.json"),
        ]
        removed.append(directory)
        original(directory)

    monkeypatch.setattr(runtime.shutil, "rmtree", cleanup)
    result = runtime.plan(request())
    assert len(state.calls) == 1
    assert len(removed) == 1 and removed[0].parent == state.root
    assert not removed[0].exists()
    assert (unrelated / "keep.txt").read_text() == "unrelated evidence"
    assert json.loads(state.objects[request().output_path + "/result.json"]) == result


@pytest.mark.parametrize("failure", ["write", "readback"])
def test_partial_publication_preserves_completed_journal(
    completed_plan, monkeypatch, failure
):
    state = completed_plan
    original_read, original_write = runtime.read_bytes_uri, runtime.write_bytes_uri

    def read(uri):
        data = original_read(uri)
        return (
            b"altered remote bytes"
            if failure == "readback" and uri.endswith("result.json")
            else data
        )

    def write(uri, data):
        if failure == "write" and uri.endswith("result.json"):
            raise OSError("simulated object-store write failure")
        original_write(uri, data)

    monkeypatch.setattr(runtime, "read_bytes_uri", read)
    monkeypatch.setattr(runtime, "write_bytes_uri", write)
    monkeypatch.setattr(
        runtime.shutil, "rmtree", lambda *_: pytest.fail("cleaned failed publication")
    )
    with pytest.raises((OSError, CuroboError)):
        runtime.plan(request())
    directories = list(state.root.iterdir())
    assert len(directories) == 1
    assert directories[0].stat().st_mode & 0o777 == 0o700
    assert (directories[0] / "output/problems.jsonl").read_bytes() == canonical(
        plan_row()
    ) + b"\n"
    assert (
        directories[0] / "runtime.log"
    ).read_bytes() == b"completed solver fixture\n"
    assert request().output_path + "/problems.jsonl" in state.objects
    assert len(state.calls) == 1


@pytest.mark.parametrize("error_type", [PermissionError, RuntimeError])
def test_cleanup_failure_keeps_success_and_emits_only_fixed_warning(
    completed_plan, monkeypatch, caplog, error_type
):
    state = completed_plan

    def fail(directory):
        raise error_type(f"private diagnostic at {directory}")

    monkeypatch.setattr(runtime.shutil, "rmtree", fail)
    with caplog.at_level("WARNING", logger="npa.workbench.curobo.runtime"):
        result = runtime.plan(request())
    assert result["summary"]["kinematic"]["success"] == 1
    assert json.loads(state.objects[request().output_path + "/result.json"]) == result
    assert len(state.calls) == 1
    assert len(list(state.root.iterdir())) == 1
    assert caplog.messages == [
        "cuRobo artifacts verified; local working-file cleanup failed"
    ]
    assert "private diagnostic" not in caplog.text
    assert str(state.root) not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_validation_recomputes_facts_and_detects_hash_tampering(monkeypatch):
    journal = canonical(plan_row()) + b"\n"
    report = {
        "schema_version": "npa.curobo.result.v1",
        "engine": "nvidia-curobo-v2",
        "source_revision": SOURCE_REVISION,
        "kind": "plan",
        "requested_modes": ["kinematic"],
        "run_id": "unit-run",
        "journal_sha256": hashlib.sha256(journal).hexdigest(),
        "summary": summarize([plan_row()]),
    }
    objects = {
        "s3://example-bucket/input/problems.jsonl": journal,
        "s3://example-bucket/input/result.json": canonical(report),
    }
    monkeypatch.setattr(runtime, "read_bytes_uri", lambda uri: objects[uri])
    monkeypatch.setattr(
        runtime,
        "write_bytes_uri",
        lambda uri, payload: objects.__setitem__(uri, payload),
    )
    monkeypatch.setattr(
        runtime,
        "replay_rows",
        lambda _rows, _report: {
            "valid": True,
            "terminal_goal_distance_m": {"max": 0.0, "mean": 0.0},
            "terminal_goal_orientation_rad": {"max": 0.0, "mean": 0.0},
        },
    )
    assert runtime.validate(request())["valid"] is True
    objects["s3://example-bucket/input/problems.jsonl"] += b" "
    with pytest.raises(CuroboError, match="hash mismatch"):
        runtime.validate(request())


@pytest.mark.parametrize(
    "distance,orientation,accepted",
    [
        (0.005, 0.05, True),
        (0.005001, 0.0, False),
        (0.0, 0.050001, False),
        (float("nan"), 0.0, False),
        (0.0, float("nan"), False),
    ],
)
def test_validation_requires_independent_terminal_pose_replay(
    distance, orientation, accepted
):
    replay = {
        "terminal_goal_distance_m": {"max": distance},
        "terminal_goal_orientation_rad": {"max": orientation},
    }
    if accepted:
        runtime._require_replay_tolerance(replay)
    else:
        with pytest.raises(CuroboError, match="terminal goal"):
            runtime._require_replay_tolerance(replay)


@pytest.mark.parametrize(
    "endpoint", ["prepare", "plan", "benchmark", "run", "validate", "visualize"]
)
def test_service_authorization_and_failure_mapping(endpoint, monkeypatch):
    client = TestClient(
        create_app(token="unit-token", allowed_s3_roots=["s3://example-bucket/"])
    )
    body = (
        {"output_path": "s3://example-bucket/output"}
        if endpoint == "prepare"
        else request().model_dump()
    )
    assert client.post("/" + endpoint, json=body).status_code == 401

    def fail(_):
        raise CuroboError("synthetic failure")

    monkeypatch.setattr(runtime, "benchmark" if endpoint == "run" else endpoint, fail)
    response = client.post(
        "/" + endpoint, json=body, headers={"Authorization": "Bearer unit-token"}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "synthetic failure"


def test_service_storage_scope_refuses_cross_root(monkeypatch):
    client = TestClient(
        create_app(
            token="unit-token", allowed_s3_roots=["s3://example-bucket/allowed/"]
        )
    )
    monkeypatch.setattr(
        runtime, "read_bytes_uri", lambda *_: pytest.fail("cross-root read")
    )
    response = client.post(
        "/plan",
        json=request().model_dump(),
        headers={"Authorization": "Bearer unit-token"},
    )
    assert response.status_code == 400
    assert "outside" in response.json()["detail"]


def test_factual_rrd_round_trip(tmp_path):

    journal = tmp_path / "problems.jsonl"
    journal.write_bytes(canonical(row()) + b"\n")
    target = tmp_path / "planning.rrd"
    result = build_rrd(journal, target, run_id="unit-rrd")
    assert result["successful_trajectories"] == 1
    assert result["run_id"] == "unit-rrd"
    assert result["journal_sha256"] == hashlib.sha256(journal.read_bytes()).hexdigest()
    assert result["goal_markers"] == 1
    assert result["trajectory_samples"] == 2
    decoded_path = tmp_path / "rrd-print.txt"
    decoded = decode_rrd(
        target, rows=[row()], run_id="unit-rrd", decoded_output=decoded_path
    )
    assert decoded["verify"] == "passed"
    assert decoded["print"] == "passed"
    assert decoded["print_bytes"] == decoded_path.stat().st_size
    assert decoded["chunk_mismatches"] == []
    assert decoded["decoded_chunk_rows"] == 23
    import sys

    executable = Path(sys.executable).with_name("rerun")
    verified = subprocess.run(
        [str(executable), "rrd", "verify", str(target)], capture_output=True, text=True
    )
    assert verified.returncode == 0, verified.stderr
    printed = subprocess.run(
        [str(executable), "rrd", "print", "-vv", str(target)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for entity in (
        "npa.curobo",
        "unit-rrd",
        "problem_index",
        "trajectory_time",
        "tool_path",
        "tool_quaternion/w",
        "problems/000000/goal",
        "joints/0/position",
    ):
        assert entity in printed


@pytest.mark.parametrize(
    "mutation", ["joint", "tool", "quaternion", "timeline", "metric"]
)
def test_decoded_rrd_rejects_same_cardinality_wrong_semantics(tmp_path, mutation):
    import rerun as rr
    from npa.workbench.curobo.artifacts import log_trajectory_columns

    original = row()
    journal_bytes = canonical(original) + b"\n"
    target = tmp_path / "adversarial.rrd"
    recording = rr.RecordingStream("npa.curobo", recording_id="semantic-run")
    recording.save(str(target))
    recording.log(
        "provenance",
        rr.TextDocument(
            json.dumps(
                {
                    "producer": "npa.workbench.curobo",
                    "source_revision": SOURCE_REVISION,
                    "dataset_revision": None,
                    "run_id": "semantic-run",
                    "journal_sha256": hashlib.sha256(journal_bytes).hexdigest(),
                    "limitations": "FK tool paths and joint traces; no rendered robot meshes or independent collision certification.",
                }
            )
        ),
        static=True,
    )
    recording.set_time("problem_index", sequence=0)
    recording.log(
        "problems/000000/status",
        rr.TextDocument(
            json.dumps(
                {
                    key: original[key]
                    for key in ("problem_id", "mode", "dataset", "status")
                }
            )
        ),
    )
    recording.log(
        "problems/000000/goal",
        rr.Points3D(
            [original["query"]["goal_pose"]["position_xyz"]],
            radii=0.015,
            colors=[0, 255, 0],
        ),
    )
    changed = copy.deepcopy(original["trajectory"])
    metrics = copy.deepcopy(original["metrics"])
    if mutation == "joint":
        changed["position"][1][0] += 0.25
    elif mutation == "tool":
        changed["tool_position"][1][0] += 0.25
    elif mutation == "quaternion":
        changed["tool_quaternion"][1] = [0.0, 1.0, 0.0, 0.0]
    elif mutation == "timeline":
        changed["dt"] = 0.2
    elif mutation == "metric":
        metrics["wall_plan_seconds"] += 0.25
    for name, value in metrics.items():
        recording.log(f"metrics/{name}", rr.Scalars(value))
    recording.log(
        "problems/000000/joint_names",
        rr.TextDocument(json.dumps(original["trajectory"]["joint_names"])),
    )
    recording.log(
        "problems/000000/tool_path",
        rr.LineStrips3D([changed["tool_position"]]),
    )
    log_trajectory_columns(recording, "problems/000000", changed, problem_index=0)
    recording.flush()
    del recording

    with pytest.raises(CuroboError, match="values or factual timelines"):
        decode_rrd(target, rows=[original], run_id="semantic-run")


def test_decoded_rrd_coverage_rejects_any_missing_problem_or_sample_chunk(tmp_path):
    solved = plan_row()
    failed = {
        "mode": "kinematic",
        "dataset": "operator",
        "problem_id": "failed",
        "status": "failed",
        "query": copy.deepcopy(solved["query"]),
        "metrics": {"wall_plan_seconds": 0.1},
    }
    expected = _expected_rrd_chunks([solved, failed])
    decoded = tmp_path / "truncated-print.txt"
    lines = [b'npa.curobo "coverage-run"']
    for entity, count in expected.items():
        if entity == "problems/000000/joints/0/position":
            continue
        lines.append(f"Chunk(x) with {count} rows (1 B) - /{entity} -".encode())
    decoded.write_bytes(b"\n".join(lines) + b"\n")
    _digest, _size, mismatches = _scan_decoded_chunks(
        decoded, expected, run_id="coverage-run"
    )
    assert mismatches == ["problems/000000/joints/0/position: expected 2, decoded 0"]


def test_functional_smoke_retains_complete_positive_and_failure_evidence(
    tmp_path, monkeypatch, capsys
):
    from npa.smoke import test_curobo_functional as smoke

    def execute(_kind, manifest, output, *, run_id):
        assert [problem["id"] for problem in manifest["problems"]] == [
            "franka-pose",
            "blocked-goal-control",
        ]
        solved = plan_row()
        solved["problem_id"] = "franka-pose"
        failed = {
            "mode": "kinematic",
            "dataset": "operator",
            "problem_id": "blocked-goal-control",
            "status": "failed",
            "query": copy.deepcopy(solved["query"]),
            "metrics": {"wall_plan_seconds": 0.02},
        }
        rows = [solved, failed]
        output.mkdir(parents=True)
        (output / "problems.jsonl").write_bytes(
            b"".join(canonical(item) + b"\n" for item in rows)
        )
        report = {
            "schema_version": "npa.curobo.result.v1",
            "engine": "nvidia-curobo-v2",
            "source_revision": SOURCE_REVISION,
            "dataset_revision": None,
            "kind": "plan",
            "requested_modes": ["kinematic"],
            "run_id": run_id,
            "gpu": {
                "name": "synthetic-gpu",
                "compute_capability": [10, 0],
                "torch_version": "test",
                "cuda_version": "test",
            },
            "summary": summarize(rows),
            "limitations": ["unit fixture"],
        }
        (output / "result.json").write_bytes(canonical(report))
        return report

    monkeypatch.setattr(smoke, "execute", execute)
    monkeypatch.setenv("NPA_SMOKE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("NPA_SMOKE_RUN_ID", "retained-smoke")
    monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", "a" * 40)
    monkeypatch.setenv("NPA_IMAGE_DIGEST", "sha256:" + "b" * 64)
    monkeypatch.setenv("NPA_EXPECTED_IMAGE_DIGEST", "sha256:" + "b" * 64)
    monkeypatch.setenv("NPA_OUTPUT_PATH", "s3://example-bucket/golden/")
    uploaded = {}
    monkeypatch.setattr(
        smoke,
        "write_bytes_uri",
        lambda uri, payload: uploaded.__setitem__(uri, payload),
    )
    monkeypatch.setattr(smoke, "read_bytes_uri", lambda uri: uploaded[uri])
    monkeypatch.setattr(
        smoke,
        "replay_rows",
        lambda _rows, _report: {
            "valid": True,
            "terminal_goal_distance_m": {"max": 0.0, "mean": 0.0},
            "terminal_goal_orientation_rad": {"max": 0.0, "mean": 0.0},
        },
    )
    smoke.main()
    root = tmp_path / "retained-smoke"
    assert root.stat().st_mode & 0o777 == 0o700
    expected = {
        "artifact-manifest.json",
        "controls.json",
        "independent-validation.json",
        "input.json",
        "output/problems.jsonl",
        "output/result.json",
        "planning.rrd",
        "rrd-manifest.json",
        "rrd-print.txt",
        "upload-receipt.json",
    }
    assert {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    } == expected
    manifest = json.loads((root / "artifact-manifest.json").read_text())
    assert manifest["source_commit"] == "a" * 40
    assert manifest["image_digest"] == "sha256:" + "b" * 64
    assert manifest["objective"] == {
        "success": 1,
        "failed_control": 1,
        "invalid": 0,
        "independent_validation": True,
        "independent_replay": True,
        "rrd_decode": "passed",
    }
    assert (
        json.loads((root / "controls.json").read_text())["valid_but_infeasible"][
            "observed_status"
        ]
        == "failed"
    )
    receipt = json.loads((root / "upload-receipt.json").read_text())
    assert receipt["readback_verified"] is True
    assert receipt["workload_status"] == "passed"
    assert receipt["errors"] == []
    assert {item["path"] for item in receipt["objects"]} == expected - {
        "upload-receipt.json"
    }
    assert set(uploaded) == {"s3://example-bucket/golden/" + path for path in expected}
    assert json.loads(capsys.readouterr().out)["status"] == "passed"


def test_functional_smoke_uploads_partial_artifacts_and_failure_receipt(
    tmp_path, monkeypatch, capsys
):
    from npa.smoke import test_curobo_functional as smoke

    def fail_after_partial_output(_kind, _manifest, output, *, run_id):
        output.mkdir(parents=True)
        (output / "partial.log").write_text(f"{run_id}: planner failed")
        raise RuntimeError("synthetic planner failure")

    monkeypatch.setattr(smoke, "execute", fail_after_partial_output)
    monkeypatch.setenv("NPA_SMOKE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("NPA_SMOKE_RUN_ID", "failed-smoke")
    monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", "a" * 40)
    monkeypatch.setenv("NPA_IMAGE_DIGEST", "sha256:" + "b" * 64)
    monkeypatch.setenv("NPA_EXPECTED_IMAGE_DIGEST", "sha256:" + "b" * 64)
    monkeypatch.setenv("NPA_OUTPUT_PATH", "s3://example-bucket/golden/")
    uploaded = {}
    monkeypatch.setattr(
        smoke,
        "write_bytes_uri",
        lambda uri, payload: uploaded.__setitem__(uri, payload),
    )
    monkeypatch.setattr(smoke, "read_bytes_uri", lambda uri: uploaded[uri])

    with pytest.raises(RuntimeError, match="synthetic planner"):
        smoke.main()

    root = tmp_path / "failed-smoke"
    receipt = json.loads((root / "upload-receipt.json").read_text())
    assert receipt["workload_status"] == "failed"
    assert receipt["readback_verified"] is True
    assert {item["path"] for item in receipt["objects"]} == {
        "failure.json",
        "input.json",
        "output/partial.log",
    }
    assert "s3://example-bucket/golden/failure.json" in uploaded
    assert "s3://example-bucket/golden/upload-receipt.json" in uploaded
    failure_output = json.loads(capsys.readouterr().out)
    assert failure_output["status"] == "failed"
    assert failure_output["evidence_upload_failure_type"] is None


def test_functional_smoke_retains_partial_upload_failure_receipt(
    tmp_path, monkeypatch, capsys
):
    from npa.smoke import test_curobo_functional as smoke

    (tmp_path / "kept.txt").write_text("kept")
    (tmp_path / "rejected.txt").write_text("rejected")
    uploaded = {}

    def publish(uri, payload):
        if uri.endswith("/rejected.txt"):
            raise RuntimeError("synthetic upload failure")
        uploaded[uri] = payload

    monkeypatch.setattr(smoke, "_publish", publish)
    with pytest.raises(RuntimeError, match="upload or read-back"):
        smoke._upload_tree(
            tmp_path,
            output_uri="s3://example-bucket/golden/",
            run_id="partial-upload",
            image_digest="sha256:" + "b" * 64,
            workload_status="failed",
        )
    receipt = json.loads((tmp_path / "upload-receipt.json").read_text())
    assert receipt["readback_verified"] is False
    assert receipt["errors"] == [{"path": "rejected.txt", "error": "RuntimeError"}]
    assert "s3://example-bucket/golden/upload-receipt.json" in uploaded
    assert json.loads(capsys.readouterr().out)["status"] == "evidence-upload-failed"


@pytest.mark.parametrize(
    "missing,error",
    [
        ("NPA_IMAGE_SOURCE_SHA", "NPA_IMAGE_SOURCE_SHA"),
        ("NPA_IMAGE_DIGEST", "NPA_IMAGE_DIGEST"),
        ("NPA_EXPECTED_IMAGE_DIGEST", "NPA_EXPECTED_IMAGE_DIGEST"),
        ("NPA_OUTPUT_PATH", "NPA_OUTPUT_PATH"),
        ("NPA_SMOKE_OUTPUT_DIR", "NPA_SMOKE_OUTPUT_DIR"),
    ],
)
def test_functional_smoke_requires_immutable_durable_identity(
    tmp_path, monkeypatch, missing, error
):
    from npa.smoke import test_curobo_functional as smoke

    values = {
        "NPA_IMAGE_SOURCE_SHA": "a" * 40,
        "NPA_IMAGE_DIGEST": "sha256:" + "b" * 64,
        "NPA_EXPECTED_IMAGE_DIGEST": "sha256:" + "b" * 64,
        "NPA_OUTPUT_PATH": "s3://example-bucket/golden/",
        "NPA_SMOKE_OUTPUT_DIR": str(tmp_path),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(missing)
    monkeypatch.setattr(
        smoke, "execute", lambda *_a, **_k: pytest.fail("planner called")
    )
    with pytest.raises(RuntimeError, match=error):
        smoke.main()
    assert not list(tmp_path.iterdir())


def test_functional_smoke_requires_absolute_output_directory(tmp_path, monkeypatch):
    from npa.smoke import test_curobo_functional as smoke

    monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", "a" * 40)
    monkeypatch.setenv("NPA_IMAGE_DIGEST", "sha256:" + "b" * 64)
    monkeypatch.setenv("NPA_EXPECTED_IMAGE_DIGEST", "sha256:" + "b" * 64)
    monkeypatch.setenv("NPA_OUTPUT_PATH", "s3://example-bucket/golden/")
    monkeypatch.setenv("NPA_SMOKE_OUTPUT_DIR", "relative-output")
    monkeypatch.setattr(
        smoke, "execute", lambda *_a, **_k: pytest.fail("planner called")
    )
    with pytest.raises(RuntimeError, match="must be absolute"):
        smoke.main()
    assert not list(tmp_path.iterdir())


def test_functional_smoke_rejects_different_frozen_digest_before_planner(
    tmp_path, monkeypatch
):
    from npa.smoke import test_curobo_functional as smoke

    monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", "a" * 40)
    monkeypatch.setenv("NPA_IMAGE_DIGEST", "sha256:" + "b" * 64)
    monkeypatch.setenv("NPA_EXPECTED_IMAGE_DIGEST", "sha256:" + "c" * 64)
    monkeypatch.setenv("NPA_OUTPUT_PATH", "s3://example-bucket/golden/")
    monkeypatch.setenv("NPA_SMOKE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        smoke, "execute", lambda *_a, **_k: pytest.fail("planner called")
    )
    with pytest.raises(RuntimeError, match="frozen candidate"):
        smoke.main()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "distance,orientation,accepted",
    [
        (0.005, 0.05, True),
        (0.0050001, 0.0, False),
        (0.0, 0.050001, False),
        (float("nan"), 0.0, False),
        (0.0, float("nan"), False),
    ],
)
def test_functional_smoke_enforces_terminal_pose_threshold(
    distance, orientation, accepted
):
    from npa.smoke import test_curobo_functional as smoke

    report = {
        "terminal_goal_distance_m": {"max": distance},
        "terminal_goal_orientation_rad": {"max": orientation},
    }
    if accepted:
        smoke._require_feasible_distance(report)
    else:
        with pytest.raises(RuntimeError, match="tolerance"):
            smoke._require_feasible_distance(report)


def test_partial_benchmark_cannot_pass_with_self_consistent_summary():
    from npa.workbench.curobo.schemas import DATASET_REVISION

    sample = row()
    sample["dataset"] = "motion_benchmaker"
    report = {
        "schema_version": "npa.curobo.result.v1",
        "engine": "nvidia-curobo-v2",
        "source_revision": SOURCE_REVISION,
        "dataset_revision": DATASET_REVISION,
        "run_id": "test",
        "kind": "benchmark",
        "requested_modes": ["kinematic"],
        "summary": summarize([sample]),
    }
    with pytest.raises(CuroboError, match="incomplete benchmark"):
        validate_report(report, [sample], run_id="test")


@pytest.mark.parametrize(
    "mutation", ["none", "drop", "swap_id", "extra", "dataset", "mode", "exclude"]
)
def test_plan_result_requires_exact_requested_population(
    monkeypatch, tmp_path, mutation
):
    monkeypatch.setenv("NPA_CUROBO_WORK_DIR", str(tmp_path))
    template = {
        "start": [0, 0, 0, 0, 0, 0, 0],
        "goal_pose": {"position_xyz": [0.5, 0, 0.3], "quaternion_wxyz": [1, 0, 0, 0]},
    }
    manifest = PlanManifest(
        problems=[{"id": "first", **template}, {"id": "second", **template}]
    ).model_dump(mode="json")
    objects = {"s3://example-bucket/input": canonical(manifest)}
    monkeypatch.setattr(runtime, "read_bytes_uri", lambda uri: objects[uri])
    monkeypatch.setattr(
        runtime,
        "write_bytes_uri",
        lambda uri, payload: objects.__setitem__(uri, payload),
    )

    def run(argv, **kwargs):
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir()
        rows = [{**plan_row(), "problem_id": name} for name in ("first", "second")]
        if mutation == "drop":
            rows.pop()
        elif mutation == "swap_id":
            rows[0]["problem_id"] = "replacement"
        elif mutation == "extra":
            rows.append({**plan_row(), "problem_id": "extra"})
        elif mutation == "dataset":
            rows[0]["dataset"] = "unexpected"
        elif mutation == "mode":
            rows[0]["mode"] = "dynamics"
        elif mutation == "exclude":
            rows[0] = {
                k: v for k, v in rows[0].items() if k not in ("trajectory", "metrics")
            }
            rows[0]["status"] = "invalid"
        (output / "problems.jsonl").write_bytes(
            b"".join(canonical(r) + b"\n" for r in rows)
        )
        (output / "result.json").write_bytes(
            canonical(
                {
                    "schema_version": "npa.curobo.result.v1",
                    "engine": "nvidia-curobo-v2",
                    "source_revision": SOURCE_REVISION,
                    "run_id": "unit-run",
                    "kind": "plan",
                    "requested_modes": ["kinematic"],
                    "summary": summarize(rows),
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime.subprocess, "run", run)
    if mutation == "none":
        assert runtime.plan(request())["summary"]["kinematic"]["input_count"] == 2
        assert len(objects) == 3
    else:
        with pytest.raises(CuroboError, match="(identities|exclude)"):
            runtime.plan(request())
        assert len(objects) == 1


def test_rrd_column_batches_preserve_all_joint_and_fk_samples(tmp_path):
    import sys
    import rerun as rr
    from npa.workbench.curobo.artifacts import log_trajectory_columns

    trajectory = row()["trajectory"]
    expected = tmp_path / "expected.rrd"
    actual = tmp_path / "actual.rrd"
    reference = rr.RecordingStream("comparison", recording_id="same")
    reference.save(str(expected))
    for frame in range(len(trajectory["position"])):
        reference.set_time("problem_index", sequence=4)
        reference.set_time("trajectory_time", duration=frame * trajectory["dt"])
        reference.log(
            "trajectory/tool", rr.Points3D([trajectory["tool_position"][frame]])
        )
        for component, name in enumerate(("w", "x", "y", "z")):
            reference.log(
                f"trajectory/tool_quaternion/{name}",
                rr.Scalars(trajectory["tool_quaternion"][frame][component]),
            )
        for field in ("position", "velocity", "acceleration", "jerk"):
            reference.log(
                f"trajectory/joints/0/{field}", rr.Scalars(trajectory[field][frame][0])
            )
    reference.flush()
    del reference
    batched = rr.RecordingStream("comparison", recording_id="same")
    batched.save(str(actual))
    log_trajectory_columns(batched, "trajectory", trajectory, problem_index=4)
    batched.flush()
    del batched
    executable = str(Path(sys.executable).with_name("rerun"))
    outputs = []
    for path in (expected, actual):
        filtered = path.with_name(path.stem + "-filtered.rrd")
        subprocess.run(
            [
                executable,
                "rrd",
                "filter",
                "--drop-timeline",
                "log_tick",
                "--drop-timeline",
                "log_time",
                "--output",
                str(filtered),
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        outputs.append(str(filtered))
    comparison = subprocess.run(
        [executable, "rrd", "compare", "--unordered", *outputs],
        capture_output=True,
        text=True,
    )
    assert comparison.returncode == 0, comparison.stdout + comparison.stderr


@pytest.mark.parametrize(
    "revision, package_version", [("wrong", "0.8.0"), (SOURCE_REVISION, "0.7.0")]
)
def test_runtime_revision_or_legacy_package_cannot_claim_v2(
    monkeypatch, tmp_path, revision, package_version
):
    from npa.workbench.curobo import runner

    (tmp_path / "NPA_SOURCE_REVISION").write_text(revision)
    monkeypatch.setenv("NPA_CUROBO_SOURCE", str(tmp_path))
    monkeypatch.setattr(runner, "version", lambda _name: package_version)
    with pytest.raises(CuroboError, match="reviewed V2"):
        runner._runtime_source()
