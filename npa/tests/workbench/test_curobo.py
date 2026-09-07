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
    build_rrd,
    canonical,
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
        "metrics": {"wall_plan_seconds": 0.01},
        "trajectory": {
            "joint_names": ["joint"],
            "dt": 0.1,
            "position": [[0.0], [0.2]],
            "velocity": [[0.0], [0.1]],
            "acceleration": [[0.0], [0.0]],
            "jerk": [[0.0], [0.0]],
            "tool_position": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
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
    "mutation", ["nan", "shape", "dt", "tool", "duplicate", "false_solution"]
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


def test_gpu_subprocess_failure_retains_evidence_and_never_uploads(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("NPA_CUROBO_WORK_DIR", str(tmp_path))
    monkeypatch.setattr(
        runtime,
        "read_bytes_uri",
        lambda uri: canonical(BenchmarkManifest().model_dump()),
    )
    monkeypatch.setattr(
        runtime, "write_bytes_uri", lambda *a: pytest.fail("uploaded failed operation")
    )
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        kwargs["stdout"].write(b"upstream failure")
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(runtime.subprocess, "run", run)
    with pytest.raises(CuroboError, match="exit code 7"):
        runtime.benchmark(request())
    assert len(calls) == 1
    logs = list(tmp_path.glob("*/runtime.log"))
    assert len(logs) == 1 and logs[0].read_bytes() == b"upstream failure"
    assert logs[0].parent.stat().st_mode & 0o777 == 0o700


@pytest.fixture
def completed_plan(monkeypatch, tmp_path):
    """Real artifact validation/publication, with only GPU and storage mocked."""
    monkeypatch.setenv("NPA_CUROBO_WORK_DIR", str(tmp_path))
    manifest = PlanManifest(problems=[{
        "id": "case",
        "start": [0, 0, 0, 0, 0, 0, 0],
        "goal_pose": {"position_xyz": [0.5, 0, 0.3], "quaternion_wxyz": [1, 0, 0, 0]},
    }]).model_dump(mode="json")
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
        (output / "result.json").write_bytes(canonical({
            "schema_version": "npa.curobo.result.v1",
            "engine": "nvidia-curobo-v2",
            "source_revision": SOURCE_REVISION,
            "run_id": "unit-run",
            "kind": "plan",
            "requested_modes": ["kinematic"],
            "summary": summarize([plan_row()]),
        }))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime, "read_bytes_uri", read)
    monkeypatch.setattr(runtime, "write_bytes_uri", write)
    monkeypatch.setattr(runtime.subprocess, "run", run)
    return SimpleNamespace(root=tmp_path, objects=objects, events=events, calls=calls)


def test_success_cleans_only_owned_directory_after_both_readbacks(completed_plan, monkeypatch):
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
def test_partial_publication_preserves_completed_journal(completed_plan, monkeypatch, failure):
    state = completed_plan
    original_read, original_write = runtime.read_bytes_uri, runtime.write_bytes_uri

    def read(uri):
        data = original_read(uri)
        return b"altered remote bytes" if failure == "readback" and uri.endswith("result.json") else data

    def write(uri, data):
        if failure == "write" and uri.endswith("result.json"):
            raise OSError("simulated object-store write failure")
        original_write(uri, data)

    monkeypatch.setattr(runtime, "read_bytes_uri", read)
    monkeypatch.setattr(runtime, "write_bytes_uri", write)
    monkeypatch.setattr(runtime.shutil, "rmtree", lambda *_: pytest.fail("cleaned failed publication"))
    with pytest.raises((OSError, CuroboError)):
        runtime.plan(request())
    directories = list(state.root.iterdir())
    assert len(directories) == 1
    assert directories[0].stat().st_mode & 0o777 == 0o700
    assert (directories[0] / "output/problems.jsonl").read_bytes() == canonical(plan_row()) + b"\n"
    assert (directories[0] / "runtime.log").read_bytes() == b"completed solver fixture\n"
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
    assert caplog.messages == ["cuRobo artifacts verified; local working-file cleanup failed"]
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
    assert runtime.validate(request())["valid"] is True
    objects["s3://example-bucket/input/problems.jsonl"] += b" "
    with pytest.raises(CuroboError, match="hash mismatch"):
        runtime.validate(request())


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
        "joints/0/position",
    ):
        assert entity in printed


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
        rows = [
            {**plan_row(), "problem_id": name}
            for name in ("first", "second")
        ]
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
