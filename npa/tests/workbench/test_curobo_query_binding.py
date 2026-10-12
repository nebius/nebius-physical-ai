"""Prove request provenance and actionable failures across real operation surfaces."""

from copy import deepcopy
import hashlib
import json

from fastapi.testclient import TestClient
import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.main import app
from npa.workbench.curobo import audit, query_binding, runtime
from npa.workbench.curobo.artifacts import CuroboError, canonical, summarize
from npa.workbench.curobo.schemas import DATASET_REVISION, PlanManifest, Pose
from npa.workbench.curobo.service import create_app
import test_curobo
from test_curobo import request
from test_curobo_audit import plan_bytes

completed_plan = test_curobo.completed_plan


def _invoke(surface, operation, run_request):
    if surface == "cli":
        result = CliRunner().invoke(
            app,
            [
                "workbench",
                "curobo",
                operation,
                "--input-path",
                run_request.input_path,
                "--output-path",
                run_request.output_path,
                "--run-id",
                run_request.run_id,
            ],
        )
        return result.exit_code, result.output
    client = TestClient(
        create_app(token="unit-token", allowed_s3_roots=["s3://example-bucket"])
    )
    with client:
        response = client.post(
            "/" + operation,
            json=run_request.model_dump(),
            headers={"Authorization": "Bearer unit-token"},
        )
    return response.status_code, response.text


@pytest.mark.parametrize("surface,expected", [("cli", 0), ("service", 200)])
def test_accepted_quaternion_completes_through_real_operation(
    completed_plan, surface, expected
):
    state = completed_plan
    manifest = json.loads(state.objects[request().input_path])
    manifest["problems"][0]["goal_pose"]["quaternion_wxyz"] = [1.00002, 0, 0, 0]
    state.objects[request().input_path] = canonical(manifest)
    status, text = _invoke(surface, "plan", request())
    assert status == expected, text
    result = json.loads(state.objects[request().output_path + "/result.json"])
    query = result["input_manifest"]["problems"][0]
    assert query["goal_pose"]["quaternion_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert len(state.calls) == 1


@pytest.mark.parametrize("surface,expected", [("cli", 1), ("service", 400)])
@pytest.mark.parametrize("substitution", ["goal", "scene"])
def test_operation_rejects_substituted_query_before_publication(
    completed_plan, surface, expected, substitution
):
    state = completed_plan
    manifest = json.loads(state.objects[request().input_path])
    problem = manifest["problems"][0]
    if substitution == "goal":
        problem["goal_pose"]["position_xyz"] = [0.5, 0.0, 0.3]
    else:
        problem["cuboids"] = {"box": {"dims": [1, 1, 1], "pose": [0, 0, 0, 1, 0, 0, 0]}}
    state.objects[request().input_path] = canonical(manifest)
    status, text = _invoke(surface, "plan", request())
    assert status == expected, text
    assert "executed query differs" in text
    assert len(state.calls) == 1
    assert not any(event[0] == "write" for event in state.events)


@pytest.mark.parametrize("surface,expected", [("cli", 1), ("service", 400)])
def test_audit_only_invalid_artifact_has_actionable_public_error(
    monkeypatch, tmp_path, surface, expected
):
    result_bytes, journal = plan_bytes()
    report, row = json.loads(result_bytes), json.loads(journal)
    row["metrics"]["joint_path_length_rad"] = 0.1
    journal = canonical(row) + b"\n"
    report["summary"] = summarize([row])
    report["journal_sha256"] = hashlib.sha256(journal).hexdigest()
    objects = {"problems.jsonl": journal, "result.json": canonical(report)}
    monkeypatch.setenv("NPA_CUROBO_WORK_DIR", str(tmp_path))
    monkeypatch.setattr(
        runtime, "read_bytes_uri", lambda uri: objects[uri.rsplit("/", 1)[1]]
    )
    monkeypatch.setattr(
        runtime, "replay_rows", lambda *_a: pytest.fail("native replay reached")
    )
    monkeypatch.setattr(
        runtime, "_publish", lambda *_a: pytest.fail("invalid result published")
    )
    run_request = request().model_copy(update={"run_id": "audit-run"})
    status, text = _invoke(surface, "validate", run_request)
    assert status == expected, text
    assert "independent artifact audit failed" in text
    assert "joint path length" in text
    with pytest.raises(CuroboError) as error:
        runtime.validate(run_request)
    assert isinstance(error.value.__cause__, audit.AuditError)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "mutation", ["goal", "scene", "request_hash", "missing_request"]
)
def test_independent_audit_rejects_rehashed_query_substitution(mutation):
    result_bytes, journal = plan_bytes()
    report, row = json.loads(result_bytes), json.loads(journal)
    if mutation == "goal":
        # Change both retained goal and FK endpoint: the geometry remains coherent.
        row["query"]["goal_pose"]["position_xyz"][0] = 0.2
        row["trajectory"]["tool_position"][-1][0] = 0.2
        row["metrics"]["tool_path_length_m"] = 0.2
    elif mutation == "scene":
        row["query"]["scene"]["cuboid"]["box"] = {
            "dims": [1, 1, 1],
            "pose": [0, 0, 0, 1, 0, 0, 0],
        }
    elif mutation == "request_hash":
        report["input_sha256"] = "0" * 64
    else:
        del report["input_manifest"]
    journal = canonical(row) + b"\n"
    report["journal_sha256"] = hashlib.sha256(journal).hexdigest()
    report["summary"] = summarize([row])
    with pytest.raises(audit.AuditError, match="binding"):
        audit.audit_bytes(canonical(report), journal, run_id="audit-run")


@pytest.fixture
def pinned_dataset(monkeypatch, tmp_path):
    root = tmp_path / "dataset"
    directory = root / "robometrics/content/dataset"
    directory.mkdir(parents=True)
    (root / "NPA_SOURCE_REVISION").write_text(DATASET_REVISION)
    problems = [
        {
            "start": [0.0] * 7,
            "goal_pose": {
                "position_xyz": [float(i), 0.0, 0.0],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "obstacles": {"cuboid": {}},
            "collision_buffer_ik": 0.0,
        }
        for i in range(2)
    ]
    payload = yaml.safe_dump({"group": problems}).encode()
    path = directory / "cases.yaml"
    path.write_bytes(payload)
    monkeypatch.setenv("NPA_CUROBO_DATASET_SOURCE", str(root))
    monkeypatch.setattr(
        query_binding,
        "DATASET_FILES",
        {"test": ("cases.yaml", hashlib.sha256(payload).hexdigest())},
    )
    monkeypatch.setattr(query_binding, "BENCHMARK_GROUPS", {"test": {"group": (2, ())}})
    return path


def test_pinned_dataset_reader_binds_content_not_just_labels(pinned_dataset):
    expected = query_binding.benchmark_queries()
    assert expected[("test", "group/0")] != expected[("test", "group/1")]
    manifest = {
        "schema_version": "npa.curobo.benchmark.v1",
        "modes": ["kinematic"],
        "dataset_revision": DATASET_REVISION,
    }
    report = {
        "kind": "benchmark",
        "input_manifest": manifest,
        "input_sha256": hashlib.sha256(canonical(manifest)).hexdigest(),
    }
    rows = [
        {
            "mode": "kinematic",
            "dataset": dataset,
            "problem_id": identity,
            "status": "failed",
            "query": deepcopy(query),
        }
        for (dataset, identity), query in expected.items()
    ]
    query_binding.validate_query_binding(report, rows)
    rows[1]["query"] = deepcopy(rows[0]["query"])
    with pytest.raises(query_binding.QueryBindingError, match="executed query differs"):
        query_binding.validate_query_binding(report, rows)


@pytest.mark.parametrize("mutation", ["bytes", "missing"])
def test_pinned_dataset_reader_fails_closed(pinned_dataset, mutation):
    if mutation == "bytes":
        pinned_dataset.write_bytes(pinned_dataset.read_bytes() + b"\n")
    else:
        pinned_dataset.unlink()
    with pytest.raises(query_binding.QueryBindingError, match="unavailable"):
        query_binding.benchmark_queries()


def test_quaternion_canonicalization_is_idempotent_for_goal_and_scene():
    pose = Pose(position_xyz=[0, 0, 0], quaternion_wxyz=[1.00002, 0, 0, 0])
    manifest = PlanManifest(
        problems=[
            {
                "id": "pose",
                "start": [0] * 7,
                "goal_pose": pose,
                "cuboids": {
                    "box": {"dims": [1, 1, 1], "pose": [0, 0, 0, 1.00002, 0, 0, 0]}
                },
            }
        ]
    )
    data = manifest.model_dump(mode="json")
    assert data == PlanManifest.model_validate(data).model_dump(mode="json")
    assert data["problems"][0]["cuboids"]["box"]["pose"][3:] == [1.0, 0.0, 0.0, 0.0]
