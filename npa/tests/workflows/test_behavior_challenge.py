"""Exercise challenge splits, original artifact preservation, and single-attempt execution."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import io
import json
import subprocess
import zipfile
from pathlib import Path

import av
import numpy as np
import pytest

from npa.clients.storage import StoragePreconditionFailed
from npa.workflows.behavior_challenge import artifacts, execution, protocol


@pytest.fixture
def upstream(tmp_path):
    root = tmp_path / "upstream"
    directory = root / "docs/challenge"
    directory.mkdir(parents=True)
    tasks = ["turning_on_radio"] + [f"fixture_task_{i}" for i in range(99)]
    (directory / "task_data.json").write_text(
        json.dumps({"tasks": [{"id": t} for t in tasks]})
    )
    evaluator = root / protocol.EVAL_DIRECTORY
    (evaluator / "wrappers").mkdir(parents=True)
    (evaluator / "wrappers/rgbd_full_res_wrapper.py").write_text(
        "# synthetic wrapper fixture\n"
    )
    (evaluator / "r1pro.yaml").write_text("model: r1pro\n")
    (root / "OmniGibson/LICENSE").write_text("Synthetic license fixture\n")
    return root


@pytest.fixture
def recipe():
    return {
        "schema": "npa.behavior.recipe.v1",
        "split": "report",
        "tasks": ["turning_on_radio"],
        "policy_checkpoint_sha256": "a" * 64,
    }


def _write_rollout(output, case, score=0.25):
    stem = f"{case['task']}_{case['instance_id']}_0"
    metrics = {k: case[k] for k in ("task", "instance_id", "rollout_id")}
    metrics.update(
        {
            "steps": 2,
            "success": False,
            "q_score": {"final": score},
            "agent_distance": {k: 1.0 for k in ("base", "left", "right")},
            "normalized_agent_distance": {k: 1.0 for k in ("base", "left", "right")},
            "time": {
                "simulator_steps": 2,
                "simulator_time": 2 / 30,
                "normalized_time": 10.0,
            },
        }
    )
    (output / "json").mkdir(exist_ok=True)
    (output / "videos").mkdir(exist_ok=True)
    metrics_path = output / f"json/{stem}.json"
    metrics_path.write_text(json.dumps(metrics, indent=3) + "\n")
    with av.open(str(output / f"videos/{stem}.mp4"), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for value in (0, 128):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), value, dtype=np.uint8), format="rgb24"
            )
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return metrics_path


def test_official_report_indices_are_not_output_instance_ids(upstream, recipe):
    plan = protocol.make_plan(recipe, upstream)
    assert [case["index"] for case in plan["cases"]] == list(range(10))
    assert [case["instance_id"] for case in plan["cases"]] == list(range(301, 311))
    assert all(case["rollout_id"] == 0 for case in plan["cases"])
    assert plan["challenge_denominator"] == 1000
    recipe["split"] = "development"
    development = protocol.make_plan(recipe, upstream)
    assert [case["index"] for case in development["cases"]] == list(range(10, 20))
    assert not development["eligible_for_reporting"]


def test_all_tasks_freezes_exactly_1000_cases(upstream, recipe):
    recipe["tasks"] = "all"
    tasks = json.loads((upstream / "docs/challenge/task_data.json").read_text())[
        "tasks"
    ]
    recipe["policy_ports"] = {task["id"]: 8000 + i for i, task in enumerate(tasks)}
    plan = protocol.make_plan(recipe, upstream)
    assert len(plan["cases"]) == 1000
    assert len({case["task"] for case in plan["cases"]}) == 100


def test_multiple_tasks_require_explicit_task_configured_endpoints(
    upstream, recipe, tmp_path
):
    recipe["tasks"] = ["turning_on_radio", "fixture_task_0"]
    with pytest.raises(ValueError, match="explicit policy port"):
        protocol.make_plan(recipe, upstream)
    recipe["policy_ports"] = {"turning_on_radio": 8000, "fixture_task_0": 8001}
    plan = protocol.make_plan(recipe, upstream)
    argv = protocol.evaluator_argv(
        plan["cases"][-1],
        root=upstream,
        python="python",
        host="localhost",
        port=9000,
        output=tmp_path,
    )
    assert argv[argv.index("--port") + 1] == "8001"
    recipe["policy_ports"]["fixture_task_0"] = 8000
    with pytest.raises(ValueError, match="distinct"):
        protocol.make_plan(recipe, upstream)


@pytest.mark.parametrize(
    "ports",
    [[], {"unknown": 8000}, {"turning_on_radio": True}, {"turning_on_radio": 70000}],
)
def test_invalid_policy_ports_fail(upstream, recipe, ports):
    recipe["policy_ports"] = ports
    with pytest.raises(ValueError):
        protocol.make_plan(recipe, upstream)


@pytest.mark.parametrize(
    "change",
    [
        {"split": "hidden_test"},
        {"split": "train"},
        {"split": []},
        {"tasks": []},
        {"tasks": ["turning_on_radio"] * 2},
        {"tasks": ["unknown"]},
        {"tasks": "turning_on_radio"},
        {"max_steps": 10},
        {"num_rollouts": 3},
        {"policy_checkpoint_sha256": "latest"},
        {"schema": "2025"},
    ],
)
def test_recipe_rejects_protocol_changes(upstream, recipe, change):
    recipe.update(change)
    with pytest.raises(ValueError):
        protocol.make_plan(recipe, upstream)


def test_evaluator_preserves_wrapper_video_and_default_timeout(
    upstream, recipe, tmp_path
):
    case = protocol.make_plan(recipe, upstream)["cases"][0]
    argv = protocol.evaluator_argv(
        case,
        root=upstream,
        python="/runtime/python",
        host="localhost",
        port=8000,
        output=tmp_path,
    )
    assert argv[:3] == ["/runtime/python", "-m", "omnigibson.eval.eval"]
    assert argv[argv.index("--instance-indices") + 1] == "0"
    assert argv[argv.index("--env-wrapper") + 1] == protocol.WRAPPER
    assert argv[argv.index("--num-rollouts") + 1] == "1"
    assert argv[argv.index("--mode") + 1] == "public_test"
    assert "--write-video" in argv
    assert argv[argv.index("--num-envs") + 1] == "1"
    assert argv[argv.index("--replay-action-chunk-size") + 1] == "0"
    assert "--max-steps" not in argv and "--policy" not in argv


def test_legacy_evaluator_argv_remains_byte_compatible(upstream, recipe, tmp_path):
    legacy = protocol.UPSTREAM_COMMITS["3.9.2"]
    recipe["upstream_commit"] = legacy
    case = protocol.make_plan(recipe, upstream)["cases"][0]
    argv = protocol.evaluator_argv(
        case,
        root=upstream,
        python="python",
        host="localhost",
        port=8000,
        output=tmp_path,
        upstream_commit=legacy,
    )
    assert "--num-envs" not in argv
    assert "--replay-action-chunk-size" not in argv
    assert argv[-2:] == ["--write-video", "--headless"]


def test_recipe_pins_supported_evaluator_revision(upstream, recipe):
    legacy = protocol.UPSTREAM_COMMITS["3.9.2"]
    recipe["upstream_commit"] = legacy
    assert protocol.make_plan(recipe, upstream)["upstream_commit"] == legacy
    recipe["upstream_commit"] = "0" * 40
    with pytest.raises(ValueError, match="supported official"):
        protocol.make_plan(recipe, upstream)


@pytest.mark.parametrize(
    "revision,dirty", [("wrong", ""), (protocol.UPSTREAM_COMMIT, " M eval.py")]
)
def test_source_verification_refuses_drift(monkeypatch, tmp_path, revision, dirty):
    monkeypatch.setattr(
        protocol.subprocess,
        "check_output",
        lambda argv, **kw: revision if "rev-parse" in argv else dirty,
    )
    with pytest.raises(ValueError):
        protocol.verify_upstream(tmp_path)


def test_source_verification_uses_explicit_legacy_pin(monkeypatch, tmp_path):
    legacy = protocol.UPSTREAM_COMMITS["3.9.2"]
    monkeypatch.setattr(
        protocol.subprocess,
        "check_output",
        lambda argv, **kw: legacy if "rev-parse" in argv else "",
    )
    protocol.verify_upstream(tmp_path, legacy)
    with pytest.raises(ValueError, match="pinned commit"):
        protocol.verify_upstream(tmp_path, protocol.UPSTREAM_COMMIT)


def test_original_failed_rollout_and_infinite_normalization_are_preserved(tmp_path):
    case = {"task": "turning_on_radio", "index": 0, "instance_id": 301, "rollout_id": 0}
    path = _write_rollout(tmp_path, case, score=0)
    metrics = json.loads(path.read_text())
    metrics["agent_distance"]["base"] = 0
    metrics["normalized_agent_distance"]["base"] = float("inf")
    path.write_text(json.dumps(metrics))
    original = path.read_bytes()
    record = artifacts.inspect_rollout(tmp_path, case)
    assert record["q_score"] == 0 and record["video_frames"] == 2
    assert path.read_bytes() == original


@pytest.mark.parametrize("score", [-1, 1.1, float("nan"), float("inf"), True])
def test_invalid_primary_scores_fail(tmp_path, score):
    case = {"task": "turning_on_radio", "index": 0, "instance_id": 301, "rollout_id": 0}
    _write_rollout(tmp_path, case, score=score)
    with pytest.raises(ValueError, match="q_score"):
        artifacts.inspect_rollout(tmp_path, case)


def test_mismatched_case_and_missing_or_corrupt_video_fail(tmp_path):
    case = {"task": "turning_on_radio", "index": 0, "instance_id": 301, "rollout_id": 0}
    path = _write_rollout(tmp_path, case)
    metrics = json.loads(path.read_text())
    metrics["instance_id"] = 0
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="instance_id"):
        artifacts.inspect_rollout(tmp_path, case)
    _write_rollout(tmp_path, case)
    video = next((tmp_path / "videos").iterdir())
    video.write_bytes(b"invalid video fixture")
    with pytest.raises(av.FFmpegError):
        artifacts.inspect_rollout(tmp_path, case)
    video.unlink()
    with pytest.raises((OSError, av.FFmpegError)):
        artifacts.inspect_rollout(tmp_path, case)


def test_summary_counts_missing_cases_as_zero_and_rejects_duplicates(
    upstream, recipe, tmp_path
):
    plan = protocol.make_plan(recipe, upstream)
    _write_rollout(tmp_path, plan["cases"][0])
    record = artifacts.inspect_rollout(tmp_path, plan["cases"][0])
    summary = artifacts.write_summary(tmp_path, plan, [record])
    assert summary["challenge_score"] == 0.25 / 1000
    assert summary["missing_challenge_rollouts"] == 999 and not summary["complete"]
    with pytest.raises(ValueError, match="Duplicate"):
        artifacts.write_summary(tmp_path, plan, [record, record])
    plan["eligible_for_reporting"] = False
    assert artifacts.write_summary(tmp_path, plan, [record])["challenge_score"] is None


class _Body(io.BytesIO):
    def iter_chunks(self, chunk_size):
        while chunk := self.read(chunk_size):
            yield chunk


class _Storage:
    def __init__(self, recipe):
        self.objects = {
            "input/recipe.json": json.dumps(recipe).encode(),
            "input/policy.md": b"Synthetic policy serving instructions",
        }
        self.s3 = self

    def download_file(self, uri, target):
        Path(target).write_bytes(self.objects[execution._s3_location(uri)[1]])

    def upload_file(self, path, uri):
        self.objects[execution._s3_location(uri)[1]] = Path(path).read_bytes()

    def put_bytes_conditional(self, payload, uri, **kwargs):
        assert kwargs["if_none_match"]
        key = execution._s3_location(uri)[1]
        if key in self.objects:
            raise StoragePreconditionFailed("already claimed")
        self.objects[key] = payload

    def get_object(self, *, Bucket, Key):
        return {"Body": _Body(self.objects[Key])}


@pytest.fixture
def execution_fixture(upstream, recipe, monkeypatch):
    storage = _Storage(recipe)
    monkeypatch.setattr(execution, "verify_upstream", lambda root, commit=None: None)
    monkeypatch.setattr(execution, "_runtime_environment", lambda args: {})
    monkeypatch.setattr(execution.StorageClient, "from_environment", lambda: storage)
    args = argparse.Namespace(
        input_path="s3://fixture/input/recipe.json",
        output_path="s3://fixture/output/",
        policy_readme_uri="s3://fixture/input/policy.md",
        upstream_root=upstream,
        evaluator_python="/runtime/python",
        host="localhost",
        port=8000,
    )
    calls = []

    def run(command, **kwargs):
        index = int(command[command.index("--instance-indices") + 1])
        calls.append(index)
        output = Path(command[command.index("--output-dir") + 1])
        case = {
            "task": "turning_on_radio",
            "index": index,
            "instance_id": 301 + index,
            "rollout_id": 0,
        }
        _write_rollout(output, case)

    monkeypatch.setattr(execution.subprocess, "run", run)
    return args, storage, calls


def test_prescribed_run_preserves_zip_bytes_and_refuses_reexecution(execution_fixture):
    args, storage, calls = execution_fixture
    summary = execution.evaluate(args)
    assert summary["complete"] and summary["completed"] == 10
    assert calls == list(range(10))
    with zipfile.ZipFile(
        io.BytesIO(storage.objects["output/submission.zip"])
    ) as archive:
        for name in archive.namelist():
            assert archive.read(name) == storage.objects[f"output/{name}"]
        assert "evaluator/r1pro.yaml" in archive.namelist()
        assert "policy.md" in archive.namelist()
        assert not any(name.endswith(".mp4") for name in archive.namelist())
    with pytest.raises(StoragePreconditionFailed):
        execution.evaluate(args)
    assert calls == list(range(10))


def test_submission_delivers_policy_launcher_license(execution_fixture, monkeypatch):
    args, storage, _calls = execution_fixture
    adapter = b"# Synthetic policy launcher fixture\n"

    @contextmanager
    def managed_policy(_args, _plan, output):
        (output / "policy-server.py").write_bytes(adapter)
        (output / "policy-provenance.json").write_text("{}\n")
        yield

    monkeypatch.setattr(execution, "managed_policy", managed_policy)
    execution.evaluate(args)
    license_bytes = (Path(__file__).resolve().parents[3] / "LICENSE").read_bytes()
    with zipfile.ZipFile(
        io.BytesIO(storage.objects["output/submission.zip"])
    ) as archive:
        assert archive.read("policy-server.py") == adapter
        assert archive.read("policy-server.LICENSE") == license_bytes
        assert archive.read("evaluator/LICENSE") == b"Synthetic license fixture\n"
    assert storage.objects["output/policy-server.LICENSE"] == license_bytes


def test_failed_evaluator_preserves_attempt_and_partial_outputs(
    execution_fixture, monkeypatch
):
    args, storage, calls = execution_fixture
    real = execution.subprocess.run

    def fail_second(command, **kwargs):
        if calls:
            raise subprocess.CalledProcessError(1, command)
        real(command, **kwargs)

    monkeypatch.setattr(execution.subprocess, "run", fail_second)
    with pytest.raises(subprocess.CalledProcessError):
        execution.evaluate(args)
    summary = json.loads(storage.objects["output/summary.json"])
    assert summary["completed"] == 1 and not summary["complete"]
    assert len(json.loads(storage.objects["output/attempts.json"])) == 2
    assert "output/json/turning_on_radio_301_0.json" in storage.objects
    assert "output/submission.zip" not in storage.objects


def test_storage_readback_detects_changed_bytes(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"original")
    storage = _Storage({})
    storage.get_object = lambda **kwargs: {"Body": _Body(b"changed")}
    with pytest.raises(ValueError, match="readback"):
        execution._upload_verified(storage, path, "s3://fixture/output/artifact")


def test_active_policy_log_upload_uses_a_stable_snapshot(tmp_path):
    log = tmp_path / "policy.log"
    log.write_bytes(b"ready\n")
    storage = _Storage({})
    upload = storage.upload_file

    def append_during_upload(path, uri):
        with log.open("ab") as stream:
            stream.write(b"new event\n")
        upload(path, uri)

    storage.upload_file = append_during_upload
    published = {}
    execution._publish(storage, tmp_path, "s3://fixture/output/", published)
    assert storage.objects["output/policy.log"] == b"ready\n"
    execution._publish(storage, tmp_path, "s3://fixture/output/", published)
    assert storage.objects["output/policy.log"] == b"ready\nnew event\n"


def test_submission_rejects_post_validation_changes(upstream, recipe, tmp_path):
    plan = protocol.make_plan(recipe, upstream)
    records = []
    for case in plan["cases"]:
        _write_rollout(tmp_path, case)
        records.append(artifacts.inspect_rollout(tmp_path, case))
    next((tmp_path / "json").iterdir()).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        artifacts.build_submission(tmp_path, plan, records)


def test_development_run_never_creates_submission(execution_fixture):
    args, storage, calls = execution_fixture
    recipe = json.loads(storage.objects["input/recipe.json"])
    recipe["split"] = "development"
    storage.objects["input/recipe.json"] = json.dumps(recipe).encode()
    summary = execution.evaluate(args)
    assert calls == list(range(10, 20))
    assert summary["challenge_score"] is None
    assert "output/submission.zip" not in storage.objects


def test_policy_startup_failure_publishes_diagnostics(execution_fixture, monkeypatch):
    from contextlib import contextmanager

    args, storage, calls = execution_fixture

    @contextmanager
    def failed_policy(args, plan, output):
        (output / "policy.log").write_text("Policy startup failed\n")
        raise RuntimeError("startup failed")
        yield

    monkeypatch.setattr(execution, "managed_policy", failed_policy)
    with pytest.raises(RuntimeError, match="startup failed"):
        execution.evaluate(args)
    assert calls == []
    assert storage.objects["output/policy.log"] == b"Policy startup failed\n"
    assert json.loads(storage.objects["output/summary.json"])["completed"] == 0


def test_missing_licensed_assets_fail_before_simulator_import(tmp_path, monkeypatch):
    args = argparse.Namespace(data_root=str(tmp_path))
    monkeypatch.setattr(
        execution.subprocess,
        "check_output",
        lambda *args, **kwargs: pytest.fail("must not import simulator"),
    )
    with pytest.raises(ValueError, match="authorized BEHAVIOR assets"):
        execution._runtime_environment(args)


@pytest.mark.parametrize(
    "vulkan_output,accepted",
    [
        ("GPU0:\n vendorID = 0x10de\n deviceName = NVIDIA RTX\n", True),
        ("GPU0:\n vendorID = 0x10005\n deviceName = llvmpipe\n", False),
        ("NVIDIA loader available, but no physical devices found\n", False),
    ],
)
def test_graphics_requires_an_enumerated_nvidia_device(
    monkeypatch, vulkan_output, accepted
):
    def check(argv, **kwargs):
        assert kwargs["env"] == {"PATH": "/runtime/bin"}
        return vulkan_output if argv[0] == "vulkaninfo" else b""

    monkeypatch.setattr(execution.subprocess, "check_output", check)
    if accepted:
        execution._verify_graphics("/runtime/python", {"PATH": "/runtime/bin"})
    else:
        with pytest.raises(RuntimeError, match="NVIDIA physical device"):
            execution._verify_graphics("/runtime/python", {"PATH": "/runtime/bin"})


def test_missing_graphics_libraries_fail_even_when_cuda_is_available(monkeypatch):
    def check(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, output=b"libGLX_nvidia missing")

    monkeypatch.setattr(execution.subprocess, "check_output", check)
    with pytest.raises(RuntimeError, match="graphics-qualified"):
        execution._verify_graphics("/runtime/python", {})


def test_render_preserves_operator_image_and_readonly_asset_mount():
    import yaml

    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        render_skypilot_yaml,
    )
    from npa.orchestration.npa_workflow.spec import load_spec

    root = Path(__file__).resolve().parents[3]
    spec = load_spec(root / "workflows/testing/behavior-challenge-eval.yaml")
    plan = build_plan(spec, run_id="fixture-render")
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="fixture-render",
        options=SkypilotRenderOptions(materialize_registry_secrets=False),
    )
    task = list(yaml.safe_load_all(rendered))[-1]
    assert task["resources"]["accelerators"] == "L40S:1"
    assert task["resources"]["image_id"] == "docker:" + spec.config["runtime_image"]
    pod = task["config"]["kubernetes"]["pod_config"]["spec"]
    assert pod["runtimeClassName"] == "nvidia"
    assert {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "all"} in pod["containers"][
        0
    ]["env"]
    assert pod["volumes"][0]["persistentVolumeClaim"]["claimName"] == "behavior-assets"
    mount = pod["containers"][0]["volumeMounts"][0]
    assert mount["readOnly"] and mount["mountPath"] == "/data/behavior"
    assert "npa.workflows.behavior_challenge evaluate" in task["run"]
    assert "--max-steps" not in task["run"]
