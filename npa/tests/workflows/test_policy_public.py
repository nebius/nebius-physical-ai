"""Exercise public policy lineage, gate authenticity, serving contracts and privacy failures."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.submit import spec_requires_runtime
from npa.workflows.policy_training.public_vla_data import write_json
from npa.workflows.policy_training.turnkey_server import Observation
from npa.workflows.policy_training.turnkey_serving import gang_identity
from npa.workflows.policy_training.turnkey_store import record, require_identity
from npa.workflows.policy_training.turnkey_training import decide

SPEC = (
    Path(__file__).resolve().parents[3]
    / "workflows/testing/robot-policy-train-and-serve.yaml"
)


@pytest.mark.parametrize("offset", [True, -1, 19, 41])
def test_deployment_partition_rejects_overlap_and_wrap(offset):
    from npa.workflows.policy_training.turnkey_data import _deployment_partition

    with pytest.raises(ValueError, match="disjoint"):
        _deployment_partition(
            {"deployment_state_offset": offset, "evaluation_episodes": 10}
        )


@pytest.mark.parametrize("offset", [30, 40])
def test_deployment_partition_accepts_reserved_states(offset):
    from npa.workflows.policy_training.turnkey_data import _deployment_partition

    _deployment_partition(
        {"deployment_state_offset": offset, "evaluation_episodes": 10}
    )


def test_standard_runtime_covers_all_stages_and_two_gpu_roles():
    spec = load_spec(SPEC.resolve())
    validate_spec(spec)
    plan = build_plan(spec, run_id="public-test", assume_decision="promote_checkpoint")
    assert [step.state for step in plan.steps] == [
        "prepare",
        "curate",
        "split",
        "pretrain",
        "evaluate-pretrain",
        "gate-pretrain",
        "finetune",
        "evaluate-finetune",
        "gate-finetune",
        "export",
        "serve",
        "report",
    ]
    assert spec_requires_runtime(spec)
    serving = next(step for step in plan.steps if step.state == "serve")
    assert serving.resources_profile["num_nodes"] == 2
    assert serving.resources_profile["accelerators"].endswith(":1")
    for name in ("pretrain-loop", "finetune-loop"):
        assert spec.states[name].loop.max is None
    assert all(
        step.tool_ref.startswith("workflow.policy_public.") for step in plan.steps
    )


def _evaluation(successes=7):
    return {
        "engine": "lerobot-libero-native",
        "successes": successes,
        "trials": 10,
        "success_rate": successes / 10,
        "checkpoint_sha256": "a" * 64,
        "candidate_uri": "s3://example-bucket/run/candidate/",
        "phase": "pretrain",
        "iteration": 1,
    }


def test_stage_bootstrap_uses_the_complete_overlay_and_native_interpreter(monkeypatch):
    import yaml
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        render_skypilot_yaml,
    )

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source/")
    spec = load_spec(SPEC.resolve())
    plan = build_plan(spec, run_id="public-test", assume_decision="promote_checkpoint")
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="public-test",
        options=SkypilotRenderOptions(materialize_registry_secrets=False),
    )
    tasks = [task for task in yaml.safe_load_all(rendered) if "resources" in task]
    for task, step in zip(tasks, plan.steps, strict=True):
        vendor = "lerobot" if step.resources in ("gpu", "serving") else "fiftyone"
        assert task["envs"]["NPA_BAKED_PYTHON"] == f"/opt/{vendor}/venv/bin/python"
        assert task["envs"]["NPA_SRC_OVERLAY"] == "1"


@pytest.mark.parametrize(
    "successes,expected", [(6, "loop_back"), (7, "promote_checkpoint")]
)
def test_gate_uses_measured_counts_and_preserves_checkpoint(successes, expected):
    decision = decide(
        _evaluation(successes), {"evaluation_episodes": 10, "minimum_success": 0.7}
    )
    assert decision["decision"] == expected
    assert decision["checkpoint_sha256"] == "a" * 64


@pytest.mark.parametrize(
    "change",
    [
        {"engine": "numpy-planar-reference"},
        {"successes": True},
        {"trials": 9},
        {"success_rate": float("nan")},
        {"success_rate": 1},
        {"checkpoint_sha256": "bad"},
    ],
)
def test_forged_or_incomplete_evaluation_cannot_promote(change):
    with pytest.raises(ValueError):
        decide(
            _evaluation() | change, {"evaluation_episodes": 10, "minimum_success": 0.7}
        )


def test_curated_corpus_identity_cannot_change_between_training_and_gate(tmp_path):
    recipe, corpus = {"seed": 42}, {"episodes": [{"episode_index": 1}]}
    write_json(tmp_path / "recipe.json", recipe)
    write_json(tmp_path / "corpus.json", corpus)
    record(tmp_path, "train", {"engine": "lerobot-smolvla-torchrun"})
    require_identity(tmp_path, recipe, corpus)
    with pytest.raises(ValueError, match="corpus identity"):
        require_identity(tmp_path, recipe, {"episodes": [{"episode_index": 2}]})


def _gang():
    return {
        "SKYPILOT_NODE_RANK": "0",
        "SKYPILOT_NUM_NODES": "2",
        "SKYPILOT_NUM_GPUS_PER_NODE": "1",
        "SKYPILOT_NODE_IPS": "192.0.2.1\n192.0.2.2",
    }


@pytest.mark.parametrize(
    "change",
    [
        {"SKYPILOT_NODE_RANK": "2"},
        {"SKYPILOT_NUM_NODES": "1"},
        {"SKYPILOT_NUM_GPUS_PER_NODE": "0"},
        {"SKYPILOT_NODE_IPS": "192.0.2.1\n192.0.2.1"},
    ],
)
def test_serving_refuses_an_incomplete_or_collapsed_gang(change):
    with pytest.raises(ValueError):
        gang_identity(_gang() | change)


def test_gang_assigns_distinct_policy_and_renderer_workers():
    assert gang_identity(_gang()) == (0, ["192.0.2.1", "192.0.2.2"])
    assert gang_identity(_gang() | {"SKYPILOT_NODE_RANK": "1"})[0] == 1


def test_serving_observation_refuses_boolean_step_and_wrong_state_size():
    request = {
        "episode": 0,
        "step": 0,
        "task": "place bowl",
        "state": [0.0] * 8,
        "image": "",
        "wrist": "",
    }
    Observation(**request)
    with pytest.raises(ValueError):
        Observation(**(request | {"step": True}))
    with pytest.raises(ValueError):
        Observation(**(request | {"state": [0.0] * 7}))


def test_resume_rewrites_worker_paths_without_touching_optimizer_state(tmp_path):
    from npa.workflows.policy_training.turnkey_training import _resume_checkpoint

    source = tmp_path / "original"
    model = source / "pretrained_model"
    write_json(
        model / "train_config.json",
        {
            "dataset": {},
            "policy": {},
            "scheduler": {"num_decay_steps": 100},
            "steps": 100,
        },
    )
    write_json(model / "config.json", {})
    write_json(
        model / "policy_preprocessor.json", {"tokenizer_name": "/old-worker/backbone"}
    )
    optimizer = source / "training_state/optimizer_state.safetensors"
    optimizer.parent.mkdir()
    optimizer.write_bytes(b"opaque-native-optimizer")
    worker = tmp_path / "new-worker"
    _resume_checkpoint(worker, source, {}, "generalist", {"episodes": [1, 3]}, 200)
    resumed = worker / "generalist/checkpoints/last"
    config = json.loads((resumed / "pretrained_model/train_config.json").read_text())
    assert config["steps"] == 200 and config["resume"] is True
    assert config["dataset"]["episodes"] == [1, 3]
    assert config["scheduler"]["num_decay_steps"] == 200
    assert (
        resumed / "training_state/optimizer_state.safetensors"
    ).read_bytes() == optimizer.read_bytes()


def test_public_report_omits_private_candidate_uris_and_model_paths(tmp_path):
    from npa.workflows.policy_training.turnkey_report import _public_history

    write_json(
        tmp_path / "history/pretrain-1-train.json",
        {
            "stage": "pretrain-1-train",
            "engine": "lerobot-smolvla-torchrun",
            "candidate_uri": "s3://operator-private/run/",
            "export": {"path": "/private/model"},
            "checkpoint_sha256": "a" * 64,
            "selection": {"episodes": [1], "frames": 50},
        },
    )
    public = json.dumps(_public_history(tmp_path))
    assert "operator-private" not in public and "/private/model" not in public
    assert "training_frames" in public


def test_http_session_authentication_and_monotonic_sequence(monkeypatch, tmp_path):
    import threading
    from fastapi.testclient import TestClient
    from npa.workflows.policy_training import turnkey_server as module

    component = SimpleNamespace(reset=lambda: None)
    state = {
        "policy": component,
        "pre": component,
        "post": component,
        "runtime": {},
        "lock": threading.Lock(),
        "episode": None,
        "next_step": 0,
    }
    monkeypatch.setattr(module, "_load", lambda *args: state)
    monkeypatch.setattr(module, "_batch", lambda request: {})
    monkeypatch.setattr(
        module,
        "_action",
        lambda state, batch, request: {
            "episode": request.episode,
            "step": request.step,
        },
    )
    server = SimpleNamespace(should_exit=False)
    client = TestClient(
        module.create_app(tmp_path, tmp_path, "a" * 64, "session-token", server)
    )
    assert client.get("/health").status_code == 401
    headers = {"Authorization": "Bearer session-token"}
    assert client.post("/reset/0", headers=headers).status_code == 200
    request = {
        "episode": 0,
        "step": 0,
        "task": "place bowl",
        "state": [0.0] * 8,
        "image": "",
        "wrist": "",
    }
    assert client.post("/infer", json=request, headers=headers).status_code == 200
    assert client.post("/infer", json=request, headers=headers).status_code == 409
    assert (
        client.post("/finish", params={"completed": True}, headers=headers).status_code
        == 200
    )
    assert server.should_exit is True


def test_private_diagnostics_directory_is_not_nested_or_misnamed(monkeypatch):
    from npa.workflows.policy_training import diagnostics

    written = []
    monkeypatch.setattr(
        diagnostics,
        "write_json_uri",
        lambda uri, payload: written.append((uri, payload)),
    )
    diagnostics._failure(
        "s3://example-bucket/run/train-diagnostics/",
        ValueError("private worker failure"),
    )
    assert len(written) == 1
    uri, payload = written[0]
    assert uri.startswith("s3://example-bucket/run/train-diagnostics/")
    assert uri.count("diagnostics") == 1 and uri.endswith(".json")
    assert payload["error_type"] == "ValueError"


def test_interrupted_recovery_rejects_a_changed_training_selection(
    monkeypatch, tmp_path
):
    from npa.workflows.policy_training import turnkey_runtime

    monkeypatch.setattr(
        turnkey_runtime,
        "_recovery_pointer",
        lambda uri: {
            "recipe_sha256": "a" * 64,
            "selection_sha256": "b" * 64,
        },
    )
    with pytest.raises(ValueError, match="selection changed"):
        turnkey_runtime.partial_recovery(
            "s3://example-bucket/recovery/", tmp_path, "a" * 64, "c" * 64
        )


def test_interrupted_recovery_does_not_hide_storage_authorization_failure(
    monkeypatch, tmp_path
):
    from botocore.exceptions import ClientError
    from npa.workbench.dataset import storage
    from npa.workflows.policy_training.turnkey_runtime import partial_recovery

    def denied(uri):
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")

    monkeypatch.setattr(storage, "read_json_uri", denied)
    with pytest.raises(ClientError):
        partial_recovery("s3://example-bucket/recovery/", tmp_path, "a" * 64, "b" * 64)
