from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pytest

from npa.workflows.behavior_challenge import (
    campaign,
    campaign_runner,
    campaign_workflow,
    native_comet_server,
    nonreporting_train,
    policy,
    released_comet_checkpoint,
    released_comet_policy,
    serving_identity,
)
from npa.workflows.behavior_challenge.comet_policy import (
    COMET12_PROFILE,
    MODEL_REPOSITORY,
    SOURCE_COMMIT,
)
from npa.workflows.behavior_challenge.native_training_checkpoint import file_identity


def _released_fixture(tmp_path: Path, monkeypatch):
    root = tmp_path / "inputs"
    root.mkdir()
    evidence = {}
    for role in sorted(released_comet_checkpoint._ROLES):
        path = root / f"{role}.json"
        path.write_text(role + "\n")
        evidence[role] = {"path": path.name, "identity": file_identity(path)}
    archive = tmp_path / "checkpoint.zip"
    archive.write_bytes(b"released-params")
    checkpoint = tmp_path / COMET12_PROFILE.checkpoint
    checkpoint.mkdir()
    serving = {"sha256": "a" * 64, "bytes": 111}
    protocol = nonreporting_train.declare_train_protocol(
        "picking_up_trash",
        {
            "schema": "npa.behavior.nonreporting-train-task-mapping.v1",
            "split": "train",
            "task": "picking_up_trash",
            "data_namespace": "dataset",
            "data_task_id": 1,
            "source_manifest": evidence["task_mapping"]["identity"],
            "split_manifest": evidence["task_mapping"]["identity"],
            "mapping_artifact": evidence["task_mapping"]["identity"],
        },
        [{"instance_id": 200, "rollout_id": 0}],
        {
            "behavior_upstream_commit": "1" * 40,
            "task_registry": evidence["task_mapping"]["identity"],
            "dataset": evidence["task_mapping"]["identity"],
            "dataset_view": evidence["task_mapping"]["identity"],
            "normalization": evidence["normalization"]["identity"],
            "tokenizer": evidence["tokenizer"]["identity"],
            "action_semantics": evidence["task_mapping"]["identity"],
        },
        {
            "argv_contract": evidence["train_evaluator_argv"]["identity"],
            "evaluator_source": evidence["evaluator_source"]["identity"],
            "controller_source": evidence["controller_source"]["identity"],
            "robot_config": evidence["robot_config"]["identity"],
            "rng_contract": evidence["rng_contract"]["identity"],
            "wrapper": "omnigibson.eval.wrappers.RGBDFullResWrapper",
            "mode": "train",
            "num_envs": 1,
            "num_rollouts": 1,
            "write_video": True,
            "max_steps_argument": None,
            "model_prediction_horizon": 32,
            "executed_prefix": 32,
            "fresh_policy_process_per_case": True,
            "qualification_process_discarded": True,
        },
    )
    policy = campaign.freeze_policy_identity(
        "released-test-policy",
        {"checkpoint": file_identity(archive), "serving": serving},
    )
    panel = nonreporting_train.declare_train_panel(protocol, policy)
    policy_identity = panel["policy_binding_sha256"]
    panel_id = panel["panel_id"]
    inventory = Path(released_comet_checkpoint.__file__).with_name(
        COMET12_PROFILE.inventory
    )
    binding = {
        "schema": released_comet_checkpoint.BINDING_SCHEMA,
        "profile": "comet12",
        "source_commit": SOURCE_COMMIT,
        "task": "picking_up_trash",
        "task_id": 1,
        "panel_id": panel_id,
        "policy_binding_sha256": policy_identity,
        "checkpoint": {
            "archive": file_identity(archive),
            "inventory": file_identity(inventory),
            "name": COMET12_PROFILE.checkpoint,
            "repository": MODEL_REPOSITORY,
            "revision": COMET12_PROFILE.revision,
        },
        "evidence": evidence,
        "trace": {"enabled": False},
    }
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(json.dumps(binding))
    monkeypatch.setattr(
        released_comet_checkpoint, "load_checkpoint_manifest", lambda *_: {}
    )
    monkeypatch.setattr(
        released_comet_checkpoint,
        "verify_checkpoint_archive",
        lambda *_args, **_kwargs: {"params/value": "e" * 64},
    )
    return binding_path, root, archive, checkpoint, panel, serving


def _validate(fixture):
    binding, root, archive, checkpoint, panel, serving = fixture
    return released_comet_checkpoint.validate_released_comet_admission(
        binding,
        root,
        archive,
        checkpoint,
        panel,
        expected_serving_identity=serving,
    )


def test_released_admission_is_params_only(tmp_path, monkeypatch):
    admission = _validate(_released_fixture(tmp_path, monkeypatch))
    assert admission["status"] == "released_params_checkpoint_and_train_protocol_exact"
    assert admission["full_train_state_claimed"] is False
    assert admission["optimizer_state_claimed"] is False
    assert admission["checkpoint_file_count"] == 1


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda binding, panel: binding.update(task="wrong"), "TRAIN panel"),
        (
            lambda binding, panel: binding["checkpoint"].update(revision="wrong"),
            "authority",
        ),
        (
            lambda binding, panel: panel.update(panel_id="0" * 64),
            "canonical declaration|identity digest",
        ),
        (
            lambda binding, panel: panel["protocol"]["evaluator_contract"].update(
                rng_contract={"bytes": 1, "sha256": "1" * 64}
            ),
            "canonical declaration",
        ),
        (
            lambda binding, panel: panel["policy_binding"]["artifacts"].update(
                serving={"bytes": 2, "sha256": "2" * 64}
            ),
            "canonical declaration|identity digest",
        ),
    ],
)
def test_released_admission_rejects_substitution(tmp_path, monkeypatch, mutate, match):
    fixture = _released_fixture(tmp_path, monkeypatch)
    binding_path, *_rest, panel, _serving = fixture
    binding = json.loads(binding_path.read_text())
    changed_panel = deepcopy(panel)
    mutate(binding, changed_panel)
    binding_path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match=match):
        _validate((*fixture[:4], changed_panel, fixture[-1]))


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("task_mapping", "data_task_id", 999),
        ("evaluator_contract", "model_prediction_horizon", 64),
        ("evaluator_contract", "executed_prefix", 16),
    ],
)
def test_released_admission_rejects_canonically_rehashed_protocol_change(
    tmp_path, monkeypatch, section, field, value
):
    fixture = _released_fixture(tmp_path, monkeypatch)
    binding_path, *_rest, panel, _serving = fixture
    changed_protocol = deepcopy(panel["protocol"])
    changed_protocol[section][field] = value
    changed_protocol.pop("protocol_sha256")
    changed_protocol = nonreporting_train.declare_train_protocol(
        changed_protocol["task"],
        changed_protocol["task_mapping"],
        changed_protocol["prescribed_cases"],
        changed_protocol["science_lineage"],
        changed_protocol["evaluator_contract"],
    )
    changed_panel = nonreporting_train.declare_train_panel(
        changed_protocol, panel["policy_binding"]
    )
    binding = json.loads(binding_path.read_text())
    binding["panel_id"] = changed_panel["panel_id"]
    binding_path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="TRAIN panel"):
        _validate((*fixture[:4], changed_panel, fixture[-1]))


def test_released_admission_rejects_changed_evidence(tmp_path, monkeypatch):
    fixture = _released_fixture(tmp_path, monkeypatch)
    (fixture[1] / "tokenizer.json").write_text("changed\n")
    with pytest.raises(ValueError, match="tokenizer evidence"):
        _validate(fixture)


def test_released_admission_rejects_evidence_symlink(tmp_path, monkeypatch):
    fixture = _released_fixture(tmp_path, monkeypatch)
    target = fixture[1] / "tokenizer.json"
    moved = fixture[1] / "tokenizer-real.json"
    target.rename(moved)
    target.symlink_to(moved.name)
    with pytest.raises(ValueError, match="contains a symlink"):
        _validate(fixture)


def _process_args(layout="released"):
    return argparse.Namespace(
        checkpoint_layout=layout,
        case_id="case",
        task_name="picking_up_trash",
        instance_id=200,
        rollout_id=0,
        case_seed=293639851,
        checkpoint_sha256="1" * 64,
        rng_contract_sha256="2" * 64,
        trace_configuration_sha256="3" * 64,
        task_prompt_override=None,
        process_identity_sha256="unused",
        action_trace=None,
    )


def test_released_process_receipt_uses_distinct_schema():
    args = _process_args()
    initial = "4" * 64
    args.process_identity_sha256 = native_comet_server._process_identity(args, initial)
    receipt = native_comet_server._process_receipt(args, initial, initial, 0, 0)
    assert receipt["schema"] == "npa.behavior.comet-released-serving-process.v1"
    native_comet_server.validate_process_receipt(receipt)


def test_native_process_receipt_schema_stays_unchanged():
    args = _process_args("native")
    initial = "4" * 64
    args.process_identity_sha256 = native_comet_server._process_identity(args, initial)
    receipt = native_comet_server._process_receipt(args, initial, initial, 0, 0)
    assert receipt["schema"] == "npa.behavior.comet-native-serving-process.v1"
    native_comet_server.validate_process_receipt(receipt)


def test_released_server_command_has_direct_checkpoint_layout():
    args = argparse.Namespace(
        policy_python=Path("/runtime/python"),
        policy_root=Path("/source"),
        policy_checkpoint=Path("/checkpoint"),
        port=8000,
    )
    admission = {
        "task": "picking_up_trash",
        "task_id": 1,
        "checkpoint": {"archive": {"sha256": "1" * 64}},
        "evidence": {"rng_contract": {"identity": {"sha256": "2" * 64}}},
        "trace": {"enabled": False},
    }
    case = {
        "case_id": "case",
        "instance_id": 200,
        "rollout_id": 0,
    }
    command = released_comet_policy._server_command(
        args, {"upstream_commit": "upstream"}, Path("/output"), admission, case, 7
    )
    assert command[command.index("--checkpoint-layout") + 1] == "released"
    assert "--manager-step" not in command
    assert "--train-experience-root" in command


def test_campaign_scope_requires_released_recording():
    args = argparse.Namespace(
        policy_kind="comet-released",
        train_experience=False,
        train_experience_depth=False,
        policy_prompt_override=None,
    )
    with pytest.raises(ValueError, match="requires experience recording"):
        campaign_runner._validate_train_experience_scope(
            args, {"schema": "npa.behavior.nonreporting-train-panel.v1"}
        )


def test_workflow_renders_released_recording_flag():
    runtime = {
        "upstream_root": "/upstream",
        "evaluator_python": "/runtime/python",
        "data_root": "/data",
        "host": "127.0.0.1",
        "port": 8000,
        "policy_kind": "comet-released",
        "policy_root": "/source",
        "policy_python": "/policy/python",
        "policy_checkpoint": "/checkpoint",
        "policy_archive": "/checkpoint.zip",
        "policy_execution_variant": "native",
        "policy_released_binding": "/inputs/binding.json",
        "policy_released_input_root": "/inputs",
        "train_experience": True,
    }
    normalized = campaign_workflow._runtime_config(runtime)
    command = campaign_workflow._worker_argv(
        {"worker_index": 0, "workspace": "/work"}, "s3://bucket/receipt", normalized
    )
    assert command.count("--train-experience") == 1
    assert command[command.index("--policy-released-binding") + 1] == (
        "{{config.policy_released_binding}}"
    )


def test_workflow_rejects_released_without_recording():
    with pytest.raises(ValueError, match="requires TRAIN experience"):
        campaign_workflow._validate_released_policy_config(
            "comet-released",
            {
                "policy_released_binding": "/binding",
                "policy_released_input_root": "/inputs",
            },
            {},
        )


def test_released_serving_identity_includes_binding_and_new_adapters(tmp_path):
    binding = tmp_path / "binding.json"
    binding.write_text(
        json.dumps(
            {
                "schema": released_comet_checkpoint.BINDING_SCHEMA,
                "profile": "comet12",
                "source_commit": SOURCE_COMMIT,
                "task": "picking_up_trash",
                "task_id": 1,
                "panel_id": "1" * 64,
                "policy_binding_sha256": "2" * 64,
                "checkpoint": {},
                "evidence": {},
                "trace": {"enabled": False},
            }
        )
    )
    args = argparse.Namespace(
        policy_kind="comet-released",
        policy_execution_variant="native",
        policy_released_binding=binding,
        train_experience=True,
        policy_prompt_override=None,
        policy_task_name="picking_up_trash",
        policy_stock_correlation_sha256=None,
        **{field: None for field in serving_identity._INPUT_FIELDS},
    )
    artifact = serving_identity.serving_artifact(args)
    assert artifact["bytes"] > 0
    payload = serving_identity._serving_source(args)
    assert "released_comet_checkpoint.py" in payload
    assert "train_official_q.py" in payload
    value = json.loads(binding.read_text())
    value["panel_id"] = "3" * 64
    value["policy_binding_sha256"] = "4" * 64
    binding.write_text(json.dumps(value))
    assert serving_identity.serving_artifact(args) == artifact


def test_managed_policy_rejects_released_inputs_on_other_kind(tmp_path):
    args = argparse.Namespace(
        policy_kind="official",
        policy_execution_variant="native",
        train_experience=False,
        train_experience_depth=False,
        policy_task_name=None,
        policy_released_binding=tmp_path / "binding.json",
        policy_released_input_root=None,
        **{field: None for field in policy.POLICY_FIELDS},
        **{field: None for field in policy.SELECTED_RLC_FIELDS},
        **{field: None for field in policy.STOCK_RLC_CORRELATION_FIELDS},
        **{field: None for field in policy.SPECIALIST_REPORT_FIELDS},
    )
    with pytest.raises(ValueError, match="Released Comet inputs require"):
        with policy.managed_policy(args, {}, tmp_path):
            pass
