"""Exercise generic native Comet checkpoint admission and trace contracts."""

from copy import deepcopy
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows.behavior_challenge import campaign
from npa.workflows.behavior_challenge import campaign_runner
from npa.workflows.behavior_challenge import native_comet_checkpoint as native
from npa.workflows.behavior_challenge import native_comet_policy
from npa.workflows.behavior_challenge import native_comet_server
from npa.workflows.behavior_challenge import nonreporting_train as train
from npa.workflows.behavior_challenge import serving_identity
from npa.workflows.behavior_challenge.native_training_checkpoint import (
    checkpoint_inventory,
    file_identity,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _provider(name: str, identity: dict) -> dict:
    return {
        "uri": f"s3://fixture-bucket/native-run/originals/{name}",
        **identity,
        "provider_readback": True,
    }


def _architecture(profile="action_expert"):
    trainable = (
        ["params/action/a"]
        if profile == "action_expert"
        else [
            "params/action/a",
            "params/backbone/b",
        ]
    )
    frozen = ["params/backbone/b"] if profile == "action_expert" else []
    optimizer = ["opt/count", "opt/mu/a"]
    partition = {
        "trainable_paths": trainable,
        "frozen_paths": frozen,
        "optimizer_paths": optimizer,
        "trainable_leaves": len(trainable),
        "frozen_leaves": len(frozen),
        "optimizer_leaves": len(optimizer),
    }
    value = {
        "schema": native.ARCHITECTURE_SCHEMA,
        "profile": profile,
        "training_dtype": "float32",
        "parameter_paths": sorted(trainable + frozen),
        "optimizer_paths": optimizer,
        "parameter_leaves": {
            path: {
                "shape": [2, 2],
                "dtype": "float32",
                "elements": 4,
                "bytes": 16,
                "sha256": chr(97 + index) * 64,
            }
            for index, path in enumerate(sorted(trainable + frozen))
        },
        "optimizer_leaves": {
            path: {
                "shape": [] if path == "opt/count" else [2, 2],
                "dtype": "int32" if path == "opt/count" else "float32",
                "elements": 1 if path == "opt/count" else 4,
                "bytes": 4 if path == "opt/count" else 16,
                "sha256": str(index + 1) * 64,
            }
            for index, path in enumerate(optimizer)
        },
        "partition": partition,
    }
    value["partition_sha256"] = native._canonical_digest(
        {
            "profile": profile,
            "training_dtype": "float32",
            "parameter_paths": value["parameter_paths"],
            "optimizer_paths": optimizer,
            "parameter_leaves": value["parameter_leaves"],
            "optimizer_leaves": value["optimizer_leaves"],
            "partition": partition,
        }
    )
    return value


def _fixture(tmp_path: Path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "1/params").mkdir(parents=True)
    (checkpoint / "1/train_state").mkdir(parents=True)
    (checkpoint / "1/params/a").write_bytes(b"params")
    (checkpoint / "1/train_state/b").write_bytes(b"optimizer")
    (checkpoint / "1/assets/behavior-1k/2025-challenge-demos").mkdir(parents=True)
    (
        checkpoint / "1/assets/behavior-1k/2025-challenge-demos/norm_stats.json"
    ).write_text("{}\n")
    inventory = checkpoint_inventory(checkpoint)
    architecture = _architecture()
    _write_json(inputs / "architecture_receipt", architecture)
    lineage = {}
    bridge_inputs = native._LINEAGE_ARTIFACTS - {
        "bridge_execution",
        "bridge_terminal",
    }
    for index, name in enumerate(sorted(bridge_inputs)):
        path = inputs / name
        path.write_text(f"{name}-{index}\n")
        lineage[name] = _provider(name, file_identity(path))
    execution = {
        "schema": native.BRIDGE_EXECUTION_SCHEMA,
        "status": "actual_full_state_restore_and_probe_completed",
        "profile": "action_expert",
        "logical_update": 2,
        "manager_step": 1,
        "cursor": {
            "logical_update": 2,
            "manager_step": 1,
            "next_manifest_position": 32,
        },
        "checkpoint_content_sha256": inventory["content_sha256"],
        "architecture_partition_sha256": architecture["partition_sha256"],
        "input_provider_rows": {name: lineage[name] for name in sorted(bridge_inputs)},
        "claims": {
            "development_or_report_read": False,
            "outcomes_read": False,
            "score_or_selection_executed": False,
        },
    }
    _write_json(inputs / "bridge_execution", execution)
    lineage["bridge_execution"] = _provider(
        "bridge_execution", file_identity(inputs / "bridge_execution")
    )
    probe = {
        "schema": native.PROBE_SCHEMA,
        "status": "actual_restore_and_discarded_finite_probe",
        "logical_update": 2,
        "manager_step": 1,
        "checkpoint_content_sha256": inventory["content_sha256"],
        "architecture_partition_sha256": architecture["partition_sha256"],
        "optimizer_train_state_restored": True,
        "serving_params_restored": True,
        "finite_loss": 1.25,
        "finite_grad_norm": 2.5,
        "update_discarded": True,
        "provider_execution": lineage["bridge_execution"],
    }
    _write_json(inputs / "restore_qualification", probe)
    prefix = "s3://fixture-bucket/native-run/checkpoint"
    member_provider_rows = [
        {
            "uri": f"{prefix}/{row['path']}",
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "provider_readback": True,
        }
        for row in inventory["files"]
    ]
    terminal = {
        "schema": native.BRIDGE_TERMINAL_SCHEMA,
        "status": "qualified_checkpoint_originals_provider_readback",
        "bridge_execution": lineage["bridge_execution"],
        "architecture_receipt": _provider(
            "architecture_receipt", file_identity(inputs / "architecture_receipt")
        ),
        "restore_qualification": _provider(
            "restore_qualification", file_identity(inputs / "restore_qualification")
        ),
        "checkpoint_content_sha256": inventory["content_sha256"],
        "member_provider_rows": member_provider_rows,
        "claims": {
            "development_or_report_read": False,
            "outcomes_read": False,
            "score_or_selection_executed": False,
        },
    }
    _write_json(inputs / "bridge_terminal", terminal)
    lineage["bridge_terminal"] = _provider(
        "bridge_terminal", file_identity(inputs / "bridge_terminal")
    )
    verified = {
        "schema": native.VERIFIED_SCHEMA,
        "status": "actual_restore_partition_and_finite_probe_verified",
        "profile": "action_expert",
        "logical_update": 2,
        "manager_step": 1,
        "cursor": {
            "logical_update": 2,
            "manager_step": 1,
            "next_manifest_position": 32,
        },
        "checkpoint_prefix": prefix,
        "checkpoint": inventory,
        "member_provider_rows": member_provider_rows,
        "serving_assets": {
            "asset_id": "behavior-1k/2025-challenge-demos",
            "norm_stats_path": "1/assets/behavior-1k/2025-challenge-demos/norm_stats.json",
        },
        "architecture": architecture,
        "frozen_parent_equality": {
            "params/backbone/b": {
                "parent_sha256": architecture["parameter_leaves"]["params/backbone/b"][
                    "sha256"
                ],
                "restored_sha256": architecture["parameter_leaves"][
                    "params/backbone/b"
                ]["sha256"],
            }
        },
        "optimizer_train_state_observation": {
            "manager_step": 1,
            "params_restored": True,
            "optimizer_state_restored": True,
            "state_step": 2,
            "training_dtype": "float32",
        },
        "serving_params_observation": {
            "manager_step": 1,
            "params_restored": True,
            "training_dtype": "float32",
        },
        "discarded_finite_probe": probe,
        "lineage": lineage,
        "claims": {
            "development_or_report_read": False,
            "outcomes_read": False,
            "score_or_selection_executed": False,
            "public_milestone_history_claimed": False,
        },
    }
    _write_json(inputs / "verified_checkpoint", verified)
    artifacts = {}
    protocol_names = set(native._PROTOCOL_ARTIFACTS)
    for index, name in enumerate(sorted(protocol_names)):
        path = inputs / name
        path.write_text(f"{name}-protocol-{index}\n")
    for name in sorted(protocol_names | native._NATIVE_ARTIFACTS):
        path = inputs / name
        identity = file_identity(path)
        artifacts[name] = {
            "path": name,
            "identity": identity,
            "provider": _provider(name, identity),
        }
    for name in native._LINEAGE_ARTIFACTS:
        artifacts[name]["provider"] = lineage[name]
    task = "fixture_task"
    mapping = {
        "schema": "npa.behavior.nonreporting-train-task-mapping.v1",
        "split": "train",
        "task": task,
        "data_namespace": "fixture_dataset",
        "data_task_id": 7,
        **{
            field: artifacts[field]["identity"]
            for field in ("source_manifest", "split_manifest", "mapping_artifact")
        },
    }
    science = {
        "behavior_upstream_commit": "6cbf70b075816096e9be53958780769f3264d25d",
        **{
            field: artifacts[field]["identity"]
            for field in (
                "task_registry",
                "dataset",
                "dataset_view",
                "normalization",
                "tokenizer",
                "action_semantics",
            )
        },
    }
    evaluator = {
        **{
            field: artifacts[field]["identity"]
            for field in (
                "argv_contract",
                "evaluator_source",
                "controller_source",
                "robot_config",
                "rng_contract",
            )
        },
        "wrapper": "fixture.Wrapper",
        "mode": "train",
        "num_envs": 1,
        "num_rollouts": 1,
        "write_video": True,
        "max_steps_argument": None,
        "model_prediction_horizon": 32,
        "executed_prefix": 32,
        "fresh_policy_process_per_case": True,
        "qualification_process_discarded": True,
    }
    protocol = train.declare_train_protocol(
        task,
        mapping,
        [{"instance_id": 0, "rollout_id": 0}],
        science,
        evaluator,
    )
    trace = {
        "schema": "npa.behavior.comet-native-action-trace.v1",
        "enabled": True,
        "action_width": 23,
        "left_command_index": 14,
        "right_command_index": 22,
        "left_proprio_indices": [24, 25],
        "right_proprio_indices": [49, 50],
    }
    binding_path = inputs / "binding.json"
    provisional = campaign.freeze_policy_identity(
        "comet-native-fixture",
        {
            "checkpoint": file_identity(inputs / "verified_checkpoint"),
            "serving": {"bytes": 1, "sha256": "0" * 64},
        },
    )
    binding = {
        "schema": native.BINDING_SCHEMA,
        "policy_identity": provisional,
        "task": task,
        "task_id": 7,
        "verified_checkpoint": file_identity(inputs / "verified_checkpoint"),
        "artifacts": artifacts,
        "trace": trace,
    }
    _write_json(binding_path, binding)
    args = SimpleNamespace(
        policy_kind="comet-native",
        policy_execution_variant="native",
        policy_native_binding=binding_path,
        policy_selected_export_receipt=None,
        policy_correlation_manifest=None,
        policy_validation_receipt=None,
        policy_stock_correlation_asset=None,
        policy_stock_correlation_sha256=None,
        policy_task_name=task,
    )
    policy = campaign.freeze_policy_identity(
        "comet-native-fixture",
        {
            "checkpoint": file_identity(inputs / "verified_checkpoint"),
            "serving": serving_identity.serving_artifact(args),
        },
    )
    binding["policy_identity"] = policy
    _write_json(binding_path, binding)
    assert serving_identity.serving_artifact(args) == policy["artifacts"]["serving"]
    panel = train.declare_train_panel(protocol, policy)
    return inputs, checkpoint, binding_path, panel, verified, binding, args


def test_native_checkpoint_admission_opens_all_artifacts_and_checkpoint(tmp_path):
    inputs, checkpoint, binding_path, panel, verified, _binding, args = _fixture(
        tmp_path
    )
    admission = native.validate_native_train_admission(
        binding_path,
        inputs,
        checkpoint,
        panel,
        expected_verified_checkpoint=inputs / "verified_checkpoint",
        expected_serving_identity=serving_identity.serving_artifact(args),
    )

    assert admission["logical_update"] == 2
    assert admission["manager_step"] == 1
    assert (
        admission["checkpoint_content_sha256"]
        == verified["checkpoint"]["content_sha256"]
    )


def test_prepare_native_policy_runs_discarded_loader_before_serving(
    tmp_path, monkeypatch
):
    inputs, checkpoint, binding_path, panel, verified, binding, args = _fixture(
        tmp_path
    )
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    _write_json(
        source / "scripts/task_mapping.json",
        {"fixture_task": {"task_index": 7, "task": "do fixture"}},
    )
    args.policy_native_input_root = inputs
    args.policy_checkpoint = checkpoint
    args.policy_archive = inputs / "verified_checkpoint"
    args.policy_root = source
    args.policy_python = Path("/runtime/python")
    args.port = 8123
    args.policy_native_panel = panel
    args.policy_native_admission = native.validate_native_train_admission(
        binding_path,
        inputs,
        checkpoint,
        panel,
        expected_verified_checkpoint=args.policy_archive,
        expected_serving_identity=serving_identity.serving_artifact(args),
    )
    monkeypatch.setattr(native_comet_policy, "verify_source", lambda _root: None)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        destination = Path(command[command.index("--qualification-output") + 1])
        case = panel["cases"][0]
        initial_rng = "1" * 64
        process_identity = native_comet_server._canonical_digest(
            {
                "schema": "npa.behavior.comet-native-serving-process-identity.v1",
                "case_id": case["case_id"],
                "task": binding["task"],
                "instance_id": case["instance_id"],
                "rollout_id": case["rollout_id"],
                "case_seed": native_comet_policy.case_seed(
                    binding["artifacts"]["rng_contract"]["identity"], case
                ),
                "checkpoint_content_sha256": verified["checkpoint"]["content_sha256"],
                "rng_contract_sha256": binding["artifacts"]["rng_contract"]["identity"][
                    "sha256"
                ],
                "trace_configuration_sha256": campaign.canonical_digest(
                    binding["trace"]
                ),
                "initial_rng_sha256": initial_rng,
            }
        )
        _write_json(
            destination,
            {
                "schema": "npa.behavior.comet-native-serving-load-qualification.v1",
                "status": "checkpoint_loaded_with_explicit_case_rng",
                "case_id": case["case_id"],
                "task": binding["task"],
                "instance_id": case["instance_id"],
                "rollout_id": case["rollout_id"],
                "case_seed": native_comet_policy.case_seed(
                    binding["artifacts"]["rng_contract"]["identity"], case
                ),
                "checkpoint_content_sha256": verified["checkpoint"]["content_sha256"],
                "manager_step": 1,
                "asset_id": "behavior-1k/2025-challenge-demos",
                "source_commit": native_comet_policy.SOURCE_COMMIT,
                "config_name": native_comet_policy.CONFIG_NAME,
                "rng_contract_sha256": binding["artifacts"]["rng_contract"]["identity"][
                    "sha256"
                ],
                "trace_configuration_sha256": campaign.canonical_digest(
                    binding["trace"]
                ),
                "initial_rng_sha256": initial_rng,
                "process_identity_sha256": process_identity,
                "inference_count": 0,
            },
        )

    monkeypatch.setattr(native_comet_policy.subprocess, "run", run)
    command = native_comet_policy.prepare_policy(
        args,
        campaign_runner._managed_plan(panel, panel["cases"][0]),
        tmp_path / "case",
    )

    assert len(calls) == 1
    assert "--qualify-only" in calls[0][0]
    assert calls[0][1] == {"cwd": source, "check": True}
    assert "--qualify-only" not in command
    assert command[command.index("--manager-step") + 1] == "1"
    assert command[command.index("--case-seed") + 1].isdigit()
    provenance = json.loads((tmp_path / "case/policy-provenance.json").read_text())
    assert (
        provenance["status"]
        == "discarded_load_qualification_complete_serving_not_started"
    )


@pytest.mark.parametrize("profile", ["action_expert", "full_sft"])
def test_native_architecture_profiles_require_exhaustive_dynamic_partitions(profile):
    value = _architecture(profile)
    assert native.validate_architecture(value) == value
    mutant = deepcopy(value)
    mutant["partition"]["trainable_paths"] = []
    with pytest.raises(ValueError, match="trainable paths|leaf coverage"):
        native.validate_architecture(mutant)


@pytest.mark.parametrize(
    "asset_id",
    [
        "../behavior",
        "/behavior",
        "behavior//dataset",
        "behavior/./dataset",
        "behavior/../dataset",
        "behavior\\dataset",
        "behavior/%2e%2e/dataset",
        "behavior/dataset?query",
    ],
)
def test_native_checkpoint_rejects_noncanonical_nested_asset_ids(tmp_path, asset_id):
    inputs, _checkpoint, _binding_path, _panel, verified, _binding, _args = _fixture(
        tmp_path
    )
    verified["serving_assets"] = {
        "asset_id": asset_id,
        "norm_stats_path": f"1/assets/{asset_id}/norm_stats.json",
    }
    with pytest.raises(ValueError, match="asset ID|serving assets"):
        native.validate_verified_checkpoint(verified)


def test_native_server_asset_id_uses_path_aware_canonical_validation():
    assert (
        native_comet_server._canonical_asset_id("behavior-1k/2025-challenge-demos")
        == "behavior-1k/2025-challenge-demos"
    )
    for value in ("../evil", "/evil", "good//evil", "good/../evil", "good%2fevil"):
        with pytest.raises(ValueError, match="asset ID"):
            native_comet_server._canonical_asset_id(value)


def test_native_checkpoint_rejects_cross_lineage_and_frozen_mutants(tmp_path):
    inputs, checkpoint, binding_path, panel, _verified, binding, args = _fixture(
        tmp_path
    )
    mutant = json.loads((inputs / "verified_checkpoint").read_text())
    mutant["discarded_finite_probe"]["provider_execution"] = _provider(
        "unrelated", {"bytes": 1, "sha256": "f" * 64}
    )
    _write_json(inputs / "verified_checkpoint", mutant)
    identity = file_identity(inputs / "verified_checkpoint")
    binding["verified_checkpoint"] = identity
    binding["artifacts"]["verified_checkpoint"].update(
        identity=identity, provider=_provider("verified_checkpoint", identity)
    )
    policy = campaign.freeze_policy_identity(
        "comet-native-fixture",
        {
            "checkpoint": identity,
            "serving": binding["policy_identity"]["artifacts"]["serving"],
        },
    )
    binding["policy_identity"] = policy
    _write_json(binding_path, binding)
    with pytest.raises(ValueError, match="probe execution lineage"):
        native.validate_native_train_admission(
            binding_path,
            inputs,
            checkpoint,
            train.declare_train_panel(panel["protocol"], policy),
            expected_verified_checkpoint=inputs / "verified_checkpoint",
            expected_serving_identity=policy["artifacts"]["serving"],
        )


def test_native_admission_rejects_noncanonical_provider_uri_and_frozen_value(
    tmp_path,
):
    inputs, checkpoint, binding_path, panel, verified, binding, args = _fixture(
        tmp_path
    )
    binding["artifacts"]["dataset"]["provider"]["uri"] += "?signature=mutable"
    _write_json(binding_path, binding)
    with pytest.raises(ValueError, match="provider URI"):
        native.validate_native_train_admission(
            binding_path,
            inputs,
            checkpoint,
            panel,
            expected_verified_checkpoint=inputs / "verified_checkpoint",
            expected_serving_identity=serving_identity.serving_artifact(args),
        )

    binding["artifacts"]["dataset"]["provider"]["uri"] = _provider(
        "dataset", binding["artifacts"]["dataset"]["identity"]
    )["uri"]
    _write_json(binding_path, binding)
    verified["frozen_parent_equality"]["params/backbone/b"] = {
        "parent_sha256": "f" * 64,
        "restored_sha256": "f" * 64,
    }
    _write_json(inputs / "verified_checkpoint", verified)
    identity = file_identity(inputs / "verified_checkpoint")
    binding["verified_checkpoint"] = identity
    binding["artifacts"]["verified_checkpoint"] = {
        "path": "verified_checkpoint",
        "identity": identity,
        "provider": _provider("verified_checkpoint", identity),
    }
    policy = campaign.freeze_policy_identity(
        "comet-native-fixture",
        {
            "checkpoint": identity,
            "serving": binding["policy_identity"]["artifacts"]["serving"],
        },
    )
    binding["policy_identity"] = policy
    _write_json(binding_path, binding)
    with pytest.raises(ValueError, match="frozen parent value"):
        native.validate_native_train_admission(
            binding_path,
            inputs,
            checkpoint,
            train.declare_train_panel(panel["protocol"], policy),
            expected_verified_checkpoint=inputs / "verified_checkpoint",
            expected_serving_identity=policy["artifacts"]["serving"],
        )


@pytest.mark.parametrize(
    "uri",
    [
        "s3://fixture-bucket/native-run/./original.json",
        "s3://fixture-bucket/native-run/original.json/",
        "s3://fixture-bucket/native-run//original.json",
    ],
)
def test_native_provider_row_rejects_normalized_noncanonical_spelling(uri):
    with pytest.raises(ValueError, match="provider URI"):
        native.canonical_provider_row(
            {
                "uri": uri,
                "bytes": 1,
                "sha256": "a" * 64,
                "provider_readback": True,
            },
            "fixture",
        )


@pytest.mark.parametrize(
    "relative", ["./artifact", "dir//artifact", "dir/./artifact", ""]
)
def test_native_local_artifact_rejects_normalized_noncanonical_spelling(
    tmp_path, relative
):
    (tmp_path / "artifact").write_text("x")
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir/artifact").write_text("x")
    with pytest.raises(ValueError, match="local path"):
        native._safe_local(tmp_path.resolve(), relative, "fixture")


def test_native_worker_rejects_bad_binding_before_startup_or_store(
    tmp_path, monkeypatch
):
    inputs, checkpoint, binding_path, panel, _verified, binding, args = _fixture(
        tmp_path
    )
    binding["artifacts"]["dataset"]["identity"]["sha256"] = "f" * 64
    _write_json(binding_path, binding)
    args.policy_native_input_root = inputs
    args.policy_checkpoint = checkpoint
    args.policy_archive = inputs / "verified_checkpoint"
    effects = []

    def unexpected(label):
        def fail(*_args, **_kwargs):
            effects.append(label)
            pytest.fail(f"native admission ran after {label}")

        return fail

    monkeypatch.setattr(
        campaign_runner, "_prepare_worker_startup", unexpected("startup")
    )
    monkeypatch.setattr(campaign_runner, "_specialist_preclaim", unexpected("preclaim"))
    monkeypatch.setattr(campaign_runner, "CaseStore", unexpected("case-store"))
    with pytest.raises(
        ValueError, match="provider identity|local bytes|protocol artifact"
    ):
        campaign_runner._execute_partition(
            args,
            panel,
            train.partition_train_panel(panel, 1),
            SimpleNamespace(objects={}),
            tmp_path / "workspace",
        )
    assert effects == []
    assert not (tmp_path / "workspace").exists()


def test_native_worker_admits_before_startup_and_preserves_admission_on_failure(
    tmp_path, monkeypatch
):
    inputs, checkpoint, binding_path, panel, _verified, _binding, args = _fixture(
        tmp_path
    )
    args.policy_native_input_root = inputs
    args.policy_checkpoint = checkpoint
    args.policy_archive = inputs / "verified_checkpoint"
    args.worker_receipt_uri = "s3://fixture-bucket/native-worker/receipt.json"
    workspace = tmp_path / "workspace"
    effects = []

    def fail_startup(_args, observed_workspace):
        effects.append("startup")
        assert observed_workspace == workspace
        assert (workspace / "native-train-admission.json").is_file()
        raise RuntimeError("startup stopped after native admission")

    class Storage:
        def __init__(self):
            self.objects = {}

        def put_bytes_conditional(self, payload, uri, *, if_none_match):
            assert if_none_match is True
            self.objects[uri] = payload

    storage = Storage()
    monkeypatch.setattr(campaign_runner, "_prepare_worker_startup", fail_startup)
    with pytest.raises(RuntimeError, match="after native admission"):
        campaign_runner._execute_partition(
            args,
            panel,
            train.partition_train_panel(panel, 1),
            storage,
            workspace,
        )
    assert effects == ["startup"]
    assert any(uri.endswith("native-train-admission.json") for uri in storage.objects)


def test_case_seed_excludes_checkpoint_and_trace_records_sent_action(tmp_path):
    case = {
        "case_id": "case-a",
        "task": "fixture_task",
        "instance_id": 2,
        "rollout_id": 0,
    }
    rng = {"bytes": 7, "sha256": "a" * 64}
    assert native_comet_policy.case_seed(rng, case) == native_comet_policy.case_seed(
        rng, case
    )
    assert native_comet_policy.case_seed(rng, case) != native_comet_policy.case_seed(
        rng, {**case, "instance_id": 3}
    )
    args = SimpleNamespace(
        case_id="case-a",
        case_seed=42,
        task_name="fixture_task",
        instance_id=2,
        rollout_id=0,
        checkpoint_sha256="b" * 64,
        rng_contract_sha256="a" * 64,
        trace_configuration_sha256="e" * 64,
        process_identity_sha256="f" * 64,
        initial_rng_sha256="c" * 64,
    )
    trace_path = tmp_path / "trace.jsonl"
    trace = native_comet_server.ActionTrace(trace_path, args)
    action = np.arange(23, dtype=np.float32)
    observation = {"robot_r1::proprio": np.arange(61, dtype=np.float32)}
    trace.append(action, observation, "c" * 64, "d" * 64, 0)
    row = json.loads(trace_path.read_text())
    assert row["sent_action"] == list(range(23))
    assert row["left_command"] == 14
    assert row["right_command"] == 22
    assert row["left_gripper_proprio"] == [24, 25]
    assert row["right_gripper_proprio"] == [49, 50]
    assert row["action_index"] == row["inference_ordinal"] == 0
    assert not ({"image", "tokens", "prompt", "parameters"} & set(row))


def test_native_trace_configuration_changes_serving_identity(tmp_path):
    inputs, _checkpoint, binding_path, _panel, _verified, binding, args = _fixture(
        tmp_path
    )
    enabled = serving_identity.serving_artifact(args)
    binding["trace"] = {
        "schema": "npa.behavior.comet-native-action-trace.v1",
        "enabled": False,
    }
    _write_json(binding_path, binding)
    disabled = serving_identity.serving_artifact(args)
    assert enabled != disabled
    assert (
        file_identity(inputs / "verified_checkpoint") == binding["verified_checkpoint"]
    )


def test_native_connection_checks_one_rng_split_and_traces_sent_23_vector(
    tmp_path, monkeypatch
):
    import jax

    class FakePacker:
        def pack(self, value):
            return value

    class FakeMsgpack:
        Packer = FakePacker

        @staticmethod
        def unpackb(value):
            return value

    class FakeWire:
        def __init__(self, _revision):
            pass

        @staticmethod
        def is_reset(_value):
            return False

        @staticmethod
        def observation_for_policy(value):
            return value

        @staticmethod
        def action_for_evaluator(value):
            return value

    class FakeSocket:
        def __init__(self, observation):
            self.observation = observation
            self.sent = []

        async def send(self, value):
            self.sent.append(value)

        def __aiter__(self):
            async def values():
                yield self.observation
                yield self.observation

            return values()

    class FakePolicy:
        def __init__(self):
            self._rng = jax.random.key(42)

    class FakeWrapper:
        def __init__(self, policy):
            self.policy = policy
            self.chunk = None
            self.index = 0

        def reset(self):
            pass

        def act(self, _inputs):
            if self.chunk is None:
                self.policy._rng, _ = jax.random.split(self.policy._rng)
                self.chunk = np.arange(32 * 23, dtype=np.float32).reshape(32, 23)
            action = self.chunk[self.index : self.index + 1]
            self.index += 1
            return action

    monkeypatch.setitem(
        sys.modules, "openpi_client", SimpleNamespace(msgpack_numpy=FakeMsgpack)
    )
    monkeypatch.setattr(native_comet_server, "EvaluatorWire", FakeWire)
    monkeypatch.setattr(
        native_comet_server,
        "policy_observation",
        lambda observation: observation,
    )
    args = SimpleNamespace(
        upstream_commit="fixture",
        case_id="case-a",
        case_seed=42,
        task_name="fixture_task",
        instance_id=2,
        rollout_id=0,
        checkpoint_sha256="b" * 64,
        rng_contract_sha256="a" * 64,
        trace_configuration_sha256="e" * 64,
        initial_rng_sha256=native_comet_server._key_identity(jax.random.key(42)),
        process_receipt=tmp_path / "native-process.json",
        action_trace=tmp_path / "native-actions.jsonl",
    )
    args.process_identity_sha256 = native_comet_server._process_identity(
        args, args.initial_rng_sha256
    )
    observation = {"robot_r1::proprio": np.arange(61, dtype=np.float32)}
    socket = FakeSocket(observation)
    policy = FakePolicy()
    wrapper = FakeWrapper(policy)
    trace = native_comet_server.ActionTrace(tmp_path / "native-actions.jsonl", args)
    progress = native_comet_server.ProcessProgress(
        tmp_path / "native-process-progress.jsonl", args
    )
    native_comet_server._atomic_json(
        args.process_receipt,
        native_comet_server._process_receipt(
            args, args.initial_rng_sha256, args.initial_rng_sha256, 0, 0
        ),
    )
    asyncio.run(
        native_comet_server._connection(socket, wrapper, policy, args, trace, progress)
    )
    progress.close()

    assert wrapper.chunk.shape == (32, 23)
    assert all(np.asarray(item["action"]).shape == (23,) for item in socket.sent[1:])
    assert trace.rows == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "native-actions.jsonl").read_text().splitlines()
    ]
    assert [row["inference_ordinal"] for row in rows] == [0, 0]
    progress_rows = [
        json.loads(line)
        for line in (tmp_path / "native-process-progress.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(progress_rows) == 2
    receipt = progress_rows[-1]
    assert receipt["inference_count"] == 1
    assert receipt["initial_rng_sha256"] != receipt["current_rng_sha256"]
    final = native_comet_policy.finalize_process(tmp_path, -15)
    assert final["action_count"] == 2
    assert final["inference_count"] == 1
    assert final["trace"]["rows"] == 2


def test_disabled_native_trace_keeps_one_contiguous_progress_journal(tmp_path):
    import jax

    initial = native_comet_server._key_identity(jax.random.key(42))
    current = native_comet_server._key_identity(jax.random.key(43))
    args = SimpleNamespace(
        case_id="case-a",
        case_seed=42,
        task_name="fixture_task",
        instance_id=2,
        rollout_id=0,
        checkpoint_sha256="b" * 64,
        rng_contract_sha256="a" * 64,
        trace_configuration_sha256="e" * 64,
        action_trace=None,
        initial_rng_sha256=initial,
    )
    args.process_identity_sha256 = native_comet_server._process_identity(args, initial)
    trace = native_comet_server.ActionTrace(None, args)
    trace.append(
        np.arange(23, dtype=np.float32),
        {"robot_r1::proprio": np.arange(61, dtype=np.float32)},
        initial,
        current,
        0,
    )
    assert trace.rows == 1
    native_comet_server._atomic_json(
        tmp_path / "native-process.json",
        native_comet_server._process_receipt(args, initial, initial, 0, 0),
    )
    progress = native_comet_server.ProcessProgress(
        tmp_path / "native-process-progress.jsonl", args
    )
    progress.append(current, 1)
    progress.close()
    final = native_comet_policy.finalize_process(tmp_path, -15)
    assert final["action_count"] == final["inference_count"] == 1
    assert final["trace"] is None
    assert final["progress_journal"] == file_identity(
        tmp_path / "native-process-progress.jsonl"
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "native-process-final.json",
        "native-process-progress.jsonl",
        "native-process.json",
    ]


def test_native_connection_records_sent_action_only_after_send(tmp_path, monkeypatch):
    import jax

    class FakePacker:
        def pack(self, value):
            return value

    class FakeMsgpack:
        Packer = FakePacker

        @staticmethod
        def unpackb(value):
            return value

    class FakeWire:
        def __init__(self, _revision):
            pass

        @staticmethod
        def is_reset(_value):
            return False

        @staticmethod
        def observation_for_policy(value):
            return value

        @staticmethod
        def action_for_evaluator(value):
            return value

    class FailingSocket:
        def __init__(self, observation):
            self.observation = observation
            self.sends = 0

        async def send(self, _value):
            self.sends += 1
            if self.sends == 2:
                raise ConnectionError("action was not sent")

        def __aiter__(self):
            async def values():
                yield self.observation

            return values()

    class FakePolicy:
        def __init__(self):
            self._rng = jax.random.key(42)

    class FakeWrapper:
        def __init__(self, policy):
            self.policy = policy

        def reset(self):
            pass

        def act(self, _inputs):
            self.policy._rng, _ = jax.random.split(self.policy._rng)
            return np.arange(23, dtype=np.float32)[None, :]

    monkeypatch.setitem(
        sys.modules, "openpi_client", SimpleNamespace(msgpack_numpy=FakeMsgpack)
    )
    monkeypatch.setattr(native_comet_server, "EvaluatorWire", FakeWire)
    monkeypatch.setattr(native_comet_server, "policy_observation", lambda value: value)
    initial = native_comet_server._key_identity(jax.random.key(42))
    args = SimpleNamespace(
        upstream_commit="fixture",
        case_id="case-a",
        case_seed=42,
        task_name="fixture_task",
        instance_id=2,
        rollout_id=0,
        checkpoint_sha256="b" * 64,
        rng_contract_sha256="a" * 64,
        trace_configuration_sha256="e" * 64,
        initial_rng_sha256=initial,
        action_trace=tmp_path / "native-actions.jsonl",
    )
    args.process_identity_sha256 = native_comet_server._process_identity(args, initial)
    trace = native_comet_server.ActionTrace(args.action_trace, args)
    progress = native_comet_server.ProcessProgress(
        tmp_path / "native-process-progress.jsonl", args
    )
    policy = FakePolicy()
    with pytest.raises(ConnectionError, match="not sent"):
        asyncio.run(
            native_comet_server._connection(
                FailingSocket({"robot_r1::proprio": np.arange(61, dtype=np.float32)}),
                FakeWrapper(policy),
                policy,
                args,
                trace,
                progress,
            )
        )
    progress.close()
    assert trace.rows == 0
    assert (tmp_path / "native-actions.jsonl").read_bytes() == b""
    assert (tmp_path / "native-process-progress.jsonl").read_bytes() == b""


def test_native_finalizer_rejects_consistently_sized_trace_substitution(
    tmp_path, monkeypatch
):
    test_native_connection_checks_one_rng_split_and_traces_sent_23_vector(
        tmp_path, monkeypatch
    )
    (tmp_path / "native-process-final.json").unlink()
    rows = (tmp_path / "native-actions.jsonl").read_text().splitlines()
    mutant = json.loads(rows[0])
    mutant["left_command"] = mutant["left_command"] + 1
    rows[0] = json.dumps(mutant, sort_keys=True, separators=(",", ":"))
    (tmp_path / "native-actions.jsonl").write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="trace values"):
        native_comet_policy.finalize_process(tmp_path, -15)


def test_native_finalizer_rejects_noncontiguous_progress_journal(tmp_path, monkeypatch):
    test_native_connection_checks_one_rng_split_and_traces_sent_23_vector(
        tmp_path, monkeypatch
    )
    (tmp_path / "native-process-final.json").unlink()
    path = tmp_path / "native-process-progress.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["action_count"] = 3
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    with pytest.raises(ValueError, match="action count"):
        native_comet_policy.finalize_process(tmp_path, -15)
