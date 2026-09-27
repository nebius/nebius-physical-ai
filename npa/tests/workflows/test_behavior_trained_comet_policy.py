"""Exercise selected-parity Comet export admission and runner wiring."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows.behavior_challenge import campaign
from npa.workflows.behavior_challenge import campaign_runner
from npa.workflows.behavior_challenge import native_comet_server
from npa.workflows.behavior_challenge import nonreporting_train
from npa.workflows.behavior_challenge import serving_identity
from npa.workflows.behavior_challenge import trained_comet_checkpoint as trained
from npa.workflows.behavior_challenge import trained_comet_policy
from npa.workflows.behavior_challenge import trained_comet_producer
from npa.workflows.behavior_challenge.native_training_checkpoint import file_identity

ACTUAL_CHECKPOINT_SHA256 = (
    "0f0685e3eb31585c679996611517f5d4d046fd47d2c66edbdd92cc42909f9ffd"
)
SELECTED_STEP = 20_000


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _provider(path: Path, name: str) -> dict:
    return {
        "uri": f"s3://fixture-bucket/trained/{name}",
        **file_identity(path),
        "provider_readback": True,
    }


def _canonical(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _native_inventory() -> dict:
    names = (
        "_METADATA",
        "_sharding",
        "array_metadatas/process_0",
        "d/data",
        "manifest.ocdbt",
        "ocdbt.process_0/d/data",
        "ocdbt.process_0/manifest.ocdbt",
    )
    files = [
        {"path": f"params/{name}", "bytes": 3, "sha256": f"{index + 1:064x}"}
        for index, name in enumerate(names)
    ]
    return {
        "schema": "npa.behavior.orbax-params-inventory.v1",
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
    }


def _precision(checkpoint_sha256: str = ACTUAL_CHECKPOINT_SHA256) -> dict:
    leaves = [
        {
            "ordinal": ordinal,
            "shape": [ordinal + 1],
            "dtype": "bfloat16",
            "bytes": 2 * (ordinal + 1),
            "sha256": f"{ordinal + 100:064x}",
        }
        for ordinal in range(51)
    ]
    return {
        "schema": "npa.behavior.comet12-serving-bf16-parameter-inventory.v2",
        "status": "native_restore_params_bfloat16_cast_inventory_complete",
        "checkpoint_inventory_sha256": checkpoint_sha256,
        "loader_call": (
            "config.model.load(model.restore_params(checkpoint/'params', "
            "dtype=jnp.bfloat16))"
        ),
        "source_precision": "float32_parameters",
        "checkpoint_role": "trained_milestone_full_state",
        "deployed_precision": "bfloat16",
        "leaves": leaves,
        "leaf_count": 51,
        "total_parameter_bytes": sum(row["bytes"] for row in leaves),
        "cast_inventory_sha256": _canonical(leaves),
    }


def _process(
    label: str,
    pid: int,
    schema_sha256: str,
    npz: dict,
    action: dict,
    full_state_sha256: str,
) -> dict:
    return {
        "schema": ("npa.behavior.comet12-trained-milestone-serving-parity-process.v1"),
        "status": "exact_bf16_policy_action_parity_from_selected_milestone",
        "process_label": label,
        "pid": pid,
        "python": "3.11.16",
        "cold_compilation_cache": f"/fixture/jax-cache-{label}",
        "actions": {"native": action, "exported": action},
        "trained_milestone": True,
        "rollout_quality": False,
        "fp32_full_state_inventory_sha256": full_state_sha256,
        "device_count": 1,
        "device_platform": "gpu",
        "parameter_precision": "bfloat16_for_native_and_exported",
        "input_schema": {
            "sha256": schema_sha256,
            "npz": npz,
            "sample": None,
            "synthetic": False,
        },
    }


def _state_inventory(leaves: int, elements: int, byte_count: int, seed: int) -> dict:
    return {
        "leaf_count": leaves,
        "total_elements": elements,
        "total_bytes": byte_count,
        "content_sha256": f"{seed:064x}",
    }


def _state_contract() -> dict:
    return {
        "train_state_step": SELECTED_STEP,
        "inventories": {
            "all_params": _state_inventory(51, 100, 400, 201),
            "trainable_params": _state_inventory(19, 40, 160, 202),
            "frozen_params": _state_inventory(32, 60, 240, 203),
            "optimizer": {"leaf_count": 40, "content_sha256": f"{204:064x}"},
            "adamw_mu": {"leaf_count": 19, "content_sha256": f"{205:064x}"},
            "adamw_nu": {"leaf_count": 19, "content_sha256": f"{206:064x}"},
        },
        "optimizer_scalar_progress": {
            "[1][0].count": SELECTED_STEP,
            "[1][2].count": SELECTED_STEP,
        },
        "optimizer_paths": {"bytes": 3, "sha256": f"{207:064x}"},
        "all_params_fp32": True,
    }


def _full_checkpoint(norm: Path) -> tuple[dict, dict]:
    native = _native_inventory()
    rows = [
        {**row, "path": f"19999/{row['path']}", "mode": "0o644"}
        for row in native["files"]
    ]
    train_names = (
        "_METADATA",
        "_sharding",
        "array_metadatas/process_0",
        "d/data",
        "manifest.ocdbt",
        "ocdbt.process_0/d/data",
        "ocdbt.process_0/manifest.ocdbt",
    )
    rows.extend(
        {
            "path": f"19999/train_state/{name}",
            "bytes": 4,
            "sha256": f"{index + 20:064x}",
            "mode": "0o644",
        }
        for index, name in enumerate(train_names)
    )
    rows.extend(
        (
            {
                "path": "19999/_CHECKPOINT_METADATA",
                "bytes": 2,
                "sha256": f"{40:064x}",
                "mode": "0o644",
            },
            {
                "path": "19999/assets/behavior-1k/2025-challenge-demos/norm_stats.json",
                **file_identity(norm),
                "mode": "0o644",
            },
        )
    )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    full = {
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "max_member_bytes": max(row["bytes"] for row in rows),
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    serving_rows = [row for row in rows if row["path"].startswith("19999/params/")]
    serving = {
        "root": "19999/params",
        "dtype": "float32_native_training_state",
        "bf16_serving_derivative_created": False,
        "file_count": len(serving_rows),
        "total_bytes": sum(row["bytes"] for row in serving_rows),
        "files": serving_rows,
    }
    return full, serving


def _selected_reader(chosen: dict, score: dict, score_path: Path) -> dict:
    fields = {
        "checkpoint_inventory_sha256",
        "ordered_input_sha256",
        "process_identity",
        "rng_schedule_sha256",
        "sample_inventory_sha256",
        "scorer_contract_sha256",
        "serving_precision_receipt_sha256",
        "step",
        "tasks",
    }
    return {
        "schema": "npa.behavior.comet12-dp4-score-independent-readback.v2",
        "status": "one_exact_dp4_train_score_provider_originals_verified",
        "candidate_arm": chosen["arm"],
        "logical_update": chosen["logical_update"],
        "selection_executed": False,
        "development_or_report_read": False,
        "score": {key: score["score"].get(key) for key in fields},
        "originals": {"score-process.json": file_identity(score_path)},
    }


def _parity(export: Path, norm: Path, precision: dict, evidence_root: Path) -> dict:
    native_inventory = _native_inventory()
    checkpoint_sha256 = _canonical(native_inventory)
    assert precision["checkpoint_inventory_sha256"] == checkpoint_sha256
    export_row = _provider(export, "exported-checkpoint/params/data")
    parity_log = evidence_root / "parity.log"
    parity_log.write_bytes(b"")
    bootstrap = evidence_root / "runtime-bootstrap.json"
    locked = evidence_root / "runtime-inputs/locked-runtime.json"
    installed = evidence_root / "runtime-inputs/installed-distributions.json"
    runtime = evidence_root / "runtime-contract.json"
    result = evidence_root / "parity-result.json"
    milestone_manifest = evidence_root / "prepared-run/inputs/milestone_manifest.json"
    milestone_receipt = evidence_root / "prepared-run/inputs/milestone_receipt.json"
    selection_path = evidence_root / "prepared-run/inputs/selection_receipt.json"
    reader_path = evidence_root / "selected-reader.json"
    score_path = evidence_root / "prepared-run/inputs/score_receipt.json"
    fixed_schema = evidence_root / "prepared-run/inputs/fixed_input_schema.json"
    fixed_npz = evidence_root / "prepared-run/inputs/fixed_input_npz.npz"
    fixed_npz.parent.mkdir(parents=True)
    fixed_npz.write_bytes(b"fixed-input")
    _write(
        locked,
        {
            "schema": "npa.behavior.comet-locked-runtime.v2",
            "status": "exact_lock_environment_with_upstream_import_ready",
            "source_commit": trained.SOURCE_COMMIT,
            "venv_path": "/fixture/venv",
            "dataset_or_model_downloaded": False,
        },
    )
    _write(installed, {})
    _write(
        bootstrap,
        {
            "schema": "npa.behavior.comet12-trained-parity-runtime-bootstrap.v1",
            "status": "runtime_base_locked_venv_and_nested_npa_source_ready",
            "locked_receipt": file_identity(locked),
            "locked_inventory": file_identity(installed),
            "npa_import_root": "/fixture/npa/src",
            "python_executable": "/fixture/venv/bin/python",
            "venv_path": "/fixture/venv",
        },
    )
    _write(
        fixed_schema,
        {
            "schema": "npa.behavior.comet12-real-native-fixed-input.v1",
            "status": "real_r6_cpu_transformed_observation_and_fixed_noise",
            "noise": {
                "shape": [32, 32],
                "dtype": "float32",
                "sha256": trained.FIXED_NOISE["expected_c_sha256"],
            },
            "npz": file_identity(fixed_npz),
            "provenance": {
                "noise_recipe": trained.FIXED_NOISE,
                "observation_contract": {
                    "episode_index": 200,
                    "ordinal": 0,
                    "split": "TRAIN",
                    "task_id": 1,
                },
            },
            "synthetic": False,
        },
    )
    inventory = {
        "schema": "npa.behavior.orbax-params-inventory.v1",
        "file_count": 1,
        "total_bytes": export_row["bytes"],
        "files": [
            {
                "path": "params/data",
                "bytes": export_row["bytes"],
                "sha256": export_row["sha256"],
            }
        ],
    }
    full_inventory, serving_params = _full_checkpoint(norm)
    full_state_sha256 = _canonical(full_inventory)
    cursor = {
        "epoch": SELECTED_STEP // 3729,
        "batch_offset": SELECTED_STEP % 3729,
        "committed_global_batch": SELECTED_STEP,
    }
    progress = {
        "bytes": 3,
        "sha256": f"{301:064x}",
        "line_count": SELECTED_STEP - 15_000,
        "first_logical_update": 15_001,
        "last_logical_update": SELECTED_STEP,
    }
    _write(
        milestone_receipt,
        {
            "arm": "action_expert",
            "start_mode": "milestone_15000",
            "logical_update_count": SELECTED_STEP,
            "manager_step": SELECTED_STEP - 1,
            "train_state_step": SELECTED_STEP,
            "cursor": cursor,
            "progress": progress,
            "state_contract": _state_contract(),
            "checkpoint": full_inventory,
            "serving_params": serving_params,
        },
    )
    _write(
        milestone_manifest,
        {
            "schema": "npa.behavior.comet12-native-training-milestone.v2",
            "status": "complete_full_state_and_serving_params_provider_readback",
            "outcome": "success",
            "logical_update_count": SELECTED_STEP,
            "manager_step": SELECTED_STEP - 1,
            "full_state_resume_ready": True,
            "serving_export_qualified": False,
            "scoring_executed": False,
            "cursor": cursor,
            "checkpoint": full_inventory,
            "serving_params": serving_params,
            "files": {},
        },
    )
    milestone = json.loads(milestone_manifest.read_text())
    provider_root = "s3://fixture-bucket/milestone/originals"
    milestone["files"] = {
        "milestone-receipt.json": {
            "uri": f"{provider_root}/milestone-receipt.json",
            **file_identity(milestone_receipt),
            "provider_readback": True,
        },
        **{
            f"checkpoint/{row['path']}": {
                "uri": f"{provider_root}/checkpoint/{row['path']}",
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "provider_readback": True,
            }
            for row in full_inventory["files"]
        },
    }
    _write(milestone_manifest, milestone)
    score = {
        "schema": "npa.behavior.comet12-native-scorer-process.v2",
        "status": "native_bf16_shape1_jit_score_complete",
        "step": SELECTED_STEP,
        "checkpoint_role": "trained_milestone_full_state",
        "fp32_parameter_source_inventory_sha256": checkpoint_sha256,
        "precision": precision,
        "score": {
            "schema": "npa.behavior.comet12-holdout-score-draft.v5",
            "checkpoint_inventory_sha256": checkpoint_sha256,
            "serving_precision_receipt_sha256": _canonical(precision),
            "step": SELECTED_STEP,
            "ordered_input_sha256": f"{501:064x}",
            "process_identity": "fixture-score-process",
            "rng_schedule_sha256": f"{502:064x}",
            "sample_inventory_sha256": f"{503:064x}",
            "scorer_contract_sha256": f"{504:064x}",
            "tasks": {"1": {"mean": 0.5}},
            "parameter_precision": (
                "native_serving_bfloat16_cast_from_fp32_parameters"
            ),
            "rng": {
                "construction": "jax.random.fold_in(base_key, global_example_ordinal)",
                "example_count": 1,
                "seed": 4049,
            },
            "official_development_or_report_examples_used": False,
            "train": False,
        },
    }
    _write(score_path, score)
    chosen = {
        "arm": "action_expert",
        "logical_update": SELECTED_STEP,
        "eligible": True,
        "checkpoint_inventory_sha256": checkpoint_sha256,
        "serving_precision_receipt_sha256": _canonical(precision),
    }
    _write(reader_path, _selected_reader(chosen, score, score_path))
    chosen["reader"] = file_identity(reader_path)
    candidates = [
        {
            **chosen,
            "arm": arm,
            "logical_update": step,
            "eligible": eligible,
            "reader": {"bytes": 100 + index, "sha256": f"{400 + index:064x}"},
        }
        for index, (arm, step, eligible) in enumerate(
            (
                ("action_expert", 15_000, True),
                ("full_sft", 15_000, False),
                ("full_sft", 20_000, False),
            )
        )
    ]
    candidates.append(chosen)
    selection = {
        "schema": "npa.private.comet12-dp4-b200-parent-finalizer-adapter.v1",
        "status": "frozen_selection_completed_from_accepted_b200_parent",
        "selection": {
            "schema": "npa.behavior.comet12-dp4-four-candidate-selection.v1",
            "status": "complete_train_only_dp4_selection",
            "candidate_set_complete": True,
            "development_or_report_used": False,
            "one_gpu_candidate_relabelled": False,
            "candidates": candidates,
            "selected": chosen,
        },
    }
    _write(selection_path, selection)
    action_array = np.zeros((32, 32), dtype=np.float32)
    action = {
        "shape": [32, 32],
        "dtype": "float32",
        "sha256": hashlib.sha256(action_array.tobytes(order="C")).hexdigest(),
    }
    processes = [
        _process(
            "cold-a",
            101,
            file_identity(fixed_schema)["sha256"],
            file_identity(fixed_npz),
            action,
            full_state_sha256,
        ),
        _process(
            "cold-b",
            202,
            file_identity(fixed_schema)["sha256"],
            file_identity(fixed_npz),
            action,
            full_state_sha256,
        ),
    ]
    cast = {
        "schema": "npa.behavior.comet12-bf16-serving-cast-inventory.v1",
        "source_precision": "float32_native_training_state",
        "deployed_precision": "bfloat16",
        "leaves": precision["leaves"],
        "leaf_count": 51,
        "total_bytes": precision["total_parameter_bytes"],
        "canonical_sha256": _canonical(precision["leaves"]),
    }
    _write(
        runtime,
        {
            "schema": (
                "npa.behavior.comet12-trained-selected-milestone-parity-input.v1"
            ),
            "model_kind": "qualified_comet12",
            "selected_step": SELECTED_STEP,
            "runtime_bootstrap": file_identity(bootstrap),
            "source_commit": trained.SOURCE_COMMIT,
            "trained_milestone": True,
            "milestone_parity_satisfied": False,
            "official_24gb_hardware_qualified": False,
            "static_reconstruction": trained.SCORER_STATIC_RECONSTRUCTION,
            "exported_inventory_sha256": _canonical(inventory),
            "fp32_full_state_inventory": full_inventory,
            "fp32_full_state_inventory_sha256": full_state_sha256,
            "native_inventory": native_inventory,
            "native_inventory_sha256": checkpoint_sha256,
            "bf16_cast_inventory": cast,
            "selected_v13_precision_receipt": precision,
            "selection_receipt": selection,
            "score_receipt_identity": {
                name: _provider(score_path, "prepared-run/inputs/score_receipt.json")[
                    name
                ]
                for name in ("uri", "bytes", "sha256")
            },
            "milestone_receipt": json.loads(milestone_receipt.read_text()),
            "input_schema_sha256": file_identity(fixed_schema)["sha256"],
        },
    )
    claims = {
        "full_model_weights_loaded": True,
        "gpu_execution_verified": True,
        "official_24gb_hardware_qualified": False,
        "rollout_quality": False,
        "trained_milestone": True,
        "milestone_parity_satisfied": True,
        "selected_milestone_serving_parity_admission": True,
    }
    _write(
        result,
        {
            "schema": (
                "npa.behavior.comet12-trained-selected-milestone-serving-parity.v1"
            ),
            "status": "two_cold_processes_exact_selected_milestone_action_parity",
            "processes": processes,
            "claims": claims,
        },
    )
    action_files = {}
    for process in processes:
        label = process["process_label"]
        process_path = evidence_root / f"work/result-{label}.json"
        raw_path = evidence_root / f"work/raw-actions-{label}.npz"
        receipt_path = evidence_root / f"work/raw-actions-{label}.json"
        _write(process_path, process)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(raw_path, native=action_array, exported=action_array)
        _write(
            receipt_path,
            {
                "schema": (
                    "npa.behavior.comet12-trained-milestone-parity-raw-actions.v1"
                ),
                "status": (
                    "raw_native_and_exported_actions_recorded_before_equality_gate"
                ),
                "process_label": label,
                "pid": process["pid"],
                "actions": process["actions"],
                "npz": file_identity(raw_path),
                "admission_granted": False,
            },
        )
        action_files.update(
            {
                f"work/result-{label}.json": _provider(
                    process_path, f"work/result-{label}.json"
                ),
                f"work/raw-actions-{label}.json": _provider(
                    receipt_path, f"work/raw-actions-{label}.json"
                ),
                f"work/raw-actions-{label}.npz": _provider(
                    raw_path, f"work/raw-actions-{label}.npz"
                ),
            }
        )
    return {
        "schema": trained.PARITY_SCHEMA,
        "status": trained.PARITY_STATUS,
        "outcome": "success",
        "run_id": "selected-parity-fixture",
        "exit_code": 0,
        "official_24gb_hardware_qualified": False,
        "selected_milestone_serving_parity_admission": True,
        "files": {
            "parity.log": _provider(parity_log, "parity.log"),
            "runtime-bootstrap.json": _provider(bootstrap, "runtime-bootstrap.json"),
            "runtime-inputs/locked-runtime.json": _provider(
                locked, "runtime-inputs/locked-runtime.json"
            ),
            "runtime-inputs/installed-distributions.json": _provider(
                installed, "runtime-inputs/installed-distributions.json"
            ),
            "runtime-contract.json": _provider(runtime, "runtime-contract.json"),
            "parity-result.json": _provider(result, "parity-result.json"),
            "prepared-run/inputs/milestone_manifest.json": _provider(
                milestone_manifest, "prepared-run/inputs/milestone_manifest.json"
            ),
            "prepared-run/inputs/milestone_receipt.json": _provider(
                milestone_receipt, "prepared-run/inputs/milestone_receipt.json"
            ),
            "prepared-run/inputs/selection_receipt.json": _provider(
                selection_path, "prepared-run/inputs/selection_receipt.json"
            ),
            "prepared-run/inputs/score_receipt.json": _provider(
                score_path, "prepared-run/inputs/score_receipt.json"
            ),
            "prepared-run/inputs/fixed_input_schema.json": _provider(
                fixed_schema, "prepared-run/inputs/fixed_input_schema.json"
            ),
            "prepared-run/inputs/fixed_input_npz.npz": _provider(
                fixed_npz, "prepared-run/inputs/fixed_input_npz.npz"
            ),
            "exported-checkpoint/params/data": export_row,
            **action_files,
        },
        "success_evidence": {
            "selected_step": SELECTED_STEP,
            "processes": processes,
            "fp32_full_state_inventory_sha256": full_state_sha256,
            "native_inventory_sha256": checkpoint_sha256,
            "exported_inventory_sha256": _canonical(inventory),
            "selected_v13_precision_receipt": precision,
            "bf16_cast_inventory": cast,
            "selection_receipt": selection,
        },
    }


def _trace() -> dict:
    return {
        "schema": "npa.behavior.comet-native-action-trace.v1",
        "enabled": False,
        "action_width": 23,
        "left_command_index": 14,
        "right_command_index": 22,
        "left_proprio_indices": [24, 25],
        "right_proprio_indices": [49, 50],
    }


def _args(admission: Path, checkpoint: Path, inputs: Path) -> SimpleNamespace:
    return SimpleNamespace(
        policy_kind="comet-trained",
        policy_execution_variant="native",
        policy_archive=admission,
        policy_checkpoint=checkpoint,
        policy_trained_input_root=inputs,
        policy_task_name="picking_up_trash",
        policy_selected_export_receipt=None,
        policy_correlation_manifest=None,
        policy_validation_receipt=None,
        policy_stock_correlation_asset=None,
        policy_stock_correlation_sha256=None,
        train_experience=False,
        train_experience_depth=False,
    )


def _artifact(marker: str) -> dict:
    return {"bytes": 1, "sha256": marker * 64}


def _train_panel(
    args,
    admission: Path,
    *,
    wrapper: str = "omnigibson.evaluator.Evaluator",
    robot_config: dict | None = None,
) -> dict:
    admitted = json.loads(admission.read_text())
    normalization = {
        name: admitted["normalization"]["provider"][name]
        for name in ("bytes", "sha256")
    }
    rng_contract = {
        name: admitted["rng_contract"]["provider"][name] for name in ("bytes", "sha256")
    }
    protocol = nonreporting_train.declare_train_protocol(
        "picking_up_trash",
        {
            "schema": "npa.behavior.nonreporting-train-task-mapping.v1",
            "split": "train",
            "task": "picking_up_trash",
            "data_namespace": "behavior_1k",
            "data_task_id": 1,
            "source_manifest": _artifact("1"),
            "split_manifest": _artifact("2"),
            "mapping_artifact": _artifact("3"),
        },
        [{"instance_id": 0, "rollout_id": 0}],
        {
            "behavior_upstream_commit": "6cbf70b075816096e9be53958780769f3264d25d",
            "task_registry": _artifact("4"),
            "dataset": _artifact("5"),
            "dataset_view": _artifact("6"),
            "normalization": normalization,
            "tokenizer": _artifact("8"),
            "action_semantics": _artifact("9"),
        },
        {
            "argv_contract": _artifact("a"),
            "evaluator_source": _artifact("b"),
            "controller_source": _artifact("c"),
            "robot_config": robot_config or _artifact("d"),
            "rng_contract": rng_contract,
            "wrapper": wrapper,
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
        "dp4-action-expert-20k-train-recorder",
        {
            "checkpoint": file_identity(admission),
            "serving": serving_identity.serving_artifact(args),
        },
    )
    return nonreporting_train.declare_train_panel(protocol, policy)


def _fixture(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    export = checkpoint / "20000/params/data"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"bf16-export")
    norm = checkpoint / "20000/assets/behavior-1k/2025-challenge-demos/norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}\n")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    rng = inputs / "rng-contract.json"
    rng.write_text("{}\n")
    precision = _precision(_canonical(_native_inventory()))
    parity = _parity(export, norm, precision, inputs)
    parity_path = inputs / "parity-output-manifest.json"
    _write(parity_path, parity)
    actual = {
        path.relative_to(checkpoint).as_posix(): file_identity(path)
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file()
    }
    admission = tmp_path / "trained-admission.json"
    _write(
        admission,
        {
            "schema": trained.ADMISSION_SCHEMA,
            "status": "actual_selected_parity_export_and_normalization_admitted",
            "profile": "action_expert",
            "selected_step": 20000,
            "manager_step": 20000,
            "selected_checkpoint_inventory_sha256": precision[
                "checkpoint_inventory_sha256"
            ],
            "selected_precision_receipt_sha256": _canonical(precision),
            "task": "picking_up_trash",
            "task_id": 1,
            "source_commit": trained.SOURCE_COMMIT,
            "parity_output": _provider(parity_path, "parity-output-manifest.json"),
            "selected_reader": {
                "path": "selected-reader.json",
                "provider": _provider(
                    inputs / "selected-reader.json", "selected-reader.json"
                ),
            },
            "normalization": {
                "path": norm.relative_to(checkpoint).as_posix(),
                "asset_id": "behavior-1k/2025-challenge-demos",
                "provider": _provider(norm, "norm_stats.json"),
            },
            "rng_contract": {
                "path": rng.name,
                "provider": _provider(rng, rng.name),
            },
            "trace": _trace(),
            "serving_tree_sha256": campaign.canonical_digest(actual),
            "claims": {
                "optimizer_or_train_state_claimed": False,
                "development_or_report_read": False,
                "rollout_quality_claimed": False,
            },
        },
    )
    args = _args(admission, checkpoint, inputs)
    policy = campaign.freeze_policy_identity(
        "dp4-action-expert-20k",
        {
            "checkpoint": file_identity(admission),
            "serving": serving_identity.serving_artifact(args),
        },
    )
    registry = [f"task-{index}" for index in range(100)]
    registry[1] = "picking_up_trash"
    panel = campaign.declare_panel(
        policy, registry, ["picking_up_trash"], "development"
    )
    return admission, checkpoint, inputs, panel, args


def test_selected_export_admission_composes_with_dev_panel(
    tmp_path: Path, monkeypatch
) -> None:
    admission, checkpoint, inputs, panel, args = _fixture(tmp_path, monkeypatch)
    result = trained.validate_trained_comet_admission(
        admission,
        inputs,
        checkpoint,
        panel,
        expected_serving_identity=serving_identity.serving_artifact(args),
    )
    assert result["selected_step"] == 20000
    assert result["status"] == "selected_bf16_export_ready_for_fresh_panel_process"


def test_export_mutation_rejects_before_panel_claim(
    tmp_path: Path, monkeypatch
) -> None:
    admission, checkpoint, inputs, panel, args = _fixture(tmp_path, monkeypatch)
    (checkpoint / "20000/params/data").write_bytes(b"changed")
    with pytest.raises(ValueError, match="serving tree differs"):
        trained.validate_trained_comet_admission(
            admission,
            inputs,
            checkpoint,
            panel,
            expected_serving_identity=serving_identity.serving_artifact(args),
        )


def test_failed_parity_rejects(tmp_path: Path, monkeypatch) -> None:
    admission, checkpoint, inputs, panel, args = _fixture(tmp_path, monkeypatch)
    parity = json.loads((inputs / "parity-output-manifest.json").read_text())
    precision = parity["success_evidence"]["selected_v13_precision_receipt"]
    parity["outcome"] = "failure"
    with pytest.raises(ValueError, match="parity manifest differs"):
        trained.validate_parity_manifest(
            parity,
            selected_step=SELECTED_STEP,
            checkpoint_sha256=precision["checkpoint_inventory_sha256"],
            precision_sha256=_canonical(precision),
        )


def test_empty_producer_log_requires_exact_empty_digest(
    tmp_path: Path, monkeypatch
) -> None:
    _admission, _checkpoint, inputs, _panel, _args = _fixture(tmp_path, monkeypatch)
    parity = json.loads((inputs / "parity-output-manifest.json").read_text())
    precision = parity["success_evidence"]["selected_v13_precision_receipt"]
    parity["files"]["parity.log"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="identity differs"):
        trained.validate_parity_manifest(
            parity,
            selected_step=SELECTED_STEP,
            checkpoint_sha256=precision["checkpoint_inventory_sha256"],
            precision_sha256=_canonical(precision),
        )


@pytest.mark.parametrize("field", ["checkpoint", "precision"])
def test_different_selection_or_precision_rejects(
    tmp_path: Path, monkeypatch, field: str
) -> None:
    _admission, _checkpoint, inputs, _panel, _args = _fixture(tmp_path, monkeypatch)
    parity = json.loads((inputs / "parity-output-manifest.json").read_text())
    precision = parity["success_evidence"]["selected_v13_precision_receipt"]
    selection = parity["success_evidence"]["selection_receipt"]["selection"]["selected"]
    key = {
        "checkpoint": "checkpoint_inventory_sha256",
        "precision": "serving_precision_receipt_sha256",
    }[field]
    selection[key] = "f" * 64
    with pytest.raises(ValueError, match="selection differs"):
        trained.validate_parity_manifest(
            parity,
            selected_step=SELECTED_STEP,
            checkpoint_sha256=precision["checkpoint_inventory_sha256"],
            precision_sha256=_canonical(precision),
        )


def test_native_exported_action_bytes_must_match(tmp_path: Path, monkeypatch) -> None:
    _admission, _checkpoint, inputs, _panel, _args = _fixture(tmp_path, monkeypatch)
    parity = json.loads((inputs / "parity-output-manifest.json").read_text())
    precision = parity["success_evidence"]["selected_v13_precision_receipt"]
    for row in parity["success_evidence"]["processes"]:
        row["actions"]["exported"] = {
            **row["actions"]["native"],
            "sha256": "e" * 64,
        }
    with pytest.raises(ValueError, match="cold process differs"):
        trained.validate_parity_manifest(
            parity,
            selected_step=SELECTED_STEP,
            checkpoint_sha256=precision["checkpoint_inventory_sha256"],
            precision_sha256=_canonical(precision),
        )


@pytest.mark.parametrize("field", ["schema", "status"])
def test_consistently_rehashed_invalid_precision_contract_rejects(
    tmp_path: Path, monkeypatch, field: str
) -> None:
    _admission, _checkpoint, inputs, _panel, _args = _fixture(tmp_path, monkeypatch)
    parity = json.loads((inputs / "parity-output-manifest.json").read_text())
    precision = parity["success_evidence"]["selected_v13_precision_receipt"]
    precision[field] = f"attacker.invalid-{field}.v1"
    digest = _canonical(precision)
    parity["success_evidence"]["selection_receipt"]["selection"]["selected"][
        "serving_precision_receipt_sha256"
    ] = digest
    with pytest.raises(ValueError, match="precision receipt differs"):
        trained.validate_parity_manifest(
            parity,
            selected_step=SELECTED_STEP,
            checkpoint_sha256=precision["checkpoint_inventory_sha256"],
            precision_sha256=digest,
        )


def test_selection_without_complete_four_candidate_set_rejects(
    tmp_path: Path, monkeypatch
) -> None:
    _admission, _checkpoint, inputs, _panel, _args = _fixture(tmp_path, monkeypatch)
    parity = json.loads((inputs / "parity-output-manifest.json").read_text())
    evidence = parity["success_evidence"]
    evidence["selection_receipt"]["selection"]["candidates"].pop()
    precision = evidence["selected_v13_precision_receipt"]
    with pytest.raises(ValueError, match="four-candidate selection differs"):
        trained.validate_parity_manifest(
            parity,
            selected_step=SELECTED_STEP,
            checkpoint_sha256=precision["checkpoint_inventory_sha256"],
            precision_sha256=_canonical(precision),
        )


def test_non_comet12_precision_leaf_count_rejects(tmp_path: Path, monkeypatch) -> None:
    _admission, _checkpoint, inputs, _panel, _args = _fixture(tmp_path, monkeypatch)
    parity = json.loads((inputs / "parity-output-manifest.json").read_text())
    precision = parity["success_evidence"]["selected_v13_precision_receipt"]
    precision["leaves"].pop()
    precision["leaf_count"] = len(precision["leaves"])
    precision["total_parameter_bytes"] = sum(
        row["bytes"] for row in precision["leaves"]
    )
    precision["cast_inventory_sha256"] = _canonical(precision["leaves"])
    with pytest.raises(ValueError, match="precision leaf count differs"):
        trained.validate_parity_manifest(
            parity,
            selected_step=SELECTED_STEP,
            checkpoint_sha256=precision["checkpoint_inventory_sha256"],
            precision_sha256=_canonical(precision),
        )


def test_consistently_rehashed_selected_reader_rejects(
    tmp_path: Path, monkeypatch
) -> None:
    admission_path, _checkpoint, inputs, _panel, _args = _fixture(tmp_path, monkeypatch)
    reader_path = inputs / "selected-reader.json"
    reader = json.loads(reader_path.read_text())
    reader["candidate_arm"] = "full_sft"
    _write(reader_path, reader)
    admission = json.loads(admission_path.read_text())
    admission["selected_reader"]["provider"] = _provider(reader_path, reader_path.name)
    parity = json.loads((inputs / "parity-output-manifest.json").read_text())
    with pytest.raises(ValueError, match="reader differs"):
        trained._validate_parity_originals(
            inputs, parity["files"], parity["success_evidence"], admission
        )


def test_milestone_cursor_and_provider_members_are_required(
    tmp_path: Path, monkeypatch
) -> None:
    _admission, _checkpoint, inputs, _panel, _args = _fixture(tmp_path, monkeypatch)
    manifest = json.loads(
        (inputs / "prepared-run/inputs/milestone_manifest.json").read_text()
    )
    receipt = json.loads(
        (inputs / "prepared-run/inputs/milestone_receipt.json").read_text()
    )
    receipt["cursor"]["batch_offset"] += 1
    row = manifest["files"]["milestone-receipt.json"]
    with pytest.raises(ValueError, match="receipt lineage differs"):
        trained_comet_producer._validate_milestone(
            manifest, receipt, row, SELECTED_STEP, "action_expert"
        )
    receipt["cursor"]["batch_offset"] -= 1
    manifest["files"].pop(
        next(name for name in manifest["files"] if name.startswith("checkpoint/"))
    )
    with pytest.raises(ValueError, match="provider member set differs"):
        trained_comet_producer._validate_milestone(
            manifest, receipt, row, SELECTED_STEP, "action_expert"
        )


def test_substitute_normalization_rejects_even_when_serving_tree_is_rehashed(
    tmp_path: Path, monkeypatch
) -> None:
    admission_path, checkpoint, inputs, panel, args = _fixture(tmp_path, monkeypatch)
    admission = json.loads(admission_path.read_text())
    norm = checkpoint / admission["normalization"]["path"]
    norm.write_text('{"substitute": true}\n')
    admission["normalization"]["provider"] = _provider(norm, "norm_stats.json")
    actual = {
        path.relative_to(checkpoint).as_posix(): file_identity(path)
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file()
    }
    admission["serving_tree_sha256"] = campaign.canonical_digest(actual)
    _write(admission_path, admission)
    policy = campaign.freeze_policy_identity(
        "dp4-action-expert-20k",
        {
            "checkpoint": file_identity(admission_path),
            "serving": serving_identity.serving_artifact(args),
        },
    )
    registry = [f"task-{index}" for index in range(100)]
    registry[1] = "picking_up_trash"
    panel = campaign.declare_panel(
        policy, registry, ["picking_up_trash"], "development"
    )
    with pytest.raises(ValueError, match="differs from selected milestone"):
        trained.validate_trained_comet_admission(
            admission_path,
            inputs,
            checkpoint,
            panel,
            expected_serving_identity=serving_identity.serving_artifact(args),
        )


@pytest.mark.parametrize(("field", "value"), [("profile", "full_sft"), ("task_id", 2)])
def test_admission_role_and_task_must_equal_producer_lineage(
    tmp_path: Path, monkeypatch, field: str, value: object
) -> None:
    admission_path, _checkpoint, inputs, _panel, _args = _fixture(tmp_path, monkeypatch)
    admission = json.loads(admission_path.read_text())
    admission[field] = value
    parity = json.loads((inputs / "parity-output-manifest.json").read_text())
    with pytest.raises(ValueError, match="runtime candidate|fixed observation"):
        trained._validate_parity_originals(
            inputs, parity["files"], parity["success_evidence"], admission
        )


def test_consistently_rehashed_result_claims_reject(
    tmp_path: Path, monkeypatch
) -> None:
    admission, _checkpoint, inputs, _panel, _args = _fixture(tmp_path, monkeypatch)
    result_path = inputs / "parity-result.json"
    result = json.loads(result_path.read_text())
    result["claims"]["rollout_quality"] = True
    _write(result_path, result)
    parity_path = inputs / "parity-output-manifest.json"
    parity = json.loads(parity_path.read_text())
    parity["files"]["parity-result.json"] = _provider(result_path, "parity-result.json")
    admitted = json.loads(admission.read_text())
    with pytest.raises(ValueError, match="parity result differs"):
        trained._validate_parity_originals(
            inputs, parity["files"], parity["success_evidence"], admitted
        )


def test_raw_action_archive_mutation_rejects(tmp_path: Path, monkeypatch) -> None:
    admission, checkpoint, inputs, panel, args = _fixture(tmp_path, monkeypatch)
    raw = inputs / "work/raw-actions-cold-a.npz"
    np.savez(
        raw,
        native=np.ones((32, 32), dtype=np.float32),
        exported=np.ones((32, 32), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="local bytes differ"):
        trained.validate_trained_comet_admission(
            admission,
            inputs,
            checkpoint,
            panel,
            expected_serving_identity=serving_identity.serving_artifact(args),
        )


def test_fixed_noise_original_is_provider_byte_bound(
    tmp_path: Path, monkeypatch
) -> None:
    admission, checkpoint, inputs, panel, args = _fixture(tmp_path, monkeypatch)
    schema = inputs / "prepared-run/inputs/fixed_input_schema.json"
    value = json.loads(schema.read_text())
    value["noise"]["sha256"] = "0" * 64
    _write(schema, value)
    with pytest.raises(ValueError, match="local bytes differ"):
        trained.validate_trained_comet_admission(
            admission,
            inputs,
            checkpoint,
            panel,
            expected_serving_identity=serving_identity.serving_artifact(args),
        )


def test_provider_original_ancestor_symlink_rejects(
    tmp_path: Path, monkeypatch
) -> None:
    admission, checkpoint, inputs, panel, args = _fixture(tmp_path, monkeypatch)
    prepared = inputs / "prepared-run"
    outside = tmp_path / "outside-prepared-run"
    shutil.move(prepared, outside)
    prepared.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="parent is absent or unsafe"):
        trained.validate_trained_comet_admission(
            admission,
            inputs,
            checkpoint,
            panel,
            expected_serving_identity=serving_identity.serving_artifact(args),
        )


def test_preclaim_records_actual_admission(tmp_path: Path, monkeypatch) -> None:
    admission, checkpoint, inputs, panel, args = _fixture(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    campaign_runner._trained_comet_preclaim(args, panel, workspace)
    assert args.policy_trained_execution_admission["selected_step"] == 20000
    assert (workspace / "trained-comet-admission.json").is_file()


def test_preclaim_rejects_missing_trained_input_root_before_workspace_write(
    tmp_path: Path, monkeypatch
) -> None:
    _admission, _checkpoint, _inputs, panel, args = _fixture(tmp_path, monkeypatch)
    args.policy_trained_input_root = None
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="provider-read input root"):
        campaign_runner._trained_comet_preclaim(args, panel, workspace)
    assert list(workspace.iterdir()) == []


def test_real_policy_command_qualifies_selected_export_interface(
    tmp_path: Path, monkeypatch
) -> None:
    admission, checkpoint, inputs, panel, args = _fixture(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    campaign_runner._trained_comet_preclaim(args, panel, workspace)
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    _write(
        source / "scripts/task_mapping.json",
        {"picking_up_trash": {"task_index": 1, "task": "pick up the trash"}},
    )
    args.policy_root = source
    args.policy_python = Path("/runtime/python")
    args.port = 8000
    plan = campaign_runner._managed_plan(panel, panel["cases"][0])
    output = tmp_path / "case"
    monkeypatch.setattr(trained_comet_policy, "verify_source", lambda _root: None)

    def qualify(command, *, cwd, check):
        assert cwd == source
        assert check is True
        parsed = native_comet_server.parser().parse_args(command[2:])
        expected = trained_comet_policy._expected_qualification(
            args.policy_trained_execution_admission,
            plan["cases"][0],
            parsed.case_seed,
        )
        expected["initial_rng_sha256"] = "d" * 64
        expected["process_identity_sha256"] = _canonical(
            {
                "schema": "npa.behavior.comet-native-serving-process-identity.v1",
                "case_id": expected["case_id"],
                "task": expected["task"],
                "instance_id": expected["instance_id"],
                "rollout_id": expected["rollout_id"],
                "case_seed": expected["case_seed"],
                "checkpoint_content_sha256": expected["checkpoint_content_sha256"],
                "rng_contract_sha256": expected["rng_contract_sha256"],
                "trace_configuration_sha256": expected["trace_configuration_sha256"],
                "initial_rng_sha256": expected["initial_rng_sha256"],
            }
        )
        _write(parsed.qualification_output, expected)

    monkeypatch.setattr(trained_comet_policy.subprocess, "run", qualify)
    command = trained_comet_policy.prepare_policy(args, plan, output)

    parsed = native_comet_server.parser().parse_args(command[2:])
    assert parsed.policy_label == "comet-trained-selected"
    assert parsed.checkpoint == checkpoint
    assert parsed.manager_step == 20000
    assert parsed.case_id == panel["cases"][0]["case_id"]
    assert (output / "policy-provenance.json").is_file()


def test_train_panel_cannot_relabel_selected_export(
    tmp_path: Path, monkeypatch
) -> None:
    admission, checkpoint, inputs, panel, args = _fixture(tmp_path, monkeypatch)
    train = {"schema": "npa.behavior.nonreporting-train-panel.v1"}
    with pytest.raises(ValueError, match="panel"):
        trained.validate_trained_comet_admission(
            admission,
            inputs,
            checkpoint,
            train,
            expected_serving_identity=serving_identity.serving_artifact(args),
        )


def test_selected_export_composes_with_distinct_train_recording_panel(
    tmp_path: Path, monkeypatch
) -> None:
    admission, checkpoint, inputs, _dev_panel, args = _fixture(tmp_path, monkeypatch)
    args.train_experience = True
    panel = _train_panel(args, admission)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    campaign_runner._trained_comet_preclaim(args, panel, workspace)
    assert args.policy_trained_execution_admission["selected_step"] == 20_000
    assert panel["cases"] == [
        {
            "split": "train",
            "task": "picking_up_trash",
            "instance_id": 0,
            "rollout_id": 0,
            "case_id": panel["cases"][0]["case_id"],
        }
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task_id", 2, "task ID differs"),
        ("model_prediction_horizon", 31, "action horizon differs"),
        ("executed_prefix", 31, "action horizon differs"),
        ("rng_contract", _artifact("f"), "RNG contract differs"),
        ("normalization", _artifact("0"), "normalization differs"),
    ],
)
def test_train_protocol_runtime_values_must_match_selected_admission(
    tmp_path: Path, monkeypatch, field: str, value: object, message: str
) -> None:
    admission, checkpoint, inputs, _dev_panel, args = _fixture(tmp_path, monkeypatch)
    args.train_experience = True
    panel = _train_panel(args, admission)
    protocol = copy.deepcopy(panel["protocol"])
    if field == "task_id":
        protocol["task_mapping"]["data_task_id"] = value
    elif field == "normalization":
        protocol["science_lineage"][field] = value
    else:
        protocol["evaluator_contract"][field] = value
        if field == "model_prediction_horizon":
            protocol["evaluator_contract"]["executed_prefix"] = value
    protocol = nonreporting_train.declare_train_protocol(
        protocol["task"],
        protocol["task_mapping"],
        protocol["prescribed_cases"],
        protocol["science_lineage"],
        protocol["evaluator_contract"],
    )
    changed = nonreporting_train.declare_train_panel(protocol, panel["policy_binding"])
    with pytest.raises(ValueError, match=message):
        trained.validate_trained_comet_admission(
            admission,
            inputs,
            checkpoint,
            changed,
            expected_serving_identity=serving_identity.serving_artifact(args),
        )


def test_trained_policy_stages_recorders_and_exact_train_config(
    tmp_path: Path, monkeypatch
) -> None:
    admission, checkpoint, inputs, _dev_panel, args = _fixture(tmp_path, monkeypatch)
    args.train_experience = True
    panel = _train_panel(args, admission)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    campaign_runner._trained_comet_preclaim(args, panel, workspace)
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    _write(
        source / "scripts/task_mapping.json",
        {"picking_up_trash": {"task_index": 1, "task": "pick up the trash"}},
    )
    args.policy_root = source
    args.policy_python = Path("/runtime/python")
    args.port = 8000
    plan = campaign_runner._managed_plan(panel, panel["cases"][0])
    output = tmp_path / "case"
    monkeypatch.setattr(trained_comet_policy, "verify_source", lambda _root: None)

    def qualify(command, *, cwd, check):
        parsed = native_comet_server.parser().parse_args(command[2:])
        expected = trained_comet_policy._expected_qualification(
            args.policy_trained_execution_admission,
            plan["cases"][0],
            parsed.case_seed,
        )
        expected["initial_rng_sha256"] = "d" * 64
        expected["process_identity_sha256"] = _canonical(
            {
                "schema": "npa.behavior.comet-native-serving-process-identity.v1",
                "case_id": expected["case_id"],
                "task": expected["task"],
                "instance_id": expected["instance_id"],
                "rollout_id": expected["rollout_id"],
                "case_seed": expected["case_seed"],
                "checkpoint_content_sha256": expected["checkpoint_content_sha256"],
                "rng_contract_sha256": expected["rng_contract_sha256"],
                "trace_configuration_sha256": expected["trace_configuration_sha256"],
                "initial_rng_sha256": expected["initial_rng_sha256"],
            }
        )
        _write(parsed.qualification_output, expected)

    monkeypatch.setattr(trained_comet_policy.subprocess, "run", qualify)
    command = trained_comet_policy.prepare_policy(args, plan, output)
    parsed = native_comet_server.parser().parse_args(command[2:])
    config = json.loads((output / "train-experience/config.json").read_text())

    assert parsed.train_experience_root == output / "train-experience"
    assert config["panel_sha256"] == panel["panel_id"]
    assert (
        config["checkpoint_sha256"]
        == args.policy_trained_execution_admission["serving_tree_sha256"]
    )
    assert (
        config["rng_contract_sha256"]
        == args.policy_trained_execution_admission["rng_contract"]["sha256"]
    )
    assert (output / "train_experience_evaluator.py").is_file()
    assert (output / "semantic_monitor/interface.py").is_file()


def test_trained_export_train_requires_recording(tmp_path: Path, monkeypatch) -> None:
    admission, _checkpoint, _inputs, _dev_panel, args = _fixture(tmp_path, monkeypatch)
    args.train_experience = True
    panel = _train_panel(args, admission)
    plan = campaign_runner._managed_plan(panel, panel["cases"][0])
    args.train_experience = False
    with pytest.raises(ValueError, match="requires experience recording"):
        trained_comet_policy._case(
            args,
            plan,
            {"task": "picking_up_trash"},
        )


def test_trained_train_admission_precedes_startup_and_case_claim(
    tmp_path: Path, monkeypatch
) -> None:
    admission, checkpoint, inputs, _dev_panel, args = _fixture(tmp_path, monkeypatch)
    args.train_experience = True
    args.policy_root = tmp_path / "source"
    args.policy_python = Path("/runtime/python")
    args.policy_checkpoint = checkpoint
    args.policy_archive = admission
    args.workspace = tmp_path / "workspace"
    args.workspace.mkdir()
    args.worker_index = 0
    args.output_path = "s3://fixture-bucket/train"
    args.worker_receipt_uri = "s3://fixture-bucket/train/worker.json"
    panel = _train_panel(args, admission)
    partition = nonreporting_train.partition_train_panel(panel, 1)
    events = []
    original = campaign_runner._trained_comet_preclaim

    def preclaim(*values):
        original(*values)
        events.append("admission")

    def evaluator_preclaim(*_values):
        events.append("evaluator-contract")

    def startup(*_values):
        events.append("startup")
        raise RuntimeError("stop before claim")

    monkeypatch.setattr(campaign_runner, "_trained_comet_preclaim", preclaim)
    monkeypatch.setattr(
        campaign_runner, "_train_evaluator_preclaim", evaluator_preclaim
    )
    monkeypatch.setattr(campaign_runner, "_prepare_worker_startup", startup)
    monkeypatch.setattr(campaign_runner, "_publish_worker_provenance", lambda *_: None)
    monkeypatch.setattr(
        campaign_runner,
        "CaseStore",
        lambda *_: pytest.fail("case store opened before TRAIN admission"),
    )

    with pytest.raises(RuntimeError, match="stop before claim"):
        campaign_runner._execute_partition(
            args, panel, partition, object(), args.workspace
        )
    assert events == ["evaluator-contract", "admission", "startup"]


def test_train_evaluator_source_rejects_before_policy_or_case_side_effects(
    tmp_path: Path, monkeypatch
) -> None:
    admission, checkpoint, _inputs, _dev_panel, args = _fixture(tmp_path, monkeypatch)
    args.train_experience = True
    args.policy_checkpoint = checkpoint
    args.workspace = tmp_path / "workspace"
    args.workspace.mkdir()
    args.upstream_root = tmp_path / "upstream"
    args.worker_index = 0

    for name in ("_trained_comet_preclaim", "_prepare_worker_startup", "CaseStore"):
        monkeypatch.setattr(
            campaign_runner,
            name,
            lambda *_args, gate=name: pytest.fail(
                f"{gate} ran before source rejection"
            ),
        )

    invalid_wrapper = _train_panel(args, admission)
    invalid_partition = nonreporting_train.partition_train_panel(invalid_wrapper, 1)
    with pytest.raises(ValueError, match="official evaluator wrapper"):
        campaign_runner._execute_partition(
            args, invalid_wrapper, invalid_partition, object(), args.workspace
        )

    robot = args.upstream_root / nonreporting_train.EVAL_DIRECTORY / "r1pro.yaml"
    robot.parent.mkdir(parents=True)
    robot.write_bytes(b"wrong robot configuration\n")
    monkeypatch.setattr(nonreporting_train, "verify_upstream", lambda *_: None)
    wrong_robot = _train_panel(
        args,
        admission,
        wrapper=nonreporting_train.WRAPPER,
        robot_config=_artifact("d"),
    )
    wrong_partition = nonreporting_train.partition_train_panel(wrong_robot, 1)
    with pytest.raises(ValueError, match="robot configuration bytes differ"):
        campaign_runner._execute_partition(
            args, wrong_robot, wrong_partition, object(), args.workspace
        )


def test_trained_train_without_recording_rejects_before_any_runtime_gate(
    tmp_path: Path, monkeypatch
) -> None:
    admission, checkpoint, inputs, _dev_panel, args = _fixture(tmp_path, monkeypatch)
    args.train_experience = True
    panel = _train_panel(args, admission)
    args.train_experience = False
    args.workspace = tmp_path / "workspace"
    args.workspace.mkdir()
    partition = nonreporting_train.partition_train_panel(panel, 1)
    monkeypatch.setattr(
        campaign_runner,
        "_validate_worker_startup_binding",
        lambda *_: pytest.fail("startup binding checked before recording gate"),
    )
    monkeypatch.setattr(
        campaign_runner,
        "_trained_comet_preclaim",
        lambda *_: pytest.fail("admission checked before recording gate"),
    )
    monkeypatch.setattr(
        campaign_runner,
        "CaseStore",
        lambda *_: pytest.fail("case store opened before recording gate"),
    )
    with pytest.raises(ValueError, match="requires experience recording"):
        campaign_runner._execute_partition(
            args, panel, partition, object(), args.workspace
        )


def test_server_parser_preserves_train_default_and_accepts_selected_label() -> None:
    parser = native_comet_server.parser()
    common = [
        "--source-root",
        "/source",
        "--checkpoint",
        "/checkpoint",
        "--manager-step",
        "20000",
        "--asset-id",
        "behavior-1k/2025-challenge-demos",
        "--task-id",
        "1",
        "--task-name",
        "picking_up_trash",
        "--port",
        "8000",
        "--upstream-commit",
        "6cbf70b075816096e9be53958780769f3264d25d",
        "--case-id",
        "case",
        "--instance-id",
        "311",
        "--rollout-id",
        "0",
        "--case-seed",
        "1",
        "--checkpoint-sha256",
        "a" * 64,
        "--rng-contract-sha256",
        "b" * 64,
        "--trace-configuration-sha256",
        "c" * 64,
        "--process-receipt",
        "/receipt",
    ]
    assert parser.parse_args(common).policy_label == "comet-native-train"
    assert (
        parser.parse_args(
            [*common, "--policy-label", "comet-trained-selected"]
        ).policy_label
        == "comet-trained-selected"
    )
