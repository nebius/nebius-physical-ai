"""Validate selected DP4 producer receipts used by trained Comet serving."""

from __future__ import annotations

import hashlib
import json
import re

from .native_comet_checkpoint import canonical_provider_row
from .native_training_checkpoint import (
    canonical_member,
    validate_complete_checkpoint,
)

SOURCE_COMMIT = "4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


SCORER_STATIC_RECONSTRUCTION = {
    "batch_size": 256,
    "ema_decay": None,
    "fsdp_devices": 1,
    "loader_transport_bound_separately": True,
    "lr_schedule": {
        "decay_lr": 0.0,
        "decay_steps": 20_000,
        "peak_lr": 2.5e-6,
        "warmup_steps": 1_000,
    },
    "model": {
        "action_dim": 32,
        "action_expert_variant": "gemma_300m",
        "action_horizon": 32,
        "discrete_state_input": True,
        "dtype": "bfloat16",
        "max_token_len": 200,
        "paligemma_variant": "gemma_2b",
        "pcd": False,
        "pi05": True,
        "pointnet_variant": "pcd",
    },
    "native_sources": {
        "scripts/train.py": "d770752454ee221835b919918be6331583c9e9f5727aae7f22016cced2b84518",
        "src/openpi/models/pi0_config.py": "508aae13d4cd96959918dc5cdee44b8baf1ad262a2d14c3cd4c094fa2aed4445",
        "src/openpi/training/checkpoints.py": "a031c2621c9eb1270d68d362e5bc25b7eb05d475ae9a50234440173a0fb31ecc",
        "src/openpi/training/config.py": "f5a4a363b19e8956745962af494996efd51088f36692d0560c6b70b1d56c18dd",
        "src/openpi/training/data_loader.py": "5caf2dbf4ac311282cca8569394f762eeecb5aed41189cf21fb65cb337d9b4c3",
        "src/openpi/training/optimizer.py": "8db48c8bc3c0906394fb0642524f50668334a86fdabb5716eb30bf6f486d8dad",
        "src/openpi/training/utils.py": "b684e674343f86ef5e976f279a586d3fe2f885e901f396ac899ed136abc7eca6",
    },
    "optimizer": {
        "b1": 0.9,
        "b2": 0.95,
        "clip_gradient_norm": 1.0,
        "eps": 1e-8,
        "weight_decay": 1e-10,
    },
    "parent_profile": "comet12",
    "parent_role": "selected_future_training_parent_not_yet_trained",
    "seed": 42,
    "source_commit": SOURCE_COMMIT,
}


def _precision_rows(precision: object) -> list[dict]:
    rows = precision.get("leaves") if isinstance(precision, dict) else None
    if not isinstance(rows, list) or len(rows) != 51:
        raise ValueError("Selected parity precision leaf count differs")
    for ordinal, row in enumerate(rows):
        valid = (
            isinstance(row, dict)
            and row.get("ordinal") == ordinal
            and row.get("dtype") == "bfloat16"
            and isinstance(row.get("shape"), list)
            and all(type(size) is int and size >= 0 for size in row["shape"])
            and type(row.get("bytes")) is int
            and row["bytes"] >= 0
            and _SHA256.fullmatch(str(row.get("sha256"))) is not None
        )
        if not valid:
            raise ValueError("Selected parity precision leaf differs")
    return rows


def _validate_precision_receipt(
    precision: object, checkpoint_sha256: str, precision_sha256: str
) -> list[dict]:
    rows = _precision_rows(precision)
    expected_loader = (
        "config.model.load(model.restore_params(checkpoint/'params', "
        "dtype=jnp.bfloat16))"
    )
    valid = (
        _canonical_sha256(precision) == precision_sha256
        and precision.get("schema")
        == "npa.behavior.comet12-serving-bf16-parameter-inventory.v2"
        and precision.get("status")
        == "native_restore_params_bfloat16_cast_inventory_complete"
        and precision.get("loader_call") == expected_loader
        and precision.get("source_precision") == "float32_parameters"
        and precision.get("checkpoint_role") == "trained_milestone_full_state"
        and precision.get("deployed_precision") == "bfloat16"
        and precision.get("checkpoint_inventory_sha256") == checkpoint_sha256
        and precision.get("leaf_count") == 51
        and precision.get("total_parameter_bytes") == sum(row["bytes"] for row in rows)
        and precision.get("cast_inventory_sha256") == _canonical_sha256(rows)
    )
    if not valid:
        raise ValueError("Selected parity precision receipt differs")
    return rows


def _validate_cast_inventory(cast: object, precision: dict, rows: list[dict]) -> None:
    valid = (
        isinstance(cast, dict)
        and cast.get("schema") == "npa.behavior.comet12-bf16-serving-cast-inventory.v1"
        and cast.get("source_precision") == "float32_native_training_state"
        and cast.get("deployed_precision") == "bfloat16"
        and cast.get("leaves") == rows
        and cast.get("leaf_count") == 51
        and cast.get("total_bytes") == precision.get("total_parameter_bytes")
        and cast.get("canonical_sha256") == _canonical_sha256(rows)
    )
    if not valid:
        raise ValueError("Selected parity BF16 cast inventory differs")


def _selected_candidate(selection: object) -> tuple[dict, list[dict]]:
    result = selection.get("selection") if isinstance(selection, dict) else None
    candidates = result.get("candidates") if isinstance(result, dict) else None
    chosen = result.get("selected") if isinstance(result, dict) else None
    valid = (
        selection.get("schema")
        == "npa.private.comet12-dp4-b200-parent-finalizer-adapter.v1"
        and selection.get("status")
        == "frozen_selection_completed_from_accepted_b200_parent"
        and result.get("schema")
        == "npa.behavior.comet12-dp4-four-candidate-selection.v1"
        and result.get("status") == "complete_train_only_dp4_selection"
        and result.get("candidate_set_complete") is True
        and result.get("development_or_report_used") is False
        and result.get("one_gpu_candidate_relabelled") is False
        and isinstance(candidates, list)
        and len(candidates) == 4
        and all(isinstance(row, dict) for row in candidates)
        and isinstance(chosen, dict)
        and candidates.count(chosen) == 1
    )
    if not valid:
        raise ValueError("Selected parity four-candidate selection differs")
    return chosen, candidates


def _validate_selected_candidate(
    selection: object, step: int, checkpoint_sha256: str, precision_sha256: str
) -> dict:
    chosen, candidates = _selected_candidate(selection)
    identities = {(row.get("arm"), row.get("logical_update")) for row in candidates}
    valid = (
        len(identities) == 4
        and isinstance(chosen.get("arm"), str)
        and bool(chosen["arm"])
        and chosen.get("logical_update") == step
        and chosen.get("eligible") is True
        and chosen.get("checkpoint_inventory_sha256") == checkpoint_sha256
        and chosen.get("serving_precision_receipt_sha256") == precision_sha256
        and set(chosen.get("reader", {})) == {"bytes", "sha256"}
        and type(chosen["reader"].get("bytes")) is int
        and chosen["reader"]["bytes"] > 0
        and _SHA256.fullmatch(str(chosen["reader"].get("sha256"))) is not None
    )
    if not valid:
        raise ValueError("Selected parity selected candidate differs")
    return chosen


def _validate_precision(
    evidence: dict, selected_step: int, checkpoint_sha256: str, precision_sha256: str
) -> None:
    precision = evidence.get("selected_v13_precision_receipt")
    rows = _validate_precision_receipt(precision, checkpoint_sha256, precision_sha256)
    _validate_cast_inventory(evidence.get("bf16_cast_inventory"), precision, rows)
    _validate_selected_candidate(
        evidence.get("selection_receipt"),
        selected_step,
        checkpoint_sha256,
        precision_sha256,
    )


def _validate_selection_and_score(
    selection: dict,
    score: dict,
    precision: dict,
    selected_step: int,
    checkpoint_sha256: str,
    precision_sha256: str,
) -> str:
    chosen = _validate_selected_candidate(
        selection, selected_step, checkpoint_sha256, precision_sha256
    )
    scored = score.get("score")
    if (
        score.get("schema") != "npa.behavior.comet12-native-scorer-process.v2"
        or score.get("status") != "native_bf16_shape1_jit_score_complete"
        or score.get("step") != selected_step
        or score.get("checkpoint_role") != "trained_milestone_full_state"
        or score.get("fp32_parameter_source_inventory_sha256") != checkpoint_sha256
        or score.get("precision") != precision
        or not isinstance(scored, dict)
        or scored.get("schema") != "npa.behavior.comet12-holdout-score-draft.v5"
        or scored.get("checkpoint_inventory_sha256") != checkpoint_sha256
        or scored.get("serving_precision_receipt_sha256") != precision_sha256
        or scored.get("step") != selected_step
        or scored.get("official_development_or_report_examples_used") is not False
        or scored.get("train") is not False
    ):
        raise ValueError("Selected parity producer score differs")
    _validate_score_payload(scored)
    return chosen["arm"]


def _validate_score_payload(value: dict) -> None:
    digest_fields = {
        "ordered_input_sha256",
        "rng_schedule_sha256",
        "sample_inventory_sha256",
        "scorer_contract_sha256",
    }
    rng = value.get("rng")
    valid = (
        all(_SHA256.fullmatch(str(value.get(field))) for field in digest_fields)
        and value.get("parameter_precision")
        == "native_serving_bfloat16_cast_from_fp32_parameters"
        and isinstance(value.get("process_identity"), str)
        and bool(value["process_identity"])
        and isinstance(value.get("tasks"), dict)
        and bool(value["tasks"])
        and isinstance(rng, dict)
        and rng.get("construction")
        == "jax.random.fold_in(base_key, global_example_ordinal)"
        and type(rng.get("example_count")) is int
        and rng["example_count"] > 0
        and type(rng.get("seed")) is int
    )
    if not valid:
        raise ValueError("Selected parity score payload differs")


def _full_state_inventory(value: object) -> dict:
    rows = value.get("files") if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Selected milestone full-state inventory differs")
    paths = [row.get("path") for row in rows if isinstance(row, dict)]
    valid = all(
        set(row) == {"path", "bytes", "sha256", "mode"}
        and isinstance(row["path"], str)
        and canonical_member(row["path"]) == row["path"]
        and type(row["bytes"]) is int
        and row["bytes"] >= 0
        and _SHA256.fullmatch(str(row["sha256"])) is not None
        and isinstance(row["mode"], str)
        and row["mode"].startswith("0o")
        for row in rows
        if isinstance(row, dict)
    )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    aggregates = (
        value.get("file_count") == len(rows)
        and value.get("total_bytes") == sum(row["bytes"] for row in rows)
        and value.get("max_member_bytes") == max(row["bytes"] for row in rows)
        and value.get("content_sha256") == hashlib.sha256(encoded).hexdigest()
    )
    if (
        not valid
        or len(paths) != len(rows)
        or len(set(paths)) != len(paths)
        or not aggregates
    ):
        raise ValueError("Selected milestone full-state inventory differs")
    return value


def _parameter_inventory(value: object, leaves: int, label: str) -> dict:
    required = {"leaf_count", "total_elements", "total_bytes", "content_sha256"}
    valid = (
        isinstance(value, dict)
        and set(value) == required
        and value.get("leaf_count") == leaves
        and type(value.get("total_elements")) is int
        and value["total_elements"] >= 0
        and type(value.get("total_bytes")) is int
        and value["total_bytes"] >= 0
        and _SHA256.fullmatch(str(value.get("content_sha256"))) is not None
    )
    if not valid:
        raise ValueError(f"Selected milestone {label} inventory differs")
    return value


def _optimizer_inventory(value: object, leaves: int, label: str) -> None:
    valid = (
        isinstance(value, dict)
        and set(value) == {"leaf_count", "content_sha256"}
        and value.get("leaf_count") == leaves
        and _SHA256.fullmatch(str(value.get("content_sha256"))) is not None
    )
    if not valid:
        raise ValueError(f"Selected milestone {label} inventory differs")


def _validate_state_contract(value: object, step: int, profile: str) -> None:
    required = {
        "train_state_step",
        "inventories",
        "optimizer_scalar_progress",
        "optimizer_paths",
        "all_params_fp32",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Selected milestone state contract differs")
    counts = {"action_expert": (19, 32), "full_sft": (51, 0)}
    if profile not in counts:
        raise ValueError("Selected milestone profile differs")
    trainable_count, frozen_count = counts[profile]
    inventories = value.get("inventories")
    names = {
        "all_params",
        "trainable_params",
        "frozen_params",
        "optimizer",
        "adamw_mu",
        "adamw_nu",
    }
    if not isinstance(inventories, dict) or set(inventories) != names:
        raise ValueError("Selected milestone state inventory set differs")
    all_params = _parameter_inventory(inventories["all_params"], 51, "all params")
    trainable = _parameter_inventory(
        inventories["trainable_params"], trainable_count, "trainable"
    )
    frozen = _parameter_inventory(inventories["frozen_params"], frozen_count, "frozen")
    _validate_state_totals(value, step, all_params, trainable, frozen)
    _optimizer_inventory(inventories["optimizer"], 2 * trainable_count + 2, "optimizer")
    _optimizer_inventory(inventories["adamw_mu"], trainable_count, "AdamW mu")
    _optimizer_inventory(inventories["adamw_nu"], trainable_count, "AdamW nu")


def _validate_state_totals(value, step, all_params, trainable, frozen) -> None:
    valid = (
        value.get("train_state_step") == step
        and value.get("all_params_fp32") is True
        and all_params["total_bytes"]
        == trainable["total_bytes"] + frozen["total_bytes"]
        and all_params["total_elements"]
        == trainable["total_elements"] + frozen["total_elements"]
        and value.get("optimizer_scalar_progress")
        == {"[1][0].count": step, "[1][2].count": step}
        and set(value.get("optimizer_paths", {})) == {"bytes", "sha256"}
        and type(value["optimizer_paths"].get("bytes")) is int
        and value["optimizer_paths"]["bytes"] >= 0
        and _SHA256.fullmatch(str(value["optimizer_paths"].get("sha256"))) is not None
    )
    if not valid:
        raise ValueError("Selected milestone state totals differ")


def _validate_native_layout(checkpoint: dict, manager_step: int) -> None:
    validate_complete_checkpoint(checkpoint, manager_step)
    paths = {row["path"] for row in checkpoint["files"]}
    if any(not path.startswith(f"{manager_step}/") for path in paths):
        raise ValueError("Selected milestone checkpoint root differs")
    required = {f"{manager_step}/_CHECKPOINT_METADATA"}
    for subtree in ("train_state", "params"):
        root = f"{manager_step}/{subtree}"
        required |= {
            f"{root}/_METADATA",
            f"{root}/_sharding",
            f"{root}/manifest.ocdbt",
            f"{root}/array_metadatas/process_0",
            f"{root}/ocdbt.process_0/manifest.ocdbt",
        }
        if not any(path.startswith(f"{root}/d/") for path in paths):
            raise ValueError("Selected milestone checkpoint data differs")
        if not any(path.startswith(f"{root}/ocdbt.process_0/d/") for path in paths):
            raise ValueError("Selected milestone checkpoint data differs")
    if not required <= paths:
        raise ValueError("Selected milestone checkpoint metadata differs")


def _validate_milestone_members(
    manifest: dict, checkpoint: dict, receipt_row: dict
) -> None:
    files = manifest.get("files")
    expected = {"milestone-receipt.json"} | {
        f"checkpoint/{row['path']}" for row in checkpoint["files"]
    }
    if not isinstance(files, dict) or set(files) != expected:
        raise ValueError("Selected milestone provider member set differs")
    checkpoint_rows = {row["path"]: row for row in checkpoint["files"]}
    for name, provider in files.items():
        canonical_provider_row(provider, f"selected milestone {name}")
        source = (
            receipt_row
            if name == "milestone-receipt.json"
            else checkpoint_rows[name.removeprefix("checkpoint/")]
        )
        if any(provider.get(key) != source.get(key) for key in ("bytes", "sha256")):
            raise ValueError("Selected milestone provider identity differs")
    prefixes = {row["uri"].removesuffix("/" + name) for name, row in files.items()}
    if len(prefixes) != 1:
        raise ValueError("Selected milestone provider prefix differs")


def _normalized_serving(serving: object, checkpoint: dict, manager_step: int) -> dict:
    root = f"{manager_step}/params"
    rows = serving.get("files") if isinstance(serving, dict) else None
    checkpoint_rows = {row["path"]: row for row in checkpoint["files"]}
    valid = (
        isinstance(rows, list)
        and bool(rows)
        and serving.get("root") == root
        and serving.get("dtype") == "float32_native_training_state"
        and serving.get("bf16_serving_derivative_created") is False
        and serving.get("file_count") == len(rows)
        and serving.get("total_bytes") == sum(row.get("bytes", -1) for row in rows)
        and all(checkpoint_rows.get(row.get("path")) == row for row in rows)
    )
    if not valid:
        raise ValueError("Selected milestone serving inventory differs")
    return {
        "schema": "npa.behavior.orbax-params-inventory.v1",
        "file_count": len(rows),
        "total_bytes": serving["total_bytes"],
        "files": [
            {
                "path": "params/" + row["path"].removeprefix(root + "/"),
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
            for row in rows
        ],
    }


def _validate_milestone(manifest, receipt, receipt_row, step, profile) -> dict:
    manager_step = step - 1
    expected = {
        "schema": "npa.behavior.comet12-native-training-milestone.v2",
        "status": "complete_full_state_and_serving_params_provider_readback",
        "outcome": "success",
        "logical_update_count": step,
        "manager_step": manager_step,
        "full_state_resume_ready": True,
        "serving_export_qualified": False,
        "scoring_executed": False,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Selected milestone manifest differs")
    _validate_milestone_receipt(receipt, step, manager_step, profile)
    checkpoint = _full_state_inventory(receipt.get("checkpoint"))
    _validate_native_layout(checkpoint, manager_step)
    if (
        manifest.get("checkpoint") != checkpoint
        or manifest.get("cursor") != receipt["cursor"]
    ):
        raise ValueError("Selected milestone manifest lineage differs")
    serving = _normalized_serving(
        receipt.get("serving_params"), checkpoint, manager_step
    )
    if manifest.get("serving_params") != receipt.get("serving_params"):
        raise ValueError("Selected milestone manifest serving differs")
    _validate_milestone_members(manifest, checkpoint, receipt_row)
    return {"checkpoint": checkpoint, "serving": serving}


def _validate_milestone_receipt(receipt, step, manager_step, profile) -> None:
    cursor = receipt.get("cursor") if isinstance(receipt, dict) else None
    progress = receipt.get("progress") if isinstance(receipt, dict) else None
    valid = (
        receipt.get("arm") == profile
        and receipt.get("logical_update_count") == step
        and receipt.get("manager_step") == manager_step
        and receipt.get("train_state_step") == step
        and receipt.get("start_mode") in {"boundary", "milestone_15000"}
        and isinstance(cursor, dict)
        and set(cursor) == {"epoch", "batch_offset", "committed_global_batch"}
        and cursor.get("epoch") == step // 3729
        and cursor.get("batch_offset") == step % 3729
        and cursor.get("committed_global_batch") == step
        and isinstance(progress, dict)
        and progress.get("last_logical_update") == step
    )
    if not valid:
        raise ValueError("Selected milestone receipt lineage differs")
    _validate_state_contract(receipt.get("state_contract"), step, profile)
