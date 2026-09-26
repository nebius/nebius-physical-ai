"""Validate one provider-read native Comet checkpoint for TRAIN serving."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit

from .campaign import validate_policy_identity
from .native_training_checkpoint import (
    checkpoint_inventory,
    file_identity,
    validate_complete_checkpoint,
)

VERIFIED_SCHEMA = "npa.behavior.comet-native-verified-checkpoint.v1"
BINDING_SCHEMA = "npa.behavior.comet-native-policy-binding.v1"
ARCHITECTURE_SCHEMA = "npa.behavior.comet-native-architecture.v1"
PROBE_SCHEMA = "npa.behavior.comet-native-restore-qualification.v1"
BRIDGE_EXECUTION_SCHEMA = "npa.behavior.comet-native-checkpoint-bridge-execution.v1"
BRIDGE_TERMINAL_SCHEMA = "npa.behavior.comet-native-checkpoint-bridge-terminal.v1"
SUPPORTED_PROFILES = {"action_expert", "full_sft"}

_PROTOCOL_ARTIFACTS = {
    "source_manifest": ("task_mapping", "source_manifest"),
    "split_manifest": ("task_mapping", "split_manifest"),
    "mapping_artifact": ("task_mapping", "mapping_artifact"),
    "task_registry": ("science_lineage", "task_registry"),
    "dataset": ("science_lineage", "dataset"),
    "dataset_view": ("science_lineage", "dataset_view"),
    "normalization": ("science_lineage", "normalization"),
    "tokenizer": ("science_lineage", "tokenizer"),
    "action_semantics": ("science_lineage", "action_semantics"),
    "argv_contract": ("evaluator_contract", "argv_contract"),
    "evaluator_source": ("evaluator_contract", "evaluator_source"),
    "controller_source": ("evaluator_contract", "controller_source"),
    "robot_config": ("evaluator_contract", "robot_config"),
    "rng_contract": ("evaluator_contract", "rng_contract"),
}
_NATIVE_ARTIFACTS = {
    "verified_checkpoint",
    "architecture_receipt",
    "restore_qualification",
    "bridge_manifest",
    "bridge_execution",
    "bridge_terminal",
    "training_terminal",
    "resume_receipt",
    "same_arm_qualification",
    "runtime_receipt",
    "source_receipt",
    "parent_receipt",
}
_LINEAGE_ARTIFACTS = _NATIVE_ARTIFACTS - {
    "verified_checkpoint",
    "architecture_receipt",
    "restore_qualification",
}


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _identity(value: object, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"bytes", "sha256"}:
        raise ValueError(f"{label} identity differs")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise ValueError(f"{label} identity differs")
    return value


def canonical_provider_row(value: object, label: str) -> dict:
    """Require one canonical provider-read S3 object row.

    Args:
        value: Candidate object row.
        label: Name used in validation errors.
    Returns:
        The unchanged validated row.
    Raises:
        ValueError: The URI, identity, or readback claim is invalid.
    """
    if not isinstance(value, dict) or set(value) != {
        "uri",
        "bytes",
        "sha256",
        "provider_readback",
    }:
        raise ValueError(f"{label} provider row differs")
    parsed = urlsplit(str(value.get("uri", "")))
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or "\\" in parsed.path
        or "%" in parsed.path
        or "//" in parsed.path
    ):
        raise ValueError(f"{label} provider URI differs")
    path = PurePosixPath(parsed.path.removeprefix("/"))
    if (
        not path.parts
        or parsed.path != "/" + path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} provider URI differs")
    _identity({key: value[key] for key in ("bytes", "sha256")}, label)
    if value.get("provider_readback") is not True:
        raise ValueError(f"{label} provider readback differs")
    return value


def canonical_asset_id(value: object) -> str:
    """Require a canonical relative OpenPI asset identifier.

    OpenPI dataset asset IDs may contain multiple path components, for example
    ``behavior-1k/2025-challenge-demos``.  The identifier remains confined below
    the checkpoint's ``assets`` directory.
    """
    if not isinstance(value, str) or not value or "\\" in value or "%" in value:
        raise ValueError("native checkpoint asset ID differs")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", part) is None
            for part in path.parts
        )
    ):
        raise ValueError("native checkpoint asset ID differs")
    return value


def _path_list(value: object, label: str, *, allow_empty: bool = False) -> list[str]:
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
        or value != sorted(value)
    ):
        raise ValueError(f"{label} paths differ")
    return value


def _leaf_inventory(value: object, label: str) -> dict[str, dict]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} leaf inventory differs")
    for path, row in value.items():
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(row, dict)
            or set(row) != {"shape", "dtype", "elements", "bytes", "sha256"}
            or not isinstance(row["shape"], list)
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in row["shape"]
            )
            or row["dtype"] not in {"float32", "int32"}
            or isinstance(row["elements"], bool)
            or not isinstance(row["elements"], int)
            or row["elements"] <= 0
            or isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or row["bytes"] != row["elements"] * 4
            or re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])) is None
        ):
            raise ValueError(f"{label} leaf inventory differs")
        elements = 1
        for dimension in row["shape"]:
            elements *= dimension
        if row["elements"] != elements:
            raise ValueError(f"{label} leaf shape differs")
    return value


def validate_architecture(value: object) -> dict:
    """Validate an exact dynamic native leaf partition.

    Args:
        value: Architecture observation produced by checkpoint qualification.
    Returns:
        The unchanged validated architecture.
    Raises:
        ValueError: Paths, shapes, dtypes, values, or coverage differ.
    """
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "profile",
        "training_dtype",
        "parameter_paths",
        "optimizer_paths",
        "parameter_leaves",
        "optimizer_leaves",
        "partition",
        "partition_sha256",
    }:
        raise ValueError("native architecture contract differs")
    profile = value.get("profile")
    if value.get("schema") != ARCHITECTURE_SCHEMA or profile not in SUPPORTED_PROFILES:
        raise ValueError("native architecture profile differs")
    if value.get("training_dtype") != "float32":
        raise ValueError("native architecture training dtype differs")
    parameter_leaves = _leaf_inventory(value.get("parameter_leaves"), "parameter")
    optimizer_leaves = _leaf_inventory(value.get("optimizer_leaves"), "optimizer")
    parameters = _path_list(value.get("parameter_paths"), "parameter")
    optimizers = _path_list(value.get("optimizer_paths"), "optimizer")
    if parameters != sorted(parameter_leaves) or optimizers != sorted(optimizer_leaves):
        raise ValueError("native architecture leaf path inventory differs")
    if any(
        row["dtype"] != value["training_dtype"] for row in parameter_leaves.values()
    ):
        raise ValueError("native architecture parameter dtype differs")
    partition = value.get("partition")
    if not isinstance(partition, dict) or set(partition) != {
        "trainable_paths",
        "frozen_paths",
        "optimizer_paths",
        "trainable_leaves",
        "frozen_leaves",
        "optimizer_leaves",
    }:
        raise ValueError("native architecture partition differs")
    trainable = _path_list(partition.get("trainable_paths"), "trainable")
    frozen = _path_list(partition.get("frozen_paths"), "frozen", allow_empty=True)
    declared_optimizer = _path_list(partition.get("optimizer_paths"), "optimizer")
    if (
        set(trainable) & set(frozen)
        or sorted(trainable + frozen) != parameters
        or declared_optimizer != optimizers
        or partition.get("trainable_leaves") != len(trainable)
        or partition.get("frozen_leaves") != len(frozen)
        or partition.get("optimizer_leaves") != len(optimizers)
    ):
        raise ValueError("native architecture leaf coverage differs")
    expected = _canonical_digest(
        {
            "profile": profile,
            "training_dtype": "float32",
            "parameter_paths": parameters,
            "optimizer_paths": optimizers,
            "parameter_leaves": parameter_leaves,
            "optimizer_leaves": optimizer_leaves,
            "partition": partition,
        }
    )
    if value.get("partition_sha256") != expected:
        raise ValueError("native architecture partition digest differs")
    return value


def _validate_probe(value: object, verified: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "status",
        "logical_update",
        "manager_step",
        "checkpoint_content_sha256",
        "architecture_partition_sha256",
        "optimizer_train_state_restored",
        "serving_params_restored",
        "finite_loss",
        "finite_grad_norm",
        "update_discarded",
        "provider_execution",
    }:
        raise ValueError("native restore qualification differs")
    for name in ("finite_loss", "finite_grad_norm"):
        metric = value.get(name)
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ValueError("native restore qualification metric differs")
        if not (-float("inf") < float(metric) < float("inf")):
            raise ValueError("native restore qualification metric is nonfinite")
    if (
        value.get("schema") != PROBE_SCHEMA
        or value.get("status") != "actual_restore_and_discarded_finite_probe"
        or value.get("logical_update") != verified["logical_update"]
        or value.get("manager_step") != verified["manager_step"]
        or value.get("checkpoint_content_sha256")
        != verified["checkpoint"]["content_sha256"]
        or value.get("architecture_partition_sha256")
        != verified["architecture"]["partition_sha256"]
        or value.get("optimizer_train_state_restored") is not True
        or value.get("serving_params_restored") is not True
        or value.get("update_discarded") is not True
    ):
        raise ValueError("native restore qualification differs")
    canonical_provider_row(value.get("provider_execution"), "probe execution")
    return value


def _provider_rows_equal(
    value: object, names: set[str], artifacts: dict, label: str
) -> dict:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError(f"{label} provider rows differ")
    for name, row in value.items():
        canonical_provider_row(row, f"{label} {name}")
        if row != artifacts[name]["provider"]:
            raise ValueError(f"{label} provider row differs: {name}")
    return value


def _validate_bridge_execution(value: object, verified: dict, artifacts: dict) -> dict:
    inputs = _LINEAGE_ARTIFACTS - {"bridge_execution", "bridge_terminal"}
    required = {
        "schema",
        "status",
        "profile",
        "logical_update",
        "manager_step",
        "cursor",
        "checkpoint_content_sha256",
        "architecture_partition_sha256",
        "input_provider_rows",
        "claims",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("native checkpoint bridge execution differs")
    if (
        value.get("schema") != BRIDGE_EXECUTION_SCHEMA
        or value.get("status") != "actual_full_state_restore_and_probe_completed"
        or value.get("profile") != verified["profile"]
        or value.get("logical_update") != verified["logical_update"]
        or value.get("manager_step") != verified["manager_step"]
        or value.get("cursor") != verified["cursor"]
        or value.get("checkpoint_content_sha256")
        != verified["checkpoint"]["content_sha256"]
        or value.get("architecture_partition_sha256")
        != verified["architecture"]["partition_sha256"]
        or value.get("claims")
        != {
            "development_or_report_read": False,
            "outcomes_read": False,
            "score_or_selection_executed": False,
        }
    ):
        raise ValueError("native checkpoint bridge execution differs")
    _provider_rows_equal(
        value.get("input_provider_rows"), inputs, artifacts, "bridge execution input"
    )
    return value


def _validate_bridge_terminal(value: object, verified: dict, artifacts: dict) -> dict:
    required = {
        "schema",
        "status",
        "bridge_execution",
        "architecture_receipt",
        "restore_qualification",
        "checkpoint_content_sha256",
        "member_provider_rows",
        "claims",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("native checkpoint bridge terminal differs")
    for name in ("bridge_execution", "architecture_receipt", "restore_qualification"):
        canonical_provider_row(value.get(name), f"bridge terminal {name}")
        if value[name] != artifacts[name]["provider"]:
            raise ValueError(f"bridge terminal provider row differs: {name}")
    if (
        value.get("schema") != BRIDGE_TERMINAL_SCHEMA
        or value.get("status") != "qualified_checkpoint_originals_provider_readback"
        or value.get("checkpoint_content_sha256")
        != verified["checkpoint"]["content_sha256"]
        or value.get("member_provider_rows") != verified["member_provider_rows"]
        or value.get("claims")
        != {
            "development_or_report_read": False,
            "outcomes_read": False,
            "score_or_selection_executed": False,
        }
    ):
        raise ValueError("native checkpoint bridge terminal differs")
    return value


def validate_verified_checkpoint(value: object) -> dict:
    """Validate one observed native checkpoint qualification record.

    Args:
        value: Generic verified-checkpoint record.
    Returns:
        The unchanged validated record.
    Raises:
        ValueError: Checkpoint, state, partition, or provider lineage differs.
    """
    required = {
        "schema",
        "status",
        "profile",
        "logical_update",
        "manager_step",
        "cursor",
        "checkpoint_prefix",
        "checkpoint",
        "member_provider_rows",
        "serving_assets",
        "architecture",
        "frozen_parent_equality",
        "optimizer_train_state_observation",
        "serving_params_observation",
        "discarded_finite_probe",
        "lineage",
        "claims",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("verified native checkpoint fields differ")
    if (
        value.get("schema") != VERIFIED_SCHEMA
        or value.get("status") != "actual_restore_partition_and_finite_probe_verified"
        or value.get("profile") not in SUPPORTED_PROFILES
    ):
        raise ValueError("verified native checkpoint contract differs")
    step = value.get("logical_update")
    manager = value.get("manager_step")
    cursor = value.get("cursor")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step <= 0
        or manager != step - 1
        or not isinstance(cursor, dict)
        or set(cursor) != {"logical_update", "manager_step", "next_manifest_position"}
        or cursor.get("logical_update") != step
        or cursor.get("manager_step") != manager
        or isinstance(cursor.get("next_manifest_position"), bool)
        or not isinstance(cursor.get("next_manifest_position"), int)
        or cursor["next_manifest_position"] <= 0
    ):
        raise ValueError("verified native checkpoint chronology differs")
    architecture = validate_architecture(value.get("architecture"))
    if architecture["profile"] != value["profile"]:
        raise ValueError("verified native checkpoint profile differs")
    checkpoint = value.get("checkpoint")
    validate_complete_checkpoint(checkpoint, manager)
    rows = checkpoint.get("files") if isinstance(checkpoint, dict) else None
    provider_rows = value.get("member_provider_rows")
    prefix = value.get("checkpoint_prefix")
    if (
        not isinstance(rows, list)
        or not rows
        or not isinstance(provider_rows, list)
        or len(provider_rows) != len(rows)
        or not isinstance(prefix, str)
    ):
        raise ValueError("verified native checkpoint members differ")
    serving_assets = value.get("serving_assets")
    if not isinstance(serving_assets, dict) or set(serving_assets) != {
        "asset_id",
        "norm_stats_path",
    }:
        raise ValueError("verified native checkpoint serving assets differ")
    asset_id = canonical_asset_id(serving_assets.get("asset_id"))
    expected_norm = f"{manager}/assets/{asset_id}/norm_stats.json"
    if serving_assets.get("norm_stats_path") != expected_norm or expected_norm not in {
        row["path"] for row in rows
    }:
        raise ValueError("verified native checkpoint serving assets differ")
    canonical_provider_row(
        {
            "uri": prefix + "/sentinel",
            "bytes": 1,
            "sha256": "0" * 64,
            "provider_readback": True,
        },
        "checkpoint prefix",
    )
    for inventory_row, provider in zip(rows, provider_rows, strict=True):
        canonical_provider_row(provider, "checkpoint member")
        if provider != {
            "uri": f"{prefix}/{inventory_row['path']}",
            "bytes": inventory_row["bytes"],
            "sha256": inventory_row["sha256"],
            "provider_readback": True,
        }:
            raise ValueError("verified native checkpoint provider member differs")
    frozen = architecture["partition"]["frozen_paths"]
    equality = value.get("frozen_parent_equality")
    if not isinstance(equality, dict) or sorted(equality) != frozen:
        raise ValueError("frozen parent equality coverage differs")
    for path, row in equality.items():
        if (
            not isinstance(row, dict)
            or set(row) != {"parent_sha256", "restored_sha256"}
            or row["parent_sha256"] != row["restored_sha256"]
            or row["restored_sha256"]
            != architecture["parameter_leaves"][path]["sha256"]
            or re.fullmatch(r"[0-9a-f]{64}", str(row["parent_sha256"])) is None
        ):
            raise ValueError(f"frozen parent value differs: {path}")
    state = value.get("optimizer_train_state_observation")
    serving = value.get("serving_params_observation")
    if state != {
        "manager_step": manager,
        "params_restored": True,
        "optimizer_state_restored": True,
        "state_step": step,
        "training_dtype": architecture["training_dtype"],
    }:
        raise ValueError("optimizer/TrainState observation differs")
    if serving != {
        "manager_step": manager,
        "params_restored": True,
        "training_dtype": architecture["training_dtype"],
    }:
        raise ValueError("serving parameter observation differs")
    probe = _validate_probe(value.get("discarded_finite_probe"), value)
    lineage = value.get("lineage")
    if not isinstance(lineage, dict) or set(lineage) != _LINEAGE_ARTIFACTS:
        raise ValueError("verified native checkpoint lineage differs")
    for name, row in lineage.items():
        canonical_provider_row(row, name)
    if probe["provider_execution"] != lineage["bridge_execution"]:
        raise ValueError("native restore probe execution lineage differs")
    if value.get("claims") != {
        "development_or_report_read": False,
        "outcomes_read": False,
        "score_or_selection_executed": False,
        "public_milestone_history_claimed": False,
    }:
        raise ValueError("verified native checkpoint claims differ")
    return value


def _safe_local(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} local path differs")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"{label} local path differs")
    path = root.joinpath(*pure.parts)
    current = root
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} local artifact differs")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"{label} local artifact differs")
    return path


def _artifact_identity(panel: dict, path: tuple[str, str]) -> dict:
    return panel["protocol"][path[0]][path[1]]


def _load_json(path: Path, label: str) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON differs")
    return value


def validate_native_train_admission(
    binding_path: Path,
    input_root: Path,
    checkpoint_root: Path,
    panel: dict,
    *,
    expected_verified_checkpoint: Path | None = None,
    expected_serving_identity: dict | None = None,
) -> dict:
    """Open and validate every native TRAIN input before worker side effects.

    Args:
        binding_path: Native policy binding JSON.
        input_root: Directory containing every declared input artifact.
        checkpoint_root: Locally restored complete checkpoint tree.
        panel: Frozen non-reporting TRAIN panel.
        expected_verified_checkpoint: Exact verified-checkpoint local path.
        expected_serving_identity: Serving code and configuration identity.
    Returns:
        Immutable admission record for the worker.
    Raises:
        ValueError: Any local byte, schema, provider row, or lineage join differs.
    """
    if input_root.is_symlink() or not input_root.is_dir():
        raise ValueError("native policy input root differs")
    input_root = input_root.resolve()
    binding = _load_json(binding_path, "native binding")
    if not isinstance(binding, dict) or set(binding) != {
        "schema",
        "policy_identity",
        "task",
        "task_id",
        "verified_checkpoint",
        "artifacts",
        "trace",
    }:
        raise ValueError("native policy binding fields differ")
    if binding.get("schema") != BINDING_SCHEMA:
        raise ValueError("native policy binding schema differs")
    policy = validate_policy_identity(binding.get("policy_identity"))
    if policy != panel.get("policy_binding"):
        raise ValueError("native policy identity differs from TRAIN panel")
    if binding.get("task") != panel["protocol"]["task"]:
        raise ValueError("native policy task differs")
    if (
        isinstance(binding.get("task_id"), bool)
        or not isinstance(binding.get("task_id"), int)
        or binding["task_id"] < 0
        or binding["task_id"] != panel["protocol"]["task_mapping"]["data_task_id"]
    ):
        raise ValueError("native policy task ID differs")
    artifacts = binding.get("artifacts")
    required = set(_PROTOCOL_ARTIFACTS) | _NATIVE_ARTIFACTS
    if not isinstance(artifacts, dict) or set(artifacts) != required:
        raise ValueError("native policy artifact set differs")
    loaded = {}
    for name, row in artifacts.items():
        if not isinstance(row, dict) or set(row) != {"path", "identity", "provider"}:
            raise ValueError(f"native artifact row differs: {name}")
        expected = _identity(row.get("identity"), name)
        provider = canonical_provider_row(row.get("provider"), name)
        if {key: provider[key] for key in ("bytes", "sha256")} != expected:
            raise ValueError(f"native artifact provider identity differs: {name}")
        path = _safe_local(input_root, row.get("path"), name)
        if file_identity(path) != expected:
            raise ValueError(f"native artifact local bytes differ: {name}")
        loaded[name] = path
    for name, declaration in _PROTOCOL_ARTIFACTS.items():
        if artifacts[name]["identity"] != _artifact_identity(panel, declaration):
            raise ValueError(f"native protocol artifact differs: {name}")
    verified_path = loaded["verified_checkpoint"]
    if (
        expected_verified_checkpoint is not None
        and expected_verified_checkpoint.resolve() != verified_path.resolve()
    ):
        raise ValueError("native verified-checkpoint path differs")
    if file_identity(verified_path) != binding.get("verified_checkpoint"):
        raise ValueError("verified checkpoint binding identity differs")
    verified = validate_verified_checkpoint(
        _load_json(verified_path, "verified checkpoint")
    )
    architecture = _load_json(loaded["architecture_receipt"], "architecture")
    probe = _load_json(loaded["restore_qualification"], "restore qualification")
    if (
        architecture != verified["architecture"]
        or probe != verified["discarded_finite_probe"]
    ):
        raise ValueError("verified checkpoint observed receipts differ")
    for name in _LINEAGE_ARTIFACTS:
        if artifacts[name]["provider"] != verified["lineage"][name]:
            raise ValueError(f"verified checkpoint lineage differs: {name}")
    _validate_bridge_execution(
        _load_json(loaded["bridge_execution"], "bridge execution"),
        verified,
        artifacts,
    )
    _validate_bridge_terminal(
        _load_json(loaded["bridge_terminal"], "bridge terminal"),
        verified,
        artifacts,
    )
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise ValueError("native checkpoint root differs")
    root = checkpoint_root.resolve()
    observed = checkpoint_inventory(root)
    if observed != verified["checkpoint"]:
        raise ValueError("native checkpoint local inventory differs")
    if binding.get("trace") not in (
        {"schema": "npa.behavior.comet-native-action-trace.v1", "enabled": False},
        {
            "schema": "npa.behavior.comet-native-action-trace.v1",
            "enabled": True,
            "action_width": 23,
            "left_command_index": 14,
            "right_command_index": 22,
            "left_proprio_indices": [24, 25],
            "right_proprio_indices": [49, 50],
        },
    ):
        raise ValueError("native action trace configuration differs")
    serving = _identity(expected_serving_identity, "native serving")
    if serving != policy["artifacts"]["serving"]:
        raise ValueError("native serving identity differs")
    admission = {
        "schema": "npa.behavior.comet-native-train-admission.v1",
        "panel_id": panel["panel_id"],
        "policy_id": policy["policy_id"],
        "task": binding["task"],
        "profile": verified["profile"],
        "logical_update": verified["logical_update"],
        "manager_step": verified["manager_step"],
        "checkpoint_content_sha256": observed["content_sha256"],
        "architecture_partition_sha256": architecture["partition_sha256"],
        "verified_checkpoint": binding["verified_checkpoint"],
        "serving_identity": serving,
        "trace": binding["trace"],
        "artifact_rows_sha256": _canonical_digest(artifacts),
    }
    admission["admission_sha256"] = _canonical_digest(admission)
    return admission
