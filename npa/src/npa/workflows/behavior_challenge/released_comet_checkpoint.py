"""Validate a released Comet inference checkpoint for recorded TRAIN use."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re

from .campaign import canonical_digest
from .comet_policy import (
    COMET12_PROFILE,
    MODEL_REPOSITORY,
    SOURCE_COMMIT,
    load_checkpoint_manifest,
    verify_checkpoint_archive,
)
from .native_training_checkpoint import file_identity
from .nonreporting_train import validate_train_panel

BINDING_SCHEMA = "npa.behavior.comet-released-policy-binding.v1"
ADMISSION_SCHEMA = "npa.behavior.comet-released-train-admission.v1"
_ROLES = {
    "rng_contract",
    "normalization",
    "tokenizer",
    "task_mapping",
    "train_evaluator_argv",
    "evaluator_source",
    "controller_source",
    "robot_config",
}


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Released Comet {label} SHA-256 differs")
    return value


def _identity(value: object, label: str) -> dict[str, int | str]:
    if (
        not isinstance(value, dict)
        or set(value) != {"bytes", "sha256"}
        or isinstance(value["bytes"], bool)
        or not isinstance(value["bytes"], int)
        or value["bytes"] < 0
    ):
        raise ValueError(f"Released Comet {label} identity differs")
    _sha(value["sha256"], label)
    return value


def _relative(value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "%" in value:
        raise ValueError("Released Comet evidence path differs")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError("Released Comet evidence path differs")
    return Path(*pure.parts)


def _evidence(binding: dict, input_root: Path) -> dict[str, dict]:
    rows = binding.get("evidence")
    if not isinstance(rows, dict) or set(rows) != _ROLES:
        raise ValueError("Released Comet evidence roles differ")
    if input_root.is_symlink() or not input_root.is_dir():
        raise ValueError("Released Comet input root differs")
    root = input_root.resolve(strict=True)
    verified = {}
    for role, row in rows.items():
        if not isinstance(row, dict) or set(row) != {"path", "identity"}:
            raise ValueError("Released Comet evidence row differs")
        relative = _relative(row["path"])
        target = _contained_file(input_root, root, relative)
        if file_identity(target) != _identity(row["identity"], role):
            raise ValueError(f"Released Comet {role} evidence differs")
        verified[role] = row
    return verified


def _contained_file(input_root: Path, root: Path, relative: Path) -> Path:
    current = input_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Released Comet evidence path contains a symlink")
    if not current.is_file() or not current.resolve(strict=True).is_relative_to(root):
        raise ValueError("Released Comet evidence path escapes its input root")
    return current


def _validate_binding(value: object) -> dict:
    required = {
        "schema",
        "profile",
        "source_commit",
        "task",
        "task_id",
        "panel_id",
        "policy_binding_sha256",
        "checkpoint",
        "evidence",
        "trace",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Released Comet binding fields differ")
    checkpoint = _checkpoint_binding(value["checkpoint"])
    if (
        value["schema"] != BINDING_SCHEMA
        or value["profile"] != COMET12_PROFILE.kind
        or value["source_commit"] != SOURCE_COMMIT
        or not isinstance(value["task"], str)
        or not value["task"]
        or isinstance(value["task_id"], bool)
        or value["task_id"] not in COMET12_PROFILE.task_ids
        or value["trace"] != {"enabled": False}
    ):
        raise ValueError("Released Comet binding authority differs")
    _sha(value["panel_id"], "panel")
    _sha(value["policy_binding_sha256"], "policy binding")
    _identity(checkpoint["archive"], "archive")
    _identity(checkpoint["inventory"], "inventory")
    return value


def _checkpoint_binding(value: object) -> dict:
    checkpoint = value
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "archive",
        "inventory",
        "name",
        "repository",
        "revision",
    }:
        raise ValueError("Released Comet checkpoint binding differs")
    if (
        checkpoint["name"] != COMET12_PROFILE.checkpoint
        or checkpoint["repository"] != MODEL_REPOSITORY
        or checkpoint["revision"] != COMET12_PROFILE.revision
    ):
        raise ValueError("Released Comet checkpoint authority differs")
    return checkpoint


def _protocol_identities(panel: dict) -> dict[str, dict]:
    protocol = panel["protocol"]
    contract = protocol["evaluator_contract"]
    science = protocol["science_lineage"]
    mapping = protocol["task_mapping"]
    return {
        "rng_contract": contract["rng_contract"],
        "normalization": science["normalization"],
        "tokenizer": science["tokenizer"],
        "task_mapping": mapping["mapping_artifact"],
        "train_evaluator_argv": contract["argv_contract"],
        "evaluator_source": contract["evaluator_source"],
        "controller_source": contract["controller_source"],
        "robot_config": contract["robot_config"],
    }


def validate_released_comet_admission(
    binding_path: Path,
    input_root: Path,
    checkpoint_archive: Path,
    checkpoint_root: Path,
    panel: dict,
    *,
    expected_serving_identity: dict,
) -> dict:
    """Validate exact release bytes and a one-case TRAIN protocol before startup."""
    if binding_path.is_symlink() or not binding_path.is_file():
        raise ValueError("Released Comet binding must be a regular file")
    binding = _validate_binding(json.loads(binding_path.read_text()))
    evidence = _evidence(binding, input_root)
    _panel_binding(binding, panel, expected_serving_identity)
    actual_protocol = _protocol_identities(panel)
    if any(
        evidence[role]["identity"] != identity
        for role, identity in actual_protocol.items()
    ):
        raise ValueError("Released Comet protocol evidence differs")
    files = _checkpoint_files(binding, checkpoint_archive, checkpoint_root)
    return _admission(binding_path, binding, evidence, files)


def _panel_binding(binding: dict, panel: dict, serving: dict) -> None:
    panel = validate_train_panel(panel)
    protocol = panel["protocol"]
    mapping = protocol["task_mapping"]
    evaluator = protocol["evaluator_contract"]
    if panel.get("panel_id") != binding["panel_id"]:
        raise ValueError("Released Comet panel identity differs")
    cases = panel.get("cases")
    if (
        protocol["split"] != "train"
        or not isinstance(cases, list)
        or len(cases) != 1
        or cases[0].get("task") != binding["task"]
        or cases[0].get("rollout_id") != 0
        or protocol["task"] != binding["task"]
        or mapping["data_task_id"] != binding["task_id"]
        or evaluator["model_prediction_horizon"] != 32
        or evaluator["executed_prefix"] != 32
        or panel.get("policy_binding_sha256") != binding["policy_binding_sha256"]
    ):
        raise ValueError("Released Comet TRAIN panel differs")
    policy = panel.get("policy_binding")
    if (
        not isinstance(policy, dict)
        or policy.get("identity_sha256") != binding["policy_binding_sha256"]
        or policy.get("artifacts", {}).get("checkpoint")
        != binding["checkpoint"]["archive"]
        or policy.get("artifacts", {}).get("serving") != serving
    ):
        raise ValueError("Released Comet policy identity differs")


def _checkpoint_files(binding: dict, archive: Path, root: Path) -> dict:
    inventory = Path(__file__).with_name(COMET12_PROFILE.inventory)
    load_checkpoint_manifest(inventory, COMET12_PROFILE)
    if file_identity(inventory) != binding["checkpoint"]["inventory"]:
        raise ValueError("Released Comet checkpoint inventory differs")
    if file_identity(archive) != binding["checkpoint"]["archive"]:
        raise ValueError("Released Comet checkpoint archive differs")
    return verify_checkpoint_archive(
        archive,
        root,
        binding["checkpoint"]["archive"]["sha256"],
        COMET12_PROFILE,
    )


def _admission(binding_path, binding, evidence, files) -> dict:
    return {
        "schema": ADMISSION_SCHEMA,
        "status": "released_params_checkpoint_and_train_protocol_exact",
        "binding_sha256": file_identity(binding_path)["sha256"],
        "panel_id": binding["panel_id"],
        "policy_binding_sha256": binding["policy_binding_sha256"],
        "task": binding["task"],
        "task_id": binding["task_id"],
        "checkpoint": binding["checkpoint"],
        "checkpoint_file_count": len(files),
        "evidence": evidence,
        "trace": binding["trace"],
        "full_train_state_claimed": False,
        "optimizer_state_claimed": False,
        "admission_sha256": canonical_digest(binding),
    }


__all__ = ["validate_released_comet_admission"]
