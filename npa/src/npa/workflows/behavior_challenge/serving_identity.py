"""Bind campaign policy identity to actual serving code and inference settings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .protocol import file_digest

_SERVING_FILES = (
    "campaign_runner.py",
    "nonreporting_train.py",
    "evaluator_versions.py",
    "evaluator_wire.py",
    "policy.py",
    "policy_server.py",
    "rlc_policy.py",
    "rlc_server.py",
    "rlc_selected.py",
    "rlc_transition.py",
    "rlc_execution.py",
    "rlc_observations.py",
    "rlc_correlation.py",
    "rlc_selected_server.py",
    "rlc-checkpoints.json",
    "comet_policy.py",
    "comet_server.py",
    "native_comet_checkpoint.py",
    "native_comet_policy.py",
    "native_comet_server.py",
    "comet12-checkpoint.json",
    "comet50-checkpoint.json",
)
_SPECIALIST_SERVING_FILES = (
    "rlc_specialist.py",
    "rlc-specialist-checkpoint.json",
)
_TRAINED_COMET_SERVING_FILES = (
    "trained_comet_checkpoint.py",
    "trained_comet_policy.py",
    "trained_comet_producer.py",
)
_TRAIN_EXPERIENCE_SERVING_FILES = (
    "train_experience.py",
    "train_experience_evaluator.py",
    "semantic_monitor/__init__.py",
    "semantic_monitor/interface.py",
)
_INPUT_FIELDS = (
    "policy_selected_export_receipt",
    "policy_correlation_manifest",
    "policy_validation_receipt",
    "policy_stock_correlation_asset",
)


def _serving_source(args) -> dict[str, str]:
    source_names = _SERVING_FILES
    if args.policy_kind == "rlc-specialist":
        source_names += _SPECIALIST_SERVING_FILES
    if args.policy_kind == "comet-trained":
        source_names += _TRAINED_COMET_SERVING_FILES
    if getattr(args, "train_experience", False):
        source_names += _TRAIN_EXPERIENCE_SERVING_FILES
    root = Path(__file__).parent
    return {name: file_digest(root / name) for name in source_names}


def serving_artifact(args) -> dict:
    """Fingerprint exact adapter bytes, execution settings, and auxiliary inputs.

    Args:
        args: Managed policy arguments; optional receipt and correlation paths
            must resolve to the same bytes on planning and serving machines.
    Returns:
        SHA-256 and byte count usable as a frozen policy's serving artifact.
    Raises:
        OSError: Serving code or a declared auxiliary input cannot be read.
    """
    payload = {
        "schema": "npa.behavior.serving-identity.v1",
        "kind": args.policy_kind,
        "execution_variant": args.policy_execution_variant,
        "episode_lifecycle": "fresh-managed-process-per-case-v1",
        "source": _serving_source(args),
        "inputs": {
            field: file_digest(Path(getattr(args, field)))
            for field in _INPUT_FIELDS
            if getattr(args, field, None)
        },
        "stock_correlation_sha256": getattr(
            args, "policy_stock_correlation_sha256", None
        ),
        "task_name": getattr(args, "policy_task_name", None),
        "native_configuration": (
            _native_configuration(Path(args.policy_native_binding))
            if args.policy_kind == "comet-native"
            else None
        ),
        "trained_configuration": (
            _trained_configuration(Path(args.policy_archive))
            if args.policy_kind == "comet-trained"
            else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}


def _native_configuration(path: Path) -> dict:
    value = json.loads(path.read_text())
    return {
        "schema": value.get("schema"),
        "task": value.get("task"),
        "task_id": value.get("task_id"),
        "trace": value.get("trace"),
    }


def _trained_configuration(path: Path) -> dict:
    value = json.loads(path.read_text())
    return {
        "schema": value.get("schema"),
        "profile": value.get("profile"),
        "selected_step": value.get("selected_step"),
        "manager_step": value.get("manager_step"),
        "selected_checkpoint_inventory_sha256": value.get(
            "selected_checkpoint_inventory_sha256"
        ),
        "selected_precision_receipt_sha256": value.get(
            "selected_precision_receipt_sha256"
        ),
        "task": value.get("task"),
        "task_id": value.get("task_id"),
        "parity_output": value.get("parity_output"),
        "normalization": value.get("normalization"),
        "rng_contract": value.get("rng_contract"),
        "trace": value.get("trace"),
        "serving_tree_sha256": value.get("serving_tree_sha256"),
    }


def verify_serving_identity(args, policy: dict) -> None:
    """Reject policy execution if checkpoint or serving identities have changed.

    Args:
        args: Managed policy paths and execution settings.
        policy: Frozen campaign policy identity.
    Returns:
        None.
    Raises:
        ValueError: Loaded artifacts differ from the declared policy.
        OSError: The checkpoint archive or adapter inputs cannot be read.
    """
    checkpoint = Path(args.policy_archive)
    actual = {"sha256": file_digest(checkpoint), "bytes": checkpoint.stat().st_size}
    if actual != policy["artifacts"]["checkpoint"]:
        raise ValueError("Campaign checkpoint differs from its frozen artifact")
    if serving_artifact(args) != policy["artifacts"]["serving"]:
        raise ValueError("Campaign serving code or configuration differs")
