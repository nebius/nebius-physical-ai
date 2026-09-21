"""Bind campaign policy identity to actual serving code and inference settings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .protocol import file_digest

_SERVING_FILES = (
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
)
_INPUT_FIELDS = (
    "policy_selected_export_receipt",
    "policy_correlation_manifest",
    "policy_validation_receipt",
    "policy_stock_correlation_asset",
)


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
    root = Path(__file__).parent
    payload = {
        "schema": "npa.behavior.serving-identity.v1",
        "kind": args.policy_kind,
        "execution_variant": args.policy_execution_variant,
        "source": {name: file_digest(root / name) for name in _SERVING_FILES},
        "inputs": {
            field: file_digest(Path(getattr(args, field)))
            for field in _INPUT_FIELDS
            if getattr(args, field, None)
        },
        "stock_correlation_sha256": getattr(
            args, "policy_stock_correlation_sha256", None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}


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
