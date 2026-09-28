"""Inspect one frozen campaign policy identity without creating runtime state."""

from __future__ import annotations

import json
from pathlib import Path

from .campaign_runner import _is_train_panel, _validate_execution_panel
from .serving_identity import verify_serving_identity


def inspect_policy_identity(args, panel_path: Path) -> dict:
    """Verify local policy bytes and settings against one frozen panel.

    Args:
        args: Policy arguments accepted by the production campaign worker.
        panel_path: Local frozen development, reporting, or TRAIN panel.
    Returns:
        A machine-readable identity receipt with an explicit effects boundary.
    Raises:
        ValueError: The panel scope or policy identity differs.
        OSError: A required local artifact cannot be read.
    """
    panel = _validate_execution_panel(json.loads(panel_path.read_bytes()))
    train_panel = _is_train_panel(panel)
    if train_panel != bool(args.train_experience):
        raise ValueError(
            "TRAIN panels require --train-experience and DEV/REPORT panels forbid it"
        )
    policy = panel["policy_binding"] if train_panel else panel["policy"]
    identity = verify_serving_identity(args, policy)
    return {
        "schema": "npa.behavior.policy-identity-inspection.v1",
        "status": "checkpoint_and_serving_identity_verified",
        "panel": {
            "schema": panel["schema"],
            "id": panel["panel_id"],
            "split": "train" if train_panel else panel["split"],
        },
        "policy": {"identity_sha256": policy["identity_sha256"], **identity},
        "effects": {
            "case_claimed": False,
            "policy_started": False,
            "simulator_started": False,
            "workspace_created": False,
        },
    }
