"""Verify the public reference's runtime-fetched locomotion controller before load."""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile

CONTROLLER_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/"
    "Isaac/IsaacLab/Policies/ANYmal-C/Blind/policy.pt"
)
CONTROLLER_SHA256 = "8cb6cfe8987bbd8da5da4031aa3ffee9e09cc644d140febd25371b84a97ac555"


def create_action(cfg, env):
    """Load the upstream native action from the exact controller bytes in the recipe.

    Args:
        cfg: Upstream pretrained locomotion action configuration.
        env: Native environment carrying the sealed controller digest.
    Returns:
        Upstream PreTrainedPolicyAction with its actual loaded identity recorded.
    Raises:
        ValueError: The vendor URL or downloaded digest differs from the contract.
        RuntimeError: Native loading fails.
    """
    from isaaclab.utils.assets import read_file

    if cfg.policy_path != CONTROLLER_URL:
        raise ValueError("reference controller must use the canonical vendor URL")
    payload = read_file(CONTROLLER_URL).getvalue()
    return _load_verified(cfg, env, payload, env.cfg.npa_reference_controller_sha256)


def _load_verified(cfg, env, payload, expected):
    digest = hashlib.sha256(payload).hexdigest()
    if not payload or digest != expected:
        raise ValueError("reference controller SHA-256 differs from sealed recipe")
    from isaaclab_tasks.manager_based.navigation.mdp.pre_trained_policy_action import (
        PreTrainedPolicyAction,
    )

    with tempfile.TemporaryDirectory(prefix="npa-locomotion-") as temporary:
        path = Path(temporary) / "controller.pt"
        path.write_bytes(payload)
        local = deepcopy(cfg)
        local.policy_path = str(path)
        action = PreTrainedPolicyAction(local, env)
    env.npa_reference_controller = {
        "source_url": CONTROLLER_URL,
        "sha256": digest,
        "bytes": len(payload),
        "verified_before_load": True,
    }
    return action


def controller_evidence(env, recipe):
    """Require evidence from the actual verified native controller load.

    Args:
        env: Initialized native environment.
        recipe: Sealed reference recipe or independent BYOF recipe.
    Returns:
        Actual reference controller identity, or None for external BYOF tasks.
    Raises:
        ValueError: A reference task omitted or changed the verified controller.
    """
    if recipe.reference_controller_sha256 is None:
        return None
    evidence = getattr(env, "npa_reference_controller", {})
    if (
        evidence.get("sha256") != recipe.reference_controller_sha256
        or evidence.get("verified_before_load") is not True
        or evidence.get("source_url") != CONTROLLER_URL
    ):
        raise ValueError("native reference omitted its verified controller identity")
    return dict(evidence)
