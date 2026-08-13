"""Sim2Real compatibility exports without eager legacy-controller imports.

The canonical workflow invokes focused modules such as ``workflow_stage``.
Archived callers may still import the historical public names from this package;
those modules are loaded only when the corresponding attribute is requested.
The compatibility target is removal in 0.5.0, no earlier than 2027-02-01.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "artifact_uris": ("npa.workflows.sim2real.config", "artifact_uris"),
    "build_config_from_env": (
        "npa.workflows.sim2real.config",
        "build_config_from_env",
    ),
    "byo_seams": ("npa.workflows.sim2real.config", "byo_seams"),
    "SCHEMA_E2E_REPORT": ("npa.workflows.sim2real.constants", "SCHEMA_E2E_REPORT"),
    "SCHEMA_HELDOUT_REPORT": (
        "npa.workflows.sim2real.constants",
        "SCHEMA_HELDOUT_REPORT",
    ),
    "SCHEMA_RL_SIGNAL": ("npa.workflows.sim2real.constants", "SCHEMA_RL_SIGNAL"),
    "SCHEMA_THRESHOLD_DECISION": (
        "npa.workflows.sim2real.constants",
        "SCHEMA_THRESHOLD_DECISION",
    ),
    "SCHEMA_VLM_EVAL": ("npa.workflows.sim2real.constants", "SCHEMA_VLM_EVAL"),
    "SIM_BACKEND_GENESIS": (
        "npa.workflows.sim2real.constants",
        "SIM_BACKEND_GENESIS",
    ),
    "SIM_BACKEND_ISAAC": (
        "npa.workflows.sim2real.constants",
        "SIM_BACKEND_ISAAC",
    ),
    "SIM_BACKENDS": ("npa.workflows.sim2real.constants", "SIM_BACKENDS"),
    "convert_vlm_eval_to_rl_signal": (
        "npa.workflows.sim2real.engine",
        "convert_vlm_eval_to_rl_signal",
    ),
    "evaluate_rollout_with_vlm": (
        "npa.workflows.sim2real.engine",
        "evaluate_rollout_with_vlm",
    ),
    "generate_action_rollouts": (
        "npa.workflows.sim2real.engine",
        "generate_action_rollouts",
    ),
    "run_finalize": ("npa.workflows.sim2real.engine", "run_finalize"),
    "run_heldout_eval": ("npa.workflows.sim2real.engine", "run_heldout_eval"),
    "run_inner_loop": ("npa.workflows.sim2real.engine", "run_inner_loop"),
    "run_preamble": ("npa.workflows.sim2real.engine", "run_preamble"),
    "run_single_outer_iteration": (
        "npa.workflows.sim2real.engine",
        "run_single_outer_iteration",
    ),
    "signal_mapping_rules": (
        "npa.workflows.sim2real.engine",
        "signal_mapping_rules",
    ),
    "upload_run_artifacts": (
        "npa.workflows.sim2real.engine",
        "upload_run_artifacts",
    ),
    "ComponentRecord": ("npa.workflows.sim2real.models", "ComponentRecord"),
    "Sim2RealLoopConfig": ("npa.workflows.sim2real.models", "Sim2RealLoopConfig"),
    "Sim2RealLoopError": ("npa.workflows.sim2real.models", "Sim2RealLoopError"),
    "default_augment_image": (
        "npa.workflows.sim2real.models",
        "default_augment_image",
    ),
    "default_envgen_image": (
        "npa.workflows.sim2real.models",
        "default_envgen_image",
    ),
    "default_eval_image": ("npa.workflows.sim2real.models", "default_eval_image"),
    "default_isaac_image": ("npa.workflows.sim2real.models", "default_isaac_image"),
    "default_policy_image": (
        "npa.workflows.sim2real.models",
        "default_policy_image",
    ),
    "default_trainer_image": (
        "npa.workflows.sim2real.models",
        "default_trainer_image",
    ),
    "default_vlm_image": ("npa.workflows.sim2real.models", "default_vlm_image"),
    "new_run_id": ("npa.workflows.sim2real.models", "new_run_id"),
    "Sim2RealWorkflow": ("npa.workflows.sim2real.runner", "Sim2RealWorkflow"),
    "run_full_loop": ("npa.workflows.sim2real.runner", "run_full_loop"),
    "WorkflowState": ("npa.workflows.sim2real.state", "WorkflowState"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
