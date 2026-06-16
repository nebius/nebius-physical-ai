"""Sim2Real VLM-to-RL workflow package.

Canonical orchestration: ``Sim2RealWorkflow`` in ``runner``.
Stage implementations (K8s siblings, sim backends, VLM glue): ``engine``.
"""

from __future__ import annotations

from npa.workflows.sim2real.config import artifact_uris, build_config_from_env, byo_seams
from npa.workflows.sim2real.constants import (
    SCHEMA_E2E_REPORT,
    SCHEMA_HELDOUT_REPORT,
    SCHEMA_RL_SIGNAL,
    SCHEMA_THRESHOLD_DECISION,
    SCHEMA_VLM_EVAL,
    SIM_BACKEND_GENESIS,
    SIM_BACKEND_ISAAC,
    SIM_BACKENDS,
)
from npa.workflows.sim2real.engine import (
    convert_vlm_eval_to_rl_signal,
    evaluate_rollout_with_vlm,
    generate_action_rollouts,
    run_finalize,
    run_heldout_eval,
    run_inner_loop,
    run_preamble,
    run_single_outer_iteration,
    signal_mapping_rules,
    upload_run_artifacts,
)
from npa.workflows.sim2real.models import (
    ComponentRecord,
    Sim2RealLoopConfig,
    Sim2RealLoopError,
    default_augment_image,
    default_envgen_image,
    default_eval_image,
    default_isaac_image,
    default_policy_image,
    default_trainer_image,
    default_vlm_image,
    new_run_id,
)
from npa.workflows.sim2real.runner import Sim2RealWorkflow, run_full_loop
from npa.workflows.sim2real.state import WorkflowState

__all__ = [
    "ComponentRecord",
    "SCHEMA_E2E_REPORT",
    "SCHEMA_HELDOUT_REPORT",
    "SCHEMA_RL_SIGNAL",
    "SCHEMA_THRESHOLD_DECISION",
    "SCHEMA_VLM_EVAL",
    "SIM_BACKEND_GENESIS",
    "SIM_BACKEND_ISAAC",
    "SIM_BACKENDS",
    "Sim2RealLoopConfig",
    "Sim2RealLoopError",
    "Sim2RealWorkflow",
    "WorkflowState",
    "artifact_uris",
    "build_config_from_env",
    "byo_seams",
    "convert_vlm_eval_to_rl_signal",
    "default_augment_image",
    "default_envgen_image",
    "default_eval_image",
    "default_isaac_image",
    "default_policy_image",
    "default_trainer_image",
    "default_vlm_image",
    "evaluate_rollout_with_vlm",
    "generate_action_rollouts",
    "new_run_id",
    "run_finalize",
    "run_full_loop",
    "run_heldout_eval",
    "run_inner_loop",
    "run_preamble",
    "run_single_outer_iteration",
    "signal_mapping_rules",
    "upload_run_artifacts",
]
