"""Compatibility import surface for the durable Sim2Real controller.

The executable implementation lives in :mod:`npa.workflows.sim2real`.  This
module intentionally contains no controller, Kubernetes, bootstrap, or stage
logic; historical imports and ``python -m npa.workflows.sim2real_loop`` resolve
to the exact same package and CLI used by the canonical live workflow.
"""

# This module intentionally re-exports historical names for downstream callers.
# ruff: noqa: F401,F403

from __future__ import annotations

from npa.workflows.sim2real.config import (
    artifact_uris,
    build_config_from_env,
    byo_seams,
)
from npa.workflows.sim2real.constants import *  # noqa: F403
from npa.workflows.sim2real.engine import (
    _apply_reference_adapter_heldout_gate,
    _component_heldout_payload,
    _component_vlm_payload,
    _consume_stage_assets,
    _image_pull_policy,
    _inner_loop_progress_score,
    _isaac_import_mesh_to_usd,
    _normalize_heldout_report,
    _read_component_env_records,
    _resolve_env_records_s3_uri,
    _resolve_heldout_robot,
    _resolve_heldout_scene,
    _resolve_isaac_scene,
    _run_policy_rollouts_via_command,
    _storage_client,
    _write_stage,
    convert_vlm_eval_to_rl_signal,
    evaluate_rollout_with_vlm,
    generate_action_rollouts,
    run_cosmos2_transfer_component_from_s3,
    run_finalize,
    run_heldout_eval,
    run_heldout_eval_component_from_s3,
    run_inner_loop,
    run_policy_actions_component_from_s3,
    run_preamble,
    run_single_outer_iteration,
    run_vlm_eval_component_from_s3,
    signal_mapping_rules,
    threshold_decision,
    upload_run_artifacts,
)
from npa.workflows.sim2real.k8s_components import _kubernetes_component_env
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
from npa.workflows.sim2real.reference_helpers import (
    _heldout_env_score,
    _signal_diversity_report,
    _write_env_manifest,
    _write_train_heldout_split,
)
from npa.workflows.sim2real.runner import Sim2RealWorkflow, run_full_loop
from npa.workflows.sim2real.state import WorkflowState


def main(argv: list[str] | None = None) -> int:
    """Delegate the historical module entrypoint to the canonical CLI."""

    from npa.workflows.sim2real.cli import main as controller_main

    return controller_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
