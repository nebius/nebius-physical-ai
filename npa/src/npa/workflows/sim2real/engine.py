"""Compatibility facade for the retired direct-Kubernetes Sim2Real controller.

The canonical operator path is the compositional npa.workflow/v0.0.1 spec and
its stateless workflow_stage adapters. This facade preserves imports used by
archived-run inspection and migration tooling; it does not orchestrate the canonical
workflow. Legacy implementation is split by solution boundary so new work cannot
accumulate in another monolith.

Compatibility is limited to callers and artifacts created before the canonical
standard-runtime conversion. It is scheduled for removal no earlier than 0.5.0
and 2027-02-01; no new workflow may import this module.
"""

# Compatibility deliberately re-exports imported symbols and installs legacy
# function names after importing the bounded implementation modules.
# ruff: noqa: F401,E402

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
import types
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # The runtime facade installs these names from the bounded legacy modules
    # below. Explicit type-only imports keep that dynamic compatibility surface
    # reviewable without making canonical code import the legacy implementation.
    from npa.workflows.sim2real.legacy_components import (
        _apply_reference_adapter_heldout_gate as _apply_reference_adapter_heldout_gate,
        _config_from_workflow_state as _config_from_workflow_state,
        _convert_eval_to_signal as _convert_eval_to_signal,
        _effective_k8s_parallelism as _effective_k8s_parallelism,
        _inner_loop_progress_score as _inner_loop_progress_score,
        _run_policy_rollouts_via_command as _run_policy_rollouts_via_command,
        _run_trainer_via_command as _run_trainer_via_command,
        _signal_training_imports as _signal_training_imports,
        _storage_client as _storage_client,
        convert_vlm_eval_to_rl_signal as convert_vlm_eval_to_rl_signal,
        evaluate_rollout_with_vlm as evaluate_rollout_with_vlm,
        generate_action_rollouts as generate_action_rollouts,
        run_cosmos2_transfer_component as run_cosmos2_transfer_component,
        run_envgen_sharded_component as run_envgen_sharded_component,
        run_inner_loop as run_inner_loop,
        run_policy_rollout_component as run_policy_rollout_component,
        signal_mapping_rules as signal_mapping_rules,
    )
    from npa.workflows.sim2real.legacy_artifacts import (
        _write_stage as _write_stage,
        run_cosmos2_transfer_component_from_s3 as run_cosmos2_transfer_component_from_s3,
    )
    from npa.workflows.sim2real.legacy_heldout import (
        _component_heldout_payload as _component_heldout_payload,
        _component_vlm_payload as _component_vlm_payload,
        _consume_stage_assets as _consume_stage_assets,
        _normalize_heldout_report as _normalize_heldout_report,
        _resolve_heldout_robot as _resolve_heldout_robot,
        _resolve_heldout_scene as _resolve_heldout_scene,
        _resolve_isaac_scene as _resolve_isaac_scene,
        run_heldout_eval as run_heldout_eval,
        run_heldout_eval_component_from_s3 as run_heldout_eval_component_from_s3,
        run_vlm_eval_component_from_s3 as run_vlm_eval_component_from_s3,
    )
    from npa.workflows.sim2real.legacy_isaac import (
        _heldout_render_step_indices as _heldout_render_step_indices,
        _isaac_extract_rgb_frame as _isaac_extract_rgb_frame,
        _isaac_import_mesh_to_usd as _isaac_import_mesh_to_usd,
        _read_component_env_records as _read_component_env_records,
        _resolve_env_records_s3_uri as _resolve_env_records_s3_uri,
        _write_render_png as _write_render_png,
    )
    from npa.workflows.sim2real.legacy_orchestration import (
        run_finalize as run_finalize,
        run_preamble as run_preamble,
        run_single_outer_iteration as run_single_outer_iteration,
    )


LEGACY_COMPATIBILITY_UNTIL = "2027-02-01"
LEGACY_REMOVAL_VERSION = "0.5.0"
LEGACY_SCOPE = "pre-standard-runtime callers and archived artifact replay"


from npa.clients.storage import StorageClient
from npa.workflows.sim2real.artifact_upload import (
    _upload_final_report,
    upload_run_artifacts,  # noqa: F401 - public engine import surface
)
from npa.workbench.cosmos.reason import (
    CosmosReasonError,
    merge_dual_reason_evaluations,
    resolve_cosmos_reason_model_id,
    run_cosmos_reason_vlm,
    task_description_from_manifest,
)
from npa.workflows.sim2real.config import artifact_uris, byo_seams
from npa.workflows.sim2real.component_records import (
    _expand_envgen_component_records,
    _loop_component_records,
    _persisted_loop_component_records,
)
from npa.workflows.sim2real.capture import runtime_parameter_metadata
from npa.workflows.sim2real.constants import (
    CORRECTIVE_TARGETS,
    DEFAULT_ISAAC_TASK,
    DEFAULT_REFERENCE_VLM_MODEL,
    DEFAULT_SIM_BACKEND,
    DEFAULT_THRESHOLD,
    DEFAULT_VLM_SEAM_EVIDENCE,
    ERROR_SEVERITY,
    SCHEMA_E2E_REPORT,
    SCHEMA_HELDOUT_REPORT,
    SCHEMA_RL_SIGNAL,
    SCHEMA_VLM_EVAL,
    SIM_BACKEND_GENESIS,
    SIM_BACKEND_ISAAC,
    SIM_BACKENDS,
)
from npa.workflows.sim2real.models import (
    ComponentRecord,
    Sim2RealLoopConfig,
    Sim2RealLoopError,
)
from npa.workflows.sim2real.decision import threshold_decision
from npa.workflows.sim2real.gpu_fallback import (
    GpuCapacityExhausted,
    GpuJobFailure,
    gpu_fallback_report_contract,
    minimum_vram_for_workload,
    run_gpu_job_with_fallback,
    workload_kind,
)
from npa.workflows.sim2real.k8s_components import (
    _component_job_manifest,
    _component_job_script as _component_job_script,
    _image_pull_policy,
    _indexed_component_job_manifest,
    _kubernetes_component_env as _kubernetes_component_env,
)
from npa.workflows.sim2real.workflow_transfer import (
    run_real_cosmos_transfer as _run_real_cosmos_transfer,
)
from npa.workflows.sim2real.reporting import build_progress_metrics
from npa.workflows.sim2real.policy_actions_stage import (
    run_policy_actions_component_from_s3 as run_policy_actions_component_from_s3,
)
from npa.workflows.sim2real.reference_helpers import (
    _heldout_env_score,
    _signal_diversity_report as _signal_diversity_report,
    _signal_mean_reward as _signal_mean_reward,
    _write_env_manifest as _write_env_manifest,
    _write_ppm,
    _write_train_heldout_split as _write_train_heldout_split,
)
from npa.workflows.sim2real.utils import (
    _artifact_root_uri,
    _bool_value,
    _serviceaccount_namespace,
    _split_csv,
    _utc_now,
    _write_json_artifact,
)
from npa.workflows.sim2real.viz_contract import visualization_run_metadata
from npa.workflows.sim2real.workflow_state_io import (
    _read_workflow_state,
    _workflow_state_path,  # noqa: F401 - legacy engine import surface
    _write_workflow_state,
    emit_active_progress_rerun,  # noqa: F401 - imported by runner from engine
    sync_workflow_state_to_s3,  # noqa: F401 - imported by runner from engine
)

# Isaac Sim app handle — closed only after held-out report upload.
_ISAAC_SIMULATION_APP: Any = None
HELDOUT_VIZ_CAMERA_NAME = "heldout_viz_camera"
DEFAULT_HELDOUT_RENDER_FRAMES = 8
SCHEMA_HELDOUT_RENDERS = "npa.sim2real.heldout_renders.v1"
if TYPE_CHECKING:
    from npa.workbench.lerobot.policy_container import VlmSignalUpdateResult


from npa.workflows.sim2real import (
    legacy_artifacts as _legacy_artifacts,
    legacy_components as _legacy_components,
    legacy_heldout as _legacy_heldout,
    legacy_isaac as _legacy_isaac,
    legacy_orchestration as _legacy_orchestration,
)

_LEGACY_MODULES = (
    _legacy_orchestration,
    _legacy_components,
    _legacy_heldout,
    _legacy_isaac,
    _legacy_artifacts,
)
for _legacy_module in _LEGACY_MODULES:
    for _legacy_name in _legacy_module.__all__:
        globals()[_legacy_name] = getattr(_legacy_module, _legacy_name)

__all__ = sorted({name for module in _LEGACY_MODULES for name in module.__all__})


class _EngineFacadeModule(types.ModuleType):
    """Forward compatibility monkeypatches to the bounded legacy modules."""

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name.startswith("_legacy_") or name == "_LEGACY_MODULES":
            return
        for module in _LEGACY_MODULES:
            if hasattr(module, name):
                setattr(module, name, value)


sys.modules[__name__].__class__ = _EngineFacadeModule
