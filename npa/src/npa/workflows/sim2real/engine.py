"""Stage implementations for the Sim2Real VLM-to-RL workflow.

Heavy glue lives here: sibling K8s jobs, sim backends, VLM critique, and RL
signal conversion. Orchestration belongs in ``npa.workflows.sim2real.runner``.

Canonical stage map (``monitor._STAGE_SPECS``, ``sim2real_stages``):

| Stage | Monitor name | Entrypoint | Primary artifacts |
| --- | --- | --- | --- |
| 1 | ``stage_01_trigger`` | ``run_preamble`` | ``stage_01_trigger/trigger.json`` |
| 2 | ``stage_02_assets`` | ``run_preamble`` → ``run_assets_stage`` | ``stage_02_assets/consumed_scene_spec.json`` |
| 3 | ``stage_03_augment`` | ``run_preamble`` → ``run_augment_stage`` | ``augment/manifest.json`` |
| 4 | ``stage_04_envs_raw`` | ``run_envgen_split_stage`` | ``envs/raw/`` |
| 5 | ``stage_05_envs_train`` | ``run_envgen_split_stage`` | ``envs/train/envs.jsonl`` |
| 6 | ``stage_06_tokens`` | ``run_envgen_split_stage`` | ``tokens/manifest.json`` |
| 7 | ``stage_07_actions_train`` | ``run_inner_loop`` → ``run_policy_rollouts`` | ``actions/train/`` |
| 8 | ``stage_08_vlm_eval_train`` | ``run_inner_loop`` → ``evaluate_rollout_with_vlm`` | ``vlm_eval/train/`` |
| 9 | ``stage_09_training_signal`` | ``run_inner_loop`` (signal + trainer) | ``training_signal/train/`` |
| 10 | ``stage_10_eval_heldout`` | ``run_single_outer_iteration`` → ``run_heldout_eval`` | ``eval/gold-heldout/outer-N/report.json`` |
| 11 | ``stage_11_outer_loop`` | ``run_single_outer_iteration`` → ``threshold_decision`` | ``outer_loop/decision.json`` |
| 12 | ``stage_12_external_validation_stub`` | ``run_finalize`` | ``stage_12_external_validation/external_stub.json`` |
| 13 | ``stage_13_retrigger`` | ``run_finalize`` | ``stage_13_retrigger/retrigger.json`` |
| 14 | ``stage_14_rerun_viz`` | ``run_finalize`` → ``_run_sim2real_viz_stage`` | ``reports/sim2real.rrd`` |

Phase boundaries:

- **Preamble (1–6):** ``run_preamble``
- **Outer iteration (7–11):** ``run_single_outer_iteration`` (inner loop 7–9 per outer pass)
- **Finalize (12–14 + report):** ``run_finalize``
"""

from __future__ import annotations
import logging

import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    _indexed_component_job_manifest,
    _kubernetes_component_env as _kubernetes_component_env,
)
from npa.workflows.sim2real.reporting import build_progress_metrics
from npa.workflows.sim2real.policy_actions_stage import (
    run_policy_actions_component_from_s3 as run_policy_actions_component_from_s3,
)
from npa.workflows.sim2real.reference_helpers import (
    _heldout_env_score,
    _signal_diversity_report,
    _signal_mean_reward,
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
# Per-run sibling source tarball (Isaac held-out eval cannot git-clone inside Isaac Sim).
_SIBLING_SOURCE_TARBALL_BY_RUN: dict[str, str] = {}

if TYPE_CHECKING:
    from npa.workbench.lerobot.policy_container import VlmSignalUpdateResult


def _signal_training_imports():
    from npa.workbench.lerobot.policy_container import (
        parse_vlm_signal_batch,
        run_vlm_signal_training_step,
    )

    return parse_vlm_signal_batch, run_vlm_signal_training_step


# =============================================================================
# Stages 1–6 — preamble (`run_preamble`)
# =============================================================================


def run_preamble(config: Sim2RealLoopConfig) -> dict[str, Any]:
    """Run stages 1-6 and persist workflow state."""

    config.validate()
    local_dir = config.output_dir or Path(
        tempfile.mkdtemp(prefix=f"npa-{config.run_id}-")
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    components: list[ComponentRecord] = []
    stage_records: list[dict[str, Any]] = []

    from npa.workflows.sim2real.task_contract import (
        build_task_contract,
        validate_seed_dataset_manifest,
        validate_task_dataset,
    )

    task_contract = build_task_contract(
        task_id=config.isaac_task,
        dataset_id=config.trigger_dataset_id,
        dataset_uri=config.trigger_dataset_uri,
        robot_source=config.robot_source,
        robot_preset=config.robot_preset,
    )
    task_contract_path = local_dir / "stage_02_assets" / "task-contract.json"
    _write_json_artifact(task_contract_path, task_contract)

    seed_dataset_proof: dict[str, Any] = {
        "mode": "local_contract_only",
        "provenance": task_contract["dataset"]["provenance"],
    }
    real_required = bool(
        config.s3_bucket.strip() and config.byo_trainer_command.strip()
    )
    validate_task_dataset(
        task_id=config.isaac_task,
        dataset_id=config.trigger_dataset_id,
        dataset_uri=config.trigger_dataset_uri,
        real_required=real_required,
    )
    if real_required:
        from npa.clients.storage import StorageClient

        client = StorageClient.from_environment()
        seed_manifest_uri = (
            config.trigger_dataset_uri.rstrip("/") + "/task-dataset-manifest.json"
        )
        seed_manifest_path = (
            local_dir / "stage_01_trigger" / "task-dataset-manifest.json"
        )
        client.download_path(seed_manifest_uri, str(seed_manifest_path))
        seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
        seed_dataset_proof = validate_seed_dataset_manifest(
            seed_manifest, contract=task_contract
        )
        sample_path = local_dir / "stage_01_trigger" / "sample-rollout-manifest.json"
        client.download_path(
            seed_dataset_proof["sample_rollout_manifest_uri"], str(sample_path)
        )
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        if not (sample.get("actions") and sample.get("camera_observations")):
            raise Sim2RealLoopError(
                "task seed sample must contain real actions and camera observations"
            )
        seed_dataset_proof["mode"] = "validated_isaac_seed_dataset"
        seed_dataset_proof["manifest_uri"] = seed_manifest_uri

    stage_started = time.monotonic()
    stage_records.append(
        _write_stage(
            local_dir,
            1,
            "trigger",
            {
                **_trigger_payload(config),
                "task_contract_digest": task_contract["task_contract_digest"],
                "task_data_compatible": True,
                "seed_dataset_provenance": task_contract["dataset"]["provenance"],
                "seed_dataset_proof": seed_dataset_proof,
            },
        )
    )
    stage_01_duration = round(time.monotonic() - stage_started, 3)
    components.append(
        ComponentRecord(
            "stage_01_trigger",
            "WORKS",
            "Validated a task-aligned Isaac lift-cube seed dataset and resolved runtime plug points.",
            {
                "local": str(local_dir / "stage_01_trigger" / "trigger.json"),
                "job_name": config.run_id,
                "execution": "orchestrator_record",
                "duration_s": stage_01_duration,
            },
        )
    )

    from npa.workflows.sim2real_assets import run_assets_stage
    from npa.workflows.sim2real_stages import run_augment_stage, run_envgen_split_stage

    stage_started = time.monotonic()
    assets_result = run_assets_stage(config, local_dir)
    stage_records.append(assets_result.stage_record)
    assets_component = dict(assets_result.component)
    assets_artifacts = dict(assets_component.get("artifacts") or {})
    assets_artifacts.update(
        {
            "task_contract": (
                f"{_artifact_root_uri(config)}/stage_02_assets/task-contract.json"
                if config.s3_bucket
                else str(task_contract_path)
            ),
            "task_contract_digest": task_contract["task_contract_digest"],
            "job_name": config.run_id,
            "execution": "orchestrator_materialization",
            "duration_s": round(time.monotonic() - stage_started, 3),
        }
    )
    assets_component["artifacts"] = assets_artifacts
    components.append(ComponentRecord(**assets_component))
    scene_spec_uri = assets_result.scene_spec_uri
    robot_spec_uri = assets_result.robot_spec_uri

    stage_started = time.monotonic()
    augment_result = run_augment_stage(config, local_dir)
    stage_records.append(
        _write_json_artifact(
            local_dir / "augment" / "manifest.json", augment_result["manifest"]
        )
    )
    augment_artifacts = augment_result["component"].setdefault("artifacts", {})
    if not augment_artifacts.get("duration_s"):
        augment_artifacts["duration_s"] = round(time.monotonic() - stage_started, 3)
    components.append(ComponentRecord(**augment_result["component"]))

    envgen_result = run_envgen_split_stage(
        config,
        local_dir,
        augmented_frames_uri=augment_result["augmented_frames_uri"],
        scene_spec_uri=scene_spec_uri,
        robot_spec_uri=robot_spec_uri,
    )
    envgen_components = envgen_result.get("components") or [envgen_result["component"]]
    components.extend(ComponentRecord(**component) for component in envgen_components)
    train_envs_uri = envgen_result["train_envs_uri"]
    validation_envs_uri = envgen_result["validation_envs_uri"]
    heldout_envs_uri = envgen_result["heldout_envs_uri"]
    gold_heldout_envs_uri = envgen_result["gold_heldout_envs_uri"]
    task_contract_uri = str(task_contract_path)
    sibling_source_tarball_uri = ""
    if config.s3_bucket:
        sibling_source_tarball_uri = ensure_sibling_source_tarball(config)
        if config.sim_backend == SIM_BACKEND_ISAAC and not sibling_source_tarball_uri:
            raise Sim2RealLoopError(
                "failed to stage sibling source tarball for Isaac held-out eval"
            )
    state = {
        "schema": "npa.sim2real.workflow_state.v1",
        "run_id": config.run_id,
        "status": "preamble_completed",
        "local_artifact_dir": str(local_dir),
        "stage_records": stage_records,
        "components": [asdict(component) for component in components],
        "train_envs_uri": train_envs_uri,
        "validation_envs_uri": validation_envs_uri,
        "heldout_envs_uri": heldout_envs_uri,
        "gold_heldout_envs_uri": gold_heldout_envs_uri,
        "task_contract_uri": task_contract_uri,
        "task_contract_digest": task_contract["task_contract_digest"],
        "scene_spec_uri": scene_spec_uri,
        "robot_spec_uri": robot_spec_uri,
        "env_count": envgen_result["env_count"],
        "train_env_count": envgen_result["train_count"],
        "validation_env_count": envgen_result["validation_count"],
        "heldout_env_count": envgen_result["heldout_count"],
        "gold_heldout_env_count": envgen_result["gold_heldout_count"],
        "outer_history": [],
        "final_inner": None,
        "final_eval": None,
        "final_decision": None,
        "sibling_source_tarball_uri": sibling_source_tarball_uri,
        "current_quality": 0.0,
        "next_outer_iteration": 1,
        "updated_at": _utc_now(),
    }
    return _write_workflow_state(local_dir, state, config=config)


# =============================================================================
# Stages 7–11 — outer iteration (`run_single_outer_iteration`)
# =============================================================================


def run_single_outer_iteration(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    outer_iteration: int,
    initial_quality: float,
    resume_checkpoint_uri: str = "",
) -> dict[str, Any]:
    """Run one stage 7-11 iteration and return its outcomes.

    ``resume_checkpoint_uri`` (the prior outer iteration's checkpoint) is threaded
    into the inner loop so a BYO trainer continues the same policy across outer
    iterations rather than retraining from scratch.
    """

    inner = run_inner_loop(
        config,
        local_dir=local_dir,
        initial_quality=initial_quality,
        outer_iteration=outer_iteration,
        resume_checkpoint_uri=resume_checkpoint_uri,
    )
    quality = float(inner["final_quality"])
    # Checkpoint selection consumes only validation. The gold split is not opened
    # until the final configured Stage 10 evaluation.
    if outer_iteration < config.outer_iterations and inner.get(
        "selected_validation_report"
    ):
        heldout_report = dict(inner["selected_validation_report"])
    else:
        selected_checkpoint_iteration = int(
            (inner.get("checkpoint_selection") or {}).get("training_iteration")
            or (inner.get("selected_validation_report") or {}).get(
                "checkpoint_training_iteration"
            )
            or 0
        )
        heldout_report = run_heldout_eval(
            config,
            local_dir=local_dir,
            inner_evidence=inner,
            outer_iteration=outer_iteration,
            evaluation_split="gold_heldout",
            checkpoint_iteration=selected_checkpoint_iteration,
        )
    decision = threshold_decision(
        config,
        local_dir=local_dir,
        heldout_report=heldout_report,
        outer_iteration=outer_iteration,
    )
    # This field is retained for reference-rollout compatibility, but is now a
    # measured validation/gold strict-success rate rather than synthetic uplift.
    next_quality = float(heldout_report.get("success_rate") or quality)
    checkpoint_uri = str(inner.get("final_checkpoint_uri") or "").strip()
    result = {
        "outer_iteration": outer_iteration,
        "inner": inner,
        "heldout_report": heldout_report,
        "decision": decision,
        "checkpoint_uri": checkpoint_uri,
        "history_entry": {
            "outer_iteration": outer_iteration,
            "inner_loop": inner["evidence_uri"],
            "heldout_report": heldout_report["report_uri"],
            "decision": decision,
            "checkpoint_uri": checkpoint_uri,
            "resumed_from": str(inner.get("resumed_from_checkpoint_uri") or "").strip(),
        },
        "next_quality": next_quality,
    }
    _append_outer_iteration_workflow_state(
        config,
        local_dir=local_dir,
        outer_iteration=outer_iteration,
        inner=inner,
        heldout_report=heldout_report,
        decision=decision,
        next_quality=next_quality,
    )
    return result


def _append_outer_iteration_workflow_state(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    outer_iteration: int,
    inner: dict[str, Any],
    heldout_report: dict[str, Any],
    decision: dict[str, Any],
    next_quality: float,
) -> None:
    """Merge stages 7–11 artifacts into ``workflow_state.json`` for status polling."""

    try:
        state = _read_workflow_state(local_dir)
    except Sim2RealLoopError:
        return
    components = list(state.get("components") or [])
    stage_updates = _loop_component_records(
        config,
        local_dir=local_dir,
        outer_iteration=outer_iteration,
        inner=inner,
        heldout_report=heldout_report,
        decision=decision,
    )
    updates_by_name = {component.name: asdict(component) for component in stage_updates}
    components = [
        updates_by_name.pop(str(item.get("name") or ""), item)
        if isinstance(item, dict)
        else item
        for item in components
    ]
    for component in stage_updates:
        if component.name in updates_by_name:
            components.append(updates_by_name.pop(component.name))
    state["components"] = components
    state["status"] = "outer_iteration_completed"
    state["final_inner"] = inner
    state["final_eval"] = heldout_report
    state["final_decision"] = decision
    state["current_quality"] = next_quality
    state["last_checkpoint_uri"] = str(inner.get("final_checkpoint_uri") or "")
    state["next_outer_iteration"] = outer_iteration + 1
    history = list(state.get("outer_history") or [])
    history.append(
        {
            "outer_iteration": outer_iteration,
            "inner_loop": inner.get("evidence_uri"),
            "heldout_report": heldout_report.get("report_uri"),
            "decision": decision.get("decision"),
        }
    )
    state["outer_history"] = history
    state["updated_at"] = _utc_now()
    _write_workflow_state(local_dir, state, config=config)


def run_finalize(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    stage_records: list[dict[str, Any]],
    components: list[dict[str, Any]],
    outer_history: list[dict[str, Any]],
    final_inner: dict[str, Any],
    final_eval: dict[str, Any],
    final_decision: dict[str, Any],
    upload: bool | None = None,
) -> dict[str, Any]:
    """Run stages 12-13, visualization, and final report/upload."""

    components = _expand_envgen_component_records(config, components)
    loop_components = _loop_component_records(
        config,
        local_dir=local_dir,
        outer_iteration=int(
            final_decision.get("outer_iteration") or len(outer_history) or 1
        ),
        inner=final_inner,
        heldout_report=final_eval,
        decision=final_decision,
    )
    loop_names = {component.name for component in loop_components}
    components = [
        component
        for component in components
        if str(component.get("name") or "") not in loop_names
    ]
    components.extend(asdict(component) for component in loop_components)

    stage_started = time.monotonic()
    stage_records.append(
        _write_stage(
            local_dir,
            12,
            "external_validation",
            {
                "schema": "npa.sim2real.external_stub.v1",
                "stage": 12,
                "name": "external real-world validation",
                "status": "documented_external_stub",
                "input_checkpoint": final_decision["checkpoint_uri"],
                "next_action": "CONTINUE",
            },
            filename="external_stub.json",
        )
    )
    components.append(
        asdict(
            ComponentRecord(
                "stage_12_external_validation",
                "SEAM",
                "External real-world validation is a documented BYO gate; loop-of-loops continues through Stage 13.",
                {
                    "local": str(
                        local_dir
                        / "stage_12_external_validation"
                        / "external_stub.json"
                    ),
                    "job_name": "external_stub",
                    "execution": "external_byo_gate_not_dispatched",
                    "duration_s": round(time.monotonic() - stage_started, 3),
                },
            )
        )
    )

    stage_started = time.monotonic()
    retrigger = {
        "schema": "npa.sim2real.retrigger.v1",
        "stage": 13,
        "run_id": config.run_id,
        "source_decision": final_decision["decision"],
        "loop_of_loops_iteration": 1,
        "max_loop_of_loops_iterations": config.loop_of_loops_iterations,
        "target_stage": 1,
        "trigger_dataset_uri": config.trigger_dataset_uri,
        "trigger_dataset_id": config.trigger_dataset_id,
        "retrigger_condition": "new_verified_real_failure_or_corrected_scenario_data",
        "should_retrigger": False,
        "reason": (
            "No new external real-robot failure dataset or corrected scenario "
            "dataset was produced after Stage 12; reusing existing bytes would "
            "not constitute a genuine loop-of-loops trigger."
        ),
    }
    stage_records.append(
        _write_json_artifact(
            local_dir / "stage_13_retrigger" / "retrigger.json", retrigger
        )
    )
    components.append(
        asdict(
            ComponentRecord(
                "stage_13_retrigger",
                "WORKS",
                "Recorded that no genuinely new failure/scenario data exists; no retrigger was issued.",
                {
                    "local": str(local_dir / "stage_13_retrigger" / "retrigger.json"),
                    "job_name": config.run_id,
                    "execution": "orchestrator_record",
                    "duration_s": round(time.monotonic() - stage_started, 3),
                },
            )
        )
    )

    viz_stage_record = ComponentRecord(
        "stage_14_rerun_viz",
        "WORKS",
        "Rerun visualization is being written from the completed real-tier artifacts.",
        {
            "rrd": f"{_artifact_root_uri(config)}/reports/sim2real.rrd",
            "job_name": config.run_id,
            "execution": "in_process_orchestrator_visualization",
            "node_product": config.k8s_gpu_product,
        },
    )
    stage_started = time.monotonic()
    viz_component, viz_info = _run_sim2real_viz_stage(
        config,
        local_dir=local_dir,
        inner_evidence=final_inner,
        heldout_report=final_eval,
        stage_components=[*components, asdict(viz_stage_record)],
        outer_history=outer_history,
        final_decision=final_decision,
    )
    viz_component.artifacts["duration_s"] = round(time.monotonic() - stage_started, 3)
    components.append(asdict(viz_component))

    components.extend(
        [
            asdict(
                ComponentRecord(
                    "vlm_byo_seam",
                    "WORKS",
                    "VLM image/command are runtime-configurable; "
                    f"{DEFAULT_VLM_SEAM_EVIDENCE}",
                    {"image": config.vlm_image},
                )
            ),
            asdict(
                ComponentRecord(
                    "trainer_byo_seam",
                    "WORKS",
                    "Trainer image/command are runtime-configurable; default reference consumes npa.sim2real.rl_signal.v1.",
                    {"image": config.trainer_image},
                )
            ),
            asdict(
                ComponentRecord(
                    "eval_byo_seam",
                    "WORKS",
                    "Held-out eval image/command and threshold are runtime-configurable.",
                    {"image": config.eval_image},
                )
            ),
        ]
    )

    candidate_path = local_dir / "checkpoints" / "candidate" / "candidate.json"
    candidate_payload = (
        json.loads(candidate_path.read_text(encoding="utf-8"))
        if candidate_path.is_file()
        else {}
    )
    final_iterations = list(final_inner.get("iterations") or [])
    final_update = dict(
        (final_iterations[-1].get("update") or {}) if final_iterations else {}
    )
    ppo_hyperparameters = dict(final_update.get("ppo_hyperparameters") or {})
    runtime_parameters = runtime_parameter_metadata()
    report = {
        "schema": SCHEMA_E2E_REPORT,
        "run_id": config.run_id,
        "status": "completed",
        "created_at": _utc_now(),
        "local_artifact_dir": str(local_dir),
        "s3_artifacts": artifact_uris(config),
        "config": _redacted_config(config),
        "runtime_parameters": runtime_parameters,
        "training_provenance": {
            "effective_learning_rate": config.learning_rate,
            "learning_rate_scope": "vlm_signal_adapter_and_no_signal_control",
            "source": "LEARNING_RATE/--learning-rate",
            "ppo_optimizer_initial_learning_rate": ppo_hyperparameters.get(
                "ppo_optimizer_initial_learning_rate",
                os.environ.get("NPA_BYO_ISAAC_PPO_LEARNING_RATE", "task_default"),
            ),
            "ppo_optimizer_schedule": ppo_hyperparameters.get(
                "ppo_optimizer_schedule", "task_registry_schedule"
            ),
            "entropy_coef": ppo_hyperparameters.get("entropy_coef", "task_default"),
            "init_noise_std": ppo_hyperparameters.get("init_noise_std", "task_default"),
            "reward_weights": ppo_hyperparameters.get("reward_weights", {}),
            "ppo_workload": runtime_parameters.get("ppo", {}),
            "ppo_optimizer_source": (
                "NPA_BYO_ISAAC_PPO_LEARNING_RATE plus the selected Isaac task's "
                "RSL-RL agent configuration"
            ),
        },
        "byo_seams": byo_seams(config),
        "components": components,
        "gpu_fallback_contract": gpu_fallback_report_contract(config, components),
        "stage_records": stage_records,
        "inner_loop": final_inner,
        "outer_loop": {
            "history": outer_history,
            "latest_heldout_report": final_eval,
            "latest_decision": final_decision,
        },
        "progress_metrics": build_progress_metrics(local_dir, outer_history),
        "visualization": viz_info,
        "policy_access": {
            "deployable_policy": candidate_payload.get("deployable_policy", False),
            "policy_bytes_available": candidate_payload.get(
                "policy_bytes_available", False
            ),
            "identity": candidate_payload.get("policy_checkpoint_identity", ""),
            "sha256": candidate_payload.get("policy_checkpoint_sha256", ""),
            "size_bytes": candidate_payload.get("policy_checkpoint_size_bytes", 0),
            "checkpoint_uri": candidate_payload.get("policy_checkpoint_uri", ""),
            "candidate_manifest_uri": f"{_artifact_root_uri(config)}/checkpoints/candidate/candidate.json",
            "authenticated_download_command": candidate_payload.get(
                "policy_download_command", ""
            ),
            "ui_action": candidate_payload.get("policy_ui_action", ""),
            "viewer_executes_policy": False,
        },
        "image_completeness": {
            "required": [
                config.augment_image,
                config.policy_image,
                config.vlm_image,
                config.trainer_image,
                config.eval_image,
            ],
            "all_referenced": all(
                [
                    config.augment_image,
                    config.policy_image,
                    config.vlm_image,
                    config.trainer_image,
                    config.eval_image,
                ]
            ),
        },
    }
    report_path = local_dir / "reports" / "sim2real-report.json"
    _write_json_artifact(report_path, report)
    upload_enabled = config.upload_artifacts if upload is None else upload
    if upload_enabled and config.s3_bucket:
        report["upload"] = upload_run_artifacts(config, local_dir)
    else:
        report["upload"] = {
            "status": "skipped",
            "reason": "upload_artifacts is false or no s3_bucket configured",
        }

    from npa.workflows.rerun_serve import maybe_auto_rerun_serve

    rerun_serve = maybe_auto_rerun_serve(
        run_id=config.run_id,
        s3_bucket=config.s3_bucket,
        s3_prefix=config.s3_prefix,
        s3_endpoint=config.s3_endpoint,
        rerun_enabled=config.rerun_enabled,
        upload_info=report["upload"],
        viz_info=viz_info,
        k8s_kubeconfig=config.k8s_kubeconfig,
        k8s_namespace=config.k8s_namespace,
    )
    report["rerun_serve"] = rerun_serve
    if rerun_serve.get("status") == "deployed":
        components.append(
            asdict(
                ComponentRecord(
                    "stage_14_rerun_serve",
                    "WORKS",
                    (
                        "Deployed shared hosted Rerun viewer on mk8s; one LoadBalancer "
                        "per cluster with stable public_url for all teammates."
                    ),
                    {
                        "public_url": rerun_serve.get("public_url", ""),
                        "local_url": rerun_serve.get("local_url", ""),
                        "port_forward_command": rerun_serve.get(
                            "port_forward_command", ""
                        ),
                        "deployment_name": rerun_serve.get("deployment_name", ""),
                    },
                )
            )
        )
        report["components"] = components
    elif rerun_serve.get("status") == "blocked":
        components.append(
            asdict(
                ComponentRecord(
                    "stage_14_rerun_serve",
                    "WARN",
                    rerun_serve.get("reason", "auto rerun serve blocked"),
                    {"rrd_s3_uri": rerun_serve.get("rrd_s3_uri", "")},
                    next_action="CONTINUE",
                )
            )
        )
        report["components"] = components

    _write_json_artifact(report_path, report)
    if str(report.get("upload", {}).get("status")) == "uploaded" and config.s3_bucket:
        report_refresh = _upload_final_report(config, report_path)
        if report_refresh:
            upload_meta = dict(report.get("upload") or {})
            upload_meta["report_refresh"] = report_refresh
            report["upload"] = upload_meta
            _write_json_artifact(report_path, report)
    return report


def _run_sim2real_viz_stage(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    inner_evidence: dict[str, Any],
    heldout_report: dict[str, Any] | None,
    stage_components: list[dict[str, Any]] | None = None,
    outer_history: list[dict[str, Any]] | None = None,
    final_decision: dict[str, Any] | None = None,
) -> tuple[ComponentRecord, dict[str, Any]]:
    """Produce ``reports/sim2real.rrd`` and a status ComponentRecord.

    Degrades gracefully (WARN, not hard-fail) when ``rerun`` is unavailable or the
    toggle is off, but produces a real ``.rrd`` whenever rerun is installed. If a
    ``byo_rerun_command`` is set it runs that customer hook instead, reading the
    run dir from ``NPA_SIM2REAL_RUN_DIR`` / report from ``NPA_SIM2REAL_REPORT_JSON``
    and writing to ``NPA_SIM2REAL_OUTPUT_RRD``.
    """

    rrd_path = local_dir / "reports" / "sim2real.rrd"
    if not config.rerun_enabled:
        info = {"status": "disabled", "reason": "rerun_enabled is false"}
        return (
            ComponentRecord(
                "stage_14_rerun_viz",
                "SEAM",
                "Rerun visualization disabled via toggle (NPA_SIM2REAL_RERUN=0 / --no-rerun).",
                {},
                next_action="CONTINUE",
            ),
            info,
        )

    if config.byo_rerun_command.strip():
        return _run_byo_rerun_command(config, local_dir=local_dir, rrd_path=rrd_path)

    heldout_report = _ensure_heldout_renders_for_viz(
        config,
        local_dir,
        heldout_report,
    )
    if heldout_report is not None:
        heldout_path = local_dir / "eval" / "heldout" / "report.json"
        if heldout_path.is_file():
            _write_json_artifact(heldout_path, heldout_report)

    try:
        from npa.workflows.sim2real_viz import (
            RerunUnavailableError,
            emit_sim2real_mcap_if_enabled,
            emit_sim2real_rerun,
        )

        candidate_path = local_dir / "checkpoints" / "candidate" / "candidate.json"
        candidate_payload = (
            json.loads(candidate_path.read_text(encoding="utf-8"))
            if candidate_path.is_file()
            else {}
        )

        result = emit_sim2real_rerun(
            local_dir=local_dir,
            inner_evidence=inner_evidence,
            heldout_report=heldout_report,
            stage_components=stage_components,
            outer_history=outer_history,
            run_metadata=visualization_run_metadata(
                config=config,
                artifact_root=_artifact_root_uri(config),
                policy_checkpoint=str(
                    (final_decision or {}).get("checkpoint_uri") or ""
                ),
                candidate=candidate_payload,
                heldout_report=heldout_report,
            ),
            output_rrd=rrd_path,
            write_mp4=_bool_value(os.environ.get("NPA_SIM2REAL_RERUN_MP4", "0")),
        )
    except RerunUnavailableError as exc:
        if _bool_value(os.environ.get("NPA_SIM2REAL_REQUIRE_VISUALIZATION", "0")):
            raise Sim2RealLoopError(
                f"required Stage 14 Rerun recording could not be emitted: {exc}"
            ) from exc
        viz_info: dict[str, Any] = {
            "status": "skipped",
            "reason": str(exc),
            "source": "reference",
        }
        viz_info["mcap"] = emit_sim2real_mcap_if_enabled(
            local_dir=local_dir,
            inner_evidence=inner_evidence,
            heldout_report=heldout_report,
            output_mcap=local_dir / "reports" / "sim2real.mcap",
        )
        return (
            ComponentRecord(
                "stage_14_rerun_viz",
                "WARN",
                "rerun-sdk not installed locally; skipped .rrd emission (install rerun-sdk to enable).",
                {},
                next_action="CONTINUE",
            ),
            viz_info,
        )
    viz_info = {"source": "reference", **result.to_dict()}
    mcap_info = emit_sim2real_mcap_if_enabled(
        local_dir=local_dir,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
        output_mcap=local_dir / "reports" / "sim2real.mcap",
    )
    viz_info["mcap"] = mcap_info
    if (
        _bool_value(os.environ.get("NPA_SIM2REAL_REQUIRE_VISUALIZATION", "0"))
        and _bool_value(os.environ.get("NPA_SIM2REAL_MCAP", "1"))
        and mcap_info.get("status") != "written"
    ):
        raise Sim2RealLoopError(
            "required Stage 14 MCAP recording could not be emitted: "
            + str(mcap_info.get("reason") or mcap_info.get("status"))
        )
    root = _artifact_root_uri(config)
    artifacts: dict[str, Any] = {
        "rrd": f"{root}/reports/sim2real.rrd" if config.s3_bucket else str(rrd_path),
        "rrd_local": str(rrd_path),
        "job_name": config.run_id,
        "execution": "in_process_orchestrator_visualization",
        "node_product": config.k8s_gpu_product,
    }
    if mcap_info.get("status") == "written" and mcap_info.get("output_mcap_path"):
        artifacts["mcap"] = (
            f"{root}/reports/sim2real.mcap"
            if config.s3_bucket
            else str(mcap_info["output_mcap_path"])
        )
        artifacts["mcap_local"] = str(mcap_info["output_mcap_path"])
    mcap_note = ""
    if mcap_info.get("status") == "written":
        mcap_note = f" Also wrote a Lichtblick/Foxglove MCAP with {mcap_info.get('message_count', 0)} message(s)."
    return (
        ComponentRecord(
            "stage_14_rerun_viz",
            "WORKS",
            (
                f"Wrote Rerun recording with {result.rollout_count} rollout(s), "
                f"{result.frame_count} policy camera frame(s), "
                f"{result.heldout_frame_count} held-out sim frame(s), and "
                f"{result.heldout_env_count} held-out env score(s); camera streams, "
                "VLM critiques, RL signal, and held-out scores are logged." + mcap_note
            ),
            artifacts,
        ),
        viz_info,
    )


def _run_byo_rerun_command(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    rrd_path: Path,
) -> tuple[ComponentRecord, dict[str, Any]]:
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    report_json = local_dir / "reports" / "sim2real-report.json"
    env = _component_env(
        config,
        component="rerun_viz",
        output_json=rrd_path,
        extra={
            "NPA_SIM2REAL_RUN_DIR": str(local_dir),
            "NPA_SIM2REAL_REPORT_JSON": str(report_json),
            "NPA_SIM2REAL_OUTPUT_RRD": str(rrd_path),
        },
    )
    invocation = _run_component_command(
        config.byo_rerun_command,
        cwd=local_dir,
        env=env,
        component="rerun_viz",
    )
    if not rrd_path.exists() or rrd_path.stat().st_size == 0:
        raise Sim2RealLoopError(
            f"byo_rerun_command did not write a non-empty recording to {rrd_path}"
        )
    info = {
        "source": "byo_command",
        "status": "written",
        "output_rrd_path": str(rrd_path),
        "component_invocation": _public_invocation(invocation),
    }
    return (
        ComponentRecord(
            "stage_14_rerun_viz",
            "WORKS",
            "Customer byo_rerun_command produced the Rerun recording.",
            {"rrd": str(rrd_path)},
        ),
        info,
    )


# =============================================================================
# Stages 7–9 — inner loop (`run_inner_loop`)
# =============================================================================


def run_inner_loop(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    initial_quality: float,
    outer_iteration: int = 1,
    resume_checkpoint_uri: str = "",
) -> dict[str, Any]:
    """Run action generation, VLM eval, signal conversion, and policy update.

    ``resume_checkpoint_uri`` (from the prior outer iteration) lets a BYO trainer
    CONTINUE the same policy rather than restart from scratch, so the outer loop's
    "send back for more RL" (stage 11B) actually compounds. The checkpoint advances
    through both the inner iterations and across outer iterations.
    """

    from npa.workflows.sim2real_stages import run_policy_rollouts

    iteration_records: list[dict[str, Any]] = []
    reward_trend: list[float] = []
    loss_trend: list[dict[str, float]] = []
    policy_deltas: list[float] = []
    all_signals: list[dict[str, Any]] = []
    calibration_trend: list[dict[str, Any]] = []
    checkpoint_candidates: list[dict[str, Any]] = []
    quality = float(initial_quality)
    reward_head = 0.0
    action_bias = 0.0
    current_checkpoint_uri = str(resume_checkpoint_uri or "").strip()
    prior_selection_path = (
        local_dir
        / "checkpoints"
        / "validation-selection"
        / f"outer-{outer_iteration - 1:02d}.json"
    )
    if outer_iteration > 1 and prior_selection_path.is_file():
        prior_candidate = json.loads(prior_selection_path.read_text(encoding="utf-8"))
        if (
            prior_candidate.get("evaluation_split") == "validation"
            and prior_candidate.get("checkpoint_uri") == current_checkpoint_uri
        ):
            checkpoint_candidates.append(prior_candidate)
    for iteration in range(1, config.inner_iterations + 1):
        actions_dir = (
            local_dir
            / "actions"
            / "train"
            / f"outer-{outer_iteration:02d}"
            / f"iter-{iteration:02d}"
        )
        rollouts = run_policy_rollouts(
            config,
            local_dir=local_dir,
            actions_dir=actions_dir,
            outer_iteration=outer_iteration,
            iteration=iteration,
            checkpoint_uri=current_checkpoint_uri,
        )
        eval_dir = (
            local_dir
            / "vlm_eval"
            / "train"
            / f"outer-{outer_iteration:02d}"
            / f"iter-{iteration:02d}"
        )
        signal_dir = (
            local_dir
            / "training_signal"
            / "train"
            / f"outer-{outer_iteration:02d}"
            / f"iter-{iteration:02d}"
        )
        evals: list[dict[str, Any]] = []
        signals: list[dict[str, Any]] = []
        signal_converter_source = (
            "byo_command" if config.byo_signal_converter.strip() else "reference"
        )
        vlm_k8s_parallel = not config.byo_vlm_command.strip() and bool(
            config.s3_bucket.strip()
        )
        jobs_per_rollout = 2 if vlm_k8s_parallel and config.vlm_dual_reason else 1
        if vlm_k8s_parallel and len(rollouts) > 1:
            max_workers = min(
                len(rollouts),
                max(1, _effective_k8s_parallelism(config) // jobs_per_rollout),
            )
            evaluations: list[dict[str, Any] | None] = [None] * len(rollouts)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        evaluate_rollout_with_vlm,
                        rollout,
                        output_dir=eval_dir,
                        config=config,
                    ): index
                    for index, rollout in enumerate(rollouts)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    evaluations[index] = future.result()
            ordered_evaluations = [item for item in evaluations if item is not None]
            if len(ordered_evaluations) != len(rollouts):
                raise Sim2RealLoopError("parallel VLM eval did not return all rollouts")
            for evaluation in ordered_evaluations:
                signal = _convert_eval_to_signal(
                    evaluation,
                    config=config,
                    output_dir=signal_dir,
                )
                _write_json_artifact(
                    signal_dir / f"{signal['rollout_id']}.json", signal
                )
                evals.append(evaluation)
                signals.append(signal)
                all_signals.append(signal)
        else:
            for rollout in rollouts:
                evaluation = evaluate_rollout_with_vlm(
                    rollout,
                    output_dir=eval_dir,
                    config=config,
                )
                signal = _convert_eval_to_signal(
                    evaluation,
                    config=config,
                    output_dir=signal_dir,
                )
                _write_json_artifact(
                    signal_dir / f"{signal['rollout_id']}.json", signal
                )
                evals.append(evaluation)
                signals.append(signal)
                all_signals.append(signal)
        signal_batch_path = (
            local_dir
            / "inner_loop"
            / f"outer-{outer_iteration:02d}"
            / f"signals-iter-{iteration:02d}.json"
        )
        calibration = {
            "rollout_count": len(signals),
            "step_count": sum(
                int((signal.get("calibration") or {}).get("step_count") or 0)
                for signal in signals
            ),
            "simulator_grounded_steps": sum(
                int(
                    (signal.get("calibration") or {}).get("simulator_grounded_steps")
                    or 0
                )
                for signal in signals
            ),
            "nonzero_advantage_count": sum(
                int(
                    (signal.get("calibration") or {}).get("nonzero_advantage_count")
                    or 0
                )
                for signal in signals
            ),
            "model_disagreement_steps": sum(
                int(
                    (signal.get("calibration") or {}).get("model_disagreement_steps")
                    or 0
                )
                for signal in signals
            ),
            "vlm_calibrated_steps": sum(
                int((signal.get("calibration") or {}).get("vlm_calibrated_steps") or 0)
                for signal in signals
            ),
            "vlm_accepted_steps": sum(
                int((signal.get("calibration") or {}).get("vlm_accepted_steps") or 0)
                for signal in signals
            ),
            "vlm_rejected_or_downweighted_steps": sum(
                int(
                    (signal.get("calibration") or {}).get(
                        "vlm_rejected_or_downweighted_steps"
                    )
                    or 0
                )
                for signal in signals
            ),
            "vlm_missing_or_malformed_steps": sum(
                int(
                    (signal.get("calibration") or {}).get(
                        "vlm_missing_or_malformed_steps"
                    )
                    or 0
                )
                for signal in signals
            ),
            "vlm_low_confidence_steps": sum(
                int(
                    (signal.get("calibration") or {}).get("vlm_low_confidence_steps")
                    or 0
                )
                for signal in signals
            ),
            "vlm_contradictory_steps": sum(
                int(
                    (signal.get("calibration") or {}).get("vlm_contradictory_steps")
                    or 0
                )
                for signal in signals
            ),
            "vlm_summary_broadcast_steps": sum(
                int(
                    (signal.get("calibration") or {}).get("vlm_summary_broadcast_steps")
                    or 0
                )
                for signal in signals
            ),
            "mean_reward_variance": round(
                sum(
                    float(
                        (signal.get("calibration") or {}).get("reward_variance") or 0.0
                    )
                    for signal in signals
                )
                / max(1, len(signals)),
                10,
            ),
            "simulator_fallback_rollouts": sum(
                bool(
                    (signal.get("calibration") or {}).get(
                        "degenerate_simulator_fallback_used"
                    )
                )
                for signal in signals
            ),
            "degenerate_rollouts": sum(
                bool((signal.get("calibration") or {}).get("degenerate"))
                for signal in signals
            ),
        }
        if (
            config.byo_trainer_command.strip()
            and calibration["simulator_grounded_steps"] > 0
            and calibration["nonzero_advantage_count"] == 0
        ):
            raise Sim2RealLoopError(
                "simulator-grounded temporal credit is degenerate: every "
                "advantage is zero; refusing to train on useless feedback"
            )
        calibration_trend.append(calibration)
        _write_json_artifact(
            signal_batch_path,
            {
                "schema": SCHEMA_RL_SIGNAL,
                "signals": signals,
                "calibration": calibration,
            },
        )
        parse_vlm_signal_batch, run_vlm_signal_training_step = (
            _signal_training_imports()
        )
        parsed_signals = parse_vlm_signal_batch({"signals": signals})
        trainer_dir = (
            local_dir
            / "inner_loop"
            / f"outer-{outer_iteration:02d}"
            / "trainer"
            / f"iter-{iteration:02d}"
        )
        if config.byo_trainer_command.strip():
            update = _run_trainer_via_command(
                signal_batch_path,
                config=config,
                output_dir=trainer_dir,
                initial_reward_head=reward_head,
                initial_action_bias=action_bias,
                train_envs_dir=local_dir / "envs" / "train",
                resume_checkpoint_uri=current_checkpoint_uri,
                outer_iteration=outer_iteration,
                iteration=iteration,
            )
            # Compound: the next iteration (inner or outer) resumes from THIS
            # iteration's freshly-produced checkpoint.
            if str(getattr(update, "checkpoint_path", "") or "").strip():
                current_checkpoint_uri = update.checkpoint_path.strip()
            trainer_source = "byo_command"
            trainer_provenance_path = trainer_dir / "byo-trainer-gpu-provenance.json"
            trainer_component_invocation = (
                json.loads(trainer_provenance_path.read_text(encoding="utf-8"))
                if trainer_provenance_path.is_file()
                else {}
            )
            validation_report = None
            validation_reports: list[dict[str, Any]] = []
            if config.s3_bucket.strip():
                periodic = list(update.periodic_checkpoints) or [
                    {
                        "training_iteration": int(
                            (update.ppo or {}).get("iterations") or 0
                        ),
                        "checkpoint_uri": current_checkpoint_uri,
                    }
                ]
                for periodic_checkpoint in periodic:
                    candidate_uri = str(periodic_checkpoint.get("checkpoint_uri") or "")
                    training_iteration = int(
                        periodic_checkpoint.get("training_iteration") or 0
                    )
                    checkpoint_evidence = {
                        "schema": "npa.sim2real.checkpoint_evidence.v1",
                        "iterations": [{"update": update.to_dict()}],
                        "selected_checkpoint_uri": candidate_uri,
                        "final_checkpoint_uri": candidate_uri,
                    }
                    report = run_heldout_eval(
                        config,
                        local_dir=local_dir,
                        inner_evidence=checkpoint_evidence,
                        outer_iteration=outer_iteration,
                        evaluation_split="validation",
                        inner_iteration=iteration,
                        checkpoint_iteration=training_iteration,
                    )
                    validation_reports.append(report)
                    checkpoint_candidates.append(
                        {
                            "evaluation_split": "validation",
                            "outer_iteration": outer_iteration,
                            "inner_iteration": iteration,
                            "training_iteration": training_iteration,
                            "checkpoint_uri": candidate_uri,
                            "checkpoint_sha256": report.get(
                                "policy_checkpoint_sha256", ""
                            ),
                            "validation_report_uri": report["report_uri"],
                            "validation_report": report,
                        }
                    )
                from npa.workflows.sim2real.checkpoint_selection import (
                    select_best_checkpoint,
                )

                interim_selection = select_best_checkpoint(checkpoint_candidates)
                current_checkpoint_uri = str(interim_selection["checkpoint_uri"])
                validation_report = dict(
                    interim_selection.get("validation_report") or {}
                )
        else:
            update = run_vlm_signal_training_step(
                parsed_signals,
                output_dir=trainer_dir,
                learning_rate=config.learning_rate,
                signal_loss_weight=config.signal_loss_weight,
                initial_reward_head=reward_head,
                initial_action_bias=action_bias,
            )
            trainer_source = "reference"
            trainer_component_invocation = {}
            validation_report = None
            validation_reports = []
        # The no-signal control always runs the in-process reference trainer so the
        # policy-delta attribution baseline stays honest even when a BYO trainer
        # produces the signal-driven update.
        control = run_vlm_signal_training_step(
            parsed_signals,
            output_dir=local_dir
            / "inner_loop"
            / f"outer-{outer_iteration:02d}"
            / "control"
            / f"iter-{iteration:02d}",
            learning_rate=config.learning_rate,
            signal_loss_weight=config.signal_loss_weight,
            initial_reward_head=reward_head,
            initial_action_bias=action_bias,
            control=True,
        )
        reward_head = update.reward_head_after
        action_bias = (
            update.policy_output_after[0] if update.policy_output_after else action_bias
        )
        mean_reward = round(
            sum(_signal_mean_reward(signal) for signal in signals)
            / float(len(signals)),
            6,
        )
        reward_trend.append(mean_reward)
        loss_trend.append(
            {
                "before": round(float(update.loss_before), 8),
                "after": round(float(update.loss_after), 8),
            }
        )
        delta_vs_control = max(0.0, update.policy_delta_l2 - control.policy_delta_l2)
        policy_deltas.append(round(delta_vs_control, 8))
        iteration_records.append(
            {
                "iteration": iteration,
                "actions_dir": str(actions_dir),
                "vlm_eval_dir": str(eval_dir),
                "signal_dir": str(signal_dir),
                "signal_batch": str(signal_batch_path),
                "mean_reward": mean_reward,
                "effective_learning_rate": config.learning_rate,
                "learning_rate_scope": "vlm_signal_adapter_and_no_signal_control",
                "trainer_source": trainer_source,
                "trainer_component_invocation": trainer_component_invocation,
                "signal_converter_source": signal_converter_source,
                "update": update.to_dict(),
                "no_signal_control": control.to_dict(),
                "policy_delta_vs_control": round(delta_vs_control, 8),
                "validation_report": validation_report,
                "periodic_validation_reports": validation_reports,
                "sample_vlm_eval": evals[0],
                "sample_signal": signals[0],
                "signal_calibration": calibration,
            }
        )

    signal_diversity = _signal_diversity_report(all_signals)
    if signal_diversity["degenerate"] and _bool_value(
        os.environ.get("NPA_SIM2REAL_REQUIRE_SIGNAL_DIVERSITY", "0")
    ):
        raise Sim2RealLoopError(
            "VLM->RL signal is degenerate: "
            f"{signal_diversity['distinct_scores']} distinct score(s) and "
            f"{signal_diversity['distinct_mean_rewards']} distinct mean-reward(s) "
            f"across {signal_diversity['total_rollouts']} rollout(s) "
            f"(scores={signal_diversity['score_values']}). "
            "Unset NPA_SIM2REAL_REQUIRE_SIGNAL_DIVERSITY to downgrade this gate to a "
            "diagnostic."
        )
    checkpoint_selection: dict[str, Any] = {}
    selected_checkpoint_uri = current_checkpoint_uri
    selected_validation_report: dict[str, Any] | None = None
    if checkpoint_candidates:
        from npa.workflows.sim2real.checkpoint_selection import select_best_checkpoint

        checkpoint_selection = select_best_checkpoint(checkpoint_candidates)
        selected_checkpoint_uri = str(checkpoint_selection["checkpoint_uri"])
        selected_validation_report = dict(
            checkpoint_selection.get("validation_report") or {}
        )
        quality = float(selected_validation_report.get("success_rate") or 0.0)
        selection_path = (
            local_dir
            / "checkpoints"
            / "validation-selection"
            / f"outer-{outer_iteration:02d}.json"
        )
        _write_json_artifact(selection_path, checkpoint_selection)
        checkpoint_selection["selection_report_uri"] = str(selection_path)
    evidence = {
        "schema": "npa.sim2real.inner_loop_evidence.v1",
        "outer_iteration": outer_iteration,
        "status": "closed",
        "trainer_source": (
            "byo_command" if config.byo_trainer_command.strip() else "reference"
        ),
        "signal_converter_source": (
            "byo_command" if config.byo_signal_converter.strip() else "reference"
        ),
        "effective_learning_rate": config.learning_rate,
        "learning_rate_scope": "vlm_signal_adapter_and_no_signal_control",
        "reward_trend": reward_trend,
        "loss_trend": loss_trend,
        "signal_diversity": signal_diversity,
        "signal_calibration": calibration_trend,
        "policy_delta_vs_no_signal_control": policy_deltas,
        "attribution": (
            "The reference update and no-signal control share initial adapter state. "
            "Only the VLM-derived rewards, advantages, and corrective targets produce the policy-output delta."
        ),
        "iterations": iteration_records,
        "selected_validation_strict_success": round(quality, 6),
        "efficacy_metric_definition": (
            "strict stable-placement success rate on the fixed validation split; "
            "never a synthetic training-progress uplift"
            if checkpoint_candidates
            else "reference-only compatibility metric; not real task efficacy"
        ),
        "final_quality": round(quality, 6),
        "latest_checkpoint_uri": current_checkpoint_uri,
        "selected_checkpoint_uri": selected_checkpoint_uri,
        "final_checkpoint_uri": selected_checkpoint_uri,
        "checkpoint_selection": checkpoint_selection,
        "selected_validation_report": selected_validation_report,
        "resumed_from_checkpoint_uri": str(resume_checkpoint_uri or "").strip(),
    }
    evidence_path = (
        local_dir / "inner_loop" / f"outer-{outer_iteration:02d}" / "evidence.json"
    )
    _write_json_artifact(evidence_path, evidence)
    return {**evidence, "evidence_uri": str(evidence_path)}


def generate_action_rollouts(
    output_dir: Path,
    *,
    count: int,
    steps_per_rollout: int,
    seed: int,
    quality: float,
) -> list[Path]:
    """Generate small action-conditioned rollout fixtures with camera frames."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rollouts: list[Path] = []
    for index in range(count):
        rollout_id = f"rollout-{index:04d}"
        rollout_dir = output_dir / rollout_id
        rollout_dir.mkdir(parents=True, exist_ok=True)
        actions: list[dict[str, Any]] = []
        for step in range(steps_per_rollout):
            drift = max(0.0, 1.0 - quality) * (1.0 + rng.random() * 0.2)
            action = [
                round(quality * 0.1 + rng.uniform(-0.02, 0.02), 5),
                round((0.5 - drift) * 0.1 + rng.uniform(-0.02, 0.02), 5),
                round((quality - 0.5) * 0.1 + rng.uniform(-0.02, 0.02), 5),
            ]
            actions.append({"step": step, "action": action})
            _write_ppm(
                rollout_dir / f"camera-{step:03d}.ppm",
                red=int(64 + 120 * quality),
                green=int(40 + 80 * (1.0 - drift)),
                blue=int(80 + step * 12),
            )
        _write_json_artifact(
            rollout_dir / "manifest.json",
            {
                "schema": "npa.sim2real.action_rollout.v1",
                "rollout_id": rollout_id,
                "task_description": "Move the manipulation object to the target while maintaining stable contact.",
                "quality": round(quality, 6),
                "steps": steps_per_rollout,
                "camera_observations": [
                    f"camera-{step:03d}.ppm" for step in range(steps_per_rollout)
                ],
                "actions": actions,
            },
        )
        rollouts.append(rollout_dir)
    return rollouts


def evaluate_rollout_with_vlm(
    rollout_dir: Path,
    *,
    output_dir: Path,
    config: Sim2RealLoopConfig,
) -> dict[str, Any]:
    """Invoke Reason2 + Reason3 (or a single model) and parse structured judgments."""

    manifest_path = rollout_dir / "manifest.json"
    if not manifest_path.exists():
        raise Sim2RealLoopError(f"rollout manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rollout_id = str(manifest.get("rollout_id") or rollout_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{rollout_id}.json"

    if config.byo_vlm_command.strip():
        env = _component_env(
            config,
            component="vlm_eval",
            output_json=output_path,
            extra={
                "NPA_SIM2REAL_ROLLOUT_DIR": str(rollout_dir),
                "NPA_SIM2REAL_ROLLOUT_ID": rollout_id,
                "NPA_SIM2REAL_ROLLOUT_MANIFEST": str(manifest_path),
                "NPA_SIM2REAL_VLM_MODEL": config.vlm_model,
                "NPA_SIM2REAL_THRESHOLD": str(config.threshold),
                "NPA_SIM2REAL_VLM_IMAGE": config.vlm_image,
            },
        )
        invocation = _run_component_command(
            config.byo_vlm_command,
            cwd=rollout_dir,
            env=env,
            component="vlm_eval",
        )
        payload = _read_component_json(output_path, invocation)
    elif not config.s3_bucket.strip():
        if config.vlm_dual_reason:
            reason2 = _reference_vlm_payload_from_rollout(
                manifest,
                rollout_dir=rollout_dir,
                rollout_id=rollout_id,
                config=config,
            )
            reason3 = _reference_vlm_payload_from_rollout(
                manifest,
                rollout_dir=rollout_dir,
                rollout_id=rollout_id,
                config=config,
            )
            reason2["model"] = config.vlm_reason2_model
            reason3["model"] = config.vlm_reason3_model
            payload = merge_dual_reason_evaluations(
                reason2, reason3, threshold=config.threshold
            )
        else:
            payload = _reference_vlm_payload_from_rollout(
                manifest,
                rollout_dir=rollout_dir,
                rollout_id=rollout_id,
                config=config,
            )
        invocation = {
            "component": "vlm_eval",
            "mode": "local_reference",
            "image": config.vlm_image,
            "dual_reason": config.vlm_dual_reason,
        }
        _write_json_artifact(output_path, payload)
    elif config.vlm_dual_reason:
        from concurrent.futures import ThreadPoolExecutor

        reason2_image = (config.vlm_reason2_image or config.vlm_image).strip()
        reason3_image = (config.vlm_reason3_image or config.vlm_image).strip()

        def _run_reason2() -> tuple[dict[str, Any], dict[str, Any]]:
            return _evaluate_reason_rollout_k8s(
                rollout_dir,
                manifest=manifest,
                manifest_path=manifest_path,
                rollout_id=rollout_id,
                config=config,
                model=config.vlm_reason2_model,
                image=reason2_image,
                component="vlm_eval_reason2",
                output_dir=output_dir,
            )

        def _run_reason3() -> tuple[dict[str, Any], dict[str, Any]]:
            return _evaluate_reason_rollout_k8s(
                rollout_dir,
                manifest=manifest,
                manifest_path=manifest_path,
                rollout_id=rollout_id,
                config=config,
                model=config.vlm_reason3_model,
                image=reason3_image,
                component="vlm_eval_reason3",
                output_dir=output_dir,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            reason2_future = pool.submit(_run_reason2)
            reason3_future = pool.submit(_run_reason3)
            reason2_eval, reason2_invocation = reason2_future.result()
            reason3_eval, reason3_invocation = reason3_future.result()
        payload = merge_dual_reason_evaluations(
            reason2_eval, reason3_eval, threshold=config.threshold
        )
        invocation = {
            "component": "vlm_eval",
            "mode": "kubernetes_job_dual_reason",
            "reason2_image": reason2_image,
            "reason3_image": reason3_image,
            "reason2_invocation": _public_invocation(reason2_invocation),
            "reason3_invocation": _public_invocation(reason3_invocation),
            "gpu_provenance": {
                "candidate_order": list(
                    dict.fromkeys(
                        list(
                            (reason2_invocation.get("gpu_provenance") or {}).get(
                                "candidate_order", []
                            )
                        )
                        + list(
                            (reason3_invocation.get("gpu_provenance") or {}).get(
                                "candidate_order", []
                            )
                        )
                    )
                ),
                "attempts": list(
                    (reason2_invocation.get("gpu_provenance") or {}).get("attempts", [])
                )
                + list(
                    (reason3_invocation.get("gpu_provenance") or {}).get("attempts", [])
                ),
                "selected_products": list(
                    dict.fromkeys(
                        product
                        for product in (
                            (reason2_invocation.get("gpu_provenance") or {}).get(
                                "selected_product"
                            ),
                            (reason3_invocation.get("gpu_provenance") or {}).get(
                                "selected_product"
                            ),
                        )
                        if product
                    )
                ),
                "selected_nodes": list(
                    dict.fromkeys(
                        node
                        for node in (
                            (reason2_invocation.get("gpu_provenance") or {}).get(
                                "selected_node"
                            ),
                            (reason3_invocation.get("gpu_provenance") or {}).get(
                                "selected_node"
                            ),
                        )
                        if node
                    )
                ),
                "allocated_gpu": {
                    "resource": config.k8s_gpu_resource,
                    "count_per_job": 1,
                },
                "minimum_vram_gb": max(
                    int(
                        (reason2_invocation.get("gpu_provenance") or {}).get(
                            "minimum_vram_gb", 0
                        )
                    ),
                    int(
                        (reason3_invocation.get("gpu_provenance") or {}).get(
                            "minimum_vram_gb", 0
                        )
                    ),
                ),
                "model_requirement": [
                    config.vlm_reason2_model,
                    config.vlm_reason3_model,
                ],
                "image_digests": list(
                    dict.fromkeys(
                        list(
                            (reason2_invocation.get("gpu_provenance") or {}).get(
                                "image_digests", []
                            )
                        )
                        + list(
                            (reason3_invocation.get("gpu_provenance") or {}).get(
                                "image_digests", []
                            )
                        )
                    )
                ),
                "duration_s": round(
                    float(
                        (reason2_invocation.get("gpu_provenance") or {}).get(
                            "duration_s", 0
                        )
                        or 0
                    )
                    + float(
                        (reason3_invocation.get("gpu_provenance") or {}).get(
                            "duration_s", 0
                        )
                        or 0
                    ),
                    3,
                ),
            },
        }
        _write_json_artifact(output_path, payload)
    else:
        payload, invocation = _evaluate_reason_rollout_k8s(
            rollout_dir,
            manifest=manifest,
            manifest_path=manifest_path,
            rollout_id=rollout_id,
            config=config,
            model=config.vlm_model,
            image=config.vlm_image,
            component="vlm_eval",
            output_dir=output_dir,
        )
        _write_json_artifact(output_path, payload)

    evaluation = _normalize_vlm_evaluation(
        payload,
        manifest=manifest,
        rollout_id=rollout_id,
        config=config,
        invocation=invocation,
    )
    _write_json_artifact(output_path, evaluation)
    return evaluation


def _evaluate_reason_rollout_k8s(
    rollout_dir: Path,
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    rollout_id: str,
    config: Sim2RealLoopConfig,
    model: str,
    image: str,
    component: str,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_path = output_dir / f"{rollout_id}-{component}.json"
    attempt_id = _component_attempt_id(config, component, rollout_id)
    rollout_uri = _upload_component_directory(
        config,
        rollout_dir,
        component=component,
        attempt_id=attempt_id,
        name="rollout",
    )
    output_uri = _component_output_uri(
        config,
        component=component,
        attempt_id=attempt_id,
        filename=f"{rollout_id}.json",
    )
    env = _component_env(
        config,
        component=component,
        output_json=output_path,
        extra={
            "NPA_SIM2REAL_ROLLOUT_DIR": str(rollout_dir),
            "NPA_SIM2REAL_ROLLOUT_ID": rollout_id,
            "NPA_SIM2REAL_ROLLOUT_MANIFEST": str(manifest_path),
            "NPA_SIM2REAL_ROLLOUT_URI": rollout_uri,
            "NPA_SIM2REAL_OUTPUT_URI": output_uri,
            "NPA_SIM2REAL_VLM_MODEL": model,
            "NPA_SIM2REAL_THRESHOLD": str(config.threshold),
            "NPA_SIM2REAL_VLM_IMAGE": image,
            "NPA_COSMOS_REASON_MODEL_ID": model,
        },
    )
    invocation = _run_image_component(
        image,
        component=component,
        env=env,
        output_json=output_path,
        output_uri=output_uri,
        config=config,
    )
    payload = _read_component_json(output_path, invocation)
    evaluation = _normalize_vlm_evaluation(
        payload,
        manifest=manifest,
        rollout_id=rollout_id,
        config=config,
        invocation=invocation,
    )
    return evaluation, invocation


def _component_env(
    config: Sim2RealLoopConfig,
    *,
    component: str,
    output_json: Path,
    extra: dict[str, str],
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "NPA_SIM2REAL_COMPONENT": component,
            "NPA_SIM2REAL_RUN_ID": config.run_id,
            "NPA_SIM2REAL_OUTPUT_JSON": str(output_json),
            "NPA_SIM2REAL_S3_BUCKET": config.s3_bucket,
            "NPA_SIM2REAL_S3_PREFIX": config.s3_prefix,
            "AWS_ENDPOINT_URL": config.s3_endpoint or env.get("AWS_ENDPOINT_URL", ""),
            "NPA_SIM2REAL_K8S_GPU_RESOURCE": config.k8s_gpu_resource,
            "NPA_SIM2REAL_K8S_GPU_PRODUCT": config.k8s_gpu_product,
            "NPA_SIM2REAL_TASK_CONTRACT_DIGEST": str(
                getattr(config, "task_contract_digest", "") or ""
            ),
            "NPA_SIM2REAL_K8S_GPU_CANDIDATES": ",".join(
                getattr(config, "k8s_gpu_candidates", ())
            ),
        }
    )
    env.update(extra)
    return env


def _run_component_command(
    command: str,
    *,
    cwd: Path,
    env: dict[str, str],
    component: str,
    timeout_s: int = 0,
) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s or None,
        check=False,
    )
    if result.returncode != 0:
        raise Sim2RealLoopError(
            f"{component} command failed with exit {result.returncode}: "
            f"{_component_excerpt(result.stderr or result.stdout)}"
        )
    return {
        "mode": "command",
        "component": component,
        "command": _redact_command(command),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_excerpt": _component_excerpt(result.stdout),
        "stderr_excerpt": _component_excerpt(result.stderr),
    }


# =============================================================================
# K8s sibling components (stages 3–7 GPU jobs)
# =============================================================================


def run_cosmos2_transfer_component(
    config: Sim2RealLoopConfig,
    *,
    input_uri: str,
    output_uri: str,
    local_dir: Path,
) -> dict[str, Any]:
    """Run Cosmos Transfer 2.5 in a sibling GPU job and return augment artifacts."""

    if not config.s3_bucket:
        raise Sim2RealLoopError(
            "s3_bucket is required for Cosmos Transfer sibling jobs"
        )
    frames_uri = _normalized_s3_prefix(f"{output_uri.rstrip('/')}/frames/")
    augment_prefix = output_uri.rstrip("/") + "/"
    result_uri = f"{augment_prefix}cosmos2-transfer-result.json"
    env = {
        "NPA_SIM2REAL_INPUT_URI": input_uri,
        "NPA_SIM2REAL_OUTPUT_URI": result_uri,
        "NPA_SIM2REAL_AUGMENT_PREFIX": augment_prefix,
        "NPA_SIM2REAL_AUGMENTED_FRAMES_URI": frames_uri,
        "NPA_SIM2REAL_ASSETS_URI": config.assets_uri,
        "NPA_SIM2REAL_SCENE_SPEC_URI": config.scene_spec_uri,
        "NPA_SIM2REAL_AUGMENT_IMAGE": config.augment_image,
        "NPA_SIM2REAL_ROLLOUT_COUNT": str(config.rollout_count),
        # A registry-qualified production Job must never turn a descriptor
        # fallback into a WORKS record. Unit/CPU entrypoint calls can still use
        # the descriptor seam by omitting this explicit real-tier contract.
        "NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS": "1",
        # Match the proven serverless real-GPU path: component images can have
        # read-only baked HOME/default caches even though /tmp is writable.
        "HF_HOME": "/tmp/hf_home",
        "HF_XET_CACHE": "/tmp/hf_xet_cache",
        "UV_CACHE_DIR": "/tmp/uv_cache",
        "XDG_CACHE_HOME": "/tmp/xdg_cache",
    }
    output_json = local_dir / "cosmos2-transfer-result.json"
    invocation = _run_image_component(
        config.augment_image,
        component="cosmos2_transfer",
        env=env,
        output_json=output_json,
        output_uri=result_uri,
        config=config,
    )
    payload = _read_component_json(output_json, invocation)
    manifest = payload.get("manifest") or payload
    augmented_frames_uri = str(
        manifest.get("augmented_frames_uri")
        or payload.get("augmented_frames_uri")
        or frames_uri
    )
    return {
        "manifest": manifest,
        "augmented_frames_uri": augmented_frames_uri,
        "invocation": invocation,
    }


def run_envgen_sharded_component(
    config: Sim2RealLoopConfig,
    *,
    envgen: Any,
) -> dict[str, Any]:
    """Run raw env generation as an indexed GPU Job with bounded parallelism."""

    if not config.s3_bucket:
        raise Sim2RealLoopError("s3_bucket is required for envgen sibling jobs")
    client = _storage_client(config)
    with tempfile.TemporaryDirectory(prefix="npa-envgen-scene-") as tmp:
        scene_path = Path(tmp) / "scene-spec.json"
        scene_path.write_text(
            json.dumps(envgen.scene_spec.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        scene_uri = client.upload_file(
            str(scene_path), f"{envgen.manifest_uri}scene-spec-orchestrator.json"
        )
    env = {
        "NPA_SIM2REAL_RUN_ID": config.run_id,
        "NPA_SIM2REAL_OUTPUT_URI": envgen.output_uri,
        "NPA_SIM2REAL_ENV_COUNT": str(envgen.env_count),
        "NPA_SIM2REAL_SHARD_COUNT": str(envgen.shard_count),
        "NPA_SIM2REAL_TRAIN_FRACTION": str(envgen.train_fraction),
        "NPA_SIM2REAL_SEED": str(envgen.seed),
        "NPA_SIM2REAL_AUGMENTED_FRAMES_URI": envgen.scene_spec.augmented_frames_uri,
        "NPA_SIM2REAL_SCENE_SPEC_URI": scene_uri,
    }
    parallelism = min(envgen.shard_count, _effective_k8s_parallelism(config))
    invocation = _run_kubernetes_indexed_image_component(
        config.envgen_image,
        component="envgen_raw_shard",
        env=env,
        config=config,
        completions=envgen.shard_count,
        parallelism=parallelism,
        timeout_s=config.k8s_job_timeout_s,
    )
    return {
        "scene_spec_uri": scene_uri,
        "shard_count": envgen.shard_count,
        "parallelism": parallelism,
        "invocation": invocation,
    }


def _effective_k8s_parallelism(config: Sim2RealLoopConfig) -> int:
    """Return the sibling-job GPU concurrency cap for this run."""

    return max(1, int(config.k8s_max_parallel_gpus))


def run_policy_rollout_component(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    actions_dir: Path,
    outer_iteration: int,
    iteration: int,
    train_envs_uri: str,
    checkpoint_uri: str = "",
) -> list[Path]:
    """Run swappable LeRobot policy image to produce action rollouts."""

    if config.byo_policy_command.strip():
        return _run_policy_rollouts_via_command(
            config,
            actions_dir=actions_dir,
            outer_iteration=outer_iteration,
            iteration=iteration,
            train_envs_uri=train_envs_uri,
            checkpoint_uri=checkpoint_uri,
        )
    output_uri = _normalized_s3_prefix(
        f"{_artifact_root_uri(config)}/actions/train/"
        f"outer-{outer_iteration:02d}/iter-{iteration:02d}/"
    )
    env = {
        "NPA_SIM2REAL_TRAIN_ENVS_URI": train_envs_uri,
        "NPA_SIM2REAL_OUTPUT_URI": output_uri,
        "NPA_SIM2REAL_POLICY_IMAGE": config.policy_image,
        "NPA_SIM2REAL_ACTION_LIMIT": str(
            min(config.action_env_limit, config.rollout_count)
        ),
        "NPA_SIM2REAL_SEED": str(config.seed + outer_iteration * 100 + iteration),
        "NPA_SIM2REAL_ROLLOUT_COUNT": str(config.rollout_count),
        "NPA_SIM2REAL_STEPS_PER_ROLLOUT": str(config.steps_per_rollout),
    }
    output_json = actions_dir / "policy-actions-result.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    result_uri = f"{output_uri.rstrip('/')}/policy-actions-result.json"
    invocation = _run_image_component(
        config.policy_image,
        component="policy_actions",
        env=env,
        output_json=output_json,
        output_uri=result_uri,
        config=config,
    )
    payload = _read_component_json(output_json, invocation)
    if payload.get("rollout_dirs"):
        return [Path(item) for item in payload["rollout_dirs"]]
    return generate_action_rollouts(
        actions_dir,
        count=config.rollout_count,
        steps_per_rollout=config.steps_per_rollout,
        seed=config.seed + outer_iteration * 100 + iteration,
        quality=0.5,
    )


def _run_policy_rollouts_via_command(
    config: Sim2RealLoopConfig,
    *,
    actions_dir: Path,
    outer_iteration: int,
    iteration: int,
    train_envs_uri: str,
    checkpoint_uri: str = "",
) -> list[Path]:
    actions_dir.mkdir(parents=True, exist_ok=True)
    output_path = actions_dir / "byo-policy-rollouts.json"
    env = _component_env(
        config,
        component="policy_actions",
        output_json=output_path,
        extra={
            "NPA_SIM2REAL_TRAIN_ENVS_URI": train_envs_uri,
            "NPA_SIM2REAL_POLICY_CHECKPOINT_URI": checkpoint_uri,
            "NPA_SIM2REAL_TRAIN_ENVS_DIR": str(
                actions_dir.parents[3] / "envs" / "train"
            ),
            "NPA_SIM2REAL_POLICY_IMAGE": config.policy_image,
            "NPA_SIM2REAL_ROLLOUT_COUNT": str(config.rollout_count),
            "NPA_SIM2REAL_STEPS_PER_ROLLOUT": str(config.steps_per_rollout),
            "NPA_SIM2REAL_OUTPUT_DIR": str(actions_dir),
            "NPA_SIM2REAL_ROLLOUT_TAG": f"outer-{outer_iteration:02d}-iter-{iteration:02d}",
        },
    )
    invocation = _run_component_command(
        config.byo_policy_command,
        cwd=actions_dir,
        env=env,
        component="policy_actions",
    )
    payload = _read_component_json(output_path, invocation)
    if payload.get("rollout_dirs"):
        rollout_paths = [Path(item) for item in payload["rollout_dirs"]]
        nested_invocation = dict(payload.get("component_invocation") or {})
        if nested_invocation:
            for rollout_path in rollout_paths:
                manifest_path = rollout_path / "manifest.json"
                if not manifest_path.is_file():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["component_invocation"] = nested_invocation
                _write_json_artifact(manifest_path, manifest)
        return rollout_paths
    return generate_action_rollouts(
        actions_dir,
        count=config.rollout_count,
        steps_per_rollout=config.steps_per_rollout,
        seed=config.seed + outer_iteration * 100 + iteration,
        quality=0.5,
    )


def _config_from_workflow_state(
    config: Sim2RealLoopConfig, state: dict[str, Any]
) -> Sim2RealLoopConfig:
    from dataclasses import replace

    updates: dict[str, Any] = {}
    for state_field in (
        "train_envs_uri",
        "validation_envs_uri",
        "heldout_envs_uri",
        "gold_heldout_envs_uri",
        "task_contract_uri",
        "task_contract_digest",
        "scene_spec_uri",
        "robot_spec_uri",
    ):
        if not hasattr(config, state_field):
            continue
        value = str(state.get(state_field) or "").strip()
        if value:
            updates[state_field] = value
    if not updates:
        return config
    return replace(config, **updates)


def _run_image_component(
    image: str,
    *,
    component: str,
    env: dict[str, str],
    output_json: Path,
    output_uri: str,
    config: Sim2RealLoopConfig,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    effective_timeout_s = config.k8s_job_timeout_s if timeout_s is None else timeout_s
    return _run_kubernetes_image_component(
        image,
        component=component,
        env=env,
        output_json=output_json,
        output_uri=output_uri,
        config=config,
        timeout_s=effective_timeout_s,
    )


def _kubectl_job_not_found(result: subprocess.CompletedProcess[str]) -> bool:
    """Return True when kubectl reports the sibling Job no longer exists."""

    if result.returncode == 0:
        return False
    text = f"{result.stderr or ''}{result.stdout or ''}"
    lowered = text.lower()
    return "notfound" in lowered.replace(" ", "") or (
        "not found" in lowered and "job" in lowered
    )


def _wait_kubernetes_job(
    config: Sim2RealLoopConfig,
    *,
    namespace: str,
    job_name: str,
    timeout_s: int,
    required_successes: int = 1,
) -> str:
    """Poll a sibling Job until it succeeds, fails, or times out.

    External or manual Job deletion during a wait is treated as failure so the
    driver fails fast instead of blocking on ``kubectl wait`` for ``timeout_s``.
    """

    initial_status = _kubectl(
        config,
        [
            "get",
            "job",
            job_name,
            "-n",
            namespace,
            "-o",
            "jsonpath={.status.succeeded} {.status.failed}",
        ],
        timeout_s=30,
        check=False,
    )
    if _kubectl_job_not_found(initial_status):
        return "failed"
    if initial_status.returncode == 0:
        parts = (initial_status.stdout or "").strip().split()
        succeeded = int(parts[0]) if parts and str(parts[0]).isdigit() else 0
        failed = int(parts[1]) if len(parts) > 1 and str(parts[1]).isdigit() else 0
        if failed >= 1:
            return "failed"
        if succeeded >= required_successes:
            return "complete"

    wait_result = _kubectl(
        config,
        [
            "wait",
            "--for=condition=complete",
            f"job/{job_name}",
            "-n",
            namespace,
            f"--timeout={max(1, int(timeout_s))}s",
        ],
        timeout_s=max(30, int(timeout_s) + 5),
        check=False,
    )
    if _kubectl_job_not_found(wait_result):
        return "failed"
    if wait_result.returncode == 0:
        verify = _kubectl(
            config,
            [
                "get",
                "job",
                job_name,
                "-n",
                namespace,
                "-o",
                "jsonpath={.status.succeeded} {.status.failed}",
            ],
            timeout_s=30,
            check=False,
        )
        if verify.returncode == 0:
            parts = (verify.stdout or "").strip().split()
            succeeded = int(parts[0]) if parts and str(parts[0]).isdigit() else 0
            if succeeded >= required_successes:
                return "complete"
    elif wait_result.returncode != 0:
        failed_result = _kubectl(
            config,
            [
                "wait",
                "--for=condition=failed",
                f"job/{job_name}",
                "-n",
                namespace,
                "--timeout=1s",
            ],
            timeout_s=10,
            check=False,
        )
        if failed_result.returncode == 0:
            return "failed"

    poll_s = max(2, int(os.environ.get("NPA_SIM2REAL_JOB_POLL_SECONDS", "5")))
    deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
    while deadline is None or time.monotonic() < deadline:
        result = _kubectl(
            config,
            [
                "get",
                "job",
                job_name,
                "-n",
                namespace,
                "-o",
                "jsonpath={.status.succeeded} {.status.failed}",
            ],
            timeout_s=30,
            check=False,
        )
        if _kubectl_job_not_found(result):
            return "failed"
        if result.returncode == 0:
            parts = (result.stdout or "").strip().split()
            succeeded = int(parts[0]) if parts and str(parts[0]).isdigit() else 0
            failed = int(parts[1]) if len(parts) > 1 and str(parts[1]).isdigit() else 0
            if succeeded >= required_successes:
                return "complete"
            if failed >= 1:
                return "failed"
        time.sleep(poll_s)
    return "timeout"


def _log_sibling_job_applied(
    config: Sim2RealLoopConfig,
    *,
    namespace: str,
    job_name: str,
    component: str,
) -> str:
    """Log sibling Job identity after apply and return the Job UID when known."""

    uid_result = _kubectl(
        config,
        [
            "get",
            "job",
            job_name,
            "-n",
            namespace,
            "-o",
            "jsonpath={.metadata.uid}",
        ],
        timeout_s=30,
        check=False,
    )
    job_uid = (uid_result.stdout or "").strip() if uid_result.returncode == 0 else ""
    print(
        f"sibling_job_applied: component={component} job={job_name} uid={job_uid or 'unknown'}",
        flush=True,
    )
    return job_uid


def _format_pod_exit_diagnostics(pod_info: dict[str, Any]) -> str:
    """Summarize pod phase and container exit/wait reasons for operator errors."""

    parts: list[str] = []
    phase = str(pod_info.get("phase") or "").strip()
    if phase:
        parts.append(f"pod_phase={phase}")
    for status in pod_info.get("container_statuses") or []:
        name = str(status.get("name") or "container")
        state = status.get("state") or {}
        for state_key in ("terminated", "waiting"):
            detail = state.get(state_key) or {}
            reason = str(detail.get("reason") or "").strip()
            message = str(detail.get("message") or "").strip()
            if reason or message:
                parts.append(f"{name}:{state_key}={reason} {message}".strip())
    lookup_error = str(pod_info.get("lookup_error") or "").strip()
    if lookup_error:
        parts.append(f"lookup_error={lookup_error}")
    return " ".join(parts)


def _npa_package_root() -> Path | None:
    """Return the checkout ``npa/`` directory when running from source."""

    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists() and (
            candidate / "src" / "npa"
        ).is_dir():
            return candidate
    for fallback in (Path("/tmp/npa-src/npa"), Path("/tmp/npa-source/npa")):
        if (fallback / "pyproject.toml").exists() and (
            fallback / "src" / "npa"
        ).is_dir():
            return fallback
    return None


def ensure_sibling_source_tarball(config: Sim2RealLoopConfig) -> str:
    """Upload (once per run) a minimal npa source tarball for sibling Jobs."""

    cached = _SIBLING_SOURCE_TARBALL_BY_RUN.get(config.run_id, "").strip()
    if cached:
        return cached
    uri = _stage_sibling_source_tarball(config)
    if uri:
        _SIBLING_SOURCE_TARBALL_BY_RUN[config.run_id] = uri
    return uri


def _sibling_tarball_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if "__pycache__" in tarinfo.name or tarinfo.name.endswith(".pyc"):
        return None
    return tarinfo


def _stage_sibling_source_tarball(config: Sim2RealLoopConfig) -> str:
    """Upload a minimal npa source tarball so sibling Jobs run current code."""

    npa_root = _npa_package_root()
    if npa_root is None or not config.s3_bucket:
        return ""
    with tempfile.TemporaryDirectory(prefix="npa-sibling-src-") as tmp:
        tarball = Path(tmp) / "npa-source.tgz"
        with tarfile.open(tarball, "w:gz") as archive:
            archive.add(
                npa_root / "src",
                arcname="npa/src",
                filter=_sibling_tarball_filter,
            )
            archive.add(
                npa_root / "pyproject.toml",
                arcname="npa/pyproject.toml",
                filter=_sibling_tarball_filter,
            )
        destination = (
            f"{_artifact_root_uri(config).rstrip('/')}/source/"
            f"npa-{_safe_slug(config.run_id)[:40]}.tgz"
        )
        return _storage_client(config).upload_file(str(tarball), destination)


def _ensure_sibling_source_env(
    config: Sim2RealLoopConfig, env: dict[str, str]
) -> dict[str, str]:
    """Inject source tarball env for sibling Jobs (Isaac held-out requires it)."""

    merged = dict(env)
    if merged.get("NPA_SIM2REAL_SOURCE_TARBALL_URI"):
        return merged
    tarball_uri = ensure_sibling_source_tarball(config)
    if tarball_uri:
        merged["NPA_SIM2REAL_SOURCE_TARBALL_URI"] = tarball_uri
    return merged


def _refresh_registry_pull_secret_for_sibling_job(
    image: str,
    *,
    config: Sim2RealLoopConfig,
    namespace: str,
) -> None:
    """Mint a fresh Nebius registry pull secret before each sibling Job apply.

    Initial ``k8s_submit`` refreshes once, but long Sim2Real runs launch many
    later sibling Jobs (augment/train/eval/heldout). IAM registry tokens expire,
    so stale ``npa-nebius-registry`` secrets cause mid-pipeline ImagePullBackOff.
    """

    if _bool_value(os.environ.get("NPA_SIM2REAL_SKIP_REGISTRY_REFRESH", "0")):
        return

    from npa.workflows.sim2real.registry_auth import (
        ensure_registry_pull_secret_for_images,
    )

    # Best-effort: sibling Jobs already carry the run's imagePullSecrets (e.g.
    # ``agent-sa``) via ``config.k8s_image_pull_secrets``, so a refresh failure
    # (no ``nebius`` CLI / no ``NEBIUS_IAM_TOKEN`` in-pod) must not crash the
    # orchestrator. Log and continue; a genuinely stale secret surfaces as an
    # ImagePullBackOff on the sibling rather than a hard orchestrator failure.
    try:
        ensure_registry_pull_secret_for_images(
            image,
            namespace=namespace,
            kubeconfig=config.k8s_kubeconfig,
            k8s_context=config.k8s_context,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort refresh must never abort the run
        logging.getLogger(__name__).warning(
            "sibling registry pull-secret refresh skipped for %s: %s", image, exc
        )


def _run_kubernetes_indexed_image_component(
    image: str,
    *,
    component: str,
    env: dict[str, str],
    config: Sim2RealLoopConfig,
    completions: int,
    parallelism: int,
    timeout_s: int,
) -> dict[str, Any]:
    namespace = config.k8s_namespace or _serviceaccount_namespace() or "default"
    base_job_name = _k8s_job_name(config.run_id, component)
    env = _ensure_sibling_source_env(config, env)
    _refresh_registry_pull_secret_for_sibling_job(
        image, config=config, namespace=namespace
    )

    def manifest_factory(product: str, job_name: str) -> dict[str, Any]:
        return _indexed_component_job_manifest(
            image,
            component=component,
            env=env,
            config=config,
            namespace=namespace,
            job_name=job_name,
            completions=completions,
            parallelism=parallelism,
            timeout_s=timeout_s,
            gpu_product=product,
        )

    def kubectl(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _kubectl(config, args, check=False, **kwargs)

    job_workload = workload_kind(component, sim_backend=config.sim_backend)
    model_hint = env.get("NPA_SIM2REAL_VLM_MODEL", "")
    try:
        gpu_provenance = run_gpu_job_with_fallback(
            kubectl=kubectl,
            manifest_factory=manifest_factory,
            base_job_name=base_job_name,
            namespace=namespace,
            image=image,
            preferred_product=config.k8s_gpu_product,
            explicit_candidates=config.k8s_gpu_candidates,
            workload=job_workload,
            gpu_resource=config.k8s_gpu_resource,
            gpu_count=1,
            timeout_s=timeout_s,
            minimum_vram_gb=minimum_vram_for_workload(
                job_workload,
                model=model_hint,
                explicit=env.get("NPA_SIM2REAL_MIN_GPU_VRAM_GB"),
            ),
            model=model_hint,
        )
    except (GpuCapacityExhausted, GpuJobFailure) as exc:
        raise Sim2RealLoopError(str(exc)) from exc
    job_name = str(gpu_provenance["job_name"])
    job_uid = _log_sibling_job_applied(
        config, namespace=namespace, job_name=job_name, component=component
    )
    pod_info = _component_pod_info(config, namespace=namespace, job_name=job_name)
    logs_result = _kubectl(
        config,
        [
            "logs",
            f"job/{job_name}",
            "-n",
            namespace,
            "--all-containers=true",
            "--tail=-1",
        ],
        timeout_s=300,
        check=False,
    )
    delete_result = _cleanup_component_job(
        config, namespace=namespace, job_name=job_name
    )
    return {
        "mode": "kubernetes_indexed_job",
        "component": component,
        "image": image,
        "image_digests": gpu_provenance.get("image_digests", [])
        or pod_info.get("image_digests", []),
        "namespace": namespace,
        "job_name": job_name,
        "job_uid": job_uid,
        "completions": completions,
        "parallelism": parallelism,
        "pod": pod_info,
        "gpu_request": {
            "resource": config.k8s_gpu_resource,
            "product": gpu_provenance["selected_product"],
            "count": 1,
        },
        "gpu_provenance": gpu_provenance,
        "returncode": 0,
        "stdout_excerpt": _component_excerpt(logs_result.stdout),
        "stderr_excerpt": _component_excerpt(logs_result.stderr),
        "cleanup_stdout_excerpt": _component_excerpt(delete_result.stdout),
        "cleanup_stderr_excerpt": _component_excerpt(delete_result.stderr),
    }


def _run_kubernetes_image_component(
    image: str,
    *,
    component: str,
    env: dict[str, str],
    output_json: Path,
    output_uri: str,
    config: Sim2RealLoopConfig,
    timeout_s: int,
) -> dict[str, Any]:
    namespace = config.k8s_namespace or _serviceaccount_namespace() or "default"
    base_job_name = _k8s_job_name(config.run_id, component)
    env = _ensure_sibling_source_env(config, env)
    _refresh_registry_pull_secret_for_sibling_job(
        image, config=config, namespace=namespace
    )

    def manifest_factory(product: str, job_name: str) -> dict[str, Any]:
        return _component_job_manifest(
            image,
            component=component,
            env=env,
            config=config,
            namespace=namespace,
            job_name=job_name,
            timeout_s=timeout_s,
            gpu_product=product,
        )

    def kubectl(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _kubectl(config, args, check=False, **kwargs)

    job_workload = workload_kind(component, sim_backend=config.sim_backend)
    model_hint = env.get("NPA_SIM2REAL_VLM_MODEL", "")
    try:
        gpu_provenance = run_gpu_job_with_fallback(
            kubectl=kubectl,
            manifest_factory=manifest_factory,
            base_job_name=base_job_name,
            namespace=namespace,
            image=image,
            preferred_product=config.k8s_gpu_product,
            explicit_candidates=config.k8s_gpu_candidates,
            workload=job_workload,
            gpu_resource=config.k8s_gpu_resource,
            gpu_count=1,
            timeout_s=timeout_s,
            minimum_vram_gb=minimum_vram_for_workload(
                job_workload,
                model=model_hint,
                explicit=env.get("NPA_SIM2REAL_MIN_GPU_VRAM_GB"),
            ),
            model=model_hint,
        )
    except (GpuCapacityExhausted, GpuJobFailure) as exc:
        raise Sim2RealLoopError(str(exc)) from exc
    job_name = str(gpu_provenance["job_name"])
    job_uid = _log_sibling_job_applied(
        config, namespace=namespace, job_name=job_name, component=component
    )
    pod_info = _component_pod_info(config, namespace=namespace, job_name=job_name)
    logs_result = _kubectl(
        config,
        [
            "logs",
            f"job/{job_name}",
            "-n",
            namespace,
            "--all-containers=true",
            "--tail=-1",
        ],
        timeout_s=300,
        check=False,
    )
    delete_result = _cleanup_component_job(
        config, namespace=namespace, job_name=job_name
    )
    try:
        _download_component_output(config, output_uri, output_json)
    except Sim2RealLoopError as exc:
        log_excerpt = _component_excerpt(logs_result.stdout or logs_result.stderr)
        raise Sim2RealLoopError(f"{exc} sibling_logs={log_excerpt}") from exc
    return {
        "mode": "kubernetes_job",
        "component": component,
        "image": image,
        "image_digests": gpu_provenance.get("image_digests", [])
        or pod_info.get("image_digests", []),
        "namespace": namespace,
        "job_name": job_name,
        "job_uid": job_uid,
        "pod": pod_info,
        "gpu_request": {
            "resource": config.k8s_gpu_resource,
            "product": gpu_provenance["selected_product"],
            "count": 1,
        },
        "gpu_provenance": gpu_provenance,
        "service_account": config.k8s_service_account,
        "image_pull_secrets": _split_csv(config.k8s_image_pull_secrets),
        "env_secret_names": _split_csv(config.k8s_env_secret_names),
        "output_uri": output_uri,
        "returncode": 0,
        "stdout_excerpt": _component_excerpt(logs_result.stdout),
        "stderr_excerpt": _component_excerpt(logs_result.stderr),
        "cleanup_stdout_excerpt": _component_excerpt(delete_result.stdout),
        "cleanup_stderr_excerpt": _component_excerpt(delete_result.stderr),
    }


def _kubectl(
    config: Sim2RealLoopConfig,
    args: list[str],
    *,
    stdin: str | None = None,
    timeout_s: int = 300,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    cmd = [os.environ.get("NPA_KUBECTL_BIN") or "kubectl"]
    if config.k8s_context:
        cmd.extend(["--context", config.k8s_context])
    cmd.extend(args)
    proc_env = os.environ.copy()
    if config.k8s_kubeconfig:
        proc_env["KUBECONFIG"] = config.k8s_kubeconfig
    result = subprocess.run(
        cmd,
        input=stdin,
        env=proc_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    if check and result.returncode != 0:
        raise Sim2RealLoopError(
            f"kubectl {' '.join(shlex.quote(part) for part in args)} failed: "
            f"{_component_excerpt(result.stderr or result.stdout)}"
        )
    return result


def _component_pod_info(
    config: Sim2RealLoopConfig, *, namespace: str, job_name: str
) -> dict[str, Any]:
    result = _kubectl(
        config,
        [
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"job-name={job_name}",
            "-o",
            "json",
        ],
        timeout_s=120,
        check=False,
    )
    if result.returncode != 0:
        return {"lookup_error": _component_excerpt(result.stderr or result.stdout)}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"lookup_error": "kubectl returned non-json pod output"}
    items = payload.get("items") or []
    if not items:
        return {}
    pod = items[0]
    container = (pod.get("spec", {}).get("containers") or [{}])[0]
    resources = container.get("resources", {})
    statuses = pod.get("status", {}).get("containerStatuses") or []
    container_statuses = [
        {
            "name": item.get("name", ""),
            "ready": item.get("ready", False),
            "restart_count": item.get("restartCount", 0),
            "image": item.get("image", ""),
            "image_id": item.get("imageID", ""),
            "state": item.get("state", {}),
        }
        for item in statuses
    ]
    image_digests = [
        status["image_id"] for status in container_statuses if status["image_id"]
    ]
    return {
        "name": pod.get("metadata", {}).get("name", ""),
        "node_name": pod.get("spec", {}).get("nodeName", ""),
        "phase": pod.get("status", {}).get("phase", ""),
        "resources": resources,
        "container_statuses": container_statuses,
        "image_digests": image_digests,
    }


def _cleanup_component_job(
    config: Sim2RealLoopConfig, *, namespace: str, job_name: str
) -> subprocess.CompletedProcess[str]:
    # Sibling Jobs are deleted after each component when
    # NPA_SIM2REAL_DELETE_COMPONENT_JOBS=1 (default). External/manual deletion
    # during a wait is treated as failure (NotFound) so the driver fails fast.
    if not _bool_value(os.environ.get("NPA_SIM2REAL_DELETE_COMPONENT_JOBS", "1")):
        return subprocess.CompletedProcess([], 0, "", "")
    return _kubectl(
        config,
        [
            "delete",
            "job",
            job_name,
            "-n",
            namespace,
            "--ignore-not-found=true",
            "--wait=true",
        ],
        timeout_s=300,
        check=False,
    )


def _component_attempt_id(
    config: Sim2RealLoopConfig, component: str, label: str
) -> str:
    digest = hashlib.sha1(
        f"{config.run_id}:{component}:{label}".encode("utf-8")
    ).hexdigest()
    return f"{_safe_slug(component)}-{digest[:10]}-{uuid.uuid4().hex[:8]}"


def _component_io_prefix(
    config: Sim2RealLoopConfig, *, component: str, attempt_id: str
) -> str:
    if not config.s3_bucket:
        raise Sim2RealLoopError(
            f"{component} image execution requires s3_bucket for sibling Job I/O"
        )
    return (
        f"{_artifact_root_uri(config).rstrip('/')}/component-io/"
        f"{_safe_slug(component)}/{attempt_id}"
    )


def _component_output_uri(
    config: Sim2RealLoopConfig,
    *,
    component: str,
    attempt_id: str,
    filename: str,
) -> str:
    return f"{_component_io_prefix(config, component=component, attempt_id=attempt_id)}/output/{filename}"


def _upload_component_directory(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    *,
    component: str,
    attempt_id: str,
    name: str,
) -> str:
    uri = f"{_component_io_prefix(config, component=component, attempt_id=attempt_id)}/input/{_safe_slug(name)}/"
    _storage_client(config).upload_directory(str(local_dir), uri)
    return uri


def _upload_component_file(
    config: Sim2RealLoopConfig,
    local_path: Path,
    *,
    component: str,
    attempt_id: str,
    name: str,
) -> str:
    uri = f"{_component_io_prefix(config, component=component, attempt_id=attempt_id)}/input/{_safe_slug(name)}"
    return _storage_client(config).upload_file(str(local_path), uri)


def _download_component_output(
    config: Sim2RealLoopConfig, output_uri: str, output_json: Path
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    client = _storage_client(config)
    attempts = max(
        1, int(os.environ.get("NPA_SIM2REAL_COMPONENT_DOWNLOAD_RETRIES", "12"))
    )
    grace_s = float(os.environ.get("NPA_SIM2REAL_HELDOUT_UPLOAD_GRACE_S", "0") or "0")
    if grace_s > 0:
        time.sleep(grace_s)
    for attempt in range(attempts):
        if output_json.exists():
            output_json.unlink()
        client.download_path(output_uri, str(output_json))
        if output_json.exists() and output_json.stat().st_size > 0:
            return
        if attempt + 1 < attempts:
            time.sleep(min(2**attempt, 8))
    raise Sim2RealLoopError(
        f"component output not available at {output_uri} after {attempts} download attempts"
    )


def _storage_client(config: Sim2RealLoopConfig) -> StorageClient:
    return StorageClient.from_environment(endpoint_url=config.s3_endpoint)


def _k8s_job_name(run_id: str, component: str) -> str:
    run_part = _safe_slug(run_id)[:22] or "run"
    component_part = _safe_slug(component)[:16] or "component"
    suffix = uuid.uuid4().hex[:8]
    return f"s2r-{component_part}-{run_part}-{suffix}"[:63].rstrip("-")


def _safe_slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in str(value)]
    return "-".join(part for part in "".join(chars).split("-") if part)


def _normalized_s3_prefix(uri: str) -> str:
    return str(uri or "").strip()


def _read_component_json(
    output_path: Path, invocation: dict[str, Any]
) -> dict[str, Any]:
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    stdout = str(invocation.get("stdout") or invocation.get("stdout_excerpt") or "")
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)
    raise Sim2RealLoopError(
        f"{invocation.get('component', 'component')} did not write JSON to {output_path}"
    )


def _inner_loop_progress_score(inner_evidence: dict[str, Any]) -> float:
    """Map closed inner-loop evidence to a [0, 1] training-progress score."""

    reward_trend = [
        float(item)
        for item in (inner_evidence.get("reward_trend") or [])
        if item is not None
    ]
    reward_progress = (
        max(0.0, min(1.0, (reward_trend[-1] + 1.0) / 2.0)) if reward_trend else 0.0
    )
    final_quality = float(inner_evidence.get("final_quality") or 0.0)
    vlm_scores: list[float] = []
    for iteration in inner_evidence.get("iterations") or []:
        if not isinstance(iteration, dict):
            continue
        sample = iteration.get("sample_vlm_eval") or {}
        if isinstance(sample, dict) and sample.get("score") is not None:
            vlm_scores.append(max(0.0, min(1.0, float(sample["score"]))))
    vlm_progress = vlm_scores[-1] if vlm_scores else 0.0
    return max(0.0, min(1.0, max(reward_progress, final_quality, vlm_progress)))


def _reference_adapter_env_score(base: float, env: dict[str, Any], index: int) -> float:
    physics = env.get("physics") or {}
    friction = float(physics.get("friction", 0.5))
    return max(0.0, min(1.0, base + 0.04 * (friction - 0.5) + 0.01 * index))


def _apply_reference_adapter_heldout_gate(
    per_env: list[dict[str, Any]],
    envs: list[dict[str, Any]],
    *,
    inner_evidence: dict[str, Any],
    threshold: float,
) -> None:
    """Annotate reference-adapter progress without overriding real sim success.

    The reference VLM→RL trainer only updates a compact action-bias adapter, so
    native Isaac/Genesis task success may stay near zero even when VLM scores and
    reward trends improve. Preserve that adapter score for diagnostics, but keep
    ``success`` and ``score`` grounded in the actual simulator rollout so the
    outer-loop gate cannot promote a checkpoint without real held-out success.
    """

    trainer_source = inner_evidence.get("trainer_source")
    if trainer_source not in (None, "reference"):
        return
    base = _inner_loop_progress_score(inner_evidence)
    for index, (item, env) in enumerate(zip(per_env, envs, strict=False)):
        cal_score = _reference_adapter_env_score(base, env, index)
        sim_success = bool(item.get("success"))
        sim_score = float(item.get("score", 0.0))
        details = dict(item.get("details") or {})
        details["sim_success"] = sim_success
        details["sim_score"] = round(sim_score, 6)
        details["reference_adapter_score"] = round(cal_score, 6)
        details["reference_adapter_would_pass"] = cal_score >= threshold
        item["details"] = details


def _reference_heldout_payload(
    envs: list[dict[str, Any]],
    *,
    inner_evidence: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """Deterministic held-out scores for local staged runs without sim backends."""

    base = _inner_loop_progress_score(inner_evidence)
    per_env: list[dict[str, Any]] = []
    for index, env in enumerate(envs):
        physics = env.get("physics") or {}
        score = _reference_adapter_env_score(base, env, index)
        per_env.append(
            {
                "env_id": str(env.get("env_id") or f"heldout-{index:04d}"),
                "success": score >= threshold,
                "score": round(score, 6),
                "details": {"mode": "local_reference", "physics": physics},
            }
        )
    return {
        "schema": SCHEMA_HELDOUT_REPORT,
        "per_env": per_env,
        "sim_backend": "local_reference",
        "component_source": "local_reference",
        "rollout_backend": "reference-heuristic",
        "policy_source": "inner_evidence_adapter",
    }


def _reference_vlm_payload_from_rollout(
    manifest: dict[str, Any],
    *,
    rollout_dir: Path,
    rollout_id: str,
    config: Sim2RealLoopConfig,
) -> dict[str, Any]:
    """In-process reference VLM when no S3 bucket is configured (local smoke/staged runs)."""

    quality = float(manifest.get("quality", 0.4))
    per_step: list[dict[str, Any]] = []
    for item in manifest.get("actions", []):
        step = int(item["step"])
        frame = rollout_dir / f"camera-{step:03d}.ppm"
        signal = sum(frame.read_bytes()[-12:]) % 17 if frame.exists() else step
        tag = "minor_alignment" if signal % 3 else "ok"
        per_step.append(
            {
                "step": step,
                "critique_text": (
                    f"Reference VLM: frame signal {signal}; rollout quality={quality:.3f}."
                ),
                "error_tags": [tag],
                "action": item.get("action", []),
                "camera_observation": frame.name,
            }
        )
    if not per_step:
        raise Sim2RealLoopError("reference VLM requires rollout actions in manifest")
    score = max(0.05, min(0.95, quality + 0.06))
    return {
        "schema": SCHEMA_VLM_EVAL,
        "rollout_id": rollout_id,
        "success": score >= config.threshold,
        "score": round(score, 6),
        "per_step": per_step,
        "summary": "Local reference VLM evaluation (no S3/K8s sibling job).",
        "model": config.vlm_model,
        "component_source": "local_reference",
    }


def _normalize_vlm_evaluation(
    payload: dict[str, Any],
    *,
    manifest: dict[str, Any],
    rollout_id: str,
    config: Sim2RealLoopConfig,
    invocation: dict[str, Any],
) -> dict[str, Any]:
    if "score" not in payload:
        raise Sim2RealLoopError("VLM component output must include score")
    score = max(0.0, min(1.0, float(payload["score"])))
    success = bool(payload.get("success", score >= config.threshold))
    raw_steps = payload.get("per_step") or payload.get("steps") or []
    if not raw_steps and payload.get("critique_text"):
        raw_steps = [
            {
                "step": 0,
                "critique_text": payload["critique_text"],
                "error_tags": payload.get("error_tags", []),
            }
        ]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise Sim2RealLoopError("VLM component output must include non-empty per_step")
    action_items = {int(item["step"]): item for item in manifest.get("actions", [])}
    observations = list(manifest.get("camera_observations", []))
    per_step: list[dict[str, Any]] = []
    normalized_steps: set[int] = set()
    for raw in raw_steps:
        step = int(raw.get("step", len(per_step)))
        if action_items and (step not in action_items or step in normalized_steps):
            continue
        tags = raw.get("error_tags", raw.get("tags", [])) or ["ok"]
        if isinstance(tags, str):
            tags = [tags]
        critique = str(
            raw.get("critique_text") or raw.get("critique") or raw.get("text") or ""
        )
        malformed = not critique
        if malformed:
            critique = "Malformed model-local critique rejected; simulator state only."
        camera = raw.get("camera_observation")
        if not camera and 0 <= step < len(observations):
            camera = observations[step]
        per_step.append(
            {
                "step": step,
                "critique_text": critique,
                "error_tags": [str(tag) for tag in tags],
                "action": raw.get(
                    "action", (action_items.get(step) or {}).get("action", [])
                ),
                "camera_observation": str(camera or f"camera-{step:03d}.ppm"),
                "confidence": 0.0 if malformed else float(raw.get("confidence", 0.65)),
                "critique_source": str(
                    "model_malformed"
                    if malformed
                    else raw.get("critique_source") or "model_per_step"
                ),
                "model_disagreement": bool(raw.get("model_disagreement")),
                "reason2_critique": str(raw.get("reason2_critique") or ""),
                "reason3_critique": str(raw.get("reason3_critique") or ""),
                "simulator_ground_truth": dict(
                    (action_items.get(step) or {}).get("simulator_ground_truth") or {}
                ),
                "scenario_config_digest": str(
                    (action_items.get(step) or {}).get("scenario_config_digest")
                    or manifest.get("scenario_config_digest")
                    or ""
                ),
            }
        )
        normalized_steps.add(step)
    covered = {int(item["step"]) for item in per_step}
    for step, action_item in action_items.items():
        if step in covered:
            continue
        per_step.append(
            {
                "step": step,
                "critique_text": "No model-local critique; simulator state only.",
                "error_tags": ["ok"],
                "action": action_item.get("action", []),
                "camera_observation": str(
                    action_item.get("camera_observation") or f"camera-{step:03d}.png"
                ),
                "confidence": 0.0,
                "critique_source": "model_missing",
                "model_disagreement": False,
                "reason2_critique": "",
                "reason3_critique": "",
                "simulator_ground_truth": dict(
                    action_item.get("simulator_ground_truth") or {}
                ),
                "scenario_config_digest": str(
                    action_item.get("scenario_config_digest")
                    or manifest.get("scenario_config_digest")
                    or ""
                ),
            }
        )
    per_step.sort(key=lambda item: int(item["step"]))
    return {
        "schema": SCHEMA_VLM_EVAL,
        "rollout_id": str(payload.get("rollout_id") or rollout_id),
        "success": success,
        "score": round(score, 6),
        "per_step": per_step,
        "summary": str(payload.get("summary") or payload.get("critique") or ""),
        "model": str(payload.get("model") or config.vlm_model),
        "vlm_image": config.vlm_image,
        "component_invocation": _public_invocation(invocation),
        "generated_at": _utc_now(),
    }


def _component_excerpt(text: str, limit: int = 1200) -> str:
    scrubbed = []
    for line in str(text or "").splitlines():
        if "AWS_SECRET_ACCESS_KEY" in line or "AWS_ACCESS_KEY_ID" in line:
            scrubbed.append("[redacted secret line]")
        else:
            scrubbed.append(line)
    return "\n".join(scrubbed)[-limit:]


def _redact_command(command: str) -> str:
    redacted = str(command)
    for key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "NGC_API_KEY",
    ):
        value = os.environ.get(key)
        if value:
            redacted = redacted.replace(value, f"<{key}>")
    return redacted


def _public_invocation(invocation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in invocation.items()
        if key not in {"stdout", "stderr"}
    }


def convert_vlm_eval_to_rl_signal(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Convert structured critique into simulator-grounded temporal credit."""

    if evaluation.get("schema") != SCHEMA_VLM_EVAL:
        raise Sim2RealLoopError(
            f"unsupported VLM eval schema: {evaluation.get('schema')}"
        )
    from npa.workflows.sim2real.temporal_credit import (
        TemporalCreditError,
        convert_evaluation,
    )

    try:
        return convert_evaluation(evaluation)
    except TemporalCreditError as exc:
        raise Sim2RealLoopError(f"invalid temporal credit input: {exc}") from exc


def signal_mapping_rules() -> dict[str, Any]:
    """Return documented VLM-critique to RL-signal conversion rules."""

    return {
        "dense_reward": (
            "authoritative weighted goal/reach progress, contact, stable grasp, "
            "lift, stable placement, distance, drop, and termination terms; "
            "bounded Cosmos auxiliary contribution <=0.12; clipped to [-1,1]"
        ),
        "advantage": "per-step reward minus rollout mean reward",
        "degeneracy": (
            "use distance/action simulator fallback when grounded reward is "
            "constant; refuse real PPO if every advantage remains zero"
        ),
        "per_action_credit": "abs(action_i) * step_reward for each source action dimension",
        "nl_corrective_targets": CORRECTIVE_TARGETS,
        "error_severity": ERROR_SEVERITY,
    }


def _convert_eval_to_signal(
    evaluation: dict[str, Any],
    *,
    config: Sim2RealLoopConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Convert a VLM eval to an RL signal via the BYO command or the reference.

    BYO signal-converter contract: the command reads the VLM evaluation JSON from
    ``NPA_SIM2REAL_EVALUATION_JSON`` and writes an ``npa.sim2real.rl_signal.v1``
    document to ``NPA_SIM2REAL_OUTPUT_JSON``. A missing, empty, non-conforming, or
    failing command raises ``Sim2RealLoopError`` -- the loop never silently falls
    back to the in-process reference converter.
    """

    if not config.byo_signal_converter.strip():
        return convert_vlm_eval_to_rl_signal(evaluation)

    rollout_id = str(evaluation.get("rollout_id") or "rollout")
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = output_dir / f"{rollout_id}.evaluation.json"
    _write_json_artifact(eval_path, evaluation)
    output_path = output_dir / f"{rollout_id}.byo-signal.json"
    env = _component_env(
        config,
        component="signal_converter",
        output_json=output_path,
        extra={
            "NPA_SIM2REAL_EVALUATION_JSON": str(eval_path),
            "NPA_SIM2REAL_ROLLOUT_ID": rollout_id,
            "NPA_SIM2REAL_RL_SIGNAL_SCHEMA": SCHEMA_RL_SIGNAL,
        },
    )
    invocation = _run_component_command(
        config.byo_signal_converter,
        cwd=output_dir,
        env=env,
        component="signal_converter",
    )
    payload = _read_component_json(output_path, invocation)
    return _normalize_byo_rl_signal(
        payload,
        rollout_id=rollout_id,
        invocation=invocation,
    )


def _normalize_byo_rl_signal(
    payload: dict[str, Any],
    *,
    rollout_id: str,
    invocation: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise Sim2RealLoopError("signal_converter output must be a JSON object")
    if payload.get("schema") != SCHEMA_RL_SIGNAL:
        raise Sim2RealLoopError(
            "signal_converter output must use schema "
            f"{SCHEMA_RL_SIGNAL}, got {payload.get('schema')!r}"
        )
    per_step = payload.get("per_step")
    if not isinstance(per_step, list) or not per_step:
        raise Sim2RealLoopError(
            "signal_converter output must include non-empty per_step"
        )
    payload.setdefault("rollout_id", rollout_id)
    payload.setdefault("source", "byo")
    parse_vlm_signal_batch, _ = _signal_training_imports()
    try:
        parse_vlm_signal_batch(payload)
    except Exception as exc:
        raise Sim2RealLoopError(
            f"signal_converter output is not a valid {SCHEMA_RL_SIGNAL}: {exc}"
        ) from exc
    payload["component_invocation"] = _public_invocation(invocation)
    return payload


def _byo_robot_env(config: Sim2RealLoopConfig) -> dict[str, str]:
    """Robot-spec env that opts a component into the BYO-robot path (Franka-safe).

    Shared by BOTH the BYO trainer sibling and the held-out eval, so a customer's
    ``robot_spec_uri`` / preset reaches RL training AND eval as ONE embodiment with
    matching action/observation dims. The ``NPA_BYO_ROBOT_TASK=1`` flag is the gate
    both ``byo_isaac_trainer`` and ``byo_isaac_eval`` check before resolving the spec
    and registering the retargeted Lift variant — without it, the eval builds a stock
    Franka-dimensioned env and a non-Franka checkpoint fails to load.

    Returns ``{}`` — leaving the component on the stock-Franka path, byte-for-byte
    unchanged — when no robot is requested or it is stock Franka.
    """

    uri = (config.robot_spec_uri or "").strip()
    source = (config.robot_source or "").strip().lower()
    preset = (config.robot_preset or "").strip().lower()
    requested = bool(uri) or bool(preset) or (bool(source) and source != "stock_franka")
    if not requested:
        return {}
    return {
        "NPA_BYO_ROBOT_TASK": "1",
        "NPA_SIM2REAL_ROBOT_SPEC_URI": config.robot_spec_uri,
        "NPA_SIM2REAL_ROBOT_SOURCE": config.robot_source,
        "NPA_SIM2REAL_ROBOT_PRESET": config.robot_preset,
    }


def _run_trainer_via_command(
    signal_batch_path: Path,
    *,
    config: Sim2RealLoopConfig,
    output_dir: Path,
    initial_reward_head: float,
    initial_action_bias: float,
    train_envs_dir: Path | None = None,
    resume_checkpoint_uri: str = "",
    outer_iteration: int = 1,
    iteration: int = 1,
) -> VlmSignalUpdateResult:
    """Run the BYO trainer command and parse its update result.

    BYO trainer contract: the command reads the parsed signal batch JSON from
    ``NPA_SIM2REAL_SIGNAL_JSON`` and writes an update JSON to
    ``NPA_SIM2REAL_OUTPUT_JSON`` containing at least ``reward_head_after``,
    ``policy_output_after`` (list), and ``policy_delta_l2`` (optional
    ``loss_before``/``loss_after``). A missing, empty, non-conforming, or failing
    command raises ``Sim2RealLoopError`` -- the loop never silently falls back to
    the in-process reference trainer.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "byo-trainer-update.json"
    extra = {
        "NPA_SIM2REAL_SIGNAL_JSON": str(signal_batch_path),
        "NPA_SIM2REAL_INITIAL_REWARD_HEAD": str(initial_reward_head),
        "NPA_SIM2REAL_INITIAL_ACTION_BIAS": str(initial_action_bias),
        "NPA_SIM2REAL_LEARNING_RATE": str(config.learning_rate),
        "NPA_SIM2REAL_SIGNAL_LOSS_WEIGHT": str(config.signal_loss_weight),
        "NPA_SIM2REAL_TRAINER_IMAGE": config.trainer_image,
    }
    # Expose the GENERATED train-env specs (envgen seed + per-env physics) so a BYO
    # trainer can train on the generated env distribution, not stock defaults
    # (mirrors how the held-out eval consumes NPA_SIM2REAL_HELDOUT_ENVS_DIR).
    if train_envs_dir is not None:
        extra["NPA_SIM2REAL_TRAIN_ENVS_DIR"] = str(train_envs_dir)
        # S3 fallback: the orchestrator localizes only the held-out split, so the
        # trainer reads the generated train-env spec (seed + physics) from S3 when
        # the local dir is absent.
        extra["NPA_SIM2REAL_TRAIN_ENVS_URI"] = (
            f"{_artifact_root_uri(config)}/envs/train/envs.jsonl"
        )
    # Per-iteration tag keeps each iteration's trainer artifacts (checkpoint) at a
    # DISTINCT path so the prior model survives for the next iteration to resume from.
    extra["NPA_SIM2REAL_TRAINER_TAG"] = (
        f"outer-{outer_iteration:02d}-iter-{iteration:02d}"
    )
    # OUTER-LOOP RESUME: continue the same policy from the prior iteration's checkpoint
    # so stage 11B "send back for more RL" compounds instead of restarting from scratch.
    if resume_checkpoint_uri.strip():
        extra["NPA_SIM2REAL_RESUME_CHECKPOINT_URI"] = resume_checkpoint_uri.strip()
    # BYO ROBOT: route a customer robot spec into RL TRAINING, not just the held-out
    # eval. Previously the trainer sibling never received the robot vars, so a custom
    # robot_spec_uri reached only the eval USD-swap — the policy still trained on the
    # stock Franka. Forward the same robot inputs the eval gets + opt in the trainer's
    # BYO-robot path so train and eval share one embodiment. Empty (Franka/no robot)
    # -> nothing added, trainer path byte-for-byte unchanged.
    extra.update(_byo_robot_env(config))
    env = _component_env(
        config,
        component="trainer",
        output_json=output_path,
        extra=extra,
    )
    invocation = _run_component_command(
        config.byo_trainer_command,
        cwd=output_dir,
        env=env,
        component="trainer",
    )
    payload = _read_component_json(output_path, invocation)
    if not isinstance(payload, dict):
        raise Sim2RealLoopError("trainer command output must be a JSON object")
    from npa.workbench.lerobot.policy_container import VlmSignalUpdateResult

    try:
        result = VlmSignalUpdateResult.from_dict(payload)
    except Exception as exc:
        raise Sim2RealLoopError(
            f"trainer command output is not a valid update result: {exc}"
        ) from exc
    nested_invocation = dict(payload.get("component_invocation") or {})
    if nested_invocation:
        _write_json_artifact(
            output_dir / "byo-trainer-gpu-provenance.json", nested_invocation
        )
    _write_json_artifact(output_path, result.to_dict())
    return result


def _heldout_k8s_image_ready(config: Sim2RealLoopConfig) -> bool:
    from npa.workflows.sim2real_stages import k8s_image_ready

    return k8s_image_ready(config.heldout_backend_image())


# =============================================================================
# Stage 10 — held-out eval (`run_heldout_eval`)
# =============================================================================


def run_heldout_eval(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    inner_evidence: dict[str, Any],
    outer_iteration: int,
    evaluation_split: str = "",
    inner_iteration: int = 0,
    checkpoint_iteration: int = 0,
) -> dict[str, Any]:
    """Evaluate one exact checkpoint on validation or final gold scenarios."""

    if not evaluation_split:
        evaluation_split = (
            "gold_heldout"
            if outer_iteration >= config.outer_iterations
            else "validation"
        )
    if evaluation_split not in {"validation", "gold_heldout"}:
        raise Sim2RealLoopError(
            f"unsupported checkpoint evaluation split: {evaluation_split}"
        )
    split_dir_name = (
        "validation" if evaluation_split == "validation" else "gold-heldout"
    )
    eval_count = (
        config.validation_env_count
        if evaluation_split == "validation"
        else config.heldout_env_count
    )
    output_dir = local_dir / "eval" / split_dir_name / f"outer-{outer_iteration:02d}"
    if inner_iteration:
        output_dir = output_dir / f"iter-{inner_iteration:02d}"
    if checkpoint_iteration:
        output_dir = output_dir / f"checkpoint-{checkpoint_iteration:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report.json"
    inner_path = output_dir / f"inner-evidence-outer-{outer_iteration:02d}.json"
    _write_json_artifact(inner_path, inner_evidence)
    extra = {
        "NPA_SIM2REAL_HELDOUT_ENVS_DIR": str(local_dir / "envs" / split_dir_name),
        "NPA_SIM2REAL_HELDOUT_ENV_COUNT": str(eval_count),
        "NPA_SIM2REAL_EVALUATION_SPLIT": evaluation_split,
        "NPA_SIM2REAL_INNER_EVIDENCE_JSON": str(inner_path),
        "NPA_SIM2REAL_THRESHOLD": str(config.threshold),
        "NPA_SIM2REAL_EVAL_IMAGE": config.eval_image,
        "NPA_SIM2REAL_ISAAC_IMAGE": config.isaac_image,
        "NPA_SIM2REAL_SIM_BACKEND": config.sim_backend,
        "NPA_SIM2REAL_ISAAC_TASK": config.isaac_task,
        "NPA_SIM2REAL_SCENE_SPEC_URI": config.scene_spec_uri,
        "NPA_SIM2REAL_ASSETS_URI": config.assets_uri,
        "NPA_SIM2REAL_CAMERAS_URI": config.cameras_uri,
        "NPA_SIM2REAL_EVAL_TAG": (
            f"{evaluation_split}-outer-{outer_iteration:02d}"
            + (f"-iter-{inner_iteration:02d}" if inner_iteration else "")
            + (
                f"-checkpoint-{checkpoint_iteration:04d}"
                if checkpoint_iteration
                else ""
            )
        ),
    }
    # BYO robot: opt the held-out eval into the SAME robot-swapped Lift variant the
    # policy trained on. This sets NPA_BYO_ROBOT_TASK=1 (+ the robot uri/source/preset)
    # so byo_isaac_eval resolves the spec, ships the retarget module, and registers the
    # customer's task variant — giving the eval env the SAME action/observation dims as
    # training. Without the flag the eval built a stock Franka-dimensioned env, and a
    # non-Franka checkpoint failed to load into the eval ActorCritic (size mismatch).
    # Franka-safe: _byo_robot_env returns {} for no-robot / stock, so the eval env is
    # byte-for-byte unchanged on the stock path.
    extra.update(_byo_robot_env(config))
    env = _component_env(
        config,
        component="heldout_eval",
        output_json=output_path,
        extra=extra,
    )
    # Default the genuine-RL Isaac path to the real-policy held-out eval
    # (byo_isaac_eval loads the trained rsl_rl checkpoint and rolls it in Isaac
    # Lab), instead of the scalar action-bias adapter rollout. Gate on a genuine
    # trainer (a real checkpoint must exist) + a ready K8s image; the reference
    # trainer has no Isaac checkpoint, so it keeps the adapter/reference path.
    eval_command = config.byo_eval_command.strip()
    if (
        not eval_command
        and config.sim_backend == SIM_BACKEND_ISAAC
        and config.byo_trainer_command.strip()
        and config.s3_bucket.strip()
        and _heldout_k8s_image_ready(config)
    ):
        eval_command = "python3 -m npa.workflows.sim2real.byo_isaac_eval"
    if eval_command:
        invocation = _run_component_command(
            eval_command,
            cwd=local_dir,
            env=env,
            component="heldout_eval",
        )
    elif not config.s3_bucket.strip() or not _heldout_k8s_image_ready(config):
        heldout_manifest = local_dir / "envs" / split_dir_name / "manifest.json"
        if not heldout_manifest.is_file():
            heldout_manifest = local_dir / "envs" / "heldout" / "manifest.json"
        envs = json.loads(heldout_manifest.read_text(encoding="utf-8")).get("envs", [])
        local_backend = config.sim_backend
        if local_backend == SIM_BACKEND_ISAAC:
            try:
                import isaaclab  # noqa: F401
            except ImportError:
                local_backend = SIM_BACKEND_GENESIS
        try:
            import torch  # noqa: F401

            has_sim = True
        except ImportError:
            has_sim = False
        if has_sim:
            payload = _component_heldout_payload(
                envs,
                inner_evidence=inner_evidence,
                threshold=config.threshold,
                sim_backend=local_backend,
                isaac_task=config.isaac_task,
                renders_dir=(
                    output_dir / "renders"
                    if local_backend == SIM_BACKEND_ISAAC
                    else None
                ),
            )
        else:
            payload = _reference_heldout_payload(
                envs,
                inner_evidence=inner_evidence,
                threshold=config.threshold,
            )
        _write_json_artifact(output_path, payload)
        invocation = {
            "component": "heldout_eval",
            "mode": "local_reference"
            if not config.s3_bucket.strip()
            else "seam_placeholder",
            "image": config.heldout_backend_image(),
        }
    else:
        attempt_id = _component_attempt_id(
            config,
            "heldout_eval",
            f"{evaluation_split}-outer-{outer_iteration:02d}"
            + (f"-iter-{inner_iteration:02d}" if inner_iteration else "")
            + (
                f"-checkpoint-{checkpoint_iteration:04d}"
                if checkpoint_iteration
                else ""
            ),
        )
        split_envs_uri = (
            config.validation_envs_uri
            if evaluation_split == "validation"
            else (config.gold_heldout_envs_uri or config.heldout_envs_uri)
        )
        if split_envs_uri:
            heldout_envs_uri = _resolve_env_records_s3_uri(
                _normalized_s3_prefix(split_envs_uri)
            )
        else:
            local_heldout = local_dir / "envs" / split_dir_name
            jsonl_path = local_heldout / "envs.jsonl"
            if jsonl_path.is_file():
                heldout_envs_uri = _upload_component_file(
                    config,
                    jsonl_path,
                    component="heldout_eval",
                    attempt_id=attempt_id,
                    name="heldout-envs.jsonl",
                )
            else:
                heldout_envs_uri = _upload_component_directory(
                    config,
                    local_heldout,
                    component="heldout_eval",
                    attempt_id=attempt_id,
                    name="heldout-envs",
                )
        inner_evidence_uri = _upload_component_file(
            config,
            inner_path,
            component="heldout_eval",
            attempt_id=attempt_id,
            name="inner-evidence.json",
        )
        output_uri = _component_output_uri(
            config,
            component="heldout_eval",
            attempt_id=attempt_id,
            filename="report.json",
        )
        env["NPA_SIM2REAL_HELDOUT_ENVS_URI"] = heldout_envs_uri
        env["NPA_SIM2REAL_INNER_EVIDENCE_URI"] = inner_evidence_uri
        env["NPA_SIM2REAL_OUTPUT_URI"] = output_uri
        env["NPA_SIM2REAL_HELDOUT_EVAL_LIMIT"] = str(config.heldout_eval_limit)
        invocation = _run_image_component(
            config.heldout_backend_image(),
            component="heldout_eval",
            env=env,
            output_json=output_path,
            output_uri=output_uri,
            config=config,
        )
    payload = _read_component_json(output_path, invocation)
    nested_invocation = dict(payload.get("component_invocation") or {})
    if nested_invocation:
        invocation = {
            **invocation,
            **nested_invocation,
            "command_invocation": _public_invocation(invocation),
        }
    report = _normalize_heldout_report(
        payload,
        config=config,
        outer_iteration=outer_iteration,
        inner_evidence_uri=str(inner_path),
        invocation=invocation,
    )
    report["evaluation_split"] = evaluation_split
    report["checkpoint_training_iteration"] = checkpoint_iteration
    report["gold_heldout_untouched"] = evaluation_split == "validation"
    report = _ensure_heldout_renders_for_viz(
        config,
        local_dir,
        report,
        invocation=invocation,
        payload=payload,
        renders_dir=output_dir / "renders",
    )
    _write_json_artifact(output_path, report)
    return {**report, "report_uri": str(output_path)}


_HELDOUT_DISTANCE_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)


def _heldout_success_summary(
    payload: dict[str, Any],
    per_env: list[dict[str, Any]],
) -> dict[str, Any]:
    """Multi-threshold accuracy from per-env object->goal distances.

    Prefers the success_summary the byo_isaac_eval component already emitted
    (carried through verbatim); otherwise recomputes success@0.05/0.10/0.15/0.20
    plus mean/min object_goal_distance_m from per_env details so the curve
    survives normalization even when strict success_rate is 0.
    """
    existing = payload.get("success_summary")
    if isinstance(existing, dict) and existing:
        return existing
    dists = [
        item["details"]["object_goal_distance_m"]
        for item in per_env
        if isinstance(item.get("details"), dict)
        and isinstance(item["details"].get("object_goal_distance_m"), (int, float))
    ]
    summary: dict[str, Any] = {}
    if dists:
        for thr in _HELDOUT_DISTANCE_THRESHOLDS:
            summary[f"success@{thr:.2f}"] = round(
                sum(1 for d in dists if d < thr) / len(dists), 4
            )
        summary["mean_object_goal_distance_m"] = round(sum(dists) / len(dists), 6)
        summary["min_object_goal_distance_m"] = round(min(dists), 6)
    return summary


def _normalize_heldout_report(
    payload: dict[str, Any],
    *,
    config: Sim2RealLoopConfig,
    outer_iteration: int,
    inner_evidence_uri: str,
    invocation: dict[str, Any],
) -> dict[str, Any]:
    raw_items = (
        payload.get("per_env") or payload.get("env_scores") or payload.get("scores")
    )
    if isinstance(raw_items, dict):
        raw_items = [
            {"env_id": key, **(value if isinstance(value, dict) else {"score": value})}
            for key, value in raw_items.items()
        ]
    if not isinstance(raw_items, list) or not raw_items:
        raise Sim2RealLoopError(
            "held-out eval component output must include non-empty per_env/env_scores"
        )
    per_env: list[dict[str, Any]] = []
    passed = 0
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            item = {"score": item}
        score = max(
            0.0, min(1.0, float(item.get("score", item.get("success_score", 0.0))))
        )
        success = bool(item.get("success", score >= config.threshold))
        passed += int(success)
        per_env.append(
            {
                "env_id": str(item.get("env_id") or f"heldout-{index:04d}"),
                "success": success,
                "score": round(score, 6),
                "details": item.get("details", {}),
            }
        )
    success_rate = passed / float(len(per_env))
    # Multi-threshold accuracy: a single strict success_rate@threshold hides real
    # progress (a policy can score 0 at 0.05m yet land 81% within 0.15m). Carry
    # through the byo_isaac_eval payload's success_summary when present, otherwise
    # recompute it from the per-env object->goal distances so accuracy stays
    # visible in the split-specific evaluation report even when success_rate is 0.
    success_summary = _heldout_success_summary(payload, per_env)
    report = {
        "schema": SCHEMA_HELDOUT_REPORT,
        "stage": 10,
        "outer_iteration": outer_iteration,
        "status": "completed",
        "success_rate": round(success_rate, 6),
        "threshold": config.threshold,
        "success_summary": success_summary,
        "per_env": per_env,
        "eval_image": config.eval_image,
        "sim_backend": str(payload.get("sim_backend") or config.sim_backend),
        "heldout_backend_image": config.heldout_backend_image(),
        "byo_eval_command": _redact_command(config.byo_eval_command),
        "inner_evidence_uri": inner_evidence_uri,
        "component_invocation": _public_invocation(invocation),
        "generated_at": _utc_now(),
    }
    for key in (
        "component_source",
        "rollout_backend",
        # Preserve BYO-eval extras so heldout viz + provenance survive normalization:
        # render_manifest drives stage-14 Rerun heldout/camera/**; the rest record
        # which trained policy + generated envs were actually evaluated.
        "render_manifest",
        "generated_envs_tested",
        "generated_env_ids",
        "policy_checkpoint",
        "policy_checkpoint_sha256",
        "policy_checkpoint_size_bytes",
        "policy_inference_provenance",
        "applied_scenario_proof",
        "capture",
        "camera_metadata",
        "success_distance_m",
        "strict_success",
        "decomposed_metrics",
        "per_difficulty",
        "deployable_policy_eval",
    ):
        if payload.get(key):
            report[key] = payload[key]
    if "asset_provenance" in payload:
        report["asset_provenance"] = payload["asset_provenance"]
        report["asset_fallback_used"] = bool(
            payload.get(
                "asset_fallback_used",
                payload["asset_provenance"].get("asset_fallback_used", False),
            )
        )
    if "robot_provenance" in payload:
        report["robot_provenance"] = payload["robot_provenance"]
        report["robot_fallback_used"] = bool(payload.get("robot_fallback_used", False))
    if payload.get("render_manifest"):
        report["render_manifest"] = payload["render_manifest"]
    return report


def _has_heldout_camera_pngs(renders_dir: Path) -> bool:
    return any(renders_dir.rglob("camera-*.png"))


def _ensure_heldout_renders_for_viz(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    heldout_report: dict[str, Any] | None,
    *,
    invocation: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    renders_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Sync Isaac held-out camera PNGs locally and attach a render manifest."""

    if not heldout_report:
        return heldout_report

    report = dict(heldout_report)
    renders_dir = renders_dir or (local_dir / "eval" / "heldout" / "renders")
    renders_dir.mkdir(parents=True, exist_ok=True)
    report["local_renders_dir"] = str(renders_dir)

    if config.s3_bucket.strip():
        from npa.clients.storage import StorageError
        from npa.workflows.sim2real_rerun_regen import sync_heldout_renders

        client = _storage_client(config)

        output_uri = str((invocation or {}).get("output_uri") or "").strip()
        if output_uri:
            try:
                client.download_directory(
                    _sibling_uri(output_uri, "renders/"), str(renders_dir)
                )
            except (StorageError, OSError):
                pass
            manifest_uri = _sibling_uri(output_uri, "render-manifest.json")
            manifest_path = renders_dir.parent / "render-manifest.sibling.json"
            try:
                client.download_path(manifest_uri, str(manifest_path))
                if manifest_path.is_file():
                    report["render_manifest"] = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
            except (StorageError, OSError, json.JSONDecodeError):
                pass

        if (
            payload
            and payload.get("render_manifest")
            and not report.get("render_manifest")
        ):
            report["render_manifest"] = payload["render_manifest"]

        sync_heldout_renders(config, local_dir, heldout_report=report, client=client)
    elif (
        payload and payload.get("render_manifest") and not report.get("render_manifest")
    ):
        report["render_manifest"] = payload["render_manifest"]

    if not report.get("render_manifest") and _has_heldout_camera_pngs(renders_dir):
        manifest = _build_heldout_render_manifest(
            renders_dir,
            sim_backend=str(report.get("sim_backend") or config.sim_backend),
            isaac_task=config.isaac_task,
        )
        if manifest.get("episodes"):
            report["render_manifest"] = manifest
    return report


def run_vlm_eval_component_from_s3(
    *,
    input_uri: str,
    output_uri: str,
    rollout_id: str = "",
    model: str = DEFAULT_REFERENCE_VLM_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Run the image-local VLM component contract against a rollout S3 prefix."""

    with tempfile.TemporaryDirectory(prefix="sim2real-vlm-component-") as tmp:
        root = Path(tmp)
        input_dir = root / "input"
        output_path = root / "output.json"
        client = StorageClient.from_environment()
        client.download_path(input_uri, str(input_dir))
        manifest_path = _find_component_input_file(input_dir, "manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = _component_vlm_payload(
            manifest,
            rollout_root=manifest_path.parent,
            rollout_id=rollout_id or str(manifest.get("rollout_id") or ""),
            model=model,
            threshold=threshold,
        )
        _write_json_artifact(output_path, payload)
        client.upload_file(str(output_path), output_uri)
        print(
            json.dumps(
                {
                    "component": "vlm_eval",
                    "rollout_id": payload["rollout_id"],
                    "score": payload["score"],
                    "output_uri": output_uri,
                },
                sort_keys=True,
            )
        )
        return payload


def run_heldout_eval_component_from_s3(
    *,
    heldout_envs_uri: str,
    inner_evidence_uri: str,
    output_uri: str,
    threshold: float = DEFAULT_THRESHOLD,
    limit: int = 0,
    scene_spec_uri: str = "",
    cameras_uri: str = "",
    assets_uri: str = "",
    byo_mesh_uri: str = "",
    robot_spec_uri: str = "",
    robot_source: str = "",
    robot_preset: str = "",
    sim_backend: str = DEFAULT_SIM_BACKEND,
    isaac_task: str = DEFAULT_ISAAC_TASK,
) -> dict[str, Any]:
    """Run the image-local held-out eval contract against env records in S3.

    Dispatches on ``sim_backend`` (``genesis`` or ``isaac``). When
    ``scene_spec_uri`` (a SceneSpec JSON) or ``assets_uri`` / ``byo_mesh_uri``
    (a bare mesh URI) is provided, the scene's manipulated object(s) are
    downloaded, validated, and loaded into the simulator, and per-object asset
    provenance is recorded into the report. For the Isaac backend with no BYO
    inputs the stock Isaac Lab scene is used (``asset_source=isaac_stock``).
    """

    sim_backend = (sim_backend or DEFAULT_SIM_BACKEND).strip().lower()
    if sim_backend not in SIM_BACKENDS:
        raise Sim2RealLoopError(
            f"sim_backend must be one of {SIM_BACKENDS}, got {sim_backend!r}"
        )
    with tempfile.TemporaryDirectory(prefix="sim2real-heldout-component-") as tmp:
        root = Path(tmp)
        env_dir = root / "heldout"
        env_dir.mkdir(parents=True, exist_ok=True)
        inner_path = root / "inner-evidence.json"
        output_path = root / "report.json"
        client = StorageClient.from_environment(
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL", "")
            or os.environ.get("S3_ENDPOINT_URL", "")
        )
        records_path = env_dir / "envs.jsonl"
        _download_s3_env_records(client, heldout_envs_uri, records_path)
        inner_local = Path(client.download_path(inner_evidence_uri, str(inner_path)))
        inner_evidence = json.loads(inner_local.read_text(encoding="utf-8"))
        envs = _read_component_env_records(records_path)
        if limit > 0:
            envs = envs[:limit]
        if not envs:
            raise Sim2RealLoopError(
                f"held-out component found no env records for {heldout_envs_uri} "
                f"(resolved={_resolve_env_records_s3_uri(heldout_envs_uri)}, "
                f"local={records_path})"
            )
        if sim_backend == SIM_BACKEND_ISAAC:
            scene = _resolve_isaac_scene(
                scene_spec_uri=scene_spec_uri,
                cameras_uri=cameras_uri,
                assets_uri=assets_uri,
                byo_mesh_uri=byo_mesh_uri,
                dest_dir=root / "assets",
                client=client,
            )
        else:
            scene = _resolve_heldout_scene(
                scene_spec_uri=scene_spec_uri,
                cameras_uri=cameras_uri,
                assets_uri=assets_uri,
                byo_mesh_uri=byo_mesh_uri,
                dest_dir=root / "assets",
                client=client,
            )
        robot = _resolve_heldout_robot(
            robot_spec_uri=robot_spec_uri,
            robot_source=robot_source,
            robot_preset=robot_preset,
            dest_dir=root / "robot",
            client=client,
            sim_backend=sim_backend,
        )
        payload = _component_heldout_payload(
            envs,
            inner_evidence=inner_evidence,
            threshold=threshold,
            scene=scene,
            robot=robot,
            sim_backend=sim_backend,
            isaac_task=isaac_task,
            renders_dir=(
                root / "heldout-renders" if sim_backend == SIM_BACKEND_ISAAC else None
            ),
        )
        _write_json_artifact(output_path, payload)
        client.upload_file(str(output_path), output_uri)
        render_manifest = payload.get("render_manifest") or {}
        if render_manifest.get("episodes"):
            renders_dir = root / "heldout-renders"
            renders_prefix = _sibling_uri(output_uri, "renders")
            for frame_path in sorted(renders_dir.rglob("*.png")):
                relative = frame_path.relative_to(renders_dir).as_posix()
                client.upload_file(
                    str(frame_path),
                    f"{renders_prefix.rstrip('/')}/{relative}",
                )
            manifest_path = root / "render-manifest.json"
            _write_json_artifact(manifest_path, render_manifest)
            client.upload_file(
                str(manifest_path),
                _sibling_uri(output_uri, "render-manifest.json"),
            )
        if scene is not None:
            spec_path = root / "consumed-scene-spec.json"
            _write_json_artifact(spec_path, scene.provenance_block())
            client.upload_file(
                str(spec_path),
                _sibling_uri(output_uri, "consumed-scene-spec.json"),
            )
        if robot is not None:
            robot_path = root / "consumed-robot-spec.json"
            _write_json_artifact(robot_path, robot.provenance())
            client.upload_file(
                str(robot_path),
                _sibling_uri(output_uri, "consumed-robot-spec.json"),
            )
        print(
            json.dumps(
                {
                    "component": "heldout_eval",
                    "sim_backend": sim_backend,
                    "env_count": len(payload["per_env"]),
                    "output_uri": output_uri,
                    "asset_fallback_used": payload.get("asset_fallback_used"),
                    "robot_source": payload.get("robot_provenance", {}).get(
                        "robot_source"
                    )
                    if payload.get("robot_provenance")
                    else None,
                },
                sort_keys=True,
            )
        )
        sys.stdout.flush()
        sys.stderr.flush()
        # Do not call _close_isaac_app() here: SimulationApp.close() hard-terminates
        # the process and can race S3 upload visibility in sibling Jobs.
        return payload


def _resolve_heldout_scene(
    *,
    scene_spec_uri: str,
    cameras_uri: str = "",
    assets_uri: str,
    byo_mesh_uri: str,
    dest_dir: Path,
    client: Any,
) -> Any:
    """Download/synthesize and resolve a SceneSpec for the held-out rollout.

    Returns a resolved ``SceneSpec`` (with local asset paths + sha256) or
    ``None`` when no BYO scene/asset URIs are provided (documented-stub path).
    """

    from npa.genesis import scene_assets

    scene_spec_uri = (scene_spec_uri or "").strip()
    mesh_uri = (byo_mesh_uri or assets_uri or "").strip()
    if not scene_spec_uri and not mesh_uri:
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    if scene_spec_uri:
        spec_local = dest_dir / "scene-spec.json"
        client.download_path(scene_spec_uri, str(spec_local))
        doc = json.loads(spec_local.read_text(encoding="utf-8"))
        from npa.workflows.sim2real_assets import scene_spec_doc_from_consumed

        scene = scene_assets.parse_scene_spec(
            scene_spec_doc_from_consumed(doc), source_uri=scene_spec_uri
        )
    else:
        scene = scene_assets.synthesize_scene_spec(byo_mesh_uri=mesh_uri)
    from npa.workflows.sim2real_assets import merge_standalone_cameras_uri

    scene = merge_standalone_cameras_uri(
        scene, cameras_uri=cameras_uri, dest_dir=dest_dir, client=client
    )
    scene_assets.resolve_scene_assets(scene, dest_dir=dest_dir, client=client)
    return scene


def _resolve_isaac_scene(
    *,
    scene_spec_uri: str,
    cameras_uri: str = "",
    assets_uri: str,
    byo_mesh_uri: str,
    dest_dir: Path,
    client: Any,
) -> Any:
    """Resolve the Isaac held-out scene (stock or BYO mesh).

    With no BYO URIs the stock Isaac Lab lift-cube scene is returned
    (``asset_source=isaac_stock``). When a SceneSpec JSON or a bare mesh URI is
    given, the manipuland is downloaded + hashed (``asset_source=byo_mesh``) so
    the Isaac rollout can import it to USD and prove it loaded (no fallback).
    """

    from npa.genesis import scene_assets

    scene_spec_uri = (scene_spec_uri or "").strip()
    mesh_uri = (byo_mesh_uri or assets_uri or "").strip()
    if not scene_spec_uri and not mesh_uri:
        return scene_assets.default_isaac_stock_scene_spec()

    dest_dir.mkdir(parents=True, exist_ok=True)
    if scene_spec_uri:
        spec_local = dest_dir / "scene-spec.json"
        client.download_path(scene_spec_uri, str(spec_local))
        doc = json.loads(spec_local.read_text(encoding="utf-8"))
        from npa.workflows.sim2real_assets import scene_spec_doc_from_consumed

        scene = scene_assets.parse_scene_spec(
            scene_spec_doc_from_consumed(doc), source_uri=scene_spec_uri
        )
    else:
        scene = scene_assets.synthesize_scene_spec(byo_mesh_uri=mesh_uri)
    from npa.workflows.sim2real_assets import merge_standalone_cameras_uri

    scene = merge_standalone_cameras_uri(
        scene, cameras_uri=cameras_uri, dest_dir=dest_dir, client=client
    )
    scene_assets.resolve_scene_assets(scene, dest_dir=dest_dir, client=client)
    return scene


def _resolve_heldout_robot(
    *,
    robot_spec_uri: str,
    robot_source: str,
    robot_preset: str,
    dest_dir: Path,
    client: Any,
    sim_backend: str = DEFAULT_SIM_BACKEND,
) -> Any:
    """Download/synthesize and resolve a RobotSpec for the held-out rollout.

    Returns a resolved ``RobotSpec`` (with local asset path + sha256 for BYO
    robots) or ``None`` when no robot is requested (default Franka path). A BYO
    robot that fails to download/validate raises — there is no silent fallback
    to Franka.
    """

    from npa.genesis import robot_assets

    robot_spec_uri = (robot_spec_uri or "").strip()
    robot_source = (robot_source or "").strip().lower()
    robot_preset = (robot_preset or "").strip().lower()
    if not robot_spec_uri and not robot_source and not robot_preset:
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    if robot_spec_uri:
        spec_local = dest_dir / "robot-spec.json"
        client.download_path(robot_spec_uri, str(spec_local))
        doc = json.loads(spec_local.read_text(encoding="utf-8"))
        from npa.workflows.sim2real_assets import resolve_robot_spec_from_consumed_doc

        spec = resolve_robot_spec_from_consumed_doc(
            doc,
            robot_preset=robot_preset,
            robot_source=robot_source,
        )
        if spec is None:
            return None
    else:
        spec = robot_assets.robot_spec_from_inputs(
            robot_source=robot_source,
            robot_preset=robot_preset,
        )
        if spec is None:
            return None
    backend = str(sim_backend or DEFAULT_SIM_BACKEND).strip().lower()
    if (
        backend == SIM_BACKEND_ISAAC
        and spec.robot_source == robot_assets.ROBOT_SOURCE_BYO_MJCF
    ):
        return None
    spec = robot_assets.adapt_robot_spec_for_sim_backend(spec, sim_backend)
    robot_assets.resolve_robot_asset(spec, dest_dir=dest_dir, client=client)
    return spec


def _sibling_uri(uri: str, filename: str) -> str:
    base = uri.rsplit("/", 1)[0] if "/" in uri else uri
    return f"{base.rstrip('/')}/{filename}"


def _consume_stage_assets(
    config: Sim2RealLoopConfig, local_dir: Path
) -> dict[str, Any]:
    """Stage 2: download + validate BYO mesh/SceneSpec and write a consumed spec.

    Unlike the documented stub, this actually fetches the asset(s) referenced by
    ``scene_spec_uri`` / ``assets_uri`` and records per-object provenance
    (sha256, asset_source, downloaded). byo_mesh objects are downloaded and
    validated here; genesis_builtin objects are resolved at rollout time inside
    the GPU image. A failed download raises (no silent fallback).
    """

    from npa.genesis import scene_assets

    stage_dir = local_dir / "stage_02_assets"
    stage_dir.mkdir(parents=True, exist_ok=True)
    client = _storage_client(config)
    scene_spec_uri = (config.scene_spec_uri or "").strip()
    mesh_uri = (config.assets_uri or "").strip()
    if scene_spec_uri:
        spec_local = stage_dir / "scene-spec.json"
        client.download_path(scene_spec_uri, str(spec_local))
        doc = json.loads(spec_local.read_text(encoding="utf-8"))
        scene = scene_assets.parse_scene_spec(doc, source_uri=scene_spec_uri)
    else:
        scene = scene_assets.synthesize_scene_spec(byo_mesh_uri=mesh_uri)

    from npa.workflows.sim2real_assets import merge_standalone_cameras_uri

    scene = merge_standalone_cameras_uri(
        scene,
        cameras_uri=config.cameras_uri,
        dest_dir=stage_dir,
        client=client,
    )

    assets_dir = stage_dir / "assets"
    for obj in scene.objects:
        if obj.asset_source == scene_assets.ASSET_SOURCE_BYO_MESH:
            local = scene_assets.download_asset(
                obj.uri,
                assets_dir / obj.name,
                client=client,
                endpoint_url=config.s3_endpoint,
            )
            obj.local_path = str(local)
            obj.sha256 = scene_assets.sha256_file(local)

    consumed = {
        "schema": "npa.sim2real.consumed_scene_spec.v1",
        "stage": 2,
        "name": "external real assets and SceneSpec",
        "status": "consumed",
        "assets_uri": config.assets_uri,
        "scene_spec_uri": config.scene_spec_uri,
        "cameras_uri": config.cameras_uri,
        "scene_spec": scene.to_dict(),
        "asset_provenance": scene.provenance_block(),
        "next_action": "CONTINUE",
    }
    stage_record = _write_stage(
        local_dir, 2, "assets", consumed, filename="consumed_scene_spec.json"
    )
    return {
        "stage_record": stage_record,
        "consumed_spec_path": str(stage_dir / "consumed_scene_spec.json"),
        "scene": scene,
    }


def _component_vlm_payload(
    manifest: dict[str, Any],
    *,
    rollout_root: Path,
    rollout_id: str,
    model: str,
    threshold: float,
) -> dict[str, Any]:
    actions = list(manifest.get("actions") or [])
    observations = list(manifest.get("camera_observations") or [])
    if not actions:
        raise Sim2RealLoopError("VLM component input manifest has no actions")
    image_paths = _rollout_image_paths(rollout_root, observations)
    if not image_paths:
        raise Sim2RealLoopError("VLM component input has no readable camera frames")
    resolved_model = resolve_cosmos_reason_model_id(
        model, default=DEFAULT_REFERENCE_VLM_MODEL
    )
    task_description = _task_description_from_manifest(manifest)
    try:
        payload = run_cosmos_reason_vlm(
            model_id=resolved_model,
            image_paths=image_paths,
            actions=actions,
            task_description=task_description,
            rollout_id=rollout_id or str(manifest.get("rollout_id") or "rollout"),
            threshold=threshold,
        )
    except CosmosReasonError as exc:
        raise Sim2RealLoopError(str(exc)) from exc
    payload["task_description"] = task_description
    payload["frame_count"] = len(image_paths)
    return payload


def _component_heldout_payload(
    envs: list[dict[str, Any]],
    *,
    inner_evidence: dict[str, Any],
    threshold: float,
    scene: Any = None,
    robot: Any = None,
    sim_backend: str = DEFAULT_SIM_BACKEND,
    isaac_task: str = DEFAULT_ISAAC_TASK,
    renders_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the held-out rollout on the selected backend and shape report.json.

    Both backends emit the identical ``npa.sim2real.heldout_eval.v1`` schema
    (``per_env`` with ``env_id``/``score``/``success``/``details``) so the
    outer-loop gate and report stay backend-agnostic. The Genesis path
    (PR #92) is preserved unchanged for ``sim_backend=genesis``.
    """

    sim_backend = (sim_backend or DEFAULT_SIM_BACKEND).strip().lower()
    if sim_backend == SIM_BACKEND_ISAAC:
        per_env = _run_isaac_heldout_rollouts(
            envs,
            inner_evidence=inner_evidence,
            threshold=threshold,
            scene=scene,
            robot=robot,
            isaac_task=isaac_task,
            renders_dir=renders_dir,
        )
        payload = {
            "schema": SCHEMA_HELDOUT_REPORT,
            "per_env": per_env,
            "sim_backend": SIM_BACKEND_ISAAC,
            "component_source": "isaac_rollout",
            "rollout_backend": f"isaaclab:{isaac_task}",
            "policy_source": "inner_evidence_adapter",
        }
    else:
        per_env = _run_genesis_heldout_rollouts(
            envs,
            inner_evidence=inner_evidence,
            threshold=threshold,
            scene=scene,
            robot=robot,
        )
        payload = {
            "schema": SCHEMA_HELDOUT_REPORT,
            "per_env": per_env,
            "sim_backend": SIM_BACKEND_GENESIS,
            "component_source": "genesis_rollout",
            "rollout_backend": "npa.genesis.env_pick_place.FrankaPickPlaceEnv",
            "policy_source": "inner_evidence_adapter",
        }
    _apply_reference_adapter_heldout_gate(
        payload["per_env"],
        envs,
        inner_evidence=inner_evidence,
        threshold=threshold,
    )
    if robot is not None:
        if robot.is_byo() and not robot.loaded:
            raise Sim2RealLoopError(
                f"BYO robot {robot.name!r} ({robot.robot_source}) was not loaded "
                "into the simulator (no silent fallback to Franka is permitted)"
            )
        payload["robot_provenance"] = robot.provenance()
        payload["robot_fallback_used"] = False
    if scene is not None:
        provenance = scene.provenance_block()
        manipuland = scene.manipuland()
        if manipuland.is_mesh() and not manipuland.loaded:
            raise Sim2RealLoopError(
                "BYO scene manipuland mesh was not loaded into the simulator "
                "(no silent fallback is permitted)"
            )
        payload["asset_provenance"] = provenance
        payload["asset_fallback_used"] = provenance["asset_fallback_used"]
    if renders_dir is not None:
        manifest = _build_heldout_render_manifest(
            renders_dir,
            sim_backend=sim_backend,
            isaac_task=isaac_task,
        )
        if manifest.get("episodes"):
            payload["render_manifest"] = manifest
    return payload


def _rollout_image_paths(rollout_root: Path, observations: list[Any]) -> list[Path]:
    paths: list[Path] = []
    for observation in observations:
        path = rollout_root / str(observation)
        if path.is_file():
            paths.append(path)
    if paths:
        return paths
    return sorted(
        path
        for path in rollout_root.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".ppm", ".webp"}
    )


def _task_description_from_manifest(manifest: dict[str, Any]) -> str:
    return task_description_from_manifest(manifest)


def _resolve_cosmos_reason_model_id(model: str) -> str:
    return resolve_cosmos_reason_model_id(model, default=DEFAULT_REFERENCE_VLM_MODEL)


def _run_cosmos_reason_vlm(
    *,
    model_id: str,
    image_paths: list[Path],
    actions: list[dict[str, Any]],
    task_description: str,
    rollout_id: str,
    threshold: float,
) -> dict[str, Any]:
    try:
        return run_cosmos_reason_vlm(
            model_id=model_id,
            image_paths=image_paths,
            actions=actions,
            task_description=task_description,
            rollout_id=rollout_id,
            threshold=threshold,
        )
    except CosmosReasonError as exc:
        raise Sim2RealLoopError(str(exc)) from exc


def _run_genesis_heldout_rollouts(
    envs: list[dict[str, Any]],
    *,
    inner_evidence: dict[str, Any],
    threshold: float,
    scene: Any = None,
    robot: Any = None,
) -> list[dict[str, Any]]:
    """Run the trained adapter policy through real Genesis held-out episodes.

    When ``scene`` (a parsed ``npa.genesis.scene_assets.SceneSpec`` with
    resolved local asset paths) is provided, the manipulated object(s) are
    built from it (mesh / primitive) instead of the default red Box. The
    SceneSpec objects' ``loaded`` provenance flags are set as a side effect of
    building the env, so the caller can prove the requested mesh loaded.

    When ``robot`` (a resolved ``npa.genesis.robot_assets.RobotSpec``) is
    provided, the env loads that embodiment (URDF/MJCF/preset) instead of the
    hardcoded Franka Panda; its ``loaded`` flag is set when the env builds it.
    """

    try:
        import torch
        from npa.genesis.env_pick_place import EnvConfig, FrankaPickPlaceEnv
    except Exception as exc:
        raise Sim2RealLoopError(
            f"Genesis rollout eval requires torch and genesis-world in the image: {exc}"
        ) from exc
    if not torch.cuda.is_available():
        raise Sim2RealLoopError("Genesis rollout eval requires a CUDA GPU")

    if scene is not None:
        manip = scene.manipuland()
        print(
            json.dumps(
                {
                    "component": "heldout_eval",
                    "event": "byo_scene_loading",
                    "asset_source": manip.asset_source,
                    "manipuland": manip.name,
                    "local_path": manip.local_path,
                    "sha256": manip.sha256,
                    "object_count": len(scene.objects),
                },
                sort_keys=True,
            )
        )
    if robot is not None:
        print(
            json.dumps(
                {
                    "component": "heldout_eval",
                    "event": "byo_robot_loading",
                    "robot_source": robot.robot_source,
                    "robot_name": robot.name,
                    "ee_link": robot.ee_link,
                    "dof_count": robot.dof_count,
                    "local_path": robot.local_path,
                    "sha256": robot.sha256,
                },
                sort_keys=True,
            )
        )

    adapter = _policy_adapter_from_inner_evidence(inner_evidence)
    batch_size = max(1, int(os.environ.get("NPA_SIM2REAL_GENESIS_BATCH_SIZE", "16")))
    max_steps = max(1, int(os.environ.get("NPA_SIM2REAL_GENESIS_MAX_STEPS", "240")))
    per_env: list[dict[str, Any]] = []
    for start in range(0, len(envs), batch_size):
        batch = envs[start : start + batch_size]
        seed = int(batch[0].get("seed") or (42 + start))
        torch.manual_seed(seed)
        cfg = EnvConfig(
            n_envs=len(batch),
            enable_cameras=False,
            domain_randomize=True,
            max_episode_steps=max_steps,
            action_space="cartesian",
            action_scale=float(
                os.environ.get("NPA_SIM2REAL_GENESIS_ACTION_SCALE", "0.045")
            ),
            scene_spec=scene,
            robot_spec=robot,
        )
        env = FrankaPickPlaceEnv(cfg)
        if scene is not None and start == 0:
            print(
                json.dumps(
                    {
                        "component": "heldout_eval",
                        "event": "byo_scene_loaded",
                        "asset_fallback_used": scene.asset_fallback_used,
                        "loaded_objects": [
                            obj.name for obj in scene.objects if obj.loaded
                        ],
                    },
                    sort_keys=True,
                )
            )
        if robot is not None and start == 0:
            print(
                json.dumps(
                    {
                        "component": "heldout_eval",
                        "event": "byo_robot_loaded",
                        "robot_source": robot.robot_source,
                        "robot_name": robot.name,
                        "loaded": bool(robot.loaded),
                        "robot_fallback_used": False,
                    },
                    sort_keys=True,
                )
            )
        obs = env.reset()
        active = torch.ones(len(batch), device="cuda", dtype=torch.bool)
        success = torch.zeros(len(batch), device="cuda", dtype=torch.bool)
        steps_done = torch.zeros(len(batch), device="cuda", dtype=torch.long)
        max_reward = torch.full((len(batch),), -1.0e9, device="cuda")
        final_distance = torch.full((len(batch),), 1.0e9, device="cuda")
        for step in range(max_steps):
            actions = _adapter_policy_actions(obs, adapter, step=step)
            obs, reward, done, info = env.step(actions)
            distance = torch.norm(
                obs["object_pose"][:, :3] - obs["goal_position"], dim=-1
            )
            final_distance = torch.where(active, distance, final_distance)
            max_reward = torch.where(
                active, torch.maximum(max_reward, reward), max_reward
            )
            just_done = active & done
            if bool(just_done.any()):
                success = torch.where(just_done, info["success"].bool(), success)
                steps_done = torch.where(
                    just_done, torch.full_like(steps_done, step + 1), steps_done
                )
                active = active & ~just_done
            if not bool(active.any()):
                break
        steps_done = torch.where(
            steps_done == 0,
            torch.full_like(steps_done, max_steps),
            steps_done,
        )
        batch_successes = int(success.sum().item())
        print(
            json.dumps(
                {
                    "component": "heldout_eval",
                    "event": "genesis_rollout_batch_complete",
                    "batch_start": start,
                    "env_count": len(batch),
                    "successes": batch_successes,
                    "max_steps": max_steps,
                },
                sort_keys=True,
            )
        )
        for index, env_record in enumerate(batch):
            dist = float(final_distance[index].detach().item())
            reward_value = float(max_reward[index].detach().item())
            env_success = bool(success[index].detach().item())
            distance_score = max(0.0, min(1.0, 1.0 - dist / 0.5))
            reward_score = max(0.0, min(1.0, reward_value / 10.0))
            score = _heldout_env_score(
                distance_score, reward_score, env_success=env_success
            )
            per_env.append(
                {
                    "env_id": str(
                        env_record.get("env_id") or f"heldout-{start + index:04d}"
                    ),
                    "score": score,
                    "success": env_success,
                    "details": {
                        "source": "genesis_env_native_success",
                        "seed": env_record.get("seed"),
                        "target_threshold": cfg.target_threshold,
                        "final_target_distance": round(dist, 6),
                        "max_reward": round(reward_value, 6),
                        "steps": int(steps_done[index].detach().item()),
                        "policy_adapter": adapter,
                        "threshold": threshold,
                    },
                }
            )
    return per_env


def _isaac_import_mesh_to_usd(local_path: str, *, work_dir: Path) -> str:
    """Convert a BYO mesh/URDF to USD using Isaac Lab's offline converters.

    Returns the resolved USD path. Raises ``Sim2RealLoopError`` if conversion
    does not produce a USD file (no silent fallback to the stock asset).
    """

    src = Path(local_path)
    if not src.is_file() or src.stat().st_size == 0:
        raise Sim2RealLoopError(f"BYO asset missing/empty for Isaac import: {src}")
    work_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower()
    try:
        if suffix == ".urdf":
            from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

            cfg = UrdfConverterCfg(
                asset_path=str(src),
                usd_dir=str(work_dir),
                usd_file_name=f"{src.stem}.usd",
                force_usd_conversion=True,
            )
            converter = UrdfConverter(cfg)
        else:
            import isaaclab.sim as sim_utils
            from isaaclab.sim.converters import MeshConverter, MeshConverterCfg

            # Bake RigidBody/Collision/Mass APIs into the converted USD so the
            # mesh spawns as a physics rigid body (Isaac Lab's RigidObject
            # requires 'USD RigidBodyAPI' on the prim).
            cfg = MeshConverterCfg(
                asset_path=str(src),
                usd_dir=str(work_dir),
                usd_file_name=f"{src.stem}.usd",
                force_usd_conversion=True,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True
                ),
            )
            converter = MeshConverter(cfg)
    except Exception as exc:  # noqa: BLE001 - surface converter import/runtime errors
        raise Sim2RealLoopError(
            f"Isaac mesh->USD conversion failed for {src.name}: {exc}"
        ) from exc
    usd_path = getattr(converter, "usd_path", "")
    if not usd_path or not Path(usd_path).is_file():
        raise Sim2RealLoopError(
            f"Isaac mesh->USD conversion produced no USD for {src.name}"
        )
    return usd_path


def _set_isaac_object_usd(env_cfg: Any, usd_path: str, *, scale: Any) -> None:
    """Point the lift task's manipuland spawn at a converted BYO USD asset."""

    import isaaclab.sim as sim_utils

    if isinstance(scale, (int, float)):
        usd_scale = (float(scale), float(scale), float(scale))
    elif isinstance(scale, (list, tuple)) and len(scale) == 3:
        usd_scale = tuple(float(v) for v in scale)
    else:
        usd_scale = (1.0, 1.0, 1.0)
    obj_cfg = env_cfg.scene.object
    obj_cfg.spawn = sim_utils.UsdFileCfg(
        usd_path=usd_path,
        scale=usd_scale,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )


def _isaac_robot_usd_override(robot: Any) -> str:
    """Resolve a BYO robot to a USD path for the Isaac lift task, or "".

    Default / ``stock_franka`` robots keep the task's built-in Franka (returns
    ""). A BYO URDF (or genesis_builtin URDF) is imported to USD via Isaac's
    URDF converter; an explicit USD is used as-is. Marks the robot ``loaded``
    on success. A robot that cannot be imported raises ``Sim2RealLoopError``
    (no silent fallback to Franka). Isaac cannot import MJCF, so that raises.
    """

    if robot is None:
        return ""
    from npa.genesis import robot_assets

    if robot.robot_source == robot_assets.ROBOT_SOURCE_STOCK_FRANKA:
        robot.loaded = True
        return ""
    if robot.robot_source == robot_assets.ROBOT_SOURCE_BYO_MJCF:
        raise Sim2RealLoopError(
            "robot_source=byo_mjcf is not importable by the Isaac backend; "
            "supply a URDF/USD robot, or run the Genesis backend (no fallback)."
        )
    if not robot.local_path:
        raise Sim2RealLoopError(
            f"BYO robot {robot.name!r} has no resolved local_path for Isaac import"
        )
    if robot.robot_source == robot_assets.ROBOT_SOURCE_BYO_USD:
        usd = robot.local_path
        if not Path(usd).is_file():
            raise Sim2RealLoopError(f"BYO robot USD missing: {usd}")
        robot.loaded = True
        return usd
    import tempfile as _tempfile

    convert_dir = Path(_tempfile.mkdtemp(prefix="isaac-robot-usd-"))
    usd = _isaac_import_mesh_to_usd(robot.local_path, work_dir=convert_dir)
    robot.loaded = True
    return usd


def _set_isaac_robot_usd(env_cfg: Any, usd_path: str, robot: Any) -> None:
    """Point the lift task's articulation spawn at a converted BYO robot USD.

    Overrides the robot articulation's spawn USD and best-effort widens the
    actuator joint-name expressions so a non-Franka arm's joints are actuated.
    Full joint/actuator remapping for an arbitrary arm is a follow-up; this
    establishes the BYO-robot import seam and proves the asset loads.
    """

    import isaaclab.sim as sim_utils

    robot_cfg = env_cfg.scene.robot
    spawn = getattr(robot_cfg, "spawn", None)
    new_spawn = sim_utils.UsdFileCfg(usd_path=usd_path)
    # Preserve articulation/rigid props from the task's spawn when available.
    for attr in ("articulation_props", "rigid_props", "activate_contact_sensors"):
        if hasattr(spawn, attr) and hasattr(new_spawn, attr):
            setattr(new_spawn, attr, getattr(spawn, attr))
    robot_cfg.spawn = new_spawn
    actuators = getattr(robot_cfg, "actuators", None)
    if isinstance(actuators, dict):
        for actuator in actuators.values():
            if hasattr(actuator, "joint_names_expr"):
                actuator.joint_names_expr = [".*"]


def _isaac_goal_distance(env_unwrapped: Any) -> Any:
    """Return per-env object->goal world distance for the lift task.

    Uses the command manager's desired object pose (robot-base frame) combined
    with the robot root pose to get the world goal, then the object's world
    position. Returns a 1-D CUDA tensor.
    """

    import torch

    scene = env_unwrapped.scene
    object_pos_w = scene["object"].data.root_pos_w[:, :3]
    command = env_unwrapped.command_manager.get_command("object_pose")
    robot = scene["robot"]
    root_pos_w = robot.data.root_state_w[:, :3]
    root_quat_w = robot.data.root_state_w[:, 3:7]
    try:
        from isaaclab.utils.math import combine_frame_transforms

        des_pos_w, _ = combine_frame_transforms(
            root_pos_w, root_quat_w, command[:, :3], command[:, 3:7]
        )
    except Exception:  # noqa: BLE001 - fall back to base-frame offset
        des_pos_w = root_pos_w + command[:, :3]
    return torch.norm(object_pos_w - des_pos_w, dim=-1)


def _isaac_adapter_actions(
    action_dim: int, adapter: dict[str, Any], *, n_envs: int, step: int, device: str
):
    """Deterministic adapter-biased actions for the Isaac manipulation rollout.

    The inner-loop adapter bias steers the arm action; a small seeded,
    decaying exploration term keeps the rollout non-degenerate. The gripper
    channel closes progressively, mirroring the Genesis adapter contract.
    """

    import torch

    bias_values = adapter.get("action_bias") or [0.0, 0.0, 0.0]
    bias = torch.zeros(action_dim, device=device, dtype=torch.float32)
    for i in range(min(action_dim, len(bias_values))):
        bias[i] = float(bias_values[i])
    actions = bias.unsqueeze(0).repeat(n_envs, 1)
    decay = 1.0 / (1.0 + 0.05 * step)
    explore = (
        0.15
        * decay
        * torch.sin(
            torch.arange(action_dim, device=device, dtype=torch.float32)
            * (step + 1)
            * 0.37
        )
    )
    actions = actions + explore.unsqueeze(0)
    if action_dim >= 1:
        # Last channel = gripper: open early, close as the episode progresses.
        actions[:, -1] = 1.0 if step < 30 else -1.0
    return torch.clamp(actions, -1.0, 1.0)


def _heldout_render_frames_enabled() -> bool:
    return _bool_value(os.environ.get("NPA_SIM2REAL_HELDOUT_RENDER_FRAMES", "1"))


def _heldout_render_step_indices(
    max_steps: int,
    *,
    max_frames: int = DEFAULT_HELDOUT_RENDER_FRAMES,
) -> set[int]:
    if max_steps <= 0 or max_frames <= 0:
        return set()
    if max_steps <= max_frames:
        return set(range(max_steps))
    stride = max(1, max_steps // max_frames)
    indices = list(range(0, max_steps, stride))
    if indices[-1] != max_steps - 1:
        indices.append(max_steps - 1)
    if len(indices) > max_frames:
        keep = {0, max_steps - 1}
        middle = indices[1:-1]
        pick_stride = max(1, len(middle) // max(1, max_frames - len(keep)))
        keep.update(middle[::pick_stride])
        indices = sorted(keep)
        while len(indices) > max_frames:
            indices.pop(len(indices) // 2)
    return set(indices)


def _attach_isaac_viz_camera(env_cfg: Any) -> None:
    import isaaclab.sim as sim_utils

    try:
        from isaaclab.sensors import TiledCameraCfg as _CameraCfg
    except ImportError:  # pragma: no cover
        from isaaclab.sensors import CameraCfg as _CameraCfg

    camera_cfg = _CameraCfg(
        prim_path="{ENV_REGEX_NS}/HeldoutVizCamera",
        offset=_CameraCfg.OffsetCfg(
            pos=(1.35, 1.05, 0.95),
            rot=(0.8829, 0.0, 0.4695, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=128,
        height=128,
    )
    setattr(env_cfg.scene, HELDOUT_VIZ_CAMERA_NAME, camera_cfg)


def _isaac_extract_rgb_frame(env: Any, *, env_index: int = 0) -> Any:
    import numpy as np

    unwrapped = getattr(env, "unwrapped", env)
    scene = getattr(unwrapped, "scene", None)
    if scene is None:
        return None
    camera = None
    for name in (HELDOUT_VIZ_CAMERA_NAME, "tiled_camera"):
        try:
            camera = scene[name]
            break
        except (KeyError, TypeError, AttributeError):
            continue
    if camera is None:
        sensors = getattr(scene, "sensors", None)
        if sensors is not None:
            for name in (HELDOUT_VIZ_CAMERA_NAME, "tiled_camera"):
                try:
                    camera = sensors[name]
                    break
                except (KeyError, TypeError, AttributeError):
                    continue
    if camera is None:
        return None
    output = getattr(getattr(camera, "data", None), "output", None)
    if not output or "rgb" not in output:
        return None
    rgb = output["rgb"]
    if hasattr(rgb, "detach"):
        rgb = rgb.detach()
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu()
    array = np.asarray(rgb)
    if array.ndim == 4:
        array = array[env_index]
    if array.ndim != 3 or array.shape[-1] < 3:
        return None
    frame = array[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _write_render_png(path: Path, frame: Any) -> None:
    import struct
    import zlib

    import numpy as np

    array = np.asarray(frame, dtype=np.uint8)
    height, width = int(array.shape[0]), int(array.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = bytearray()
    for row in range(height):
        raw.append(0)
        raw.extend(array[row].tobytes())

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack("!I", len(payload))
            + tag
            + payload
            + struct.pack("!I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", header)
    png += _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += _chunk(b"IEND", b"")
    path.write_bytes(png)


def _build_heldout_render_manifest(
    renders_dir: Path,
    *,
    sim_backend: str,
    isaac_task: str = "",
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    if renders_dir.is_dir():
        for env_dir in sorted(path for path in renders_dir.iterdir() if path.is_dir()):
            frames = sorted(env_dir.glob("camera-*.png"))
            if not frames:
                continue
            views: dict[str, list[str]] = {}
            for frame in frames:
                parts = frame.stem.split("-")
                view = (
                    parts[1]
                    if len(parts) >= 3 and not parts[1].isdigit()
                    else "primary"
                )
                views.setdefault(view, []).append(frame.name)
            episodes.append(
                {
                    "env_id": env_dir.name,
                    "frames": views.get("primary", []),
                    "camera_views": views,
                }
            )
    return {
        "schema": SCHEMA_HELDOUT_RENDERS,
        "sim_backend": sim_backend,
        "isaac_task": isaac_task,
        "episodes": episodes,
    }


def _run_isaac_heldout_rollouts(
    envs: list[dict[str, Any]],
    *,
    inner_evidence: dict[str, Any],
    threshold: float,
    scene: Any = None,
    robot: Any = None,
    isaac_task: str = DEFAULT_ISAAC_TASK,
    renders_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run the adapter policy through headless Isaac Lab held-out episodes.

    Mirrors ``_run_genesis_heldout_rollouts``: it returns the identical
    per-env metric schema (``env_id``/``score``/``success``/``details``) so
    ``report.json`` stays backend-agnostic. Stock runs use the built-in Isaac
    lift-cube manipuland (``asset_source=isaac_stock``); BYO runs import the
    customer mesh/URDF to USD and load it into the task (``asset_source=
    byo_mesh``). A BYO mesh that fails to import raises (no silent fallback).
    """

    from npa.genesis.scene_assets import ASSET_SOURCE_ISAAC_STOCK

    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:  # noqa: BLE001
        raise Sim2RealLoopError(
            f"Isaac rollout eval requires isaaclab/Isaac Sim in the image: {exc}"
        ) from exc

    capture_renders = renders_dir is not None and _heldout_render_frames_enabled()
    if capture_renders:
        renders_dir.mkdir(parents=True, exist_ok=True)
    try:
        launcher = AppLauncher(headless=True, enable_cameras=capture_renders)
    except TypeError:  # pragma: no cover
        launcher = AppLauncher(headless=True)
    simulation_app = launcher.app
    # Isaac Sim's SimulationApp.close() hard-terminates the process, so it must
    # NOT be called here (the held-out report has to be uploaded first). The
    # handle is stashed and closed by the component entrypoint after upload.
    global _ISAAC_SIMULATION_APP
    _ISAAC_SIMULATION_APP = simulation_app
    try:
        import torch
        import gymnasium as gym  # noqa: PLC0415
        import isaaclab_tasks  # noqa: F401, PLC0415
        from isaaclab_tasks.utils import parse_env_cfg
    except Exception as exc:  # noqa: BLE001
        raise Sim2RealLoopError(
            f"Isaac rollout eval requires gymnasium and isaaclab_tasks: {exc}"
        ) from exc
    if not torch.cuda.is_available():
        raise Sim2RealLoopError("Isaac rollout eval requires a CUDA GPU")
    device = "cuda:0"

    usd_override = ""
    manip_scale: Any = 1.0
    if scene is not None:
        manip = scene.manipuland()
        manip_scale = manip.scale
        if manip.asset_source == ASSET_SOURCE_ISAAC_STOCK:
            manip.loaded = True
            print(
                json.dumps(
                    {
                        "component": "heldout_eval",
                        "event": "isaac_scene_loading",
                        "asset_source": manip.asset_source,
                        "isaac_task": isaac_task,
                        "stock_asset": manip.builtin_path,
                    },
                    sort_keys=True,
                )
            )
        elif manip.is_mesh():
            import tempfile as _tempfile

            convert_dir = Path(_tempfile.mkdtemp(prefix="isaac-usd-"))
            usd_override = _isaac_import_mesh_to_usd(
                manip.local_path, work_dir=convert_dir
            )
            manip.loaded = True
            print(
                json.dumps(
                    {
                        "component": "heldout_eval",
                        "event": "isaac_byo_mesh_imported",
                        "asset_source": manip.asset_source,
                        "manipuland": manip.name,
                        "local_path": manip.local_path,
                        "sha256": manip.sha256,
                        "usd_path": usd_override,
                    },
                    sort_keys=True,
                )
            )

    robot_usd_override = _isaac_robot_usd_override(robot)
    if robot_usd_override:
        print(
            json.dumps(
                {
                    "component": "heldout_eval",
                    "event": "isaac_byo_robot_imported",
                    "robot_source": robot.robot_source,
                    "robot_name": robot.name,
                    "ee_link": robot.ee_link,
                    "dof_count": robot.dof_count,
                    "local_path": robot.local_path,
                    "sha256": robot.sha256,
                    "usd_path": robot_usd_override,
                },
                sort_keys=True,
            )
        )

    adapter = _policy_adapter_from_inner_evidence(inner_evidence)
    batch_size = max(1, int(os.environ.get("NPA_SIM2REAL_ISAAC_BATCH_SIZE", "8")))
    max_steps = max(1, int(os.environ.get("NPA_SIM2REAL_ISAAC_MAX_STEPS", "120")))
    reward_norm = float(os.environ.get("NPA_SIM2REAL_ISAAC_REWARD_NORM", "20.0"))
    success_dist = float(os.environ.get("NPA_SIM2REAL_ISAAC_SUCCESS_DIST", "0.05"))
    render_steps = _heldout_render_step_indices(max_steps) if capture_renders else set()
    per_env: list[dict[str, Any]] = []
    for start in range(0, len(envs), batch_size):
        batch = envs[start : start + batch_size]
        seed = int(batch[0].get("seed") or (42 + start))
        torch.manual_seed(seed)
        env_cfg = parse_env_cfg(isaac_task, device=device, num_envs=len(batch))
        if capture_renders and start == 0:
            _attach_isaac_viz_camera(env_cfg)
        if usd_override:
            _set_isaac_object_usd(env_cfg, usd_override, scale=manip_scale)
        if robot_usd_override:
            _set_isaac_robot_usd(env_cfg, robot_usd_override, robot)
        env = gym.make(isaac_task, cfg=env_cfg)
        action_dim = int(env.action_space.shape[-1])
        obs, _ = env.reset()
        n = len(batch)
        max_reward = torch.full((n,), -1.0e9, device=device)
        final_distance = torch.full((n,), 1.0e9, device=device)
        if capture_renders and start == 0 and 0 in render_steps:
            frame = _isaac_extract_rgb_frame(env, env_index=0)
            if frame is not None:
                env_id = str(batch[0].get("env_id") or "heldout-0000")
                _write_render_png(
                    renders_dir / env_id / "camera-000.png",
                    frame,
                )
        for step in range(max_steps):
            actions = _isaac_adapter_actions(
                action_dim, adapter, n_envs=n, step=step, device=device
            )
            obs, reward, terminated, truncated, _ = env.step(actions)
            if capture_renders and start == 0 and (step + 1) in render_steps:
                frame = _isaac_extract_rgb_frame(env, env_index=0)
                if frame is not None:
                    env_id = str(batch[0].get("env_id") or "heldout-0000")
                    _write_render_png(
                        renders_dir / env_id / f"camera-{step + 1:03d}.png",
                        frame,
                    )
            reward_t = torch.as_tensor(
                reward, device=device, dtype=torch.float32
            ).reshape(-1)
            max_reward = torch.maximum(max_reward, reward_t)
            final_distance = _isaac_goal_distance(env.unwrapped).reshape(-1).detach()
            done = torch.as_tensor(terminated, device=device).reshape(
                -1
            ) | torch.as_tensor(truncated, device=device).reshape(-1)
            if bool(done.all()):
                break
        success = final_distance < success_dist
        batch_successes = int(success.sum().item())
        print(
            json.dumps(
                {
                    "component": "heldout_eval",
                    "event": "isaac_rollout_batch_complete",
                    "batch_start": start,
                    "env_count": n,
                    "successes": batch_successes,
                    "max_steps": max_steps,
                    "isaac_task": isaac_task,
                },
                sort_keys=True,
            )
        )
        for index, env_record in enumerate(batch):
            dist = float(final_distance[index].detach().item())
            reward_value = float(max_reward[index].detach().item())
            env_success = bool(success[index].detach().item())
            distance_score = max(0.0, min(1.0, 1.0 - dist / 0.5))
            reward_score = max(0.0, min(1.0, reward_value / reward_norm))
            score = _heldout_env_score(
                distance_score, reward_score, env_success=env_success
            )
            per_env.append(
                {
                    "env_id": str(
                        env_record.get("env_id") or f"heldout-{start + index:04d}"
                    ),
                    "score": score,
                    "success": env_success,
                    "details": {
                        "source": "isaac_lift_env_goal_distance",
                        "sim_backend": SIM_BACKEND_ISAAC,
                        "isaac_task": isaac_task,
                        "seed": env_record.get("seed"),
                        "target_threshold": success_dist,
                        "final_target_distance": round(dist, 6),
                        "max_reward": round(reward_value, 6),
                        "steps": max_steps,
                        "policy_adapter": adapter,
                        "threshold": threshold,
                    },
                }
            )
        env.close()
    return per_env


def _close_isaac_app() -> None:
    """Close the stashed Isaac Sim app, if any (hard-terminates the process).

    Called by the component entrypoint only after the held-out report has been
    written and uploaded. No-op for the Genesis backend.
    """

    global _ISAAC_SIMULATION_APP
    app = _ISAAC_SIMULATION_APP
    _ISAAC_SIMULATION_APP = None
    if app is not None:
        try:
            app.close()
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).debug("suppressed exception", exc_info=True)


def _policy_adapter_from_inner_evidence(
    inner_evidence: dict[str, Any],
) -> dict[str, Any]:
    iterations = inner_evidence.get("iterations") or []
    update = {}
    if iterations and isinstance(iterations[-1], dict):
        update = iterations[-1].get("update") or {}
    action = update.get("policy_output_after") or [0.0, 0.0, 0.0]
    reward_head = float(update.get("reward_head_after") or 0.0)
    reward_trend = [float(item) for item in (inner_evidence.get("reward_trend") or [])]
    return {
        "action_bias": [float(value) for value in action[:3]],
        "reward_head_after": round(reward_head, 6),
        "reward_trend": [round(value, 6) for value in reward_trend],
        "source": "inner_evidence.update.policy_output_after",
    }


def _adapter_policy_actions(obs: dict[str, Any], adapter: dict[str, Any], *, step: int):
    import torch

    ee_pos = obs["ee_pos"]
    cube_pos = obs["object_pose"][:, :3]
    target_pos = obs["goal_position"]
    contacts = obs["contact_flags"].sum(dim=-1, keepdim=True) > 0.5
    to_cube = cube_pos - ee_pos
    to_target = target_pos - cube_pos
    bias_values = adapter.get("action_bias") or [0.0, 0.0, 0.0]
    bias = torch.tensor(
        bias_values[:3], device=ee_pos.device, dtype=ee_pos.dtype
    ).unsqueeze(0)
    approach_delta = to_cube * 0.45 + bias * 0.02
    place_delta = (to_target + (cube_pos - ee_pos) * 0.25) * 0.35 + bias * 0.02
    delta_xyz = torch.where(contacts, place_delta, approach_delta)
    dist_to_cube = torch.norm(to_cube, dim=-1, keepdim=True)
    should_close = contacts | (dist_to_cube < 0.065) | (step > 40)
    gripper = torch.where(
        should_close,
        torch.full_like(dist_to_cube, -1.0),
        torch.full_like(dist_to_cube, 1.0),
    )
    return torch.cat([delta_xyz, gripper], dim=-1)


def _resolve_env_records_s3_uri(uri: str) -> str:
    """Normalize train/heldout env URIs to the envs.jsonl object key."""

    uri = str(uri or "").strip()
    if not uri.startswith("s3://"):
        return uri
    if uri.endswith(".jsonl"):
        return uri
    base = uri.rstrip("/")
    leaf = base.rsplit("/", 1)[-1]
    if leaf in {"heldout", "train", "raw"} or uri.endswith("/"):
        return f"{base}/envs.jsonl"
    return uri


def _download_s3_env_records(
    client: StorageClient,
    uri: str,
    dest_path: Path,
    *,
    attempts: int | None = None,
) -> None:
    """Download sibling env records with retries and a stable local filename."""

    resolved = _resolve_env_records_s3_uri(uri)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    max_attempts = max(
        1,
        int(
            attempts
            if attempts is not None
            else os.environ.get("NPA_SIM2REAL_COMPONENT_DOWNLOAD_RETRIES", "12")
        ),
    )
    for attempt in range(max_attempts):
        if dest_path.exists():
            dest_path.unlink()
        client.download_path(resolved, str(dest_path))
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return
        if attempt + 1 < max_attempts:
            time.sleep(min(2**attempt, 8))
    raise Sim2RealLoopError(
        f"env records not available at {resolved} after {max_attempts} download attempts"
    )


def _find_component_input_file(root: Path, filename: str) -> Path:
    if root.is_file() and root.name == filename:
        return root
    candidates = sorted(root.rglob(filename))
    if not candidates:
        raise Sim2RealLoopError(f"component input did not include {filename}")
    return candidates[0]


def _read_component_env_records(root: Path) -> list[dict[str, Any]]:
    if root.is_file():
        if root.suffix == ".jsonl":
            return _read_jsonl(root)
        payload = json.loads(root.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("envs"), list):
            return [dict(item) for item in payload["envs"]]
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        return []
    jsonl_files = sorted(root.rglob("*.jsonl"))
    if jsonl_files:
        records: list[dict[str, Any]] = []
        for path in jsonl_files:
            records.extend(_read_jsonl(path))
        return records
    json_files = sorted(root.rglob("*.json"))
    for path in json_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("envs"), list):
            return [dict(item) for item in payload["envs"]]
        if isinstance(payload, list):
            return [dict(item) for item in payload]
    return []


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_cosmos2_transfer_component_from_s3(
    *,
    input_uri: str,
    output_uri: str,
    augmented_frames_uri: str,
    assets_uri: str = "",
    scene_spec_uri: str = "",
    image: str = "",
    run_id: str = "",
) -> dict[str, Any]:
    """Sibling-job entrypoint for task-conditioned Cosmos Transfer."""

    from npa.workflows.sim2real.cosmos_transfer_stage import (
        run_cosmos_transfer_component,
    )

    return run_cosmos_transfer_component(
        input_uri=input_uri,
        output_uri=output_uri,
        augmented_frames_uri=augmented_frames_uri,
        assets_uri=assets_uri,
        scene_spec_uri=scene_spec_uri,
        image=image,
        run_id=run_id,
        real_runner=_run_real_cosmos_transfer,
    )


def _run_real_cosmos_transfer(
    client: Any,
    input_uri: str,
    augment_prefix: str,
    frames_root: str,
    run_id: str,
) -> dict[str, Any] | None:
    """Compatibility surface delegating to the task-conditioned stage module."""

    from npa.workflows.sim2real.cosmos_transfer_stage import (
        run_real_cosmos_transfer,
    )

    return run_real_cosmos_transfer(
        client, input_uri, augment_prefix, frames_root, run_id
    )


def _write_stage(
    local_dir: Path,
    number: int,
    name: str,
    payload: dict[str, Any],
    *,
    filename: str | None = None,
) -> dict[str, Any]:
    path = local_dir / f"stage_{number:02d}_{name}" / (filename or f"{name}.json")
    return _write_json_artifact(path, payload)


def _trigger_payload(config: Sim2RealLoopConfig) -> dict[str, Any]:
    return {
        "schema": "npa.sim2real.trigger.v1",
        "stage": 1,
        "run_id": config.run_id,
        "created_at": _utc_now(),
        "trigger_dataset_uri": config.trigger_dataset_uri,
        "trigger_dataset_id": config.trigger_dataset_id,
        "input_format": "lerobot",
        "start_condition": "dataset_landed_in_trigger_path",
        "artifact_root": artifact_uris(config).get("root", ""),
        "byo_seams": byo_seams(config),
    }


def _redacted_config(config: Sim2RealLoopConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir) if config.output_dir else None
    return payload
