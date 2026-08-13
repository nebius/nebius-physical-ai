"""Retired controller orchestration compatibility; the canonical workflow never imports this module."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any


from npa.workflows.sim2real.artifact_upload import (
    _upload_final_report,
    upload_run_artifacts,  # noqa: F401 - public engine import surface
)
from npa.workflows.sim2real.config import artifact_uris, byo_seams
from npa.workflows.sim2real.component_records import (
    _expand_envgen_component_records,
    _loop_component_records,
    _persisted_loop_component_records,
)
from npa.workflows.sim2real.capture import runtime_parameter_metadata
from npa.workflows.sim2real.constants import (
    DEFAULT_VLM_SEAM_EVIDENCE,
    SCHEMA_E2E_REPORT,
    SIM_BACKEND_ISAAC,
)
from npa.serverless_common.env import require_isaac_eula_acceptance
from npa.workflows.sim2real.models import (
    ComponentRecord,
    Sim2RealLoopConfig,
    Sim2RealLoopError,
)
from npa.workflows.sim2real.decision import threshold_decision
from npa.workflows.sim2real.gpu_fallback import (
    gpu_fallback_report_contract,
)
from npa.workflows.sim2real.k8s_components import (
    _component_job_script as _component_job_script,
    _kubernetes_component_env as _kubernetes_component_env,
)
from npa.workflows.sim2real.reporting import build_progress_metrics
from npa.workflows.sim2real.policy_actions_stage import (
    run_policy_actions_component_from_s3 as run_policy_actions_component_from_s3,
)
from npa.workflows.sim2real.reference_helpers import (
    _signal_diversity_report as _signal_diversity_report,
    _signal_mean_reward as _signal_mean_reward,
    _write_env_manifest as _write_env_manifest,
    _write_train_heldout_split as _write_train_heldout_split,
)
from npa.workflows.sim2real.utils import (
    _artifact_root_uri,
    _bool_value,
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
    pass


_HELDOUT_DISTANCE_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)


def _compat_call(symbol: str, *args: Any, **kwargs: Any) -> Any:
    from npa.workflows.sim2real import engine

    return getattr(engine, symbol)(*args, **kwargs)


def _component_env(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_component_env", *args, **kwargs)


def _ensure_heldout_renders_for_viz(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_ensure_heldout_renders_for_viz", *args, **kwargs)


def _public_invocation(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_public_invocation", *args, **kwargs)


def _redacted_config(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_redacted_config", *args, **kwargs)


def _run_component_command(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_run_component_command", *args, **kwargs)


def _storage_client(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_storage_client", *args, **kwargs)


def _trigger_payload(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_trigger_payload", *args, **kwargs)


def _write_stage(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_write_stage", *args, **kwargs)


def run_heldout_eval(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("run_heldout_eval", *args, **kwargs)


def run_inner_loop(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("run_inner_loop", *args, **kwargs)


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

    if (
        config.sim_backend == SIM_BACKEND_ISAAC
        and not config.byo_eval_command
        and (config.k8s_context or config.k8s_kubeconfig)
    ):
        require_isaac_eula_acceptance(
            context=f"Sim2Real run {config.run_id}",
            resume_command=shlex.join([sys.executable, *sys.argv]),
        )
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
    source_sha = os.environ.get("NPA_SIM2REAL_SOURCE_SHA", "").strip()
    runtime_image = os.environ.get("NPA_SIM2REAL_RUNTIME_IMAGE", "").strip()
    if real_required:
        from npa.workflows.sim2real.job_scheduling import require_image_digest

        if not source_sha:
            raise Sim2RealLoopError(
                "NPA_SIM2REAL_SOURCE_SHA is required for a real durable controller"
            )
        require_image_digest(runtime_image)
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
        "source_sha": source_sha,
        "runtime_image": runtime_image,
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
    from npa.workflows.sim2real.resume_state import DurableStateStore, canonical_digest

    durable = DurableStateStore(config, local_dir)
    quality = float(inner["final_quality"])
    # Checkpoint selection consumes only validation. The gold split is not opened
    # until the final configured Stage 10 evaluation.
    selected_checkpoint_iteration = int(
        (inner.get("checkpoint_selection") or {}).get("training_iteration")
        or (inner.get("selected_validation_report") or {}).get(
            "checkpoint_training_iteration"
        )
        or 0
    )
    evaluation_split = (
        "validation"
        if outer_iteration < config.outer_iterations
        and inner.get("selected_validation_report")
        else "gold_heldout"
    )
    stage10_input = {
        "outer_iteration": outer_iteration,
        "evaluation_split": evaluation_split,
        "selected_checkpoint_iteration": selected_checkpoint_iteration,
        "selected_checkpoint_uri": inner.get("final_checkpoint_uri"),
        "inner_evidence_digest": canonical_digest(inner),
        "validation_envs_uri": getattr(config, "validation_envs_uri", ""),
        "gold_heldout_envs_uri": getattr(config, "gold_heldout_envs_uri", ""),
        "eval_image": config.eval_image,
        "isaac_image": config.isaac_image,
    }
    cached_stage10 = durable.load_unit(
        f"outer-o{outer_iteration:02d}-stage10-evaluation", stage10_input
    )
    if cached_stage10 is not None:
        heldout_report = dict(cached_stage10["heldout_report"])
    else:
        if evaluation_split == "validation":
            heldout_report = dict(inner["selected_validation_report"])
        else:
            heldout_report = run_heldout_eval(
                config,
                local_dir=local_dir,
                inner_evidence=inner,
                outer_iteration=outer_iteration,
                evaluation_split="gold_heldout",
                checkpoint_iteration=selected_checkpoint_iteration,
            )
        durable.commit_unit(
            f"outer-o{outer_iteration:02d}-stage10-evaluation",
            stage10_input,
            {"heldout_report": heldout_report},
        )
    stage11_input = {
        "outer_iteration": outer_iteration,
        "heldout_report_digest": canonical_digest(heldout_report),
        "threshold": config.threshold,
        "early_exit": getattr(config, "early_exit", True),
    }
    cached_stage11 = durable.load_unit(
        f"outer-o{outer_iteration:02d}-stage11-decision", stage11_input
    )
    if cached_stage11 is not None:
        decision = dict(cached_stage11["decision"])
    else:
        decision = threshold_decision(
            config,
            local_dir=local_dir,
            heldout_report=heldout_report,
            outer_iteration=outer_iteration,
        )
        durable.commit_unit(
            f"outer-o{outer_iteration:02d}-stage11-decision",
            stage11_input,
            {"decision": decision},
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

    from npa.workflows.sim2real.resume_state import DurableStateStore, canonical_digest

    durable = DurableStateStore(config, local_dir)
    finalize_input = {
        "final_decision": final_decision,
        "final_checkpoint": final_decision.get("checkpoint_uri"),
        "final_eval_digest": canonical_digest(final_eval),
        "outer_history_digest": canonical_digest(outer_history),
        "component_digest": canonical_digest(components),
    }
    completed_finalize = durable.load_unit("finalize-complete", finalize_input)
    if completed_finalize is not None:
        return dict(completed_finalize["report"])

    components = _expand_envgen_component_records(config, components)
    loop_components = _persisted_loop_component_records(config, components)
    if loop_components is None:
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
    stage14_input = {
        **finalize_input,
        "stage_components_digest": canonical_digest(components),
        "rerun_enabled": config.rerun_enabled,
        "mcap_enabled": os.environ.get("NPA_SIM2REAL_MCAP", ""),
    }
    cached_viz = durable.load_unit("finalize-stage-14", stage14_input)
    if cached_viz is not None:
        for key, filename in (
            ("rrd_uri", "sim2real.rrd"),
            ("mcap_uri", "sim2real.mcap"),
        ):
            uri = str(cached_viz.get(key) or "")
            if uri:
                _storage_client(config).download_file(
                    uri, str(local_dir / "reports" / filename)
                )
        viz_component = ComponentRecord(**dict(cached_viz["component"]))
        viz_info = dict(cached_viz["viz_info"])
    else:
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
        viz_component.artifacts["duration_s"] = round(
            time.monotonic() - stage_started, 3
        )
        viz_payload: dict[str, Any] = {
            "component": asdict(viz_component),
            "viz_info": viz_info,
            "rrd_uri": "",
            "mcap_uri": "",
        }
        if durable.enabled:
            for key, filename in (
                ("rrd_uri", "sim2real.rrd"),
                ("mcap_uri", "sim2real.mcap"),
            ):
                path = local_dir / "reports" / filename
                if path.is_file() and path.stat().st_size > 0:
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    viz_payload[key] = _storage_client(config).upload_file(
                        str(path),
                        f"{durable.root}/finalization-artifacts/sha256-{digest}-{filename}",
                    )
        durable.commit_unit("finalize-stage-14", stage14_input, viz_payload)
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
    durable_candidate = final_decision.get("candidate")
    if not candidate_path.is_file() and isinstance(durable_candidate, dict):
        _write_json_artifact(candidate_path, durable_candidate)
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
    durable.commit_unit("finalize-complete", finalize_input, {"report": report})
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

    if os.environ.get("NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS", "").strip() == "1":
        if (heldout_report or {}).get("evaluation_split") != "gold_heldout":
            raise Sim2RealLoopError(
                "Stage 14 real-tier footage must come from the exact gold evaluation"
            )
    heldout_report = _ensure_heldout_renders_for_viz(
        config,
        local_dir,
        heldout_report,
    )
    if os.environ.get("NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS", "").strip() == "1":
        lineage = dict((heldout_report or {}).get("render_lineage") or {})
        expected_checkpoint = str((final_decision or {}).get("checkpoint_uri") or "")
        if (
            lineage.get("evaluation_split") != "gold_heldout"
            or not lineage.get("render_manifest_sha256")
            or lineage.get("checkpoint_uri") != expected_checkpoint
        ):
            raise Sim2RealLoopError(
                "gold render lineage does not match the final selected checkpoint"
            )
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


__all__ = [
    "_append_outer_iteration_workflow_state",
    "_run_byo_rerun_command",
    "_run_sim2real_viz_stage",
    "_signal_training_imports",
    "run_finalize",
    "run_preamble",
    "run_single_outer_iteration",
]
