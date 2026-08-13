"""Retired controller held-out evaluation contracts and normalization."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any


from npa.clients.storage import StorageClient
from npa.workflows.sim2real.artifact_upload import (
    upload_run_artifacts,  # noqa: F401 - public engine import surface
)
from npa.workbench.cosmos.reason import (
    CosmosReasonError,
    resolve_cosmos_reason_model_id,
    run_cosmos_reason_vlm,
)
from npa.workflows.sim2real.constants import (
    DEFAULT_ISAAC_TASK,
    DEFAULT_REFERENCE_VLM_MODEL,
    DEFAULT_SIM_BACKEND,
    DEFAULT_THRESHOLD,
    SCHEMA_HELDOUT_REPORT,
    SIM_BACKEND_GENESIS,
    SIM_BACKEND_ISAAC,
    SIM_BACKENDS,
)
from npa.workflows.sim2real.models import (
    Sim2RealLoopConfig,
    Sim2RealLoopError,
)
from npa.workflows.sim2real.k8s_components import (
    _component_job_script as _component_job_script,
    _kubernetes_component_env as _kubernetes_component_env,
)
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
    _utc_now,
    _write_json_artifact,
)
from npa.workflows.sim2real.workflow_state_io import (
    _workflow_state_path,  # noqa: F401 - legacy engine import surface
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


def _apply_reference_adapter_heldout_gate(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_apply_reference_adapter_heldout_gate", *args, **kwargs)


def _build_heldout_render_manifest(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_build_heldout_render_manifest", *args, **kwargs)


def _byo_robot_env(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_byo_robot_env", *args, **kwargs)


def _component_attempt_id(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_component_attempt_id", *args, **kwargs)


def _component_env(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_component_env", *args, **kwargs)


def _component_output_uri(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_component_output_uri", *args, **kwargs)


def _download_s3_env_records(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_download_s3_env_records", *args, **kwargs)


def _find_component_input_file(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_find_component_input_file", *args, **kwargs)


def _normalized_s3_prefix(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_normalized_s3_prefix", *args, **kwargs)


def _public_invocation(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_public_invocation", *args, **kwargs)


def _read_component_env_records(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_read_component_env_records", *args, **kwargs)


def _read_component_json(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_read_component_json", *args, **kwargs)


def _redact_command(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_redact_command", *args, **kwargs)


def _reference_heldout_payload(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_reference_heldout_payload", *args, **kwargs)


def _resolve_env_records_s3_uri(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_resolve_env_records_s3_uri", *args, **kwargs)


def _rollout_image_paths(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_rollout_image_paths", *args, **kwargs)


def _run_component_command(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_run_component_command", *args, **kwargs)


def _run_genesis_heldout_rollouts(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_run_genesis_heldout_rollouts", *args, **kwargs)


def _run_image_component(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_run_image_component", *args, **kwargs)


def _run_isaac_heldout_rollouts(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_run_isaac_heldout_rollouts", *args, **kwargs)


def _storage_client(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_storage_client", *args, **kwargs)


def _task_description_from_manifest(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_task_description_from_manifest", *args, **kwargs)


def _upload_component_directory(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_upload_component_directory", *args, **kwargs)


def _upload_component_file(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_upload_component_file", *args, **kwargs)


def _write_stage(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_write_stage", *args, **kwargs)


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
    configured_split_envs_uri = (
        config.validation_envs_uri
        if evaluation_split == "validation"
        else (config.gold_heldout_envs_uri or config.heldout_envs_uri)
    )
    scenario_records_uri = (
        _resolve_env_records_s3_uri(_normalized_s3_prefix(configured_split_envs_uri))
        if configured_split_envs_uri
        else ""
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
    if scenario_records_uri:
        # The exact split object is the source of truth across controller Pod
        # replacement. The local directory is merely a disposable cache.
        extra["NPA_SIM2REAL_HELDOUT_ENVS_URI"] = scenario_records_uri
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
        if (
            config.s3_bucket.strip()
            and os.environ.get("NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS", "").strip()
            == "1"
            and not scenario_records_uri
        ):
            raise Sim2RealLoopError(
                f"real {evaluation_split} evaluation requires an exact durable "
                "scenario-records S3 URI"
            )
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
        if scenario_records_uri:
            heldout_envs_uri = scenario_records_uri
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
            scenario_records_uri = heldout_envs_uri
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
    report["scenario_records_uri"] = scenario_records_uri
    report["scenario_records_source"] = (
        "durable_s3_object" if scenario_records_uri else "local_ephemeral"
    )
    report["checkpoint_training_iteration"] = checkpoint_iteration
    report["gold_heldout_untouched"] = evaluation_split == "validation"
    rendered_report = _ensure_heldout_renders_for_viz(
        config,
        local_dir,
        report,
        invocation=invocation,
        payload=payload,
        renders_dir=output_dir / "renders",
    )
    if rendered_report is None:
        raise Sim2RealLoopError("held-out render lineage unexpectedly lost its report")
    report = rendered_report
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
        # Canonical Isaac evaluator evidence emitted by the Stage 10 sibling.
        # Stage 11 still owns the threshold decision; Stage 14 still owns RRD/MCAP.
        "isaac_eval_summary",
        "isaac_eval_summary_uri",
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
    lineage = dict(report.get("render_lineage") or {})
    if renders_dir is None:
        relative = str(lineage.get("local_relative_dir") or "").strip()
        if relative:
            renders_dir = local_dir / relative
        elif report.get("evaluation_split") == "gold_heldout":
            outer = int(report.get("outer_iteration") or config.outer_iterations)
            renders_dir = (
                local_dir / "eval" / "gold-heldout" / f"outer-{outer:02d}" / "renders"
            )
        else:
            renders_dir = local_dir / "eval" / "heldout" / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    report["local_renders_dir"] = str(renders_dir)

    output_uri = str((invocation or {}).get("output_uri") or "").strip()
    if output_uri:
        lineage = {
            "schema": "npa.sim2real.render_lineage.v1",
            "evaluation_split": report.get("evaluation_split"),
            "outer_iteration": report.get("outer_iteration"),
            "checkpoint_training_iteration": report.get(
                "checkpoint_training_iteration"
            ),
            "checkpoint_uri": report.get("policy_checkpoint")
            or (report.get("policy_inference_provenance") or {}).get("checkpoint_uri"),
            "local_relative_dir": str(renders_dir.relative_to(local_dir)),
            "renders_s3_uri": _sibling_uri(output_uri, "renders/"),
            "manifest_s3_uri": _sibling_uri(output_uri, "render-manifest.json"),
        }
    elif report.get("render_manifest"):
        render_manifest = dict(report.get("render_manifest") or {})
        lineage = {
            **lineage,
            "schema": "npa.sim2real.render_lineage.v1",
            "evaluation_split": report.get("evaluation_split"),
            "outer_iteration": report.get("outer_iteration"),
            "checkpoint_training_iteration": report.get(
                "checkpoint_training_iteration"
            ),
            "checkpoint_uri": report.get("policy_checkpoint")
            or (report.get("policy_inference_provenance") or {}).get("checkpoint_uri"),
            "local_relative_dir": str(renders_dir.relative_to(local_dir)),
            "renders_s3_uri": render_manifest.get("renders_s3_uri") or "",
        }
    report["render_lineage"] = lineage

    if config.s3_bucket.strip():
        from npa.clients.storage import StorageError

        client = _storage_client(config)
        renders_uri = str(lineage.get("renders_s3_uri") or "").strip()
        manifest_uri = str(lineage.get("manifest_s3_uri") or "").strip()
        if renders_uri:
            try:
                client.download_directory(renders_uri, str(renders_dir))
            except (StorageError, OSError):
                pass
        if manifest_uri:
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
    manifest = dict(report.get("render_manifest") or {})
    if manifest:
        from npa.workflows.sim2real.resume_state import canonical_digest

        lineage["render_manifest_sha256"] = canonical_digest(manifest)
        lineage["capture"] = manifest.get("capture") or report.get("capture") or {}
        report["render_lineage"] = lineage
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


__all__ = [
    "_component_heldout_payload",
    "_component_vlm_payload",
    "_consume_stage_assets",
    "_ensure_heldout_renders_for_viz",
    "_has_heldout_camera_pngs",
    "_heldout_k8s_image_ready",
    "_heldout_success_summary",
    "_normalize_heldout_report",
    "_resolve_heldout_robot",
    "_resolve_heldout_scene",
    "_resolve_isaac_scene",
    "_sibling_uri",
    "run_heldout_eval",
    "run_heldout_eval_component_from_s3",
    "run_vlm_eval_component_from_s3",
]
