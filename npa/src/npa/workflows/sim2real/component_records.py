"""Component-tier evidence records for the canonical Sim2Real engine."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.constants import SIM_BACKEND_ISAAC
from npa.workflows.sim2real.models import (
    ComponentRecord,
    Sim2RealLoopConfig,
    Sim2RealLoopError,
)
from npa.workflows.sim2real.utils import _artifact_root_uri


def _placement_artifacts(invocation: dict[str, Any]) -> dict[str, Any]:
    provenance = dict(invocation.get("gpu_provenance") or {})
    selected = (
        provenance.get("selected_product") or provenance.get("selected_products") or ""
    )
    nodes = provenance.get("selected_node") or provenance.get("selected_nodes") or ""
    return {
        "job_name": str(provenance.get("job_name") or invocation.get("job_name") or ""),
        "gpu_candidate_order": provenance.get("candidate_order", []),
        "gpu_attempts": provenance.get("attempts", []),
        "selected_gpu_product": selected,
        "selected_gpu_node": nodes,
        "allocated_gpu": provenance.get("allocated_gpu", {}),
        "minimum_vram_gb": provenance.get("minimum_vram_gb", 0),
        "model_requirement": provenance.get("model_requirement", ""),
        "image_digests": provenance.get("image_digests", []),
        "duration_s": provenance.get("duration_s", ""),
    }


def _loop_component_records(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    outer_iteration: int,
    inner: dict[str, Any],
    heldout_report: dict[str, Any],
    decision: dict[str, Any],
) -> list[ComponentRecord]:
    """Build strict per-stage records for the real stages 7–11 path."""

    from npa.workflows.sim2real_stages import k8s_image_ready

    iterations = list(inner.get("iterations") or [])
    action_manifests = list(
        (local_dir / "actions" / "train" / f"outer-{outer_iteration:02d}").glob(
            "iter-*/rollout-*/manifest.json"
        )
    )
    rollout_sources: set[str] = set()
    rollout_scenario_digests: set[str] = set()
    rollout_invocations: list[dict[str, Any]] = []
    for manifest_path in action_manifests:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rollout_sources.add(str(payload.get("source") or ""))
        rollout_scenario_digests.add(str(payload.get("scenario_config_digest") or ""))
        if payload.get("component_invocation"):
            rollout_invocations.append(dict(payload["component_invocation"]))
    stage_07_works = bool(
        iterations
        and action_manifests
        and config.sim_backend == SIM_BACKEND_ISAAC
        and config.byo_policy_command.strip()
        and k8s_image_ready(config.policy_image)
        and k8s_image_ready(config.isaac_image)
        and rollout_sources == {"byo_isaac_policy_rollout"}
        and rollout_invocations
        and {str(invocation.get("mode") or "") for invocation in rollout_invocations}
        == {"kubernetes_job"}
        and all(
            (invocation.get("gpu_provenance") or {}).get("selected_product")
            and (invocation.get("gpu_provenance") or {}).get("image_digests")
            for invocation in rollout_invocations
        )
        and "" not in rollout_scenario_digests
        and len(action_manifests) >= config.rollout_count * config.inner_iterations
    )

    vlm_modes = {
        str(
            ((item.get("sample_vlm_eval") or {}).get("component_invocation") or {}).get(
                "mode"
            )
            or ""
        )
        for item in iterations
    }
    vlm_invocation = dict(
        (
            (iterations[-1].get("sample_vlm_eval") or {}).get("component_invocation")
            or {}
        )
        if iterations
        else {}
    )
    stage_08_works = bool(
        iterations
        and vlm_modes == {"kubernetes_job_dual_reason"}
        and vlm_invocation
        and (vlm_invocation.get("gpu_provenance") or {}).get("selected_products")
        and (vlm_invocation.get("gpu_provenance") or {}).get("image_digests")
        and k8s_image_ready(config.vlm_image)
    )

    trainer_backends = {
        str((item.get("update") or {}).get("backend") or "") for item in iterations
    }
    trainer_sources = {str(item.get("trainer_source") or "") for item in iterations}
    checkpoint_uris = [
        str((item.get("update") or {}).get("checkpoint_path") or "")
        for item in iterations
    ]
    trainer_scenario_proofs = [
        dict((item.get("update") or {}).get("applied_scenario_proof") or {})
        for item in iterations
    ]
    trainer_telemetry_uris = [
        str((item.get("update") or {}).get("ppo_telemetry_uri") or "")
        for item in iterations
    ]
    trainer_invocation = dict(
        iterations[-1].get("trainer_component_invocation") or {} if iterations else {}
    )
    stage_09_works = bool(
        iterations
        and trainer_sources == {"byo_command"}
        and trainer_backends == {"isaac_rsl_rl_ppo"}
        and all(
            uri.startswith("s3://") and uri.endswith(".pt") for uri in checkpoint_uris
        )
        and all(
            float(proof.get("coverage_rate") or 0.0) >= 0.90
            and int(proof.get("applied_unique_config_digests") or 0) > 0
            for proof in trainer_scenario_proofs
        )
        and all(uri.startswith("s3://") for uri in trainer_telemetry_uris)
        and bool(trainer_invocation.get("job_name"))
        and (trainer_invocation.get("gpu_provenance") or {}).get("selected_product")
        and (trainer_invocation.get("gpu_provenance") or {}).get("image_digests")
        and k8s_image_ready(config.trainer_image)
        and k8s_image_ready(config.isaac_image)
    )

    eval_invocation = dict(heldout_report.get("component_invocation") or {})
    heldout_image = str(heldout_report.get("heldout_backend_image") or "")
    eval_inference = dict(heldout_report.get("policy_inference_provenance") or {})
    eval_scenario_proof = dict(heldout_report.get("applied_scenario_proof") or {})
    eval_split = str(heldout_report.get("evaluation_split") or "")
    required_eval_count = (
        config.heldout_env_count
        if eval_split == "gold_heldout"
        else config.validation_env_count
        if eval_split == "validation"
        else 1
    )
    stage_10_works = bool(
        heldout_report.get("status") == "completed"
        and heldout_report.get("sim_backend") == SIM_BACKEND_ISAAC
        and heldout_report.get("deployable_policy_eval") is True
        and int(eval_invocation.get("returncode", 1)) == 0
        and eval_inference.get("loaded_for_inference") is True
        and eval_inference.get("stock_or_scripted_policy") is False
        and bool(heldout_report.get("policy_checkpoint_sha256"))
        and eval_scenario_proof.get("exact_digest_match") is True
        and len(heldout_report.get("per_env") or []) >= required_eval_count
        and (eval_invocation.get("gpu_provenance") or {}).get("selected_product")
        and (eval_invocation.get("gpu_provenance") or {}).get("image_digests")
        and k8s_image_ready(heldout_image)
    )

    root = _artifact_root_uri(config)
    from npa.workflows.sim2real.byo_isaac_trainer import k8s_job_name

    isaac_rollout_job = k8s_job_name(
        "s2r-byo-isaac-roll",
        config.run_id,
        f"outer-{outer_iteration:02d}-iter-{config.inner_iterations:02d}",
    )
    isaac_trainer_job = k8s_job_name("s2r-byo-isaac-train", config.run_id)
    isaac_eval_job = k8s_job_name(
        "s2r-byo-isaac-eval", config.run_id, f"outer-{outer_iteration:02d}"
    )
    rollout_placement = _placement_artifacts(
        rollout_invocations[-1] if rollout_invocations else {}
    )
    vlm_placement = _placement_artifacts(vlm_invocation)
    trainer_placement = _placement_artifacts(trainer_invocation)
    eval_placement = _placement_artifacts(eval_invocation)
    isaac_rollout_job = str(rollout_placement.pop("job_name", "")) or isaac_rollout_job
    vlm_job = str(vlm_placement.pop("job_name", "")) or "s2r-vlm-eval-reason{2,3}-*"
    isaac_trainer_job = str(trainer_placement.pop("job_name", "")) or isaac_trainer_job
    isaac_eval_job = str(eval_placement.pop("job_name", "")) or isaac_eval_job
    local_report_uri = str(heldout_report.get("report_uri") or "")
    try:
        report_relative = Path(local_report_uri).relative_to(local_dir).as_posix()
    except (TypeError, ValueError):
        report_relative = ""
    report_artifact_uri = (
        f"{root}/{report_relative}" if report_relative else local_report_uri
    )
    last_update = dict((iterations[-1].get("update") or {}) if iterations else {})
    last_calibration = dict(
        (iterations[-1].get("signal_calibration") or {}) if iterations else {}
    )
    records = [
        ComponentRecord(
            "stage_07_actions_train",
            "WORKS" if stage_07_works else "SEAM",
            (
                f"Real Isaac policy rollouts completed for outer-{outer_iteration:02d} "
                f"({config.rollout_count} rollouts × {config.inner_iterations} inner iterations)."
                if stage_07_works
                else "Policy rollouts did not prove the real Isaac Kubernetes contract."
            ),
            {
                "prefix": f"{root}/actions/train/outer-{outer_iteration:02d}/",
                "applied_scenario_config_digests": sorted(
                    digest for digest in rollout_scenario_digests if digest
                ),
                "applied_scenario_count": len(
                    {digest for digest in rollout_scenario_digests if digest}
                ),
                "job_name": isaac_rollout_job,
                "image": config.isaac_image,
                "gpu_request": {
                    "resource": config.k8s_gpu_resource,
                    "product": config.k8s_gpu_product,
                    "count": 1,
                },
                **rollout_placement,
            },
        ),
        ComponentRecord(
            "stage_08_vlm_eval_train",
            "WORKS" if stage_08_works else "SEAM",
            (
                "Dual Cosmos-Reason Kubernetes GPU critiques merged for train rollouts."
                if stage_08_works
                else "VLM evaluation did not prove the dual Kubernetes GPU contract."
            ),
            {
                "prefix": f"{root}/vlm_eval/train/outer-{outer_iteration:02d}/",
                "signal_calibration": last_calibration,
                "job_name": vlm_job,
                "image": config.vlm_image,
                "gpu_request": {
                    "resource": config.k8s_gpu_resource,
                    "product": config.k8s_gpu_product,
                    "count": 1,
                },
                **vlm_placement,
            },
        ),
        ComponentRecord(
            "stage_09_training_signal",
            "WORKS" if stage_09_works else "SEAM",
            (
                "VLM signals drove genuine Isaac RSL-RL PPO and produced resumable PyTorch checkpoints."
                if stage_09_works
                else "Trainer output did not prove the genuine Isaac RSL-RL PPO contract."
            ),
            {
                "prefix": f"{root}/training_signal/train/outer-{outer_iteration:02d}/",
                "checkpoint": str(inner.get("final_checkpoint_uri") or ""),
                "ppo_telemetry": str(last_update.get("ppo_telemetry_uri") or ""),
                "ppo_raw_log": str(last_update.get("ppo_raw_log_uri") or ""),
                "ppo_hyperparameters": dict(
                    last_update.get("ppo_hyperparameters") or {}
                ),
                "applied_scenarios": str(
                    last_update.get("applied_scenarios_uri") or ""
                ),
                "applied_scenario_proof": dict(
                    last_update.get("applied_scenario_proof") or {}
                ),
                "checkpoint_selection": str(
                    (inner.get("checkpoint_selection") or {}).get(
                        "selection_report_uri", ""
                    )
                ),
                "job_name": isaac_trainer_job,
                "image": config.isaac_image,
                "gpu_request": {
                    "resource": config.k8s_gpu_resource,
                    "product": config.k8s_gpu_product,
                    "count": 1,
                },
                **trainer_placement,
            },
        ),
        ComponentRecord(
            "stage_10_eval_heldout",
            "WORKS" if stage_10_works else "SEAM",
            (
                f"Real Isaac held-out rollout completed (success_rate={heldout_report.get('success_rate', 'n/a')})."
                if stage_10_works
                else "Held-out evaluation did not prove the real Isaac Kubernetes contract."
            ),
            {
                "report": report_artifact_uri,
                "evaluation_split": str(heldout_report.get("evaluation_split") or ""),
                "applied_scenario_proof": heldout_report.get(
                    "applied_scenario_proof", {}
                ),
                "checkpoint_sha256": heldout_report.get("policy_checkpoint_sha256", ""),
                "job_name": isaac_eval_job,
                "image": heldout_image,
                "gpu_request": {
                    "resource": config.k8s_gpu_resource,
                    "product": config.k8s_gpu_product,
                    "count": 1,
                },
                **eval_placement,
            },
        ),
        ComponentRecord(
            "stage_11_outer_loop",
            "WORKS",
            f"Persisted threshold decision: {decision.get('decision', 'unknown')}.",
            {
                "decision": f"{root}/outer_loop/decision.json",
                "job_name": config.run_id,
                "execution": "orchestrator_record",
                "duration_s": decision.get("duration_s", 0.0),
            },
        ),
    ]
    if os.environ.get("NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS", "").strip() == "1":
        failed = [record.name for record in records[:4] if record.tier != "WORKS"]
        if failed:
            raise Sim2RealLoopError(
                "real-tier component proof failed for: " + ", ".join(failed)
            )
    return records


def _expand_envgen_component_records(
    config: Sim2RealLoopConfig, components: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Ensure stages 4, 5, and 6 each have an explicit ComponentRecord."""

    expanded = list(components)
    names = {str(component.get("name") or "") for component in expanded}
    if {
        "stage_04_envs_raw",
        "stage_05_envs_train",
        "stage_06_tokens",
    }.issubset(names):
        return expanded
    grouped = next(
        (
            component
            for component in expanded
            if component.get("name") == "stage_04_06_env_gen_split_tokens"
        ),
        {},
    )
    expanded = [
        component
        for component in expanded
        if component.get("name") != "stage_04_06_env_gen_split_tokens"
    ]
    tier = str(grouped.get("tier") or "SEAM")
    evidence = str(
        grouped.get("evidence") or "Environment generation record unavailable."
    )
    artifacts = dict(grouped.get("artifacts") or {})
    root = _artifact_root_uri(config)
    stage_04_common = {
        # A resumed legacy state did not persist the attempt-specific hash. Do not
        # invent an exact name; fresh runs carry the invocation's real Job name.
        "job_name": str(artifacts.get("job_name") or "s2r-envgen-raw-shard-*"),
        "image": config.envgen_image,
        "gpu_request": {
            "resource": config.k8s_gpu_resource,
            "product": config.k8s_gpu_product,
            "count": 1,
        },
        "gpu_candidate_order": artifacts.get("gpu_candidate_order", []),
        "gpu_attempts": artifacts.get("gpu_attempts", []),
        "selected_gpu_product": artifacts.get("selected_gpu_product", ""),
        "selected_gpu_node": artifacts.get("selected_gpu_node", ""),
        "allocated_gpu": artifacts.get("allocated_gpu", {}),
        "image_digests": artifacts.get("image_digests", []),
        "duration_s": artifacts.get("duration_s", ""),
    }
    additions = [
        ComponentRecord(
            "stage_04_envs_raw",
            tier,
            evidence,
            {"raw_envs": f"{root}/envs/raw/", **stage_04_common},
        ),
        ComponentRecord(
            "stage_05_envs_train",
            tier,
            "Curated deterministic, disjoint, stratified train, validation, and gold-heldout records.",
            {
                "train_envs": artifacts.get(
                    "train_envs", f"{root}/envs/train/envs.jsonl"
                ),
                "validation_envs": artifacts.get(
                    "validation_envs", f"{root}/envs/validation/envs.jsonl"
                ),
                "heldout_envs": artifacts.get(
                    "heldout_envs", f"{root}/envs/heldout/envs.jsonl"
                ),
                "gold_heldout_envs": artifacts.get(
                    "gold_heldout_envs", f"{root}/envs/gold-heldout/envs.jsonl"
                ),
                "split_manifest": artifacts.get(
                    "split_manifest", f"{root}/envs/manifest/split-manifest.json"
                ),
                "curation_manifest": artifacts.get(
                    "curation_manifest",
                    f"{root}/envs/manifest/curation-manifest.json",
                ),
                "job_name": config.run_id,
                "execution": "orchestrator_record_from_stage_04_gpu_outputs",
                "upstream_job_name": stage_04_common["job_name"],
            },
        ),
        ComponentRecord(
            "stage_06_tokens",
            tier,
            (
                "Recorded scenario features as lineage/reporting inputs for state PPO; "
                "tokens and pixels are not policy observations."
            ),
            {
                "tokens": f"{root}/tokens/manifest.json",
                "learning_consumer": "lineage_and_reporting_only_for_state_ppo",
                "policy_observation_consumer": False,
                "rollout_consumer": "scenario_config_digest",
                "job_name": config.run_id,
                "execution": "orchestrator_record_from_stage_04_gpu_outputs",
                "upstream_job_name": stage_04_common["job_name"],
            },
        ),
    ]
    for component in additions:
        if component.name not in names:
            expanded.append(asdict(component))
    return expanded
