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
    selected = provenance.get("selected_product") or provenance.get("selected_products") or ""
    nodes = provenance.get("selected_node") or provenance.get("selected_nodes") or ""
    return {
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
    rollout_invocations: list[dict[str, Any]] = []
    for manifest_path in action_manifests:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rollout_sources.add(str(payload.get("source") or ""))
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
    stage_08_works = bool(
        iterations
        and vlm_modes == {"kubernetes_job_dual_reason"}
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
    stage_09_works = bool(
        iterations
        and trainer_sources == {"byo_command"}
        and trainer_backends == {"isaac_rsl_rl_ppo"}
        and all(uri.startswith("s3://") and uri.endswith(".pt") for uri in checkpoint_uris)
        and k8s_image_ready(config.trainer_image)
        and k8s_image_ready(config.isaac_image)
    )

    eval_invocation = dict(heldout_report.get("component_invocation") or {})
    heldout_image = str(heldout_report.get("heldout_backend_image") or "")
    stage_10_works = bool(
        heldout_report.get("status") == "completed"
        and heldout_report.get("sim_backend") == SIM_BACKEND_ISAAC
        and heldout_report.get("deployable_policy_eval") is True
        and int(eval_invocation.get("returncode", 1)) == 0
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
    vlm_invocation = dict(
        ((iterations[-1].get("sample_vlm_eval") or {}).get("component_invocation") or {})
        if iterations
        else {}
    )
    vlm_placement = _placement_artifacts(vlm_invocation)
    trainer_invocation = dict(
        iterations[-1].get("trainer_component_invocation") or {}
        if iterations
        else {}
    )
    trainer_placement = _placement_artifacts(trainer_invocation)
    eval_placement = _placement_artifacts(eval_invocation)
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
                "job_name": "s2r-vlm-eval-reason{2,3}-*",
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
                "report": f"{root}/eval/heldout/report.json",
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
    evidence = str(grouped.get("evidence") or "Environment generation record unavailable.")
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
            "Split the generated environment catalog into train and held-out records.",
            {
                "train_envs": artifacts.get("train_envs", f"{root}/envs/train/envs.jsonl"),
                "heldout_envs": artifacts.get(
                    "heldout_envs", f"{root}/envs/heldout/envs.jsonl"
                ),
                "job_name": config.run_id,
                "execution": "orchestrator_record_from_stage_04_gpu_outputs",
                "upstream_job_name": stage_04_common["job_name"],
            },
        ),
        ComponentRecord(
            "stage_06_tokens",
            tier,
            "Wrote the token manifest from the generated train/held-out catalog.",
            {
                "tokens": f"{root}/tokens/manifest.json",
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
