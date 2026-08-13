"""Retired controller component and training compatibility adapters."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any


from npa.clients.storage import StorageClient
from npa.workflows.sim2real.artifact_upload import (
    upload_run_artifacts,  # noqa: F401 - public engine import surface
)
from npa.workbench.cosmos.reason import (
    merge_dual_reason_evaluations,
)
from npa.workflows.sim2real.constants import (
    CORRECTIVE_TARGETS,
    ERROR_SEVERITY,
    SCHEMA_HELDOUT_REPORT,
    SCHEMA_RL_SIGNAL,
    SCHEMA_VLM_EVAL,
)
from npa.workflows.sim2real.models import (
    Sim2RealLoopConfig,
    Sim2RealLoopError,
)
from npa.workflows.sim2real.gpu_fallback import (
    GpuCapacityExhausted,
    GpuJobFailure,
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
from npa.workflows.sim2real.policy_actions_stage import (
    run_policy_actions_component_from_s3 as run_policy_actions_component_from_s3,
)
from npa.workflows.sim2real.reference_helpers import (
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
    from npa.workbench.lerobot.policy_container import VlmSignalUpdateResult


_HELDOUT_DISTANCE_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)


def _compat_call(symbol: str, *args: Any, **kwargs: Any) -> Any:
    from npa.workflows.sim2real import engine

    return getattr(engine, symbol)(*args, **kwargs)


def _signal_training_imports(*args: Any, **kwargs: Any) -> Any:
    return _compat_call("_signal_training_imports", *args, **kwargs)


def run_inner_loop(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    initial_quality: float,
    outer_iteration: int = 1,
    resume_checkpoint_uri: str = "",
) -> dict[str, Any]:
    """Delegate durable Stage 7–9 execution to the review-sized executor."""

    from npa.workflows.sim2real.stage_execution import run_inner_loop as execute

    return execute(
        config,
        local_dir=local_dir,
        initial_quality=initial_quality,
        outer_iteration=outer_iteration,
        resume_checkpoint_uri=resume_checkpoint_uri,
    )


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
            "NPA_SIM2REAL_ISAAC_CACHE_PVC": config.k8s_isaac_cache_pvc,
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


def _pod_info_from_snapshot(snapshot: Any) -> dict[str, Any]:
    """Serialize typed Pod fields used by ComponentRecord provenance."""

    pods = list(snapshot.pods)
    if not pods:
        return {}
    pod = pods[0]
    return {
        "name": pod.name,
        "uid": pod.uid,
        "owner_uid": pod.owner_uid,
        "node_name": pod.node_name,
        "phase": pod.phase,
        "deletion_timestamp": pod.deletion_timestamp,
        "scheduled_status": pod.scheduled_status,
        "scheduled_reason": pod.scheduled_reason,
        "resources": {"requests": pod.resource_requests},
        "container_statuses": [
            {
                "name": status.name,
                "image": status.image,
                "image_id": status.image_id,
                "restart_count": status.restart_count,
                "waiting_reason": status.waiting_reason,
                "terminated_reason": status.terminated_reason,
                "exit_code": status.exit_code,
                "signal": status.signal,
            }
            for status in pod.containers
        ],
        "image_digests": snapshot.image_digests,
        "kueue": snapshot.kueue.__dict__,
    }


def _refresh_registry_pull_secret_for_sibling_job(
    image: str,
    *,
    config: Sim2RealLoopConfig,
    namespace: str,
) -> None:
    """Compatibility-only refresh before an archived sibling Job apply.

    Pre-standard-runtime runs refreshed during the retired ``k8s_submit`` path,
    but later sibling Jobs could outlive IAM registry tokens. The standard
    compositional workflow does not import or call this helper.
    """

    if _bool_value(os.environ.get("NPA_SIM2REAL_SKIP_REGISTRY_REFRESH", "0")):
        return

    from npa.workflows.sim2real.registry_auth import (
        ensure_registry_pull_secret_for_images,
    )

    try:
        ensure_registry_pull_secret_for_images(
            image,
            namespace=namespace,
            kubeconfig=config.k8s_kubeconfig,
            k8s_context=config.k8s_context,
        )
    except Exception as exc:
        if os.environ.get("NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS", "").strip() == "1":
            raise Sim2RealLoopError(
                f"could not refresh the registry pull secret for {image}"
            ) from exc
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
    base_job_name = _k8s_job_name(
        config.run_id,
        component,
        identity=env.get("NPA_SIM2REAL_OUTPUT_URI", ""),
    )
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

    job_workload = workload_kind(component, sim_backend=config.sim_backend)
    model_hint = env.get("NPA_SIM2REAL_VLM_MODEL", "")
    from npa.workflows.sim2real.k8s_client import KubernetesJobClient

    client = KubernetesJobClient.from_environment(
        namespace=namespace,
        kubeconfig=config.k8s_kubeconfig,
        context=config.k8s_context,
    )
    try:
        gpu_provenance = run_gpu_job_with_fallback(
            client=client,
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
    job_uid = str(gpu_provenance.get("job_uid") or "")
    snapshot = client.snapshot(job_name, namespace=namespace)
    pod_info = _pod_info_from_snapshot(snapshot)
    logs = client.pod_logs(snapshot, tail_lines=10000)
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
        "stdout_excerpt": _component_excerpt(logs),
        "stderr_excerpt": "",
        "cleanup": "preserved_for_durable_reconciliation",
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
    base_job_name = _k8s_job_name(
        config.run_id,
        component,
        identity=output_uri or env.get("NPA_SIM2REAL_OUTPUT_URI", ""),
    )
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

    job_workload = workload_kind(component, sim_backend=config.sim_backend)
    model_hint = env.get("NPA_SIM2REAL_VLM_MODEL", "")
    from npa.workflows.sim2real.k8s_client import KubernetesJobClient

    client = KubernetesJobClient.from_environment(
        namespace=namespace,
        kubeconfig=config.k8s_kubeconfig,
        context=config.k8s_context,
    )
    try:
        gpu_provenance = run_gpu_job_with_fallback(
            client=client,
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
    job_uid = str(gpu_provenance.get("job_uid") or "")
    snapshot = client.snapshot(job_name, namespace=namespace)
    pod_info = _pod_info_from_snapshot(snapshot)
    logs = client.pod_logs(snapshot, tail_lines=10000)
    try:
        _download_component_output(config, output_uri, output_json)
    except Sim2RealLoopError as exc:
        log_excerpt = _component_excerpt(logs)
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
        "stdout_excerpt": _component_excerpt(logs),
        "stderr_excerpt": "",
        "cleanup": "preserved_for_durable_reconciliation",
    }


def _component_attempt_id(
    config: Sim2RealLoopConfig, component: str, label: str
) -> str:
    """Return the durable identity for one logical component execution.

    A restarted exact-SHA controller must address the same immutable component
    input/output prefix. Random attempt suffixes made partially completed Stage 8
    work undiscoverable and caused duplicate GPU Jobs after driver replacement.
    """

    digest = hashlib.sha256(
        f"{config.run_id}:{component}:{label}".encode("utf-8")
    ).hexdigest()
    return f"{_safe_slug(component)}-{digest[:24]}"


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


def _k8s_job_name(run_id: str, component: str, *, identity: str = "") -> str:
    """Return a stable Job base name suitable for create-or-adopt reconciliation."""

    run_part = _safe_slug(run_id)[:18] or "run"
    component_part = _safe_slug(component)[:16] or "component"
    suffix = hashlib.sha256(
        f"{run_id}:{component}:{identity}".encode("utf-8")
    ).hexdigest()[:12]
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


__all__ = [
    "_apply_reference_adapter_heldout_gate",
    "_byo_robot_env",
    "_component_attempt_id",
    "_component_env",
    "_component_excerpt",
    "_component_io_prefix",
    "_component_output_uri",
    "_config_from_workflow_state",
    "_convert_eval_to_signal",
    "_download_component_output",
    "_effective_k8s_parallelism",
    "_evaluate_reason_rollout_k8s",
    "_inner_loop_progress_score",
    "_k8s_job_name",
    "_normalize_byo_rl_signal",
    "_normalize_vlm_evaluation",
    "_normalized_s3_prefix",
    "_pod_info_from_snapshot",
    "_public_invocation",
    "_read_component_json",
    "_redact_command",
    "_reference_adapter_env_score",
    "_reference_heldout_payload",
    "_reference_vlm_payload_from_rollout",
    "_refresh_registry_pull_secret_for_sibling_job",
    "_run_component_command",
    "_run_image_component",
    "_run_kubernetes_image_component",
    "_run_kubernetes_indexed_image_component",
    "_run_policy_rollouts_via_command",
    "_run_trainer_via_command",
    "_safe_slug",
    "_storage_client",
    "_upload_component_directory",
    "_upload_component_file",
    "convert_vlm_eval_to_rl_signal",
    "evaluate_rollout_with_vlm",
    "generate_action_rollouts",
    "run_cosmos2_transfer_component",
    "run_envgen_sharded_component",
    "run_inner_loop",
    "run_policy_rollout_component",
    "signal_mapping_rules",
]
