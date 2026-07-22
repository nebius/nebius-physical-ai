"""Build Sim2Real runtime config and artifact URI maps."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from npa.deploy.images import registry_from_env
from npa.workflows.sim2real.constants import (
    DEFAULT_ACTION_ENV_LIMIT,
    DEFAULT_ENVGEN_SHARD_COUNT,
    DEFAULT_K8S_MAX_PARALLEL_GPUS,
    DEFAULT_INNER_ITERATIONS,
    DEFAULT_ISAAC_TASK,
    DEFAULT_LEROBOT_DATASET_ID,
    DEFAULT_LOOP_OF_LOOPS_ITERATIONS,
    DEFAULT_OUTER_ITERATIONS,
    DEFAULT_PREFIX,
    DEFAULT_REFERENCE_VLM_MODEL,
    DEFAULT_REASON2_MODEL,
    DEFAULT_REASON3_MODEL,
    DEFAULT_ROLLOUT_COUNT,
    DEFAULT_S3_ENDPOINT,
    DEFAULT_SIM_BACKEND,
    DEFAULT_STEPS_PER_ROLLOUT,
    DEFAULT_THRESHOLD,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_HELDOUT_ENVS,
)
from npa.workflows.sim2real.models import (
    Sim2RealLoopConfig,
    default_augment_image,
    default_envgen_image,
    default_eval_image,
    default_isaac_image,
    default_policy_image,
    default_trainer_image,
    default_vlm_image,
    new_run_id,
)
from npa.workflows.sim2real.utils import (
    _artifact_root_uri,
    _bool_value,
    _serviceaccount_namespace,
    _split_csv,
)

def build_config_from_env(**overrides: Any) -> Sim2RealLoopConfig:
    """Build a Sim2Real loop config from explicit values and env fallbacks."""

    run_id = str(
        overrides.get("run_id") or os.environ.get("NPA_SIM2REAL_RUN_ID") or new_run_id()
    )
    if "s3_bucket" in overrides:
        bucket = str(overrides.get("s3_bucket") or "")
    else:
        bucket = str(
            os.environ.get("NPA_SIM2REAL_BUCKET")
            or os.environ.get("NPA_S3_BUCKET")
            or os.environ.get("S3_BUCKET")
            or ""
        )
    registry = registry_from_env()
    if "s3_prefix" in overrides and overrides.get("s3_prefix") is not None:
        s3_prefix = str(overrides["s3_prefix"])
    elif "NPA_SIM2REAL_PREFIX" in os.environ:
        s3_prefix = os.environ.get("NPA_SIM2REAL_PREFIX", "")
    else:
        s3_prefix = DEFAULT_PREFIX
    action_rollouts_uri = str(
        overrides.get("action_rollouts_uri")
        or os.environ.get("ACTION_ROLLOUTS_URI")
        or (f"s3://{bucket}/sim2real-a-*/actions/train/" if bucket else "")
    )
    return Sim2RealLoopConfig(
        run_id=run_id,
        output_dir=Path(overrides["output_dir"])
        if overrides.get("output_dir")
        else None,
        s3_bucket=bucket,
        s3_prefix=s3_prefix,
        s3_endpoint=str(
            overrides.get("s3_endpoint")
            or os.environ.get("AWS_ENDPOINT_URL")
            or os.environ.get("S3_ENDPOINT_URL")
            or DEFAULT_S3_ENDPOINT
        ),
        trigger_dataset_uri=str(
            overrides.get("trigger_dataset_uri")
            or os.environ.get("NPA_SIM2REAL_TRIGGER_DATASET_URI")
            or os.environ.get("TRIGGER_DATASET_URI")
            or (f"s3://{bucket}/sim2real-triggers/{run_id}/" if bucket else "")
        ),
        trigger_dataset_id=str(
            overrides.get("trigger_dataset_id")
            or os.environ.get("NPA_SIM2REAL_TRIGGER_DATASET_ID")
            or os.environ.get("TRIGGER_DATASET_ID")
            or DEFAULT_LEROBOT_DATASET_ID
        ),
        action_rollouts_uri=action_rollouts_uri,
        train_envs_uri=str(
            overrides.get("train_envs_uri") or os.environ.get("TRAIN_ENVS_URI") or ""
        ),
        heldout_envs_uri=str(
            overrides.get("heldout_envs_uri")
            or os.environ.get("HELDOUT_ENVS_URI")
            or ""
        ),
        assets_uri=str(
            overrides.get("assets_uri") or os.environ.get("ASSETS_URI") or ""
        ),
        scene_spec_uri=str(
            overrides.get("scene_spec_uri") or os.environ.get("SCENE_SPEC_URI") or ""
        ),
        cameras_uri=str(
            overrides.get("cameras_uri")
            or os.environ.get("NPA_SIM2REAL_CAMERAS_URI")
            or os.environ.get("CAMERAS_URI")
            or ""
        ),
        robot_spec_uri=str(
            overrides.get("robot_spec_uri")
            or os.environ.get("ROBOT_SPEC_URI")
            or os.environ.get("NPA_SIM2REAL_ROBOT_SPEC_URI")
            or ""
        ),
        robot_source=str(
            overrides.get("robot_source")
            or os.environ.get("ROBOT_SOURCE")
            or os.environ.get("NPA_SIM2REAL_ROBOT_SOURCE")
            or ""
        ).strip().lower(),
        robot_preset=str(
            overrides.get("robot_preset")
            or os.environ.get("ROBOT_PRESET")
            or os.environ.get("NPA_SIM2REAL_ROBOT_PRESET")
            or ""
        ).strip().lower(),
        augment_image=str(
            overrides.get("augment_image")
            or os.environ.get("AUGMENT_IMAGE")
            or default_augment_image(registry=registry or None)
        ),
        envgen_image=str(
            overrides.get("envgen_image")
            or os.environ.get("ENVGEN_IMAGE")
            or default_envgen_image(registry=registry or None)
        ),
        env_count=int(
            overrides.get("env_count", os.environ.get("NPA_ENV_COUNT", "0"))
        ),
        train_fraction=float(
            overrides.get(
                "train_fraction",
                os.environ.get("NPA_TRAIN_FRACTION", DEFAULT_TRAIN_FRACTION),
            )
        ),
        envgen_shard_count=int(
            overrides.get(
                "envgen_shard_count",
                os.environ.get("NPA_ENVGEN_SHARD_COUNT", DEFAULT_ENVGEN_SHARD_COUNT),
            )
        ),
        action_env_limit=int(
            overrides.get(
                "action_env_limit",
                os.environ.get("NPA_ACTION_ENV_LIMIT", DEFAULT_ACTION_ENV_LIMIT),
            )
        ),
        policy_image=str(
            overrides.get("policy_image")
            or os.environ.get("POLICY_IMAGE")
            or default_policy_image(registry=registry or None)
        ),
        trainer_image=str(
            overrides.get("trainer_image")
            or os.environ.get("TRAINER_IMAGE")
            or default_trainer_image(registry=registry or None)
        ),
        vlm_image=str(
            overrides.get("vlm_image")
            or os.environ.get("VLM_IMAGE")
            or default_vlm_image(registry=registry or None)
        ),
        vlm_reason2_image=str(
            overrides.get("vlm_reason2_image")
            or os.environ.get("VLM_REASON2_IMAGE")
            or os.environ.get("VLM_IMAGE")
            or default_vlm_image(registry=registry or None)
        ),
        vlm_reason3_image=str(
            overrides.get("vlm_reason3_image")
            or os.environ.get("VLM_REASON3_IMAGE")
            or os.environ.get("VLM_IMAGE")
            or default_vlm_image(registry=registry or None)
        ),
        eval_image=str(
            overrides.get("eval_image")
            or os.environ.get("EVAL_IMAGE")
            or default_eval_image(registry=registry or None)
        ),
        isaac_image=str(
            overrides.get("isaac_image")
            or os.environ.get("ISAAC_IMAGE")
            or default_isaac_image(registry=registry or None)
        ),
        sim_backend=str(
            overrides.get("sim_backend")
            or os.environ.get("NPA_SIM2REAL_SIM_BACKEND")
            or DEFAULT_SIM_BACKEND
        ).strip().lower(),
        isaac_task=str(
            overrides.get("isaac_task")
            or os.environ.get("NPA_SIM2REAL_ISAAC_TASK")
            or DEFAULT_ISAAC_TASK
        ),
        vlm_model=str(
            overrides.get("vlm_model")
            or os.environ.get("VLM_MODEL")
            or DEFAULT_REFERENCE_VLM_MODEL
        ),
        vlm_reason2_model=str(
            overrides.get("vlm_reason2_model")
            or os.environ.get("VLM_REASON2_MODEL")
            or os.environ.get("VLM_MODEL")
            or DEFAULT_REASON2_MODEL
        ),
        vlm_reason3_model=str(
            overrides.get("vlm_reason3_model")
            or os.environ.get("VLM_REASON3_MODEL")
            or os.environ.get("NPA_COSMOS_REASON3_MODEL_ID")
            or DEFAULT_REASON3_MODEL
        ),
        vlm_dual_reason=_bool_value(
            overrides.get(
                "vlm_dual_reason",
                os.environ.get("NPA_SIM2REAL_VLM_DUAL_REASON", "1"),
            )
        ),
        threshold=float(
            overrides.get(
                "threshold", os.environ.get("SUCCESS_THRESHOLD", DEFAULT_THRESHOLD)
            )
        ),
        inner_iterations=int(
            overrides.get(
                "inner_iterations",
                os.environ.get("INNER_ITERATIONS", DEFAULT_INNER_ITERATIONS),
            )
        ),
        outer_iterations=int(
            overrides.get(
                "outer_iterations",
                os.environ.get("OUTER_ITERATIONS", DEFAULT_OUTER_ITERATIONS),
            )
        ),
        loop_of_loops_iterations=int(
            overrides.get(
                "loop_of_loops_iterations",
                os.environ.get(
                    "LOOP_OF_LOOPS_ITERATIONS", DEFAULT_LOOP_OF_LOOPS_ITERATIONS
                ),
            )
        ),
        rollout_count=int(
            overrides.get(
                "rollout_count", os.environ.get("ROLLOUT_COUNT", DEFAULT_ROLLOUT_COUNT)
            )
        ),
        steps_per_rollout=int(
            overrides.get(
                "steps_per_rollout",
                os.environ.get("STEPS_PER_ROLLOUT", DEFAULT_STEPS_PER_ROLLOUT),
            )
        ),
        heldout_env_count=int(
            overrides.get(
                "heldout_env_count",
                os.environ.get("HELDOUT_ENV_COUNT", DEFAULT_HELDOUT_ENVS),
            )
        ),
        seed=int(overrides.get("seed", os.environ.get("SEED", "42"))),
        upload_artifacts=_bool_value(
            overrides.get("upload_artifacts", os.environ.get("UPLOAD_ARTIFACTS", "0"))
        ),
        no_guardrails=_bool_value(
            overrides.get("no_guardrails", os.environ.get("NO_GUARDRAILS", "0"))
        ),
        signal_loss_weight=float(
            overrides.get(
                "signal_loss_weight", os.environ.get("SIGNAL_LOSS_WEIGHT", "1.0")
            )
        ),
        learning_rate=float(
            overrides.get("learning_rate", os.environ.get("LEARNING_RATE", "0.05"))
        ),
        byo_signal_converter=str(
            overrides.get("byo_signal_converter")
            or os.environ.get("BYO_SIGNAL_CONVERTER")
            or ""
        ),
        byo_trainer_command=str(
            overrides.get("byo_trainer_command")
            or os.environ.get("BYO_TRAINER_COMMAND")
            or ""
        ),
        byo_vlm_command=str(
            overrides.get("byo_vlm_command") or os.environ.get("BYO_VLM_COMMAND") or ""
        ),
        byo_eval_command=str(
            overrides.get("byo_eval_command")
            or os.environ.get("BYO_EVAL_COMMAND")
            or ""
        ),
        byo_rerun_command=str(
            overrides.get("byo_rerun_command")
            or os.environ.get("BYO_RERUN_COMMAND")
            or ""
        ),
        byo_policy_command=str(
            overrides.get("byo_policy_command")
            or os.environ.get("BYO_POLICY_COMMAND")
            or ""
        ),
        rerun_enabled=_bool_value(
            overrides.get("rerun_enabled", os.environ.get("NPA_SIM2REAL_RERUN", "1"))
        ),
        k8s_namespace=str(
            overrides.get("k8s_namespace")
            or os.environ.get("NPA_SIM2REAL_K8S_NAMESPACE")
            or _serviceaccount_namespace()
            or "default"
        ),
        k8s_service_account=str(
            overrides.get("k8s_service_account")
            or os.environ.get("NPA_SIM2REAL_K8S_SERVICE_ACCOUNT")
            or "agent-sa"
        ),
        k8s_image_pull_secrets=str(
            overrides.get("k8s_image_pull_secrets")
            or os.environ.get("NPA_SIM2REAL_K8S_IMAGE_PULL_SECRETS")
            or "agent-sa,ngc-nvcr-imagepullsecret,npa-nebius-registry"
        ),
        k8s_env_secret_names=str(
            overrides.get("k8s_env_secret_names")
            or os.environ.get("NPA_SIM2REAL_K8S_ENV_SECRET_NAMES")
            or "hf-ngc-tokens,npa-storage-credentials"
        ),
        k8s_gpu_resource=str(
            overrides.get("k8s_gpu_resource")
            or os.environ.get("NPA_SIM2REAL_K8S_GPU_RESOURCE")
            or "nvidia.com/gpu"
        ),
        k8s_gpu_product=str(
            overrides.get("k8s_gpu_product")
            or os.environ.get("NPA_SIM2REAL_K8S_GPU_PRODUCT")
            or "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
        ),
        k8s_kubeconfig=str(
            overrides.get("k8s_kubeconfig")
            or os.environ.get("KUBECONFIG")
            or os.environ.get("NPA_SIM2REAL_KUBECONFIG")
            or ""
        ),
        k8s_context=str(
            overrides.get("k8s_context")
            or os.environ.get("NPA_SIM2REAL_K8S_CONTEXT")
            or ""
        ),
        k8s_job_timeout_s=int(
            overrides.get(
                "k8s_job_timeout_s",
                os.environ.get("NPA_SIM2REAL_K8S_JOB_TIMEOUT_S", "7200"),
            )
        ),
        k8s_max_parallel_gpus=int(
            overrides.get(
                "k8s_max_parallel_gpus",
                os.environ.get(
                    "NPA_SIM2REAL_K8S_MAX_PARALLEL_GPUS",
                    DEFAULT_K8S_MAX_PARALLEL_GPUS,
                ),
            )
        ),
        source_repo=str(
            overrides.get("source_repo")
            or os.environ.get("NPA_SOURCE_REPO")
            or ""
        ),
        source_ref=str(
            overrides.get("source_ref")
            or os.environ.get("NPA_SOURCE_REF")
            or ""
        ),
        heldout_eval_limit=int(
            overrides.get(
                "heldout_eval_limit",
                os.environ.get("NPA_SIM2REAL_HELDOUT_EVAL_LIMIT", "0"),
            )
        ),
    )


def artifact_uris(config: Sim2RealLoopConfig) -> dict[str, str]:
    """Return canonical S3 artifact URIs for the full 14-stage workflow."""

    if not config.s3_bucket:
        return {}
    root = _artifact_root_uri(config)
    return {
        "root": f"{root}/",
        "trigger_dataset": config.trigger_dataset_uri,
        "stage_01_trigger": f"{root}/stage_01_trigger/trigger.json",
        "stage_02_assets": f"{root}/stage_02_assets/consumed_scene_spec.json",
        "stage_02_assets_stub": f"{root}/stage_02_assets/consumed_scene_spec.json",
        "stage_03_augment": f"{root}/augment/cosmos2-transfer-result.json",
        "stage_04_envs_raw": f"{root}/envs/raw/",
        "stage_05_envs_train": f"{root}/envs/train/",
        "stage_06_tokens": f"{root}/tokens/manifest.json",
        "stage_07_actions_train": f"{root}/actions/train/",
        "stage_08_vlm_eval_train": f"{root}/vlm_eval/train/",
        "stage_09_training_signal": f"{root}/training_signal/train/",
        "inner_loop": f"{root}/inner_loop/",
        "stage_10_eval_heldout": f"{root}/eval/heldout/report.json",
        "stage_11_outer_loop": f"{root}/outer_loop/decision.json",
        "candidate_checkpoint": f"{root}/checkpoints/candidate/",
        "stage_12_external_validation_stub": f"{root}/stage_12_external_validation/external_stub.json",
        "stage_13_retrigger": f"{root}/stage_13_retrigger/retrigger.json",
        "report": f"{root}/reports/sim2real-report.json",
    }


def byo_seams(config: Sim2RealLoopConfig) -> dict[str, Any]:
    """Return every runtime-configurable plug point."""

    return {
        "s3_endpoint": config.s3_endpoint,
        "s3_bucket": config.s3_bucket,
        "trigger_dataset_uri": config.trigger_dataset_uri,
        "trigger_dataset_id": config.trigger_dataset_id,
        "assets_uri": config.assets_uri,
        "scene_spec_uri": config.scene_spec_uri,
        "cameras_uri": config.cameras_uri,
        "augment_image": config.augment_image,
        "action_rollouts_uri": config.action_rollouts_uri,
        "train_envs_uri": config.train_envs_uri,
        "heldout_envs_uri": config.heldout_envs_uri,
        "policy_image": config.policy_image,
        "trainer_image": config.trainer_image,
        "vlm_image": config.vlm_image,
        "eval_image": config.eval_image,
        "isaac_image": config.isaac_image,
        "sim_backend": config.sim_backend,
        "isaac_task": config.isaac_task,
        "threshold": config.threshold,
        "inner_iterations": config.inner_iterations,
        "outer_iterations": config.outer_iterations,
        "loop_of_loops_iterations": config.loop_of_loops_iterations,
        "byo_signal_converter": config.byo_signal_converter,
        "byo_trainer_command": config.byo_trainer_command,
        "byo_vlm_command": config.byo_vlm_command,
        "byo_eval_command": config.byo_eval_command,
        "byo_rerun_command": config.byo_rerun_command,
        "rerun_enabled": config.rerun_enabled,
        "k8s_namespace": config.k8s_namespace,
        "k8s_service_account": config.k8s_service_account,
        "k8s_image_pull_secrets": _split_csv(config.k8s_image_pull_secrets),
        "k8s_env_secret_names": _split_csv(config.k8s_env_secret_names),
        "k8s_gpu_request": {
            "resource": config.k8s_gpu_resource,
            "product": config.k8s_gpu_product,
            "count": 1,
        },
        "source_ref": config.source_ref,
        "heldout_eval_limit": config.heldout_eval_limit,
    }
