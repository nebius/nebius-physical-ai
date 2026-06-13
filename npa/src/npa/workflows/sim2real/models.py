"""Sim2Real configuration models and image defaults."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from npa.deploy.images import container_image_for_tool

from npa.workflows.sim2real.constants import (
    DEFAULT_ACTION_ENV_LIMIT,
    DEFAULT_COSMOS2_TRANSFER_TAG,
    DEFAULT_ENVGEN_SHARD_COUNT,
    DEFAULT_K8S_MAX_PARALLEL_GPUS,
    DEFAULT_ENVGEN_TAG,
    DEFAULT_EVAL_TAG,
    DEFAULT_HELDOUT_ENVS,
    DEFAULT_INNER_ITERATIONS,
    DEFAULT_ISAAC_TAG,
    DEFAULT_ISAAC_TASK,
    DEFAULT_LEROBOT_DATASET_ID,
    DEFAULT_LOOP_OF_LOOPS_ITERATIONS,
    DEFAULT_OUTER_ITERATIONS,
    DEFAULT_PREFIX,
    DEFAULT_REFERENCE_POLICY_TAG,
    DEFAULT_REFERENCE_VLM_MODEL,
    DEFAULT_REASON2_MODEL,
    DEFAULT_REASON3_MODEL,
    DEFAULT_ROLLOUT_COUNT,
    DEFAULT_S3_ENDPOINT,
    DEFAULT_SIM_BACKEND,
    DEFAULT_STEPS_PER_ROLLOUT,
    DEFAULT_THRESHOLD,
    DEFAULT_TRAINER_TAG,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_VLM_IMAGE_TAG,
    SIM_BACKEND_ISAAC,
    SIM_BACKENDS,
)

class Sim2RealLoopError(Exception):
    """Raised when the Sim2Real loop cannot produce a valid artifact."""


@dataclass(frozen=True)
class ComponentRecord:
    """Status line for one workflow stage or plug point."""

    name: str
    tier: str
    evidence: str
    artifacts: dict[str, str] = field(default_factory=dict)
    next_action: str = "CONTINUE"


@dataclass(frozen=True)
class Sim2RealLoopConfig:
    """Runtime configuration for the Sim2Real Stage 7-13 loop."""

    run_id: str
    output_dir: Path | None = None
    s3_bucket: str = ""
    s3_prefix: str = DEFAULT_PREFIX
    s3_endpoint: str = DEFAULT_S3_ENDPOINT
    trigger_dataset_uri: str = ""
    trigger_dataset_id: str = DEFAULT_LEROBOT_DATASET_ID
    action_rollouts_uri: str = ""
    train_envs_uri: str = ""
    heldout_envs_uri: str = ""
    assets_uri: str = ""
    scene_spec_uri: str = ""
    # BYO robot embodiment (alongside the object SceneSpec). ``robot_spec_uri``
    # points at a RobotSpec JSON; ``robot_preset`` selects a built-in preset
    # (franka/ur5e/ur10e/flexiv); ``robot_source`` selects a bare source. All
    # empty => the default Franka Panda robot (no change to today's behavior).
    robot_spec_uri: str = ""
    robot_source: str = ""
    robot_preset: str = ""
    augment_image: str = f"npa-cosmos2-transfer:{DEFAULT_COSMOS2_TRANSFER_TAG}"
    envgen_image: str = f"npa-sim2real-envgen:{DEFAULT_ENVGEN_TAG}"
    env_count: int = 0
    train_fraction: float = DEFAULT_TRAIN_FRACTION
    envgen_shard_count: int = DEFAULT_ENVGEN_SHARD_COUNT
    action_env_limit: int = DEFAULT_ACTION_ENV_LIMIT
    policy_image: str = f"npa-sim2real-reference-policy:{DEFAULT_REFERENCE_POLICY_TAG}"
    trainer_image: str = f"npa-lerobot-vlm-rl:{DEFAULT_TRAINER_TAG}"
    vlm_image: str = f"npa-cosmos3-reason:{DEFAULT_VLM_IMAGE_TAG}"
    vlm_reason2_image: str = ""
    vlm_reason3_image: str = ""
    eval_image: str = f"npa-sim2real-eval:{DEFAULT_EVAL_TAG}"
    isaac_image: str = f"npa-isaac-lab:{DEFAULT_ISAAC_TAG}"
    sim_backend: str = DEFAULT_SIM_BACKEND
    isaac_task: str = DEFAULT_ISAAC_TASK
    vlm_model: str = DEFAULT_REFERENCE_VLM_MODEL
    vlm_reason2_model: str = DEFAULT_REASON2_MODEL
    vlm_reason3_model: str = DEFAULT_REASON3_MODEL
    vlm_dual_reason: bool = True
    threshold: float = DEFAULT_THRESHOLD
    inner_iterations: int = DEFAULT_INNER_ITERATIONS
    outer_iterations: int = DEFAULT_OUTER_ITERATIONS
    loop_of_loops_iterations: int = DEFAULT_LOOP_OF_LOOPS_ITERATIONS
    rollout_count: int = DEFAULT_ROLLOUT_COUNT
    steps_per_rollout: int = DEFAULT_STEPS_PER_ROLLOUT
    heldout_env_count: int = DEFAULT_HELDOUT_ENVS
    seed: int = 42
    upload_artifacts: bool = False
    no_guardrails: bool = False
    signal_loss_weight: float = 1.0
    learning_rate: float = 0.05
    byo_signal_converter: str = ""
    byo_trainer_command: str = ""
    byo_vlm_command: str = ""
    byo_eval_command: str = ""
    byo_rerun_command: str = ""
    byo_policy_command: str = ""
    rerun_enabled: bool = True
    k8s_namespace: str = ""
    k8s_service_account: str = "agent-sa"
    k8s_image_pull_secrets: str = "agent-sa,ngc-nvcr-imagepullsecret,npa-nebius-registry"
    k8s_env_secret_names: str = "hf-ngc-tokens,npa-storage-credentials"
    k8s_gpu_resource: str = "nvidia.com/gpu"
    k8s_gpu_product: str = "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
    k8s_kubeconfig: str = ""
    k8s_context: str = ""
    k8s_job_timeout_s: int = 7200
    k8s_max_parallel_gpus: int = DEFAULT_K8S_MAX_PARALLEL_GPUS
    source_repo: str = ""
    source_ref: str = ""
    heldout_eval_limit: int = 0

    def validate(self) -> None:
        if not self.run_id:
            raise Sim2RealLoopError("run_id must not be empty")
        if not 0.0 <= self.threshold <= 1.0:
            raise Sim2RealLoopError(
                f"threshold must be in [0, 1], got {self.threshold}"
            )
        if self.inner_iterations <= 0:
            raise Sim2RealLoopError("inner_iterations must be positive")
        if self.outer_iterations <= 0:
            raise Sim2RealLoopError("outer_iterations must be positive")
        if self.loop_of_loops_iterations <= 0:
            raise Sim2RealLoopError("loop_of_loops_iterations must be positive")
        if self.rollout_count <= 0:
            raise Sim2RealLoopError("rollout_count must be positive")
        if self.steps_per_rollout <= 0:
            raise Sim2RealLoopError("steps_per_rollout must be positive")
        if self.heldout_env_count <= 0:
            raise Sim2RealLoopError("heldout_env_count must be positive")
        if self.learning_rate <= 0:
            raise Sim2RealLoopError("learning_rate must be positive")
        if self.signal_loss_weight < 0:
            raise Sim2RealLoopError("signal_loss_weight must be non-negative")
        if self.k8s_job_timeout_s <= 0:
            raise Sim2RealLoopError("k8s_job_timeout_s must be positive")
        if self.k8s_max_parallel_gpus <= 0:
            raise Sim2RealLoopError("k8s_max_parallel_gpus must be positive")
        if self.heldout_eval_limit < 0:
            raise Sim2RealLoopError("heldout_eval_limit must be non-negative")
        if self.env_count < 0:
            raise Sim2RealLoopError("env_count must be non-negative")
        if not 0.0 < self.train_fraction < 1.0:
            raise Sim2RealLoopError("train_fraction must be in (0, 1)")
        if self.envgen_shard_count <= 0:
            raise Sim2RealLoopError("envgen_shard_count must be positive")
        if self.action_env_limit <= 0:
            raise Sim2RealLoopError("action_env_limit must be positive")
        if self.sim_backend not in SIM_BACKENDS:
            raise Sim2RealLoopError(
                f"sim_backend must be one of {SIM_BACKENDS}, got {self.sim_backend!r}"
            )

    def heldout_backend_image(self) -> str:
        """Return the container image that runs the held-out rollout backend.

        Genesis runs inside the reference eval image; Isaac runs inside the
        Isaac Lab image (Isaac Sim headless, RT cores required).
        """

        if self.sim_backend == SIM_BACKEND_ISAAC:
            return self.isaac_image
        return self.eval_image


def new_run_id(prefix: str = "sim2real-b") -> str:
    """Return a run id suitable for S3 prefixes and local artifact paths."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def default_vlm_image(*, registry: str | None = None) -> str:
    """Return the reference Cosmos3-reason-compatible VLM image."""

    if registry or os.environ.get("NPA_REGISTRY"):
        return container_image_for_tool("cosmos3-reason", registry=registry)
    return f"npa-cosmos3-reason:{DEFAULT_VLM_IMAGE_TAG}"


def default_envgen_image(*, registry: str | None = None) -> str:
    """Return the reference env-generation image used by Stages 3-6."""

    if registry or os.environ.get("NPA_REGISTRY"):
        return container_image_for_tool("sim2real-envgen", registry=registry)
    return f"npa-sim2real-envgen:{DEFAULT_ENVGEN_TAG}"


def default_augment_image(*, registry: str | None = None) -> str:
    """Return the reference Cosmos2 transfer image used by Stage 3."""

    if registry or os.environ.get("NPA_REGISTRY"):
        return container_image_for_tool("cosmos2-transfer", registry=registry)
    return f"npa-cosmos2-transfer:{DEFAULT_COSMOS2_TRANSFER_TAG}"


def default_policy_image(*, registry: str | None = None) -> str:
    """Return the reference action-generation policy image."""

    if registry or os.environ.get("NPA_REGISTRY"):
        return container_image_for_tool("sim2real-reference-policy", registry=registry)
    return f"npa-sim2real-reference-policy:{DEFAULT_REFERENCE_POLICY_TAG}"


def default_trainer_image(*, registry: str | None = None) -> str:
    """Return the reference VLM-signal LeRobot trainer image."""

    if registry or os.environ.get("NPA_REGISTRY"):
        return container_image_for_tool("lerobot-vlm-rl", registry=registry)
    return f"npa-lerobot-vlm-rl:{DEFAULT_TRAINER_TAG}"


def default_eval_image(*, registry: str | None = None) -> str:
    """Return the reference held-out eval harness image."""

    if registry or os.environ.get("NPA_REGISTRY"):
        return container_image_for_tool("sim2real-eval", registry=registry)
    return f"npa-sim2real-eval:{DEFAULT_EVAL_TAG}"


def default_isaac_image(*, registry: str | None = None) -> str:
    """Return the Isaac Lab held-out rollout image (Isaac Sim headless)."""

    if registry or os.environ.get("NPA_REGISTRY"):
        return container_image_for_tool("isaac-lab", registry=registry)
    return f"npa-isaac-lab:{DEFAULT_ISAAC_TAG}"

