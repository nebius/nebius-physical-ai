"""Concrete Sim2Real VLM-to-RL loop and end-to-end runbook runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from npa.clients.storage import StorageClient
from npa.deploy.images import container_image_for_tool
from npa.workbench.lerobot.policy_container import (
    parse_vlm_signal_batch,
    run_vlm_signal_training_step,
)


DEFAULT_S3_ENDPOINT = ""
DEFAULT_BUCKET = ""
DEFAULT_PREFIX = "sim2real-b"
DEFAULT_VLM_IMAGE_TAG = "3.0.1-genuine-sm120"
DEFAULT_ENVGEN_TAG = "0.1.1"
DEFAULT_REFERENCE_POLICY_TAG = "0.1.1"
DEFAULT_TRAINER_TAG = "0.1.0"
DEFAULT_EVAL_TAG = "0.1.1-genuine-sm120"
DEFAULT_ISAAC_TAG = "2.3.2.post1"
# Pluggable held-out sim backend. Genesis remains fully supported; Isaac Lab
# (Isaac Sim headless) is the default and requires RT-core GPUs (L40S / RTX Pro).
SIM_BACKEND_GENESIS = "genesis"
SIM_BACKEND_ISAAC = "isaac"
SIM_BACKENDS = (SIM_BACKEND_GENESIS, SIM_BACKEND_ISAAC)
DEFAULT_SIM_BACKEND = SIM_BACKEND_ISAAC
# Default headless Isaac Lab manipulation task for the stock held-out rollout.
DEFAULT_ISAAC_TASK = "Isaac-Lift-Cube-Franka-v0"
DEFAULT_THRESHOLD = 0.75
DEFAULT_INNER_ITERATIONS = 2
DEFAULT_OUTER_ITERATIONS = 1
DEFAULT_LOOP_OF_LOOPS_ITERATIONS = 1
DEFAULT_ROLLOUT_COUNT = 3
DEFAULT_STEPS_PER_ROLLOUT = 4
DEFAULT_HELDOUT_ENVS = 8
DEFAULT_REFERENCE_VLM_MODEL = "nvidia/Cosmos-Reason1-7B"
DEFAULT_LEROBOT_DATASET_ID = "lerobot/pusht"
REFERENCE_VLM_ALIASES = {"", "npa-cosmos3-reason", "cosmos3-reason"}
DEFAULT_COSMOS_REASON_CACHE = "/models/cosmos-reason1"
SCHEMA_VLM_EVAL = "npa.sim2real.vlm_eval.v1"
SCHEMA_RL_SIGNAL = "npa.sim2real.rl_signal.v1"
SCHEMA_HELDOUT_REPORT = "npa.sim2real.heldout_eval.v1"
SCHEMA_THRESHOLD_DECISION = "npa.sim2real.threshold_decision.v1"
SCHEMA_E2E_REPORT = "npa.sim2real.e2e_report.v1"

ERROR_SEVERITY = {
    "collision": 0.95,
    "missed_target": 0.85,
    "unstable": 0.7,
    "late_grasp": 0.55,
    "minor_alignment": 0.3,
    "ok": 0.0,
}

CORRECTIVE_TARGETS = {
    "collision": {
        "nl_correction": "Back off from contact and retry with a shallower approach.",
        "action_delta": [-0.12, 0.0, 0.04],
    },
    "missed_target": {
        "nl_correction": "Move the end effector toward the object center before closing.",
        "action_delta": [0.12, 0.02, 0.0],
    },
    "unstable": {
        "nl_correction": "Reduce vertical speed and stabilize before release.",
        "action_delta": [0.0, 0.0, -0.08],
    },
    "late_grasp": {
        "nl_correction": "Close the gripper earlier once the object is centered.",
        "action_delta": [0.03, 0.0, 0.02],
    },
    "minor_alignment": {
        "nl_correction": "Apply a small lateral correction toward the target marker.",
        "action_delta": [0.04, 0.01, 0.0],
    },
    "ok": {
        "nl_correction": "Preserve the current action.",
        "action_delta": [0.0, 0.0, 0.0],
    },
}


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
    augment_image: str = f"npa-sim2real-envgen:{DEFAULT_ENVGEN_TAG}"
    policy_image: str = f"npa-sim2real-reference-policy:{DEFAULT_REFERENCE_POLICY_TAG}"
    trainer_image: str = f"npa-lerobot-vlm-rl:{DEFAULT_TRAINER_TAG}"
    vlm_image: str = f"npa-cosmos3-reason:{DEFAULT_VLM_IMAGE_TAG}"
    eval_image: str = f"npa-sim2real-eval:{DEFAULT_EVAL_TAG}"
    isaac_image: str = f"npa-isaac-lab:{DEFAULT_ISAAC_TAG}"
    sim_backend: str = DEFAULT_SIM_BACKEND
    isaac_task: str = DEFAULT_ISAAC_TASK
    vlm_model: str = DEFAULT_REFERENCE_VLM_MODEL
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
    k8s_namespace: str = ""
    k8s_service_account: str = "agent-sa"
    k8s_image_pull_secrets: str = "agent-sa,ngc-nvcr-imagepullsecret,npa-nebius-registry"
    k8s_env_secret_names: str = "hf-ngc-tokens,npa-storage-credentials"
    k8s_gpu_resource: str = "nvidia.com/gpu"
    k8s_gpu_product: str = "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
    k8s_kubeconfig: str = ""
    k8s_context: str = ""
    k8s_job_timeout_s: int = 7200
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
        if self.heldout_eval_limit < 0:
            raise Sim2RealLoopError("heldout_eval_limit must be non-negative")
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


def build_config_from_env(**overrides: Any) -> Sim2RealLoopConfig:
    """Build a Sim2Real loop config from explicit values and env fallbacks."""

    run_id = str(
        overrides.get("run_id") or os.environ.get("NPA_SIM2REAL_RUN_ID") or new_run_id()
    )
    bucket = str(
        overrides.get("s3_bucket")
        or os.environ.get("NPA_SIM2REAL_BUCKET")
        or os.environ.get("NPA_S3_BUCKET")
        or os.environ.get("S3_BUCKET")
        or ""
    )
    registry = os.environ.get("NPA_REGISTRY", "")
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
        augment_image=str(
            overrides.get("augment_image")
            or os.environ.get("AUGMENT_IMAGE")
            or default_envgen_image(registry=registry or None)
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
    """Return canonical S3 artifact URIs for the full 13-stage workflow."""

    if not config.s3_bucket:
        return {}
    root = _artifact_root_uri(config)
    return {
        "root": f"{root}/",
        "trigger_dataset": config.trigger_dataset_uri,
        "stage_01_trigger": f"{root}/stage_01_trigger/trigger.json",
        "stage_02_assets_stub": f"{root}/stage_02_assets/external_stub.json",
        "stage_03_augment": f"{root}/augment/manifest.json",
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


def run_full_loop(
    config: Sim2RealLoopConfig,
    *,
    upload: bool | None = None,
) -> dict[str, Any]:
    """Run the full local/executable Sim2Real loop and write all artifacts."""

    config.validate()
    local_dir = config.output_dir or Path(
        tempfile.mkdtemp(prefix=f"npa-{config.run_id}-")
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    components: list[ComponentRecord] = []
    stage_artifacts: dict[str, str] = {}
    stage_records: list[dict[str, Any]] = []

    stage_records.append(
        _write_stage(local_dir, 1, "trigger", _trigger_payload(config))
    )
    components.append(
        ComponentRecord(
            "stage_01_trigger",
            "WORKS",
            "Consumed the dedicated LeRobot dataset trigger path and resolved runtime plug points.",
            {"local": str(local_dir / "stage_01_trigger" / "trigger.json")},
        )
    )

    if config.scene_spec_uri or config.assets_uri:
        consumed = _consume_stage_assets(config, local_dir)
        stage_records.append(consumed["stage_record"])
        components.append(
            ComponentRecord(
                "stage_02_assets",
                "WORKS",
                (
                    "Downloaded and validated the BYO mesh/SceneSpec and emitted a "
                    "consumed scene spec with per-object provenance for the "
                    "held-out Genesis rollout."
                ),
                {"local": consumed["consumed_spec_path"]},
            )
        )
    else:
        stage_records.append(
            _write_stage(
                local_dir,
                2,
                "assets",
                {
                    "schema": "npa.sim2real.external_stub.v1",
                    "stage": 2,
                    "name": "external real assets and SceneSpec",
                    "status": "documented_external_stub",
                    "assets_uri": config.assets_uri,
                    "scene_spec_uri": config.scene_spec_uri,
                    "next_action": "CONTINUE",
                },
                filename="external_stub.json",
            )
        )
        components.append(
            ComponentRecord(
                "stage_02_assets",
                "SEAM",
                "External assets and SceneSpec are documented BYO inputs for this reference run.",
                {"local": str(local_dir / "stage_02_assets" / "external_stub.json")},
            )
        )

    stage_records.append(
        _write_json_artifact(
            local_dir / "augment" / "manifest.json",
            {
                "schema": "npa.sim2real.augment_manifest.v1",
                "stage": 3,
                "augment": "cosmos2-transfer",
                "image": config.augment_image or "reference-cosmos2-transfer",
                "assets_uri": config.assets_uri,
                "output_uri": "augment/",
                "status": "reference_manifest",
            },
        )
    )
    components.append(
        ComponentRecord(
            "stage_03_augment",
            "WORKS",
            "Wrote a Cosmos transfer augmentation manifest with BYO image override support.",
            {"local": str(local_dir / "augment" / "manifest.json")},
        )
    )

    raw_envs = _write_env_manifest(
        local_dir / "envs" / "raw",
        count=config.rollout_count + config.heldout_env_count,
        seed=config.seed,
    )
    train_envs, heldout_envs = _write_train_heldout_split(
        local_dir / "envs",
        raw_envs=raw_envs,
        train_count=config.rollout_count,
        heldout_count=config.heldout_env_count,
        seed=config.seed,
    )
    stage_records.extend([raw_envs, train_envs, heldout_envs])
    components.append(
        ComponentRecord(
            "stage_04_06_env_gen_split_tokens",
            "WORKS",
            "Generated deterministic raw env specs, train/heldout split, and token manifest.",
            {
                "raw_envs": str(local_dir / "envs" / "raw" / "manifest.json"),
                "train_envs": str(local_dir / "envs" / "train" / "manifest.json"),
                "heldout_envs": str(local_dir / "envs" / "heldout" / "manifest.json"),
                "tokens": str(local_dir / "tokens" / "manifest.json"),
            },
        )
    )
    _write_json_artifact(
        local_dir / "tokens" / "manifest.json",
        {
            "schema": "npa.sim2real.tokens.v1",
            "stage": 6,
            "source": "stage-a-compatible-reference",
            "train_env_count": len(train_envs["payload"]["envs"]),
            "heldout_env_count": len(heldout_envs["payload"]["envs"]),
            "status": "ready",
        },
    )

    outer_history: list[dict[str, Any]] = []
    final_inner: dict[str, Any] | None = None
    final_eval: dict[str, Any] | None = None
    final_decision: dict[str, Any] | None = None
    quality = 0.36 + rng.random() * 0.04
    for outer_iteration in range(1, config.outer_iterations + 1):
        inner = run_inner_loop(
            config,
            local_dir=local_dir,
            initial_quality=quality,
            outer_iteration=outer_iteration,
        )
        final_inner = inner
        quality = float(inner["final_quality"])
        heldout_report = run_heldout_eval(
            config,
            local_dir=local_dir,
            inner_evidence=inner,
            outer_iteration=outer_iteration,
        )
        final_eval = heldout_report
        decision = threshold_decision(
            config,
            local_dir=local_dir,
            heldout_report=heldout_report,
            outer_iteration=outer_iteration,
        )
        final_decision = decision
        outer_history.append(
            {
                "outer_iteration": outer_iteration,
                "inner_loop": inner["evidence_uri"],
                "heldout_report": heldout_report["report_uri"],
                "decision": decision,
            }
        )
        if decision["decision"] == "promote_checkpoint":
            break
        quality = min(0.95, quality + 0.12)

    if final_decision is None or final_inner is None or final_eval is None:
        raise Sim2RealLoopError("full loop did not execute an outer iteration")

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
        ComponentRecord(
            "stage_12_external_validation",
            "SEAM",
            "External real-world validation is a documented BYO gate; loop-of-loops continues through Stage 13.",
            {
                "local": str(
                    local_dir / "stage_12_external_validation" / "external_stub.json"
                )
            },
        )
    )

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
        "retrigger_condition": "real_world_lerobot_dataset_landed",
        "should_retrigger": config.loop_of_loops_iterations > 1,
    }
    stage_records.append(
        _write_json_artifact(
            local_dir / "stage_13_retrigger" / "retrigger.json", retrigger
        )
    )
    components.append(
        ComponentRecord(
            "stage_13_retrigger",
            "WORKS",
            "Wrote loop-of-loops retrigger record with max-iteration cap.",
            {"local": str(local_dir / "stage_13_retrigger" / "retrigger.json")},
        )
    )

    components.extend(
        [
            ComponentRecord(
                "vlm_byo_seam",
                "WORKS",
                "VLM image/command are runtime-configurable; default model is nvidia/Cosmos-Reason1-7B.",
                {"image": config.vlm_image},
            ),
            ComponentRecord(
                "trainer_byo_seam",
                "WORKS",
                "Trainer image/command are runtime-configurable; default reference consumes npa.sim2real.rl_signal.v1.",
                {"image": config.trainer_image},
            ),
            ComponentRecord(
                "eval_byo_seam",
                "WORKS",
                "Held-out eval image/command and threshold are runtime-configurable.",
                {"image": config.eval_image},
            ),
        ]
    )

    report = {
        "schema": SCHEMA_E2E_REPORT,
        "run_id": config.run_id,
        "status": "completed",
        "created_at": _utc_now(),
        "local_artifact_dir": str(local_dir),
        "s3_artifacts": artifact_uris(config),
        "config": _redacted_config(config),
        "byo_seams": byo_seams(config),
        "components": [asdict(component) for component in components],
        "stage_records": stage_records,
        "inner_loop": final_inner,
        "outer_loop": {
            "history": outer_history,
            "latest_heldout_report": final_eval,
            "latest_decision": final_decision,
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
    stage_artifacts["report"] = str(report_path)
    upload_enabled = config.upload_artifacts if upload is None else upload
    if upload_enabled and config.s3_bucket:
        report["upload"] = upload_run_artifacts(config, local_dir)
        _write_json_artifact(report_path, report)
    else:
        report["upload"] = {
            "status": "skipped",
            "reason": "upload_artifacts is false or no s3_bucket configured",
        }
        _write_json_artifact(report_path, report)
    return report


def run_inner_loop(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    initial_quality: float,
    outer_iteration: int = 1,
) -> dict[str, Any]:
    """Run action generation, VLM eval, signal conversion, and policy update."""

    iteration_records: list[dict[str, Any]] = []
    reward_trend: list[float] = []
    policy_deltas: list[float] = []
    all_signals: list[dict[str, Any]] = []
    quality = float(initial_quality)
    reward_head = 0.0
    action_bias = 0.0
    for iteration in range(1, config.inner_iterations + 1):
        actions_dir = (
            local_dir
            / "actions"
            / "train"
            / f"outer-{outer_iteration:02d}"
            / f"iter-{iteration:02d}"
        )
        rollouts = generate_action_rollouts(
            actions_dir,
            count=config.rollout_count,
            steps_per_rollout=config.steps_per_rollout,
            seed=config.seed + outer_iteration * 100 + iteration,
            quality=quality,
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
        for rollout in rollouts:
            evaluation = evaluate_rollout_with_vlm(
                rollout,
                output_dir=eval_dir,
                config=config,
            )
            signal = convert_vlm_eval_to_rl_signal(evaluation)
            _write_json_artifact(signal_dir / f"{signal['rollout_id']}.json", signal)
            evals.append(evaluation)
            signals.append(signal)
            all_signals.append(signal)
        signal_batch_path = (
            local_dir
            / "inner_loop"
            / f"outer-{outer_iteration:02d}"
            / f"signals-iter-{iteration:02d}.json"
        )
        _write_json_artifact(
            signal_batch_path, {"schema": SCHEMA_RL_SIGNAL, "signals": signals}
        )
        parsed_signals = parse_vlm_signal_batch({"signals": signals})
        update = run_vlm_signal_training_step(
            parsed_signals,
            output_dir=local_dir
            / "inner_loop"
            / f"outer-{outer_iteration:02d}"
            / "trainer"
            / f"iter-{iteration:02d}",
            learning_rate=config.learning_rate,
            signal_loss_weight=config.signal_loss_weight,
            initial_reward_head=reward_head,
            initial_action_bias=action_bias,
        )
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
        delta_vs_control = max(0.0, update.policy_delta_l2 - control.policy_delta_l2)
        policy_deltas.append(round(delta_vs_control, 8))
        quality = min(
            0.98, quality + max(0.06, min(0.18, delta_vs_control * 2.0 + 0.07))
        )
        iteration_records.append(
            {
                "iteration": iteration,
                "actions_dir": str(actions_dir),
                "vlm_eval_dir": str(eval_dir),
                "signal_dir": str(signal_dir),
                "signal_batch": str(signal_batch_path),
                "mean_reward": mean_reward,
                "update": update.to_dict(),
                "no_signal_control": control.to_dict(),
                "policy_delta_vs_control": round(delta_vs_control, 8),
                "next_rollout_quality": round(quality, 6),
                "sample_vlm_eval": evals[0],
                "sample_signal": signals[0],
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
    evidence = {
        "schema": "npa.sim2real.inner_loop_evidence.v1",
        "outer_iteration": outer_iteration,
        "status": "closed",
        "reward_trend": reward_trend,
        "signal_diversity": signal_diversity,
        "policy_delta_vs_no_signal_control": policy_deltas,
        "attribution": (
            "The reference update and no-signal control share initial adapter state. "
            "Only the VLM-derived rewards, advantages, and corrective targets produce the policy-output delta."
        ),
        "iterations": iteration_records,
        "final_quality": round(quality, 6),
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
    """Invoke the configured VLM component and parse its structured judgment."""

    manifest_path = rollout_dir / "manifest.json"
    if not manifest_path.exists():
        raise Sim2RealLoopError(f"rollout manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rollout_id = str(manifest.get("rollout_id") or rollout_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{rollout_id}.json"
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
    if config.byo_vlm_command.strip():
        invocation = _run_component_command(
            config.byo_vlm_command,
            cwd=rollout_dir,
            env=env,
            component="vlm_eval",
        )
    else:
        attempt_id = _component_attempt_id(config, "vlm_eval", rollout_id)
        rollout_uri = _upload_component_directory(
            config,
            rollout_dir,
            component="vlm_eval",
            attempt_id=attempt_id,
            name="rollout",
        )
        output_uri = _component_output_uri(
            config,
            component="vlm_eval",
            attempt_id=attempt_id,
            filename=f"{rollout_id}.json",
        )
        env["NPA_SIM2REAL_ROLLOUT_URI"] = rollout_uri
        env["NPA_SIM2REAL_OUTPUT_URI"] = output_uri
        invocation = _run_image_component(
            config.vlm_image,
            component="vlm_eval",
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
    _write_json_artifact(output_path, evaluation)
    return evaluation


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
    timeout_s: int = 7200,
) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
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


def _run_image_component(
    image: str,
    *,
    component: str,
    env: dict[str, str],
    output_json: Path,
    output_uri: str,
    config: Sim2RealLoopConfig,
    timeout_s: int = 7200,
) -> dict[str, Any]:
    return _run_kubernetes_image_component(
        image,
        component=component,
        env=env,
        output_json=output_json,
        output_uri=output_uri,
        config=config,
        timeout_s=timeout_s,
    )


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
    job_name = _k8s_job_name(config.run_id, component)
    manifest = _component_job_manifest(
        image,
        component=component,
        env=env,
        config=config,
        namespace=namespace,
        job_name=job_name,
        timeout_s=timeout_s,
    )
    apply_result = _kubectl(
        config,
        ["apply", "-f", "-"],
        stdin=json.dumps(manifest),
        timeout_s=120,
    )
    wait_result = _kubectl(
        config,
        [
            "wait",
            "--for=condition=complete",
            f"job/{job_name}",
            "-n",
            namespace,
            f"--timeout={timeout_s}s",
        ],
        timeout_s=timeout_s + 60,
        check=False,
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
    events_excerpt = ""
    if wait_result.returncode != 0:
        events = _kubectl(
            config,
            [
                "get",
                "events",
                "-n",
                namespace,
                "--field-selector",
                f"involvedObject.name={job_name}",
                "-o",
                "json",
            ],
            timeout_s=120,
            check=False,
        )
        events_excerpt = _component_excerpt(events.stdout or events.stderr)
    delete_result = _cleanup_component_job(
        config, namespace=namespace, job_name=job_name
    )
    if wait_result.returncode != 0:
        raise Sim2RealLoopError(
            f"{component} Kubernetes Job {job_name} did not complete: "
            f"{_component_excerpt(wait_result.stderr or wait_result.stdout)} "
            f"{_component_excerpt(logs_result.stdout or logs_result.stderr)} "
            f"{events_excerpt}"
        )
    _download_component_output(config, output_uri, output_json)
    return {
        "mode": "kubernetes_job",
        "component": component,
        "image": image,
        "image_digests": pod_info.get("image_digests", []),
        "namespace": namespace,
        "job_name": job_name,
        "pod": pod_info,
        "gpu_request": {
            "resource": config.k8s_gpu_resource,
            "product": config.k8s_gpu_product,
            "count": 1,
        },
        "service_account": config.k8s_service_account,
        "image_pull_secrets": _split_csv(config.k8s_image_pull_secrets),
        "env_secret_names": _split_csv(config.k8s_env_secret_names),
        "output_uri": output_uri,
        "returncode": wait_result.returncode,
        "apply_stdout_excerpt": _component_excerpt(apply_result.stdout),
        "stdout_excerpt": _component_excerpt(logs_result.stdout),
        "stderr_excerpt": _component_excerpt(logs_result.stderr),
        "cleanup_stdout_excerpt": _component_excerpt(delete_result.stdout),
        "cleanup_stderr_excerpt": _component_excerpt(delete_result.stderr),
    }


def _component_job_manifest(
    image: str,
    *,
    component: str,
    env: dict[str, str],
    config: Sim2RealLoopConfig,
    namespace: str,
    job_name: str,
    timeout_s: int,
) -> dict[str, Any]:
    env_values = _kubernetes_component_env(env, config)
    pull_secrets = [
        {"name": name} for name in _split_csv(config.k8s_image_pull_secrets)
    ]
    env_from = [
        {"secretRef": {"name": name, "optional": True}}
        for name in _split_csv(config.k8s_env_secret_names)
    ]
    template_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "serviceAccountName": config.k8s_service_account,
        "containers": [
            {
                "name": "component",
                "image": image,
                "imagePullPolicy": _image_pull_policy(image),
                "command": ["bash", "-lc"],
                "args": [_component_job_script(component, sim_backend=config.sim_backend)],
                "env": [
                    {"name": key, "value": value}
                    for key, value in sorted(env_values.items())
                    if value != ""
                ],
                "envFrom": env_from,
                "resources": {
                    "requests": {
                        "cpu": "4",
                        "memory": "16Gi",
                        config.k8s_gpu_resource: 1,
                    },
                    "limits": {
                        config.k8s_gpu_resource: 1,
                    },
                },
            }
        ],
        "nodeSelector": {
            "nvidia.com/gpu.compute.major": "12",
            "nvidia.com/gpu.compute.minor": "0",
            "nvidia.com/gpu.product": config.k8s_gpu_product,
        },
    }
    if pull_secrets:
        template_spec["imagePullSecrets"] = pull_secrets
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "sim2real-sibling-component",
                "app.kubernetes.io/component": component.replace("_", "-"),
                "sim2real.local/run-id": _label_value(config.run_id),
            },
            "annotations": {
                "sim2real.local/gpu-request": (
                    "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
                )
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": timeout_s,
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/name": "sim2real-sibling-component",
                        "app.kubernetes.io/component": component.replace("_", "-"),
                        "sim2real.local/run-id": _label_value(config.run_id),
                    }
                },
                "spec": template_spec,
            },
        },
    }


def _component_job_script(component: str, *, sim_backend: str = DEFAULT_SIM_BACKEND) -> str:
    if component == "vlm_eval":
        subcommand = (
            "component-vlm-eval "
            "--input-uri \"${NPA_SIM2REAL_ROLLOUT_URI}\" "
            "--output-uri \"${NPA_SIM2REAL_OUTPUT_URI}\" "
            "--rollout-id \"${NPA_SIM2REAL_ROLLOUT_ID}\" "
            "--model \"${NPA_SIM2REAL_VLM_MODEL}\" "
            "--threshold \"${NPA_SIM2REAL_THRESHOLD}\""
        )
    elif component == "heldout_eval":
        subcommand = (
            "component-heldout-eval "
            "--heldout-envs-uri \"${NPA_SIM2REAL_HELDOUT_ENVS_URI}\" "
            "--inner-evidence-uri \"${NPA_SIM2REAL_INNER_EVIDENCE_URI}\" "
            "--output-uri \"${NPA_SIM2REAL_OUTPUT_URI}\" "
            "--threshold \"${NPA_SIM2REAL_THRESHOLD}\" "
            "--limit \"${NPA_SIM2REAL_HELDOUT_EVAL_LIMIT:-0}\" "
            "--sim-backend \"${NPA_SIM2REAL_SIM_BACKEND:-genesis}\" "
            "--isaac-task \"${NPA_SIM2REAL_ISAAC_TASK:-}\" "
            "--scene-spec-uri \"${NPA_SIM2REAL_SCENE_SPEC_URI:-}\" "
            "--assets-uri \"${NPA_SIM2REAL_ASSETS_URI:-}\""
        )
    else:
        raise Sim2RealLoopError(f"unsupported image component: {component}")
    # The Isaac Lab image ships Isaac Sim + isaaclab only under its bundled
    # interpreter (/isaac-sim/python.sh) and bakes no npa code. Branch npa code
    # is injected at start either from an S3 source tarball
    # (NPA_SIM2REAL_SOURCE_TARBALL_URI, using the pod's mounted S3 creds) or via
    # a git clone (NPA_SOURCE_REPO/NPA_SOURCE_REF when the repo is reachable).
    # boto3 is installed to a writable target dir for the S3 client.
    if component == "heldout_eval" and sim_backend == SIM_BACKEND_ISAAC:
        return f"""set -euo pipefail
PYBIN=/isaac-sim/python.sh
if [ ! -x "$PYBIN" ]; then PYBIN=python; fi
DEPS=/tmp/npa-pydeps
"$PYBIN" -c "import boto3" 2>/dev/null || "$PYBIN" -m pip install --quiet --target "$DEPS" boto3 botocore
export PYTHONPATH="$DEPS:${{PYTHONPATH:-}}"
if [ -n "${{NPA_SIM2REAL_SOURCE_TARBALL_URI:-}}" ]; then
  rm -rf /tmp/npa-source && mkdir -p /tmp/npa-source
  "$PYBIN" - "${{NPA_SIM2REAL_SOURCE_TARBALL_URI}}" <<'PYB'
import os, sys, tarfile, urllib.parse, boto3
u = urllib.parse.urlparse(sys.argv[1])
ep = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT_URL") or None
boto3.client("s3", endpoint_url=ep).download_file(u.netloc, u.path.lstrip("/"), "/tmp/npa-src.tgz")
with tarfile.open("/tmp/npa-src.tgz") as tar:
    tar.extractall("/tmp/npa-source")
PYB
  export PYTHONPATH="/tmp/npa-source/npa/src:${{PYTHONPATH:-}}"
elif [ -n "${{NPA_SOURCE_REPO:-}}" ] && [ -n "${{NPA_SOURCE_REF:-}}" ]; then
  rm -rf /tmp/npa-source
  git clone --quiet --depth 1 --branch "${{NPA_SOURCE_REF}}" "${{NPA_SOURCE_REPO}}" /tmp/npa-source
  export PYTHONPATH="/tmp/npa-source/npa/src:${{PYTHONPATH:-}}"
fi
"$PYBIN" -m npa.workflows.sim2real_loop {subcommand}
"""
    return f"""set -euo pipefail
if [ -n "${{NPA_SOURCE_REPO:-}}" ] && [ -n "${{NPA_SOURCE_REF:-}}" ]; then
  rm -rf /tmp/npa-source
  git clone --quiet --depth 1 --branch "${{NPA_SOURCE_REF}}" "${{NPA_SOURCE_REPO}}" /tmp/npa-source
  export PYTHONPATH="/tmp/npa-source/npa/src:${{PYTHONPATH:-}}"
fi
python -m npa.workflows.sim2real_loop {subcommand}
"""


def _kubernetes_component_env(
    env: dict[str, str], config: Sim2RealLoopConfig
) -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in env.items():
        if key.startswith("NPA_SIM2REAL"):
            safe[key] = value
    safe["AWS_ENDPOINT_URL"] = config.s3_endpoint or env.get("AWS_ENDPOINT_URL", "")
    safe["S3_ENDPOINT_URL"] = config.s3_endpoint or env.get("S3_ENDPOINT_URL", "")
    safe["NPA_SOURCE_REPO"] = config.source_repo or env.get("NPA_SOURCE_REPO", "")
    safe["NPA_SOURCE_REF"] = config.source_ref or env.get("NPA_SOURCE_REF", "")
    return safe


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
    digest = hashlib.sha1(f"{config.run_id}:{component}:{label}".encode("utf-8")).hexdigest()
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
    _storage_client(config).download_path(output_uri, str(output_json))


def _storage_client(config: Sim2RealLoopConfig) -> StorageClient:
    return StorageClient.from_environment(endpoint_url=config.s3_endpoint)


def _k8s_job_name(run_id: str, component: str) -> str:
    run_part = _safe_slug(run_id)[:22] or "run"
    component_part = _safe_slug(component)[:16] or "component"
    suffix = uuid.uuid4().hex[:8]
    return f"s2r-{component_part}-{run_part}-{suffix}"[:63].rstrip("-")


def _label_value(value: str) -> str:
    return (_safe_slug(value)[:63] or "run").rstrip("-")


def _safe_slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in str(value)]
    return "-".join(part for part in "".join(chars).split("-") if part)


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _serviceaccount_namespace() -> str:
    path = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def _normalized_s3_prefix(uri: str) -> str:
    return str(uri or "").strip()


def _read_component_json(output_path: Path, invocation: dict[str, Any]) -> dict[str, Any]:
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    stdout = str(invocation.get("stdout") or "")
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)
    raise Sim2RealLoopError(
        f"{invocation.get('component', 'component')} did not write JSON to {output_path}"
    )


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
        raw_steps = [{"step": 0, "critique_text": payload["critique_text"], "error_tags": payload.get("error_tags", [])}]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise Sim2RealLoopError("VLM component output must include non-empty per_step")
    actions = {int(item["step"]): item.get("action", []) for item in manifest.get("actions", [])}
    observations = list(manifest.get("camera_observations", []))
    per_step: list[dict[str, Any]] = []
    for raw in raw_steps:
        step = int(raw.get("step", len(per_step)))
        tags = raw.get("error_tags", raw.get("tags", [])) or ["ok"]
        if isinstance(tags, str):
            tags = [tags]
        critique = str(raw.get("critique_text") or raw.get("critique") or raw.get("text") or "")
        if not critique:
            raise Sim2RealLoopError("VLM component per_step entries must include critique text")
        camera = raw.get("camera_observation")
        if not camera and 0 <= step < len(observations):
            camera = observations[step]
        per_step.append(
            {
                "step": step,
                "critique_text": critique,
                "error_tags": [str(tag) for tag in tags],
                "action": raw.get("action", actions.get(step, [])),
                "camera_observation": str(camera or f"camera-{step:03d}.ppm"),
            }
        )
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
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN", "NGC_API_KEY"):
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
    """Convert structured VLM critique JSON into a dense RL signal."""

    if evaluation.get("schema") != SCHEMA_VLM_EVAL:
        raise Sim2RealLoopError(
            f"unsupported VLM eval schema: {evaluation.get('schema')}"
        )
    raw_steps = evaluation.get("per_step")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise Sim2RealLoopError("VLM eval must include non-empty per_step")
    success = bool(evaluation["success"])
    step_items: list[dict[str, Any]] = []
    rewards: list[float] = []
    for raw_step in raw_steps:
        tags = [str(tag) for tag in raw_step.get("error_tags", [])] or ["ok"]
        severity = max(ERROR_SEVERITY.get(tag, 0.4) for tag in tags)
        success_bonus = 0.35 if success else -0.15
        reward = max(
            -1.0, min(1.0, success_bonus + 0.65 * (1.0 - severity) - 0.75 * severity)
        )
        rewards.append(reward)
        target = _merge_targets(tags)
        source_action = raw_step.get("action") or []
        credit = [
            round(abs(float(value)) * reward, 6)
            for value in source_action
            if isinstance(value, int | float)
        ]
        step_items.append(
            {
                "step": int(raw_step["step"]),
                "reward": round(reward, 6),
                "target": target,
                "critique_text": str(raw_step.get("critique_text") or ""),
                "error_tags": tags,
                "action_credit": {
                    "source_action": source_action,
                    "credit": credit,
                },
            }
        )
    baseline = sum(rewards) / float(len(rewards))
    for item in step_items:
        item["advantage"] = round(float(item["reward"]) - baseline, 6)
    return {
        "schema": SCHEMA_RL_SIGNAL,
        "rollout_id": str(evaluation["rollout_id"]),
        "source": "vlm",
        "success": success,
        "score": evaluation.get("score"),
        "per_step": step_items,
        "mapping_rules": signal_mapping_rules(),
    }


def signal_mapping_rules() -> dict[str, Any]:
    """Return documented VLM-critique to RL-signal conversion rules."""

    return {
        "dense_reward": (
            "reward = success_bonus + 0.65 * (1 - max_tag_severity) - "
            "0.75 * max_tag_severity, clipped to [-1, 1]."
        ),
        "success_bonus": {"success": 0.35, "failure": -0.15},
        "advantage": "per-step reward minus rollout mean reward",
        "per_action_credit": "abs(action_i) * step_reward for each source action dimension",
        "nl_corrective_targets": CORRECTIVE_TARGETS,
        "error_severity": ERROR_SEVERITY,
    }


def run_heldout_eval(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    inner_evidence: dict[str, Any],
    outer_iteration: int,
) -> dict[str, Any]:
    """Invoke the configured held-out eval component and write report.json."""

    output_dir = local_dir / "eval" / "heldout"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report.json"
    inner_path = output_dir / f"inner-evidence-outer-{outer_iteration:02d}.json"
    _write_json_artifact(inner_path, inner_evidence)
    env = _component_env(
        config,
        component="heldout_eval",
        output_json=output_path,
        extra={
            "NPA_SIM2REAL_HELDOUT_ENVS_DIR": str(local_dir / "envs" / "heldout"),
            "NPA_SIM2REAL_HELDOUT_ENV_COUNT": str(config.heldout_env_count),
            "NPA_SIM2REAL_INNER_EVIDENCE_JSON": str(inner_path),
            "NPA_SIM2REAL_THRESHOLD": str(config.threshold),
            "NPA_SIM2REAL_EVAL_IMAGE": config.eval_image,
            "NPA_SIM2REAL_ISAAC_IMAGE": config.isaac_image,
            "NPA_SIM2REAL_SIM_BACKEND": config.sim_backend,
            "NPA_SIM2REAL_ISAAC_TASK": config.isaac_task,
            "NPA_SIM2REAL_SCENE_SPEC_URI": config.scene_spec_uri,
            "NPA_SIM2REAL_ASSETS_URI": config.assets_uri,
        },
    )
    if config.byo_eval_command.strip():
        invocation = _run_component_command(
            config.byo_eval_command,
            cwd=local_dir,
            env=env,
            component="heldout_eval",
        )
    else:
        attempt_id = _component_attempt_id(
            config, "heldout_eval", f"outer-{outer_iteration:02d}"
        )
        if config.heldout_envs_uri:
            heldout_envs_uri = _normalized_s3_prefix(config.heldout_envs_uri)
        else:
            heldout_envs_uri = _upload_component_directory(
                config,
                local_dir / "envs" / "heldout",
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
    report = _normalize_heldout_report(
        payload,
        config=config,
        outer_iteration=outer_iteration,
        inner_evidence_uri=str(inner_path),
        invocation=invocation,
    )
    _write_json_artifact(output_path, report)
    return {**report, "report_uri": str(output_path)}


def _normalize_heldout_report(
    payload: dict[str, Any],
    *,
    config: Sim2RealLoopConfig,
    outer_iteration: int,
    inner_evidence_uri: str,
    invocation: dict[str, Any],
) -> dict[str, Any]:
    raw_items = payload.get("per_env") or payload.get("env_scores") or payload.get("scores")
    if isinstance(raw_items, dict):
        raw_items = [
            {"env_id": key, **(value if isinstance(value, dict) else {"score": value})}
            for key, value in raw_items.items()
        ]
    if not isinstance(raw_items, list) or not raw_items:
        raise Sim2RealLoopError("held-out eval component output must include non-empty per_env/env_scores")
    per_env: list[dict[str, Any]] = []
    passed = 0
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            item = {"score": item}
        score = max(0.0, min(1.0, float(item.get("score", item.get("success_score", 0.0)))))
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
    report = {
        "schema": SCHEMA_HELDOUT_REPORT,
        "stage": 10,
        "outer_iteration": outer_iteration,
        "status": "completed",
        "success_rate": round(success_rate, 6),
        "threshold": config.threshold,
        "per_env": per_env,
        "eval_image": config.eval_image,
        "byo_eval_command": _redact_command(config.byo_eval_command),
        "inner_evidence_uri": inner_evidence_uri,
        "component_invocation": _public_invocation(invocation),
        "generated_at": _utc_now(),
    }
    if "asset_provenance" in payload:
        report["asset_provenance"] = payload["asset_provenance"]
        report["asset_fallback_used"] = bool(
            payload.get(
                "asset_fallback_used",
                payload["asset_provenance"].get("asset_fallback_used", False),
            )
        )
    return report


def threshold_decision(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    heldout_report: dict[str, Any],
    outer_iteration: int,
) -> dict[str, Any]:
    """Apply Stage 11 threshold gate and write promote/loop-back artifacts."""

    success_rate = float(heldout_report["success_rate"])
    promoted = success_rate >= config.threshold
    checkpoint_dir = local_dir / "checkpoints" / "candidate"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_uri = str(checkpoint_dir)
    decision = {
        "schema": SCHEMA_THRESHOLD_DECISION,
        "stage": 11,
        "outer_iteration": outer_iteration,
        "success_rate": round(success_rate, 6),
        "threshold": config.threshold,
        "decision": "promote_checkpoint" if promoted else "loop_back_to_inner_loop",
        "checkpoint_uri": checkpoint_uri,
        "max_outer_iterations": config.outer_iterations,
        "remaining_outer_iterations": max(0, config.outer_iterations - outer_iteration),
    }
    if promoted:
        _write_json_artifact(
            checkpoint_dir / "candidate.json",
            {
                "schema": "npa.sim2real.candidate_checkpoint.v1",
                "run_id": config.run_id,
                "source": "vlm-rl-reference-update",
                "heldout_success_rate": round(success_rate, 6),
                "threshold": config.threshold,
                "promoted_at": _utc_now(),
            },
        )
    else:
        _write_json_artifact(
            local_dir / "outer_loop" / "loopback.json",
            {
                "schema": "npa.sim2real.loopback.v1",
                "from_stage": 11,
                "to_stage": 7,
                "reason": "heldout threshold not met",
                "decision": decision,
            },
        )
    path = local_dir / "outer_loop" / "decision.json"
    _write_json_artifact(path, decision)
    return {**decision, "decision_uri": str(path)}


def upload_run_artifacts(config: Sim2RealLoopConfig, local_dir: Path) -> dict[str, Any]:
    """Upload the run artifact tree to S3-compatible storage."""

    if not config.s3_bucket:
        return {"status": "skipped", "reason": "s3_bucket is not configured"}
    try:
        client = StorageClient.from_environment(endpoint_url=config.s3_endpoint)
        destination = f"{_artifact_root_uri(config)}/"
        uploaded = client.upload_directory(str(local_dir), destination)
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": f"S3 upload failed: {exc}",
            "next_action": "CONTINUE",
        }
    return {"status": "uploaded", "uri": uploaded}


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
    assets_uri: str = "",
    byo_mesh_uri: str = "",
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
        inner_path = root / "inner-evidence.json"
        output_path = root / "report.json"
        client = StorageClient.from_environment()
        client.download_path(heldout_envs_uri, str(env_dir))
        client.download_path(inner_evidence_uri, str(inner_path))
        inner_evidence = json.loads(inner_path.read_text(encoding="utf-8"))
        envs = _read_component_env_records(env_dir)
        if limit > 0:
            envs = envs[:limit]
        if not envs:
            raise Sim2RealLoopError("held-out component found no env records")
        if sim_backend == SIM_BACKEND_ISAAC:
            scene = _resolve_isaac_scene(
                scene_spec_uri=scene_spec_uri,
                assets_uri=assets_uri,
                byo_mesh_uri=byo_mesh_uri,
                dest_dir=root / "assets",
                client=client,
            )
        else:
            scene = _resolve_heldout_scene(
                scene_spec_uri=scene_spec_uri,
                assets_uri=assets_uri,
                byo_mesh_uri=byo_mesh_uri,
                dest_dir=root / "assets",
                client=client,
            )
        payload = _component_heldout_payload(
            envs,
            inner_evidence=inner_evidence,
            threshold=threshold,
            scene=scene,
            sim_backend=sim_backend,
            isaac_task=isaac_task,
        )
        _write_json_artifact(output_path, payload)
        client.upload_file(str(output_path), output_uri)
        if scene is not None:
            spec_path = root / "consumed-scene-spec.json"
            _write_json_artifact(spec_path, scene.provenance_block())
            client.upload_file(
                str(spec_path),
                _sibling_uri(output_uri, "consumed-scene-spec.json"),
            )
        print(
            json.dumps(
                {
                    "component": "heldout_eval",
                    "sim_backend": sim_backend,
                    "env_count": len(payload["per_env"]),
                    "output_uri": output_uri,
                    "asset_fallback_used": payload.get("asset_fallback_used"),
                },
                sort_keys=True,
            )
        )
        return payload


def _resolve_heldout_scene(
    *,
    scene_spec_uri: str,
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
        scene = scene_assets.parse_scene_spec(doc, source_uri=scene_spec_uri)
    else:
        scene = scene_assets.synthesize_scene_spec(byo_mesh_uri=mesh_uri)
    scene_assets.resolve_scene_assets(scene, dest_dir=dest_dir, client=client)
    return scene


def _resolve_isaac_scene(
    *,
    scene_spec_uri: str,
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
        scene = scene_assets.parse_scene_spec(doc, source_uri=scene_spec_uri)
    else:
        scene = scene_assets.synthesize_scene_spec(byo_mesh_uri=mesh_uri)
    scene_assets.resolve_scene_assets(scene, dest_dir=dest_dir, client=client)
    return scene


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
    resolved_model = _resolve_cosmos_reason_model_id(model)
    task_description = _task_description_from_manifest(manifest)
    payload = _run_cosmos_reason_vlm(
        model_id=resolved_model,
        image_paths=image_paths,
        actions=actions,
        task_description=task_description,
        rollout_id=rollout_id or str(manifest.get("rollout_id") or "rollout"),
        threshold=threshold,
    )
    payload["component_source"] = "cosmos_reason_vlm"
    payload["model"] = resolved_model
    payload["task_description"] = task_description
    payload["frame_count"] = len(image_paths)
    return payload


def _component_heldout_payload(
    envs: list[dict[str, Any]],
    *,
    inner_evidence: dict[str, Any],
    threshold: float,
    scene: Any = None,
    sim_backend: str = DEFAULT_SIM_BACKEND,
    isaac_task: str = DEFAULT_ISAAC_TASK,
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
            isaac_task=isaac_task,
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
        )
        payload = {
            "schema": SCHEMA_HELDOUT_REPORT,
            "per_env": per_env,
            "sim_backend": SIM_BACKEND_GENESIS,
            "component_source": "genesis_rollout",
            "rollout_backend": "npa.genesis.env_pick_place.FrankaPickPlaceEnv",
            "policy_source": "inner_evidence_adapter",
        }
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


def _resolve_cosmos_reason_model_id(model: str) -> str:
    candidate = str(model or "").strip()
    if candidate in REFERENCE_VLM_ALIASES:
        candidate = os.environ.get("NPA_COSMOS_REASON_MODEL_ID", DEFAULT_REFERENCE_VLM_MODEL)
    return candidate


def _task_description_from_manifest(manifest: dict[str, Any]) -> str:
    for key in ("task_description", "task", "instruction", "prompt"):
        value = str(manifest.get(key) or "").strip()
        if value:
            return value
    return (
        "Evaluate whether the robot rollout completes the manipulation task. "
        "Use the camera frames and the listed actions to judge physical success, "
        "stability, target alignment, and contact mistakes."
    )


def _run_cosmos_reason_vlm(
    *,
    model_id: str,
    image_paths: list[Path],
    actions: list[dict[str, Any]],
    task_description: str,
    rollout_id: str,
    threshold: float,
) -> dict[str, Any]:
    """Run real Cosmos-Reason1/Qwen-VL inference and parse its JSON judgment."""

    try:
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:
        raise Sim2RealLoopError(
            "Cosmos-Reason1 VLM inference requires torch, Pillow, transformers, "
            f"and qwen-vl-utils in the image: {exc}"
        ) from exc

    if not image_paths:
        raise Sim2RealLoopError("Cosmos-Reason1 inference requires at least one frame")
    if not torch.cuda.is_available():
        raise Sim2RealLoopError("Cosmos-Reason1 inference requires a CUDA GPU")

    cache_dir = os.environ.get("NPA_COSMOS_REASON_CACHE", DEFAULT_COSMOS_REASON_CACHE)
    max_frames = int(os.environ.get("NPA_COSMOS_REASON_MAX_FRAMES", "8"))
    selected_paths = image_paths[:max(1, max_frames)]
    for path in selected_paths:
        with Image.open(path) as img:
            img.verify()

    prompt = _cosmos_reason_prompt(
        task_description=task_description,
        actions=actions,
        frame_names=[path.name for path in selected_paths],
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image", "image": str(path)} for path in selected_paths)
    messages = [{"role": "user", "content": content}]

    print(
        json.dumps(
            {
                "component": "vlm_eval",
                "event": "cosmos_reason_inference_start",
                "model": model_id,
                "frames": [path.name for path in selected_paths],
            },
            sort_keys=True,
        )
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    first_device = next(model.parameters()).device
    inputs = inputs.to(first_device)
    max_new_tokens = int(os.environ.get("NPA_COSMOS_REASON_MAX_NEW_TOKENS", "768"))
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated, strict=False)
    ]
    model_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    payload = _parse_cosmos_reason_output(
        model_text,
        actions=actions,
        rollout_id=rollout_id,
        threshold=threshold,
    )
    payload["raw_model_output_excerpt"] = _component_excerpt(model_text, limit=900)
    print(
        json.dumps(
            {
                "component": "vlm_eval",
                "event": "cosmos_reason_inference_complete",
                "model": model_id,
                "rollout_id": payload["rollout_id"],
                "score": payload["score"],
                "success": payload["success"],
                "tags": sorted({tag for step in payload["per_step"] for tag in step["error_tags"]}),
            },
            sort_keys=True,
        )
    )
    return payload


def _cosmos_reason_prompt(
    *,
    task_description: str,
    actions: list[dict[str, Any]],
    frame_names: list[str],
) -> str:
    action_excerpt = json.dumps(actions[:16], sort_keys=True)
    return (
        "You are NVIDIA Cosmos-Reason1 evaluating a physical robot rollout.\n"
        f"Task description: {task_description}\n"
        f"Frame order: {frame_names}\n"
        f"Actions by step: {action_excerpt}\n"
        "Return JSON only. The JSON must contain: success (boolean), "
        "score (number from 0 to 1), summary (natural-language critique), and "
        "per_step (array of objects with step, critique_text, error_tags, "
        "camera_observation). Use only these error tags when applicable: "
        "collision, missed_target, unstable, late_grasp, minor_alignment, ok. "
        "Judge actual visual rollout behavior, not metadata or requested actions."
    )


def _parse_cosmos_reason_output(
    model_text: str,
    *,
    actions: list[dict[str, Any]],
    rollout_id: str,
    threshold: float,
) -> dict[str, Any]:
    payload = _json_object_from_text(model_text)
    if payload is None:
        payload = _parse_unstructured_vlm_output(model_text)
    if "score" not in payload:
        raise Sim2RealLoopError(
            "Cosmos-Reason1 output did not include a numeric score"
        )
    score = max(0.0, min(1.0, float(payload["score"])))
    success = bool(payload.get("success", score >= threshold))
    raw_steps = payload.get("per_step") or payload.get("steps") or []
    if not raw_steps:
        critique = str(
            payload.get("summary")
            or payload.get("critique")
            or payload.get("critique_text")
            or model_text
        ).strip()
        tags = payload.get("error_tags") or _tags_from_text(critique)
        # The model returned no per-step breakdown, so the single summary critique
        # is broadcast across every action step. Mark it as such so a degenerate
        # (all-identical) per-step signal is visible rather than masquerading as
        # genuine per-step granularity.
        raw_steps = [
            {
                "step": int(action.get("step", index)),
                "critique_text": critique,
                "error_tags": tags,
                "critique_source": "summary_broadcast",
                "camera_observation": f"camera-{int(action.get('step', index)):03d}.ppm",
            }
            for index, action in enumerate(actions)
        ]
    per_step: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raw = {"critique_text": str(raw)}
        step = int(raw.get("step", index))
        tags = raw.get("error_tags") or raw.get("tags") or _tags_from_text(str(raw))
        if isinstance(tags, str):
            tags = [tags]
        normalized_tags = _normalize_error_tags(tags)
        critique = str(
            raw.get("critique_text")
            or raw.get("critique")
            or raw.get("text")
            or payload.get("summary")
            or ""
        ).strip()
        if not critique:
            raise Sim2RealLoopError("Cosmos-Reason1 per_step output lacks critique text")
        per_step.append(
            {
                "step": step,
                "critique_text": critique,
                "error_tags": normalized_tags,
                "action": actions[index].get("action", []) if index < len(actions) else [],
                "camera_observation": str(
                    raw.get("camera_observation") or f"camera-{step:03d}.ppm"
                ),
            }
        )
    return {
        "schema": SCHEMA_VLM_EVAL,
        "rollout_id": str(payload.get("rollout_id") or rollout_id),
        "success": success,
        "score": round(score, 6),
        "per_step": per_step,
        "summary": str(payload.get("summary") or payload.get("critique") or "").strip(),
    }


def _json_object_from_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_unstructured_vlm_output(text: str) -> dict[str, Any]:
    lowered = text.lower()
    score_match = re.search(r"(?:score|confidence|rating)\D+([01](?:\.\d+)?)", lowered)
    if not score_match:
        raise Sim2RealLoopError("Cosmos-Reason1 output was not parseable JSON")
    score = float(score_match.group(1))
    if "success" in lowered or "pass" in lowered:
        success = True
    elif "fail" in lowered or "unsuccess" in lowered:
        success = False
    else:
        success = score >= DEFAULT_THRESHOLD
    return {
        "success": success,
        "score": score,
        "summary": text.strip(),
        "error_tags": _tags_from_text(text),
    }


def _tags_from_text(text: str) -> list[str]:
    lowered = text.lower().replace("-", "_").replace(" ", "_")
    tags = [tag for tag in ERROR_SEVERITY if tag != "ok" and tag in lowered]
    if not tags and re.search(r"\b(ok|success|stable|complete)\b", text.lower()):
        tags = ["ok"]
    return tags or ["minor_alignment"]


def _normalize_error_tags(tags: list[Any]) -> list[str]:
    known = set(ERROR_SEVERITY)
    normalized = []
    for tag in tags:
        value = str(tag).strip().lower().replace("-", "_").replace(" ", "_")
        normalized.append(value if value in known else "minor_alignment")
    return normalized or ["minor_alignment"]


def _run_genesis_heldout_rollouts(
    envs: list[dict[str, Any]],
    *,
    inner_evidence: dict[str, Any],
    threshold: float,
    scene: Any = None,
) -> list[dict[str, Any]]:
    """Run the trained adapter policy through real Genesis held-out episodes.

    When ``scene`` (a parsed ``npa.genesis.scene_assets.SceneSpec`` with
    resolved local asset paths) is provided, the manipulated object(s) are
    built from it (mesh / primitive) instead of the default red Box. The
    SceneSpec objects' ``loaded`` provenance flags are set as a side effect of
    building the env, so the caller can prove the requested mesh loaded.
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
            action_scale=float(os.environ.get("NPA_SIM2REAL_GENESIS_ACTION_SCALE", "0.045")),
            scene_spec=scene,
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
        obs = env.reset()
        active = torch.ones(len(batch), device="cuda", dtype=torch.bool)
        success = torch.zeros(len(batch), device="cuda", dtype=torch.bool)
        steps_done = torch.zeros(len(batch), device="cuda", dtype=torch.long)
        max_reward = torch.full((len(batch),), -1.0e9, device="cuda")
        final_distance = torch.full((len(batch),), 1.0e9, device="cuda")
        for step in range(max_steps):
            actions = _adapter_policy_actions(obs, adapter, step=step)
            obs, reward, done, info = env.step(actions)
            distance = torch.norm(obs["object_pose"][:, :3] - obs["goal_position"], dim=-1)
            final_distance = torch.where(active, distance, final_distance)
            max_reward = torch.where(active, torch.maximum(max_reward, reward), max_reward)
            just_done = active & done
            if bool(just_done.any()):
                success = torch.where(just_done, info["success"].bool(), success)
                steps_done = torch.where(just_done, torch.full_like(steps_done, step + 1), steps_done)
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
                    "env_id": str(env_record.get("env_id") or f"heldout-{start + index:04d}"),
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
            from isaaclab.sim.converters import MeshConverter, MeshConverterCfg

            cfg = MeshConverterCfg(
                asset_path=str(src),
                usd_dir=str(work_dir),
                usd_file_name=f"{src.stem}.usd",
                force_usd_conversion=True,
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
    )


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


def _isaac_adapter_actions(action_dim: int, adapter: dict[str, Any], *, n_envs: int, step: int, device: str):
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
    explore = 0.15 * decay * torch.sin(
        torch.arange(action_dim, device=device, dtype=torch.float32) * (step + 1) * 0.37
    )
    actions = actions + explore.unsqueeze(0)
    if action_dim >= 1:
        # Last channel = gripper: open early, close as the episode progresses.
        actions[:, -1] = 1.0 if step < 30 else -1.0
    return torch.clamp(actions, -1.0, 1.0)


def _run_isaac_heldout_rollouts(
    envs: list[dict[str, Any]],
    *,
    inner_evidence: dict[str, Any],
    threshold: float,
    scene: Any = None,
    isaac_task: str = DEFAULT_ISAAC_TASK,
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

    simulation_app = AppLauncher(headless=True).app
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

    adapter = _policy_adapter_from_inner_evidence(inner_evidence)
    batch_size = max(1, int(os.environ.get("NPA_SIM2REAL_ISAAC_BATCH_SIZE", "8")))
    max_steps = max(1, int(os.environ.get("NPA_SIM2REAL_ISAAC_MAX_STEPS", "120")))
    reward_norm = float(os.environ.get("NPA_SIM2REAL_ISAAC_REWARD_NORM", "20.0"))
    success_dist = float(os.environ.get("NPA_SIM2REAL_ISAAC_SUCCESS_DIST", "0.05"))
    per_env: list[dict[str, Any]] = []
    try:
        for start in range(0, len(envs), batch_size):
            batch = envs[start : start + batch_size]
            seed = int(batch[0].get("seed") or (42 + start))
            torch.manual_seed(seed)
            env_cfg = parse_env_cfg(isaac_task, device=device, num_envs=len(batch))
            if usd_override:
                _set_isaac_object_usd(env_cfg, usd_override, scale=manip_scale)
            env = gym.make(isaac_task, cfg=env_cfg)
            action_dim = int(env.action_space.shape[-1])
            obs, _ = env.reset()
            n = len(batch)
            max_reward = torch.full((n,), -1.0e9, device=device)
            final_distance = torch.full((n,), 1.0e9, device=device)
            for step in range(max_steps):
                actions = _isaac_adapter_actions(
                    action_dim, adapter, n_envs=n, step=step, device=device
                )
                obs, reward, terminated, truncated, _ = env.step(actions)
                reward_t = torch.as_tensor(reward, device=device, dtype=torch.float32).reshape(-1)
                max_reward = torch.maximum(max_reward, reward_t)
                final_distance = _isaac_goal_distance(env.unwrapped).reshape(-1).detach()
                done = torch.as_tensor(terminated, device=device).reshape(-1) | torch.as_tensor(
                    truncated, device=device
                ).reshape(-1)
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
    finally:
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001
            pass
    return per_env


def _policy_adapter_from_inner_evidence(inner_evidence: dict[str, Any]) -> dict[str, Any]:
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
    bias = torch.tensor(bias_values[:3], device=ee_pos.device, dtype=ee_pos.dtype).unsqueeze(0)
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


def _find_component_input_file(root: Path, filename: str) -> Path:
    if root.is_file() and root.name == filename:
        return root
    candidates = sorted(root.rglob(filename))
    if not candidates:
        raise Sim2RealLoopError(f"component input did not include {filename}")
    return candidates[0]


def _read_component_env_records(root: Path) -> list[dict[str, Any]]:
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


def main(argv: list[str] | None = None) -> int:
    """Module CLI for raw SkyPilot YAML and local smoke runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    full = subparsers.add_parser(
        "full-loop", help="Run the full Stage 1-13 Sim2Real workflow."
    )
    _add_common_args(full)
    inner = subparsers.add_parser(
        "inner-loop", help="Run only the VLM-to-RL inner loop."
    )
    _add_common_args(inner)
    convert = subparsers.add_parser(
        "convert-signal", help="Convert one VLM eval JSON to RL signal JSON."
    )
    convert.add_argument("--vlm-json", type=Path, required=True)
    convert.add_argument("--output-json", type=Path, required=True)
    component_vlm = subparsers.add_parser(
        "component-vlm-eval", help="Run one sibling-image VLM component contract."
    )
    component_vlm.add_argument("--input-uri", required=True)
    component_vlm.add_argument("--output-uri", required=True)
    component_vlm.add_argument("--rollout-id", default="")
    component_vlm.add_argument("--model", default=DEFAULT_REFERENCE_VLM_MODEL)
    component_vlm.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    component_heldout = subparsers.add_parser(
        "component-heldout-eval",
        help="Run one sibling-image held-out eval component contract.",
    )
    component_heldout.add_argument("--heldout-envs-uri", required=True)
    component_heldout.add_argument("--inner-evidence-uri", required=True)
    component_heldout.add_argument("--output-uri", required=True)
    component_heldout.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    component_heldout.add_argument("--limit", type=int, default=0)
    component_heldout.add_argument("--scene-spec-uri", default="")
    component_heldout.add_argument("--assets-uri", default="")
    component_heldout.add_argument("--byo-mesh-uri", default="")
    component_heldout.add_argument(
        "--sim-backend",
        default=os.environ.get("NPA_SIM2REAL_SIM_BACKEND", DEFAULT_SIM_BACKEND),
        choices=list(SIM_BACKENDS),
    )
    component_heldout.add_argument(
        "--isaac-task",
        default=os.environ.get("NPA_SIM2REAL_ISAAC_TASK", DEFAULT_ISAAC_TASK),
    )
    args = parser.parse_args(argv)

    if args.command == "convert-signal":
        payload = json.loads(args.vlm_json.read_text(encoding="utf-8"))
        _write_json_artifact(args.output_json, convert_vlm_eval_to_rl_signal(payload))
        return 0
    if args.command == "component-vlm-eval":
        run_vlm_eval_component_from_s3(
            input_uri=args.input_uri,
            output_uri=args.output_uri,
            rollout_id=args.rollout_id,
            model=args.model,
            threshold=args.threshold,
        )
        return 0
    if args.command == "component-heldout-eval":
        run_heldout_eval_component_from_s3(
            heldout_envs_uri=args.heldout_envs_uri,
            inner_evidence_uri=args.inner_evidence_uri,
            output_uri=args.output_uri,
            threshold=args.threshold,
            limit=args.limit,
            scene_spec_uri=args.scene_spec_uri,
            assets_uri=args.assets_uri,
            byo_mesh_uri=args.byo_mesh_uri,
            sim_backend=args.sim_backend,
            isaac_task=args.isaac_task,
        )
        return 0

    config = build_config_from_env(
        run_id=args.run_id,
        output_dir=args.output_dir,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        s3_endpoint=args.s3_endpoint,
        trigger_dataset_uri=args.trigger_dataset_uri,
        trigger_dataset_id=args.trigger_dataset_id,
        action_rollouts_uri=args.action_rollouts_uri,
        train_envs_uri=args.train_envs_uri,
        heldout_envs_uri=args.heldout_envs_uri,
        assets_uri=args.assets_uri,
        scene_spec_uri=args.scene_spec_uri,
        augment_image=args.augment_image,
        policy_image=args.policy_image,
        trainer_image=args.trainer_image,
        vlm_image=args.vlm_image,
        eval_image=args.eval_image,
        isaac_image=args.isaac_image,
        sim_backend=args.sim_backend,
        isaac_task=args.isaac_task,
        vlm_model=args.vlm_model,
        threshold=args.threshold,
        inner_iterations=args.inner_iterations,
        outer_iterations=args.outer_iterations,
        loop_of_loops_iterations=args.loop_of_loops_iterations,
        rollout_count=args.rollout_count,
        steps_per_rollout=args.steps_per_rollout,
        heldout_env_count=args.heldout_env_count,
        seed=args.seed,
        upload_artifacts=args.upload_artifacts,
        no_guardrails=args.no_guardrails,
        signal_loss_weight=args.signal_loss_weight,
        learning_rate=args.learning_rate,
        byo_signal_converter=args.byo_signal_converter,
        byo_trainer_command=args.byo_trainer_command,
        byo_vlm_command=args.byo_vlm_command,
        byo_eval_command=args.byo_eval_command,
        k8s_namespace=args.k8s_namespace,
        k8s_service_account=args.k8s_service_account,
        k8s_image_pull_secrets=args.k8s_image_pull_secrets,
        k8s_env_secret_names=args.k8s_env_secret_names,
        k8s_gpu_resource=args.k8s_gpu_resource,
        k8s_gpu_product=args.k8s_gpu_product,
        k8s_kubeconfig=args.k8s_kubeconfig,
        k8s_context=args.k8s_context,
        k8s_job_timeout_s=args.k8s_job_timeout_s,
        source_repo=args.source_repo,
        source_ref=args.source_ref,
        heldout_eval_limit=args.heldout_eval_limit,
    )
    if args.command == "full-loop":
        report = run_full_loop(config)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "inner-loop":
        config.validate()
        local_dir = config.output_dir or Path(
            tempfile.mkdtemp(prefix=f"npa-{config.run_id}-")
        )
        evidence = run_inner_loop(config, local_dir=local_dir, initial_quality=0.38)
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0
    return 2


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-id", default=os.environ.get("NPA_SIM2REAL_RUN_ID", new_run_id())
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--s3-bucket",
        default=os.environ.get("NPA_SIM2REAL_BUCKET", os.environ.get("S3_BUCKET", "")),
    )
    parser.add_argument(
        "--s3-prefix", default=os.environ.get("NPA_SIM2REAL_PREFIX", DEFAULT_PREFIX)
    )
    parser.add_argument(
        "--s3-endpoint", default=os.environ.get("AWS_ENDPOINT_URL", DEFAULT_S3_ENDPOINT)
    )
    parser.add_argument(
        "--trigger-dataset-uri",
        default=os.environ.get("NPA_SIM2REAL_TRIGGER_DATASET_URI", ""),
    )
    parser.add_argument(
        "--trigger-dataset-id",
        default=os.environ.get(
            "NPA_SIM2REAL_TRIGGER_DATASET_ID", DEFAULT_LEROBOT_DATASET_ID
        ),
    )
    parser.add_argument(
        "--action-rollouts-uri", default=os.environ.get("ACTION_ROLLOUTS_URI", "")
    )
    parser.add_argument(
        "--train-envs-uri", default=os.environ.get("TRAIN_ENVS_URI", "")
    )
    parser.add_argument(
        "--heldout-envs-uri", default=os.environ.get("HELDOUT_ENVS_URI", "")
    )
    parser.add_argument("--assets-uri", default=os.environ.get("ASSETS_URI", ""))
    parser.add_argument(
        "--scene-spec-uri", default=os.environ.get("SCENE_SPEC_URI", "")
    )
    parser.add_argument("--augment-image", default=os.environ.get("AUGMENT_IMAGE", ""))
    parser.add_argument("--policy-image", default=os.environ.get("POLICY_IMAGE", ""))
    parser.add_argument("--trainer-image", default=os.environ.get("TRAINER_IMAGE", ""))
    parser.add_argument("--vlm-image", default=os.environ.get("VLM_IMAGE", ""))
    parser.add_argument("--eval-image", default=os.environ.get("EVAL_IMAGE", ""))
    parser.add_argument("--isaac-image", default=os.environ.get("ISAAC_IMAGE", ""))
    parser.add_argument(
        "--sim-backend",
        default=os.environ.get("NPA_SIM2REAL_SIM_BACKEND", DEFAULT_SIM_BACKEND),
        choices=list(SIM_BACKENDS),
    )
    parser.add_argument(
        "--isaac-task",
        default=os.environ.get("NPA_SIM2REAL_ISAAC_TASK", DEFAULT_ISAAC_TASK),
    )
    parser.add_argument(
        "--vlm-model", default=os.environ.get("VLM_MODEL", DEFAULT_REFERENCE_VLM_MODEL)
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ.get("SUCCESS_THRESHOLD", DEFAULT_THRESHOLD)),
    )
    parser.add_argument(
        "--inner-iterations", type=int, default=DEFAULT_INNER_ITERATIONS
    )
    parser.add_argument(
        "--outer-iterations", type=int, default=DEFAULT_OUTER_ITERATIONS
    )
    parser.add_argument(
        "--loop-of-loops-iterations", type=int, default=DEFAULT_LOOP_OF_LOOPS_ITERATIONS
    )
    parser.add_argument("--rollout-count", type=int, default=DEFAULT_ROLLOUT_COUNT)
    parser.add_argument(
        "--steps-per-rollout", type=int, default=DEFAULT_STEPS_PER_ROLLOUT
    )
    parser.add_argument("--heldout-env-count", type=int, default=DEFAULT_HELDOUT_ENVS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload-artifacts", action="store_true")
    parser.add_argument("--no-guardrails", action="store_true")
    parser.add_argument("--signal-loss-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--byo-signal-converter", default="")
    parser.add_argument("--byo-trainer-command", default="")
    parser.add_argument("--byo-vlm-command", default="")
    parser.add_argument("--byo-eval-command", default="")
    parser.add_argument(
        "--k8s-namespace",
        default=os.environ.get("NPA_SIM2REAL_K8S_NAMESPACE", ""),
    )
    parser.add_argument(
        "--k8s-service-account",
        default=os.environ.get("NPA_SIM2REAL_K8S_SERVICE_ACCOUNT", "agent-sa"),
    )
    parser.add_argument(
        "--k8s-image-pull-secrets",
        default=os.environ.get(
            "NPA_SIM2REAL_K8S_IMAGE_PULL_SECRETS",
            "agent-sa,ngc-nvcr-imagepullsecret,npa-nebius-registry",
        ),
    )
    parser.add_argument(
        "--k8s-env-secret-names",
        default=os.environ.get(
            "NPA_SIM2REAL_K8S_ENV_SECRET_NAMES",
            "hf-ngc-tokens,npa-storage-credentials",
        ),
    )
    parser.add_argument(
        "--k8s-gpu-resource",
        default=os.environ.get("NPA_SIM2REAL_K8S_GPU_RESOURCE", "nvidia.com/gpu"),
    )
    parser.add_argument(
        "--k8s-gpu-product",
        default=os.environ.get(
            "NPA_SIM2REAL_K8S_GPU_PRODUCT",
            "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        ),
    )
    parser.add_argument(
        "--k8s-kubeconfig",
        default=os.environ.get("NPA_SIM2REAL_KUBECONFIG", os.environ.get("KUBECONFIG", "")),
    )
    parser.add_argument(
        "--k8s-context",
        default=os.environ.get("NPA_SIM2REAL_K8S_CONTEXT", ""),
    )
    parser.add_argument(
        "--k8s-job-timeout-s",
        type=int,
        default=int(os.environ.get("NPA_SIM2REAL_K8S_JOB_TIMEOUT_S", "7200")),
    )
    parser.add_argument("--source-repo", default=os.environ.get("NPA_SOURCE_REPO", ""))
    parser.add_argument("--source-ref", default=os.environ.get("NPA_SOURCE_REF", ""))
    parser.add_argument(
        "--heldout-eval-limit",
        type=int,
        default=int(os.environ.get("NPA_SIM2REAL_HELDOUT_EVAL_LIMIT", "0")),
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


def _write_json_artifact(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"path": str(path), "payload": payload}


def _write_env_manifest(root: Path, *, count: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    envs = [
        {
            "env_id": f"env-{index:04d}",
            "seed": rng.randrange(1, 2**31 - 1),
            "asset_ref": f"asset-{index:04d}",
            "physics": {
                "friction": round(0.5 + rng.random() * 0.5, 4),
                "mass_scale": round(0.85 + rng.random() * 0.3, 4),
                "lighting": round(0.4 + rng.random() * 0.5, 4),
            },
        }
        for index in range(count)
    ]
    return _write_json_artifact(
        root / "manifest.json",
        {"schema": "npa.sim2real.env_manifest.v1", "stage": 4, "envs": envs},
    )


def _write_train_heldout_split(
    root: Path,
    *,
    raw_envs: dict[str, Any],
    train_count: int,
    heldout_count: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    envs = list(raw_envs["payload"]["envs"])
    expected = train_count + heldout_count
    if len(envs) != expected:
        raise Sim2RealLoopError(
            f"raw env count {len(envs)} must equal train+heldout count {expected}"
        )
    rng = random.Random(seed)
    rng.shuffle(envs)
    train = envs[:train_count]
    heldout = envs[train_count:train_count + heldout_count]
    if len(train) != train_count or len(heldout) != heldout_count:
        raise Sim2RealLoopError("train/heldout split did not preserve requested counts")
    train_record = _write_json_artifact(
        root / "train" / "manifest.json",
        {
            "schema": "npa.sim2real.env_split.v1",
            "stage": 5,
            "split": "train",
            "envs": train,
        },
    )
    heldout_record = _write_json_artifact(
        root / "heldout" / "manifest.json",
        {
            "schema": "npa.sim2real.env_split.v1",
            "stage": 5,
            "split": "heldout",
            "envs": heldout,
        },
    )
    return train_record, heldout_record


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


def _tags_for_quality(quality: float, *, step: int) -> list[str]:
    if quality < 0.45:
        return ["missed_target", "unstable"] if step % 2 == 0 else ["late_grasp"]
    if quality < 0.65:
        return ["minor_alignment"] if step % 2 == 0 else ["late_grasp"]
    if quality < 0.8:
        return ["minor_alignment"]
    return ["ok"]


def _critique_for_tags(tags: list[str], *, quality: float) -> str:
    if tags == ["ok"]:
        return f"Step is stable; estimated rollout quality {quality:.2f}."
    corrections = [
        CORRECTIVE_TARGETS.get(tag, CORRECTIVE_TARGETS["minor_alignment"])[
            "nl_correction"
        ]
        for tag in tags
    ]
    return " ".join(corrections)


def _merge_targets(tags: list[str]) -> dict[str, Any]:
    corrections = [
        CORRECTIVE_TARGETS.get(tag, CORRECTIVE_TARGETS["minor_alignment"])
        for tag in tags
    ]
    action_dim = max(len(item["action_delta"]) for item in corrections)
    merged = [0.0 for _ in range(action_dim)]
    for item in corrections:
        for index, value in enumerate(item["action_delta"]):
            merged[index] += float(value) / float(len(corrections))
    return {
        "nl_correction": " ".join(str(item["nl_correction"]) for item in corrections),
        "action_delta": [round(value, 6) for value in merged],
    }


def _signal_mean_reward(signal: dict[str, Any]) -> float:
    steps = signal.get("per_step") or []
    return sum(float(step["reward"]) for step in steps) / float(len(steps))


def _heldout_env_score(
    distance_score: float, reward_score: float, *, env_success: bool
) -> float:
    """Map per-env distance/reward to a continuous held-out score.

    Successful and failed envs occupy separate bands, but the score stays
    continuous in the env's own distance/reward so the held-out report keeps a
    gradient instead of collapsing to a flat ``1.0`` across every env (which
    produced an uninformative, incoherent signal).
    """

    quality = max(0.0, min(1.0, 0.7 * distance_score + 0.3 * reward_score))
    if env_success:
        return round(0.75 + 0.25 * quality, 6)
    return round(0.6 * quality, 6)


def _signal_diversity_report(signals: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize cross-rollout diversity of the VLM->RL signal.

    A genuine signal varies across rollouts; a single distinct score/reward
    across every rollout is the degenerate ("hollow") pattern. These metrics
    (``distinct_scores``, ``coherent``) are emitted into loop evidence so a run
    is self-describing instead of requiring an external validator to infer them.
    """

    scores = [round(float(signal.get("score") or 0.0), 4) for signal in signals]
    mean_rewards = [round(_signal_mean_reward(signal), 4) for signal in signals]
    distinct_scores = sorted({score for score in scores})
    distinct_rewards = sorted({reward for reward in mean_rewards})
    total = len(signals)
    coherent = total > 1 and len(distinct_scores) > 1 and len(distinct_rewards) > 1
    return {
        "total_rollouts": total,
        "distinct_scores": len(distinct_scores),
        "distinct_mean_rewards": len(distinct_rewards),
        "score_values": distinct_scores,
        "mean_reward_values": distinct_rewards,
        "coherent": coherent,
        "degenerate": not coherent,
    }


def _image_pull_policy(image: str) -> str:
    """Choose the imagePullPolicy for a sibling component image.

    Provenance-sensitive ``-genuine-`` builds are pulled fresh so a stale image
    cached under the same tag cannot silently masquerade as the genuine build.
    A digest-pinned reference (``@sha256:``) is already immutable.
    """

    override = os.environ.get("NPA_SIM2REAL_IMAGE_PULL_POLICY", "").strip()
    if override:
        return override
    if "@sha256:" in image:
        return "IfNotPresent"
    tag = image.rsplit(":", 1)[-1] if ":" in image.rsplit("/", 1)[-1] else ""
    if "genuine" in tag:
        return "Always"
    return "IfNotPresent"


def _write_ppm(path: Path, *, red: int, green: int, blue: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 32
    height = 32
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    pixel = bytes(
        [max(0, min(255, red)), max(0, min(255, green)), max(0, min(255, blue))]
    )
    path.write_bytes(header + pixel * width * height)


def _redacted_config(config: Sim2RealLoopConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir) if config.output_dir else None
    return payload


def _artifact_root_uri(config: Sim2RealLoopConfig) -> str:
    parts = [part for part in (config.s3_prefix.strip("/"), config.run_id) if part]
    return f"s3://{config.s3_bucket}/{'/'.join(parts)}"


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise Sim2RealLoopError(f"expected s3:// URI, got {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


__all__ = [
    "SCHEMA_HELDOUT_REPORT",
    "SCHEMA_RL_SIGNAL",
    "SCHEMA_THRESHOLD_DECISION",
    "SCHEMA_VLM_EVAL",
    "DEFAULT_LEROBOT_DATASET_ID",
    "DEFAULT_ISAAC_TASK",
    "SIM_BACKENDS",
    "SIM_BACKEND_GENESIS",
    "SIM_BACKEND_ISAAC",
    "Sim2RealLoopConfig",
    "Sim2RealLoopError",
    "artifact_uris",
    "build_config_from_env",
    "byo_seams",
    "convert_vlm_eval_to_rl_signal",
    "default_envgen_image",
    "default_eval_image",
    "default_isaac_image",
    "default_policy_image",
    "default_trainer_image",
    "default_vlm_image",
    "evaluate_rollout_with_vlm",
    "generate_action_rollouts",
    "new_run_id",
    "run_full_loop",
    "run_heldout_eval_component_from_s3",
    "run_heldout_eval",
    "run_inner_loop",
    "run_vlm_eval_component_from_s3",
    "signal_mapping_rules",
    "threshold_decision",
]


if __name__ == "__main__":
    sys.exit(main())
