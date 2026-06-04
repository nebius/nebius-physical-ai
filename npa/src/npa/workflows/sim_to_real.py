"""Tiered sim-to-real training and evaluation pipeline contracts.

This module keeps the generic platform pipeline honest: it builds a validated
local/structural spine and records live infrastructure gaps as component tiers
instead of treating unavailable GPU, S3, simulator, or VLM access as success.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from npa.clients.storage import StorageClient
from npa.deploy.images import container_image_for_tool, supported_tool_version
from npa.viz.adapters.lerobot_to_rerun import RerunAdapterError, lerobot_dataset_logical_to_rerun
from npa.workbench.lerobot.policy_container import (
    DEFAULT_POLICY_TYPE,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_TRAIN_NUM_WORKERS,
    PolicyContainerError,
    assert_lerobot_importable,
    parse_feedback_batch,
    run_feedback_training_step,
    run_lerobot_eval,
    run_lerobot_training,
    validate_lerobot_checkpoint,
)
from npa.workflows.lerobot_dataset import (
    DEFAULT_PUBLIC_LEROBOT_LICENSE,
    DEFAULT_PUBLIC_LEROBOT_REPO,
    DEFAULT_PUBLIC_LEROBOT_REVISION,
    LeRobotDatasetError,
    default_public_dataset_uri,
    default_staged_dataset_uri,
    download_public_lerobot_dataset,
    materialize_lerobot_dataset,
    resolve_dataset_source,
    seeded_episode_split,
    stage_dataset_to_s3,
    summarize_lerobot_dataset,
    write_episode_split_manifest,
)


DEFAULT_S3_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
DEFAULT_FEEDBACK_SOURCE = "rollout"
DEFAULT_SIM_BACKEND = "lerobot-dataset"
DEFAULT_EVAL_BACKEND = "pusht"
DEFAULT_VLM_EVAL_BACKEND = "api"
DEFAULT_VLM_EVAL_MODEL = "vlm-eval"
DEFAULT_SPLIT_FRACTION = 0.8
DEFAULT_THRESHOLD = 0.75
DEFAULT_RERUN_MAX_FRAMES_PER_EPISODE = 32
DEFAULT_MAX_TRAINING_ITERATIONS = 3
DEFAULT_TRAIN_STEP_BUDGET = 6000
DEFAULT_MIN_EVAL_IMPROVEMENT = 0.0
APPLICATION_ID = "npa_sim_to_real_pipeline"


class SimToRealError(Exception):
    """Raised when a sim-to-real structural contract is invalid."""


class Tier(str, Enum):
    """Validation tier for a component in the pipeline report."""

    WORKS = "WORKS"
    PARTIAL = "PARTIAL"
    SEAM = "SEAM"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class ComponentStatus:
    """Tier and evidence for one pipeline component."""

    name: str
    tier: Tier
    evidence: str
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SimEnvSpec:
    """Seeded raw simulator environment specification."""

    env_id: str
    backend: str
    seed: int
    instruction: str
    asset_ref: str
    domain_randomization: dict[str, float | str]


@dataclass(frozen=True)
class FeedbackResult:
    """Normalized VLM/VLA feedback result."""

    success: bool
    score: float
    rationale: str
    critique: str = ""
    source: str = DEFAULT_FEEDBACK_SOURCE


@dataclass(frozen=True)
class SimToRealConfig:
    """Configuration for a tiered sim-to-real pipeline run."""

    run_id: str
    s3_endpoint: str = DEFAULT_S3_ENDPOINT
    s3_bucket: str = ""
    s3_prefix: str = ""
    input_data_uri: str = ""
    dataset_repo_id: str = DEFAULT_PUBLIC_LEROBOT_REPO
    dataset_revision: str = DEFAULT_PUBLIC_LEROBOT_REVISION
    policy_image: str = ""
    sim_backend: str = DEFAULT_SIM_BACKEND
    eval_backend: str = DEFAULT_EVAL_BACKEND
    feedback_source: str = DEFAULT_FEEDBACK_SOURCE
    split_fraction: float = DEFAULT_SPLIT_FRACTION
    env_count: int = 10
    episodes: int = 4
    train_steps: int = 50
    eval_episodes: int = 2
    threshold: float = DEFAULT_THRESHOLD
    seed: int = 42
    gpu: str = "H100:1"
    max_training_iterations: int = DEFAULT_MAX_TRAINING_ITERATIONS
    train_step_budget: int = DEFAULT_TRAIN_STEP_BUDGET
    min_eval_improvement: float = DEFAULT_MIN_EVAL_IMPROVEMENT
    policy_type: str = DEFAULT_POLICY_TYPE
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE
    train_num_workers: int = DEFAULT_TRAIN_NUM_WORKERS
    policy_device: str = "cuda"
    vlm_eval_backend: str = DEFAULT_VLM_EVAL_BACKEND
    vlm_eval_model: str = DEFAULT_VLM_EVAL_MODEL
    vlm_eval_endpoint_url: str = ""
    vlm_eval_frame_selection: str = "keyframes"
    vlm_eval_max_frames: int = 4
    vlm_eval_score: float | None = None
    trainer_command: str = ""
    checkpoint_uri: str = ""
    rrd_path: str = ""
    rerun_max_frames_per_episode: int = DEFAULT_RERUN_MAX_FRAMES_PER_EPISODE
    output_dir: Path | None = None

    def validate(self) -> None:
        """Validate local configuration invariants."""

        if not self.run_id:
            raise SimToRealError("run_id must not be empty")
        if not 0.0 < self.split_fraction < 1.0:
            raise SimToRealError(f"split_fraction must be in (0, 1), got {self.split_fraction}")
        if self.env_count < 2:
            raise SimToRealError(f"env_count must be at least 2, got {self.env_count}")
        if self.episodes <= 0:
            raise SimToRealError(f"episodes must be positive, got {self.episodes}")
        if self.train_steps <= 0:
            raise SimToRealError(f"train_steps must be positive, got {self.train_steps}")
        if self.eval_episodes <= 0:
            raise SimToRealError(f"eval_episodes must be positive, got {self.eval_episodes}")
        if not 0.0 <= self.threshold <= 1.0:
            raise SimToRealError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.max_training_iterations <= 0:
            raise SimToRealError(
                f"max_training_iterations must be positive, got {self.max_training_iterations}"
            )
        if self.train_step_budget <= 0:
            raise SimToRealError(f"train_step_budget must be positive, got {self.train_step_budget}")
        if self.train_step_budget < self.train_steps:
            raise SimToRealError(
                f"train_step_budget must be >= train_steps ({self.train_steps}), got {self.train_step_budget}"
            )
        if self.min_eval_improvement < 0.0:
            raise SimToRealError(f"min_eval_improvement must be non-negative, got {self.min_eval_improvement}")
        if self.train_batch_size <= 0:
            raise SimToRealError(f"train_batch_size must be positive, got {self.train_batch_size}")
        if self.train_num_workers < 0:
            raise SimToRealError(f"train_num_workers must be non-negative, got {self.train_num_workers}")
        if self.vlm_eval_max_frames <= 0:
            raise SimToRealError(f"vlm_eval_max_frames must be positive, got {self.vlm_eval_max_frames}")
        if self.vlm_eval_score is not None and not 0.0 <= self.vlm_eval_score <= 1.0:
            raise SimToRealError(f"vlm_eval_score must be in [0, 1], got {self.vlm_eval_score}")
        if self.rerun_max_frames_per_episode <= 0:
            raise SimToRealError(
                f"rerun_max_frames_per_episode must be positive, got {self.rerun_max_frames_per_episode}"
            )


@dataclass
class SimToRealReport:
    """Structured output from a local or live sim-to-real run."""

    run_id: str
    status: str
    created_at: str
    config: dict[str, Any]
    interfaces: dict[str, Any]
    artifacts: dict[str, str]
    components: list[ComponentStatus]
    feedback: FeedbackResult
    training_signal: dict[str, Any]
    outer_loop: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report payload."""

        return asdict(self)


def new_run_id(prefix: str = "sim-to-real") -> str:
    """Return a unique run id suitable for S3 prefixes and SkyPilot names."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def default_policy_image(*, registry: str | None = None) -> str:
    """Return the default BYO-compatible LeRobot policy image."""

    if registry or os.environ.get("NPA_REGISTRY"):
        return container_image_for_tool("lerobot-policy", registry=registry)
    return f"npa-lerobot-policy:{supported_tool_version('lerobot-policy')}"


def default_s3_prefix(run_id: str) -> str:
    """Return the default S3 prefix for a run."""

    return f"sim-to-real/{run_id}"


def build_config_from_env(**overrides: Any) -> SimToRealConfig:
    """Build a config from explicit overrides and environment fallbacks."""

    run_id = str(overrides.get("run_id") or os.environ.get("NPA_SIM_TO_REAL_RUN_ID") or new_run_id())
    s3_bucket = str(overrides.get("s3_bucket") or os.environ.get("S3_BUCKET") or os.environ.get("NPA_S3_BUCKET") or "")
    s3_prefix = str(overrides.get("s3_prefix") or os.environ.get("NPA_S3_PREFIX") or default_s3_prefix(run_id))
    dataset_repo_id = str(
        overrides.get("dataset_repo_id")
        or os.environ.get("LEROBOT_DATASET_REPO_ID")
        or DEFAULT_PUBLIC_LEROBOT_REPO
    )
    dataset_revision = str(
        overrides.get("dataset_revision")
        or os.environ.get("LEROBOT_DATASET_REVISION")
        or DEFAULT_PUBLIC_LEROBOT_REVISION
    )
    input_data_uri = resolve_dataset_source(
        str(
            overrides.get("input_data_uri")
            or os.environ.get("LEROBOT_DATASET_URI")
            or os.environ.get("INPUT_DATA_URI")
            or ""
        ),
        bucket=s3_bucket,
    )
    policy_image = str(overrides.get("policy_image") or os.environ.get("POLICY_IMAGE") or default_policy_image())
    checkpoint_uri = str(
        overrides.get("checkpoint_uri")
        or os.environ.get("CHECKPOINT_URI")
        or (f"s3://{s3_bucket}/{s3_prefix}/checkpoints/policy/" if s3_bucket else "")
    )
    rrd_path = str(
        overrides.get("rrd_path")
        or os.environ.get("RERUN_RRD_PATH")
        or (f"s3://{s3_bucket}/{s3_prefix}/viz/{run_id}.rrd" if s3_bucket else "")
    )
    output_dir = overrides.get("output_dir")
    vlm_score = overrides.get("vlm_eval_score")
    return SimToRealConfig(
        run_id=run_id,
        s3_endpoint=str(
            overrides.get("s3_endpoint")
            or os.environ.get("S3_ENDPOINT_URL")
            or os.environ.get("NEBIUS_S3_ENDPOINT")
            or os.environ.get("AWS_ENDPOINT_URL")
            or DEFAULT_S3_ENDPOINT
        ),
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        input_data_uri=input_data_uri,
        dataset_repo_id=dataset_repo_id,
        dataset_revision=dataset_revision,
        policy_image=policy_image,
        sim_backend=str(overrides.get("sim_backend") or os.environ.get("SIM_BACKEND") or DEFAULT_SIM_BACKEND),
        eval_backend=str(overrides.get("eval_backend") or os.environ.get("EVAL_BACKEND") or DEFAULT_EVAL_BACKEND),
        feedback_source=str(overrides.get("feedback_source") or os.environ.get("FEEDBACK_SOURCE") or DEFAULT_FEEDBACK_SOURCE),
        split_fraction=float(overrides.get("split_fraction", os.environ.get("SPLIT_FRACTION", DEFAULT_SPLIT_FRACTION))),
        env_count=int(overrides.get("env_count", os.environ.get("ENV_COUNT", "10"))),
        episodes=int(overrides.get("episodes", os.environ.get("EPISODES", "4"))),
        train_steps=int(overrides.get("train_steps", os.environ.get("TRAIN_STEPS", "2000"))),
        eval_episodes=int(overrides.get("eval_episodes", os.environ.get("EVAL_EPISODES", "10"))),
        threshold=float(overrides.get("threshold", os.environ.get("SUCCESS_THRESHOLD", DEFAULT_THRESHOLD))),
        seed=int(overrides.get("seed", os.environ.get("SEED", "42"))),
        gpu=str(overrides.get("gpu") or os.environ.get("GPU") or "H100:1"),
        max_training_iterations=int(
            overrides.get(
                "max_training_iterations",
                os.environ.get("MAX_TRAINING_ITERATIONS", str(DEFAULT_MAX_TRAINING_ITERATIONS)),
            )
        ),
        train_step_budget=int(
            overrides.get("train_step_budget", os.environ.get("TRAIN_STEP_BUDGET", str(DEFAULT_TRAIN_STEP_BUDGET)))
        ),
        min_eval_improvement=float(
            overrides.get(
                "min_eval_improvement",
                os.environ.get("MIN_EVAL_IMPROVEMENT", str(DEFAULT_MIN_EVAL_IMPROVEMENT)),
            )
        ),
        policy_type=str(overrides.get("policy_type") or os.environ.get("POLICY_TYPE") or DEFAULT_POLICY_TYPE),
        train_batch_size=int(
            overrides.get("train_batch_size", os.environ.get("TRAIN_BATCH_SIZE", str(DEFAULT_TRAIN_BATCH_SIZE)))
        ),
        train_num_workers=int(
            overrides.get("train_num_workers", os.environ.get("TRAIN_NUM_WORKERS", str(DEFAULT_TRAIN_NUM_WORKERS)))
        ),
        policy_device=str(overrides.get("policy_device") or os.environ.get("POLICY_DEVICE") or "cuda"),
        vlm_eval_backend=str(
            overrides.get("vlm_eval_backend")
            or os.environ.get("VLM_EVAL_BACKEND")
            or DEFAULT_VLM_EVAL_BACKEND
        ),
        vlm_eval_model=str(
            overrides.get("vlm_eval_model")
            or os.environ.get("VLM_EVAL_MODEL")
            or DEFAULT_VLM_EVAL_MODEL
        ),
        vlm_eval_endpoint_url=str(
            overrides.get("vlm_eval_endpoint_url") or os.environ.get("VLM_EVAL_ENDPOINT_URL") or ""
        ),
        vlm_eval_frame_selection=str(
            overrides.get("vlm_eval_frame_selection")
            or os.environ.get("VLM_EVAL_FRAME_SELECTION")
            or "keyframes"
        ),
        vlm_eval_max_frames=int(
            overrides.get("vlm_eval_max_frames", os.environ.get("VLM_EVAL_MAX_FRAMES", "4"))
        ),
        vlm_eval_score=float(vlm_score) if vlm_score is not None and str(vlm_score) else None,
        trainer_command=str(overrides.get("trainer_command") or os.environ.get("CUSTOM_LEROBOT_TRAINER_COMMAND") or ""),
        checkpoint_uri=checkpoint_uri,
        rrd_path=rrd_path,
        rerun_max_frames_per_episode=int(
            overrides.get(
                "rerun_max_frames_per_episode",
                os.environ.get("RERUN_MAX_FRAMES_PER_EPISODE", str(DEFAULT_RERUN_MAX_FRAMES_PER_EPISODE)),
            )
        ),
        output_dir=Path(output_dir) if output_dir else None,
    )


def artifact_uris(config: SimToRealConfig) -> dict[str, str]:
    """Return the canonical S3 artifact layout for a run."""

    root = f"s3://{config.s3_bucket}/{config.s3_prefix.strip('/')}" if config.s3_bucket else ""
    if not root:
        return {}
    return {
        "root": f"{root}/",
        "input_data": config.input_data_uri,
        "example_dataset": default_staged_dataset_uri(config.s3_bucket),
        "dataset_summary": f"{root}/datasets/lerobot-summary.json",
        "raw_envs": f"{root}/raw-envs/",
        "train_envs": f"{root}/splits/train/",
        "heldout_envs": f"{root}/splits/heldout/",
        "rollouts": f"{root}/rollouts/",
        "feedback": f"{root}/feedback/",
        "training_signal": f"{root}/training-signal.json",
        "checkpoint": config.checkpoint_uri or f"{root}/checkpoints/policy/",
        "report": f"{root}/reports/sim-to-real-report.json",
        "rrd": config.rrd_path or f"{root}/viz/{config.run_id}.rrd",
        "lancedb_cache": f"{root}/lancedb/",
    }


def build_policy_container_contract(config: SimToRealConfig) -> dict[str, Any]:
    """Return the BYO LeRobot policy container I/O contract."""

    paths = artifact_uris(config)
    return {
        "image": config.policy_image or default_policy_image(),
        "input_path": paths.get("train_envs", config.input_data_uri),
        "output_path": paths.get("checkpoint", config.checkpoint_uri),
        "endpoints": {
            "health": "GET /health",
            "infer": "POST /infer",
            "rollout": "POST /rollout",
            "feedback_train_step": "POST /feedback/train-step",
        },
        "commands": {
            "check_import": "python -m npa.workbench.lerobot.policy_container check-import",
            "train": "python -m npa.workbench.lerobot.policy_container train",
            "eval": "python -m npa.workbench.lerobot.policy_container eval",
            "validate_checkpoint": "python -m npa.workbench.lerobot.policy_container validate-checkpoint",
        },
        "env": {
            "POLICY_IMAGE": config.policy_image or default_policy_image(),
            "INPUT_DATA_URI": config.input_data_uri,
            "CHECKPOINT_URI": paths.get("checkpoint", config.checkpoint_uri),
            "FEEDBACK_SOURCE": config.feedback_source,
            "TRAIN_STEPS": str(config.train_steps),
            "TRAIN_STEP_BUDGET": str(config.train_step_budget),
            "MAX_TRAINING_ITERATIONS": str(config.max_training_iterations),
            "POLICY_TYPE": config.policy_type,
            "TRAIN_BATCH_SIZE": str(config.train_batch_size),
            "TRAIN_NUM_WORKERS": str(config.train_num_workers),
            "POLICY_DEVICE": config.policy_device,
            "VLM_EVAL_BACKEND": config.vlm_eval_backend,
            "VLM_EVAL_MODEL": config.vlm_eval_model,
            "CUSTOM_LEROBOT_TRAINER_COMMAND": config.trainer_command,
        },
        "observation_schema": {
            "observation.images.workspace": {"dtype": "uint8", "shape": ["T", "H", "W", 3]},
            "observation.images.wrist": {"dtype": "uint8", "shape": ["T", "H", "W", 3]},
            "observation.state": {"dtype": "float32", "shape": ["T", "state_dim"]},
            "language.instruction": {"dtype": "string", "shape": ["T"]},
        },
        "action_schema": {
            "action": {"dtype": "float32", "shape": ["T", "action_dim"]},
        },
        "feedback_schema": {
            "success": {"dtype": "bool", "required": True},
            "score": {"dtype": "float32", "range": [0.0, 1.0], "required": True},
            "rationale": {"dtype": "string", "required": True},
            "source": {"dtype": "string", "required": False, "examples": ["vlm", "vla"]},
        },
    }


def generate_raw_envs(
    *,
    count: int,
    seed: int,
    backend: str = DEFAULT_SIM_BACKEND,
    instruction: str = "Pick and place the object at the target.",
) -> list[SimEnvSpec]:
    """Generate deterministic raw sim environment specs."""

    if count <= 0:
        raise SimToRealError(f"count must be positive, got {count}")
    rng = random.Random(seed)
    envs: list[SimEnvSpec] = []
    for index in range(count):
        env_seed = rng.randrange(1, 2**31 - 1)
        envs.append(
            SimEnvSpec(
                env_id=f"{backend}-raw-{index:05d}",
                backend=backend,
                seed=env_seed,
                instruction=instruction,
                asset_ref=f"{backend}:pick-place:{index:05d}",
                domain_randomization={
                    "lighting": round(rng.uniform(0.25, 1.0), 4),
                    "friction": round(rng.uniform(0.4, 1.2), 4),
                    "texture": f"seed-{rng.randrange(0, 9999):04d}",
                },
            )
        )
    return envs


def seeded_train_heldout_split(
    envs: list[SimEnvSpec],
    *,
    train_fraction: float = DEFAULT_SPLIT_FRACTION,
    seed: int = 42,
) -> tuple[list[SimEnvSpec], list[SimEnvSpec]]:
    """Split raw envs into seeded train and held-out partitions."""

    if not 0.0 < train_fraction < 1.0:
        raise SimToRealError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if len(envs) < 2:
        raise SimToRealError("at least two envs are required for train/held-out split")
    indices = list(range(len(envs)))
    random.Random(seed).shuffle(indices)
    train_size = max(1, min(len(envs) - 1, int(round(len(envs) * train_fraction))))
    train_indices = set(indices[:train_size])
    train = [env for idx, env in enumerate(envs) if idx in train_indices]
    heldout = [env for idx, env in enumerate(envs) if idx not in train_indices]
    return train, heldout


def parse_feedback_result(payload: dict[str, Any], *, source: str = DEFAULT_FEEDBACK_SOURCE) -> FeedbackResult:
    """Parse and guard feedback from a VLM or VLA backend."""

    missing = [key for key in ("success", "score", "rationale") if key not in payload]
    if missing:
        raise SimToRealError(f"feedback missing required keys: {', '.join(missing)}")
    score = float(payload["score"])
    if not 0.0 <= score <= 1.0:
        raise SimToRealError(f"feedback score must be in [0, 1], got {score}")
    rationale = str(payload["rationale"]).strip()
    if not rationale:
        raise SimToRealError("feedback rationale must not be empty")
    return FeedbackResult(
        success=bool(payload["success"]),
        score=score,
        rationale=rationale,
        critique=str(payload.get("critique", "")),
        source=str(payload.get("source") or source),
    )


def evaluate_feedback(
    config: SimToRealConfig,
    *,
    rollout_path: Path,
    output_path: Path,
    task: str = "sim-to-real",
) -> tuple[FeedbackResult, ComponentStatus]:
    """Evaluate a rollout through the existing VLM eval interface."""

    if config.feedback_source != "vlm":
        feedback = FeedbackResult(
            success=False,
            score=0.0,
            rationale=f"Feedback source {config.feedback_source!r} is configured as a typed seam.",
            critique="Configure FEEDBACK_SOURCE=vlm to use the current VLM eval implementation.",
            source=config.feedback_source,
        )
        return feedback, ComponentStatus(
            name=f"{config.feedback_source}_feedback",
            tier=Tier.SEAM,
            evidence=f"{config.feedback_source} feedback is modeled but has no backend implementation.",
        )

    try:
        from npa.workbench.vlm_eval import VlmEvalError, evaluate_vlm, write_result
    except ImportError as exc:
        fallback = FeedbackResult(
            success=False,
            score=0.0,
            rationale=f"VLM eval interface could not be imported: {exc}",
            source="vlm",
        )
        return fallback, ComponentStatus(
            name="vlm_feedback",
            tier=Tier.BLOCKED,
            evidence=str(exc),
        )

    try:
        result = evaluate_vlm(
            input_path=str(rollout_path),
            output_path=str(output_path),
            task=task,
            backend=config.vlm_eval_backend,
            model=config.vlm_eval_model,
            endpoint_url=config.vlm_eval_endpoint_url,
            frame_selection=config.vlm_eval_frame_selection,
            max_frames=config.vlm_eval_max_frames,
            success_threshold=config.threshold,
            score=config.vlm_eval_score,
        )
        payload = asdict(result)
        payload["written_uri"] = write_result(payload, result_uri=result.result_uri)
    except (OSError, VlmEvalError) as exc:
        fallback = FeedbackResult(
            success=False,
            score=0.0,
            rationale=f"VLM eval interface could not score the rollout: {exc}",
            source="vlm",
        )
        return fallback, ComponentStatus(
            name="vlm_feedback",
            tier=Tier.BLOCKED,
            evidence=str(exc),
        )

    feedback = FeedbackResult(
        success=result.passed,
        score=result.score,
        rationale=result.rationale or result.status,
        critique=result.rationale,
        source="vlm",
    )
    tier = Tier.PARTIAL if result.backend == "stub" else Tier.WORKS
    evidence = (
        "Existing vlm-eval stub backend produced schema-compatible feedback."
        if result.backend == "stub"
        else f"Existing vlm-eval backend {result.backend!r} scored rollout frames."
    )
    return feedback, ComponentStatus(
        name="vlm_feedback",
        tier=tier,
        evidence=evidence,
        artifacts={"result_uri": payload["written_uri"]},
    )


def feedback_to_training_signal(feedback: FeedbackResult) -> dict[str, Any]:
    """Convert feedback into a bounded training signal."""

    reward = feedback.score if feedback.success else min(feedback.score, 0.5)
    return {
        "schema": "npa.sim_to_real.training_signal.v1",
        "scalar_reward": round(float(reward), 6),
        "success": feedback.success,
        "score": feedback.score,
        "natural_language_critique": feedback.critique or feedback.rationale,
        "loss_weight": round(1.0 + (1.0 - feedback.score), 6),
    }


def outer_loop_decision(score: float, threshold: float, checkpoint_uri: str) -> dict[str, Any]:
    """Decide whether to promote a checkpoint or request another loop."""

    if not 0.0 <= score <= 1.0:
        raise SimToRealError(f"score must be in [0, 1], got {score}")
    if not 0.0 <= threshold <= 1.0:
        raise SimToRealError(f"threshold must be in [0, 1], got {threshold}")
    promoted = score >= threshold
    return {
        "score": score,
        "threshold": threshold,
        "decision": "promote_checkpoint" if promoted else "retrain",
        "checkpoint_uri": checkpoint_uri,
    }


def s3_roundtrip(config: SimToRealConfig, *, local_dir: Path) -> ComponentStatus:
    """Attempt a small S3 round trip and return component evidence."""

    if not config.s3_bucket:
        return ComponentStatus(
            name="nebius_s3",
            tier=Tier.BLOCKED,
            evidence="NPA_S3_BUCKET is not configured.",
        )
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not access_key or not secret_key:
        return ComponentStatus(
            name="nebius_s3",
            tier=Tier.BLOCKED,
            evidence="AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY is not configured.",
        )

    marker = local_dir / "s3-roundtrip.json"
    marker.write_text(
        json.dumps({"run_id": config.run_id, "created_at": _utc_now()}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    uri = f"s3://{config.s3_bucket}/{config.s3_prefix.strip('/')}/health/s3-roundtrip.json"
    try:
        client = StorageClient.from_environment(
            endpoint_url=config.s3_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        uploaded = client.upload_file(str(marker), uri)
        downloaded = local_dir / "s3-roundtrip.downloaded.json"
        client.download_path(uploaded, str(downloaded))
    except Exception as exc:
        return ComponentStatus(
            name="nebius_s3",
            tier=Tier.BLOCKED,
            evidence=f"S3 round trip failed: {exc}",
            artifacts={"target_uri": uri},
        )
    if not downloaded.exists() or downloaded.read_text(encoding="utf-8") != marker.read_text(encoding="utf-8"):
        return ComponentStatus(
            name="nebius_s3",
            tier=Tier.BLOCKED,
            evidence="S3 round trip downloaded content mismatch.",
            artifacts={"target_uri": uri},
        )
    return ComponentStatus(
        name="nebius_s3",
        tier=Tier.WORKS,
        evidence="Uploaded and downloaded a run marker through Nebius S3-compatible storage.",
        artifacts={"roundtrip_uri": uri},
    )


def write_rerun_summary(report_payload: dict[str, Any], output_path: Path) -> ComponentStatus:
    """Write a small Rerun recording, falling back to a labeled partial artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import rerun as rr

        recording = rr.RecordingStream(APPLICATION_ID)
        rr.save(output_path, recording=recording)
        scalar = rr.Scalars if hasattr(rr, "Scalars") else rr.Scalar
        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("run_step", 0, recording=recording)
        else:
            rr.set_time("run_step", sequence=0, recording=recording)
        rr.log("pipeline/outer_score", scalar(float(report_payload["outer_loop"]["score"])), recording=recording)
        rr.log("pipeline/threshold", scalar(float(report_payload["outer_loop"]["threshold"])), recording=recording)
        rr.disconnect(recording=recording)
    except Exception as exc:
        output_path.write_text(
            json.dumps({"format": "npa.rerun.partial.v1", "error": str(exc), **report_payload}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return ComponentStatus(
            name="rerun",
            tier=Tier.PARTIAL,
            evidence=f"Rerun SDK write failed; wrote labeled structural artifact instead: {exc}",
            artifacts={"rrd": str(output_path)},
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        return ComponentStatus(
            name="rerun",
            tier=Tier.BLOCKED,
            evidence="Rerun SDK did not create a non-empty .rrd file.",
            artifacts={"rrd": str(output_path)},
        )
    return ComponentStatus(
        name="rerun",
        tier=Tier.WORKS,
        evidence="Wrote a non-empty Rerun .rrd summary with rerun-sdk.",
        artifacts={"rrd": str(output_path)},
    )


def run_structural_spine(
    config: SimToRealConfig,
    *,
    attempt_s3_roundtrip: bool = False,
) -> SimToRealReport:
    """Run the largest local structural subset and write report artifacts."""

    config.validate()
    local_dir = config.output_dir or Path("/tmp") / f"npa-sim-to-real-{config.run_id}"
    local_dir.mkdir(parents=True, exist_ok=True)

    components: list[ComponentStatus] = []
    if attempt_s3_roundtrip:
        components.append(s3_roundtrip(config, local_dir=local_dir))
    else:
        components.append(
            ComponentStatus(
                name="nebius_s3",
                tier=Tier.PARTIAL if config.s3_bucket else Tier.BLOCKED,
                evidence=(
                    "S3 artifact layout is configured; live round trip was not requested."
                    if config.s3_bucket
                    else "NPA_S3_BUCKET is not configured."
                ),
            )
        )

    staged_example_uri = default_staged_dataset_uri(config.s3_bucket) if config.s3_bucket else ""
    source_is_default_staged = (
        bool(staged_example_uri)
        and config.input_data_uri.rstrip("/") == staged_example_uri.rstrip("/")
    )
    dataset_source_used = config.input_data_uri
    staging_fallback_error = ""
    try:
        local_dataset = materialize_lerobot_dataset(
            config.input_data_uri,
            local_dir / "datasets",
            repo_id=config.dataset_repo_id,
            revision=config.dataset_revision,
            s3_endpoint=config.s3_endpoint,
        )
    except LeRobotDatasetError as exc:
        if not source_is_default_staged:
            raise SimToRealError(str(exc)) from exc
        local_dataset = download_public_lerobot_dataset(
            local_dir / "datasets",
            repo_id=config.dataset_repo_id,
            revision=config.dataset_revision,
        )
        dataset_source_used = default_public_dataset_uri()
        staging_fallback_error = str(exc)

    try:
        dataset_summary = summarize_lerobot_dataset(
            local_dataset,
            source_uri=dataset_source_used,
            repo_id=config.dataset_repo_id,
            revision=config.dataset_revision,
            license=DEFAULT_PUBLIC_LEROBOT_LICENSE,
        )
    except LeRobotDatasetError as exc:
        raise SimToRealError(str(exc)) from exc
    dataset_summary_path = local_dir / "lerobot-dataset-summary.json"
    _write_json(dataset_summary_path, dataset_summary.to_dict())
    dataset_tier = Tier.WORKS if dataset_summary.loaded_with_lerobot_dataset else Tier.PARTIAL
    components.append(
        ComponentStatus(
            name="real_lerobot_dataset",
            tier=dataset_tier,
            evidence=(
                f"Validated {dataset_summary.repo_id}@{dataset_summary.revision} with "
                f"{dataset_summary.total_episodes} real episodes, {dataset_summary.total_frames} frames, "
                f"vision={dataset_summary.camera_keys}, state={dataset_summary.state_keys}, "
                f"action={dataset_summary.action_keys}; "
                + (
                    "loaded via LeRobotDataset."
                    if dataset_summary.loaded_with_lerobot_dataset
                    else dataset_summary.lerobot_dataset_error
                )
            ),
            artifacts={"summary": str(dataset_summary_path), "source": dataset_source_used},
        )
    )
    if staged_example_uri:
        try:
            staged_uri = (
                config.input_data_uri
                if config.input_data_uri.rstrip("/") == staged_example_uri.rstrip("/") and not staging_fallback_error
                else stage_dataset_to_s3(local_dataset, staged_example_uri, s3_endpoint=config.s3_endpoint)
            )
            components.append(
                ComponentStatus(
                    name="lerobot_dataset_staging",
                    tier=Tier.WORKS,
                    evidence=(
                        "Pinned public LeRobot dataset is available at the staged example S3 URI."
                        if not staging_fallback_error
                        else "Default staged dataset was missing or unreadable; downloaded the pinned public source and staged it to S3."
                    ),
                    artifacts={"staged_uri": staged_uri},
                )
            )
        except LeRobotDatasetError as exc:
            components.append(
                ComponentStatus(
                    name="lerobot_dataset_staging",
                    tier=Tier.BLOCKED,
                    evidence=(
                        f"Default staged dataset unavailable ({staging_fallback_error}); staging failed: {exc}"
                        if staging_fallback_error
                        else str(exc)
                    ),
                    artifacts={"staged_uri": staged_example_uri, "public_source": dataset_source_used},
                )
            )

    train_episodes, heldout_episodes = seeded_episode_split(
        dataset_summary.episode_indices,
        train_fraction=config.split_fraction,
        seed=config.seed,
    )
    split_path = local_dir / "split.json"
    write_episode_split_manifest(
        split_path,
        train=train_episodes,
        heldout=heldout_episodes,
        split_fraction=config.split_fraction,
        seed=config.seed,
        dataset=dataset_summary,
    )
    components.append(
        ComponentStatus(
            name="lerobot_episode_split",
            tier=Tier.WORKS,
            evidence=(
                f"Seeded split over real episodes covers all {dataset_summary.total_episodes} episodes: "
                f"train={len(train_episodes)}, heldout={len(heldout_episodes)}, seed={config.seed}."
            ),
            artifacts={"split": str(split_path)},
        )
    )

    policy_contract = build_policy_container_contract(config)
    _write_json(local_dir / "policy-container-contract.json", policy_contract)
    components.append(
        ComponentStatus(
            name="byo_lerobot_policy_container",
            tier=Tier.PARTIAL,
            evidence=(
                "Resolved BYO policy image, rollout/inference endpoints, VLA-capable observation/action schema, "
                "and vlm_eval-compatible feedback schema; no live policy container was launched in local smoke."
            ),
            artifacts={"contract": str(local_dir / "policy-container-contract.json")},
        )
    )

    rollout_dir = _write_lerobot_rollout_fixture(local_dir / "rollouts", heldout_episodes[0], dataset_summary)
    vlm_output = local_dir / "feedback" / "vlm-eval"
    feedback_request = {
        "run_id": config.run_id,
        "heldout_episode": heldout_episodes[0],
        "rollout_path": str(rollout_dir),
        "feedback_output": str(vlm_output),
        "rubric": {"success": "task completed", "score": "float in [0,1]", "rationale": "brief reason"},
        "observation": {
            "images": dataset_summary.camera_keys,
            "language": "Complete the demonstrated manipulation task.",
            "state": dataset_summary.state_keys,
        },
    }
    feedback, feedback_status = evaluate_feedback(
        config,
        rollout_path=rollout_dir,
        output_path=vlm_output,
        task="Complete the demonstrated manipulation task.",
    )
    components.append(feedback_status)
    training_signal = feedback_to_training_signal(feedback)
    _write_json(local_dir / "training-signal.json", training_signal)

    try:
        hook_result = run_feedback_training_step(
            parse_feedback_batch(asdict(feedback)),
            output_dir=local_dir / "feedback-training",
        )
        hook_path = local_dir / "feedback-training" / "feedback-update-result.json"
        _write_json(hook_path, hook_result.to_dict())
        components.append(
            ComponentStatus(
                name="inner_feedback_training_loop",
                tier=Tier.SEAM,
                evidence=(
                    f"Custom feedback trainer hook ran {hook_result.steps} update step(s) "
                    f"with backend={hook_result.backend}; calibration/convergence remains research-grade."
                ),
                artifacts={
                    "training_signal": str(local_dir / "training-signal.json"),
                    "feedback_update": str(hook_path),
                    "feedback_checkpoint": hook_result.checkpoint_path,
                },
            )
        )
    except Exception as exc:
        components.append(
            ComponentStatus(
                name="inner_feedback_training_loop",
                tier=Tier.BLOCKED,
                evidence=f"Feedback trainer hook failed: {exc}",
                artifacts={"training_signal": str(local_dir / "training-signal.json")},
            )
        )

    checkpoint_uri = config.checkpoint_uri or str(local_dir / "checkpoints" / "policy")
    outer = outer_loop_decision(feedback.score, config.threshold, checkpoint_uri)
    checkpoint_marker = local_dir / "checkpoints" / "promoted-checkpoint.json"
    checkpoint_marker.parent.mkdir(parents=True, exist_ok=True)
    _write_json(checkpoint_marker, {"run_id": config.run_id, **outer})
    components.append(
        ComponentStatus(
            name="outer_loop",
            tier=Tier.PARTIAL,
            evidence="Applied held-out score threshold logic and wrote a checkpoint marker; no live checkpoint weights were produced.",
            artifacts={"checkpoint_marker": str(checkpoint_marker)},
        )
    )

    components.extend(
        [
            ComponentStatus(
                name="lightwheel",
                tier=Tier.SEAM,
                evidence="Backend is represented as a selectable sim/eval backend; no Lightwheel implementation is present in this repo.",
            ),
            ComponentStatus(
                name="isaac_lab",
                tier=Tier.SEAM,
                evidence="Isaac Lab workbench and SkyPilot routes exist; this sim-to-real loop has only a typed backend seam.",
            ),
            ComponentStatus(
                name="cosmos_augmentation",
                tier=Tier.SEAM,
                evidence="Cosmos augmentation is an optional stage seam; no live augmentation was run in the structural spine.",
            ),
            ComponentStatus(
                name="lancedb_cache",
                tier=Tier.SEAM,
                evidence="Optional LanceDB cache URI is in the artifact layout; no cache backend was started.",
            ),
            ComponentStatus(
                name="vla_feedback",
                tier=Tier.SEAM,
                evidence="Policy I/O accepts vision and language observations; no VLA feedback backend is configured.",
            ),
        ]
    )

    report = SimToRealReport(
        run_id=config.run_id,
        status=_overall_status(components),
        created_at=_utc_now(),
        config=_redacted_config(config),
        interfaces={
            "s3": {
                "endpoint": config.s3_endpoint,
                "bucket": config.s3_bucket,
                "prefix": config.s3_prefix,
            },
            "dataset": dataset_summary.to_dict(),
            "split": {
                "train_count": len(train_episodes),
                "heldout_count": len(heldout_episodes),
                "seed": config.seed,
            },
            "policy_container": policy_contract,
            "feedback_request": feedback_request,
        },
        artifacts={**artifact_uris(config), "local_report_dir": str(local_dir)},
        components=components,
        feedback=feedback,
        training_signal=training_signal,
        outer_loop=outer,
    )
    rerun_output = _local_rrd_path(config, local_dir)
    try:
        logical_rerun = lerobot_dataset_logical_to_rerun(
            local_dataset,
            rerun_output,
            input_episode_indices=train_episodes[:1],
            rollout_episode_indices=heldout_episodes[:1],
            feedback_by_episode={int(heldout_episodes[0]): asdict(feedback)},
            max_frames_per_episode=config.rerun_max_frames_per_episode,
        )
        rerun_status = ComponentStatus(
            name="rerun",
            tier=Tier.WORKS,
            evidence="Wrote and verified logical LeRobot/Rerun entities for input demos, policy rollout, and per-episode feedback.",
            artifacts={
                "rrd": logical_rerun.output_rrd_path,
                "entity_counts": json.dumps(logical_rerun.entity_counts, sort_keys=True),
                "view_command": f"rerun {logical_rerun.output_rrd_path}",
            },
        )
    except RerunAdapterError as exc:
        rerun_status = write_rerun_summary(report.to_dict(), rerun_output)
        rerun_status = ComponentStatus(
            name=rerun_status.name,
            tier=Tier.PARTIAL if rerun_status.tier != Tier.BLOCKED else Tier.BLOCKED,
            evidence=f"Logical LeRobot/Rerun adapter failed ({exc}); {rerun_status.evidence}",
            artifacts=rerun_status.artifacts,
        )
    report.components.append(rerun_status)
    report.status = _overall_status(report.components)
    report_path = local_dir / "sim-to-real-report.json"
    _write_json(report_path, report.to_dict())
    upload_status = _upload_run_artifacts(
        config,
        {
            "training_signal": local_dir / "training-signal.json",
            "dataset_summary": dataset_summary_path,
            "split": split_path,
            "checkpoint": checkpoint_marker,
            "report": report_path,
            "rrd": _local_rrd_path(config, local_dir),
        },
    )
    if upload_status is not None:
        report.components.append(upload_status)
        report.status = _overall_status(report.components)
        _write_json(report_path, report.to_dict())
        _upload_run_artifacts(config, {"report": report_path})
    return report


def run_real_lerobot_loop(
    config: SimToRealConfig,
    *,
    attempt_s3_roundtrip: bool = False,
) -> SimToRealReport:
    """Run the real LeRobot dataset policy train/eval feedback loop."""

    config.validate()
    local_dir = config.output_dir or Path("/tmp") / f"npa-sim-to-real-{config.run_id}"
    local_dir.mkdir(parents=True, exist_ok=True)

    components: list[ComponentStatus] = []
    if attempt_s3_roundtrip:
        components.append(s3_roundtrip(config, local_dir=local_dir))
    else:
        components.append(
            ComponentStatus(
                name="nebius_s3",
                tier=Tier.PARTIAL if config.s3_bucket else Tier.BLOCKED,
                evidence=(
                    "S3 artifact layout is configured; live round trip was not requested."
                    if config.s3_bucket
                    else "NPA_S3_BUCKET is not configured."
                ),
            )
        )

    try:
        import_result = assert_lerobot_importable()
        components.append(
            ComponentStatus(
                name="lerobot_runtime_import",
                tier=Tier.WORKS,
                evidence=(
                    f"Imported lerobot {import_result.version or 'unknown'} and "
                    f"{import_result.dataset_class} in the runtime."
                ),
                artifacts=import_result.to_dict(),
            )
        )
    except PolicyContainerError as exc:
        raise SimToRealError(str(exc)) from exc

    staged_example_uri = default_staged_dataset_uri(config.s3_bucket) if config.s3_bucket else ""
    source_is_default_staged = (
        bool(staged_example_uri)
        and config.input_data_uri.rstrip("/") == staged_example_uri.rstrip("/")
    )
    dataset_source_used = config.input_data_uri
    staging_fallback_error = ""
    try:
        local_dataset = materialize_lerobot_dataset(
            config.input_data_uri,
            local_dir / "datasets",
            repo_id=config.dataset_repo_id,
            revision=config.dataset_revision,
            s3_endpoint=config.s3_endpoint,
        )
    except LeRobotDatasetError as exc:
        if not source_is_default_staged:
            raise SimToRealError(str(exc)) from exc
        local_dataset = download_public_lerobot_dataset(
            local_dir / "datasets",
            repo_id=config.dataset_repo_id,
            revision=config.dataset_revision,
        )
        dataset_source_used = default_public_dataset_uri()
        staging_fallback_error = str(exc)

    try:
        dataset_summary = summarize_lerobot_dataset(
            local_dataset,
            source_uri=dataset_source_used,
            repo_id=config.dataset_repo_id,
            revision=config.dataset_revision,
            license=DEFAULT_PUBLIC_LEROBOT_LICENSE,
        )
    except LeRobotDatasetError as exc:
        raise SimToRealError(str(exc)) from exc
    dataset_summary_path = local_dir / "lerobot-dataset-summary.json"
    _write_json(dataset_summary_path, dataset_summary.to_dict())
    if not dataset_summary.loaded_with_lerobot_dataset:
        raise SimToRealError(dataset_summary.lerobot_dataset_error)
    components.append(
        ComponentStatus(
            name="real_lerobot_dataset",
            tier=Tier.WORKS,
            evidence=(
                f"Loaded {dataset_summary.repo_id}@{dataset_summary.revision} via LeRobotDataset with "
                f"{dataset_summary.total_episodes} real episodes and {dataset_summary.total_frames} frames."
            ),
            artifacts={"summary": str(dataset_summary_path), "source": dataset_source_used},
        )
    )
    if staged_example_uri:
        try:
            staged_uri = (
                config.input_data_uri
                if config.input_data_uri.rstrip("/") == staged_example_uri.rstrip("/") and not staging_fallback_error
                else stage_dataset_to_s3(local_dataset, staged_example_uri, s3_endpoint=config.s3_endpoint)
            )
            components.append(
                ComponentStatus(
                    name="lerobot_dataset_staging",
                    tier=Tier.WORKS,
                    evidence="LeRobot dataset is available at the configured S3 dataset URI.",
                    artifacts={"staged_uri": staged_uri},
                )
            )
        except LeRobotDatasetError as exc:
            components.append(
                ComponentStatus(
                    name="lerobot_dataset_staging",
                    tier=Tier.BLOCKED,
                    evidence=str(exc),
                    artifacts={"staged_uri": staged_example_uri, "public_source": dataset_source_used},
                )
            )

    train_episodes, heldout_episodes = seeded_episode_split(
        dataset_summary.episode_indices,
        train_fraction=config.split_fraction,
        seed=config.seed,
    )
    split_path = local_dir / "split.json"
    write_episode_split_manifest(
        split_path,
        train=train_episodes,
        heldout=heldout_episodes,
        split_fraction=config.split_fraction,
        seed=config.seed,
        dataset=dataset_summary,
    )
    components.append(
        ComponentStatus(
            name="lerobot_episode_split",
            tier=Tier.WORKS,
            evidence=(
                f"Seeded train/heldout split covers all {dataset_summary.total_episodes} real episodes: "
                f"train={len(train_episodes)}, heldout={len(heldout_episodes)}."
            ),
            artifacts={"split": str(split_path)},
        )
    )

    policy_contract = build_policy_container_contract(config)
    _write_json(local_dir / "policy-container-contract.json", policy_contract)
    components.append(
        ComponentStatus(
            name="real_lerobot_policy_trainer",
            tier=Tier.WORKS,
            evidence=(
                f"Resolved real LeRobot trainer command for policy={config.policy_type}, "
                f"batch_size={config.train_batch_size}, max_iterations={config.max_training_iterations}, "
                f"step_budget={config.train_step_budget}; no local smoke trainer is used."
            ),
            artifacts={"contract": str(local_dir / "policy-container-contract.json")},
        )
    )

    rollout_env = _matching_rollout_env(config)
    if not rollout_env:
        raise SimToRealError(
            "No matching rollout environment is configured for this LeRobot dataset. "
            "Set EVAL_BACKEND to a real matching env, or run a heldout-metric implementation for the dataset."
        )

    loop_history: list[dict[str, Any]] = []
    latest_feedback = FeedbackResult(
        success=False,
        score=0.0,
        rationale="No eval has run yet.",
        source=config.feedback_source,
    )
    latest_signal = feedback_to_training_signal(latest_feedback)
    latest_checkpoint = ""
    previous_score: float | None = None
    cumulative_steps = 0
    stop_reason = "step_budget"
    training_dir = local_dir / "policy-training"
    eval_root = local_dir / "eval"

    for iteration in range(1, config.max_training_iterations + 1):
        remaining_budget = config.train_step_budget - cumulative_steps
        if remaining_budget <= 0:
            stop_reason = "step_budget"
            break
        additional_steps = min(config.train_steps, remaining_budget)
        cumulative_steps += additional_steps
        try:
            train_result = run_lerobot_training(
                dataset_path=local_dataset,
                dataset_repo_id=config.dataset_repo_id,
                output_dir=training_dir,
                steps=cumulative_steps,
                policy_type=config.policy_type,
                batch_size=config.train_batch_size,
                num_workers=config.train_num_workers,
                device=config.policy_device,
                resume=iteration > 1,
                log_path=local_dir / "logs" / f"train-iter-{iteration:02d}.log",
            )
            checkpoint_validation = validate_lerobot_checkpoint(train_result.checkpoint_path)
            eval_result = run_lerobot_eval(
                checkpoint_path=train_result.checkpoint_path,
                output_dir=eval_root / f"iter-{iteration:02d}",
                env_type=rollout_env,
                episodes=config.eval_episodes,
                device=config.policy_device,
                log_path=local_dir / "logs" / f"eval-iter-{iteration:02d}.log",
            )
        except PolicyContainerError as exc:
            raise SimToRealError(str(exc)) from exc

        latest_checkpoint = train_result.checkpoint_path
        latest_feedback = _feedback_from_eval(eval_result.to_dict(), threshold=config.threshold)
        latest_signal = feedback_to_training_signal(latest_feedback)
        signal_path = local_dir / "training-signals" / f"iter-{iteration:02d}.json"
        _write_json(signal_path, latest_signal)
        improvement = (
            None
            if previous_score is None
            else round(float(eval_result.score) - float(previous_score), 6)
        )
        loop_history.append(
            {
                "iteration": iteration,
                "train": train_result.to_dict(),
                "checkpoint_validation": checkpoint_validation.to_dict(),
                "eval": eval_result.to_dict(),
                "feedback": asdict(latest_feedback),
                "training_signal": latest_signal,
                "training_signal_path": str(signal_path),
                "improvement": improvement,
            }
        )
        previous_score = float(eval_result.score)
        if eval_result.score >= config.threshold:
            stop_reason = "threshold"
            break
        if cumulative_steps >= config.train_step_budget:
            stop_reason = "step_budget"
            break

    if not loop_history:
        raise SimToRealError("real LeRobot loop did not complete any train/eval iteration")

    trend = [round(float(item["eval"]["score"]), 6) for item in loop_history]
    improved = len(trend) < 2 or trend[-1] >= trend[0] + config.min_eval_improvement
    components.append(
        ComponentStatus(
            name="real_training",
            tier=Tier.WORKS,
            evidence=(
                f"Ran {len(loop_history)} LeRobot training iteration(s), cumulative_steps={cumulative_steps}, "
                f"checkpoint={latest_checkpoint}."
            ),
            artifacts={"checkpoint": latest_checkpoint, "training_log": loop_history[-1]["train"]["log_path"]},
        )
    )
    components.append(
        ComponentStatus(
            name="real_rollout_eval",
            tier=Tier.WORKS,
            evidence=(
                f"Measured {loop_history[-1]['eval']['metric_name']}={trend[-1]} in env={rollout_env} "
                f"over {config.eval_episodes} rollout episode(s)."
            ),
            artifacts={"eval_info": loop_history[-1]["eval"]["eval_info_path"], "eval_log": loop_history[-1]["eval"]["log_path"]},
        )
    )
    components.append(
        ComponentStatus(
            name="feedback_training_loop",
            tier=Tier.WORKS if improved else Tier.PARTIAL,
            evidence=(
                f"Closed feedback loop used measured eval scores {trend} to resume training until {stop_reason}; "
                f"final_score={trend[-1]}, threshold={config.threshold}."
            ),
            artifacts={"latest_training_signal": str(local_dir / "training-signals" / f"iter-{len(loop_history):02d}.json")},
        )
    )
    customer_note = (
        "For customer-owned non-PushT robot data, rollout task-success and VLM/VLA scoring require that "
        "robot's matching rollout environment or the real robot. The Franka simulator is not a substitute; "
        "heldout dataset metrics remain real when no matching rollout env is supplied."
    )
    outer = {
        **outer_loop_decision(latest_feedback.score, config.threshold, config.checkpoint_uri or latest_checkpoint),
        "iterations": len(loop_history),
        "stop_reason": stop_reason,
        "trend": trend,
        "cumulative_steps": cumulative_steps,
        "customer_note": customer_note,
    }
    checkpoint_manifest = local_dir / "checkpoints" / "policy-checkpoint-manifest.json"
    _write_json(
        checkpoint_manifest,
        {
            "schema": "npa.sim_to_real.real_policy_checkpoint.v1",
            "run_id": config.run_id,
            "local_checkpoint": latest_checkpoint,
            "s3_checkpoint_uri": artifact_uris(config).get("checkpoint", ""),
            "validation": loop_history[-1]["checkpoint_validation"],
            "training_steps": cumulative_steps,
            "eval": loop_history[-1]["eval"],
        },
    )

    report = SimToRealReport(
        run_id=config.run_id,
        status=_overall_status(components),
        created_at=_utc_now(),
        config=_redacted_config(config),
        interfaces={
            "s3": {
                "endpoint": config.s3_endpoint,
                "bucket": config.s3_bucket,
                "prefix": config.s3_prefix,
            },
            "dataset": dataset_summary.to_dict(),
            "split": {
                "train_count": len(train_episodes),
                "heldout_count": len(heldout_episodes),
                "seed": config.seed,
            },
            "policy_container": policy_contract,
            "matching_rollout_env": rollout_env,
            "customer_note": customer_note,
        },
        artifacts={**artifact_uris(config), "local_report_dir": str(local_dir), "local_checkpoint": latest_checkpoint},
        components=components,
        feedback=latest_feedback,
        training_signal=latest_signal,
        outer_loop=outer,
    )
    rerun_output = _local_rrd_path(config, local_dir)
    try:
        logical_rerun = lerobot_dataset_logical_to_rerun(
            local_dataset,
            rerun_output,
            input_episode_indices=train_episodes[:1],
            rollout_episode_indices=heldout_episodes[:1],
            feedback_by_episode={int(heldout_episodes[0]): asdict(latest_feedback)},
            max_frames_per_episode=config.rerun_max_frames_per_episode,
        )
        rerun_status = ComponentStatus(
            name="rerun",
            tier=Tier.WORKS,
            evidence="Wrote and verified logical LeRobot/Rerun entities for input demos, heldout rollout, and feedback.",
            artifacts={
                "rrd": logical_rerun.output_rrd_path,
                "entity_counts": json.dumps(logical_rerun.entity_counts, sort_keys=True),
            },
        )
    except RerunAdapterError as exc:
        rerun_status = write_rerun_summary(report.to_dict(), rerun_output)
        rerun_status = ComponentStatus(
            name=rerun_status.name,
            tier=Tier.PARTIAL if rerun_status.tier != Tier.BLOCKED else Tier.BLOCKED,
            evidence=f"Logical LeRobot/Rerun adapter failed ({exc}); {rerun_status.evidence}",
            artifacts=rerun_status.artifacts,
        )
    report.components.append(rerun_status)
    report.status = _overall_status(report.components)
    report_path = local_dir / "sim-to-real-report.json"
    _write_json(report_path, report.to_dict())
    upload_status = _upload_run_artifacts(
        config,
        {
            "training_signal": local_dir / "training-signals" / f"iter-{len(loop_history):02d}.json",
            "dataset_summary": dataset_summary_path,
            "split": split_path,
            "checkpoint": Path(latest_checkpoint),
            "checkpoint_manifest": checkpoint_manifest,
            "eval": Path(loop_history[-1]["eval"]["output_dir"]),
            "report": report_path,
            "rrd": _local_rrd_path(config, local_dir),
        },
    )
    if upload_status is not None:
        report.components.append(upload_status)
        assertion_status = _assert_uploaded_real_artifacts(config)
        if assertion_status is not None:
            report.components.append(assertion_status)
        report.status = _overall_status(report.components)
        _write_json(report_path, report.to_dict())
        _upload_run_artifacts(config, {"report": report_path})
    return report


def main(argv: list[str] | None = None) -> int:
    """Module CLI used by the NPA CLI and SkyPilot YAML."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    local = subparsers.add_parser("local-smoke", help="Run the local structural spine.")
    _add_common_args(local)
    local.add_argument("--attempt-s3-roundtrip", action="store_true")
    local.add_argument("--report-path", type=Path, default=None)
    real = subparsers.add_parser("real-loop", help="Run the real LeRobot train/eval feedback loop.")
    _add_common_args(real)
    real.add_argument("--attempt-s3-roundtrip", action="store_true")
    real.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.command in {"local-smoke", "real-loop"}:
        config = build_config_from_env(
            run_id=args.run_id,
            s3_endpoint=args.s3_endpoint,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            input_data_uri=args.input_data_uri,
            dataset_repo_id=args.dataset_repo_id,
            dataset_revision=args.dataset_revision,
            policy_image=args.policy_image,
            sim_backend=args.sim_backend,
            eval_backend=args.eval_backend,
            feedback_source=args.feedback_source,
            split_fraction=args.split_fraction,
            env_count=args.env_count,
            episodes=args.episodes,
            train_steps=args.train_steps,
            eval_episodes=args.eval_episodes,
            threshold=args.threshold,
            seed=args.seed,
            gpu=args.gpu,
            max_training_iterations=args.max_training_iterations,
            train_step_budget=args.train_step_budget,
            min_eval_improvement=args.min_eval_improvement,
            policy_type=args.policy_type,
            train_batch_size=args.train_batch_size,
            train_num_workers=args.train_num_workers,
            policy_device=args.policy_device,
            vlm_eval_backend=args.vlm_eval_backend,
            vlm_eval_model=args.vlm_eval_model,
            vlm_eval_endpoint_url=args.vlm_eval_endpoint_url,
            vlm_eval_frame_selection=args.vlm_eval_frame_selection,
            vlm_eval_max_frames=args.vlm_eval_max_frames,
            vlm_eval_score=args.vlm_eval_score,
            trainer_command=args.trainer_command,
            checkpoint_uri=args.checkpoint_uri,
            rrd_path=args.rrd_path,
            rerun_max_frames_per_episode=args.rerun_max_frames_per_episode,
            output_dir=args.output_dir,
        )
        try:
            if args.command == "local-smoke":
                report = run_structural_spine(config, attempt_s3_roundtrip=args.attempt_s3_roundtrip)
            else:
                report = run_real_lerobot_loop(config, attempt_s3_roundtrip=args.attempt_s3_roundtrip)
        except SimToRealError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        payload = report.to_dict()
        if args.report_path:
            _write_json(args.report_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    return 2


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", default="")
    parser.add_argument("--s3-endpoint", default="")
    parser.add_argument("--s3-bucket", default="")
    parser.add_argument("--s3-prefix", default="")
    parser.add_argument("--input-data-uri", default="")
    parser.add_argument("--dataset-repo-id", default=DEFAULT_PUBLIC_LEROBOT_REPO)
    parser.add_argument("--dataset-revision", default=DEFAULT_PUBLIC_LEROBOT_REVISION)
    parser.add_argument("--policy-image", default="")
    parser.add_argument("--sim-backend", default=DEFAULT_SIM_BACKEND)
    parser.add_argument("--eval-backend", default=DEFAULT_EVAL_BACKEND)
    parser.add_argument("--feedback-source", default=DEFAULT_FEEDBACK_SOURCE)
    parser.add_argument("--split-fraction", type=float, default=DEFAULT_SPLIT_FRACTION)
    parser.add_argument("--env-count", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="H100:1")
    parser.add_argument("--max-training-iterations", type=int, default=DEFAULT_MAX_TRAINING_ITERATIONS)
    parser.add_argument("--train-step-budget", type=int, default=DEFAULT_TRAIN_STEP_BUDGET)
    parser.add_argument("--min-eval-improvement", type=float, default=DEFAULT_MIN_EVAL_IMPROVEMENT)
    parser.add_argument("--policy-type", default=DEFAULT_POLICY_TYPE)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--train-num-workers", type=int, default=DEFAULT_TRAIN_NUM_WORKERS)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--vlm-eval-backend", default=DEFAULT_VLM_EVAL_BACKEND)
    parser.add_argument("--vlm-eval-model", default=DEFAULT_VLM_EVAL_MODEL)
    parser.add_argument("--vlm-eval-endpoint-url", default="")
    parser.add_argument("--vlm-eval-frame-selection", default="keyframes")
    parser.add_argument("--vlm-eval-max-frames", type=int, default=4)
    parser.add_argument("--vlm-eval-score", type=float, default=None)
    parser.add_argument("--trainer-command", default="")
    parser.add_argument("--checkpoint-uri", default="")
    parser.add_argument("--rrd-path", default="")
    parser.add_argument("--rerun-max-frames-per-episode", type=int, default=DEFAULT_RERUN_MAX_FRAMES_PER_EPISODE)
    parser.add_argument("--output-dir", type=Path, default=None)


def _write_lerobot_rollout_fixture(output_dir: Path, episode_index: int, dataset_summary: Any) -> Path:
    """Write a tiny rollout fixture tied to a real LeRobot episode for vlm-eval."""

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = output_dir / "frame_000.ppm"
    frame.write_bytes(
        b"P6\n4 4\n255\n"
        + bytes(
            [
                40,
                80,
                160,
                60,
                120,
                200,
                80,
                160,
                220,
                100,
                180,
                240,
            ]
            * 4
        )
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "format": "npa_sim_to_real_lerobot_rollout_fixture_v1",
            "dataset_repo_id": dataset_summary.repo_id,
            "dataset_revision": dataset_summary.revision,
            "episode_index": int(episode_index),
            "frames": [frame.name],
            "state_keys": dataset_summary.state_keys,
            "action_keys": dataset_summary.action_keys,
            "camera_keys": dataset_summary.camera_keys,
        },
    )
    return output_dir


def _write_rollout_fixture(output_dir: Path, env: SimEnvSpec) -> Path:
    """Write a tiny generic rollout fixture that vlm-eval can score."""

    output_dir.mkdir(parents=True, exist_ok=True)
    # Binary PPM avoids a new dependency and is accepted by Pillow via vlm-eval.
    frame = output_dir / "frame_000.ppm"
    frame.write_bytes(
        b"P6\n4 4\n255\n"
        + bytes(
            [
                40,
                80,
                160,
                60,
                120,
                200,
                80,
                160,
                220,
                100,
                180,
                240,
            ]
            * 4
        )
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "format": "npa_sim_to_real_rollout_fixture_v1",
            "env_id": env.env_id,
            "instruction": env.instruction,
            "frames": [frame.name],
        },
    )
    return output_dir


def _matching_rollout_env(config: SimToRealConfig) -> str:
    """Return the real LeRobot rollout env matching the configured dataset."""

    backend = config.eval_backend.strip().lower()
    if backend in {"pusht", "gym-pusht", "lerobot-pusht"}:
        return "pusht"
    if backend in {"", "auto"} and config.dataset_repo_id == DEFAULT_PUBLIC_LEROBOT_REPO:
        return "pusht"
    return ""


def _feedback_from_eval(eval_payload: dict[str, Any], *, threshold: float) -> FeedbackResult:
    """Convert measured rollout eval metrics into feedback for the next train iteration."""

    score = float(eval_payload["score"])
    metric_name = str(eval_payload.get("metric_name") or "score")
    return FeedbackResult(
        success=score >= threshold,
        score=score,
        rationale=(
            f"Measured {metric_name}={score:.6f} against threshold={threshold:.6f} "
            f"in rollout backend {eval_payload.get('backend', '')}."
        ),
        critique="Continue training from the last checkpoint." if score < threshold else "Threshold reached.",
        source="rollout",
    )


def _assert_uploaded_real_artifacts(config: SimToRealConfig) -> ComponentStatus | None:
    """Assert uploaded .rrd and real checkpoint tensors exist in S3."""

    paths = artifact_uris(config)
    rrd_uri = paths.get("rrd", "")
    checkpoint_uri = paths.get("checkpoint", "")
    if not _is_s3_uri(rrd_uri) or not _is_s3_uri(checkpoint_uri):
        return None
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not access_key or not secret_key:
        return ComponentStatus(
            name="s3_real_artifact_assertions",
            tier=Tier.BLOCKED,
            evidence="AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY is not configured for S3 assertions.",
            artifacts={"rrd": rrd_uri, "checkpoint": checkpoint_uri},
        )
    try:
        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=config.s3_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        rrd_bucket, rrd_key = _split_s3_uri(rrd_uri)
        client.head_object(Bucket=rrd_bucket, Key=rrd_key)
        checkpoint_bucket, checkpoint_prefix = _split_s3_uri(checkpoint_uri)
        checkpoint_prefix = checkpoint_prefix.rstrip("/") + "/"
        page = client.list_objects_v2(Bucket=checkpoint_bucket, Prefix=checkpoint_prefix)
        keys = [item["Key"] for item in page.get("Contents", [])]
        weight_keys = [
            key
            for key in keys
            if key.endswith("/model.safetensors")
            or key.endswith("/pytorch_model.bin")
            or key.endswith("model.safetensors")
            or key.endswith("pytorch_model.bin")
        ]
        if not weight_keys:
            raise SimToRealError(f"No real checkpoint weight object found under {checkpoint_uri}")
    except Exception as exc:
        return ComponentStatus(
            name="s3_real_artifact_assertions",
            tier=Tier.BLOCKED,
            evidence=f"S3 assertions failed: {exc}",
            artifacts={"rrd": rrd_uri, "checkpoint": checkpoint_uri},
        )
    return ComponentStatus(
        name="s3_real_artifact_assertions",
        tier=Tier.WORKS,
        evidence="Asserted the uploaded .rrd object and at least one real checkpoint weight object in S3.",
        artifacts={"rrd": rrd_uri, "checkpoint": checkpoint_uri, "weight": f"s3://{checkpoint_bucket}/{weight_keys[0]}"},
    )


def _upload_run_artifacts(config: SimToRealConfig, local_files: dict[str, Path]) -> ComponentStatus | None:
    """Upload selected run artifacts to configured S3 paths when possible."""

    paths = artifact_uris(config)
    upload_targets: dict[str, str] = {}
    if "training_signal" in local_files and paths.get("training_signal"):
        upload_targets["training_signal"] = paths["training_signal"]
    if "dataset_summary" in local_files and paths.get("dataset_summary"):
        upload_targets["dataset_summary"] = paths["dataset_summary"]
    if "split" in local_files and paths.get("train_envs"):
        upload_targets["split"] = paths["train_envs"].rstrip("/") + "/episode-split.json"
    if "checkpoint" in local_files and paths.get("checkpoint"):
        checkpoint_uri = paths["checkpoint"]
        if checkpoint_uri.endswith("/") and not local_files["checkpoint"].is_dir():
            checkpoint_uri += "promoted-checkpoint.json"
        upload_targets["checkpoint"] = checkpoint_uri
    if "checkpoint_manifest" in local_files and paths.get("checkpoint"):
        upload_targets["checkpoint_manifest"] = (
            paths["checkpoint"].rstrip("/") + "/policy-checkpoint-manifest.json"
        )
    if "eval" in local_files and paths.get("rollouts"):
        upload_targets["eval"] = paths["rollouts"].rstrip("/") + "/eval/"
    if "report" in local_files and paths.get("report"):
        upload_targets["report"] = paths["report"]
    if "rrd" in local_files and paths.get("rrd"):
        upload_targets["rrd"] = paths["rrd"]
    upload_targets = {key: uri for key, uri in upload_targets.items() if _is_s3_uri(uri)}
    if not upload_targets:
        return None

    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not access_key or not secret_key:
        return ComponentStatus(
            name="s3_artifact_upload",
            tier=Tier.BLOCKED,
            evidence="AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY is not configured for artifact upload.",
            artifacts=upload_targets,
        )

    uploaded: dict[str, str] = {}
    try:
        client = StorageClient.from_environment(
            endpoint_url=config.s3_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        for name, uri in upload_targets.items():
            uploaded[name] = client.upload_path(str(local_files[name]), uri)
    except Exception as exc:
        return ComponentStatus(
            name="s3_artifact_upload",
            tier=Tier.BLOCKED,
            evidence=f"S3 artifact upload failed: {exc}",
            artifacts=upload_targets,
        )
    return ComponentStatus(
        name="s3_artifact_upload",
        tier=Tier.WORKS,
        evidence="Uploaded report, training signal, real checkpoint weights, eval outputs, and Rerun artifact where configured.",
        artifacts=uploaded,
    )


def _local_rrd_path(config: SimToRealConfig, local_dir: Path) -> Path:
    if config.rrd_path and not _is_s3_uri(config.rrd_path):
        return Path(config.rrd_path)
    return local_dir / f"{config.run_id}.rrd"


def _overall_status(components: list[ComponentStatus]) -> str:
    if any(component.tier == Tier.BLOCKED for component in components):
        return "blocked"
    if any(component.tier in {Tier.PARTIAL, Tier.SEAM} for component in components):
        return "partial"
    return "works"


def _redacted_config(config: SimToRealConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir) if config.output_dir else ""
    for key in list(payload):
        lowered = key.lower()
        if "secret" in lowered or "token" in lowered or "password" in lowered or "access_key" in lowered:
            payload[key] = "***" if payload[key] else ""
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_s3_uri(value: str) -> bool:
    return urlparse(value).scheme == "s3"


def _split_s3_uri(value: str) -> tuple[str, str]:
    parsed = urlparse(value)
    if parsed.scheme != "s3":
        raise SimToRealError(f"Expected s3:// URI, got: {value}")
    return parsed.netloc, parsed.path.lstrip("/")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
