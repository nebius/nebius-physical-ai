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
from npa.workbench.lerobot.policy_container import parse_feedback_batch, run_feedback_training_step
from npa.workflows.eval_backends import (
    DEFAULT_EVAL_BACKEND,
    EvalBackendError,
    EvalMetric,
    RolloutContext,
    evaluate_backend,
    get_eval_backend,
)
from npa.workflows.feedback import (
    DEFAULT_FEEDBACK_SOURCE,
    DEFAULT_FEEDBACK_TYPE,
    FeedbackPayload,
    FeedbackRequest,
    FeedbackSourceError,
    FeedbackType,
    adapt_feedback_to_training_signal,
    byo_feedback_contract,
    collect_feedback,
    get_feedback_source,
    parse_feedback_type,
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
DEFAULT_SIM_BACKEND = "genesis"
DEFAULT_VLM_EVAL_BACKEND = "stub"
DEFAULT_VLM_EVAL_MODEL = "vlm-eval-stub"
DEFAULT_SPLIT_FRACTION = 0.8
DEFAULT_THRESHOLD = 0.75
DEFAULT_RERUN_MAX_FRAMES_PER_EPISODE = 32
DEFAULT_GPU_TYPE = "H100:1"
DEFAULT_GPU_FAILOVER = "H200:1,L40S:1"
SUPPORTED_SKYPILOT_ACCELERATOR_EXAMPLES = ("H100:1", "H200:1", "L40S:1", "B200:8")
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
    feedback_type: str = DEFAULT_FEEDBACK_TYPE
    split_fraction: float = DEFAULT_SPLIT_FRACTION
    env_count: int = 10
    episodes: int = 4
    train_steps: int = 50
    eval_episodes: int = 2
    threshold: float = DEFAULT_THRESHOLD
    seed: int = 42
    gpu: str = DEFAULT_GPU_TYPE
    gpu_failover: str = DEFAULT_GPU_FAILOVER
    vlm_eval_backend: str = DEFAULT_VLM_EVAL_BACKEND
    vlm_eval_model: str = DEFAULT_VLM_EVAL_MODEL
    vlm_eval_endpoint_url: str = ""
    vlm_eval_frame_selection: str = "keyframes"
    vlm_eval_max_frames: int = 4
    vlm_eval_score: float | None = None
    trainer_command: str = ""
    byo_feedback_endpoint_url: str = ""
    byo_feedback_command: str = ""
    byo_feedback_mode: str = "provided-rollout"
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
        if self.vlm_eval_max_frames <= 0:
            raise SimToRealError(f"vlm_eval_max_frames must be positive, got {self.vlm_eval_max_frames}")
        if self.vlm_eval_score is not None and not 0.0 <= self.vlm_eval_score <= 1.0:
            raise SimToRealError(f"vlm_eval_score must be in [0, 1], got {self.vlm_eval_score}")
        if self.rerun_max_frames_per_episode <= 0:
            raise SimToRealError(
                f"rerun_max_frames_per_episode must be positive, got {self.rerun_max_frames_per_episode}"
            )
        try:
            get_eval_backend(self.eval_backend)
        except EvalBackendError as exc:
            raise SimToRealError(str(exc)) from exc
        try:
            get_feedback_source(self.feedback_source)
            parse_feedback_type(self.feedback_type)
        except FeedbackSourceError as exc:
            raise SimToRealError(str(exc)) from exc
        if not accelerator_candidates(self.gpu, self.gpu_failover):
            raise SimToRealError("gpu or gpu_failover must provide at least one accelerator candidate")


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


def accelerator_candidates(gpu: str = "", gpu_failover: str = "") -> list[str]:
    """Return ordered SkyPilot accelerator candidates from primary and failover strings."""

    candidates: list[str] = []
    for raw in (gpu, gpu_failover):
        for candidate in str(raw or "").split(","):
            normalized = normalize_accelerator(candidate)
            if normalized and normalized not in candidates:
                candidates.append(normalized)
    return candidates


def normalize_accelerator(candidate: str) -> str:
    """Normalize a SkyPilot accelerator token, defaulting bare GPU names to count 1."""

    value = str(candidate or "").strip()
    if not value:
        return ""
    if ":" in value or value.startswith("${"):
        return value
    return f"{value}:1"


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
    if vlm_score is None and os.environ.get("VLM_EVAL_SCORE"):
        vlm_score = os.environ["VLM_EVAL_SCORE"]
    gpu_override = overrides.get("gpu")
    legacy_gpu = os.environ.get("GPU") or ""
    primary_candidates = accelerator_candidates(
        str(gpu_override or os.environ.get("NPA_GPU_TYPE") or legacy_gpu or DEFAULT_GPU_TYPE)
    )
    gpu = primary_candidates[0] if primary_candidates else DEFAULT_GPU_TYPE
    extra_from_primary = primary_candidates[1:]
    failover_override = overrides.get("gpu_failover")
    if failover_override is not None:
        gpu_failover = str(failover_override)
    else:
        gpu_failover = os.environ.get("NPA_GPU_FAILOVER") or ",".join(extra_from_primary) or DEFAULT_GPU_FAILOVER
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
        feedback_type=str(overrides.get("feedback_type") or os.environ.get("FEEDBACK_TYPE") or DEFAULT_FEEDBACK_TYPE),
        split_fraction=float(overrides.get("split_fraction", os.environ.get("SPLIT_FRACTION", DEFAULT_SPLIT_FRACTION))),
        env_count=int(overrides.get("env_count", os.environ.get("ENV_COUNT", "10"))),
        episodes=int(overrides.get("episodes", os.environ.get("EPISODES", "4"))),
        train_steps=int(overrides.get("train_steps", os.environ.get("TRAIN_STEPS", "50"))),
        eval_episodes=int(overrides.get("eval_episodes", os.environ.get("EVAL_EPISODES", "2"))),
        threshold=float(overrides.get("threshold", os.environ.get("SUCCESS_THRESHOLD", DEFAULT_THRESHOLD))),
        seed=int(overrides.get("seed", os.environ.get("SEED", "42"))),
        gpu=gpu,
        gpu_failover=gpu_failover,
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
        byo_feedback_endpoint_url=str(
            overrides.get("byo_feedback_endpoint_url") or os.environ.get("BYO_FEEDBACK_ENDPOINT_URL") or ""
        ),
        byo_feedback_command=str(
            overrides.get("byo_feedback_command") or os.environ.get("BYO_FEEDBACK_COMMAND") or ""
        ),
        byo_feedback_mode=str(
            overrides.get("byo_feedback_mode") or os.environ.get("BYO_FEEDBACK_MODE") or "provided-rollout"
        ),
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
        "env": {
            "POLICY_IMAGE": config.policy_image or default_policy_image(),
            "INPUT_DATA_URI": config.input_data_uri,
            "CHECKPOINT_URI": paths.get("checkpoint", config.checkpoint_uri),
            "EVAL_BACKEND": config.eval_backend,
            "FEEDBACK_SOURCE": config.feedback_source,
            "FEEDBACK_TYPE": config.feedback_type,
            "NPA_GPU_TYPE": config.gpu,
            "NPA_GPU_FAILOVER": config.gpu_failover,
            "VLM_EVAL_BACKEND": config.vlm_eval_backend,
            "VLM_EVAL_MODEL": config.vlm_eval_model,
            "BYO_FEEDBACK_MODE": config.byo_feedback_mode,
            "BYO_FEEDBACK_ENDPOINT_URL": config.byo_feedback_endpoint_url,
            "BYO_FEEDBACK_COMMAND": config.byo_feedback_command,
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
            "feedback_type": {
                "dtype": "string",
                "required": True,
                "examples": ["scalar", "dense-per-step", "pass-fail", "critique", "preference"],
            },
            "value": {"dtype": "json", "required": True},
            "score": {"dtype": "float32", "range": [0.0, 1.0], "required": False},
            "success": {"dtype": "bool", "required": False},
            "rationale": {"dtype": "string", "required": False},
            "source": {"dtype": "string", "required": False, "examples": ["none", "sim-env", "vlm", "byo-container"]},
        },
        "byo_feedback_container": byo_feedback_contract(
            declared_type=config.feedback_type,
            mode=config.byo_feedback_mode,
            endpoint_url=config.byo_feedback_endpoint_url,
            command=config.byo_feedback_command,
        ),
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
    eval_metric: EvalMetric | None = None,
    checkpoint_uri: str = "",
) -> tuple[FeedbackResult, ComponentStatus]:
    """Collect typed feedback and return the legacy normalized result."""

    payload, component = collect_feedback_payload(
        config,
        rollout_path=rollout_path,
        output_path=output_path,
        task=task,
        eval_metric=eval_metric,
        checkpoint_uri=checkpoint_uri,
    )
    return _feedback_payload_to_result(payload), component


def collect_feedback_payload(
    config: SimToRealConfig,
    *,
    rollout_path: Path,
    output_path: Path,
    task: str,
    eval_metric: EvalMetric | None = None,
    checkpoint_uri: str = "",
) -> tuple[FeedbackPayload, ComponentStatus]:
    """Collect typed feedback through the configured feedback source."""

    request = FeedbackRequest(
        rollout_path=rollout_path,
        output_path=output_path,
        task=task,
        checkpoint_uri=checkpoint_uri or config.checkpoint_uri,
        threshold=config.threshold,
        feedback_type=parse_feedback_type(config.feedback_type),
        eval_metric=eval_metric,
        vlm_backend=config.vlm_eval_backend,
        vlm_model=config.vlm_eval_model,
        vlm_endpoint_url=config.vlm_eval_endpoint_url,
        vlm_frame_selection=config.vlm_eval_frame_selection,
        vlm_max_frames=config.vlm_eval_max_frames,
        vlm_score=config.vlm_eval_score,
        byo_endpoint_url=config.byo_feedback_endpoint_url,
        byo_command=config.byo_feedback_command,
        byo_mode=config.byo_feedback_mode,
    )
    payload, status = collect_feedback(config.feedback_source, request)
    return payload, _component_from_status(status)


def _feedback_payload_to_result(payload: FeedbackPayload) -> FeedbackResult:
    critique = ""
    if payload.feedback_type == FeedbackType.CRITIQUE and isinstance(payload.value, dict):
        critique = str(payload.value.get("critique") or "")
    return FeedbackResult(
        success=payload.success,
        score=payload.score,
        rationale=payload.rationale or critique or f"{payload.feedback_type.value} feedback",
        critique=critique,
        source=payload.source,
    )


def _component_from_status(status: Any) -> ComponentStatus:
    return ComponentStatus(
        name=status.name,
        tier=Tier(status.tier),
        evidence=status.evidence,
        artifacts=dict(status.artifacts),
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
    checkpoint_uri = config.checkpoint_uri or str(local_dir / "checkpoints" / "policy")
    rollout_frame = rollout_dir / "frame_000.ppm"
    eval_metric, eval_status = evaluate_backend(
        config.eval_backend,
        checkpoint_uri=checkpoint_uri,
        context=RolloutContext(
            rollout_path=rollout_dir,
            task="Complete the demonstrated manipulation task.",
            sim_backend=config.sim_backend,
            metrics={
                "vlm_score": config.vlm_eval_score,
                "heldout_score": config.vlm_eval_score,
                "heldout_episode_count": len(heldout_episodes),
            },
            state={
                "pc_success": None if config.vlm_eval_score is None else config.vlm_eval_score >= config.threshold,
            },
            frames=(rollout_frame,) if rollout_frame.exists() else (),
        ),
        threshold=config.threshold,
    )
    components.append(_component_from_status(eval_status))
    feedback_request = {
        "run_id": config.run_id,
        "heldout_episode": heldout_episodes[0],
        "rollout_path": str(rollout_dir),
        "feedback_output": str(vlm_output),
        "eval_backend": config.eval_backend,
        "eval_metric": asdict(eval_metric),
        "feedback_source": config.feedback_source,
        "feedback_type": config.feedback_type,
        "rubric": {"success": "task completed", "score": "float in [0,1]", "rationale": "brief reason"},
        "observation": {
            "images": dataset_summary.camera_keys,
            "language": "Complete the demonstrated manipulation task.",
            "state": dataset_summary.state_keys,
        },
    }
    feedback_payload, feedback_status = collect_feedback_payload(
        config,
        rollout_path=rollout_dir,
        output_path=vlm_output,
        task="Complete the demonstrated manipulation task.",
        eval_metric=eval_metric,
        checkpoint_uri=checkpoint_uri,
    )
    feedback = _feedback_payload_to_result(feedback_payload)
    components.append(feedback_status)
    training_signal = adapt_feedback_to_training_signal(feedback_payload)
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
            "eval_backend": {
                "name": eval_metric.name,
                "score": eval_metric.score,
                "passed": eval_metric.passed,
                "metadata": eval_metric.metadata,
            },
            "feedback": {
                "source": feedback_payload.source,
                "type": feedback_payload.feedback_type.value,
                "metadata": feedback_payload.metadata,
            },
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


def main(argv: list[str] | None = None) -> int:
    """Module CLI used by the NPA CLI and SkyPilot YAML."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    local = subparsers.add_parser("local-smoke", help="Run the local structural spine.")
    _add_common_args(local)
    local.add_argument("--attempt-s3-roundtrip", action="store_true")
    local.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.command == "local-smoke":
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
            feedback_type=args.feedback_type,
            split_fraction=args.split_fraction,
            env_count=args.env_count,
            episodes=args.episodes,
            train_steps=args.train_steps,
            eval_episodes=args.eval_episodes,
            threshold=args.threshold,
            seed=args.seed,
            gpu=args.gpu,
            gpu_failover=args.gpu_failover,
            vlm_eval_backend=args.vlm_eval_backend,
            vlm_eval_model=args.vlm_eval_model,
            vlm_eval_endpoint_url=args.vlm_eval_endpoint_url,
            vlm_eval_frame_selection=args.vlm_eval_frame_selection,
            vlm_eval_max_frames=args.vlm_eval_max_frames,
            vlm_eval_score=args.vlm_eval_score,
            trainer_command=args.trainer_command,
            byo_feedback_endpoint_url=args.byo_feedback_endpoint_url,
            byo_feedback_command=args.byo_feedback_command,
            byo_feedback_mode=args.byo_feedback_mode,
            checkpoint_uri=args.checkpoint_uri,
            rrd_path=args.rrd_path,
            rerun_max_frames_per_episode=args.rerun_max_frames_per_episode,
            output_dir=args.output_dir,
        )
        try:
            report = run_structural_spine(config, attempt_s3_roundtrip=args.attempt_s3_roundtrip)
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
    parser.add_argument("--feedback-type", default=DEFAULT_FEEDBACK_TYPE)
    parser.add_argument("--split-fraction", type=float, default=DEFAULT_SPLIT_FRACTION)
    parser.add_argument("--env-count", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default=DEFAULT_GPU_TYPE)
    parser.add_argument("--gpu-failover", default=DEFAULT_GPU_FAILOVER)
    parser.add_argument("--vlm-eval-backend", default=DEFAULT_VLM_EVAL_BACKEND)
    parser.add_argument("--vlm-eval-model", default=DEFAULT_VLM_EVAL_MODEL)
    parser.add_argument("--vlm-eval-endpoint-url", default="")
    parser.add_argument("--vlm-eval-frame-selection", default="keyframes")
    parser.add_argument("--vlm-eval-max-frames", type=int, default=4)
    parser.add_argument("--vlm-eval-score", type=float, default=None)
    parser.add_argument("--trainer-command", default="")
    parser.add_argument("--byo-feedback-endpoint-url", default="")
    parser.add_argument("--byo-feedback-command", default="")
    parser.add_argument("--byo-feedback-mode", choices=("provided-rollout", "self-rollout"), default="provided-rollout")
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
        if checkpoint_uri.endswith("/"):
            checkpoint_uri += "promoted-checkpoint.json"
        upload_targets["checkpoint"] = checkpoint_uri
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
            uploaded[name] = client.upload_file(str(local_files[name]), uri)
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
        evidence="Uploaded report, training signal, checkpoint marker, and Rerun artifact where configured.",
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
