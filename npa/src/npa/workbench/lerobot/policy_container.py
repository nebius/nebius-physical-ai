"""BYO LeRobot policy container runtime and feedback-training hook."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from npa.workbench.training_config import (
    TrainingConfig,
    TrainingConfigError,
    build_training_config,
    format_override,
    upload_checkpoint_path,
    wandb_overrides,
)


DEFAULT_POLICY_CHECKPOINT = "lerobot/diffusion_pusht"
DEFAULT_ACTION_DIM = 2
DEFAULT_POLICY_TYPE = "act"
DEFAULT_TRAIN_BATCH_SIZE = 8
DEFAULT_TRAIN_NUM_WORKERS = 4
DEFAULT_TRAIN_LOG_FREQ = 10
DEFAULT_TRAIN_TIMEOUT_SECONDS = 43200
DEFAULT_EVAL_TIMEOUT_SECONDS = 7200
REAL_WEIGHT_FILENAMES = ("model.safetensors", "pytorch_model.bin")

# Request-supplied output directories are confined under this root so an
# unauthenticated /feedback/train-step caller cannot write adapter checkpoints to
# arbitrary filesystem paths (path traversal / arbitrary file write).
POLICY_OUTPUT_ROOT_ENV = "NPA_POLICY_OUTPUT_ROOT"
DEFAULT_POLICY_OUTPUT_ROOT = "/tmp/npa-lerobot"


def _policy_output_root() -> Path:
    return Path(os.environ.get(POLICY_OUTPUT_ROOT_ENV, "") or DEFAULT_POLICY_OUTPUT_ROOT).resolve()


def jail_output_dir(raw: str | None, *, default_name: str) -> Path:
    """Resolve a request-supplied output dir confined to the policy output root.

    Rejects absolute paths and ``..`` traversal that would escape the jail.
    """
    root = _policy_output_root()
    root.mkdir(parents=True, exist_ok=True)
    candidate = (raw or "").strip()
    if not candidate:
        return root / default_name
    relative = Path(candidate)
    if relative.is_absolute():
        try:
            relative = relative.relative_to(root)
        except ValueError:
            raise PolicyContainerError(
                f"output_dir must be a relative path under {root}; got absolute path {candidate!r}"
            ) from None
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise PolicyContainerError(f"output_dir {candidate!r} escapes the allowed output root {root}")
    return resolved


class PolicyContainerError(Exception):
    """Raised when policy-container inference or feedback training fails."""


@dataclass(frozen=True)
class FeedbackItem:
    """Existing vlm_eval-compatible feedback item."""

    success: bool
    score: float
    rationale: str
    source: str = "vlm"


@dataclass(frozen=True)
class FeedbackUpdateResult:
    """Result from one feedback-driven trainer hook step."""

    status: str
    backend: str
    steps: int
    loss_before: float
    loss_after: float
    weight_before: float
    weight_after: float
    checkpoint_path: str
    rationale_count: int
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VlmSignalStep:
    """One step of the VLM-derived RL signal schema."""

    step: int
    reward: float
    advantage: float | None = None
    target: dict[str, Any] = field(default_factory=dict)
    critique_text: str = ""
    error_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class VlmRlSignal:
    """Per-rollout VLM-derived signal consumed by the reference trainer fork."""

    rollout_id: str
    per_step: list[VlmSignalStep]
    schema: str = "npa.sim2real.rl_signal.v1"
    source: str = "vlm"


@dataclass(frozen=True)
class VlmSignalUpdateResult:
    """Result from one VLM-signal policy update step."""

    status: str
    backend: str
    steps: int
    loss_before: float
    loss_after: float
    reward_head_before: float
    reward_head_after: float
    policy_output_before: list[float]
    policy_output_after: list[float]
    policy_delta_l2: float
    mean_reward: float
    mean_advantage: float
    checkpoint_path: str
    signal_count: int
    control: bool
    loss_integration_point: str
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VlmSignalUpdateResult:
        """Parse a BYO trainer-command JSON result into a structured update.

        Required fields (a BYO trainer-command MUST emit these): ``reward_head_after``,
        ``policy_output_after`` (non-empty list), and ``policy_delta_l2``. Every other
        field falls back to a safe default so a minimal customer trainer hook still
        produces a usable, attributable update record.
        """

        if not isinstance(payload, dict):
            raise PolicyContainerError(
                "VlmSignalUpdateResult.from_dict requires a JSON object payload"
            )
        for required in ("reward_head_after", "policy_output_after", "policy_delta_l2"):
            if required not in payload:
                raise PolicyContainerError(
                    f"trainer update payload missing required field: {required}"
                )
        policy_output_after = payload["policy_output_after"]
        if not isinstance(policy_output_after, list) or not policy_output_after:
            raise PolicyContainerError(
                "trainer update policy_output_after must be a non-empty list"
            )
        policy_output_after = [float(item) for item in policy_output_after]
        raw_before = payload.get("policy_output_before")
        if isinstance(raw_before, list) and raw_before:
            policy_output_before = [float(item) for item in raw_before]
        else:
            policy_output_before = [0.0 for _ in policy_output_after]
        loss_before = float(payload.get("loss_before", 0.0))
        loss_after = float(payload.get("loss_after", loss_before))
        return cls(
            status=str(payload.get("status", "updated")),
            backend=str(payload.get("backend", "byo_command")),
            steps=int(payload.get("steps", 1)),
            loss_before=loss_before,
            loss_after=loss_after,
            reward_head_before=float(payload.get("reward_head_before", 0.0)),
            reward_head_after=float(payload["reward_head_after"]),
            policy_output_before=policy_output_before,
            policy_output_after=policy_output_after,
            policy_delta_l2=float(payload["policy_delta_l2"]),
            mean_reward=float(payload.get("mean_reward", 0.0)),
            mean_advantage=float(payload.get("mean_advantage", 0.0)),
            checkpoint_path=str(payload.get("checkpoint_path", "")),
            signal_count=int(payload.get("signal_count", 0)),
            control=bool(payload.get("control", False)),
            loss_integration_point=str(
                payload.get("loss_integration_point", "byo_trainer_command")
            ),
            duration_ms=float(payload.get("duration_ms", 0.0)),
        )


@dataclass(frozen=True)
class LeRobotImportResult:
    """Runtime LeRobot import proof."""

    status: str
    version: str
    module_path: str
    dataset_class: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointValidationResult:
    """Proof that a LeRobot checkpoint contains loadable real parameters."""

    status: str
    checkpoint_path: str
    weight_file: str
    tensor_count: int
    parameter_count: int
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeRobotTrainingResult:
    """Result from one real LeRobot training invocation."""

    status: str
    command: list[str]
    output_dir: str
    checkpoint_path: str
    steps: int
    resume: bool
    log_path: str
    duration_seconds: float
    exit_code: int
    checkpoint_validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeRobotEvalResult:
    """Result from one measured LeRobot rollout evaluation."""

    status: str
    backend: str
    command: list[str]
    output_dir: str
    eval_info_path: str
    score: float
    metric_name: str
    pc_success: float | None
    avg_sum_reward: float | None
    avg_max_reward: float | None
    n_episodes: int | None
    log_path: str
    duration_seconds: float
    exit_code: int
    raw_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LeRobotPolicyRuntime:
    """Thin runtime wrapper around a loaded LeRobot policy checkpoint."""

    def __init__(self, checkpoint_path: Path | str | None = None) -> None:
        raw_checkpoint = checkpoint_path or os.environ.get("NPA_POLICY_CHECKPOINT", "")
        if not str(raw_checkpoint):
            raise PolicyContainerError("NPA_POLICY_CHECKPOINT or --checkpoint is required for serving")
        resolved = Path(raw_checkpoint).expanduser()
        self.checkpoint_path = resolved
        self.validation = validate_lerobot_checkpoint(self.checkpoint_path)

    def infer(self, observation: dict[str, Any]) -> list[float]:
        raise PolicyContainerError(
            "HTTP inference requires LeRobot policy adapter wiring for this checkpoint. "
            "Use the real training and rollout-eval commands for end-to-end validation."
        )

    def rollout(self, observations: list[dict[str, Any]]) -> list[list[float]]:
        raise PolicyContainerError(
            "HTTP rollouts require a matching LeRobot environment. "
            "Use lerobot-eval through the real eval command for measured rollout success."
        )


def assert_lerobot_importable() -> LeRobotImportResult:
    """Import LeRobot and LeRobotDataset in the runtime environment."""

    try:
        lerobot = importlib.import_module("lerobot")
        dataset_module = importlib.import_module("lerobot.datasets.lerobot_dataset")
        dataset_cls = getattr(dataset_module, "LeRobotDataset")
    except Exception as exc:
        raise PolicyContainerError(f"LeRobot import failed: {exc}") from exc
    return LeRobotImportResult(
        status="ok",
        version=str(getattr(lerobot, "__version__", "")),
        module_path=str(getattr(lerobot, "__file__", "")),
        dataset_class=f"{dataset_cls.__module__}.{dataset_cls.__name__}",
    )


def build_lerobot_train_command(
    *,
    dataset_path: Path | str,
    output_dir: Path | str,
    steps: int,
    dataset_repo_id: str,
    policy_type: str = DEFAULT_POLICY_TYPE,
    batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    num_workers: int = DEFAULT_TRAIN_NUM_WORKERS,
    device: str = "cuda",
    save_freq: int | None = None,
    eval_freq: int = 1_000_000,
    log_freq: int = DEFAULT_TRAIN_LOG_FREQ,
    resume: bool = False,
    extra_args: list[str] | None = None,
    training_config: TrainingConfig | None = None,
) -> list[str]:
    """Build a real `lerobot-train` command for a local LeRobotDataset."""

    config = training_config or TrainingConfig()
    if steps <= 0:
        raise PolicyContainerError(f"steps must be positive, got {steps}")
    if batch_size <= 0:
        raise PolicyContainerError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise PolicyContainerError(f"num_workers must be non-negative, got {num_workers}")
    dataset_root = Path(dataset_path)
    repo_id = dataset_repo_id or dataset_root.name
    if save_freq is None:
        save_freq = max(1, steps)
    cmd = [
        "lerobot-train",
        f"--policy.type={policy_type}",
        "--policy.push_to_hub=false",
        f"--policy.device={device}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        f"--output_dir={output_dir}",
        f"--steps={steps}",
        f"--save_freq={save_freq}",
        f"--eval_freq={eval_freq}",
        f"--log_freq={log_freq}",
        f"--batch_size={batch_size}",
        f"--num_workers={num_workers}",
    ]
    cmd.extend(wandb_overrides(config.wandb, style="cli"))
    if resume:
        cmd.append("--resume=true")
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(format_override(override, style="cli") for override in config.overrides)
    return cmd


def run_lerobot_training(
    *,
    dataset_path: Path | str,
    output_dir: Path | str,
    steps: int,
    dataset_repo_id: str,
    policy_type: str = DEFAULT_POLICY_TYPE,
    batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    num_workers: int = DEFAULT_TRAIN_NUM_WORKERS,
    device: str = "cuda",
    resume: bool = False,
    log_path: Path | str | None = None,
    timeout_seconds: int = DEFAULT_TRAIN_TIMEOUT_SECONDS,
    extra_args: list[str] | None = None,
    training_config: TrainingConfig | None = None,
) -> LeRobotTrainingResult:
    """Run real LeRobot policy training and validate the resulting checkpoint."""

    config = training_config or TrainingConfig()
    assert_lerobot_importable()
    if shutil.which("lerobot-train") is None:
        raise PolicyContainerError("lerobot-train was not found on PATH")
    dataset_root = Path(dataset_path)
    if not (dataset_root / "meta" / "info.json").exists():
        raise PolicyContainerError(f"LeRobot dataset is missing meta/info.json: {dataset_root}")
    out = Path(output_dir)
    if out.exists() and not resume:
        raise PolicyContainerError(f"output_dir already exists; pass resume=True to continue: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    log = Path(log_path) if log_path else out.parent / f"{out.name}.train.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = build_lerobot_train_command(
        dataset_path=dataset_root,
        output_dir=out,
        steps=steps,
        dataset_repo_id=dataset_repo_id,
        policy_type=policy_type,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        save_freq=steps,
        resume=resume,
        extra_args=extra_args,
        training_config=config,
    )
    start = time.time()
    with log.open("w", encoding="utf-8") as handle:
        handle.write("+ " + _shell_join(command) + "\n")
        handle.flush()
        proc = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    duration = round(time.time() - start, 2)
    checkpoint = _find_lerobot_checkpoint(out)
    validation: CheckpointValidationResult | None = None
    if checkpoint is not None:
        validation = validate_lerobot_checkpoint(checkpoint)
    if proc.returncode != 0 or checkpoint is None or validation is None:
        raise PolicyContainerError(
            f"lerobot-train failed or did not produce loadable real weights "
            f"(exit={proc.returncode}, checkpoint={checkpoint}, log={log})"
        )
    upload_checkpoint_path(checkpoint, config)
    return LeRobotTrainingResult(
        status="success",
        command=command,
        output_dir=str(out),
        checkpoint_path=str(checkpoint),
        steps=steps,
        resume=resume,
        log_path=str(log),
        duration_seconds=duration,
        exit_code=proc.returncode,
        checkpoint_validation=validation.to_dict(),
    )


def build_lerobot_eval_command(
    *,
    checkpoint_path: Path | str,
    output_dir: Path | str,
    env_type: str,
    episodes: int,
    device: str = "cuda",
    env_task: str = "",
) -> list[str]:
    """Build a real `lerobot-eval` rollout command."""

    if episodes <= 0:
        raise PolicyContainerError(f"episodes must be positive, got {episodes}")
    cmd = [
        "lerobot-eval",
        f"--policy.path={checkpoint_path}",
        f"--env.type={env_type}",
        f"--output_dir={output_dir}",
        "--eval.batch_size=1",
        f"--eval.n_episodes={episodes}",
        f"--policy.device={device}",
        "--policy.use_amp=false",
    ]
    if env_task:
        cmd.append(f"--env.task={env_task}")
    return cmd


def run_lerobot_eval(
    *,
    checkpoint_path: Path | str,
    output_dir: Path | str,
    env_type: str,
    episodes: int,
    device: str = "cuda",
    env_task: str = "",
    log_path: Path | str | None = None,
    timeout_seconds: int = DEFAULT_EVAL_TIMEOUT_SECONDS,
) -> LeRobotEvalResult:
    """Run a measured rollout eval in the matching LeRobot environment."""

    assert_lerobot_importable()
    if shutil.which("lerobot-eval") is None:
        raise PolicyContainerError("lerobot-eval was not found on PATH")
    checkpoint = Path(checkpoint_path)
    validate_lerobot_checkpoint(checkpoint)
    out = Path(output_dir)
    if out.exists():
        raise PolicyContainerError(f"eval output_dir already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    log = Path(log_path) if log_path else out.parent / f"{out.name}.eval.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = build_lerobot_eval_command(
        checkpoint_path=checkpoint,
        output_dir=out,
        env_type=env_type,
        episodes=episodes,
        device=device,
        env_task=env_task,
    )
    start = time.time()
    with log.open("w", encoding="utf-8") as handle:
        handle.write("+ " + _shell_join(command) + "\n")
        handle.flush()
        proc = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    duration = round(time.time() - start, 2)
    eval_info_path = out / "eval_info.json"
    if proc.returncode != 0 or not eval_info_path.exists():
        raise PolicyContainerError(
            f"lerobot-eval failed or did not produce eval_info.json "
            f"(exit={proc.returncode}, eval_info={eval_info_path}, log={log})"
        )
    metrics = json.loads(eval_info_path.read_text(encoding="utf-8"))
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
    pc_success = _optional_float(overall.get("pc_success"))
    avg_sum_reward = _optional_float(overall.get("avg_sum_reward"))
    avg_max_reward = _optional_float(overall.get("avg_max_reward"))
    score = pc_success if pc_success is not None else _normalized_reward(avg_sum_reward)
    if score is None:
        raise PolicyContainerError(f"eval_info.json does not contain pc_success or avg_sum_reward: {eval_info_path}")
    n_episodes = overall.get("n_episodes")
    return LeRobotEvalResult(
        status="success",
        backend=env_type,
        command=command,
        output_dir=str(out),
        eval_info_path=str(eval_info_path),
        score=round(float(score), 6),
        metric_name="pc_success" if pc_success is not None else "normalized_avg_sum_reward",
        pc_success=pc_success,
        avg_sum_reward=avg_sum_reward,
        avg_max_reward=avg_max_reward,
        n_episodes=int(n_episodes) if n_episodes is not None else None,
        log_path=str(log),
        duration_seconds=duration,
        exit_code=proc.returncode,
        raw_metrics=metrics,
    )


def validate_lerobot_checkpoint(checkpoint_path: Path | str) -> CheckpointValidationResult:
    """Assert a checkpoint has real, loadable tensor weights."""

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists() or not checkpoint.is_dir():
        raise PolicyContainerError(f"checkpoint directory does not exist: {checkpoint}")
    weight_file = _find_weight_file(checkpoint)
    if weight_file is None:
        raise PolicyContainerError(f"checkpoint has no real weight file under {checkpoint}")
    if weight_file.name == "model.safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise PolicyContainerError("safetensors is required to validate model.safetensors") from exc
        tensors = load_file(str(weight_file), device="cpu")
    else:
        try:
            import torch
        except ImportError as exc:
            raise PolicyContainerError("torch is required to validate pytorch_model.bin") from exc
        payload = torch.load(str(weight_file), map_location="cpu")
        tensors = payload if isinstance(payload, dict) else {}
    tensor_count = 0
    parameter_count = 0
    for value in tensors.values():
        if hasattr(value, "numel"):
            tensor_count += 1
            parameter_count += int(value.numel())
    if tensor_count <= 0 or parameter_count <= 0:
        raise PolicyContainerError(f"checkpoint weight file contains no tensors: {weight_file}")
    return CheckpointValidationResult(
        status="loadable",
        checkpoint_path=str(checkpoint),
        weight_file=str(weight_file),
        tensor_count=tensor_count,
        parameter_count=parameter_count,
        bytes=int(weight_file.stat().st_size),
    )


def parse_feedback_batch(payload: dict[str, Any] | list[dict[str, Any]]) -> list[FeedbackItem]:
    """Parse `{success, score, rationale}` feedback records."""

    records = payload if isinstance(payload, list) else payload.get("feedback", [payload])
    if not isinstance(records, list) or not records:
        raise PolicyContainerError("feedback payload must contain at least one record")
    parsed: list[FeedbackItem] = []
    for item in records:
        if not isinstance(item, dict):
            raise PolicyContainerError("feedback records must be objects")
        missing = [key for key in ("success", "score", "rationale") if key not in item]
        if missing:
            raise PolicyContainerError(f"feedback record missing required keys: {', '.join(missing)}")
        score = float(item["score"])
        if not 0.0 <= score <= 1.0:
            raise PolicyContainerError(f"feedback score must be in [0, 1], got {score}")
        rationale = str(item["rationale"]).strip()
        if not rationale:
            raise PolicyContainerError("feedback rationale must not be empty")
        parsed.append(
            FeedbackItem(
                success=bool(item["success"]),
                score=score,
                rationale=rationale,
                source=str(item.get("source") or "vlm"),
            )
        )
    return parsed


def parse_vlm_signal_batch(payload: dict[str, Any] | list[dict[str, Any]]) -> list[VlmRlSignal]:
    """Parse the Stage 9 VLM-to-RL training signal schema.

    Accepted inputs are a single ``npa.sim2real.rl_signal.v1`` object, a list of
    signal objects, or a wrapper object with ``signals`` or ``training_signal``.
    """

    if isinstance(payload, list):
        records: Any = payload
    elif str(payload.get("schema", "")).startswith("npa.sim2real.rl_signal."):
        records = [payload]
    else:
        records = payload.get("signals") or payload.get("training_signal") or payload.get("feedback")
    if not isinstance(records, list) or not records:
        raise PolicyContainerError("VLM signal payload must contain at least one signal record")

    parsed: list[VlmRlSignal] = []
    for record in records:
        if not isinstance(record, dict):
            raise PolicyContainerError("VLM signal records must be objects")
        rollout_id = str(record.get("rollout_id") or "").strip()
        if not rollout_id:
            raise PolicyContainerError("VLM signal record missing rollout_id")
        raw_steps = record.get("per_step")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise PolicyContainerError(f"VLM signal {rollout_id} must include non-empty per_step")
        steps: list[VlmSignalStep] = []
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                raise PolicyContainerError("VLM signal per_step items must be objects")
            if "step" not in raw_step or "reward" not in raw_step:
                raise PolicyContainerError("VLM signal per_step items require step and reward")
            reward = _bounded_float(raw_step["reward"], lower=-1.0, upper=1.0)
            advantage = raw_step.get("advantage")
            tags = raw_step.get("error_tags") or []
            if not isinstance(tags, list):
                raise PolicyContainerError("VLM signal error_tags must be a list when present")
            target = raw_step.get("target") or {}
            if not isinstance(target, dict):
                raise PolicyContainerError("VLM signal target must be an object when present")
            steps.append(
                VlmSignalStep(
                    step=int(raw_step["step"]),
                    reward=reward,
                    advantage=(
                        None
                        if advantage is None
                        else _bounded_float(advantage, lower=-2.0, upper=2.0)
                    ),
                    target=target,
                    critique_text=str(raw_step.get("critique_text") or ""),
                    error_tags=tuple(str(tag) for tag in tags),
                )
            )
        parsed.append(
            VlmRlSignal(
                rollout_id=rollout_id,
                per_step=steps,
                schema=str(record.get("schema") or "npa.sim2real.rl_signal.v1"),
                source=str(record.get("source") or "vlm"),
            )
        )
    return parsed


def run_feedback_training_step(
    feedback: list[FeedbackItem],
    *,
    output_dir: Path,
    learning_rate: float = 0.1,
    initial_weight: float = 0.0,
) -> FeedbackUpdateResult:
    """Run one real optimizer step for the custom LeRobot feedback hook.

    The hook is intentionally small and research-grade: it converts VLM/VLA
    feedback into a scalar reward target and updates a feedback adapter weight.
    Containers with torch installed use autograd/SGD; local structural tests use
    an equivalent deterministic Python update.
    """

    if not feedback:
        raise PolicyContainerError("feedback batch is empty")
    if learning_rate <= 0:
        raise PolicyContainerError(f"learning_rate must be positive, got {learning_rate}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = sum(_feedback_reward(item) for item in feedback) / float(len(feedback))
    start = time.time()

    try:
        import torch

        weight = torch.nn.Parameter(torch.tensor([float(initial_weight)], dtype=torch.float32))
        optimizer = torch.optim.SGD([weight], lr=float(learning_rate))
        before_loss = torch.square(torch.sigmoid(weight) - torch.tensor([target], dtype=torch.float32)).mean()
        weight_before = float(weight.detach().item())
        optimizer.zero_grad()
        before_loss.backward()
        optimizer.step()
        after_loss = torch.square(torch.sigmoid(weight) - torch.tensor([target], dtype=torch.float32)).mean()
        checkpoint_path = output_dir / "feedback_adapter.pt"
        torch.save(
            {
                "schema": "npa.lerobot.feedback_adapter.v1",
                "weight": weight.detach().cpu(),
                "target": target,
                "feedback": [asdict(item) for item in feedback],
            },
            checkpoint_path,
        )
        backend = "torch"
        loss_before = float(before_loss.detach().item())
        loss_after = float(after_loss.detach().item())
        weight_after = float(weight.detach().item())
    except ImportError:
        prediction = _sigmoid(initial_weight)
        loss_before = (prediction - target) ** 2
        gradient = 2.0 * (prediction - target) * prediction * (1.0 - prediction)
        weight_after = float(initial_weight) - float(learning_rate) * gradient
        loss_after = (_sigmoid(weight_after) - target) ** 2
        weight_before = float(initial_weight)
        checkpoint_path = output_dir / "feedback_adapter.json"
        checkpoint_path.write_text(
            json.dumps(
                {
                    "schema": "npa.lerobot.feedback_adapter.v1",
                    "weight": weight_after,
                    "target": target,
                    "feedback": [asdict(item) for item in feedback],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        backend = "python"

    return FeedbackUpdateResult(
        status="updated",
        backend=backend,
        steps=1,
        loss_before=round(loss_before, 8),
        loss_after=round(loss_after, 8),
        weight_before=round(weight_before, 8),
        weight_after=round(weight_after, 8),
        checkpoint_path=str(checkpoint_path),
        rationale_count=len(feedback),
        duration_ms=round((time.time() - start) * 1000.0, 2),
    )


def run_vlm_signal_training_step(
    signals: list[VlmRlSignal],
    *,
    output_dir: Path,
    learning_rate: float = 0.05,
    signal_loss_weight: float = 1.0,
    initial_reward_head: float = 0.0,
    initial_action_bias: float = 0.0,
    control: bool = False,
) -> VlmSignalUpdateResult:
    """Run one reference VLM-in-the-loop LeRobot update.

    The integration point for a real LeRobot fork is immediately after the
    policy forward pass and before ``optimizer.step()``:

    ``loss = imitation_loss + signal_loss_weight * corrective_mse - advantage * policy_logit_proxy``

    This reference hook updates a compact reward head and action-bias adapter so
    the signal path is executable without assuming a specific LeRobot policy
    class. A production fork can replace the adapter tensors with the policy
    action head and keep the same signal schema.
    """

    if not signals:
        raise PolicyContainerError("VLM signal batch is empty")
    if learning_rate <= 0:
        raise PolicyContainerError(f"learning_rate must be positive, got {learning_rate}")
    if signal_loss_weight < 0:
        raise PolicyContainerError(f"signal_loss_weight must be non-negative, got {signal_loss_weight}")
    output_dir.mkdir(parents=True, exist_ok=True)
    flat_steps = [step for signal in signals for step in signal.per_step]
    if not flat_steps:
        raise PolicyContainerError("VLM signal batch contains no steps")

    rewards = [float(step.reward) for step in flat_steps]
    advantages = [
        float(step.advantage)
        if step.advantage is not None
        else float(step.reward) - (sum(rewards) / float(len(rewards)))
        for step in flat_steps
    ]
    action_targets = [_action_delta(step.target) for step in flat_steps]
    action_dim = max((len(item) for item in action_targets), default=1)
    if action_dim <= 0:
        action_dim = 1
    target_vector = _mean_action_target(action_targets, action_dim=action_dim)
    mean_reward = sum(rewards) / float(len(rewards))
    mean_advantage = sum(advantages) / float(len(advantages))
    reward_target = max(0.0, min(1.0, (mean_reward + 1.0) / 2.0))

    if control:
        target_vector = [0.0 for _ in range(action_dim)]
        mean_advantage = 0.0
        reward_target = 0.5

    start = time.time()
    loss_integration_point = (
        "LeRobot trainer fork: add signal_loss_weight * corrective_mse and "
        "advantage-weighted policy term after the policy forward pass, before optimizer.step()."
    )
    try:
        import torch

        reward_head = torch.nn.Parameter(torch.tensor([float(initial_reward_head)], dtype=torch.float32))
        action_bias = torch.nn.Parameter(
            torch.full((action_dim,), float(initial_action_bias), dtype=torch.float32)
        )
        optimizer = torch.optim.SGD([reward_head, action_bias], lr=float(learning_rate))
        target_reward_tensor = torch.tensor([reward_target], dtype=torch.float32)
        target_action_tensor = torch.tensor(target_vector, dtype=torch.float32)

        def loss_value() -> Any:
            predicted_reward = torch.sigmoid(reward_head)
            predicted_action = torch.tanh(action_bias)
            reward_loss = torch.square(predicted_reward - target_reward_tensor).mean()
            corrective_loss = torch.square(predicted_action - target_action_tensor).mean()
            advantage_term = -float(mean_advantage) * predicted_action.mean()
            return reward_loss + float(signal_loss_weight) * corrective_loss + 0.1 * advantage_term

        before_loss = loss_value()
        reward_head_before = float(reward_head.detach().item())
        policy_output_before = [float(item) for item in torch.tanh(action_bias.detach()).tolist()]
        optimizer.zero_grad()
        before_loss.backward()
        optimizer.step()
        after_loss = loss_value()
        reward_head_after = float(reward_head.detach().item())
        policy_output_after = [float(item) for item in torch.tanh(action_bias.detach()).tolist()]
        checkpoint_path = output_dir / "vlm_signal_adapter.pt"
        torch.save(
            {
                "schema": "npa.lerobot.vlm_signal_adapter.v1",
                "reward_head": reward_head.detach().cpu(),
                "action_bias": action_bias.detach().cpu(),
                "target_reward": reward_target,
                "target_action_delta": target_vector,
                "mean_reward": mean_reward,
                "mean_advantage": mean_advantage,
                "control": control,
                "signals": [asdict(signal) for signal in signals],
                "loss_integration_point": loss_integration_point,
            },
            checkpoint_path,
        )
        backend = "torch"
        loss_before = float(before_loss.detach().item())
        loss_after = float(after_loss.detach().item())
    except ImportError:
        reward_head_before = float(initial_reward_head)
        policy_output_before = [_tanh(float(initial_action_bias)) for _ in range(action_dim)]
        reward_prediction = _sigmoid(reward_head_before)
        reward_gradient = 2.0 * (reward_prediction - reward_target) * reward_prediction * (1.0 - reward_prediction)
        reward_head_after = reward_head_before - float(learning_rate) * reward_gradient
        policy_output_after: list[float] = []
        for before, target in zip(policy_output_before, target_vector, strict=False):
            gradient = 2.0 * (before - target) * (1.0 - before * before)
            gradient -= 0.1 * float(mean_advantage) / float(action_dim)
            bias_after = float(initial_action_bias) - float(learning_rate) * float(signal_loss_weight) * gradient
            policy_output_after.append(_tanh(bias_after))
        loss_before = _reference_signal_loss(
            reward_prediction=reward_prediction,
            reward_target=reward_target,
            policy_output=policy_output_before,
            target_vector=target_vector,
            mean_advantage=mean_advantage,
            signal_loss_weight=signal_loss_weight,
        )
        loss_after = _reference_signal_loss(
            reward_prediction=_sigmoid(reward_head_after),
            reward_target=reward_target,
            policy_output=policy_output_after,
            target_vector=target_vector,
            mean_advantage=mean_advantage,
            signal_loss_weight=signal_loss_weight,
        )
        checkpoint_path = output_dir / "vlm_signal_adapter.json"
        checkpoint_path.write_text(
            json.dumps(
                {
                    "schema": "npa.lerobot.vlm_signal_adapter.v1",
                    "reward_head": reward_head_after,
                    "policy_output": policy_output_after,
                    "target_reward": reward_target,
                    "target_action_delta": target_vector,
                    "mean_reward": mean_reward,
                    "mean_advantage": mean_advantage,
                    "control": control,
                    "signals": [asdict(signal) for signal in signals],
                    "loss_integration_point": loss_integration_point,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        backend = "python"

    policy_delta = math.sqrt(
        sum((after - before) ** 2 for before, after in zip(policy_output_before, policy_output_after, strict=False))
    )
    return VlmSignalUpdateResult(
        status="updated",
        backend=backend,
        steps=1,
        loss_before=round(loss_before, 8),
        loss_after=round(loss_after, 8),
        reward_head_before=round(reward_head_before, 8),
        reward_head_after=round(reward_head_after, 8),
        policy_output_before=[round(item, 8) for item in policy_output_before],
        policy_output_after=[round(item, 8) for item in policy_output_after],
        policy_delta_l2=round(policy_delta, 8),
        mean_reward=round(mean_reward, 8),
        mean_advantage=round(mean_advantage, 8),
        checkpoint_path=str(checkpoint_path),
        signal_count=len(signals),
        control=control,
        loss_integration_point=loss_integration_point,
        duration_ms=round((time.time() - start) * 1000.0, 2),
    )


def create_app() -> Any:
    """Create the FastAPI app used inside the policy container."""

    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise PolicyContainerError("fastapi is required to serve the policy container") from exc

    app = FastAPI(title="npa-lerobot-policy-container")
    policy: LeRobotPolicyRuntime | None = None
    checkpoint = os.environ.get("NPA_POLICY_CHECKPOINT", "")
    if checkpoint:
        policy = LeRobotPolicyRuntime(checkpoint)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        import_result = assert_lerobot_importable().to_dict()
        payload: dict[str, Any] = {"status": "ok", "lerobot": import_result}
        if policy is not None:
            payload["checkpoint"] = policy.validation.to_dict()
        else:
            payload["checkpoint"] = "not configured"
        return payload

    @app.post("/infer")
    async def infer(payload: dict[str, Any]) -> dict[str, Any]:
        if policy is None:
            raise HTTPException(status_code=503, detail="NPA_POLICY_CHECKPOINT is required for inference")
        try:
            return {"actions": policy.infer(payload), "policy": str(policy.checkpoint_path)}
        except PolicyContainerError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    @app.post("/rollout")
    async def rollout(payload: dict[str, Any]) -> dict[str, Any]:
        if policy is None:
            raise HTTPException(status_code=503, detail="NPA_POLICY_CHECKPOINT is required for rollout")
        observations = payload.get("observations", [])
        if not isinstance(observations, list):
            raise HTTPException(status_code=400, detail="observations must be a list")
        try:
            return {"actions": policy.rollout(observations), "policy": str(policy.checkpoint_path)}
        except PolicyContainerError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    @app.post("/feedback/train-step")
    async def feedback_train_step(payload: dict[str, Any]) -> dict[str, Any]:
        is_signal = bool(payload.get("signals")) or str(payload.get("schema", "")).startswith(
            "npa.sim2real.rl_signal."
        )
        try:
            output_dir = jail_output_dir(
                payload.get("output_dir"),
                default_name="vlm-signal" if is_signal else "feedback",
            )
            if is_signal:
                result = run_vlm_signal_training_step(
                    parse_vlm_signal_batch(payload),
                    output_dir=output_dir,
                    learning_rate=float(payload.get("learning_rate") or 0.05),
                    signal_loss_weight=float(payload.get("signal_loss_weight") or 1.0),
                    control=bool(payload.get("control", False)),
                )
            else:
                feedback = parse_feedback_batch(payload)
                result = run_feedback_training_step(
                    feedback,
                    output_dir=output_dir,
                    learning_rate=float(payload.get("learning_rate") or 0.1),
                )
        except PolicyContainerError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.to_dict()

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    import_cmd = subparsers.add_parser("check-import", help="Import LeRobot and LeRobotDataset.")
    import_cmd.set_defaults(_command_name="check-import")
    train_cmd = subparsers.add_parser("train", help="Run real LeRobot policy training.")
    train_cmd.add_argument("--dataset-path", type=Path, default=None)
    train_cmd.add_argument("--dataset-repo-id", default="local/lerobot-dataset")
    train_cmd.add_argument("--output-dir", type=Path, required=True)
    train_cmd.add_argument("--steps", type=int, required=True)
    train_cmd.add_argument("--policy-type", default=DEFAULT_POLICY_TYPE)
    train_cmd.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    train_cmd.add_argument("--num-workers", type=int, default=DEFAULT_TRAIN_NUM_WORKERS)
    train_cmd.add_argument("--device", default=os.environ.get("LEROBOT_POLICY_DEVICE", "cuda"))
    train_cmd.add_argument("--resume", action="store_true")
    train_cmd.add_argument("--log-path", type=Path, default=None)
    train_cmd.add_argument("--timeout-seconds", type=int, default=DEFAULT_TRAIN_TIMEOUT_SECONDS)
    train_cmd.add_argument("--data-path", default="")
    train_cmd.add_argument("--override", action="append", default=[])
    train_cmd.add_argument("--wandb", action="store_true")
    train_cmd.add_argument("--wandb-project", default="")
    train_cmd.add_argument("--wandb-run-name", default="")
    train_cmd.add_argument("--wandb-mode", default="offline")
    train_cmd.add_argument("--checkpoint-s3-uri", default="")
    train_cmd.add_argument("--checkpoint-s3-endpoint-url", default="")
    train_cmd.add_argument("--checkpoint-s3-access-key-id", default="")
    train_cmd.add_argument("--checkpoint-s3-secret-access-key", default="")
    eval_cmd = subparsers.add_parser("eval", help="Run measured LeRobot rollout eval.")
    eval_cmd.add_argument("--checkpoint-path", type=Path, required=True)
    eval_cmd.add_argument("--output-dir", type=Path, required=True)
    eval_cmd.add_argument("--env-type", default="pusht")
    eval_cmd.add_argument("--env-task", default="")
    eval_cmd.add_argument("--episodes", type=int, default=10)
    eval_cmd.add_argument("--device", default=os.environ.get("LEROBOT_POLICY_DEVICE", "cuda"))
    eval_cmd.add_argument("--log-path", type=Path, default=None)
    eval_cmd.add_argument("--timeout-seconds", type=int, default=DEFAULT_EVAL_TIMEOUT_SECONDS)
    validate_cmd = subparsers.add_parser("validate-checkpoint", help="Assert a checkpoint has loadable weights.")
    validate_cmd.add_argument("--checkpoint-path", type=Path, required=True)
    feedback_cmd = subparsers.add_parser("feedback-step", help="Run one feedback trainer-hook step.")
    feedback_cmd.add_argument("--feedback-json", type=Path, required=True)
    feedback_cmd.add_argument("--output-dir", type=Path, required=True)
    feedback_cmd.add_argument("--learning-rate", type=float, default=0.1)
    vlm_signal_cmd = subparsers.add_parser("vlm-signal-step", help="Run one VLM-signal trainer-fork step.")
    vlm_signal_cmd.add_argument("--signal-json", type=Path, required=True)
    vlm_signal_cmd.add_argument("--output-dir", type=Path, required=True)
    vlm_signal_cmd.add_argument("--learning-rate", type=float, default=0.05)
    vlm_signal_cmd.add_argument("--signal-loss-weight", type=float, default=1.0)
    vlm_signal_cmd.add_argument("--control", action="store_true")
    serve_cmd = subparsers.add_parser("serve", help="Run the FastAPI policy container.")
    serve_cmd.add_argument("--host", default=os.environ.get("NPA_POLICY_HOST", "0.0.0.0"))
    serve_cmd.add_argument("--port", type=int, default=int(os.environ.get("NPA_POLICY_PORT", "8080")))
    args = parser.parse_args(argv)

    if args.command == "check-import":
        print(json.dumps(assert_lerobot_importable().to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "train":
        try:
            training_config = build_training_config(
                data_path=args.data_path,
                overrides=args.override,
                wandb_enabled=args.wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=args.wandb_run_name,
                wandb_mode=args.wandb_mode,
                checkpoint_s3_uri=args.checkpoint_s3_uri,
                checkpoint_s3_endpoint_url=args.checkpoint_s3_endpoint_url,
                checkpoint_s3_access_key_id=args.checkpoint_s3_access_key_id,
                checkpoint_s3_secret_access_key=args.checkpoint_s3_secret_access_key,
            )
        except TrainingConfigError as exc:
            raise PolicyContainerError(str(exc)) from exc
        dataset_path = training_config.data_path or args.dataset_path
        if not dataset_path:
            raise PolicyContainerError("--data-path or --dataset-path is required")
        result = run_lerobot_training(
            dataset_path=dataset_path,
            dataset_repo_id=args.dataset_repo_id,
            output_dir=args.output_dir,
            steps=args.steps,
            policy_type=args.policy_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            resume=args.resume,
            log_path=args.log_path,
            timeout_seconds=args.timeout_seconds,
            training_config=training_config,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "eval":
        result = run_lerobot_eval(
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir,
            env_type=args.env_type,
            env_task=args.env_task,
            episodes=args.episodes,
            device=args.device,
            log_path=args.log_path,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "validate-checkpoint":
        result = validate_lerobot_checkpoint(args.checkpoint_path)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "feedback-step":
        payload = json.loads(args.feedback_json.read_text(encoding="utf-8"))
        result = run_feedback_training_step(
            parse_feedback_batch(payload),
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "vlm-signal-step":
        payload = json.loads(args.signal_json.read_text(encoding="utf-8"))
        result = run_vlm_signal_training_step(
            parse_vlm_signal_batch(payload),
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            signal_loss_weight=args.signal_loss_weight,
            control=args.control,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:
            raise PolicyContainerError("uvicorn is required to serve the policy container") from exc
        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0
    return 2


def _feedback_reward(item: FeedbackItem) -> float:
    reward = item.score if item.success else min(item.score, 0.5)
    return max(0.0, min(1.0, float(reward)))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def _tanh(value: float) -> float:
    return math.tanh(float(value))


def _bounded_float(value: Any, *, lower: float, upper: float) -> float:
    parsed = float(value)
    return max(float(lower), min(float(upper), parsed))


def _action_delta(target: dict[str, Any]) -> list[float]:
    raw = target.get("action_delta") or target.get("delta") or target.get("action_target")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise PolicyContainerError("VLM signal target action_delta must be a list")
    return [_bounded_float(item, lower=-1.0, upper=1.0) for item in raw]


def _mean_action_target(action_targets: list[list[float]], *, action_dim: int) -> list[float]:
    if action_dim <= 0:
        return [0.0]
    totals = [0.0 for _ in range(action_dim)]
    counts = [0 for _ in range(action_dim)]
    for target in action_targets:
        for index, value in enumerate(target[:action_dim]):
            totals[index] += float(value)
            counts[index] += 1
    return [round(totals[index] / counts[index], 8) if counts[index] else 0.0 for index in range(action_dim)]


def _reference_signal_loss(
    *,
    reward_prediction: float,
    reward_target: float,
    policy_output: list[float],
    target_vector: list[float],
    mean_advantage: float,
    signal_loss_weight: float,
) -> float:
    reward_loss = (float(reward_prediction) - float(reward_target)) ** 2
    if not policy_output:
        corrective_loss = 0.0
        action_mean = 0.0
    else:
        corrective_loss = sum(
            (float(output) - float(target)) ** 2
            for output, target in zip(policy_output, target_vector, strict=False)
        ) / float(len(policy_output))
        action_mean = sum(policy_output) / float(len(policy_output))
    return reward_loss + float(signal_loss_weight) * corrective_loss - 0.1 * float(mean_advantage) * action_mean


def _find_lerobot_checkpoint(output_dir: Path) -> Path | None:
    preferred = output_dir / "checkpoints" / "last" / "pretrained_model"
    candidates = [preferred]
    if (output_dir / "checkpoints").exists():
        candidates.extend(sorted((output_dir / "checkpoints").rglob("pretrained_model"), reverse=True))
    candidates.extend(sorted(output_dir.rglob("pretrained_model"), reverse=True))
    for candidate in candidates:
        if candidate.is_dir() and _find_weight_file(candidate) is not None:
            return candidate
    return None


def _find_weight_file(checkpoint: Path) -> Path | None:
    for filename in REAL_WEIGHT_FILENAMES:
        candidate = checkpoint / filename
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    for filename in REAL_WEIGHT_FILENAMES:
        matches = sorted(checkpoint.rglob(filename))
        for candidate in matches:
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_reward(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _shell_join(command: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in command)


if __name__ == "__main__":
    raise SystemExit(main())
