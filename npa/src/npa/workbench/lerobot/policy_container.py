"""BYO LeRobot policy container runtime and feedback-training hook."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_POLICY_CHECKPOINT = "lerobot/diffusion_pusht"
DEFAULT_ACTION_DIM = 2
DEFAULT_POLICY_TYPE = "act"
DEFAULT_TRAIN_BATCH_SIZE = 8
DEFAULT_TRAIN_NUM_WORKERS = 4
DEFAULT_TRAIN_LOG_FREQ = 10
DEFAULT_TRAIN_TIMEOUT_SECONDS = 43200
DEFAULT_EVAL_TIMEOUT_SECONDS = 7200
REAL_WEIGHT_FILENAMES = ("model.safetensors", "pytorch_model.bin")


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
) -> list[str]:
    """Build a real `lerobot-train` command for a local LeRobotDataset."""

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
        "--wandb.enable=false",
    ]
    if resume:
        cmd.append("--resume=true")
    if extra_args:
        cmd.extend(extra_args)
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
) -> LeRobotTrainingResult:
    """Run real LeRobot policy training and validate the resulting checkpoint."""

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
        try:
            feedback = parse_feedback_batch(payload)
            result = run_feedback_training_step(
                feedback,
                output_dir=Path(payload.get("output_dir") or "/tmp/npa-lerobot-feedback"),
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
    train_cmd.add_argument("--dataset-path", type=Path, required=True)
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
    serve_cmd = subparsers.add_parser("serve", help="Run the FastAPI policy container.")
    serve_cmd.add_argument("--host", default=os.environ.get("NPA_POLICY_HOST", "0.0.0.0"))
    serve_cmd.add_argument("--port", type=int, default=int(os.environ.get("NPA_POLICY_PORT", "8080")))
    args = parser.parse_args(argv)

    if args.command == "check-import":
        print(json.dumps(assert_lerobot_importable().to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "train":
        result = run_lerobot_training(
            dataset_path=args.dataset_path,
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
    import math

    return 1.0 / (1.0 + math.exp(-float(value)))


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
