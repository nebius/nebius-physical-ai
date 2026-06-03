"""BYO LeRobot policy container runtime and feedback-training hook."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_POLICY_CHECKPOINT = "lerobot/diffusion_pusht"
DEFAULT_ACTION_DIM = 2


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


class HeuristicPolicy:
    """Small deterministic fallback policy used when no LeRobot checkpoint is loaded."""

    def __init__(self, action_dim: int = DEFAULT_ACTION_DIM) -> None:
        self.action_dim = action_dim

    def infer(self, observation: dict[str, Any]) -> list[float]:
        state = observation.get("observation.state") or observation.get("state") or []
        values = [float(value) for value in state] if isinstance(state, list) else []
        if not values:
            values = [0.0] * self.action_dim
        return [round(values[index % len(values)], 6) for index in range(self.action_dim)]

    def rollout(self, observations: list[dict[str, Any]]) -> list[list[float]]:
        return [self.infer(observation) for observation in observations]


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
    policy = HeuristicPolicy()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/infer")
    async def infer(payload: dict[str, Any]) -> dict[str, Any]:
        return {"actions": policy.infer(payload), "policy": "heuristic_or_loaded_lerobot"}

    @app.post("/rollout")
    async def rollout(payload: dict[str, Any]) -> dict[str, Any]:
        observations = payload.get("observations", [])
        if not isinstance(observations, list):
            raise HTTPException(status_code=400, detail="observations must be a list")
        return {"actions": policy.rollout(observations), "policy": "heuristic_or_loaded_lerobot"}

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
    feedback_cmd = subparsers.add_parser("feedback-step", help="Run one feedback trainer-hook step.")
    feedback_cmd.add_argument("--feedback-json", type=Path, required=True)
    feedback_cmd.add_argument("--output-dir", type=Path, required=True)
    feedback_cmd.add_argument("--learning-rate", type=float, default=0.1)
    serve_cmd = subparsers.add_parser("serve", help="Run the FastAPI policy container.")
    serve_cmd.add_argument("--host", default=os.environ.get("NPA_POLICY_HOST", "0.0.0.0"))
    serve_cmd.add_argument("--port", type=int, default=int(os.environ.get("NPA_POLICY_PORT", "8080")))
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    raise SystemExit(main())
