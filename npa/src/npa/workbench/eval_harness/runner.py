"""Episode runner for the evaluation harness.

:func:`run_policy` evaluates one policy on a task for N episodes, applying a
success judge per episode and returning a JSON-serializable report.  A fresh
policy instance is built per episode from the episode seed, so stateful
policies start clean and paired-seed comparisons are reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from npa.workbench.eval_harness import metrics
from npa.workbench.eval_harness.judges import make_judge
from npa.workbench.eval_harness.policies import make_policy_factory
from npa.workbench.eval_harness.tasks import EvalHarnessError, get_task

FRAME_EVERY = 10


@dataclass
class EpisodeRecord:
    index: int
    seed: int
    success: bool
    steps: int
    time_to_success: int | None
    total_reward: float
    judge: str
    judge_score: float
    frames_captured: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunReport:
    schema: str
    task: str
    policy: str
    episodes: int
    seed: int
    judge: str
    max_steps: int
    records: list[EpisodeRecord] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task": self.task,
            "policy": self.policy,
            "episodes": self.episodes,
            "seed": self.seed,
            "judge": self.judge,
            "max_steps": self.max_steps,
            "records": [r.as_dict() for r in self.records],
            "summary": self.summary,
        }


def run_policy(
    *,
    task: str,
    policy: str,
    episodes: int,
    seed: int = 0,
    judge: str = "heuristic",
    max_steps: int = 500,
    vlm_endpoint_url: str = "",
) -> RunReport:
    """Evaluate *policy* on *task* for *episodes* episodes (real execution)."""
    if episodes < 1:
        raise EvalHarnessError("episodes must be >= 1")
    if max_steps < 1:
        raise EvalHarnessError("max_steps must be >= 1")
    judge_impl = make_judge(
        judge, {"endpoint_url": vlm_endpoint_url} if judge == "vlm" else None
    )
    factory = get_task(task)
    policy_factory = make_policy_factory(policy, task)

    report = RunReport(
        schema="npa.workbench.eval_harness.run.v1",
        task=task,
        policy=policy,
        episodes=episodes,
        seed=seed,
        judge=judge,
        max_steps=max_steps,
    )
    for index in range(episodes):
        episode_seed = seed + index
        report.records.append(
            _run_episode(
                task=task,
                index=index,
                seed=episode_seed,
                env_factory=factory,
                policy_factory=policy_factory,
                judge_impl=judge_impl,
                max_steps=max_steps,
            )
        )
    report.summary = metrics.summarize(
        [r.as_dict() for r in report.records]
    )
    return report


def _run_episode(
    *,
    task: str,
    index: int,
    seed: int,
    env_factory: Any,
    policy_factory: Any,
    judge_impl: Any,
    max_steps: int,
) -> EpisodeRecord:
    env = env_factory()
    if hasattr(env, "max_steps"):
        env.max_steps = max_steps
    policy_fn = policy_factory(seed)
    obs = env.reset(seed=seed)
    collect_frames = bool(judge_impl.needs_frames)
    frames: list[np.ndarray] = []
    total_reward = 0.0
    time_to_success: int | None = None
    env_success = False
    steps = 0
    render = getattr(env, "render_rgb", None)
    for step in range(1, max_steps + 1):
        action = policy_fn(obs)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps = step
        if collect_frames and render is not None and step % FRAME_EVERY == 0:
            frame = render()
            if frame is not None:
                frames.append(np.asarray(frame))
        if info.get("success") and time_to_success is None:
            time_to_success = step
            env_success = True
        if done:
            env_success = bool(info.get("success", env_success))
            break
    judged = judge_impl.judge(
        task=task, env_success=env_success, frames=frames
    )
    return EpisodeRecord(
        index=index,
        seed=seed,
        success=judged.success,
        steps=steps,
        time_to_success=time_to_success,
        total_reward=total_reward,
        judge=judged.judge,
        judge_score=judged.score,
        frames_captured=len(frames),
    )


__all__ = ["EpisodeRecord", "RunReport", "run_policy"]
