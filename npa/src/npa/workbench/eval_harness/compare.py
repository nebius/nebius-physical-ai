"""Paired A/B policy comparison for the evaluation harness.

:func:`compare_policies` runs two policies on the same task with paired
episode seeds (episode ``i`` uses seed ``base + i`` for both policies), then
reports the difference in success rate with a paired bootstrap confidence
interval.  Pairing removes environment/episode noise from the comparison, so
the interval reflects policy differences rather than seed luck.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from npa.workbench.eval_harness.runner import RunReport, run_policy
from npa.workbench.eval_harness.tasks import EvalHarnessError

N_BOOTSTRAP = 2000


@dataclass
class CompareReport:
    schema: str
    task: str
    policy_a: str
    policy_b: str
    episodes: int
    seed: int
    judge: str
    max_steps: int
    summary_a: dict[str, Any] = field(default_factory=dict)
    summary_b: dict[str, Any] = field(default_factory=dict)
    success_rate_diff: float = 0.0
    bootstrap_95: dict[str, float] = field(default_factory=dict)
    per_episode: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task": self.task,
            "policy_a": self.policy_a,
            "policy_b": self.policy_b,
            "episodes": self.episodes,
            "seed": self.seed,
            "judge": self.judge,
            "max_steps": self.max_steps,
            "summary_a": self.summary_a,
            "summary_b": self.summary_b,
            "success_rate_diff": self.success_rate_diff,
            "bootstrap_95": self.bootstrap_95,
            "per_episode": self.per_episode,
        }


def compare_policies(
    *,
    task: str,
    policy_a: str,
    policy_b: str,
    episodes: int,
    seed: int = 0,
    judge: str = "heuristic",
    max_steps: int = 500,
    vlm_endpoint_url: str = "",
    n_bootstrap: int = N_BOOTSTRAP,
) -> CompareReport:
    """A/B compare two policies with paired episode seeds (real execution)."""
    if episodes < 1:
        raise EvalHarnessError("episodes must be >= 1")
    if n_bootstrap < 100:
        raise EvalHarnessError("n_bootstrap must be >= 100")
    common: dict[str, Any] = dict(
        task=task,
        episodes=episodes,
        seed=seed,
        judge=judge,
        max_steps=max_steps,
        vlm_endpoint_url=vlm_endpoint_url,
    )
    report_a: RunReport = run_policy(policy=policy_a, **common)
    report_b: RunReport = run_policy(policy=policy_b, **common)

    diffs = np.array(
        [
            float(a.success) - float(b.success)
            for a, b in zip(report_a.records, report_b.records)
        ]
    )
    delta = float(diffs.mean()) if len(diffs) else 0.0
    lo, hi = _paired_bootstrap_ci(diffs, n_bootstrap=n_bootstrap, seed=seed)

    per_episode = [
        {
            "index": a.index,
            "seed": a.seed,
            "success_a": a.success,
            "success_b": b.success,
        }
        for a, b in zip(report_a.records, report_b.records)
    ]
    return CompareReport(
        schema="npa.workbench.eval_harness.compare.v1",
        task=task,
        policy_a=policy_a,
        policy_b=policy_b,
        episodes=episodes,
        seed=seed,
        judge=judge,
        max_steps=max_steps,
        summary_a=report_a.summary,
        summary_b=report_b.summary,
        success_rate_diff=delta,
        bootstrap_95={"lo": lo, "hi": hi},
        per_episode=per_episode,
    )


def _paired_bootstrap_ci(
    diffs: np.ndarray, *, n_bootstrap: int, seed: int
) -> tuple[float, float]:
    """95% bootstrap CI for the mean of paired per-episode differences."""
    if len(diffs) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed + 0xC0FFEE)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        means[i] = float(sample.mean())
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


__all__ = ["CompareReport", "compare_policies"]
