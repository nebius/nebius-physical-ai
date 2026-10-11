"""Success judges for evaluation-harness episodes.

* :class:`HeuristicJudge` — trusts the environment's own success signal
  (``info["success"]``).  Deterministic, offline, the default.
* :class:`VLMJudge` — a real client over :mod:`npa.workbench.vlm_eval`: it
  renders the episode's frames through a configured VLM endpoint and
  thresholds the returned score.  It requires configuration (an endpoint URL
  and rendered frames) and raises an informative error when unconfigured —
  it never invents a score.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from npa.workbench.eval_harness.tasks import EvalHarnessError


@dataclass(frozen=True)
class JudgeResult:
    success: bool
    score: float
    judge: str
    rationale: str


class HeuristicJudge:
    """Judge that trusts the environment's ``info["success"]`` signal."""

    name = "heuristic"
    needs_frames = False

    def judge(
        self,
        *,
        task: str,
        env_success: bool,
        frames: Sequence[np.ndarray] = (),
    ) -> JudgeResult:
        _ = (task, frames)
        return JudgeResult(
            success=bool(env_success),
            score=1.0 if env_success else 0.0,
            judge=self.name,
            rationale="environment info['success'] signal",
        )


class VLMJudge:
    """VLM-as-judge built on :mod:`npa.workbench.vlm_eval`.

    Requires ``endpoint_url`` (a VLM/OpenAI-compatible endpoint) and rendered
    episode frames.  When either is missing it raises
    :class:`EvalHarnessError` with remediation instead of scoring — a judge
    that cannot see the episode must not vote.
    """

    name = "vlm"
    needs_frames = True

    def __init__(
        self,
        *,
        endpoint_url: str = "",
        api_key_env: str = "VLM_EVAL_API_KEY",
        model: str = "",
        success_threshold: float = 0.8,
    ) -> None:
        if not 0.0 <= success_threshold <= 1.0:
            raise EvalHarnessError("success_threshold must be in [0, 1]")
        self.endpoint_url = endpoint_url.strip()
        self.api_key_env = api_key_env
        self.model = model
        self.success_threshold = success_threshold

    def judge(
        self,
        *,
        task: str,
        env_success: bool,
        frames: Sequence[np.ndarray] = (),
    ) -> JudgeResult:
        _ = env_success
        if not self.endpoint_url:
            raise EvalHarnessError(
                "the VLM judge requires --vlm-endpoint-url: point it at a "
                "VLM/OpenAI-compatible endpoint (the API key is read from "
                f"${self.api_key_env}), or use --judge heuristic for the "
                "environment success signal."
            )
        if not frames:
            raise EvalHarnessError(
                "the VLM judge requires rendered episode frames, but the "
                "environment produced none (headless rendering unavailable). "
                "Use --judge heuristic, or run on a host with MuJoCo EGL "
                "rendering."
            )
        score = self._score_frames(task=task, frames=frames)
        return JudgeResult(
            success=bool(score >= self.success_threshold),
            score=score,
            judge=self.name,
            rationale=(
                f"vlm score {score:.3f} vs threshold {self.success_threshold} "
                f"at {self.endpoint_url}"
            ),
        )

    def _score_frames(self, *, task: str, frames: Sequence[np.ndarray]) -> float:
        from PIL import Image

        from npa.workbench import vlm_eval

        with tempfile.TemporaryDirectory(prefix="eval-harness-vlm-") as tmp:
            tmpdir = Path(tmp)
            for i, frame in enumerate(frames):
                Image.fromarray(np.asarray(frame)).save(tmpdir / f"frame_{i:04d}.png")
            api_key = os.environ.get(self.api_key_env, "").strip()
            result = vlm_eval.evaluate_vlm(
                input_path=str(tmpdir),
                output_path=str(tmpdir / "vlm_eval.json"),
                task=f"eval_harness:{task}",
                backend="api" if api_key else "self-hosted",
                model=self.model,
                endpoint_url=self.endpoint_url,
                api_key_env=self.api_key_env,
                success_threshold=self.success_threshold,
            )
        score = float(result.score)
        if not 0.0 <= score <= 1.0:
            raise EvalHarnessError(
                f"VLM endpoint returned an out-of-range score {score!r}"
            )
        return score


def make_judge(
    name: str, judge_config: Mapping[str, Any] | None = None
) -> HeuristicJudge | VLMJudge:
    """Build the named judge (``heuristic`` or ``vlm``)."""
    config = dict(judge_config or {})
    if name == HeuristicJudge.name:
        return HeuristicJudge()
    if name == VLMJudge.name:
        return VLMJudge(**config)
    raise EvalHarnessError(
        f"unknown judge {name!r}; known judges: {[HeuristicJudge.name, VLMJudge.name]}"
    )


__all__ = [
    "HeuristicJudge",
    "JudgeResult",
    "VLMJudge",
    "make_judge",
]
