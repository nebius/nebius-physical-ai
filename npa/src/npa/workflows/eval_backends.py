"""Pluggable eval backend registry for sim-to-real workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from npa.workbench.lerobot.policy_container import PolicyContainerError, run_lerobot_eval


DEFAULT_EVAL_BACKEND = "state-success"


class EvalBackendError(ValueError):
    """Raised when an eval backend cannot be resolved or run."""


@dataclass(frozen=True)
class EvalMetric:
    """Normalized metric emitted by an eval backend."""

    name: str
    score: float
    passed: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalBackendStatus:
    """Lightweight status object converted into pipeline component status."""

    name: str
    tier: str
    evidence: str
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutContext:
    """Context available to eval backends."""

    rollout_path: Path
    task: str
    sim_backend: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    state: Mapping[str, Any] = field(default_factory=dict)
    frames: tuple[Path, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)


class EvalBackend(Protocol):
    """Protocol implemented by eval backends."""

    name: str

    def evaluate(
        self,
        *,
        checkpoint_uri: str,
        context: RolloutContext,
        threshold: float,
    ) -> tuple[EvalMetric, EvalBackendStatus]:
        """Evaluate a checkpoint/rollout pair and return a normalized metric."""


_EVAL_BACKENDS: dict[str, EvalBackend] = {}


def register_eval_backend(backend: EvalBackend, *, aliases: tuple[str, ...] = ()) -> None:
    """Register an eval backend by primary name and optional aliases."""

    names = (backend.name, *aliases)
    for name in names:
        normalized = _normalize_name(name)
        if not normalized:
            raise EvalBackendError("eval backend name must not be empty")
        _EVAL_BACKENDS[normalized] = backend


def get_eval_backend(name: str) -> EvalBackend:
    """Return a registered eval backend."""

    normalized = _normalize_name(name or DEFAULT_EVAL_BACKEND)
    try:
        return _EVAL_BACKENDS[normalized]
    except KeyError as exc:
        allowed = ", ".join(registered_eval_backends())
        raise EvalBackendError(f"unsupported eval backend '{name}'. Supported: {allowed}") from exc


def registered_eval_backends() -> tuple[str, ...]:
    """Return the canonical registered eval backend names."""

    return tuple(sorted({backend.name for backend in _EVAL_BACKENDS.values()}))


def evaluate_backend(
    name: str,
    *,
    checkpoint_uri: str,
    context: RolloutContext,
    threshold: float,
) -> tuple[EvalMetric, EvalBackendStatus]:
    """Resolve and run an eval backend."""

    return get_eval_backend(name).evaluate(
        checkpoint_uri=checkpoint_uri,
        context=context,
        threshold=threshold,
    )


class StateSuccessEvalBackend:
    """Pose/state predicate eval seam for simulator or real-env success signals."""

    name = "state-success"

    def evaluate(
        self,
        *,
        checkpoint_uri: str,
        context: RolloutContext,
        threshold: float,
    ) -> tuple[EvalMetric, EvalBackendStatus]:
        lerobot_eval = context.options.get("lerobot_eval")
        if isinstance(lerobot_eval, Mapping) and lerobot_eval:
            return _evaluate_lerobot_pc_success(
                checkpoint_uri=checkpoint_uri,
                context=context,
                threshold=threshold,
                options=lerobot_eval,
            )

        score = _score_from_keys(context.state, ("pc_success", "state_success", "success"))
        if score is None:
            score = _score_from_keys(context.metrics, ("pc_success", "state_success", "success"))
        passed = None if score is None else score >= threshold
        if score is None:
            metric = EvalMetric(
                name=self.name,
                score=0.0,
                passed=False,
                metadata={
                    "checkpoint_uri": checkpoint_uri,
                    "sim_backend": context.sim_backend,
                    "real_eval_hook": "lerobot-eval/pc_success",
                },
            )
            return metric, EvalBackendStatus(
                name="state_success_eval",
                tier="SEAM",
                evidence=(
                    "state-success is registered as the pose/state predicate backend; "
                    "no state-success or pc_success metric was provided in the rollout context."
                ),
            )
        return (
            EvalMetric(
                name=self.name,
                score=score,
                passed=passed,
                metadata={"checkpoint_uri": checkpoint_uri, "sim_backend": context.sim_backend},
            ),
            EvalBackendStatus(
                name="state_success_eval",
                tier="PARTIAL",
                evidence="Computed a normalized state-success score from rollout context.",
            ),
        )


class VlmFramesEvalBackend:
    """Frame-subset eval seam for VLM/VLA scoring."""

    name = "vlm-frames"

    def evaluate(
        self,
        *,
        checkpoint_uri: str,
        context: RolloutContext,
        threshold: float,
    ) -> tuple[EvalMetric, EvalBackendStatus]:
        score = _coerce_score(context.metrics.get("vlm_score"))
        if score is None:
            score = 0.0
            tier = "SEAM"
            evidence = "vlm-frames backend selected; render/VLM dispatch is a typed extension point."
        else:
            tier = "PARTIAL"
            evidence = "Used a mocked or precomputed VLM frame score from rollout context."
        return (
            EvalMetric(
                name=self.name,
                score=score,
                passed=score >= threshold,
                metadata={
                    "checkpoint_uri": checkpoint_uri,
                    "rollout_path": str(context.rollout_path),
                    "frame_count": len(context.frames),
                },
            ),
            EvalBackendStatus(name="vlm_frames_eval", tier=tier, evidence=evidence),
        )


class HeldoutMetricsEvalBackend:
    """Heldout imitation metrics backend."""

    name = "heldout-metrics"

    def evaluate(
        self,
        *,
        checkpoint_uri: str,
        context: RolloutContext,
        threshold: float,
    ) -> tuple[EvalMetric, EvalBackendStatus]:
        score = _coerce_score(context.metrics.get("heldout_score"))
        action_mse = _coerce_nonnegative(context.metrics.get("action_mse"))
        if score is None and action_mse is not None:
            score = 1.0 / (1.0 + action_mse)
        if score is None:
            score = 0.0
            tier = "SEAM"
            evidence = "heldout-metrics backend selected; no heldout metric payload was provided."
        else:
            tier = "PARTIAL"
            evidence = "Computed a normalized heldout imitation metric score."
        return (
            EvalMetric(
                name=self.name,
                score=score,
                passed=score >= threshold,
                metadata={"checkpoint_uri": checkpoint_uri, "action_mse": action_mse},
            ),
            EvalBackendStatus(name="heldout_metrics_eval", tier=tier, evidence=evidence),
        )


def _evaluate_lerobot_pc_success(
    *,
    checkpoint_uri: str,
    context: RolloutContext,
    threshold: float,
    options: Mapping[str, Any],
) -> tuple[EvalMetric, EvalBackendStatus]:
    output_dir = _required_path(options, "output_dir")
    env_type = str(options.get("env_type") or context.sim_backend or "pusht")
    episodes = int(options.get("episodes") or 1)
    device = str(options.get("device") or "cuda")
    env_task = str(options.get("env_task") or "")
    log_path = _optional_path(options.get("log_path"))
    timeout_seconds = int(options.get("timeout_seconds") or 0)
    kwargs: dict[str, Any] = {
        "checkpoint_path": Path(checkpoint_uri),
        "output_dir": output_dir,
        "env_type": env_type,
        "episodes": episodes,
        "device": device,
        "env_task": env_task,
        "log_path": log_path,
    }
    if timeout_seconds > 0:
        kwargs["timeout_seconds"] = timeout_seconds
    try:
        result = run_lerobot_eval(**kwargs)
    except PolicyContainerError as exc:
        raise EvalBackendError(str(exc)) from exc

    score = _coerce_score(result.score)
    if score is None:
        raise EvalBackendError(f"lerobot-eval returned an invalid score: {result.score}")
    passed = score >= threshold
    metadata = {
        "adapter": "lerobot-eval",
        "checkpoint_uri": checkpoint_uri,
        "sim_backend": context.sim_backend,
        "env_type": env_type,
        "metric_name": result.metric_name,
        "pc_success": result.pc_success,
        "avg_sum_reward": result.avg_sum_reward,
        "avg_max_reward": result.avg_max_reward,
        "n_episodes": result.n_episodes,
        "output_dir": result.output_dir,
        "eval_info_path": result.eval_info_path,
        "log_path": result.log_path,
        "raw_metrics": result.raw_metrics,
    }
    return (
        EvalMetric(name="state-success", score=score, passed=passed, metadata=metadata),
        EvalBackendStatus(
            name="state_success_eval",
            tier="WORKS",
            evidence=(
                f"Ran lerobot-eval in env={env_type!r} and adapted "
                f"{result.metric_name}={score:.6f} to state-success."
            ),
            artifacts={"eval_info": result.eval_info_path, "eval_log": result.log_path, "output_dir": result.output_dir},
        ),
    )


def _required_path(options: Mapping[str, Any], key: str) -> Path:
    value = options.get(key)
    if not value:
        raise EvalBackendError(f"lerobot-eval adapter requires {key}")
    return Path(value)


def _optional_path(value: Any) -> Path | None:
    if not value:
        return None
    return Path(value)


def _normalize_name(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _score_from_keys(values: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in values:
            continue
        value = values[key]
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        score = _coerce_score(value)
        if score is not None:
            return score
    return None


def _coerce_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= score <= 1.0:
        return None
    return score


def _coerce_nonnegative(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0.0:
        return None
    return number


register_eval_backend(StateSuccessEvalBackend(), aliases=("sim-env", "genesis", "pc-success", "pusht", "lerobot-eval"))
register_eval_backend(VlmFramesEvalBackend())
register_eval_backend(HeldoutMetricsEvalBackend())
