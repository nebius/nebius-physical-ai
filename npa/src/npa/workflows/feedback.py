"""Feedback source registry and typed training-signal adapters."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib import request as urllib_request
from urllib.error import URLError

from npa.workflows.eval_backends import EvalMetric


DEFAULT_FEEDBACK_SOURCE = "vlm"
DEFAULT_FEEDBACK_TYPE = "critique"


class FeedbackSourceError(ValueError):
    """Raised when a feedback source cannot be resolved or interpreted."""


class FeedbackType(str, Enum):
    """Standard feedback types accepted by the sim-to-real loop."""

    SCALAR = "scalar"
    DENSE_PER_STEP = "dense-per-step"
    PASS_FAIL = "pass-fail"
    CRITIQUE = "critique"
    PREFERENCE = "preference"


@dataclass(frozen=True)
class FeedbackPayload:
    """Typed feedback emitted by a feedback source."""

    source: str
    feedback_type: FeedbackType
    value: Any
    score: float = 0.0
    success: bool = False
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingSignal:
    """Normalized training signal produced by a feedback-type adapter."""

    schema: str
    scalar_reward: float
    success: bool
    score: float
    loss_weight: float
    dense_rewards: list[float] = field(default_factory=list)
    natural_language_critique: str = ""
    preference: dict[str, Any] = field(default_factory=dict)
    source: str = DEFAULT_FEEDBACK_SOURCE
    feedback_type: str = DEFAULT_FEEDBACK_TYPE

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""

        return asdict(self)


@dataclass(frozen=True)
class FeedbackSourceStatus:
    """Lightweight status object converted into pipeline component status."""

    name: str
    tier: str
    evidence: str
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FeedbackRequest:
    """Context passed to feedback sources."""

    rollout_path: Path
    output_path: Path
    task: str
    checkpoint_uri: str
    threshold: float
    feedback_type: FeedbackType
    eval_metric: EvalMetric | None = None
    vlm_backend: str = "stub"
    vlm_model: str = "vlm-eval-stub"
    vlm_endpoint_url: str = ""
    vlm_frame_selection: str = "keyframes"
    vlm_max_frames: int = 4
    vlm_score: float | None = None
    byo_endpoint_url: str = ""
    byo_command: str = ""
    byo_mode: str = "provided-rollout"


class FeedbackSource(Protocol):
    """Protocol implemented by feedback sources."""

    name: str

    def collect(self, request: FeedbackRequest) -> tuple[FeedbackPayload, FeedbackSourceStatus]:
        """Collect typed feedback."""


HttpPost = Callable[[str, dict[str, Any]], dict[str, Any]]
CommandRunner = Callable[[list[str], dict[str, Any]], dict[str, Any]]

_FEEDBACK_SOURCES: dict[str, FeedbackSource] = {}
_ADAPTERS: dict[FeedbackType, Callable[[FeedbackPayload], TrainingSignal]] = {}


def register_feedback_source(source: FeedbackSource, *, aliases: tuple[str, ...] = ()) -> None:
    """Register a feedback source by primary name and optional aliases."""

    for name in (source.name, *aliases):
        normalized = _normalize_name(name)
        if not normalized:
            raise FeedbackSourceError("feedback source name must not be empty")
        _FEEDBACK_SOURCES[normalized] = source


def get_feedback_source(name: str) -> FeedbackSource:
    """Return a registered feedback source."""

    normalized = _normalize_name(name or DEFAULT_FEEDBACK_SOURCE)
    try:
        return _FEEDBACK_SOURCES[normalized]
    except KeyError as exc:
        allowed = ", ".join(registered_feedback_sources())
        raise FeedbackSourceError(f"unsupported feedback source '{name}'. Supported: {allowed}") from exc


def registered_feedback_sources() -> tuple[str, ...]:
    """Return the canonical registered feedback source names."""

    return tuple(sorted({source.name for source in _FEEDBACK_SOURCES.values()}))


def parse_feedback_type(value: str | FeedbackType) -> FeedbackType:
    """Parse a feedback type from user input."""

    if isinstance(value, FeedbackType):
        return value
    normalized = _normalize_name(value or DEFAULT_FEEDBACK_TYPE)
    for candidate in FeedbackType:
        if candidate.value == normalized:
            return candidate
    allowed = ", ".join(item.value for item in FeedbackType)
    raise FeedbackSourceError(f"unsupported feedback type '{value}'. Supported: {allowed}")


def collect_feedback(
    source: str,
    request: FeedbackRequest,
) -> tuple[FeedbackPayload, FeedbackSourceStatus]:
    """Resolve and collect feedback from a configured source."""

    return get_feedback_source(source).collect(request)


def register_feedback_adapter(
    feedback_type: FeedbackType,
    adapter: Callable[[FeedbackPayload], TrainingSignal],
) -> None:
    """Register an adapter from a feedback type to a training signal."""

    _ADAPTERS[feedback_type] = adapter


def adapt_feedback_to_training_signal(payload: FeedbackPayload) -> dict[str, Any]:
    """Convert typed feedback into a JSON-serializable training signal."""

    try:
        adapter = _ADAPTERS[payload.feedback_type]
    except KeyError as exc:
        raise FeedbackSourceError(f"no adapter registered for feedback type '{payload.feedback_type.value}'") from exc
    return adapter(payload).to_dict()


def byo_feedback_contract(
    *,
    declared_type: str = DEFAULT_FEEDBACK_TYPE,
    mode: str = "provided-rollout",
    endpoint_url: str = "",
    command: str = "",
) -> dict[str, Any]:
    """Return the neutral BYO feedback container contract."""

    feedback_type = parse_feedback_type(declared_type)
    normalized_mode = _normalize_mode(mode)
    return {
        "source": "byo-container",
        "declared_feedback_type": feedback_type.value,
        "mode": normalized_mode,
        "invocation": {
            "http": {
                "method": "POST",
                "endpoint_url": endpoint_url,
                "request_schema": {
                    "mode": normalized_mode,
                    "task": "string",
                    "checkpoint_uri": "string",
                    "rollout_path": "string, present for provided-rollout mode",
                },
            },
            "cli": {
                "command": command,
                "stdin": "same JSON request schema as HTTP",
                "stdout": "JSON feedback payload",
            },
        },
        "response_schema": {
            "feedback_type": feedback_type.value,
            "value": "type-specific payload",
            "score": "optional float in [0, 1]",
            "success": "optional boolean",
            "rationale": "optional string",
        },
    }


class NoneFeedbackSource:
    """No-feedback source for pure imitation training."""

    name = "none"

    def collect(self, request: FeedbackRequest) -> tuple[FeedbackPayload, FeedbackSourceStatus]:
        payload = FeedbackPayload(
            source=self.name,
            feedback_type=FeedbackType.SCALAR,
            value=0.0,
            score=0.0,
            success=True,
            rationale="No feedback loop configured; pure imitation training signal.",
        )
        return payload, FeedbackSourceStatus(
            name="none_feedback",
            tier="PARTIAL",
            evidence="No feedback source selected; emitted pure-imitation training signal.",
        )


class SimEnvFeedbackSource:
    """Feedback source that adapts the selected eval/env metric."""

    name = "sim-env"

    def collect(self, request: FeedbackRequest) -> tuple[FeedbackPayload, FeedbackSourceStatus]:
        metric = request.eval_metric
        if metric is None:
            payload = FeedbackPayload(
                source=self.name,
                feedback_type=request.feedback_type,
                value=0.0,
                score=0.0,
                success=False,
                rationale="No eval metric was available for sim-env feedback.",
            )
            return payload, FeedbackSourceStatus(
                name="sim_env_feedback",
                tier="SEAM",
                evidence="sim-env feedback selected, but no eval metric has been produced.",
            )
        payload = _payload_from_score(
            source=self.name,
            feedback_type=request.feedback_type,
            score=metric.score,
            success=metric.passed if metric.passed is not None else metric.score >= request.threshold,
            rationale=f"{metric.name} score={metric.score:.6f}",
            metadata=metric.metadata,
        )
        return payload, FeedbackSourceStatus(
            name="sim_env_feedback",
            tier="PARTIAL",
            evidence=f"Adapted eval backend {metric.name!r} into {request.feedback_type.value} feedback.",
        )


class VlmFeedbackSource:
    """Feedback source backed by the existing VLM eval interface."""

    name = "vlm"

    def collect(self, request: FeedbackRequest) -> tuple[FeedbackPayload, FeedbackSourceStatus]:
        try:
            from npa.workbench.vlm_eval import VlmEvalError, evaluate_vlm, write_result
        except ImportError as exc:
            return _blocked_payload(
                source=self.name,
                feedback_type=request.feedback_type,
                evidence=f"VLM eval interface could not be imported: {exc}",
            )

        try:
            result = evaluate_vlm(
                input_path=str(request.rollout_path),
                output_path=str(request.output_path),
                task=request.task,
                backend=request.vlm_backend,
                model=request.vlm_model,
                endpoint_url=request.vlm_endpoint_url,
                frame_selection=request.vlm_frame_selection,
                max_frames=request.vlm_max_frames,
                success_threshold=request.threshold,
                score=request.vlm_score,
            )
            payload = asdict(result)
            payload["written_uri"] = write_result(payload, result_uri=result.result_uri)
        except (OSError, VlmEvalError) as exc:
            return _blocked_payload(
                source=self.name,
                feedback_type=request.feedback_type,
                evidence=f"VLM eval interface could not score the rollout: {exc}",
            )

        feedback = _payload_from_score(
            source=self.name,
            feedback_type=request.feedback_type,
            score=float(result.score),
            success=bool(result.passed),
            rationale=result.rationale or result.status,
            metadata={
                "backend": result.backend,
                "model": result.model,
                "result_uri": payload["written_uri"],
            },
        )
        tier = "PARTIAL" if result.backend == "stub" else "WORKS"
        evidence = (
            "Existing vlm-eval stub backend produced schema-compatible feedback."
            if result.backend == "stub"
            else f"Existing vlm-eval backend {result.backend!r} scored rollout frames."
        )
        return feedback, FeedbackSourceStatus(
            name="vlm_feedback",
            tier=tier,
            evidence=evidence,
            artifacts={"result_uri": payload["written_uri"]},
        )


class ByoContainerFeedbackSource:
    """Neutral BYO feedback container source supporting HTTP and CLI modes."""

    name = "byo-container"

    def __init__(
        self,
        *,
        http_post: HttpPost | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self._http_post = http_post or _default_http_post
        self._command_runner = command_runner or _default_command_runner

    def collect(self, request: FeedbackRequest) -> tuple[FeedbackPayload, FeedbackSourceStatus]:
        mode = _normalize_mode(request.byo_mode)
        invocation = {
            "mode": mode,
            "task": request.task,
            "checkpoint_uri": request.checkpoint_uri,
            "feedback_type": request.feedback_type.value,
        }
        if mode == "provided-rollout":
            invocation["rollout_path"] = str(request.rollout_path)
        if request.byo_endpoint_url:
            try:
                raw = self._http_post(request.byo_endpoint_url, invocation)
            except Exception as exc:
                return _blocked_payload(
                    source=self.name,
                    feedback_type=request.feedback_type,
                    evidence=f"BYO feedback HTTP invocation failed: {exc}",
                )
            route = "http"
        elif request.byo_command:
            try:
                raw = self._command_runner(shlex.split(request.byo_command), invocation)
            except Exception as exc:
                return _blocked_payload(
                    source=self.name,
                    feedback_type=request.feedback_type,
                    evidence=f"BYO feedback CLI invocation failed: {exc}",
                )
            route = "cli"
        else:
            return _blocked_payload(
                source=self.name,
                feedback_type=request.feedback_type,
                evidence="BYO feedback source requires BYO_FEEDBACK_ENDPOINT_URL or BYO_FEEDBACK_COMMAND.",
            )
        payload = parse_source_payload(raw, source=self.name, feedback_type=request.feedback_type)
        return payload, FeedbackSourceStatus(
            name="byo_container_feedback",
            tier="PARTIAL",
            evidence=f"Dispatched BYO feedback container via {route} in {mode} mode.",
        )


def parse_source_payload(
    payload: dict[str, Any],
    *,
    source: str,
    feedback_type: FeedbackType,
) -> FeedbackPayload:
    """Interpret a source response using its declared feedback type."""

    declared_type = parse_feedback_type(str(payload.get("feedback_type") or feedback_type.value))
    if declared_type != feedback_type:
        raise FeedbackSourceError(
            f"feedback source declared '{declared_type.value}', expected '{feedback_type.value}'"
        )
    value = payload.get("value")
    if value is None:
        value = _fallback_value(payload, feedback_type)
    score = _score_from_payload(payload, value, feedback_type)
    success = bool(payload.get("success", score >= 0.5))
    rationale = str(payload.get("rationale") or payload.get("critique") or "")
    return FeedbackPayload(
        source=str(payload.get("source") or source),
        feedback_type=feedback_type,
        value=value,
        score=score,
        success=success,
        rationale=rationale,
        metadata=dict(payload.get("metadata") or {}),
    )


def _payload_from_score(
    *,
    source: str,
    feedback_type: FeedbackType,
    score: float,
    success: bool,
    rationale: str,
    metadata: dict[str, Any] | None = None,
) -> FeedbackPayload:
    score = _bounded(score)
    if feedback_type == FeedbackType.SCALAR:
        value: Any = score
    elif feedback_type == FeedbackType.DENSE_PER_STEP:
        value = list(metadata.get("dense_rewards", [score]) if metadata else [score])
    elif feedback_type == FeedbackType.PASS_FAIL:
        value = success
    elif feedback_type == FeedbackType.CRITIQUE:
        value = {"score": score, "critique": rationale}
    else:
        value = {
            "chosen": "candidate" if success else "baseline",
            "rejected": "baseline" if success else "candidate",
            "score": score,
        }
    return FeedbackPayload(
        source=source,
        feedback_type=feedback_type,
        value=value,
        score=score,
        success=success,
        rationale=rationale,
        metadata=metadata or {},
    )


def _blocked_payload(
    *,
    source: str,
    feedback_type: FeedbackType,
    evidence: str,
) -> tuple[FeedbackPayload, FeedbackSourceStatus]:
    payload = FeedbackPayload(
        source=source,
        feedback_type=feedback_type,
        value=0.0,
        score=0.0,
        success=False,
        rationale=evidence,
    )
    return payload, FeedbackSourceStatus(name=f"{source.replace('-', '_')}_feedback", tier="BLOCKED", evidence=evidence)


def _adapt_scalar(payload: FeedbackPayload) -> TrainingSignal:
    score = _bounded(float(payload.value))
    return _base_signal(payload, score=score)


def _adapt_dense_per_step(payload: FeedbackPayload) -> TrainingSignal:
    dense = [_bounded(float(item)) for item in payload.value]
    score = sum(dense) / len(dense) if dense else 0.0
    signal = _base_signal(payload, score=score)
    return TrainingSignal(**{**signal.to_dict(), "dense_rewards": dense})


def _adapt_pass_fail(payload: FeedbackPayload) -> TrainingSignal:
    success = bool(payload.value)
    score = 1.0 if success else 0.0
    return _base_signal(payload, score=score, success=success)


def _adapt_critique(payload: FeedbackPayload) -> TrainingSignal:
    value = payload.value if isinstance(payload.value, dict) else {}
    score = _bounded(float(value.get("score", payload.score)))
    critique = str(value.get("critique") or payload.rationale)
    signal = _base_signal(payload, score=score)
    return TrainingSignal(**{**signal.to_dict(), "natural_language_critique": critique})


def _adapt_preference(payload: FeedbackPayload) -> TrainingSignal:
    value = payload.value if isinstance(payload.value, dict) else {}
    score = _bounded(float(value.get("score", payload.score)))
    signal = _base_signal(payload, score=score)
    return TrainingSignal(**{**signal.to_dict(), "preference": dict(value)})


def _base_signal(
    payload: FeedbackPayload,
    *,
    score: float,
    success: bool | None = None,
) -> TrainingSignal:
    score = _bounded(score)
    resolved_success = payload.success if success is None else success
    reward = score if resolved_success else min(score, 0.5)
    return TrainingSignal(
        schema="npa.sim_to_real.training_signal.v1",
        scalar_reward=round(float(reward), 6),
        success=bool(resolved_success),
        score=score,
        natural_language_critique=payload.rationale,
        loss_weight=round(1.0 + (1.0 - score), 6),
        source=payload.source,
        feedback_type=payload.feedback_type.value,
    )


def _fallback_value(payload: dict[str, Any], feedback_type: FeedbackType) -> Any:
    if feedback_type == FeedbackType.SCALAR:
        return payload.get("score", 0.0)
    if feedback_type == FeedbackType.DENSE_PER_STEP:
        return payload.get("dense_rewards") or payload.get("rewards") or []
    if feedback_type == FeedbackType.PASS_FAIL:
        return payload.get("success", False)
    if feedback_type == FeedbackType.CRITIQUE:
        return {"score": payload.get("score", 0.0), "critique": payload.get("critique") or payload.get("rationale", "")}
    return {
        "chosen": payload.get("chosen", "candidate"),
        "rejected": payload.get("rejected", "baseline"),
        "score": payload.get("score", 0.0),
    }


def _score_from_payload(payload: dict[str, Any], value: Any, feedback_type: FeedbackType) -> float:
    explicit = payload.get("score")
    if explicit is not None:
        return _bounded(float(explicit))
    if feedback_type == FeedbackType.SCALAR:
        return _bounded(float(value))
    if feedback_type == FeedbackType.DENSE_PER_STEP:
        values = [_bounded(float(item)) for item in value]
        return sum(values) / len(values) if values else 0.0
    if feedback_type == FeedbackType.PASS_FAIL:
        return 1.0 if bool(value) else 0.0
    if isinstance(value, dict):
        return _bounded(float(value.get("score", 0.0)))
    return 0.0


def _default_http_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise FeedbackSourceError(str(exc)) from exc


def _default_command_runner(command: list[str], payload: dict[str, Any]) -> dict[str, Any]:
    if not command:
        raise FeedbackSourceError("BYO feedback command must not be empty")
    result = subprocess.run(
        command,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        raise FeedbackSourceError(result.stderr.strip() or f"command exited {result.returncode}")
    return json.loads(result.stdout)


def _normalize_name(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _normalize_mode(value: str) -> str:
    normalized = _normalize_name(value or "provided-rollout")
    if normalized not in {"provided-rollout", "self-rollout"}:
        raise FeedbackSourceError("BYO feedback mode must be 'provided-rollout' or 'self-rollout'")
    return normalized


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


register_feedback_adapter(FeedbackType.SCALAR, _adapt_scalar)
register_feedback_adapter(FeedbackType.DENSE_PER_STEP, _adapt_dense_per_step)
register_feedback_adapter(FeedbackType.PASS_FAIL, _adapt_pass_fail)
register_feedback_adapter(FeedbackType.CRITIQUE, _adapt_critique)
register_feedback_adapter(FeedbackType.PREFERENCE, _adapt_preference)

register_feedback_source(NoneFeedbackSource())
register_feedback_source(SimEnvFeedbackSource(), aliases=("env",))
register_feedback_source(VlmFeedbackSource())
register_feedback_source(ByoContainerFeedbackSource(), aliases=("byo",))
