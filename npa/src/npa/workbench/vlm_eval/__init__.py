"""VLM evaluation helpers for sim-to-real pipeline validation."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from itertools import product
from io import BytesIO
import json
import math
import os
import posixpath
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

import httpx
import numpy as np
from PIL import Image
from urllib.parse import urlparse

if TYPE_CHECKING:
    from npa.clients.storage import StorageClient


DEFAULT_BACKEND = "self-hosted"
DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_ENDPOINT_URL = "http://127.0.0.1:8000/v1"
DEFAULT_FRAME_SELECTION = "keyframes"
DEFAULT_MAX_FRAMES = 4
DEFAULT_TIMEOUT_S = 120.0
# A self-hosted vLLM server cold-starts and loads weights for minutes; the eval
# client must wait for it to become reachable instead of failing on the first
# connection-refused. Override with NPA_VLM_READY_TIMEOUT_S.
DEFAULT_READY_TIMEOUT_S = 600.0
READY_TIMEOUT_ENV = "NPA_VLM_READY_TIMEOUT_S"
#: Set by a job that starts its own vLLM server, so the eval client requests the
#: model that server actually loaded rather than ``DEFAULT_MODEL``.
SELF_HOSTED_MODEL_ENV = "NPA_VLM_SELF_HOSTED_MODEL"
DEFAULT_API_KEY_ENV = "VLM_EVAL_API_KEY"
DEFAULT_RUBRIC = (
    "Score whether the rollout completes the requested physical task. "
    "Use 1.0 only for clear task completion, 0.0 for clear failure, and "
    "intermediate values for partial progress. Penalize unsafe, incomplete, "
    "or ambiguous outcomes."
)
#: Backend-neutral result name. The payload distinguishes fixtures from inference.
RESULT_FILENAME = "vlm_eval.json"
#: Read-only compatibility for bundles created before RESULT_FILENAME was neutral.
LEGACY_RESULT_FILENAME = "vlm_eval_stub.json"
#: The aggregate report a rollout-SET evaluation writes. Named for compatibility with
#: the retired sim-to-real-loop.yaml, whose readers key off this filename.
LOOP_REPORT_FILENAME = "task_success_report.json"
BENCHMARK_RESULT_FILENAME = "vlm_eval_benchmark.json"
JUDGE_COMPARISON_RESULT_FILENAME = "vlm_judge_disagreement.json"
BENCHMARK_DATASET_FORMAT = "npa_vlm_eval_benchmark_v1"
EVIDENCE_SCHEMA_VERSION = "npa_vlm_eval_evidence_v2"
JUDGE_COMPARISON_SCHEMA_VERSION = "npa_vlm_judge_comparison_v1"
HOSTED_RESPONSE_PARSER_VERSION = "npa_vlm_eval_hosted_json_v1"
SELF_HOSTED_RESPONSE_PARSER_VERSION = "npa_vlm_eval_compatible_json_v1"
MARKDOWN_FENCE_PARSER_SUFFIX = "+markdown-fence-v1"
UNPARSED_RESPONSE_PARSER_VERSION = "npa_vlm_eval_unparsed_v1"
DEFAULT_PRIMARY_JUDGE_MODEL = "MiniMaxAI/MiniMax-M3"
DEFAULT_SECONDARY_JUDGE_MODEL = "openbmb/MiniCPM-V-4_5"
DEFAULT_BENCHMARK_THRESHOLDS = (0.5, 0.8, 0.9)
DEFAULT_SAMPLE_BENCHMARK_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "sample_benchmark" / "benchmark.json"
)
SUPPORTED_BACKENDS = ("self-hosted", "api", "stub")
SUPPORTED_FRAME_SELECTIONS = ("final", "keyframes", "sequence")
CANONICAL_HOSTED_MODELS = frozenset(
    {
        "MiniMaxAI/MiniMax-M3",
        "google/gemma-3-27b-it",
        "nvidia/Nemotron-3_5-Lightning",
        "openbmb/MiniCPM-V-4_5",
    }
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".webp"}
VIDEO_SUFFIXES = {".avi", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


class VlmEvalError(ValueError):
    """Raised when a VLM evaluation request is invalid."""


@dataclass(frozen=True)
class VlmFrameEvidence:
    """Identify the exact normalized frame bytes submitted to a VLM.

    Args:
        label: Stable source-relative label for the selected frame.
        media_type: MIME type used in the multimodal request.
        sha256: Digest of the exact submitted bytes.
        byte_count: Size of the submitted bytes.
        width: Submitted image width in pixels.
        height: Submitted image height in pixels.
        source_kind: Input family from which the frame was selected.
        source_index: Zero-based source frame index when known.
        source_count: Number of available source frames when known.
        source_timestamp_s: Source video timestamp in seconds when known.

    Returns:
        None.

    Raises:
        None.
    """

    label: str
    media_type: str
    sha256: str
    byte_count: int
    width: int
    height: int
    source_kind: str | None = None
    source_index: int | None = None
    source_count: int | None = None
    source_timestamp_s: float | None = None


@dataclass(frozen=True)
class VlmRequestEvidence:
    """Retain a secret-free manifest of one multimodal judge request.

    Args:
        requested_at: UTC timestamp immediately before provider invocation.
        endpoint_role: Hosted API or self-hosted endpoint classification.
        prompt_sha256: Digest of the exact judge prompt.
        rubric_sha256: Digest of the exact visual rubric.
        request_manifest_sha256: Digest of the sanitized request manifest.
        request_manifest: Secret-free generation and frame metadata.
        frames: Exact submitted-frame identities.

    Returns:
        None.

    Raises:
        None.
    """

    requested_at: str
    endpoint_role: str
    prompt_sha256: str
    rubric_sha256: str
    request_manifest_sha256: str
    request_manifest: dict[str, Any]
    frames: tuple[VlmFrameEvidence, ...]


@dataclass(frozen=True)
class VlmProviderEvidence:
    """Retain exact provider completion metadata and response bytes.

    Args:
        provider_request_id: Provider-returned request identity when available.
        returned_model: Provider-returned model identity when available.
        finish_reason: Provider completion reason when available.
        latency_s: End-to-end request latency in seconds.
        status_code: HTTP status code when transport metadata is available.
        usage: Provider-returned usage object when available.
        raw_response: Exact HTTP response body, or canonical mocked response.
        raw_response_sha256: Digest of ``raw_response``.
        parser_version: Parser contract applied to the response.

    Returns:
        None.

    Raises:
        None.
    """

    provider_request_id: str | None
    returned_model: str | None
    finish_reason: str | None
    latency_s: float
    status_code: int | None
    usage: dict[str, Any] | None
    raw_response: str
    raw_response_sha256: str
    parser_version: str


@dataclass(frozen=True)
class VlmEvaluationEvidence:
    """Bind request and provider evidence to one parsed VLM verdict.

    Args:
        schema_version: Evaluation evidence schema identifier.
        request: Sanitized multimodal request provenance.
        provider: Provider completion provenance.

    Returns:
        None.

    Raises:
        None.
    """

    schema_version: str
    request: VlmRequestEvidence
    provider: VlmProviderEvidence


@dataclass(frozen=True)
class VlmEvalResult:
    status: str
    backend: str
    input_path: str
    output_path: str
    result_uri: str
    task: str
    model: str
    score: float
    success_threshold: float
    passed: bool
    generated_at: str
    frame_selection: str = DEFAULT_FRAME_SELECTION
    frame_count: int = 0
    rationale: str = ""
    served_model: str | None = None
    evidence: VlmEvaluationEvidence | None = None
    rubric: str = DEFAULT_RUBRIC
    provider_success: bool | None = None
    provider_success_matches_score_gate: bool | None = None


@dataclass(frozen=True)
class VlmJudgeError:
    """Retain one paired judge's typed failure and available provenance.

    Args:
        model: Exact requested model identity.
        stage: Request stage that failed.
        error_type: Stable transport or response failure category.
        message: Bounded diagnostic without authorization data.
        request: Request evidence created before transport.
        provider: Provider evidence when a response was received.

    Returns:
        None.

    Raises:
        None.
    """

    model: str
    stage: str
    error_type: str
    message: str
    request: VlmRequestEvidence
    provider: VlmProviderEvidence | None = None


@dataclass(frozen=True)
class VlmJudgeOutcome:
    """Represent exactly one success or error in a paired judge comparison.

    Args:
        model: Exact requested model identity.
        transport_request_sha256: Digest of the exact JSON request object.
        result: Complete scalar evaluation when parsing succeeded.
        error: Typed failure record when the attempt did not produce a verdict.

    Returns:
        None.

    Raises:
        None.
    """

    model: str
    transport_request_sha256: str
    result: VlmEvalResult | None
    error: VlmJudgeError | None


@dataclass(frozen=True)
class VlmJudgeComparisonReport:
    """Preserve two hosted judge outcomes without averaging disagreement.

    Args:
        schema_version: Comparison artifact schema identifier.
        status: Agreement, disagreement, or judge-error classification.
        passed: True only when both score-derived verdicts pass.
        escalation_required: Whether disagreement or error needs human review.
        deployment_status: Explicit audit-only qualification boundary.
        operational_rate_estimated: Always false for one comparison.
        input_path: Rollout source supplied by the caller.
        output_path: Artifact destination supplied by the caller.
        result_uri: Exact comparison artifact destination.
        task: Task text shared by both judges.
        rubric: Rubric text shared by both judges.
        success_threshold: Score-derived verdict threshold.
        frame_selection: Shared frame selection strategy.
        frame_count: Number of shared normalized frames.
        shared_frame_sha256: Ordered exact normalized-frame digests.
        common_request_sha256: Digest after excluding the model field.
        requests_differ_only_by_model: Pre-transport equivalence assertion.
        primary: Complete primary outcome.
        secondary: Complete secondary outcome.
        score_delta_secondary_minus_primary: Descriptive delta, never a mean.
        generated_at: UTC artifact timestamp.
        limitations: Explicit interpretation boundaries.

    Returns:
        None.

    Raises:
        None.
    """

    schema_version: str
    status: str
    passed: bool
    escalation_required: bool
    deployment_status: str
    operational_rate_estimated: bool
    input_path: str
    output_path: str
    result_uri: str
    task: str
    rubric: str
    success_threshold: float
    frame_selection: str
    frame_count: int
    shared_frame_sha256: tuple[str, ...]
    common_request_sha256: str
    requests_differ_only_by_model: bool
    primary: VlmJudgeOutcome
    secondary: VlmJudgeOutcome
    score_delta_secondary_minus_primary: float | None
    generated_at: str
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class VlmStructuredResponse:
    success: bool
    score: float
    rationale: str
    served_model: str | None = None
    evidence: VlmEvaluationEvidence | None = None
    parser_version: str = SELF_HOSTED_RESPONSE_PARSER_VERSION
    provider_success: bool | None = None


@dataclass(frozen=True)
class SelectedFrame:
    label: str
    media_type: str
    data: bytes
    source_kind: str | None = None
    source_index: int | None = None
    source_count: int | None = None
    source_timestamp_s: float | None = None


@dataclass(frozen=True)
class _VlmBackendResponse:
    data: dict[str, Any]
    raw_body: str
    status_code: int | None
    request_id_header: str | None
    latency_s: float


@dataclass(frozen=True)
class _VlmJudgeContext:
    input_path: str
    output_path: str
    task: str
    rubric: str
    success_threshold: float
    frame_selection: str
    max_frames: int
    endpoint_url: str
    api_key_env: str
    timeout_s: float
    prompt: str
    frames: tuple[SelectedFrame, ...]


@dataclass(frozen=True)
class VlmBenchmarkItem:
    id: str
    rollout: str
    expected_label: bool
    task: str
    fixture_score: float | None = None


@dataclass(frozen=True)
class VlmBenchmarkDataset:
    path: str
    format: str
    items: list[VlmBenchmarkItem]
    rubrics: dict[str, str]


@dataclass(frozen=True)
class VlmBenchmarkConfig:
    backend: str
    model: str
    rubric_name: str
    rubric: str
    success_threshold: float
    frame_selection: str
    max_frames: int


@dataclass(frozen=True)
class VlmBenchmarkMetrics:
    total: int
    correct: int
    agreement: float
    accuracy: float
    precision: float | None
    recall: float | None
    f1: float | None
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class VlmBenchmarkCaseResult:
    item_id: str
    rollout: str
    expected_label: bool
    predicted_label: bool
    score: float
    status: str
    passed: bool
    task: str
    rationale: str
    frame_count: int
    score_source: str
    evidence: VlmEvaluationEvidence | None
    provider_success: bool | None = None
    provider_success_matches_score_gate: bool | None = None


@dataclass(frozen=True)
class VlmBenchmarkConfigResult:
    rank: int
    config: VlmBenchmarkConfig
    metrics: VlmBenchmarkMetrics
    results: list[VlmBenchmarkCaseResult]


@dataclass(frozen=True)
class VlmBenchmarkReport:
    status: str
    dataset_path: str
    dataset_format: str
    item_count: int
    generated_at: str
    sweep: dict[str, Any]
    best_config: VlmBenchmarkConfigResult
    ranked_configs: list[VlmBenchmarkConfigResult]


__all__ = [
    "VlmBenchmarkCaseResult",
    "VlmBenchmarkConfig",
    "VlmBenchmarkConfigResult",
    "VlmBenchmarkDataset",
    "VlmBenchmarkItem",
    "VlmBenchmarkMetrics",
    "VlmBenchmarkReport",
    "VlmEvaluationEvidence",
    "VlmEvalResult",
    "VlmFrameEvidence",
    "VlmJudgeComparisonReport",
    "VlmJudgeError",
    "VlmJudgeOutcome",
    "VlmProviderEvidence",
    "VlmRequestEvidence",
    "VlmStructuredResponse",
    "benchmark_result_uri_for",
    "benchmark_vlm_eval",
    "compare_vlm_judges",
    "VlmLoopRollout",
    "aggregate_loop_report",
    "discover_rollouts",
    "evaluate_rollout_set",
    "evaluate_stub",
    "evaluate_vlm",
    "loop_report_uri_for",
    "judge_comparison_result_uri_for",
    "load_benchmark_dataset",
    "parse_structured_response",
    "result_uri_for",
    "select_rollout_frames",
    "write_benchmark_report",
    "write_result",
]


def benchmark_vlm_eval(
    *,
    dataset: str = str(DEFAULT_SAMPLE_BENCHMARK_PATH),
    thresholds: Sequence[float] = DEFAULT_BENCHMARK_THRESHOLDS,
    rubrics: Sequence[str] = ("default",),
    models: Sequence[str] = (DEFAULT_MODEL,),
    backend: str = DEFAULT_BACKEND,
    task: str = "sim-to-real",
    frame_selection: str = DEFAULT_FRAME_SELECTION,
    max_frames: int = DEFAULT_MAX_FRAMES,
    endpoint_url: str = "",
    api_key_env: str = DEFAULT_API_KEY_ENV,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    use_fixture_scores: bool = False,
) -> VlmBenchmarkReport:
    """Run a labeled VLM-eval sweep and rank configs by label agreement."""

    # Resolve the packaged sample fixture from its install location so callers
    # (and the npa.workflow twin) get a working default regardless of CWD. A
    # repo-relative path does not exist inside a rendered job; the ``sample``/
    # ``default`` sentinels (and empty) map to the packaged fixture.
    if dataset.strip().lower() in {"", "sample", "default"}:
        dataset = str(DEFAULT_SAMPLE_BENCHMARK_PATH)
    benchmark_dataset = load_benchmark_dataset(dataset, default_task=task)
    threshold_values = _normalize_thresholds(thresholds)
    model_values = _normalize_strings(models, label="models")
    rubric_values = _resolve_benchmark_rubrics(
        rubrics,
        dataset_rubrics=benchmark_dataset.rubrics,
        dataset_path=benchmark_dataset.path,
    )
    effective_backend = _normalize_backend(backend)
    effective_frame_selection = _normalize_frame_selection(frame_selection)
    if max_frames <= 0:
        raise VlmEvalError("--max-frames must be positive")
    if timeout_s <= 0:
        raise VlmEvalError("--timeout-s must be positive")

    config_results: list[VlmBenchmarkConfigResult] = []
    for model, (rubric_name, rubric_text), threshold in product(
        model_values,
        rubric_values,
        threshold_values,
    ):
        config = VlmBenchmarkConfig(
            backend=effective_backend,
            model=model,
            rubric_name=rubric_name,
            rubric=rubric_text,
            success_threshold=threshold,
            frame_selection=effective_frame_selection,
            max_frames=max_frames,
        )
        case_results = [
            _run_benchmark_case(
                item,
                config=config,
                endpoint_url=endpoint_url,
                api_key_env=api_key_env,
                timeout_s=timeout_s,
                use_fixture_score=use_fixture_scores or effective_backend == "stub",
            )
            for item in benchmark_dataset.items
        ]
        config_results.append(
            VlmBenchmarkConfigResult(
                rank=0,
                config=config,
                metrics=_benchmark_metrics(case_results),
                results=case_results,
            )
        )

    ranked = [
        VlmBenchmarkConfigResult(
            rank=index,
            config=result.config,
            metrics=result.metrics,
            results=result.results,
        )
        for index, result in enumerate(
            sorted(config_results, key=_benchmark_rank_key), start=1
        )
    ]
    if not ranked:
        raise VlmEvalError("benchmark sweep produced no configurations")

    return VlmBenchmarkReport(
        status="completed",
        dataset_path=benchmark_dataset.path,
        dataset_format=benchmark_dataset.format,
        item_count=len(benchmark_dataset.items),
        generated_at=datetime.now(timezone.utc).isoformat(),
        sweep={
            "backend": effective_backend,
            "models": model_values,
            "rubrics": [name for name, _text in rubric_values],
            "thresholds": threshold_values,
            "frame_selection": effective_frame_selection,
            "max_frames": max_frames,
            "fixture_scores": use_fixture_scores or effective_backend == "stub",
        },
        best_config=ranked[0],
        ranked_configs=ranked,
    )


def load_benchmark_dataset(
    dataset: str = str(DEFAULT_SAMPLE_BENCHMARK_PATH),
    *,
    default_task: str = "sim-to-real",
) -> VlmBenchmarkDataset:
    """Load a labeled benchmark dataset manifest from a local path or S3 URI."""

    if not dataset:
        raise VlmEvalError("--dataset is required")
    with _materialized_benchmark_manifest(dataset) as local_manifest:
        try:
            payload = json.loads(local_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VlmEvalError(
                f"benchmark dataset is not valid JSON: {local_manifest}"
            ) from exc

    if isinstance(payload, list):
        raw_items = payload
        dataset_format = BENCHMARK_DATASET_FORMAT
        rubrics: dict[str, str] = {}
        rollout_base_path = ""
    elif isinstance(payload, dict):
        raw_items = payload.get("items") or payload.get("rollouts")
        dataset_format = str(payload.get("format") or BENCHMARK_DATASET_FORMAT)
        rubrics = _coerce_rubric_map(payload.get("rubrics", {}))
        rollout_base_path = str(
            payload.get("rollout_base_path") or payload.get("base_path") or ""
        )
    else:
        raise VlmEvalError("benchmark dataset JSON must be an object or an item list")

    if not isinstance(raw_items, list) or not raw_items:
        raise VlmEvalError("benchmark dataset must include a non-empty items list")

    local_base = local_manifest.parent
    source_base = _dataset_source_base(dataset)
    rollout_base = _resolve_rollout_base(
        rollout_base_path,
        source_base=source_base,
        local_base=local_base,
    )
    items = [
        _parse_benchmark_item(
            raw_item,
            index=index,
            rollout_base=rollout_base,
            default_task=default_task,
        )
        for index, raw_item in enumerate(raw_items, start=1)
    ]
    return VlmBenchmarkDataset(
        path=dataset,
        format=dataset_format,
        items=items,
        rubrics=rubrics,
    )


def evaluate_vlm(
    *,
    input_path: str,
    output_path: str,
    task: str = "sim-to-real",
    backend: str = DEFAULT_BACKEND,
    model: str = DEFAULT_MODEL,
    success_threshold: float = 0.8,
    frame_selection: str = DEFAULT_FRAME_SELECTION,
    max_frames: int = DEFAULT_MAX_FRAMES,
    endpoint_url: str = "",
    api_key_env: str = DEFAULT_API_KEY_ENV,
    rubric: str = DEFAULT_RUBRIC,
    rubric_path: str = "",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    score: float | None = None,
) -> VlmEvalResult:
    """Evaluate rollout frames with a VLM and return a scalar score in [0, 1]."""

    _validate_common(
        input_path=input_path,
        output_path=output_path,
        success_threshold=success_threshold,
        frame_selection=frame_selection,
        max_frames=max_frames,
        timeout_s=timeout_s,
    )
    backend = _normalize_backend(backend)
    effective_rubric = _load_rubric(rubric=rubric, rubric_path=rubric_path)
    if backend == "stub":
        return evaluate_stub(
            input_path=input_path,
            output_path=output_path,
            task=task,
            model=model or "vlm-eval-stub",
            success_threshold=success_threshold,
            frame_selection=frame_selection,
            score=score,
            rubric=effective_rubric,
        )

    effective_model = model or DEFAULT_MODEL
    if backend == "self-hosted" and effective_model == DEFAULT_MODEL:
        # The job that started the vLLM server records which model it serves, so
        # the client asks for that one instead of the 7B default (a mismatch is a
        # 404 from the server). See `_vllm_serve_preamble` in the workflow render.
        effective_model = (
            os.environ.get(SELF_HOSTED_MODEL_ENV, "").strip() or effective_model
        )
    if backend == "api" and effective_model == DEFAULT_MODEL:
        # DEFAULT_MODEL is the self-hosted (vLLM) default. The hosted Token
        # Factory API does not serve it (requests 404); use the vision model
        # Token Factory actually serves unless the caller overrode --model.
        from npa.clients.token_factory import DEFAULT_VISION_MODEL

        effective_model = DEFAULT_VISION_MODEL
    if score is not None:
        _validate_score_override(score)
        structured = VlmStructuredResponse(
            success=score >= success_threshold,
            score=score,
            rationale="Score override supplied; VLM call skipped.",
        )
        frame_count = 0
        effective_task = task
    else:
        with _materialized_input(input_path) as local_input:
            effective_task = _resolve_task_text(local_input, task)
            frames = select_rollout_frames(
                local_input,
                frame_selection=frame_selection,
                max_frames=max_frames,
            )
            prompt = _build_prompt(
                task=effective_task,
                rubric=effective_rubric,
                frame_selection=frame_selection,
                frame_count=len(frames),
            )
            structured = _call_openai_compatible(
                backend=backend,
                model=effective_model,
                endpoint_url=endpoint_url,
                api_key_env=api_key_env,
                prompt=prompt,
                rubric=effective_rubric,
                frames=frames,
                timeout_s=timeout_s,
                frame_selection=frame_selection,
                max_frames=max_frames,
            )
            frame_count = len(frames)

    return _result_from_structured(
        backend=backend,
        input_path=input_path,
        output_path=output_path,
        task=effective_task,
        model=effective_model,
        success_threshold=success_threshold,
        frame_selection=frame_selection,
        frame_count=frame_count,
        rubric=effective_rubric,
        structured=structured,
    )


def compare_vlm_judges(
    *,
    input_path: str,
    output_path: str,
    primary_model: str,
    secondary_model: str,
    task: str = "sim-to-real",
    success_threshold: float = 0.8,
    frame_selection: str = DEFAULT_FRAME_SELECTION,
    max_frames: int = DEFAULT_MAX_FRAMES,
    endpoint_url: str = "",
    api_key_env: str = DEFAULT_API_KEY_ENV,
    rubric: str = DEFAULT_RUBRIC,
    rubric_path: str = "",
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> VlmJudgeComparisonReport:
    """Run two distinct hosted judges over one immutable prompt and frame set.

    Args:
        input_path: One rollout image, video, array, directory, or S3 prefix.
        output_path: Destination for the comparison artifact.
        primary_model: First hosted vision model ID.
        secondary_model: Distinct second hosted vision model ID.
        task: Shared physical-task instruction.
        success_threshold: Score threshold applied to each result separately.
        frame_selection: Shared final, keyframes, or sequence strategy.
        max_frames: Maximum number of shared normalized frames.
        endpoint_url: Optional hosted OpenAI-compatible endpoint override.
        api_key_env: Environment variable containing the hosted API key.
        rubric: Shared visual scoring rubric.
        rubric_path: Optional local file that replaces ``rubric``.
        timeout_s: Timeout for each provider request.

    Returns:
        An audit-only report retaining both outcomes without averaging.

    Raises:
        VlmEvalError: If configuration, input, or shared evidence is invalid.
    """

    _validate_common(
        input_path=input_path,
        output_path=output_path,
        success_threshold=success_threshold,
        frame_selection=frame_selection,
        max_frames=max_frames,
        timeout_s=timeout_s,
    )
    models = _comparison_models(primary_model, secondary_model)
    effective_rubric = _load_rubric(rubric=rubric, rubric_path=rubric_path)
    with _materialized_input(input_path) as local_input:
        context = _comparison_context(
            local_input=local_input,
            input_path=input_path,
            output_path=output_path,
            task=task,
            rubric=effective_rubric,
            success_threshold=success_threshold,
            frame_selection=frame_selection,
            max_frames=max_frames,
            endpoint_url=endpoint_url,
            api_key_env=api_key_env,
            timeout_s=timeout_s,
        )
        return _run_judge_comparison(context, models)


def _comparison_models(primary: str, secondary: str) -> tuple[str, str]:
    models = (primary.strip(), secondary.strip())
    if not all(models):
        raise VlmEvalError("paired judges require two nonempty model IDs")
    if models[0] == models[1]:
        raise VlmEvalError("paired judges require two distinct model IDs")
    return models


def _comparison_context(
    local_input: Path,
    input_path: str,
    output_path: str,
    task: str,
    rubric: str,
    success_threshold: float,
    frame_selection: str,
    max_frames: int,
    endpoint_url: str,
    api_key_env: str,
    timeout_s: float,
) -> _VlmJudgeContext:
    effective_task = _resolve_task_text(local_input, task)
    frames = tuple(
        select_rollout_frames(
            local_input,
            frame_selection=frame_selection,
            max_frames=max_frames,
        )
    )
    prompt = _comparison_prompt(effective_task, rubric, frame_selection, len(frames))
    return _VlmJudgeContext(
        input_path,
        output_path,
        effective_task,
        rubric,
        success_threshold,
        frame_selection,
        max_frames,
        endpoint_url,
        api_key_env,
        timeout_s,
        prompt,
        frames,
    )


def _comparison_prompt(
    task: str,
    rubric: str,
    frame_selection: str,
    frame_count: int,
) -> str:
    return _build_prompt(
        task=task,
        rubric=rubric,
        frame_selection=frame_selection,
        frame_count=frame_count,
    )


def _run_judge_comparison(
    context: _VlmJudgeContext,
    models: tuple[str, str],
) -> VlmJudgeComparisonReport:
    common_request = _common_hosted_request(
        prompt=context.prompt,
        frames=context.frames,
    )
    requests = tuple(_request_for_model(common_request, model) for model in models)
    common_sha256 = _assert_model_only_request_difference(requests)
    url = _chat_completions_url(
        _resolve_endpoint_url(backend="api", endpoint_url=context.endpoint_url)
    )
    api_key = _resolve_api_key(backend="api", api_key_env=context.api_key_env)
    outcomes = tuple(
        _call_comparison_judge(
            request=request,
            url=url,
            api_key=api_key,
            context=context,
        )
        for request in requests
    )
    return _build_judge_comparison_report(
        context=context,
        common_request_sha256=common_sha256,
        primary=outcomes[0],
        secondary=outcomes[1],
    )


def _common_hosted_request(
    *, prompt: str, frames: Sequence[SelectedFrame]
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in frames:
        encoded = base64.b64encode(frame.data).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{frame.media_type};base64,{encoded}"},
            }
        )
    return {
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
    }


def _request_for_model(common_request: dict[str, Any], model: str) -> dict[str, Any]:
    isolated_request = json.loads(_canonical_json(common_request))
    return {"model": model, **isolated_request}


def _assert_model_only_request_difference(
    requests: Sequence[dict[str, Any]],
) -> str:
    if len(requests) != 2:
        raise VlmEvalError("paired judge comparison requires exactly two requests")
    common_requests = []
    for request in requests:
        common = dict(request)
        model = common.pop("model", None)
        if not isinstance(model, str) or not model:
            raise VlmEvalError("paired judge request has no model identity")
        common_requests.append(common)
    if common_requests[0] != common_requests[1]:
        raise VlmEvalError("paired judge requests differ by more than model")
    return _sha256_json(common_requests[0])


def _call_comparison_judge(
    *,
    request: dict[str, Any],
    url: str,
    api_key: str,
    context: _VlmJudgeContext,
) -> VlmJudgeOutcome:
    model = str(request["model"])
    request_evidence = _comparison_request_evidence(request, context)
    request_sha256 = _sha256_json(request)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    response, error = _post_comparison_request(
        url=url,
        headers=headers,
        request=request,
        timeout_s=context.timeout_s,
    )
    if error is not None:
        return _comparison_transport_error_outcome(
            model=model,
            request_sha256=request_sha256,
            request=request_evidence,
            response=response,
            error=error,
        )
    if response is None:
        raise VlmEvalError("paired judge transport returned no outcome")
    return _parse_comparison_response(
        model=model,
        request_sha256=request_sha256,
        request=request_evidence,
        response=response,
        context=context,
    )


def _comparison_transport_error_outcome(
    *,
    model: str,
    request_sha256: str,
    request: VlmRequestEvidence,
    response: _VlmBackendResponse | None,
    error: VlmEvalError,
) -> VlmJudgeOutcome:
    stage, error_type = _comparison_transport_error_kind(response)
    provider = None
    if response is not None:
        choice = _available_response_choice(response.data)
        provider = _unparsed_provider_evidence(response, choice)
    return _comparison_error_outcome(
        model=model,
        request_sha256=request_sha256,
        request=request,
        stage=stage,
        error_type=error_type,
        error=error,
        provider=provider,
    )


def _comparison_request_evidence(
    request: dict[str, Any],
    context: _VlmJudgeContext,
) -> VlmRequestEvidence:
    return _build_request_evidence(
        backend="api",
        model=str(request["model"]),
        prompt=context.prompt,
        rubric=context.rubric,
        request=request,
        frames=context.frames,
        frame_selection=context.frame_selection,
        max_frames=context.max_frames,
    )


def _post_comparison_request(
    *,
    url: str,
    headers: dict[str, str],
    request: dict[str, Any],
    timeout_s: float,
) -> tuple[_VlmBackendResponse | None, VlmEvalError | None]:
    started_at = time.monotonic()
    captured: list[_VlmBackendResponse] = []
    try:
        raw_response = _post_with_readiness_retry(
            url=url,
            headers=headers,
            request=request,
            backend="api",
            timeout_s=timeout_s,
            error_response_sink=captured.append,
        )
        response = _coerce_backend_response(
            raw_response,
            fallback_latency_s=time.monotonic() - started_at,
        )
        return response, None
    except VlmEvalError as exc:
        response = captured[0] if captured else None
        return response, exc


def _parse_comparison_response(
    model: str,
    request_sha256: str,
    request: VlmRequestEvidence,
    response: _VlmBackendResponse,
    context: _VlmJudgeContext,
) -> VlmJudgeOutcome:
    choice = _available_response_choice(response.data)
    try:
        parsed_choice, message = _response_choice_and_content(response.data)
        structured = _parse_backend_verdict(
            backend="api",
            requested_model=model,
            data=response.data,
            choice=parsed_choice,
            message=message,
        )
        evidence = _build_evaluation_evidence(
            request,
            response,
            parsed_choice,
            parser_version=structured.parser_version,
        )
    except VlmEvalError as exc:
        return _comparison_error_outcome(
            model=model,
            request_sha256=request_sha256,
            request=request,
            stage="response_contract",
            error_type="response_contract_error",
            error=exc,
            provider=_unparsed_provider_evidence(response, choice),
        )
    return _comparison_success_outcome(
        model=model,
        request_sha256=request_sha256,
        structured=replace(structured, evidence=evidence),
        context=context,
    )


def _comparison_success_outcome(
    *,
    model: str,
    request_sha256: str,
    structured: VlmStructuredResponse,
    context: _VlmJudgeContext,
) -> VlmJudgeOutcome:
    result = _result_from_structured(
        backend="api",
        input_path=context.input_path,
        output_path=context.output_path,
        task=context.task,
        model=model,
        success_threshold=context.success_threshold,
        frame_selection=context.frame_selection,
        frame_count=len(context.frames),
        rubric=context.rubric,
        structured=structured,
    )
    result = replace(
        result,
        result_uri=judge_comparison_result_uri_for(context.output_path),
    )
    return VlmJudgeOutcome(model, request_sha256, result, None)


def _comparison_error_outcome(
    *,
    model: str,
    request_sha256: str,
    request: VlmRequestEvidence,
    stage: str,
    error_type: str,
    error: VlmEvalError,
    provider: VlmProviderEvidence | None = None,
) -> VlmJudgeOutcome:
    failure = VlmJudgeError(
        model=model,
        stage=stage,
        error_type=error_type,
        message=str(error)[:1000],
        request=request,
        provider=provider,
    )
    return VlmJudgeOutcome(model, request_sha256, None, failure)


def _comparison_transport_error_kind(
    response: _VlmBackendResponse | None,
) -> tuple[str, str]:
    if response is None:
        return "transport", "transport_error"
    if response.status_code is not None and response.status_code >= 400:
        return "provider_http_status", "provider_http_status_error"
    return "response_decode", "provider_response_decode_error"


def _available_response_choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _unparsed_provider_evidence(
    response: _VlmBackendResponse,
    choice: dict[str, Any],
) -> VlmProviderEvidence:
    provider_id = response.data.get("id")
    returned_model = response.data.get("model")
    finish_reason = choice.get("finish_reason")
    usage = response.data.get("usage")
    return VlmProviderEvidence(
        provider_request_id=(
            provider_id if isinstance(provider_id, str) else response.request_id_header
        ),
        returned_model=returned_model if isinstance(returned_model, str) else None,
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        latency_s=round(response.latency_s, 6),
        status_code=response.status_code,
        usage=usage if isinstance(usage, dict) else None,
        raw_response=response.raw_body,
        raw_response_sha256=_sha256_text(response.raw_body),
        parser_version=UNPARSED_RESPONSE_PARSER_VERSION,
    )


def _build_judge_comparison_report(
    *,
    context: _VlmJudgeContext,
    common_request_sha256: str,
    primary: VlmJudgeOutcome,
    secondary: VlmJudgeOutcome,
) -> VlmJudgeComparisonReport:
    expected_frames = tuple(_frame_evidence(frame) for frame in context.frames)
    _validate_shared_comparison_evidence(primary, secondary, expected_frames)
    status = _comparison_status(primary, secondary)
    return VlmJudgeComparisonReport(
        schema_version=JUDGE_COMPARISON_SCHEMA_VERSION,
        status=status,
        passed=status == "judges_agree_passed",
        escalation_required=status in {"judge_error", "judge_disagreement"},
        deployment_status="audit_only",
        operational_rate_estimated=False,
        input_path=context.input_path,
        output_path=context.output_path,
        result_uri=judge_comparison_result_uri_for(context.output_path),
        task=context.task,
        rubric=context.rubric,
        success_threshold=context.success_threshold,
        frame_selection=context.frame_selection,
        frame_count=len(context.frames),
        shared_frame_sha256=tuple(frame.sha256 for frame in expected_frames),
        common_request_sha256=common_request_sha256,
        requests_differ_only_by_model=True,
        primary=primary,
        secondary=secondary,
        score_delta_secondary_minus_primary=_comparison_score_delta(primary, secondary),
        generated_at=datetime.now(timezone.utc).isoformat(),
        limitations=_comparison_limitations(),
    )


def _validate_shared_comparison_evidence(
    primary: VlmJudgeOutcome,
    secondary: VlmJudgeOutcome,
    expected_frames: tuple[VlmFrameEvidence, ...],
) -> None:
    _validate_comparison_outcome(primary, expected_frames)
    _validate_comparison_outcome(secondary, expected_frames)
    primary_request = _outcome_request(primary)
    secondary_request = _outcome_request(secondary)
    if primary_request.prompt_sha256 != secondary_request.prompt_sha256:
        raise VlmEvalError("paired judge prompt evidence does not match")
    if primary_request.rubric_sha256 != secondary_request.rubric_sha256:
        raise VlmEvalError("paired judge rubric evidence does not match")


def _validate_comparison_outcome(
    outcome: VlmJudgeOutcome,
    expected_frames: tuple[VlmFrameEvidence, ...],
) -> None:
    if (outcome.result is None) == (outcome.error is None):
        raise VlmEvalError("judge outcome must contain exactly one result or error")
    request = _outcome_request(outcome)
    if request.frames != expected_frames:
        raise VlmEvalError("paired judge frame evidence does not match")
    if request.endpoint_role != "hosted-api":
        raise VlmEvalError("paired judge outcome is not from the hosted API")
    requested_model = request.request_manifest.get("requested_model")
    if requested_model != outcome.model:
        raise VlmEvalError("paired judge request model evidence does not match")


def _outcome_request(outcome: VlmJudgeOutcome) -> VlmRequestEvidence:
    if outcome.result is not None and outcome.result.evidence is not None:
        return outcome.result.evidence.request
    if outcome.error is not None:
        return outcome.error.request
    raise VlmEvalError("paired judge outcome has no request evidence")


def _comparison_status(
    primary: VlmJudgeOutcome,
    secondary: VlmJudgeOutcome,
) -> str:
    if primary.error is not None or secondary.error is not None:
        return "judge_error"
    if primary.result is None or secondary.result is None:
        raise VlmEvalError("paired judge result is incomplete")
    if primary.result.passed != secondary.result.passed:
        return "judge_disagreement"
    if primary.result.passed:
        return "judges_agree_passed"
    return "judges_agree_needs_iteration"


def _comparison_score_delta(
    primary: VlmJudgeOutcome,
    secondary: VlmJudgeOutcome,
) -> float | None:
    if primary.result is None or secondary.result is None:
        return None
    return round(secondary.result.score - primary.result.score, 4)


def _comparison_limitations() -> tuple[str, ...]:
    return (
        "Audit record only; this comparison does not qualify either judge.",
        "A weak judge can create disagreement, so disagreement does not prove case ambiguity.",
        "Scores are never averaged; both original outcomes remain authoritative.",
        "One comparison does not estimate an operational disagreement rate.",
        "In-image instructions remain an unresolved input-integrity risk.",
        "Judge agreement cannot prove that a critical visible defect is absent.",
        "Visual agreement does not establish physical correctness or robot safety.",
    )


def evaluate_stub(
    *,
    input_path: str,
    output_path: str,
    task: str = "sim-to-real",
    model: str = "vlm-eval-stub",
    success_threshold: float = 0.8,
    frame_selection: str = DEFAULT_FRAME_SELECTION,
    score: float | None = None,
    rubric: str = DEFAULT_RUBRIC,
) -> VlmEvalResult:
    """Return deterministic schema-compatible metrics without calling a VLM."""

    _validate_common(
        input_path=input_path,
        output_path=output_path,
        success_threshold=success_threshold,
        frame_selection=frame_selection,
        max_frames=DEFAULT_MAX_FRAMES,
        timeout_s=DEFAULT_TIMEOUT_S,
    )
    effective_score = (
        _deterministic_score(input_path, task, model) if score is None else score
    )
    _validate_score_override(effective_score)
    passed = effective_score >= success_threshold
    return VlmEvalResult(
        status="passed" if passed else "needs_iteration",
        backend="stub",
        input_path=input_path,
        output_path=output_path,
        result_uri=result_uri_for(output_path),
        task=task,
        model=model,
        score=round(effective_score, 4),
        success_threshold=success_threshold,
        passed=passed,
        generated_at=datetime.now(timezone.utc).isoformat(),
        frame_selection=frame_selection,
        frame_count=0,
        rationale="Deterministic compatibility score.",
        rubric=rubric,
    )


def select_rollout_frames(
    input_path: str | Path,
    *,
    frame_selection: str = DEFAULT_FRAME_SELECTION,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[SelectedFrame]:
    """Load selected rollout frames from image files, numpy episodes, or video files."""

    frame_selection = _normalize_frame_selection(frame_selection)
    if max_frames <= 0:
        raise VlmEvalError("--max-frames must be positive")

    path = Path(input_path)
    image_frames = _frames_from_images(
        path, frame_selection=frame_selection, max_frames=max_frames
    )
    if image_frames:
        return image_frames

    numpy_frames = _frames_from_numpy(
        path, frame_selection=frame_selection, max_frames=max_frames
    )
    if numpy_frames:
        return numpy_frames

    video_frames = _frames_from_videos(
        path, frame_selection=frame_selection, max_frames=max_frames
    )
    if video_frames:
        return video_frames

    raise VlmEvalError(
        f"No rollout frames found in {path}. Expected image files, RGB .npy/.npz arrays, or videos."
    )


def parse_structured_response(text: str) -> VlmStructuredResponse:
    """Parse a VLM JSON response and clamp its score into [0, 1]."""

    payload, deframed = _load_json_object(text)
    if "score" not in payload:
        raise VlmEvalError("VLM response JSON must include score")
    if "rationale" not in payload:
        raise VlmEvalError("VLM response JSON must include rationale")
    score = _clamp_score(payload["score"])
    success_supplied = "success" in payload
    success = _coerce_bool(payload["success"]) if success_supplied else score >= 0.5
    provider_success = payload.get("success")
    if not isinstance(provider_success, bool):
        provider_success = None
    return VlmStructuredResponse(
        success=success,
        score=score,
        rationale=str(payload["rationale"]),
        parser_version=_parser_version(SELF_HOSTED_RESPONSE_PARSER_VERSION, deframed),
        provider_success=provider_success,
    )


def _parse_api_structured_response(
    text: Any, *, served_model: str
) -> VlmStructuredResponse:
    """Validate the complete hosted judge output without repairing its verdict."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise VlmEvalError("Hosted VLM response JSON contains duplicate keys")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise VlmEvalError("Hosted VLM response JSON contains a non-finite number")

    if not isinstance(text, str):
        raise VlmEvalError("Hosted VLM response content must be a JSON string")
    deframed_text, deframed = _deframe_json_text(text)
    try:
        payload = json.loads(
            deframed_text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise VlmEvalError(
            "Hosted VLM response JSON could not be parsed in full"
        ) from exc
    if not isinstance(payload, dict):
        raise VlmEvalError("Hosted VLM response JSON must be an object")
    if not isinstance(payload.get("success"), bool):
        raise VlmEvalError("Hosted VLM response success must be a boolean")
    score = payload.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not 0 <= score <= 1
        or not math.isfinite(score)
    ):
        raise VlmEvalError(
            "Hosted VLM response score must be a finite number in [0, 1]"
        )
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise VlmEvalError("Hosted VLM response rationale must be a nonempty string")
    return VlmStructuredResponse(
        success=payload["success"],
        score=float(score),
        rationale=rationale,
        served_model=served_model,
        parser_version=_parser_version(HOSTED_RESPONSE_PARSER_VERSION, deframed),
        provider_success=payload["success"],
    )


def result_uri_for(output_path: str) -> str:
    """Return the JSON artifact URI for an output path."""

    if output_path.endswith(".json"):
        return output_path
    return output_path.rstrip("/") + f"/{RESULT_FILENAME}"


def judge_comparison_result_uri_for(output_path: str) -> str:
    """Return the distinct paired-judge artifact URI for an output path.

    Args:
        output_path: Output directory, S3 prefix, or canonical JSON path.

    Returns:
        The canonical ``vlm_judge_disagreement.json`` destination.

    Raises:
        VlmEvalError: If an explicit JSON path uses another filename.
    """

    if output_path.endswith(".json"):
        if output_path.rstrip("/").rsplit("/", 1)[-1] != (
            JUDGE_COMPARISON_RESULT_FILENAME
        ):
            raise VlmEvalError(
                "--output-path JSON filename must be "
                f"{JUDGE_COMPARISON_RESULT_FILENAME}"
            )
        return output_path
    return output_path.rstrip("/") + f"/{JUDGE_COMPARISON_RESULT_FILENAME}"


def loop_report_uri_for(output_path: str) -> str:
    """Return the aggregate task-success report URI for an output prefix."""

    if output_path.endswith(".json"):
        return output_path
    return output_path.rstrip("/") + f"/{LOOP_REPORT_FILENAME}"


def discover_rollouts(input_path: str) -> list[str]:
    """Return one URI per rollout under ``input_path``, or the prefix itself.

    Mirrors the retired ``sim-to-real-loop.yaml``: it listed the immediate child
    directories of the rollout prefix and fell back to treating the prefix as a single
    rollout when there were none. ``evaluate_vlm`` scores *one* rollout — it discovers
    frames recursively, so pointing it at a prefix of many rollouts would blend them into
    one score. That is why the set has to be enumerated here.
    """

    if not input_path.strip():
        raise VlmEvalError("--input-path is required")
    if input_path.startswith("s3://"):
        return _discover_object_rollouts(input_path)

    root = Path(input_path)
    if not root.exists():
        raise VlmEvalError(f"rollout input not found: {input_path}")
    children = sorted(child for child in root.iterdir() if child.is_dir())
    return [str(child) for child in children] or [str(root)]


def _discover_object_rollouts(input_path: str) -> list[str]:
    from npa.clients.storage import StorageClient

    base = input_path.rstrip("/") + "/"
    parsed = urlparse(base)
    bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
    client = StorageClient.from_environment()
    # A "directory" in object storage is a common prefix; the delimiter listing is the
    # object-store equivalent of `find -mindepth 1 -maxdepth 1 -type d`.
    paginator = client.s3.get_paginator("list_objects_v2")
    names: list[str] = []
    saw_object = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for common in page.get("CommonPrefixes") or ():
            child = str(common.get("Prefix") or "")
            if child and child != prefix:
                names.append(f"s3://{bucket}/{child}")
        saw_object = saw_object or bool(page.get("Contents"))
    if names:
        return sorted(names)
    if not saw_object:
        raise VlmEvalError(f"rollout input contains no objects: {input_path}")
    return [base]


@dataclass(frozen=True)
class VlmLoopRollout:
    """One rollout's contribution to the aggregate report."""

    rollout_id: str
    success: bool
    score: float
    rationale: str
    status: str
    frame_count: int
    result_uri: str


def evaluate_rollout_set(
    *,
    input_path: str,
    output_path: str,
    task: str = "sim-to-real",
    backend: str = DEFAULT_BACKEND,
    model: str = DEFAULT_MODEL,
    success_threshold: float = 0.8,
    frame_selection: str = DEFAULT_FRAME_SELECTION,
    max_frames: int = DEFAULT_MAX_FRAMES,
    endpoint_url: str = "",
    api_key_env: str = DEFAULT_API_KEY_ENV,
    rubric: str = DEFAULT_RUBRIC,
    rubric_path: str = "",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    storage_client: "StorageClient | None" = None,
) -> dict[str, Any]:
    """Score every rollout under a prefix and return the aggregate task-success report.

    This is the capability the retired ``sim-to-real-loop.yaml`` implemented in ~80 lines of
    bash and `jq`, and that ``tests/workbench/test_vlm_eval_loop_e2e.py`` re-implemented in
    Python: score each rollout, write one result per rollout, then aggregate into a coarse
    ``task_success`` gate. Field names and the gate rule (``mean_score >=
    success_threshold``) are kept identical so existing readers of the report keep working.
    """

    started_at = time.monotonic()
    rollouts: list[VlmLoopRollout] = []
    for rollout_uri in discover_rollouts(input_path):
        rollout_id = _rollout_id_for(rollout_uri)
        result = evaluate_vlm(
            input_path=rollout_uri,
            output_path=_join_uri(
                output_path.rstrip("/") + "/", f"rollouts/{rollout_id}/"
            ),
            task=task,
            backend=backend,
            model=model,
            success_threshold=success_threshold,
            frame_selection=frame_selection,
            max_frames=max_frames,
            endpoint_url=endpoint_url,
            api_key_env=api_key_env,
            rubric=rubric,
            rubric_path=rubric_path,
            timeout_s=timeout_s,
        )
        written = write_result(
            asdict(result), result_uri=result.result_uri, storage_client=storage_client
        )
        rollouts.append(
            VlmLoopRollout(
                rollout_id=rollout_id,
                success=bool(result.passed),
                score=float(result.score),
                rationale=result.rationale,
                status=result.status,
                frame_count=result.frame_count,
                result_uri=written,
            )
        )

    report = aggregate_loop_report(
        rollouts,
        model=model,
        frame_selection=_normalize_frame_selection(frame_selection),
        success_threshold=success_threshold,
        output_dir=output_path,
    )
    report["latency_s"] = round(time.monotonic() - started_at, 3)
    report["report_uri"] = write_result(
        report,
        result_uri=loop_report_uri_for(output_path),
        storage_client=storage_client,
    )
    return report


def aggregate_loop_report(
    rollouts: Sequence[VlmLoopRollout],
    *,
    model: str,
    frame_selection: str,
    success_threshold: float,
    output_dir: str,
) -> dict[str, Any]:
    """Aggregate per-rollout results exactly as the retired template's `jq -s` did."""

    total = len(rollouts)
    passed = sum(1 for rollout in rollouts if rollout.success)
    mean_score = (sum(rollout.score for rollout in rollouts) / total) if total else 0.0
    return {
        "status": "completed",
        "model": model,
        "frame_selection": frame_selection,
        "success_threshold": success_threshold,
        "output_dir": output_dir,
        "total_rollouts": total,
        "passed_rollouts": passed,
        "success_rate": (passed / total) if total else 0.0,
        "mean_score": mean_score,
        # The coarse gate is the MEAN score, not the pass rate — same as the template.
        "task_success": mean_score >= success_threshold,
        "rollouts": [asdict(rollout) for rollout in rollouts],
    }


def _rollout_id_for(rollout_uri: str) -> str:
    """Return the last path segment of a rollout URI (``basename`` for object stores)."""

    trimmed = rollout_uri.rstrip("/")
    segment = trimmed.rsplit("/", 1)[-1] if "/" in trimmed else trimmed
    return segment or "rollout"


def write_result(
    payload: dict[str, Any],
    *,
    result_uri: str,
    storage_client: "StorageClient | None" = None,
) -> str:
    """Write a VLM eval result to local disk or S3."""

    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if result_uri.startswith("s3://"):
        from npa.clients.storage import StorageClient

        client = storage_client or StorageClient.from_environment()
        with tempfile.TemporaryDirectory(prefix="npa-vlm-eval-") as tmp:
            local_path = Path(tmp) / RESULT_FILENAME
            local_path.write_text(body, encoding="utf-8")
            return client.upload_file(str(local_path), result_uri)

    path = Path(result_uri)
    if path.suffix != ".json":
        path = path / RESULT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


def benchmark_result_uri_for(output_path: str) -> str:
    """Return the JSON artifact URI for a benchmark report path."""

    if output_path.endswith(".json"):
        return output_path
    return output_path.rstrip("/") + f"/{BENCHMARK_RESULT_FILENAME}"


def write_benchmark_report(
    payload: dict[str, Any],
    *,
    output_path: str,
    storage_client: "StorageClient | None" = None,
) -> str:
    """Write a benchmark report to local disk or S3."""

    result_uri = benchmark_result_uri_for(output_path)
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if result_uri.startswith("s3://"):
        from npa.clients.storage import StorageClient

        client = storage_client or StorageClient.from_environment()
        with tempfile.TemporaryDirectory(prefix="npa-vlm-eval-benchmark-") as tmp:
            local_path = Path(tmp) / BENCHMARK_RESULT_FILENAME
            local_path.write_text(body, encoding="utf-8")
            return client.upload_file(str(local_path), result_uri)

    path = Path(result_uri)
    if path.suffix != ".json":
        path = path / BENCHMARK_RESULT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


def _result_from_structured(
    *,
    backend: str,
    input_path: str,
    output_path: str,
    task: str,
    model: str,
    success_threshold: float,
    frame_selection: str,
    frame_count: int,
    rubric: str,
    structured: VlmStructuredResponse,
) -> VlmEvalResult:
    score = round(_clamp_score(structured.score), 4)
    passed = score >= success_threshold
    provider_success = (
        structured.provider_success if structured.evidence is not None else None
    )
    provider_success_matches_score_gate = (
        provider_success == passed if provider_success is not None else None
    )
    return VlmEvalResult(
        status="passed" if passed else "needs_iteration",
        backend=backend,
        input_path=input_path,
        output_path=output_path,
        result_uri=result_uri_for(output_path),
        task=task,
        model=model,
        score=score,
        success_threshold=success_threshold,
        passed=passed,
        generated_at=datetime.now(timezone.utc).isoformat(),
        frame_selection=frame_selection,
        frame_count=frame_count,
        rationale=structured.rationale,
        rubric=rubric,
        served_model=structured.served_model,
        provider_success=provider_success,
        provider_success_matches_score_gate=provider_success_matches_score_gate,
        evidence=structured.evidence,
    )


def _run_benchmark_case(
    item: VlmBenchmarkItem,
    *,
    config: VlmBenchmarkConfig,
    endpoint_url: str,
    api_key_env: str,
    timeout_s: float,
    use_fixture_score: bool,
) -> VlmBenchmarkCaseResult:
    score = (
        item.fixture_score
        if use_fixture_score and item.fixture_score is not None
        else None
    )
    try:
        result = evaluate_vlm(
            input_path=item.rollout,
            output_path=f"vlm-eval-benchmark://{item.id}",
            task=item.task,
            backend=config.backend,
            model=config.model,
            success_threshold=config.success_threshold,
            frame_selection=config.frame_selection,
            max_frames=config.max_frames,
            endpoint_url=endpoint_url,
            api_key_env=api_key_env,
            rubric=config.rubric,
            timeout_s=timeout_s,
            score=score,
        )
    except VlmEvalError as exc:
        raise VlmEvalError(
            "benchmark item "
            f"{item.id!r} failed for model={config.model!r}, "
            f"rubric={config.rubric_name!r}, threshold={config.success_threshold}: {exc}"
        ) from exc

    return VlmBenchmarkCaseResult(
        item_id=item.id,
        rollout=item.rollout,
        expected_label=item.expected_label,
        predicted_label=result.passed,
        score=result.score,
        status=result.status,
        passed=result.passed,
        task=result.task,
        rationale=result.rationale,
        frame_count=result.frame_count,
        score_source="fixture" if score is not None else result.backend,
        provider_success=result.provider_success,
        provider_success_matches_score_gate=result.provider_success_matches_score_gate,
        evidence=result.evidence,
    )


def _benchmark_metrics(
    results: Sequence[VlmBenchmarkCaseResult],
) -> VlmBenchmarkMetrics:
    total = len(results)
    if total == 0:
        raise VlmEvalError("benchmark dataset must include at least one item")
    tp = sum(
        1 for result in results if result.expected_label and result.predicted_label
    )
    tn = sum(
        1
        for result in results
        if not result.expected_label and not result.predicted_label
    )
    fp = sum(
        1 for result in results if not result.expected_label and result.predicted_label
    )
    fn = sum(
        1 for result in results if result.expected_label and not result.predicted_label
    )
    correct = tp + tn
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = round((2 * precision * recall) / (precision + recall), 4)
    accuracy = round(correct / total, 4)
    return VlmBenchmarkMetrics(
        total=total,
        correct=correct,
        agreement=accuracy,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
    )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _benchmark_rank_key(result: VlmBenchmarkConfigResult) -> tuple[Any, ...]:
    metrics = result.metrics
    precision = -1.0 if metrics.precision is None else metrics.precision
    recall = -1.0 if metrics.recall is None else metrics.recall
    f1 = -1.0 if metrics.f1 is None else metrics.f1
    return (
        -metrics.accuracy,
        -f1,
        -precision,
        -recall,
        metrics.false_positives,
        metrics.false_negatives,
        result.config.success_threshold,
        result.config.model,
        result.config.rubric_name,
    )


@contextmanager
def _materialized_benchmark_manifest(dataset: str) -> Iterator[Path]:
    if dataset.startswith("s3://"):
        from npa.clients.storage import StorageClient

        with tempfile.TemporaryDirectory(
            prefix="npa-vlm-eval-benchmark-dataset-"
        ) as tmp:
            local = Path(StorageClient.from_environment().download_path(dataset, tmp))
            yield _find_benchmark_manifest(local)
        return

    yield _find_benchmark_manifest(Path(dataset))


def _find_benchmark_manifest(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        for name in ("benchmark.json", "dataset.json", "manifest.json"):
            candidate = path / name
            if candidate.is_file():
                return candidate
    raise VlmEvalError(
        f"benchmark dataset not found: {path}. Expected a JSON file or a directory "
        "containing benchmark.json, dataset.json, or manifest.json."
    )


def _parse_benchmark_item(
    raw_item: Any,
    *,
    index: int,
    rollout_base: str,
    default_task: str,
) -> VlmBenchmarkItem:
    if not isinstance(raw_item, dict):
        raise VlmEvalError(f"benchmark item {index} must be an object")
    rollout = (
        raw_item.get("rollout")
        or raw_item.get("rollout_path")
        or raw_item.get("input_path")
    )
    if not rollout:
        raise VlmEvalError(f"benchmark item {index} must include rollout or input_path")
    if "expected_label" in raw_item:
        raw_label = raw_item["expected_label"]
    elif "label" in raw_item:
        raw_label = raw_item["label"]
    else:
        raise VlmEvalError(f"benchmark item {index} must include expected_label")

    fixture_score = None
    if raw_item.get("fixture_score") is not None:
        fixture_score = _clamp_score(raw_item["fixture_score"])
        _validate_score_override(fixture_score)

    item_id = str(
        raw_item.get("id") or raw_item.get("name") or f"item-{index:03d}"
    ).strip()
    if not item_id:
        item_id = f"item-{index:03d}"

    return VlmBenchmarkItem(
        id=item_id,
        rollout=_resolve_relative_path(str(rollout), rollout_base),
        expected_label=_coerce_expected_label(raw_label),
        task=str(raw_item.get("task") or raw_item.get("instruction") or default_task),
        fixture_score=fixture_score,
    )


def _coerce_expected_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in {0, 1}:
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower().replace("_", "-")
        if normalized in {"1", "true", "pass", "passed", "success", "positive"}:
            return True
        if normalized in {"0", "false", "fail", "failed", "failure", "negative"}:
            return False
    raise VlmEvalError(
        "expected_label must be a boolean or one of pass/fail, success/failure, true/false"
    )


def _coerce_rubric_map(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise VlmEvalError("benchmark dataset rubrics must be an object")
    rubrics = {str(key).strip(): str(text).strip() for key, text in value.items()}
    return {key: text for key, text in rubrics.items() if key and text}


def _resolve_benchmark_rubrics(
    rubrics: Sequence[str],
    *,
    dataset_rubrics: dict[str, str],
    dataset_path: str,
) -> list[tuple[str, str]]:
    names = _normalize_strings(rubrics, label="rubrics")
    resolved: list[tuple[str, str]] = []
    for raw_name in names:
        if raw_name in dataset_rubrics:
            resolved.append((raw_name, dataset_rubrics[raw_name]))
            continue
        if raw_name == "default":
            resolved.append(("default", DEFAULT_RUBRIC))
            continue
        rubric_from_path = _rubric_from_path(raw_name, dataset_path=dataset_path)
        if rubric_from_path is not None:
            resolved.append(rubric_from_path)
            continue
        resolved.append((_slugify_rubric_name(raw_name), raw_name))
    return resolved


def _rubric_from_path(raw_name: str, *, dataset_path: str) -> tuple[str, str] | None:
    candidate = raw_name[1:] if raw_name.startswith("@") else raw_name
    paths = [Path(candidate)]
    if not _is_uri(dataset_path):
        dataset_file = Path(dataset_path)
        dataset_base = dataset_file.parent if dataset_file.suffix else dataset_file
        paths.insert(0, dataset_base / candidate)
    for path in paths:
        if path.is_file():
            return (path.stem, path.read_text(encoding="utf-8").strip())
    if raw_name.startswith("@"):
        raise VlmEvalError(f"rubric file does not exist: {candidate}")
    return None


def _normalize_thresholds(thresholds: Sequence[float]) -> list[float]:
    values: list[float] = []
    for raw_threshold in thresholds:
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError) as exc:
            raise VlmEvalError(f"invalid threshold: {raw_threshold}") from exc
        if not 0.0 <= threshold <= 1.0:
            raise VlmEvalError("--thresholds values must be between 0 and 1")
        if threshold not in values:
            values.append(threshold)
    if not values:
        raise VlmEvalError("--thresholds must include at least one value")
    return values


def _normalize_strings(values: Sequence[str], *, label: str) -> list[str]:
    normalized = [str(value).strip() for value in values if str(value).strip()]
    if not normalized:
        raise VlmEvalError(f"--{label} must include at least one value")
    return normalized


def _dataset_source_base(dataset: str) -> str:
    if dataset.startswith("s3://"):
        clean = dataset.rstrip("/")
        if clean.endswith(".json"):
            return clean.rsplit("/", 1)[0] + "/"
        return clean + "/"
    path = Path(dataset)
    if path.is_file() or path.suffix:
        return str(path.parent)
    return str(path)


def _resolve_rollout_base(
    rollout_base_path: str,
    *,
    source_base: str,
    local_base: Path,
) -> str:
    if not rollout_base_path:
        return source_base
    if _is_uri(rollout_base_path) or Path(rollout_base_path).is_absolute():
        return rollout_base_path
    if source_base.startswith("s3://"):
        return _join_uri(source_base, rollout_base_path)
    return str(local_base / rollout_base_path)


def _resolve_relative_path(value: str, base: str) -> str:
    if _is_uri(value) or Path(value).is_absolute():
        return value
    if base.startswith("s3://"):
        return _join_uri(base, value)
    return str(Path(base) / value)


def _join_uri(base: str, *parts: str) -> str:
    prefix = base.rstrip("/")
    path = posixpath.join(*(part.strip("/") for part in parts if part))
    return f"{prefix}/{path}" if path else prefix


def _is_uri(value: str) -> bool:
    return re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value) is not None


def _slugify_rubric_name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:40].strip("-") or "inline-rubric"


def _validate_common(
    *,
    input_path: str,
    output_path: str,
    success_threshold: float,
    frame_selection: str,
    max_frames: int,
    timeout_s: float,
) -> None:
    if not input_path:
        raise VlmEvalError("input_path is required")
    if not output_path:
        raise VlmEvalError("output_path is required")
    if not 0.0 <= success_threshold <= 1.0:
        raise VlmEvalError("--success-threshold must be between 0 and 1")
    _normalize_frame_selection(frame_selection)
    if max_frames <= 0:
        raise VlmEvalError("--max-frames must be positive")
    if timeout_s <= 0:
        raise VlmEvalError("--timeout-s must be positive")


def _validate_score_override(score: float) -> None:
    if not 0.0 <= float(score) <= 1.0:
        raise VlmEvalError("--score must be between 0 and 1")


def _normalize_backend(backend: str) -> str:
    value = (backend or DEFAULT_BACKEND).strip().lower()
    if value not in SUPPORTED_BACKENDS:
        allowed = ", ".join(SUPPORTED_BACKENDS)
        raise VlmEvalError(f"--backend must be one of: {allowed}")
    return value


def _normalize_frame_selection(frame_selection: str) -> str:
    value = (frame_selection or DEFAULT_FRAME_SELECTION).strip().lower()
    if value not in SUPPORTED_FRAME_SELECTIONS:
        allowed = ", ".join(SUPPORTED_FRAME_SELECTIONS)
        raise VlmEvalError(f"--frame-selection must be one of: {allowed}")
    return value


def _load_rubric(*, rubric: str, rubric_path: str) -> str:
    if rubric_path:
        path = Path(rubric_path)
        if not path.exists():
            raise VlmEvalError(f"--rubric-path does not exist: {path}")
        return path.read_text(encoding="utf-8").strip()
    return (rubric or DEFAULT_RUBRIC).strip()


@contextmanager
def _materialized_input(input_path: str) -> Iterator[Path]:
    if not input_path.startswith("s3://"):
        yield Path(input_path)
        return

    from npa.clients.storage import StorageClient

    with tempfile.TemporaryDirectory(prefix="npa-vlm-eval-input-") as tmp:
        local = StorageClient.from_environment().download_path(input_path, tmp)
        yield Path(local)


def _resolve_task_text(local_input: Path, task: str) -> str:
    if task and task != "sim-to-real":
        return task

    for candidate in (
        local_input / "meta" / "tasks.parquet",
        local_input.parent / "meta" / "tasks.parquet",
    ):
        if not candidate.exists():
            continue
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(candidate)
            if "task" in table.column_names and table.num_rows:
                value = table.column("task")[0].as_py()
                if value:
                    return str(value)
        except Exception:
            continue

    for candidate in (
        local_input / "meta" / "info.json",
        local_input / "info.json",
        local_input / "manifest.json",
    ):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("task", "instruction", "description"):
            value = payload.get(key)
            if value:
                return str(value)
    return task


def _build_prompt(
    *,
    task: str,
    rubric: str,
    frame_selection: str,
    frame_count: int,
) -> str:
    return "\n".join(
        [
            "You are scoring a robot rollout from visual evidence.",
            f"Task/instruction: {task}",
            f"Rubric: {rubric}",
            f"Frame selection: {frame_selection}; frames supplied: {frame_count}.",
            "Return only a JSON object with this schema:",
            '{"success": boolean, "score": number between 0 and 1, "rationale": string}',
            "The score is the only downstream contract; make it repeatable and calibrated.",
        ]
    )


def _ready_timeout_s() -> float:
    raw = os.environ.get(READY_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_READY_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_READY_TIMEOUT_S
    return value if value > 0 else DEFAULT_READY_TIMEOUT_S


def _call_openai_compatible(
    *,
    backend: str,
    model: str,
    endpoint_url: str,
    api_key_env: str,
    prompt: str,
    rubric: str = DEFAULT_RUBRIC,
    frames: list[SelectedFrame],
    timeout_s: float,
    frame_selection: str = DEFAULT_FRAME_SELECTION,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> VlmStructuredResponse:
    url = _chat_completions_url(
        _resolve_endpoint_url(backend=backend, endpoint_url=endpoint_url)
    )
    request = _build_openai_request(
        backend=backend, model=model, prompt=prompt, frames=frames
    )
    headers = {"Content-Type": "application/json"}
    api_key = _resolve_api_key(backend=backend, api_key_env=api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request_evidence = _build_request_evidence(
        backend=backend,
        model=model,
        prompt=prompt,
        rubric=rubric,
        request=request,
        frames=frames,
        frame_selection=frame_selection,
        max_frames=max_frames,
    )
    started_at = time.monotonic()
    raw_response = _post_with_readiness_retry(
        url=url,
        headers=headers,
        request=request,
        backend=backend,
        timeout_s=timeout_s,
    )
    response = _coerce_backend_response(
        raw_response, fallback_latency_s=time.monotonic() - started_at
    )
    choice, message = _response_choice_and_content(response.data)
    result = _parse_backend_verdict(
        backend=backend,
        requested_model=model,
        data=response.data,
        choice=choice,
        message=message,
    )
    evidence = _build_evaluation_evidence(
        request_evidence,
        response,
        choice,
        parser_version=result.parser_version,
    )
    return replace(result, evidence=evidence)


def _parse_backend_verdict(
    *,
    backend: str,
    requested_model: str,
    data: dict[str, Any],
    choice: dict[str, Any],
    message: Any,
) -> VlmStructuredResponse:
    served_model = data.get("model")
    if backend != "api":
        result = parse_structured_response(str(message))
        if served_model is None:
            return result
        if not isinstance(served_model, str) or not served_model.strip():
            raise VlmEvalError(
                "Self-hosted VLM response model must be a nonempty string"
            )
        return replace(result, served_model=served_model)
    if choice.get("finish_reason") != "stop":
        raise VlmEvalError(
            "Hosted VLM response did not complete with finish_reason=stop"
        )
    if not isinstance(served_model, str) or not served_model.strip():
        raise VlmEvalError("Hosted VLM response must identify the served model")
    if requested_model in CANONICAL_HOSTED_MODELS and served_model != requested_model:
        raise VlmEvalError(
            "Hosted VLM response model does not match the requested model"
        )
    return _parse_api_structured_response(message, served_model=served_model)


def _build_openai_request(
    *, backend: str, model: str, prompt: str, frames: Sequence[SelectedFrame]
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in frames:
        encoded = base64.b64encode(frame.data).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{frame.media_type};base64,{encoded}"},
            }
        )
    request: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}],
    }
    if backend != "api":
        return request
    from npa.clients.token_factory import default_chat_extra

    request.update(default_chat_extra(model))
    if model == "MiniMaxAI/MiniMax-M3":
        request.pop("response_format")
    return request


def _build_request_evidence(
    *,
    backend: str,
    model: str,
    prompt: str,
    rubric: str,
    request: dict[str, Any],
    frames: Sequence[SelectedFrame],
    frame_selection: str,
    max_frames: int,
) -> VlmRequestEvidence:
    frame_evidence = tuple(_frame_evidence(frame) for frame in frames)
    prompt_sha256 = _sha256_text(prompt)
    rubric_sha256 = _sha256_text(rubric)
    manifest = _request_manifest(
        backend=backend,
        model=model,
        request=request,
        frames=frames,
        frame_evidence=frame_evidence,
        frame_selection=frame_selection,
        max_frames=max_frames,
        prompt_sha256=prompt_sha256,
        rubric_sha256=rubric_sha256,
    )
    return VlmRequestEvidence(
        requested_at=datetime.now(timezone.utc).isoformat(),
        endpoint_role=manifest["endpoint_role"],
        prompt_sha256=prompt_sha256,
        rubric_sha256=rubric_sha256,
        request_manifest_sha256=_sha256_json(manifest),
        request_manifest=manifest,
        frames=frame_evidence,
    )


def _request_manifest(
    *,
    backend: str,
    model: str,
    request: dict[str, Any],
    frames: Sequence[SelectedFrame],
    frame_evidence: Sequence[VlmFrameEvidence],
    frame_selection: str,
    max_frames: int,
    prompt_sha256: str,
    rubric_sha256: str,
) -> dict[str, Any]:
    generation_parameters = {
        key: request[key]
        for key in ("temperature", "response_format", "chat_template_kwargs")
        if key in request
    }
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "endpoint_role": "hosted-api" if backend == "api" else "self-hosted",
        "requested_model": model,
        "generation_parameters": generation_parameters,
        "prompt_sha256": prompt_sha256,
        "rubric_sha256": rubric_sha256,
        "frames": [asdict(frame) for frame in frame_evidence],
        "sampling": _sampling_manifest(
            frames,
            frame_selection=frame_selection,
            max_frames=max_frames,
        ),
    }


def _sampling_manifest(
    frames: Sequence[SelectedFrame],
    *,
    frame_selection: str,
    max_frames: int,
) -> dict[str, Any]:
    source_kind, source_count = _uniform_source_metadata(frames)
    indices = [frame.source_index for frame in frames]
    timestamps = [frame.source_timestamp_s for frame in frames]
    timestamps_complete = (
        all(timestamp is not None for timestamp in timestamps)
        if source_kind == "video"
        else None
    )
    return {
        "strategy": _normalize_frame_selection(frame_selection),
        "max_frames": max_frames,
        "selected_count": len(frames),
        "source_kind": source_kind,
        "source_count": source_count,
        "selected_indices": indices,
        "selected_timestamps_s": timestamps,
        "coverage_complete": _source_indices_complete(frames, source_count),
        "timestamps_complete": timestamps_complete,
    }


def _uniform_source_metadata(
    frames: Sequence[SelectedFrame],
) -> tuple[str | None, int | None]:
    if not frames:
        return None, None
    source_kind = frames[0].source_kind
    if not source_kind or any(frame.source_kind != source_kind for frame in frames):
        source_kind = None
    source_count = frames[0].source_count
    if source_count is None or any(
        frame.source_count != source_count for frame in frames
    ):
        source_count = None
    return source_kind, source_count


def _source_indices_complete(
    frames: Sequence[SelectedFrame], source_count: int | None
) -> bool:
    if not frames or source_count is None:
        return False
    return all(
        frame.source_index is not None and 0 <= frame.source_index < source_count
        for frame in frames
    )


def _response_choice_and_content(data: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise VlmEvalError("VLM backend response missing choices[0]")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise VlmEvalError("VLM backend response missing choices[0].message.content")
    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        raise VlmEvalError("VLM backend refused the evaluation request")
    return choice, message["content"]


def _build_evaluation_evidence(
    request: VlmRequestEvidence,
    response: _VlmBackendResponse,
    choice: dict[str, Any],
    *,
    parser_version: str,
) -> VlmEvaluationEvidence:
    provider_id, returned_model, finish_reason, usage = _provider_metadata(
        response, choice
    )
    provider = VlmProviderEvidence(
        provider_request_id=provider_id,
        returned_model=returned_model,
        finish_reason=finish_reason,
        latency_s=round(response.latency_s, 6),
        status_code=response.status_code,
        usage=usage,
        raw_response=response.raw_body,
        raw_response_sha256=_sha256_text(response.raw_body),
        parser_version=parser_version,
    )
    return VlmEvaluationEvidence(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        request=request,
        provider=provider,
    )


def _provider_metadata(
    response: _VlmBackendResponse,
    choice: dict[str, Any],
) -> tuple[str | None, str | None, str | None, dict[str, Any] | None]:
    usage = response.data.get("usage")
    if usage is not None and not isinstance(usage, dict):
        raise VlmEvalError("VLM backend response usage must be an object when present")
    provider_id = response.data.get("id") or response.request_id_header
    if provider_id is not None and not isinstance(provider_id, str):
        raise VlmEvalError(
            "VLM backend response request id must be a string when present"
        )
    returned_model = response.data.get("model")
    if returned_model is not None and not isinstance(returned_model, str):
        raise VlmEvalError("VLM backend response model must be a string when present")
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise VlmEvalError(
            "VLM backend response finish_reason must be a string when present"
        )
    return provider_id or None, returned_model or None, finish_reason, usage


def _frame_evidence(frame: SelectedFrame) -> VlmFrameEvidence:
    with Image.open(BytesIO(frame.data)) as image:
        width, height = image.size
    return VlmFrameEvidence(
        label=frame.label,
        media_type=frame.media_type,
        sha256=hashlib.sha256(frame.data).hexdigest(),
        byte_count=len(frame.data),
        width=width,
        height=height,
        source_kind=frame.source_kind,
        source_index=frame.source_index,
        source_count=frame.source_count,
        source_timestamp_s=frame.source_timestamp_s,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _post_with_readiness_retry(
    *,
    url: str,
    headers: dict[str, str],
    request: dict[str, Any],
    backend: str,
    timeout_s: float,
    error_response_sink: Callable[[_VlmBackendResponse], None] | None = None,
) -> _VlmBackendResponse:
    """POST to an OpenAI-compatible endpoint, tolerating self-hosted warmup.

    A self-hosted vLLM server started alongside the eval job needs minutes to
    load weights; retry transient connection failures with backoff up to the
    readiness deadline so a cold start is a bounded wait, not an instant
    connection-refused. Hosted (``api``) backends are expected to be up and fail
    fast. This lives in the request path so callers that stub
    ``_call_openai_compatible`` in tests never incur the wait.
    """

    is_self_hosted = backend == "self-hosted"
    ready_timeout = _ready_timeout_s()
    deadline = time.monotonic() + (ready_timeout if is_self_hosted else 0.0)
    started_at = time.monotonic()
    delay = 2.0
    last_conn_error = ""
    while True:
        try:
            with httpx.Client(timeout=timeout_s) as client:
                response = client.post(url, headers=headers, json=request)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    captured = _captured_http_response(response, started_at=started_at)
                    _retain_error_response(captured, error_response_sink)
                    detail = captured.raw_body.strip().replace("\n", " ")[:1000]
                    suffix = f" response={detail}" if detail else ""
                    raise VlmEvalError(
                        f"VLM backend request failed: {exc}{suffix}"
                    ) from exc
                try:
                    data = response.json()
                except ValueError as exc:
                    captured = _captured_http_response(response, started_at=started_at)
                    _retain_error_response(captured, error_response_sink)
                    raise VlmEvalError(
                        "VLM backend returned non-JSON response"
                    ) from exc
                if not isinstance(data, dict):
                    captured = _captured_http_response(response, started_at=started_at)
                    _retain_error_response(captured, error_response_sink)
                    raise VlmEvalError(
                        "VLM backend returned a non-object JSON response"
                    )
                return _backend_response_from_http(
                    response,
                    data=data,
                    started_at=started_at,
                )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            last_conn_error = str(exc) or exc.__class__.__name__
            if is_self_hosted and time.monotonic() < deadline:
                time.sleep(delay)
                delay = min(delay * 1.5, 15.0)
                continue
            if is_self_hosted:
                raise VlmEvalError(
                    f"VLM backend not ready at {url} after {ready_timeout:.0f}s "
                    f"(last: {last_conn_error})"
                ) from exc
            raise VlmEvalError(f"VLM backend request failed: {exc}") from exc
        except httpx.HTTPError as exc:
            raise VlmEvalError(f"VLM backend request failed: {exc}") from exc


def _captured_http_response(
    response: Any,
    *,
    started_at: float,
) -> _VlmBackendResponse:
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    data = payload if isinstance(payload, dict) else {}
    return _backend_response_from_http(response, data=data, started_at=started_at)


def _backend_response_from_http(
    response: Any,
    *,
    data: dict[str, Any],
    started_at: float,
) -> _VlmBackendResponse:
    raw_body = getattr(response, "text", "") or _canonical_json(data)
    response_headers = getattr(response, "headers", {})
    return _VlmBackendResponse(
        data=data,
        raw_body=raw_body,
        status_code=getattr(response, "status_code", None),
        request_id_header=_request_id_from_headers(response_headers),
        latency_s=time.monotonic() - started_at,
    )


def _retain_error_response(
    response: _VlmBackendResponse,
    sink: Callable[[_VlmBackendResponse], None] | None,
) -> None:
    if sink is not None:
        sink(response)


def _coerce_backend_response(
    response: _VlmBackendResponse | dict[str, Any],
    *,
    fallback_latency_s: float,
) -> _VlmBackendResponse:
    if isinstance(response, _VlmBackendResponse):
        return response
    if not isinstance(response, dict):
        raise VlmEvalError("VLM backend returned a non-object JSON response")
    return _VlmBackendResponse(
        data=response,
        raw_body=_canonical_json(response),
        status_code=None,
        request_id_header=None,
        latency_s=fallback_latency_s,
    )


def _request_id_from_headers(headers: Any) -> str | None:
    if not hasattr(headers, "get"):
        return None
    for name in ("x-request-id", "x-nebius-request-id", "request-id"):
        value = headers.get(name)
        if value:
            return str(value)
    return None


def _resolve_endpoint_url(*, backend: str, endpoint_url: str) -> str:
    if endpoint_url:
        return endpoint_url
    if backend == "api":
        from npa.clients.token_factory import (
            BASE_URL_ENV_KEYS,
            DEFAULT_BASE_URL as TOKEN_FACTORY_BASE_URL,
        )

        token_factory_base = next(
            (os.environ[key] for key in BASE_URL_ENV_KEYS if os.environ.get(key)),
            "",
        )
        return (
            os.environ.get("VLM_EVAL_API_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or token_factory_base
            or TOKEN_FACTORY_BASE_URL
        )
    return (
        os.environ.get("VLM_EVAL_ENDPOINT_URL")
        or os.environ.get("VLM_EVAL_BASE_URL")
        or DEFAULT_ENDPOINT_URL
    )


def _resolve_api_key(*, backend: str, api_key_env: str) -> str:
    key = os.environ.get(api_key_env or DEFAULT_API_KEY_ENV, "")
    if backend == "api":
        key = (
            key
            or os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        if not key:
            from npa.clients.token_factory import resolve_config

            try:
                key = resolve_config(
                    api_key_env=api_key_env or DEFAULT_API_KEY_ENV,
                    require_api_key=False,
                ).api_key
            except ValueError as exc:
                raise VlmEvalError(
                    f"Unable to load Token Factory credentials: {exc}"
                ) from exc
        if not key:
            raise VlmEvalError(
                f"--backend api requires an API key in {api_key_env or DEFAULT_API_KEY_ENV}, "
                "NEBIUS_TOKEN_FACTORY_KEY, OPENAI_API_KEY, or ~/.npa/credentials.yaml"
            )
    return key


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _frames_from_images(
    path: Path,
    *,
    frame_selection: str,
    max_frames: int,
) -> list[SelectedFrame]:
    image_paths = _discover_image_paths(path)
    if not image_paths:
        return []
    source_count = len(image_paths)
    indices = _selected_indices(
        source_count, frame_selection=frame_selection, max_frames=max_frames
    )
    return [
        SelectedFrame(
            label=_image_frame_label(path, image_paths[index]),
            media_type="image/png",
            data=_image_file_to_png(image_paths[index]),
            source_kind="image-sequence",
            source_index=index,
            source_count=source_count,
        )
        for index in indices
    ]


def _image_frame_label(root: Path, frame_path: Path) -> str:
    if root.is_dir():
        return frame_path.relative_to(root).as_posix()
    return frame_path.name


def _discover_image_paths(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_SUFFIXES
        and not any(part.startswith(".") for part in file.relative_to(path).parts)
    )


def _frames_from_numpy(
    path: Path,
    *,
    frame_selection: str,
    max_frames: int,
) -> list[SelectedFrame]:
    arrays = _discover_numpy_arrays(path)
    for label, array in arrays:
        if array.ndim != 4 or array.shape[-1] != 3 or array.shape[0] == 0:
            continue
        indices = _selected_indices(
            array.shape[0], frame_selection=frame_selection, max_frames=max_frames
        )
        return [
            SelectedFrame(
                label=f"{label}:{index}",
                media_type="image/png",
                data=_array_frame_to_png(array[index]),
                source_kind="numpy-episode",
                source_index=index,
                source_count=int(array.shape[0]),
            )
            for index in indices
        ]
    return []


def _discover_numpy_arrays(path: Path) -> list[tuple[str, np.ndarray]]:
    candidates: list[Path] = []
    if path.is_file() and path.suffix.lower() in {".npy", ".npz"}:
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(
            file
            for file in path.rglob("*")
            if file.is_file() and file.suffix.lower() in {".npy", ".npz"}
        )
    candidates.sort(key=_numpy_preference_key)

    arrays: list[tuple[str, np.ndarray]] = []
    for candidate in candidates:
        try:
            if candidate.suffix.lower() == ".npz":
                bundle = np.load(candidate)
                for key in bundle.files:
                    arrays.append((f"{candidate.name}:{key}", np.asarray(bundle[key])))
            else:
                arrays.append((candidate.name, np.load(candidate, mmap_mode="r")))
        except (OSError, ValueError):
            continue
    return arrays


def _numpy_preference_key(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if "workspace" in name:
        return (0, name)
    if "image" in name or "frame" in name:
        return (1, name)
    if "wrist" in name:
        return (2, name)
    return (3, name)


def _frames_from_videos(
    path: Path,
    *,
    frame_selection: str,
    max_frames: int,
) -> list[SelectedFrame]:
    video_paths = _discover_video_paths(path)
    if not video_paths:
        return []
    if not shutil.which("ffmpeg"):
        raise VlmEvalError("Video rollout input requires ffmpeg to extract frames")

    video_path = video_paths[0]
    with tempfile.TemporaryDirectory(prefix="npa-vlm-video-") as tmp:
        output_dir = Path(tmp)
        count = _video_frame_count(video_path)
        source_indices, timestamps = _extract_selected_video_frames(
            video_path,
            output_dir,
            source_count=count,
            frame_selection=frame_selection,
            max_frames=max_frames,
        )
        return _selected_video_frames(
            video_path,
            output_dir,
            source_count=count,
            source_indices=source_indices,
            timestamps=timestamps,
        )


def _extract_selected_video_frames(
    video_path: Path,
    output_dir: Path,
    *,
    source_count: int | None,
    frame_selection: str,
    max_frames: int,
) -> tuple[list[int], list[float | None]]:
    if source_count:
        indices = _selected_indices(
            source_count,
            frame_selection=frame_selection,
            max_frames=max_frames,
        )
        return indices, _extract_video_indices(video_path, output_dir, indices)
    if frame_selection == "final":
        return [], _extract_final_video_frame(video_path, output_dir)
    return [], _extract_video_sample(video_path, output_dir, max_frames=max_frames)


def _selected_video_frames(
    video_path: Path,
    output_dir: Path,
    *,
    source_count: int | None,
    source_indices: Sequence[int],
    timestamps: Sequence[float | None],
) -> list[SelectedFrame]:
    extracted = sorted(output_dir.glob("frame-*.png"), key=_video_output_frame_number)
    return [
        SelectedFrame(
            label=f"{video_path.name}:{frame.name}",
            media_type="image/png",
            data=_image_file_to_png(frame),
            source_kind="video",
            source_index=source_indices[ordinal]
            if ordinal < len(source_indices)
            else None,
            source_count=source_count,
            source_timestamp_s=(
                timestamps[ordinal] if ordinal < len(timestamps) else None
            ),
        )
        for ordinal, frame in enumerate(extracted)
    ]


def _discover_video_paths(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in VIDEO_SUFFIXES
    )


def _video_frame_count(video_path: Path) -> int | None:
    if not shutil.which("ffprobe"):
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        streams = json.loads(proc.stdout).get("streams", [])
    except json.JSONDecodeError:
        return None
    if not streams:
        return None
    for key in ("nb_read_frames", "nb_frames"):
        value = streams[0].get(key)
        if value and str(value).isdigit():
            count = int(value)
            if count > 0:
                return count
    return None


def _extract_video_indices(
    video_path: Path, output_dir: Path, indices: list[int]
) -> list[float | None]:
    if not indices:
        return []
    timestamps = _video_frame_timestamps(video_path, indices)
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select={expression}",
        "-vsync",
        "0",
        str(output_dir / "frame-%03d.png"),
    ]
    _run_ffmpeg(cmd)
    return timestamps


def _extract_final_video_frame(
    video_path: Path, output_dir: Path
) -> list[float | None]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-sseof",
        "-0.1",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(output_dir / "frame-001.png"),
    ]
    _run_ffmpeg(cmd)
    return []


def _extract_video_sample(
    video_path: Path, output_dir: Path, *, max_frames: int
) -> list[float | None]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "fps=1",
        "-frames:v",
        str(max_frames),
        str(output_dir / "frame-%03d.png"),
    ]
    _run_ffmpeg(cmd)
    return []


def _run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VlmEvalError(f"ffmpeg frame extraction failed: {exc}") from exc
    if proc.returncode != 0:
        raise VlmEvalError(f"ffmpeg frame extraction failed: {proc.stderr[-500:]}")
    return proc


def _video_frame_timestamps(
    video_path: Path, indices: Sequence[int]
) -> list[float | None]:
    unavailable = [None] * len(indices)
    if not indices or not shutil.which("ffprobe"):
        return unavailable
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        frames = json.loads(proc.stdout).get("frames", [])
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return unavailable
    if (
        proc.returncode != 0
        or not isinstance(frames, list)
        or max(indices) >= len(frames)
    ):
        return unavailable
    return [_parsed_video_timestamp(frames[index]) for index in indices]


def _parsed_video_timestamp(frame: Any) -> float | None:
    raw_timestamp = (
        frame.get("best_effort_timestamp_time") if isinstance(frame, dict) else None
    )
    try:
        timestamp = float(raw_timestamp)
    except (TypeError, ValueError):
        return None
    return timestamp if math.isfinite(timestamp) else None


def _video_output_frame_number(path: Path) -> int:
    match = re.fullmatch(r"frame-(?P<number>\d+)\.png", path.name)
    if match is None:
        raise VlmEvalError(f"Unexpected extracted video frame name: {path.name}")
    return int(match.group("number"))


def _selected_indices(
    count: int, *, frame_selection: str, max_frames: int
) -> list[int]:
    if count <= 0:
        return []
    if frame_selection == "final":
        return [count - 1]
    selected = min(max_frames, count)
    if selected == 1:
        return [count - 1]
    return sorted({round(i * (count - 1) / (selected - 1)) for i in range(selected)})


def _image_file_to_png(path: Path) -> bytes:
    with Image.open(path) as image:
        return _pil_image_to_png(image)


def _array_frame_to_png(frame: np.ndarray) -> bytes:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and np.nanmax(array) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, "RGB")
    return _pil_image_to_png(image)


def _pil_image_to_png(image: Image.Image) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((768, 768))
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _load_json_object(text: str) -> tuple[dict[str, Any], bool]:
    stripped, deframed = _deframe_json_text(text)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise VlmEvalError("VLM response did not contain a JSON object") from None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise VlmEvalError("VLM response JSON could not be parsed") from exc
    if not isinstance(payload, dict):
        raise VlmEvalError("VLM response JSON must be an object")
    return payload, deframed


def _deframe_json_text(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    match = re.fullmatch(
        r"```(?:json)?\s*(?P<body>.*?)\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return stripped, False
    return match.group("body").strip(), True


def _parser_version(base_version: str, deframed: bool) -> str:
    if deframed:
        return base_version + MARKDOWN_FENCE_PARSER_SUFFIX
    return base_version


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise VlmEvalError("VLM response score must be numeric") from exc
    if math.isnan(score) or math.isinf(score):
        raise VlmEvalError("VLM response score must be finite")
    return max(0.0, min(1.0, score))


def _deterministic_score(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF
