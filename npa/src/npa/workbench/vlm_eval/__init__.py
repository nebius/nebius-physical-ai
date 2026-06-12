"""VLM evaluation helpers for sim-to-real pipeline validation."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
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
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import httpx
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from npa.clients.storage import StorageClient


DEFAULT_BACKEND = "self-hosted"
DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_ENDPOINT_URL = "http://127.0.0.1:8000/v1"
DEFAULT_FRAME_SELECTION = "keyframes"
DEFAULT_MAX_FRAMES = 4
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_API_KEY_ENV = "VLM_EVAL_API_KEY"
DEFAULT_RUBRIC = (
    "Score whether the rollout completes the requested physical task. "
    "Use 1.0 only for clear task completion, 0.0 for clear failure, and "
    "intermediate values for partial progress. Penalize unsafe, incomplete, "
    "or ambiguous outcomes."
)
RESULT_FILENAME = "vlm_eval_stub.json"
BENCHMARK_RESULT_FILENAME = "vlm_eval_benchmark.json"
BENCHMARK_DATASET_FORMAT = "npa_vlm_eval_benchmark_v1"
DEFAULT_BENCHMARK_THRESHOLDS = (0.5, 0.8, 0.9)
DEFAULT_SAMPLE_BENCHMARK_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "sample_benchmark" / "benchmark.json"
)
SUPPORTED_BACKENDS = ("self-hosted", "api", "stub")
SUPPORTED_FRAME_SELECTIONS = ("final", "keyframes", "sequence")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".webp"}
VIDEO_SUFFIXES = {".avi", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


class VlmEvalError(ValueError):
    """Raised when a VLM evaluation request is invalid."""


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


@dataclass(frozen=True)
class VlmStructuredResponse:
    success: bool
    score: float
    rationale: str


@dataclass(frozen=True)
class SelectedFrame:
    label: str
    media_type: str
    data: bytes


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
    "VlmEvalResult",
    "VlmStructuredResponse",
    "benchmark_result_uri_for",
    "benchmark_vlm_eval",
    "evaluate_stub",
    "evaluate_vlm",
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
        for index, result in enumerate(sorted(config_results, key=_benchmark_rank_key), start=1)
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
            raise VlmEvalError(f"benchmark dataset is not valid JSON: {local_manifest}") from exc

    if isinstance(payload, list):
        raw_items = payload
        dataset_format = BENCHMARK_DATASET_FORMAT
        rubrics: dict[str, str] = {}
        rollout_base_path = ""
    elif isinstance(payload, dict):
        raw_items = payload.get("items") or payload.get("rollouts")
        dataset_format = str(payload.get("format") or BENCHMARK_DATASET_FORMAT)
        rubrics = _coerce_rubric_map(payload.get("rubrics", {}))
        rollout_base_path = str(payload.get("rollout_base_path") or payload.get("base_path") or "")
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
    if backend == "stub":
        return evaluate_stub(
            input_path=input_path,
            output_path=output_path,
            task=task,
            model=model or "vlm-eval-stub",
            success_threshold=success_threshold,
            frame_selection=frame_selection,
            score=score,
        )

    effective_model = model or DEFAULT_MODEL
    effective_rubric = _load_rubric(rubric=rubric, rubric_path=rubric_path)
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
                frames=frames,
                timeout_s=timeout_s,
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
        structured=structured,
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
    effective_score = _deterministic_score(input_path, task, model) if score is None else score
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
    image_frames = _frames_from_images(path, frame_selection=frame_selection, max_frames=max_frames)
    if image_frames:
        return image_frames

    numpy_frames = _frames_from_numpy(path, frame_selection=frame_selection, max_frames=max_frames)
    if numpy_frames:
        return numpy_frames

    video_frames = _frames_from_videos(path, frame_selection=frame_selection, max_frames=max_frames)
    if video_frames:
        return video_frames

    raise VlmEvalError(
        f"No rollout frames found in {path}. Expected image files, RGB .npy/.npz arrays, or videos."
    )


def parse_structured_response(text: str) -> VlmStructuredResponse:
    """Parse a VLM JSON response and clamp its score into [0, 1]."""

    payload = _load_json_object(text)
    if "score" not in payload:
        raise VlmEvalError("VLM response JSON must include score")
    if "rationale" not in payload:
        raise VlmEvalError("VLM response JSON must include rationale")
    score = _clamp_score(payload["score"])
    success = _coerce_bool(payload.get("success", score >= 0.5))
    return VlmStructuredResponse(
        success=success,
        score=score,
        rationale=str(payload["rationale"]),
    )


def result_uri_for(output_path: str) -> str:
    """Return the JSON artifact URI for an output path."""

    if output_path.endswith(".json"):
        return output_path
    return output_path.rstrip("/") + f"/{RESULT_FILENAME}"


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
    structured: VlmStructuredResponse,
) -> VlmEvalResult:
    score = round(_clamp_score(structured.score), 4)
    passed = score >= success_threshold
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
    score = item.fixture_score if use_fixture_score and item.fixture_score is not None else None
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
    )


def _benchmark_metrics(results: Sequence[VlmBenchmarkCaseResult]) -> VlmBenchmarkMetrics:
    total = len(results)
    if total == 0:
        raise VlmEvalError("benchmark dataset must include at least one item")
    tp = sum(1 for result in results if result.expected_label and result.predicted_label)
    tn = sum(1 for result in results if not result.expected_label and not result.predicted_label)
    fp = sum(1 for result in results if not result.expected_label and result.predicted_label)
    fn = sum(1 for result in results if result.expected_label and not result.predicted_label)
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

        with tempfile.TemporaryDirectory(prefix="npa-vlm-eval-benchmark-dataset-") as tmp:
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
    rollout = raw_item.get("rollout") or raw_item.get("rollout_path") or raw_item.get("input_path")
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

    item_id = str(raw_item.get("id") or raw_item.get("name") or f"item-{index:03d}").strip()
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


def _call_openai_compatible(
    *,
    backend: str,
    model: str,
    endpoint_url: str,
    api_key_env: str,
    prompt: str,
    frames: list[SelectedFrame],
    timeout_s: float,
) -> VlmStructuredResponse:
    url = _chat_completions_url(_resolve_endpoint_url(backend=backend, endpoint_url=endpoint_url))
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in frames:
        encoded = base64.b64encode(frame.data).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{frame.media_type};base64,{encoded}"},
            }
        )

    headers = {"Content-Type": "application/json"}
    api_key = _resolve_api_key(backend=backend, api_key_env=api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}],
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(url, headers=headers, json=request)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise VlmEvalError(f"VLM backend request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise VlmEvalError("VLM backend returned non-JSON response") from exc

    try:
        message = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VlmEvalError("VLM backend response missing choices[0].message.content") from exc
    return parse_structured_response(str(message))


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
            or os.environ.get("NEBIUS_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        if not key:
            raise VlmEvalError(
                f"--backend api requires an API key in {api_key_env or DEFAULT_API_KEY_ENV}, "
                "NEBIUS_API_KEY, or OPENAI_API_KEY"
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
    indices = _selected_indices(len(image_paths), frame_selection=frame_selection, max_frames=max_frames)
    return [
        SelectedFrame(
            label=image_paths[index].name,
            media_type="image/png",
            data=_image_file_to_png(image_paths[index]),
        )
        for index in indices
    ]


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
        indices = _selected_indices(array.shape[0], frame_selection=frame_selection, max_frames=max_frames)
        return [
            SelectedFrame(
                label=f"{label}:{index}",
                media_type="image/png",
                data=_array_frame_to_png(array[index]),
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
        if count:
            indices = _selected_indices(count, frame_selection=frame_selection, max_frames=max_frames)
            _extract_video_indices(video_path, output_dir, indices)
        elif frame_selection == "final":
            _extract_final_video_frame(video_path, output_dir)
        else:
            _extract_video_sample(video_path, output_dir, max_frames=max_frames)
        return [
            SelectedFrame(
                label=frame.name,
                media_type="image/png",
                data=_image_file_to_png(frame),
            )
            for frame in sorted(output_dir.glob("frame-*.png"))
        ]


def _discover_video_paths(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        return [path]
    if not path.is_dir():
        return []
    return sorted(file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in VIDEO_SUFFIXES)


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


def _extract_video_indices(video_path: Path, output_dir: Path, indices: list[int]) -> None:
    if not indices:
        return
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


def _extract_final_video_frame(video_path: Path, output_dir: Path) -> None:
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


def _extract_video_sample(video_path: Path, output_dir: Path, *, max_frames: int) -> None:
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


def _run_ffmpeg(cmd: list[str]) -> None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VlmEvalError(f"ffmpeg frame extraction failed: {exc}") from exc
    if proc.returncode != 0:
        raise VlmEvalError(f"ffmpeg frame extraction failed: {proc.stderr[-500:]}")


def _selected_indices(count: int, *, frame_selection: str, max_frames: int) -> list[int]:
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


def _load_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
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
    return payload


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
