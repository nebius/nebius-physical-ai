"""npa workbench vlm-eval - VLM rollout evaluation commands."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from npa.deploy.images import DEFAULT_VLM_IMAGE_ENV, default_vlm_image
from npa.lifecycle_intent import json_stdout_contract
from npa.workbench.vlm_eval import (
    DEFAULT_BENCHMARK_THRESHOLDS,
    DEFAULT_API_KEY_ENV,
    DEFAULT_BACKEND,
    DEFAULT_FRAME_SELECTION,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_BENCHMARK_PATH,
    DEFAULT_RUBRIC,
    DEFAULT_TIMEOUT_S,
    DEFAULT_VISUAL_REVIEW_RUBRIC,
    SUPPORTED_BACKENDS,
    SUPPORTED_FRAME_SELECTIONS,
    VlmEvalError,
    VlmJudgeComparisonRequest,
    VlmPreferenceComparisonRequest,
    VlmVisualReviewReport,
    VlmVisualReviewRequest,
    benchmark_result_uri_for,
    benchmark_vlm_eval,
    compare_vlm_preference,
    compare_vlm_judges,
    evaluate_rollout_set,
    evaluate_vlm,
    review_visual as run_visual_review,
    write_benchmark_report,
    write_preference_report,
    write_result,
)

app = typer.Typer(
    name="vlm-eval",
    help="VLM evaluation for sim-to-real pipeline gating.",
    no_args_is_help=True,
)
console = Console(stderr=True)
# The `npa.workflow` specs this tool is driven by. Paths only: `vlm-eval workflow` /
# `status` print them; `npa workbench workflow submit <path>` runs them.
NPA_WORKFLOWS = Path("workflows/testing")
WORKFLOW_PATH = NPA_WORKFLOWS / "vlm-eval-single.yaml"
BENCHMARK_WORKFLOW_PATH = NPA_WORKFLOWS / "vlm-eval-benchmark.yaml"
# Zero-GPU hosted alternative: the `api` backend needs no vLLM server.
TOKEN_FACTORY_WORKFLOW_PATH = NPA_WORKFLOWS / "vlm-eval-token-factory.yaml"


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class BackendName(str, Enum):
    self_hosted = "self-hosted"
    api = "api"
    stub = "stub"


class FrameSelection(str, Enum):
    final = "final"
    keyframes = "keyframes"
    sequence = "sequence"


@dataclass(frozen=True)
class _ComparisonCliOptions:
    input_path: str
    output_path: str
    primary_model: str
    secondary_model: str
    task: str
    endpoint_url: str
    api_key_env: str
    frame_selection: str
    max_frames: int
    rubric: str
    rubric_path: str
    success_threshold: float
    timeout_s: float


@dataclass(frozen=True)
class _PreferenceCliOptions:
    baseline_path: str
    candidate_path: str
    output_path: str
    model: str
    task: str
    endpoint_url: str
    api_key_env: str
    rubric: str
    rubric_path: str
    timeout_s: float


@dataclass(frozen=True)
class _VisualReviewCliOptions:
    input_path: str
    output_path: str
    model: str
    task: str
    baseline_path: str
    frame_selection: str
    max_frames: int
    endpoint_url: str
    api_key_env: str
    rubric: str
    rubric_path: str
    objective_evidence_path: str
    matched_view_map_path: str
    timeout_s: float


_COMPARE_INPUT = typer.Option(
    ..., "--input-path", help="S3 or local artifact path to review."
)
_COMPARE_OUTPUT_PATH = typer.Option(
    ..., "--output-path", help="S3 or local path for the comparison JSON."
)
_COMPARE_PRIMARY = typer.Option(
    ..., "--primary-model", help="First hosted vision model ID."
)
_COMPARE_SECONDARY = typer.Option(
    ..., "--secondary-model", help="Distinct second hosted vision model ID."
)
_COMPARE_TASK = typer.Option("sim-to-real", "--task", help="Evaluation task label.")
_COMPARE_ENDPOINT = typer.Option(
    "",
    "--endpoint-url",
    help="Hosted OpenAI-compatible base URL or /chat/completions URL.",
)
_COMPARE_API_KEY = typer.Option(
    DEFAULT_API_KEY_ENV,
    "--api-key-env",
    help="Environment variable containing the hosted API key.",
)
_COMPARE_FRAME_SELECTION = typer.Option(
    FrameSelection.keyframes,
    "--frame-selection",
    help="Rollout frame selection: final, keyframes, or sequence.",
)
_COMPARE_MAX_FRAMES = typer.Option(
    DEFAULT_MAX_FRAMES,
    "--max-frames",
    help="Maximum shared frames sent to each judge.",
)
_COMPARE_RUBRIC = typer.Option(DEFAULT_RUBRIC, "--rubric", help="Scoring rubric text.")
_COMPARE_RUBRIC_PATH = typer.Option(
    "", "--rubric-path", help="Path to a scoring rubric text file."
)
_COMPARE_THRESHOLD = typer.Option(
    0.8,
    "--success-threshold",
    help="Score threshold used independently for both verdicts.",
)
_COMPARE_TIMEOUT = typer.Option(
    DEFAULT_TIMEOUT_S,
    "--timeout-s",
    help="Timeout for each hosted judge request.",
)
_COMPARE_DRY_RUN = typer.Option(
    False, "--dry-run", help="Run both judges without writing the artifact."
)
_COMPARE_OUTPUT = typer.Option(OutputFormat.text, "--output", help="Output format.")
_PREFERENCE_BASELINE = typer.Option(
    ..., "--baseline-path", help="First matched image path; kept out of provider text."
)
_PREFERENCE_CANDIDATE = typer.Option(
    ...,
    "--candidate-path",
    help="Second matched image path; kept out of provider text.",
)
_PREFERENCE_OUTPUT_PATH = typer.Option(
    ..., "--output-path", help="Private path for vlm_preference_comparison.json."
)
_PREFERENCE_MODEL = typer.Option(
    "MiniMaxAI/MiniMax-M3", "--model", help="Hosted vision model ID."
)
_PREFERENCE_TASK = typer.Option(
    ..., "--task", help="Shared comparison task without source-role words."
)
_PREFERENCE_ENDPOINT = typer.Option(
    "", "--endpoint-url", help="Explicit hosted OpenAI-compatible endpoint."
)
_PREFERENCE_API_KEY = typer.Option(
    DEFAULT_API_KEY_ENV,
    "--api-key-env",
    help="Environment variable containing the hosted API key.",
)
_PREFERENCE_RUBRIC = typer.Option(
    "", "--rubric", help="Shared comparison rubric without source-role words."
)
_PREFERENCE_RUBRIC_PATH = typer.Option(
    "", "--rubric-path", help="Local path to the shared comparison rubric."
)
_PREFERENCE_TIMEOUT = typer.Option(
    DEFAULT_TIMEOUT_S, "--timeout-s", help="Timeout for each one-shot hosted request."
)
_PREFERENCE_OUTPUT = typer.Option(
    "text", "--output-format", "--output", help="Output format: text or json."
)
_VISUAL_INPUT = typer.Option(
    ..., "--input-path", help="S3 or local current visual artifact."
)
_VISUAL_OUTPUT_PATH = typer.Option(
    ...,
    "--output-path",
    help="Private prefix or exact vlm_visual_review.json destination.",
)
_VISUAL_MODEL = typer.Option(..., "--model", help="Exact hosted vision model ID.")
_VISUAL_TASK = typer.Option(
    ..., "--task", help="Neutral visible-review task without source-role words."
)
_VISUAL_BASELINE = typer.Option(
    "", "--baseline-path", help="Optional private baseline visual artifact."
)
_VISUAL_FRAME_SELECTION = typer.Option(
    FrameSelection.keyframes,
    "--frame-selection",
    help="Frame selection applied independently to each source.",
)
_VISUAL_MAX_FRAMES = typer.Option(
    DEFAULT_MAX_FRAMES,
    "--max-frames",
    help="Maximum frames selected independently from each source.",
)
_VISUAL_ENDPOINT = typer.Option(
    "", "--endpoint-url", help="Explicit hosted OpenAI-compatible endpoint."
)
_VISUAL_API_KEY = typer.Option(
    DEFAULT_API_KEY_ENV,
    "--api-key-env",
    help="Environment variable containing the hosted API key.",
)
_VISUAL_RUBRIC = typer.Option(
    DEFAULT_VISUAL_REVIEW_RUBRIC, "--rubric", help="Rich visual-review rubric."
)
_VISUAL_RUBRIC_PATH = typer.Option(
    "", "--rubric-path", help="Private local rubric file replacing --rubric."
)
_VISUAL_OBJECTIVE_PATH = typer.Option(
    "",
    "--objective-evidence-path",
    help="Private JSON file of unverified objective-evidence references.",
)
_VISUAL_MATCHED_VIEW_PATH = typer.Option(
    "",
    "--matched-view-map-path",
    help="Private JSON file of unverified matched-view metadata.",
)
_VISUAL_TIMEOUT = typer.Option(
    DEFAULT_TIMEOUT_S, "--timeout-s", help="Timeout for each one-shot hosted request."
)
_VISUAL_OUTPUT = typer.Option(
    "text", "--output-format", "--output", help="Output format: text or json."
)


@app.command("run")
def run_cmd(
    input_path: str = typer.Option(
        ..., "--input-path", help="S3 or local artifact path to score."
    ),
    output_path: str = typer.Option(
        ..., "--output-path", help="S3 or local path for eval JSON."
    ),
    task: str = typer.Option("sim-to-real", "--task", help="Evaluation task label."),
    task_from: str = typer.Option(
        "",
        "--task-from",
        help=(
            "Read the task from a reasoning artifact (its `analysis` field) instead of --task, "
            "so a judge can score a rollout against a plan an earlier stage wrote."
        ),
    ),
    backend: BackendName = typer.Option(
        BackendName.self_hosted,
        "--backend",
        help="VLM backend: self-hosted, api, or stub.",
    ),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="VLM model name."),
    endpoint_url: str = typer.Option(
        "",
        "--endpoint-url",
        help="OpenAI-compatible base URL or /chat/completions URL.",
    ),
    api_key_env: str = typer.Option(
        DEFAULT_API_KEY_ENV,
        "--api-key-env",
        help="Environment variable containing the API key for --backend api.",
    ),
    frame_selection: FrameSelection = typer.Option(
        FrameSelection.keyframes,
        "--frame-selection",
        help="Rollout frame selection: final, keyframes, or sequence.",
    ),
    max_frames: int = typer.Option(
        DEFAULT_MAX_FRAMES,
        "--max-frames",
        help="Maximum frames sent to the VLM.",
    ),
    rubric: str = typer.Option(
        DEFAULT_RUBRIC,
        "--rubric",
        help="Scoring rubric text.",
    ),
    rubric_path: str = typer.Option(
        "",
        "--rubric-path",
        help="Path to a scoring rubric text file.",
    ),
    success_threshold: float = typer.Option(
        0.8,
        "--success-threshold",
        help="Score threshold required to pass.",
    ),
    score: float = typer.Option(
        -1.0,
        "--score",
        help="Override score for tests and dry validation; skips the VLM call.",
    ),
    timeout_s: float = typer.Option(
        DEFAULT_TIMEOUT_S,
        "--timeout-s",
        help="VLM request timeout in seconds.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Do not write the result artifact."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Score a rollout artifact with a VLM backend."""

    try:
        if task_from.strip():
            task = task_from_reasoning_artifact(task_from)
        result = evaluate_vlm(
            input_path=input_path,
            output_path=output_path,
            task=task,
            backend=_enum_value(backend),
            model=model,
            endpoint_url=endpoint_url,
            api_key_env=api_key_env,
            frame_selection=_enum_value(frame_selection),
            max_frames=max_frames,
            rubric=rubric,
            rubric_path=rubric_path,
            success_threshold=success_threshold,
            timeout_s=timeout_s,
            score=None if score < 0 else score,
        )
        payload = asdict(result)
        effective_dry_run = dry_run or _env_dry_run()
        payload["dry_run"] = effective_dry_run
        if not effective_dry_run:
            payload["written_uri"] = write_result(payload, result_uri=result.result_uri)
    except VlmEvalError as exc:
        _fail(str(exc))
        return
    _emit(payload, output)


def _comparison_console_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "passed": payload["passed"],
        "escalation_required": payload["escalation_required"],
        "deployment_status": payload["deployment_status"],
        "operational_rate_estimated": payload["operational_rate_estimated"],
        "requests_differ_only_by_model": payload["requests_differ_only_by_model"],
        "frame_count": payload["frame_count"],
        "primary": _judge_console_summary(payload["primary"]),
        "secondary": _judge_console_summary(payload["secondary"]),
        "artifact_written": "written_uri" in payload,
        "dry_run": payload["dry_run"],
    }


def _judge_console_summary(outcome: dict[str, Any]) -> dict[str, Any]:
    result = outcome.get("result")
    if isinstance(result, dict):
        return {
            "model": outcome["model"],
            "status": result["status"],
            "score": result["score"],
            "passed": result["passed"],
        }
    error = outcome.get("error") or {}
    return {
        "model": outcome["model"],
        "status": "error",
        "error_stage": error.get("stage"),
        "error_type": error.get("error_type"),
    }


@app.command("compare-judges")
def compare_judges_cmd(
    input_path: str = _COMPARE_INPUT,
    output_path: str = _COMPARE_OUTPUT_PATH,
    primary_model: str = _COMPARE_PRIMARY,
    secondary_model: str = _COMPARE_SECONDARY,
    task: str = _COMPARE_TASK,
    endpoint_url: str = _COMPARE_ENDPOINT,
    api_key_env: str = _COMPARE_API_KEY,
    frame_selection: FrameSelection = _COMPARE_FRAME_SELECTION,
    max_frames: int = _COMPARE_MAX_FRAMES,
    rubric: str = _COMPARE_RUBRIC,
    rubric_path: str = _COMPARE_RUBRIC_PATH,
    success_threshold: float = _COMPARE_THRESHOLD,
    timeout_s: float = _COMPARE_TIMEOUT,
    dry_run: bool = _COMPARE_DRY_RUN,
    output: OutputFormat = _COMPARE_OUTPUT,
) -> None:
    """Compare two hosted judges without averaging their outcomes."""
    options = _ComparisonCliOptions(
        input_path,
        output_path,
        primary_model,
        secondary_model,
        task,
        endpoint_url,
        api_key_env,
        _enum_value(frame_selection),
        max_frames,
        rubric,
        rubric_path,
        success_threshold,
        timeout_s,
    )
    try:
        payload = _execute_judge_comparison(options, dry_run=dry_run)
    except VlmEvalError as exc:
        _fail(str(exc))
        return
    _emit(_comparison_console_summary(payload), output)


def _execute_judge_comparison(
    options: _ComparisonCliOptions,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    report = compare_vlm_judges(VlmJudgeComparisonRequest(**asdict(options)))
    payload = asdict(report)
    payload["dry_run"] = dry_run or _env_dry_run()
    if not payload["dry_run"]:
        payload["written_uri"] = write_result(
            payload,
            result_uri=report.result_uri,
        )
    return payload


@app.command("compare-preference")
@json_stdout_contract
def compare_preference_cmd(
    baseline_path: str = _PREFERENCE_BASELINE,
    candidate_path: str = _PREFERENCE_CANDIDATE,
    output_path: str = _PREFERENCE_OUTPUT_PATH,
    model: str = _PREFERENCE_MODEL,
    task: str = _PREFERENCE_TASK,
    endpoint_url: str = _PREFERENCE_ENDPOINT,
    api_key_env: str = _PREFERENCE_API_KEY,
    rubric: str = _PREFERENCE_RUBRIC,
    rubric_path: str = _PREFERENCE_RUBRIC_PATH,
    timeout_s: float = _PREFERENCE_TIMEOUT,
    output_format: str = _PREFERENCE_OUTPUT,
) -> None:
    """Compare two images under blinded labels in both orders."""
    output = _parse_output_format(output_format)
    options = _PreferenceCliOptions(
        baseline_path,
        candidate_path,
        output_path,
        model,
        task,
        endpoint_url,
        api_key_env,
        rubric,
        rubric_path,
        timeout_s,
    )
    try:
        payload = _execute_preference_comparison(options)
    except VlmEvalError:
        _fail("Blinded preference comparison failed; inspect private evidence.")
        return
    _emit(_preference_console_summary(payload), output)


def _parse_output_format(value: str) -> OutputFormat:
    try:
        return OutputFormat(value)
    except ValueError:
        _fail("--output-format must be text or json")
        raise AssertionError("unreachable")


def _execute_preference_comparison(
    options: _PreferenceCliOptions,
) -> dict[str, Any]:
    report = compare_vlm_preference(VlmPreferenceComparisonRequest(**asdict(options)))
    payload = asdict(report)
    payload["written"] = write_preference_report(
        payload,
        result_uri=report.result_uri,
    )
    return payload


def _preference_console_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "escalation_required": payload["escalation_required"],
        "agreement_eligible": payload["agreement_eligible"],
        "deployment_status": payload["deployment_status"],
        "operational_rate_estimated": payload["operational_rate_estimated"],
        "model": payload["model"],
        "requests_counterbalanced": payload["requests_counterbalanced"],
        "first_order": _preference_order_summary(payload["first_order"]),
        "reversed_order": _preference_order_summary(payload["reversed_order"]),
        "artifact_written": "written" in payload,
    }


def _preference_order_summary(outcome: dict[str, Any]) -> dict[str, Any]:
    verdict = outcome.get("verdict")
    if isinstance(verdict, dict):
        return {
            "order_id": outcome["order_id"],
            "status": "completed",
            "preference": verdict["preference"],
            "confidence": verdict["confidence"],
        }
    error = outcome.get("error") or {}
    return {
        "order_id": outcome["order_id"],
        "status": "error",
        "error_stage": error.get("stage"),
        "error_type": error.get("error_type"),
    }


@app.command("review-visual")
@json_stdout_contract
def review_visual_cmd(
    input_path: str = _VISUAL_INPUT,
    output_path: str = _VISUAL_OUTPUT_PATH,
    model: str = _VISUAL_MODEL,
    task: str = _VISUAL_TASK,
    baseline_path: str = _VISUAL_BASELINE,
    frame_selection: FrameSelection = _VISUAL_FRAME_SELECTION,
    max_frames: int = _VISUAL_MAX_FRAMES,
    endpoint_url: str = _VISUAL_ENDPOINT,
    api_key_env: str = _VISUAL_API_KEY,
    rubric: str = _VISUAL_RUBRIC,
    rubric_path: str = _VISUAL_RUBRIC_PATH,
    objective_evidence_path: str = _VISUAL_OBJECTIVE_PATH,
    matched_view_map_path: str = _VISUAL_MATCHED_VIEW_PATH,
    timeout_s: float = _VISUAL_TIMEOUT,
    output_format: str = _VISUAL_OUTPUT,
) -> None:
    """Write a separate audit-only rich visual review."""
    arguments = dict(locals())
    output = _parse_output_format(arguments.pop("output_format"))
    arguments["frame_selection"] = _enum_value(frame_selection)
    options = _VisualReviewCliOptions(**arguments)
    try:
        report = run_visual_review(VlmVisualReviewRequest(**asdict(options)))
    except Exception:  # noqa: BLE001 - sanitize every private CLI failure
        _fail("Visual review failed; inspect private evidence.")
        return
    _emit(_visual_review_console_summary(report), output)


def _visual_review_console_summary(
    report: VlmVisualReviewReport,
) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "status": report.status,
        "escalation_required": report.escalation_required,
        "attempt_count": report.attempt_count,
        "model": report.model,
    }


#: How much of a plan to carry into the judge prompt; the retired template used this budget.
PLAN_TASK_CHARS = 900


def task_from_reasoning_artifact(uri: str) -> str:
    """Build a judge task from the `analysis` an earlier reasoning stage wrote.

    The retired `tokenfactory-scene-to-rollout-judge.yaml` did this in inline python and passed
    the result through `--task "$(…)"`. That command substitution IS the three-stage combo — the
    judge scores the rollout *against the plan the reasoner produced* — and it is exactly what a
    `toolRef` argv cannot express.
    """

    import tempfile as _tempfile

    raw = uri.strip()
    if not raw:
        _fail("--task-from needs a reasoning artifact URI")
    if raw.startswith("s3://"):
        from npa.clients.storage import StorageClient

        with _tempfile.TemporaryDirectory(prefix="npa-plan-") as tmp:
            local = Path(StorageClient.from_environment().download_path(raw, tmp))
            payload = _read_reasoning_payload(local, uri)
    else:
        payload = _read_reasoning_payload(Path(raw), uri)
    analysis = str(payload.get("analysis") or "").strip().replace("\n", " ")
    if not analysis:
        _fail(f"reasoning artifact has no `analysis` to judge against: {uri}")
    return (
        "Judge whether the robot rollout accomplishes this planned task. "
        f"Plan: {analysis[:PLAN_TASK_CHARS]}"
    )


def _read_reasoning_payload(local: Path, uri: str) -> dict[str, Any]:
    if not local.is_file():
        _fail(f"reasoning artifact not found: {uri}")
    try:
        return json.loads(local.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _fail(f"reasoning artifact is not readable JSON: {uri} ({exc})")
        return {}


@app.command("loop")
def loop_cmd(
    input_path: str = typer.Option(
        ...,
        "--input-path",
        help="S3 or local prefix containing one directory per rollout.",
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        help="S3 or local prefix for per-rollout results and the report.",
    ),
    task: str = typer.Option("sim-to-real", "--task", help="Evaluation task label."),
    backend: BackendName = typer.Option(
        BackendName.self_hosted,
        "--backend",
        help="VLM backend: self-hosted, api, or stub.",
    ),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="VLM model name."),
    endpoint_url: str = typer.Option(
        "",
        "--endpoint-url",
        help="OpenAI-compatible base URL or /chat/completions URL.",
    ),
    api_key_env: str = typer.Option(
        DEFAULT_API_KEY_ENV,
        "--api-key-env",
        help="Environment variable containing the API key for --backend api.",
    ),
    frame_selection: FrameSelection = typer.Option(
        FrameSelection.keyframes,
        "--frame-selection",
        help="Rollout frame selection: final, keyframes, or sequence.",
    ),
    max_frames: int = typer.Option(
        DEFAULT_MAX_FRAMES,
        "--max-frames",
        help="Maximum frames sent to the VLM per rollout.",
    ),
    rubric: str = typer.Option(DEFAULT_RUBRIC, "--rubric", help="Scoring rubric text."),
    rubric_path: str = typer.Option(
        "", "--rubric-path", help="Path to a scoring rubric text file."
    ),
    success_threshold: float = typer.Option(
        0.8,
        "--success-threshold",
        help="Mean-score threshold for the coarse task_success gate.",
    ),
    timeout_s: float = typer.Option(
        DEFAULT_TIMEOUT_S, "--timeout-s", help="VLM request timeout in seconds."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Score every rollout under a prefix and write an aggregate task-success report.

    ``run`` scores one rollout: it discovers frames recursively, so a prefix holding many
    rollouts would blend into a single score. The retired ``sim-to-real-loop.yaml`` did the
    enumeration and aggregation in bash and `jq`; this is the same behaviour in the tool,
    where an npa.workflow spec can reach it.
    """

    try:
        report = evaluate_rollout_set(
            input_path=input_path,
            output_path=output_path,
            task=task,
            backend=_enum_value(backend),
            model=model,
            endpoint_url=endpoint_url,
            api_key_env=api_key_env,
            frame_selection=_enum_value(frame_selection),
            max_frames=max_frames,
            rubric=rubric,
            rubric_path=rubric_path,
            success_threshold=success_threshold,
            timeout_s=timeout_s,
        )
    except VlmEvalError as exc:
        _fail(str(exc))
        return
    _emit(report, output)


@app.command("benchmark")
def benchmark_cmd(
    dataset: str = typer.Option(
        str(DEFAULT_SAMPLE_BENCHMARK_PATH),
        "--dataset",
        help="Benchmark manifest JSON or directory; defaults to the packaged sample fixture.",
    ),
    output_path: str = typer.Option(
        ...,
        "--output",
        help="Local or S3 path for the benchmark report JSON.",
    ),
    thresholds: str = typer.Option(
        ",".join(str(value) for value in DEFAULT_BENCHMARK_THRESHOLDS),
        "--thresholds",
        help="Comma-separated success thresholds to sweep.",
    ),
    rubrics: str = typer.Option(
        "default",
        "--rubrics",
        help="Comma-separated rubric names from the dataset, inline rubric text, or @file paths.",
    ),
    models: str = typer.Option(
        DEFAULT_MODEL,
        "--models",
        help="Comma-separated model names to sweep.",
    ),
    backend: BackendName = typer.Option(
        BackendName.self_hosted,
        "--backend",
        help="VLM backend: self-hosted, api, or stub.",
    ),
    task: str = typer.Option("sim-to-real", "--task", help="Fallback task label."),
    endpoint_url: str = typer.Option(
        "",
        "--endpoint-url",
        help="OpenAI-compatible base URL or /chat/completions URL.",
    ),
    api_key_env: str = typer.Option(
        DEFAULT_API_KEY_ENV,
        "--api-key-env",
        help="Environment variable containing the API key for --backend api.",
    ),
    frame_selection: FrameSelection = typer.Option(
        FrameSelection.keyframes,
        "--frame-selection",
        help="Rollout frame selection: final, keyframes, or sequence.",
    ),
    max_frames: int = typer.Option(
        DEFAULT_MAX_FRAMES,
        "--max-frames",
        help="Maximum frames sent to the VLM per rollout.",
    ),
    timeout_s: float = typer.Option(
        DEFAULT_TIMEOUT_S,
        "--timeout-s",
        help="VLM request timeout in seconds.",
    ),
    use_fixture_scores: bool = typer.Option(
        False,
        "--use-fixture-scores",
        help="Honor fixture_score values for non-stub backends; stub always uses them when present.",
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.text, "--format", help="Console output format."
    ),
) -> None:
    """Sweep VLM-eval configs over a labeled rollout benchmark set."""

    try:
        report = benchmark_vlm_eval(
            dataset=dataset,
            thresholds=_parse_thresholds(thresholds),
            rubrics=_parse_csv(rubrics),
            models=_parse_csv(models),
            backend=_enum_value(backend),
            task=task,
            frame_selection=_enum_value(frame_selection),
            max_frames=max_frames,
            endpoint_url=endpoint_url,
            api_key_env=api_key_env,
            timeout_s=timeout_s,
            use_fixture_scores=use_fixture_scores,
        )
        payload = asdict(report)
        payload["written_uri"] = benchmark_result_uri_for(output_path)
        payload["written_uri"] = write_benchmark_report(
            payload, output_path=output_path
        )
    except VlmEvalError as exc:
        _fail(str(exc))
        return
    _emit_benchmark(payload, format)


@app.command("workflow")
def workflow_cmd(
    image: str = typer.Option(
        "",
        "--image",
        envvar=DEFAULT_VLM_IMAGE_ENV,
        help="Self-hosted VLM workflow image. Also settable with NPA_VLM_IMAGE.",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Show the npa.workflow specs for VLM evaluation."""

    _emit(
        {
            "workflow": str(WORKFLOW_PATH),
            "benchmark_workflow": str(BENCHMARK_WORKFLOW_PATH),
            # Zero-GPU alternative; `self-hosted` needs a vLLM server, `api` does not.
            "token_factory_workflow": str(TOKEN_FACTORY_WORKFLOW_PATH),
            "image_env": DEFAULT_VLM_IMAGE_ENV,
            "image": image.strip() or default_vlm_image(),
        },
        output,
    )


@app.command("status")
def status_cmd(
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Show VLM eval backend status."""

    _emit(
        {
            "backend": DEFAULT_BACKEND,
            "status": "configured",
            "real_vlm_backend": True,
            "default_model": DEFAULT_MODEL,
            "default_frame_selection": DEFAULT_FRAME_SELECTION,
            "workflow": str(WORKFLOW_PATH),
            "benchmark_workflow": str(BENCHMARK_WORKFLOW_PATH),
            "token_factory_workflow": str(TOKEN_FACTORY_WORKFLOW_PATH),
            "sample_benchmark_dataset": str(DEFAULT_SAMPLE_BENCHMARK_PATH),
        },
        output,
    )


@app.command("list")
def list_cmd(
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """List available VLM eval backends."""

    _emit(
        {
            "backends": [
                {"name": "self-hosted", "real_backend": True, "default": True},
                {"name": "api", "real_backend": True, "default": False},
                {"name": "stub", "real_backend": False, "default": False},
            ],
            "frame_selections": list(SUPPORTED_FRAME_SELECTIONS),
            "supported_backends": list(SUPPORTED_BACKENDS),
        },
        output,
    )


def _emit(payload: dict[str, Any], output: OutputFormat) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        typer.echo(f"  {key}: {value}")


def _emit_benchmark(payload: dict[str, Any], output: OutputFormat) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    best = payload["best_config"]
    config = best["config"]
    metrics = best["metrics"]
    typer.echo(f"  dataset: {payload['dataset_path']}")
    typer.echo(f"  items: {payload['item_count']}")
    typer.echo(f"  written_uri: {payload['written_uri']}")
    typer.echo("  best_config:")
    typer.echo(f"    backend: {config['backend']}")
    typer.echo(f"    model: {config['model']}")
    typer.echo(f"    rubric: {config['rubric_name']}")
    typer.echo(f"    success_threshold: {config['success_threshold']}")
    typer.echo(f"    frame_selection: {config['frame_selection']}")
    typer.echo("  metrics:")
    typer.echo(f"    accuracy: {metrics['accuracy']}")
    typer.echo(f"    agreement: {metrics['agreement']}")
    typer.echo(f"    precision: {_format_metric(metrics['precision'])}")
    typer.echo(f"    recall: {_format_metric(metrics['recall'])}")
    typer.echo(f"    f1: {_format_metric(metrics['f1'])}")
    typer.echo(
        "    confusion: "
        f"tp={metrics['true_positives']} tn={metrics['true_negatives']} "
        f"fp={metrics['false_positives']} fn={metrics['false_negatives']}"
    )


def _format_metric(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _parse_csv(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise VlmEvalError("comma-separated option must include at least one value")
    return values


def _parse_thresholds(value: str) -> list[float]:
    thresholds: list[float] = []
    for raw_threshold in _parse_csv(value):
        try:
            thresholds.append(float(raw_threshold))
        except ValueError as exc:
            raise VlmEvalError(f"invalid threshold: {raw_threshold}") from exc
    return thresholds


def _env_dry_run() -> bool:
    return os.environ.get("NPA_DRY_RUN", "").lower() in {
        "1",
        "true",
        "yes",
    } or os.environ.get("DRY_RUN", "").lower() in {"1", "true", "yes"}


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(1)
