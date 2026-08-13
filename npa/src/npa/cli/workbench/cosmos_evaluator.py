"""Typer CLI for `npa workbench cosmos-evaluator`.

Exposes the NVIDIA Cosmos Evaluator checks (Apache-2.0,
https://github.com/nvidia-cosmos/cosmos-evaluator) as workbench commands:
``evaluate`` grades a whole Physical AI Data Factory run, while
``hallucination`` and ``attribute-verify`` run a single check against one clip.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

import typer

app = typer.Typer(
    name="cosmos-evaluator",
    help=(
        "Cosmos Evaluator checks plus NPA source-relative temporal and "
        "protected-appearance diagnostics."
    ),
    no_args_is_help=True,
)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class TemporalMode(str, Enum):
    advisory = "advisory"
    required = "required"


class AppearanceMode(str, Enum):
    advisory = "advisory"
    required = "required"


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


def _emit(payload: dict[str, Any], *, output: OutputFormat, text: str) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(text)


@app.command("evaluate")
def evaluate_cmd(
    augment_uri: str = typer.Option(
        ...,
        "--augment-uri",
        "--input-path",
        help="Augmented-variant prefix (cosmos_augmented/).",
    ),
    output_uri: str = typer.Option(
        ...,
        "--output-uri",
        "--output-path",
        help="Prefix (or .json path) for the evaluator report.",
    ),
    input_uri: str = typer.Option(
        "",
        "--input-uri",
        help="Run input prefix; variants match their recorded conditioned-input filename.",
    ),
    configs_uri: str = typer.Option(
        "",
        "--configs-uri",
        help="Config prefix; its manifest supplies the attribute option table.",
    ),
    original_video: str = typer.Option(
        "",
        "--original-video",
        help="Explicit original clip for the hallucination check.",
    ),
    threshold: float = typer.Option(
        0.682, "--threshold", help="Pass threshold for the run score."
    ),
    hallucination_weight: float = typer.Option(
        0.5,
        "--hallucination-weight",
        help="Weight on the hallucination score for input-conditioned variants.",
    ),
    temporal_threshold: float = typer.Option(
        0.8,
        "--temporal-threshold",
        help="Diagnostic temporal score threshold; enforced only with --temporal-mode required.",
    ),
    temporal_regions_json: str = typer.Option(
        "",
        "--temporal-regions-json",
        help="Optional JSON list of normalized rectangular regions; default is full frame plus a 2x2 grid.",
    ),
    temporal_mode: TemporalMode = typer.Option(
        TemporalMode.advisory,
        "--temporal-mode",
        help="Keep temporal consistency advisory (default) or require it as a calibrated hard check.",
    ),
    temporal_noise_floor: float = typer.Option(
        0.25,
        "--temporal-noise-floor",
        help="Fixed mean grayscale-acceleration residual treated as codec/capture noise.",
    ),
    temporal_blur_ksize: int = typer.Option(
        7,
        "--temporal-blur-ksize",
        help="Positive odd Gaussian pre-filter kernel used before temporal comparison.",
    ),
    appearance_threshold: float = typer.Option(
        0.8,
        "--appearance-threshold",
        help="Protected-appearance score threshold; enforced only with --appearance-mode required.",
    ),
    appearance_regions_json: str = typer.Option(
        "",
        "--appearance-regions-json",
        help="Optional JSON list of normalized protected regions; default is full frame plus a 2x2 grid.",
    ),
    appearance_mode: AppearanceMode = typer.Option(
        AppearanceMode.advisory,
        "--appearance-mode",
        help="Keep protected-appearance fidelity advisory (default) or require it as a hard check.",
    ),
    appearance_luminance_tolerance: float = typer.Option(
        18.0,
        "--appearance-luminance-tolerance",
        help="Allowed p95 CIELAB luminance drift before the appearance score falls.",
    ),
    appearance_global_chroma_tolerance: float = typer.Option(
        8.0,
        "--appearance-global-chroma-tolerance",
        help="Allowed p95 scene-wide CIELAB chroma drift.",
    ),
    appearance_local_chroma_tolerance: float = typer.Option(
        6.0,
        "--appearance-local-chroma-tolerance",
        help="Allowed p95 regional chroma residual after scene-wide shift.",
    ),
    appearance_chroma_instability_tolerance: float = typer.Option(
        4.0,
        "--appearance-chroma-instability-tolerance",
        help="Allowed p95 frame-to-frame change in regional chroma shift.",
    ),
    appearance_blur_ksize: int = typer.Option(
        7,
        "--appearance-blur-ksize",
        help="Positive odd Gaussian pre-filter kernel used before appearance comparison.",
    ),
    appearance_max_dimension: int = typer.Option(
        256,
        "--appearance-max-dimension",
        help="Longest decoded frame edge used by the appearance comparison.",
    ),
    question_model: str = typer.Option(
        "", "--question-model", help="Token Factory LLM for question generation."
    ),
    vlm_model: str = typer.Option(
        "", "--vlm-model", help="Token Factory VLM that answers the questions."
    ),
    max_clips: int = typer.Option(
        0, "--max-clips", help="Grade at most this many variants (0 = all)."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.json, "--output", help="Output format."
    ),
) -> None:
    """Grade every augmented variant of a run and write one evaluator report."""

    from npa.workbench.cosmos_evaluator import CosmosEvaluatorError, evaluate_run
    from npa.workbench.cosmos_evaluator.evaluate import write_report

    try:
        result = evaluate_run(
            augment_uri=augment_uri,
            output_uri=output_uri,
            input_uri=input_uri,
            configs_uri=configs_uri,
            original_video=original_video,
            threshold=threshold,
            hallucination_weight=hallucination_weight,
            temporal_threshold=temporal_threshold,
            temporal_regions_json=temporal_regions_json,
            temporal_mode=temporal_mode.value,
            temporal_noise_floor=temporal_noise_floor,
            temporal_blur_ksize=temporal_blur_ksize,
            appearance_threshold=appearance_threshold,
            appearance_regions_json=appearance_regions_json,
            appearance_mode=appearance_mode.value,
            appearance_luminance_tolerance=appearance_luminance_tolerance,
            appearance_global_chroma_tolerance=appearance_global_chroma_tolerance,
            appearance_local_chroma_tolerance=appearance_local_chroma_tolerance,
            appearance_chroma_instability_tolerance=appearance_chroma_instability_tolerance,
            appearance_blur_ksize=appearance_blur_ksize,
            appearance_max_dimension=appearance_max_dimension,
            question_model=question_model,
            vlm_model=vlm_model,
            max_clips=max_clips,
        )
    except CosmosEvaluatorError as exc:
        _fail(str(exc))
        return

    payload = result.to_dict()
    payload["written_uri"] = write_report(payload, result_uri=result.result_uri)
    _emit(
        payload,
        output=output,
        text=(
            f"score={result.score} passed={result.passed} "
            f"clips={result.clip_count} passed_clips={result.passed_clips} -> {payload['written_uri']}"
        ),
    )


@app.command("hallucination")
def hallucination_cmd(
    original_video: str = typer.Option(..., "--original-video", help="Path to the original clip."),
    augmented_video: str = typer.Option(..., "--augmented-video", help="Path to the augmented clip."),
    clip_id: str = typer.Option("clip", "--clip-id", help="Clip id recorded in the result."),
    threshold: float = typer.Option(0.682, "--threshold", help="Pass threshold for the hallucination score."),
    max_frames: int = typer.Option(0, "--max-frames", help="Stop after this many frame pairs (0 = all)."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Score hallucinated motion in one augmented clip."""

    from npa.workbench.cosmos_evaluator import CosmosEvaluatorError, check_hallucination

    try:
        result = check_hallucination(
            clip_id=clip_id,
            original_video=original_video,
            augmented_video=augmented_video,
            threshold=threshold,
            max_frames=max_frames or None,
        )
    except CosmosEvaluatorError as exc:
        _fail(str(exc))
        return
    payload = result.to_dict()
    _emit(
        payload,
        output=output,
        text=f"score={result.score} passed={result.passed} engine={result.engine} frames={result.total_frames}",
    )


@app.command("attribute-verify")
def attribute_verify_cmd(
    video: str = typer.Option("", "--video", help="Augmented clip to verify."),
    frame: str = typer.Option("", "--frame", help="Still frame to verify instead of a clip."),
    variables: str = typer.Option(
        ..., "--variables", help='JSON object of attribute -> requested value, e.g. \'{"lighting": "dim evening light"}\'.'
    ),
    options: str = typer.Option(
        "", "--options", help="JSON object of attribute -> list of all possible values."
    ),
    clip_id: str = typer.Option("clip", "--clip-id", help="Clip id recorded in the result."),
    question_model: str = typer.Option("", "--question-model", help="Token Factory LLM for question generation."),
    vlm_model: str = typer.Option("", "--vlm-model", help="Token Factory VLM that answers the questions."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Verify one clip's augmented attributes with an LLM + VLM question pass."""

    from npa.workbench.cosmos_evaluator import CosmosEvaluatorError, verify_attributes

    try:
        selected = json.loads(variables)
        option_table = json.loads(options) if options else None
    except json.JSONDecodeError as exc:
        _fail(f"--variables/--options must be JSON objects: {exc}")
        return
    if not isinstance(selected, dict):
        _fail("--variables must be a JSON object of attribute -> value")
        return

    try:
        result = verify_attributes(
            clip_id=clip_id,
            video=video or None,
            frame=frame or None,
            selected_variables={str(key): str(value) for key, value in selected.items()},
            variable_options=option_table,
            question_model=question_model,
            vlm_model=vlm_model,
        )
    except CosmosEvaluatorError as exc:
        _fail(str(exc))
        return
    payload = result.to_dict()
    _emit(
        payload,
        output=output,
        text=(
            f"score={result.score} passed={result.passed} "
            f"checks={result.passed_checks}/{result.total_checks}"
        ),
    )


@app.command("engine")
def engine_cmd(
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Report which evaluator engine this environment resolves to."""

    from npa.workbench.cosmos_evaluator.evaluate import evaluator_engine_summary

    payload = evaluator_engine_summary()
    _emit(payload, output=output, text=f"engine={payload['engine']} source={payload['upstream_source'] or '(none)'}")
