"""Typer CLI for `npa workbench cosmos-curate`.

Exposes NVIDIA Cosmos Curator (Apache-2.0,
https://github.com/nvidia-cosmos/cosmos-curate) as workbench commands:
``curate-augmented`` curates a Physical AI Data Factory run's augmented variants,
``curate-videos`` curates a local directory, ``plan-pipeline`` prints upstream's
container command, and ``engine`` reports what this environment can run.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

import typer

app = typer.Typer(
    name="cosmos-curate",
    help="NVIDIA Cosmos Curator: split, transcode, motion-score, and catalog video clips.",
    no_args_is_help=True,
)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class MotionFilter(str, Enum):
    disable = "disable"
    score_only = "score-only"
    enable = "enable"


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


def _emit(payload: dict[str, Any], *, output: OutputFormat, text: str) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(text)


@app.command("curate-augmented")
def curate_augmented_cmd(
    augment_uri: str = typer.Option(
        ..., "--augment-uri", "--input-path", help="Augmented-variant prefix (cosmos_augmented/)."
    ),
    curated_uri: str = typer.Option(
        ..., "--curated-uri", "--output-path", help="Prefix for the curator's output tree."
    ),
    report_uri: str = typer.Option("", "--report-uri", help="Where to write the curation report."),
    clip_len_s: float = typer.Option(10.0, "--clip-len-s", help="Fixed-stride clip length in seconds."),
    min_clip_length_s: float = typer.Option(2.0, "--min-clip-length-s", help="Drop clips shorter than this."),
    motion_filter: MotionFilter = typer.Option(
        MotionFilter.score_only, "--motion-filter", help="Motion stage mode: score only, filter, or skip."
    ),
    limit_clips: int = typer.Option(0, "--limit-clips", help="Clips per input video (0 = no limit)."),
    max_variants: int = typer.Option(0, "--max-variants", help="Curate at most this many variants (0 = all)."),
    require_curator: bool = typer.Option(
        False, "--require-curator", help="Fail instead of reporting 'unavailable' when the curator cannot run."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Log upstream stage progress."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Curate a run's augmented variants with the real Cosmos Curator stages."""

    from npa.workbench.cosmos_curate import CosmosCurateError, curate_augmented, write_report

    try:
        report = curate_augmented(
            augment_uri=augment_uri,
            curated_uri=curated_uri,
            report_uri=report_uri,
            clip_len_s=clip_len_s,
            min_clip_length_s=min_clip_length_s,
            motion_filter=motion_filter.value,
            limit_clips=limit_clips,
            max_variants=max_variants,
            require_curator=require_curator,
            verbose=verbose,
        )
    except CosmosCurateError as exc:
        _fail(str(exc))
        return

    payload = report.to_dict()
    payload["written_uri"] = write_report(payload, result_uri=report.result_uri)
    _emit(
        payload,
        output=output,
        text=(
            f"engine={report.engine} variants={report.variant_count} clips={report.clip_count} "
            f"filtered={report.filtered_count} -> {payload['written_uri']}"
        ),
    )


@app.command("curate-videos")
def curate_videos_cmd(
    input_dir: str = typer.Option(..., "--input-dir", help="Local directory of input videos."),
    output_dir: str = typer.Option(..., "--output-dir", help="Local directory for the curator output tree."),
    clip_len_s: float = typer.Option(10.0, "--clip-len-s", help="Fixed-stride clip length in seconds."),
    min_clip_length_s: float = typer.Option(2.0, "--min-clip-length-s", help="Drop clips shorter than this."),
    motion_filter: MotionFilter = typer.Option(
        MotionFilter.score_only, "--motion-filter", help="Motion stage mode: score only, filter, or skip."
    ),
    limit_clips: int = typer.Option(0, "--limit-clips", help="Clips per input video (0 = no limit)."),
    verbose: bool = typer.Option(False, "--verbose", help="Log upstream stage progress."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Run the curator stages over a local directory of videos."""

    from npa.workbench.cosmos_curate import CosmosCurateError, curate_videos, ingest_output

    try:
        run = curate_videos(
            input_dir=input_dir,
            output_dir=output_dir,
            clip_len_s=clip_len_s,
            min_clip_length_s=min_clip_length_s,
            limit_clips=limit_clips,
            motion_filter=motion_filter.value,
            verbose=verbose,
        )
    except CosmosCurateError as exc:
        _fail(str(exc))
        return
    payload = run.to_dict()
    payload["ingested_clips"] = len(ingest_output(output_dir)["clips"])
    _emit(
        payload,
        output=output,
        text=(
            f"engine={run.engine} videos={run.input_videos} clips={run.clips_written} "
            f"filtered={run.clips_filtered} encoder={run.encoder}"
        ),
    )


@app.command("plan-pipeline")
def plan_pipeline_cmd(
    input_video_path: str = typer.Option(..., "--input-video-path", help="Upstream --input-video-path value."),
    output_clip_path: str = typer.Option(..., "--output-clip-path", help="Upstream --output-clip-path value."),
    splitting_algorithm: str = typer.Option(
        "fixed-stride", "--splitting-algorithm", help="fixed-stride or transnetv2."
    ),
    captioning_algorithm: str = typer.Option("", "--captioning-algorithm", help="Upstream captioning algorithm."),
    embedding_algorithm: str = typer.Option("", "--embedding-algorithm", help="Upstream embedding algorithm."),
    generate_embeddings: bool = typer.Option(False, "--generate-embeddings", help="Keep embedding generation on."),
    limit: int = typer.Option(0, "--limit", help="Upstream --limit value."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Print upstream's `video-pipeline split` command for the curator container."""

    from npa.workbench.cosmos_curate import CosmosCurateError, split_pipeline_argv

    try:
        argv = split_pipeline_argv(
            input_video_path=input_video_path,
            output_clip_path=output_clip_path,
            splitting_algorithm=splitting_algorithm,
            captioning_algorithm=captioning_algorithm,
            embedding_algorithm=embedding_algorithm,
            generate_embeddings=generate_embeddings,
            limit=limit,
        )
    except CosmosCurateError as exc:
        _fail(str(exc))
        return
    _emit({"argv": argv}, output=output, text=" ".join(argv))


@app.command("fetch-models")
def fetch_models_cmd(
    models: list[str] = typer.Option(
        [],
        "--models",
        "-m",
        help="Model set or upstream model key; repeatable. Default: the split-annotate set.",
    ),
    force: bool = typer.Option(False, "--force", help="Re-download models that are already complete."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Download curator model weights with your own Hugging Face token.

    The image ships no model weights: TransNetV2, InternVideo2, Cosmos-Embed1,
    CLIP, and Qwen are third-party or NVIDIA models under their own licenses. The
    model ids and their pinned revisions come from upstream's own registry.
    """

    from npa.workbench.cosmos_curate import CosmosCurateError, fetch_models

    try:
        result = fetch_models(models, force=force)
    except CosmosCurateError as exc:
        _fail(str(exc))
        return
    payload = result.to_dict()
    _emit(
        payload,
        output=output,
        text=(
            f"status={result.status} fetched={len(result.fetched)} "
            f"present={len(result.already_present)} failed={len(result.failed)} "
            f"-> {result.weights_dir}"
        ),
    )
    if result.failed:
        raise typer.Exit(1)


@app.command("models")
def models_cmd(
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Show the curator model sets, their upstream pins, and what is present."""

    from npa.workbench.cosmos_curate import describe_models

    payload = describe_models()
    lines = [
        f"weights_dir={payload['weights_dir']} hf_token={payload['hf_token_present']} "
        f"ngc_key={payload.get('ngc_key_present')}"
    ]
    for name, entry in sorted(payload.get("sets", {}).items()):
        present = sum(1 for model in entry["models"] if model["present"])
        lines.append(f"{name}: {present}/{len(entry['models'])} present ({', '.join(entry['keys'])})")
    if payload.get("error"):
        lines.append(f"error: {payload['error']}")
    _emit(payload, output=output, text="\n".join(lines))


@app.command("engine")
def engine_cmd(
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Report whether the upstream curator can run in this environment."""

    from npa.workbench.cosmos_curate import probe_availability

    payload = probe_availability().to_dict()
    _emit(
        payload,
        output=output,
        text=(
            f"can_run_in_process={payload['can_run_in_process']} source={payload['source'] or '(none)'} "
            f"encoder={payload['encoder'] or '(none)'} {payload['reason']}"
        ),
    )
