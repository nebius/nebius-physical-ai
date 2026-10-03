"""OpenVLA-OFT workbench commands backed by native upstream stages."""

from __future__ import annotations

import json

import typer
from rich.console import Console

from npa.workflows.byof import openvla_pipeline as pipeline

app = typer.Typer(
    name="openvla",
    help="OpenVLA-OFT LIBERO preparation, training, rollout, and evidence.",
    no_args_is_help=True,
)
console = Console()


def _run(operation, config) -> None:
    try:
        console.print_json(json.dumps(operation(config), sort_keys=True))
    except pipeline.OpenVLAPipelineError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("prepare")
def prepare_cmd(
    dataset_uri: str = typer.Option(..., "--dataset-uri"),
    dataset_name: str = typer.Option(..., "--dataset-name"),
    task_suite: str = typer.Option(..., "--task-suite"),
    output_uri: str = typer.Option(..., "--output-uri"),
) -> None:
    """Inspect real RLDS data and publish normalization provenance."""
    _run(
        pipeline.prepare,
        pipeline.PrepareConfig(dataset_uri, output_uri, dataset_name, task_suite),
    )


@app.command("train")
def train_cmd(
    prepared_manifest_uri: str = typer.Option(..., "--prepared-manifest-uri"),
    output_uri: str = typer.Option(..., "--output-uri"),
    runtime_root: str = typer.Option(..., "--runtime-root"),
    model_id: str = typer.Option(pipeline.DEFAULT_MODEL_ID, "--model-id"),
    model_revision: str = typer.Option(pipeline.MODEL_REVISION, "--model-revision"),
    batch_size: int = typer.Option(pipeline.DEFAULT_BATCH_SIZE, "--batch-size"),
    max_steps: int = typer.Option(pipeline.DEFAULT_MAX_STEPS, "--max-steps"),
    learning_rate: float = typer.Option(
        pipeline.DEFAULT_LEARNING_RATE, "--learning-rate"
    ),
    lora_rank: int = typer.Option(pipeline.DEFAULT_LORA_RANK, "--lora-rank"),
    processes: int = typer.Option(8, "--processes"),
    seed: int = typer.Option(pipeline.DEFAULT_SEED, "--seed"),
) -> None:
    """Run upstream OFT fine-tuning; no stock-decoder fallback exists."""
    _run(pipeline.train, pipeline.TrainConfig(**locals()))


@app.command("rollout")
def rollout_cmd(
    training_manifest_uri: str = typer.Option(..., "--training-manifest-uri"),
    output_uri: str = typer.Option(..., "--output-uri"),
    runtime_root: str = typer.Option(..., "--runtime-root"),
    task_suite: str = typer.Option(..., "--task-suite"),
    trials_per_task: int = typer.Option(pipeline.DEFAULT_TRIALS, "--trials-per-task"),
    seed: int = typer.Option(pipeline.DEFAULT_SEED, "--seed"),
) -> None:
    """Run upstream closed-loop LIBERO rollouts and retain MP4 evidence."""
    _run(pipeline.rollout, pipeline.RolloutConfig(**locals()))


@app.command("evaluate")
def evaluate_cmd(
    rollout_manifest_uri: str = typer.Option(..., "--rollout-manifest-uri"),
    output_uri: str = typer.Option(..., "--output-uri"),
) -> None:
    """Compute held-out numerical success from verified raw rollout evidence."""
    _run(pipeline.evaluate, pipeline.EvaluateConfig(**locals()))


@app.command("visualize")
def visualize_cmd(
    evaluation_manifest_uri: str = typer.Option(..., "--evaluation-manifest-uri"),
    output_uri: str = typer.Option(..., "--output-uri"),
) -> None:
    """Create CSV/SVG comparison artifacts from verified evaluation metrics."""
    _run(pipeline.visualize, pipeline.VisualizeConfig(**locals()))
