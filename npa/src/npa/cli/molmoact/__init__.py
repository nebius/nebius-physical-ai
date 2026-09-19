"""npa molmoact - MolmoAct vision-language-action policy commands.

Finetune, serve, and evaluate MolmoAct VLA policies.  The commands share the
Typer app and `_cmd` naming conventions of the other workbench CLIs so the
``npa.workbench.molmoact`` wrappers can call them through the SDK.

Current status: scaffolding.  Each command validates its arguments and prints
a clear status line; actual trainer/server/evaluator execution is not wired
up yet and returns a stub run manifest instead of failing silently.
"""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console

app = typer.Typer(
    name="molmoact",
    help=(
        "MolmoAct VLA: validate fine-tune/serve/eval configs "
        "(planning only; execution not implemented)."
    ),
    no_args_is_help=True,
)

console = Console(stderr=True)

DEFAULT_MODEL_ID = "allenai/MolmoAct-7B-O-0812"


def _require_model_id(model_id: str) -> str:
    model_id = (model_id or "").strip()
    if not model_id:
        raise typer.BadParameter("model id must not be empty")
    if "/" not in model_id:
        raise typer.BadParameter(
            f"model id {model_id!r} does not look like a HF repo id (expected org/name)"
        )
    return model_id


def _require_s3_uri(uri: str, name: str) -> str:
    uri = (uri or "").strip()
    if not uri:
        raise typer.BadParameter(f"{name} is required")
    if not uri.startswith("s3://"):
        raise typer.BadParameter(f"{name} must be an s3:// URI, got {uri!r}")
    return uri


def _status(command: str, details: dict[str, Any]) -> dict[str, Any]:
    """Print a stub status line and return the run manifest."""
    summary = ", ".join(f"{k}={v}" for k, v in details.items())
    console.print(
        f"[bold]molmoact {command}[/bold]: validated inputs ({summary}); "
        "execution is not wired up yet - stub run manifest returned."
    )
    return {"status": "stub", "command": command, **details}


@app.command("finetune")
def finetune_cmd(
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id",
        help="Base MolmoAct model id (HF repo id).",
    ),
    dataset_uri: str = typer.Option(
        "",
        "--dataset-uri",
        help="S3 URI of the fine-tuning dataset.",
    ),
    output_s3_uri: str = typer.Option(
        "",
        "--output-s3-uri",
        help="S3 URI where the fine-tuned checkpoint is written.",
    ),
    max_steps: int = typer.Option(5000, "--max-steps", help="Maximum training steps."),
    batch_size: int = typer.Option(
        32, "--batch-size", help="Global training batch size."
    ),
    learning_rate: float = typer.Option(
        1e-5, "--learning-rate", help="Peak learning rate."
    ),
    num_gpus: int = typer.Option(
        1, "--num-gpus", help="Number of GPUs for fine-tuning."
    ),
    run_name: str = typer.Option(
        "", "--run-name", help="Run identifier recorded in the training manifest."
    ),
) -> dict[str, Any]:
    """Fine-tune a MolmoAct policy on a demonstration dataset."""
    if max_steps <= 0:
        raise typer.BadParameter("--max-steps must be positive")
    if batch_size <= 0:
        raise typer.BadParameter("--batch-size must be positive")
    if num_gpus <= 0:
        raise typer.BadParameter("--num-gpus must be positive")
    model_id = _require_model_id(model_id)
    dataset_uri = _require_s3_uri(dataset_uri, "dataset-uri")
    output_s3_uri = _require_s3_uri(output_s3_uri, "output-s3-uri")
    return _status(
        "finetune",
        {
            "model_id": model_id,
            "dataset_uri": dataset_uri,
            "output_s3_uri": output_s3_uri,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_gpus": num_gpus,
            "run_name": run_name or "molmoact-finetune",
        },
    )


@app.command("serve")
def serve_cmd(
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id",
        help="MolmoAct model id (HF repo id) or checkpoint path to serve.",
    ),
    checkpoint: str = typer.Option(
        "",
        "--checkpoint",
        help="Optional fine-tuned checkpoint path or S3 URI; defaults to the base model.",
    ),
    host: str = typer.Option("0.0.0.0", "--host", help="HTTP bind host."),
    port: int = typer.Option(8000, "--port", help="HTTP bind port."),
    device: str = typer.Option("cuda", "--device", help="Torch device for inference."),
) -> dict[str, Any]:
    """Serve a MolmoAct policy behind an HTTP endpoint."""
    if not (1 <= port <= 65535):
        raise typer.BadParameter("--port must be in [1, 65535]")
    if device not in {"cuda", "cpu"}:
        raise typer.BadParameter(
            f"unsupported --device {device!r} (expected cuda or cpu)"
        )
    model_id = _require_model_id(model_id)
    return _status(
        "serve",
        {
            "model_id": model_id,
            "checkpoint": checkpoint or model_id,
            "host": host,
            "port": port,
            "device": device,
        },
    )


@app.command("eval")
def eval_cmd(
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id",
        help="MolmoAct model id (HF repo id) under evaluation.",
    ),
    dataset_uri: str = typer.Option(
        "",
        "--dataset-uri",
        help="S3 URI of the evaluation dataset.",
    ),
    checkpoint: str = typer.Option(
        "",
        "--checkpoint",
        help="Checkpoint path or S3 URI to evaluate; defaults to the base model.",
    ),
    num_episodes: int = typer.Option(
        50, "--num-episodes", help="Number of evaluation episodes."
    ),
    output_s3_uri: str = typer.Option(
        "",
        "--output-s3-uri",
        help="S3 URI where evaluation results are written.",
    ),
) -> dict[str, Any]:
    """Evaluate a MolmoAct policy on an evaluation dataset."""
    if num_episodes <= 0:
        raise typer.BadParameter("--num-episodes must be positive")
    model_id = _require_model_id(model_id)
    dataset_uri = _require_s3_uri(dataset_uri, "dataset-uri")
    output_s3_uri = _require_s3_uri(output_s3_uri, "output-s3-uri")
    return _status(
        "eval",
        {
            "model_id": model_id,
            "checkpoint": checkpoint or model_id,
            "dataset_uri": dataset_uri,
            "num_episodes": num_episodes,
            "output_s3_uri": output_s3_uri,
        },
    )
