"""npa openvla — OpenVLA / OpenVLA-OFT training, serving, and evaluation commands.

Stub implementation for issue #500: validates arguments and prints a clear
execution plan. Heavy work (LoRA fine-tuning via the upstream OpenVLA-OFT
recipe, checkpoint serving, evaluation rollouts) is delegated to
``npa.workflows.byof.openvla_pipeline`` or a GPU workbench at run time.
"""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="openvla",
    help="OpenVLA: OFT fine-tuning, checkpoint serving, evaluation.",
    no_args_is_help=True,
)

console = Console()

DEFAULT_MODEL_ID = "openvla/openvla-7b"
UPSTREAM_REPO = "https://github.com/openvla/openvla"


def _fail(msg: str, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {msg}")
    raise typer.Exit(code)


def _require_non_empty(value: str, flag: str) -> str:
    if not value or not value.strip():
        _fail(f"{flag} must not be empty")
    return value.strip()


@app.command("train")
def train_cmd(
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id",
        help="Base OpenVLA model: HuggingFace Hub id or local path.",
    ),
    dataset_uri: str = typer.Option(
        ...,
        "--dataset-uri",
        help="Fine-tuning dataset URI (s3://, gs://, or local dir) in LeRobot/RLDS layout.",
    ),
    dataset_name: str = typer.Option(
        "finetune",
        "--dataset-name",
        help="Dataset name inside the data root dir (upstream --dataset_name).",
    ),
    output_dir: str = typer.Option(
        "runs/openvla-oft",
        "--output-dir",
        help="Run root dir for logs and checkpoints (upstream --run_root_dir).",
    ),
    batch_size: int = typer.Option(16, "--batch-size", help="Fine-tuning batch size."),
    max_steps: int = typer.Option(
        200_000, "--max-steps", help="Max fine-tuning steps."
    ),
    learning_rate: float = typer.Option(
        5e-4, "--learning-rate", help="Fine-tuning learning rate."
    ),
    lora_rank: int = typer.Option(32, "--lora-rank", help="LoRA rank."),
    lora_dropout: float = typer.Option(0.0, "--lora-dropout", help="LoRA dropout."),
    image_aug: bool = typer.Option(
        True, "--image-aug/--no-image-aug", help="Train with image augmentations."
    ),
    seed: int = typer.Option(7, "--seed", help="Random seed."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate arguments and print the plan without training.",
    ),
) -> None:
    """Fine-tune OpenVLA with the OpenVLA-OFT LoRA recipe (stub)."""
    _require_non_empty(model_id, "--model-id")
    _require_non_empty(dataset_uri, "--dataset-uri")
    _require_non_empty(output_dir, "--output-dir")
    if batch_size <= 0:
        _fail("--batch-size must be > 0")
    if max_steps <= 0:
        _fail("--max-steps must be > 0")
    if learning_rate <= 0:
        _fail("--learning-rate must be > 0")
    if lora_rank <= 0:
        _fail("--lora-rank must be > 0")
    if not 0.0 <= lora_dropout < 1.0:
        _fail("--lora-dropout must be in [0, 1)")

    console.print("[bold]OpenVLA-OFT fine-tuning plan[/bold]")
    console.print(f"  model:        {model_id}")
    console.print(f"  dataset:      {dataset_uri} (name: {dataset_name})")
    console.print(f"  output:       {output_dir}")
    console.print(
        f"  hyperparams:  batch_size={batch_size} max_steps={max_steps} "
        f"lr={learning_rate} lora_rank={lora_rank} lora_dropout={lora_dropout} "
        f"image_aug={image_aug} seed={seed}"
    )
    console.print(f"  upstream:     {UPSTREAM_REPO} vla-scripts/finetune.py")
    if dry_run:
        console.print("[yellow]dry-run:[/yellow] training not started.")
        return
    console.print(
        "[yellow]stub:[/yellow] real training runs on a GPU workbench via "
        "`npa.workflows.byof.openvla_pipeline train`."
    )


@app.command("serve")
def serve_cmd(
    checkpoint: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--checkpoint",
        help="OpenVLA checkpoint: HuggingFace Hub id or local run dir.",
    ),
    host: str = typer.Option("0.0.0.0", "--host", help="Host IP address."),
    port: int = typer.Option(8000, "--port", help="Host port."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate arguments and print the plan without serving.",
    ),
) -> None:
    """Serve an OpenVLA checkpoint over HTTP (stub)."""
    _require_non_empty(checkpoint, "--checkpoint")
    _require_non_empty(host, "--host")
    if not 1 <= port <= 65535:
        _fail("--port must be in [1, 65535]")

    console.print("[bold]OpenVLA serve plan[/bold]")
    console.print(f"  checkpoint: {checkpoint}")
    console.print(f"  endpoint:   http://{host}:{port}")
    console.print(f"  upstream:   {UPSTREAM_REPO} vla-scripts/deploy.py")
    if dry_run:
        console.print("[yellow]dry-run:[/yellow] server not started.")
        return
    console.print(
        "[yellow]stub:[/yellow] real serving runs on a GPU workbench via "
        "`npa.workflows.byof.openvla_pipeline serve`."
    )


@app.command("eval")
def eval_cmd(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        help="Fine-tuned OpenVLA checkpoint: local run dir or HuggingFace Hub id.",
    ),
    dataset_uri: str = typer.Option(
        ...,
        "--dataset-uri",
        help="Evaluation dataset URI (s3://, gs://, or local dir).",
    ),
    num_episodes: int = typer.Option(
        10, "--num-episodes", help="Number of evaluation episodes."
    ),
    output_uri: str = typer.Option(
        "", "--output-uri", help="Where to write the eval results JSON (optional)."
    ),
    seed: int = typer.Option(7, "--seed", help="Random seed."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate arguments and print the plan without evaluating.",
    ),
) -> None:
    """Evaluate an OpenVLA checkpoint (stub)."""
    _require_non_empty(checkpoint, "--checkpoint")
    _require_non_empty(dataset_uri, "--dataset-uri")
    if num_episodes <= 0:
        _fail("--num-episodes must be > 0")

    console.print("[bold]OpenVLA eval plan[/bold]")
    console.print(f"  checkpoint:   {checkpoint}")
    console.print(f"  dataset:      {dataset_uri}")
    console.print(f"  episodes:     {num_episodes} (seed={seed})")
    console.print(f"  output:       {output_uri or '(stdout only)'}")
    if dry_run:
        console.print("[yellow]dry-run:[/yellow] evaluation not started.")
        return
    console.print(
        "[yellow]stub:[/yellow] real evaluation runs against a served checkpoint via "
        "`npa.workflows.byof.openvla_pipeline eval`."
    )
