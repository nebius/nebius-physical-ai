"""Workbench Cosmos3 commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer

from npa.workbench.cosmos.text_to_image import DEFAULT_UV_GROUP
from npa.workbench.cosmos.generate import (
    DEFAULT_MODE,
    DEFAULT_NAME,
    DEFAULT_PARALLELISM_PRESET,
    GENERATE_MODES,
    Cosmos3GenerateError,
    generate_and_publish,
)
from npa.workflows.cosmos_split import (
    Cosmos3ReasonConfig,
    build_cosmos3_reason_manifest,
    write_manifest,
)

app = typer.Typer(
    name="cosmos3",
    help="Cosmos3 omni-model generation and reasoning workflow contracts.",
    no_args_is_help=True,
)


@app.command("checkpoint-eval")
def checkpoint_eval_cmd(
    campaign_config: str = typer.Option(
        ...,
        "--campaign-config",
        help="Local path or s3:// URI for the versioned checkpoint-evaluation config.",
    ),
    phase: str = typer.Option(
        ...,
        "--phase",
        help="Evaluation phase: primary or consistency.",
    ),
    output_uri: str = typer.Option(
        ...,
        "--output-uri",
        help="Durable s3:// campaign root for generated media and evidence.",
    ),
    top_checkpoint: list[str] = typer.Option(
        [],
        "--top-checkpoint",
        help="Consistency only: repeat exactly twice for the selected top two checkpoints.",
    ),
    work_dir: Path = typer.Option(
        Path("/tmp/npa-cosmos3-checkpoint-eval"),
        "--work-dir",
        help="Ephemeral local output and Hugging Face cache root.",
    ),
    run_id: str = typer.Option("", "--run-id", help="Workflow run id carried into evidence."),
    runtime_image: str = typer.Option(
        "",
        "--runtime-image",
        help="Resolved npa-cosmos3 image reference recorded in provenance.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate the config and show arms without probing a GPU or downloading weights.",
    ),
) -> None:
    """Evaluate Cosmos3 text-to-image checkpoints in guarded, load-once batches."""

    from npa.workbench.cosmos.checkpoint_eval import (
        Cosmos3CheckpointEvalError,
        execute_phase,
    )

    try:
        result = execute_phase(
            campaign_config=campaign_config,
            phase=phase,
            output_uri=output_uri,
            top_checkpoints=top_checkpoint,
            work_dir=work_dir,
            run_id=run_id,
            runtime_image=runtime_image.strip()
            or os.environ.get("NPA_TASK_IMAGE", "").strip(),
            dry_run=dry_run,
        )
    except (Cosmos3CheckpointEvalError, Cosmos3GenerateError) as exc:
        typer.echo(f"cosmos3 checkpoint-eval failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("generate")
def generate_cmd(
    prompt: str = typer.Option(..., "--prompt", help="Text prompt driving generation."),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        "--output-uri",
        help="Local output directory or s3:// prefix for the generated artifact.",
    ),
    mode: str = typer.Option(
        DEFAULT_MODE,
        "--mode",
        help=f"Generation mode: {', '.join(GENERATE_MODES)}.",
    ),
    input_path: str = typer.Option(
        "",
        "--input-path",
        "--input-uri",
        help="Conditioning image/video (local path, http(s) URL, or s3:// URI). "
        "Required for image2image, image2video, and video2video.",
    ),
    checkpoint: str = typer.Option(
        "",
        "--checkpoint",
        help="Checkpoint name (e.g. Cosmos3-Nano), local path, or s3:// URI. "
        "Named checkpoints download at runtime with the operator's HF token; a "
        "staged path still needs one unless --no-guardrails is also passed.",
    ),
    name: str = typer.Option(DEFAULT_NAME, "--name", help="Sample name / output subdirectory."),
    negative_prompt: str = typer.Option("", "--negative-prompt", help="Optional negative prompt."),
    seed: int = typer.Option(0, "--seed", help="Sampling seed for reproducible runs."),
    num_steps: int = typer.Option(0, "--num-steps", help="Override the mode's sampling steps."),
    guidance: float = typer.Option(0.0, "--guidance", help="Override classifier-free guidance."),
    no_guardrails: bool = typer.Option(
        False,
        "--no-guardrails",
        help="Disable the Cosmos content-safety guardrails (on by default).",
    ),
    parallelism_preset: str = typer.Option(
        DEFAULT_PARALLELISM_PRESET,
        "--parallelism-preset",
        help="Upstream parallelism preset: latency or throughput.",
    ),
    run_id: str = typer.Option("", "--run-id", help="Run id carried into the manifest."),
    output_json: Optional[Path] = typer.Option(
        None, "--output-json", help="Write the result manifest JSON locally."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Resolve the input sample and inference argv without running the model.",
    ),
) -> None:
    """Run a real Cosmos 3 generation with the omni model.

    Requires the ``npa-cosmos3`` image on a GPU: the container ships the framework
    but no weights, so the checkpoint downloads at runtime under the operator's own
    Hugging Face license acceptance. ``--dry-run`` resolves the plan on any host.
    """

    try:
        payload = generate_and_publish(
            mode=mode,
            prompt=prompt,
            name=name,
            checkpoint=checkpoint,
            input_path=input_path,
            output_path=output_path,
            negative_prompt=negative_prompt,
            seed=seed,
            num_steps=num_steps,
            guidance=guidance,
            no_guardrails=no_guardrails,
            parallelism_preset=parallelism_preset,
            run_id=run_id,
            dry_run=dry_run,
        )
    except Cosmos3GenerateError as exc:
        typer.echo(f"cosmos3 generate failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    if output_json is not None:
        payload = write_manifest(payload, output_json)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("reason")
def reason_cmd(
    input_uri: str = typer.Option(..., "--input-uri", help="Input rollout or frame URI."),
    output_uri: str = typer.Option(..., "--output-uri", help="Output prefix for reasoning JSON."),
    model: str = typer.Option("nvidia/Cosmos-Reason1-7B", "--model", help="Reasoning model id."),
    image: str = typer.Option("", "--image", help="BYO Cosmos3 reason image."),
    prompt: str = typer.Option("", "--prompt", help="Optional reasoning prompt."),
    run_id: str = typer.Option("", "--run-id", help="Run id carried into the manifest."),
    output_json: Optional[Path] = typer.Option(None, "--output-json", help="Write manifest JSON locally."),
) -> None:
    """Build the Cosmos3 reason stage manifest."""

    payload = build_cosmos3_reason_manifest(
        Cosmos3ReasonConfig(
            input_uri=input_uri,
            output_uri=output_uri,
            model=model,
            image=image,
            prompt=prompt,
            run_id=run_id,
        )
    )
    if output_json is not None:
        payload = write_manifest(payload, output_json)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("text-to-image")
def text_to_image_cmd(
    prompt: str = typer.Option(..., "--prompt", help="Text prompt to generate an image from."),
    output_uri: str = typer.Option(
        "", "--output-uri", help="S3 prefix to publish the image and its manifest to."
    ),
    output_dir: Path = typer.Option(
        Path("/tmp/npa-cosmos3-inference"),
        "--output-dir",
        help="Local working directory for inference outputs.",
    ),
    model_id: str = typer.Option("", "--model-id", help="HF model repo id for the checkpoint."),
    checkpoint_name: str = typer.Option(
        "Cosmos3-Nano",
        "--checkpoint-name",
        help="Checkpoint name the framework's inference entrypoint expects.",
    ),
    source_repo_url: str = typer.Option(
        "", "--source-repo-url", help="Cosmos framework source repository URL."
    ),
    cache_dir: Optional[Path] = typer.Option(
        None, "--cache-dir", help="Ephemeral runtime cache for source and checkpoint."
    ),
    uv_group: str = typer.Option(
        DEFAULT_UV_GROUP, "--uv-group", help="uv dependency group to sync in the framework repo."
    ),
    seed: int = typer.Option(0, "--seed", help="Inference seed."),
    guardrails: bool = typer.Option(
        False,
        "--guardrails/--no-guardrails",
        help="Run the framework's content guardrails (they download extra gated weights).",
    ),
    hf_token_env: str = typer.Option(
        "HF_TOKEN", "--hf-token-env", help="Environment variable holding the Hugging Face token."
    ),
    github_token_env: str = typer.Option(
        "GITHUB_TOKEN", "--github-token-env", help="Environment variable holding a GitHub token."
    ),
) -> None:
    """Generate an image from a prompt with the Cosmos3 framework, and publish it.

    Replaces `skypilot/cosmos3-text-to-image-inference.yaml`, which carried the whole procedure
    as bash inside an `envs:` block — unreachable from the CLI or the SDK, and untestable.
    """

    from npa.workbench.cosmos.cosmos3 import Cosmos3AccessConfig
    from npa.workbench.cosmos.text_to_image import Cosmos3TextToImageError, generate

    config = Cosmos3AccessConfig.from_env(
        model_id=model_id,
        source_repo_url=source_repo_url,
        cache_dir=cache_dir,
        github_token_env=github_token_env,
        hf_token_env=hf_token_env,
    )
    try:
        result = generate(
            config,
            prompt=prompt,
            output_dir=output_dir,
            seed=seed,
            guardrails=guardrails,
            uv_group=uv_group,
            checkpoint_name=checkpoint_name,
            publish_uri=output_uri,
        )
    except Cosmos3TextToImageError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result.as_dict(), indent=2, sort_keys=True))
