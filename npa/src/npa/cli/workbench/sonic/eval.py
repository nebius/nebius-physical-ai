"""SONIC ONNX eval command."""

from __future__ import annotations

from enum import Enum

import typer

from npa.cli.workbench.sonic.helpers import OutputFormat, fail, output
from npa.workbench.sonic.eval import (
    CONTAINER_BACKEND,
    DEFAULT_CONTAINER_METADATA_PATH,
    DEFAULT_CONTAINER_OUTPUT_PATH,
    DEFAULT_CONTAINER_POLICY_PATH,
    DEFAULT_CONTAINER_RUNTIME,
    DEFAULT_EVAL_ENV,
    DEFAULT_EVAL_OUTPUT_NAME,
    REFERENCE_BACKEND,
    SonicEvalError,
    evaluate_onnx_policy,
)


class EvalBackend(str, Enum):
    reference = REFERENCE_BACKEND
    container = CONTAINER_BACKEND


def eval_cmd(
    onnx_path: str = typer.Option(
        ...,
        "--onnx",
        help="Exported SONIC ONNX policy path.",
    ),
    metadata_path: str = typer.Option(
        "",
        "--metadata",
        "--sidecar",
        help="SONIC export sidecar metadata JSON. Defaults to <onnx>.metadata.json.",
    ),
    backend: EvalBackend = typer.Option(
        EvalBackend.reference,
        "--backend",
        help="Eval backend. Default: reference.",
    ),
    episodes: int = typer.Option(
        8,
        "--episodes",
        help="Evaluation episode count.",
    ),
    env: str = typer.Option(
        DEFAULT_EVAL_ENV,
        "--env",
        help="Reference simulator env name. Use smoke for the built-in smoke rollout.",
    ),
    container_image: str = typer.Option(
        "",
        "--container-image",
        help="Eval container image for --backend container.",
    ),
    container_runtime: str = typer.Option(
        DEFAULT_CONTAINER_RUNTIME,
        "--container-runtime",
        help="Container runtime command for --backend container.",
    ),
    container_policy_path: str = typer.Option(
        DEFAULT_CONTAINER_POLICY_PATH,
        "--container-policy-path",
        help="Path where the container reads the ONNX policy.",
    ),
    container_metadata_path: str = typer.Option(
        DEFAULT_CONTAINER_METADATA_PATH,
        "--container-metadata-path",
        help="Path where the container reads the sidecar metadata.",
    ),
    container_output_path: str = typer.Option(
        DEFAULT_CONTAINER_OUTPUT_PATH,
        "--container-output-path",
        help="Path where the container writes eval-result JSON.",
    ),
    container_arg: list[str] | None = typer.Option(
        None,
        "--container-arg",
        help="Additional argument appended after the container image.",
    ),
    output_path: str = typer.Option(
        DEFAULT_EVAL_OUTPUT_NAME,
        "--output",
        "-o",
        help="Local JSON file/path or s3:// target for eval results.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--output-format",
        help="CLI result format.",
    ),
) -> None:
    """Evaluate an exported SONIC ONNX locomotion policy."""

    try:
        result = evaluate_onnx_policy(
            onnx=onnx_path,
            metadata=metadata_path or None,
            backend=backend.value,
            episodes=episodes,
            env=env,
            output=output_path,
            container_image=container_image,
            container_runtime=container_runtime,
            container_policy_path=container_policy_path,
            container_metadata_path=container_metadata_path,
            container_output_path=container_output_path,
            container_args=container_arg or [],
        )
    except SonicEvalError as exc:
        fail(str(exc))

    output(result, output_format)
