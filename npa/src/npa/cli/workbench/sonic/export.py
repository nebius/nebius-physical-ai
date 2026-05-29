"""SONIC ONNX export command."""

from __future__ import annotations

from enum import Enum
from dataclasses import asdict

import typer

from npa.cli.workbench.sonic.helpers import OutputFormat, fail, output
from npa.workbench.sonic import (
    DEFAULT_EXPORT_OPSET,
    DEFAULT_METADATA_MODE,
    DEFAULT_NORMALIZE_MODE,
    SonicExportError,
    export_onnx,
)


class AxesMode(str, Enum):
    dynamic = "dynamic"
    static = "static"


class NormalizeMode(str, Enum):
    baked = "baked"
    sidecar = "sidecar"
    none = "none"


class MetadataMode(str, Enum):
    sidecar = "sidecar"
    embedded = "embedded"


def export_cmd(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Trained SONIC policy checkpoint path."),
    output_path: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="ONNX file path or output directory. Directories receive sonic_policy.onnx.",
    ),
    opset: int = typer.Option(
        DEFAULT_EXPORT_OPSET,
        "--opset",
        help="ONNX opset version. Default: 17.",
    ),
    axes: AxesMode = typer.Option(
        AxesMode.dynamic,
        "--axes",
        help="Batch axis mode. Default: dynamic.",
    ),
    normalize: NormalizeMode = typer.Option(
        NormalizeMode(DEFAULT_NORMALIZE_MODE),
        "--normalize",
        help="Observation normalization placement. Default: baked.",
    ),
    metadata: MetadataMode = typer.Option(
        MetadataMode(DEFAULT_METADATA_MODE),
        "--metadata",
        help="Metadata placement. Default: sidecar JSON.",
    ),
    obs_spec: str = typer.Option(
        "",
        "--obs-spec",
        help="YAML/JSON observation layout spec with ordering, shape, units, and fields.",
    ),
    action_spec: str = typer.Option(
        "",
        "--action-spec",
        help="YAML/JSON action layout spec with ordering, shape, units, and fields.",
    ),
    config: str = typer.Option(
        "",
        "--config",
        help="SONIC training/export config used to infer layout, normalization, and control dt.",
    ),
    verify: bool = typer.Option(
        False,
        "--verify/--no-verify",
        help="Run ONNX Runtime parity check against the PyTorch policy after export.",
    ),
    parity_atol: float = typer.Option(
        1e-4,
        "--parity-atol",
        help="Absolute tolerance for --verify parity check. Default: 1e-4.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--output-format",
        help="CLI result format.",
    ),
) -> None:
    """Export a SONIC locomotion policy to deterministic-action ONNX."""

    try:
        result = export_onnx(
            checkpoint=checkpoint,
            output=output_path,
            opset=opset,
            axes=axes.value,
            normalize=normalize.value,
            metadata=metadata.value,
            obs_spec=obs_spec or None,
            action_spec=action_spec or None,
            config=config or None,
            verify=verify,
            parity_atol=parity_atol,
        )
    except SonicExportError as exc:
        fail(str(exc))

    output(asdict(result), output_format)
