"""SONIC ONNX eval command."""

from __future__ import annotations

from enum import Enum

import typer

from npa.cli.workbench.sonic.helpers import OutputFormat, fail, output
from npa.deploy.images import container_image_for_tool
from npa.workbench.sonic.eval import (
    CONTAINER_BACKEND,
    DEFAULT_CONTAINER_DRIVER_CAPABILITIES,
    DEFAULT_CONTAINER_GLX_VENDOR,
    DEFAULT_CONTAINER_GPUS,
    DEFAULT_CONTAINER_METADATA_PATH,
    DEFAULT_CONTAINER_OUTPUT_PATH,
    DEFAULT_CONTAINER_POLICY_PATH,
    DEFAULT_CONTAINER_RENDER_FRAMES,
    DEFAULT_CONTAINER_RUNTIME,
    DEFAULT_CONTAINER_VULKAN_ICD,
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
    container_gpu_target: str = typer.Option(
        "",
        "--container-gpu-target",
        help="GPU target used to resolve the manifest image when --container-image is omitted.",
    ),
    container_image_variant: str = typer.Option(
        "",
        "--container-image-variant",
        help="SONIC image manifest variant for --backend container.",
    ),
    container_runtime: str = typer.Option(
        DEFAULT_CONTAINER_RUNTIME,
        "--container-runtime",
        help="Container runtime command for --backend container.",
    ),
    container_gpus: str = typer.Option(
        DEFAULT_CONTAINER_GPUS,
        "--container-gpus",
        help="Docker GPU request for --backend container.",
    ),
    container_driver_capabilities: str = typer.Option(
        DEFAULT_CONTAINER_DRIVER_CAPABILITIES,
        "--container-driver-capabilities",
        help="NVIDIA driver capabilities exposed to the eval container.",
    ),
    container_vulkan_icd: str = typer.Option(
        DEFAULT_CONTAINER_VULKAN_ICD,
        "--container-vulkan-icd",
        help="Vulkan ICD path exposed to the eval container.",
    ),
    container_glx_vendor: str = typer.Option(
        DEFAULT_CONTAINER_GLX_VENDOR,
        "--container-glx-vendor",
        help="GLX vendor library name exposed to the eval container.",
    ),
    container_device: list[str] | None = typer.Option(
        None,
        "--container-device",
        help="Host device path passed through to the eval container.",
    ),
    container_render_frames: int = typer.Option(
        DEFAULT_CONTAINER_RENDER_FRAMES,
        "--container-render-frames",
        help="Minimum Isaac Lab headless render frames for first-party container eval.",
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

    resolved_container_image = container_image
    if backend.value == CONTAINER_BACKEND and not resolved_container_image:
        try:
            resolved_container_image = container_image_for_tool(
                "sonic",
                gpu_target=container_gpu_target or None,
                image_variant=container_image_variant or None,
            )
        except ValueError as exc:
            fail(str(exc))

    try:
        result = evaluate_onnx_policy(
            onnx=onnx_path,
            metadata=metadata_path or None,
            backend=backend.value,
            episodes=episodes,
            env=env,
            output=output_path,
            container_image=resolved_container_image,
            container_runtime=container_runtime,
            container_gpus=container_gpus,
            container_driver_capabilities=container_driver_capabilities,
            container_vulkan_icd=container_vulkan_icd,
            container_glx_vendor=container_glx_vendor,
            container_device=container_device or [],
            container_render_frames=container_render_frames,
            container_policy_path=container_policy_path,
            container_metadata_path=container_metadata_path,
            container_output_path=container_output_path,
            container_args=container_arg or [],
        )
    except SonicEvalError as exc:
        fail(str(exc))

    output(result, output_format)
