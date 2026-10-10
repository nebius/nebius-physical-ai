"""Run the HY-World BYOF stage without interpolating operator input into a shell.

The generic ``workbench.byof.repo`` worker owns scheduling, output transport and
the S3 upload. This image-owned adapter stages one declared PNG and passes
validated values to the real HY-World runtime. The private vLLM address travels
through SkyPilot's secret environment channel, not workflow configuration.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener


EX_CONFIG = 78
SMOKE_ARTIFACT = "hy_world_image_to_world.json"
REPORT_ARTIFACTS = (
    "reports/hy_world_scene.rrd",
    "reports/hy_world_scene_rrd_manifest.json",
)
QWEN_VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


class WorkflowInputError(ValueError):
    """Signal that a supplied workflow input misses the published contract.

    Args:
        None.

    Returns:
        None.

    Raises:
        None. Raised by workflow input and stage-contract validation helpers.
    """


def _required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise WorkflowInputError(f"{name} is required")
    return value


def _decode_config(name: str) -> str:
    """Decode one base64-rendered, UTF-8 workflow configuration value."""

    try:
        return base64.b64decode(_required(name), validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise WorkflowInputError(f"{name} must be base64-encoded UTF-8") from exc


def _input_uri() -> str:
    source = _decode_config("HY_WORLD_INPUT_IMAGE_URI_B64")
    parsed = urlparse(source)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path
        or not parsed.path.lower().endswith(".png")
        or parsed.query
        or parsed.fragment
    ):
        raise WorkflowInputError(
            "input_image_uri must be one private s3:// object ending in .png"
        )
    return source


def _private_llm_address() -> str:
    """Validate the secret, host-or-address-only vLLM endpoint value."""

    value = _required("NPA_HY_WORLD_LLM_ADDR")
    if (
        value != value.strip()
        or "://" in value
        or "/" in value
        or "@" in value
        or any(ord(character) < 33 for character in value)
        or not re.fullmatch(r"(?:[A-Za-z0-9][A-Za-z0-9.-]*|\[[0-9A-Fa-f:]+\])", value)
    ):
        raise WorkflowInputError(
            "NPA_HY_WORLD_LLM_ADDR must be a reachable hostname or address without a scheme"
        )
    return value


def _llm_port() -> str:
    value = _decode_config("HY_WORLD_LLM_PORT_B64")
    if not value.isdecimal() or not 1 <= int(value) <= 65535:
        raise WorkflowInputError("llm_port must be a TCP port number")
    return value


def _llm_name() -> str:
    """The upstream trajectory code is pinned to this served model identity."""

    return QWEN_VLM_MODEL


def _prompt() -> str:
    value = _decode_config("HY_WORLD_PROMPT_B64")
    if not value.strip() or "\x00" in value:
        raise WorkflowInputError("prompt is empty or invalid")
    return value


def _download_image(source: str, destination: Path) -> None:
    """Download the declared private PNG using the worker's scoped S3 identity."""

    import boto3
    from botocore.config import Config

    parsed = urlparse(source)
    endpoint = (
        os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("NEBIUS_S3_ENDPOINT") or ""
    ).strip() or None
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        config=Config(s3={"addressing_style": "path"}),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-central1"),
    )
    client.download_file(parsed.netloc, parsed.path.lstrip("/"), str(destination))
    if not destination.is_file() or destination.stat().st_size < 256:
        raise WorkflowInputError("downloaded input image is unexpectedly small")


def _validate_image(path: Path) -> None:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "null", "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise WorkflowInputError(
            f"input image does not decode: {result.stderr.strip()}"
        )


def _run_runtime(command: str, environment: dict[str, str]) -> None:
    subprocess.run(["hy-world-runtime", command], check=True, env=environment)


def _check_vllm_endpoint(address: str, port: str, model_name: str) -> None:
    """Prove the private OpenAI-compatible endpoint exposes the required model.

    The released ``traj_generate.py`` creates ``OpenAI(base_url=http://ADDR:PORT/v1)``
    and calls chat completions. The ordinary vLLM ``/v1/models`` discovery
    surface is a request-free preflight for the same endpoint. Deliberately
    bypass proxy environment variables: the operator supplied this private
    worker-reachable address and it must not be forwarded to an ambient proxy.
    """

    request = Request(
        f"http://{address}:{port}/v1/models",
        # ``traj_generate.py`` uses OpenAI(api_key="EMPTY", ...), so use the
        # same non-secret credential shape for the preflight request.
        headers={"Accept": "application/json", "Authorization": "Bearer EMPTY"},
        method="GET",
    )
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=15) as response:
            payload = json.load(response)
    except Exception as exc:  # noqa: BLE001 - endpoint details remain secret
        raise WorkflowInputError(
            "private vLLM endpoint did not answer its OpenAI /v1/models contract"
        ) from exc
    entries = payload.get("data") if isinstance(payload, dict) else None
    model_ids = (
        {str(entry.get("id") or "") for entry in entries if isinstance(entry, dict)}
        if isinstance(entries, list)
        else set()
    )
    if model_name not in model_ids:
        raise WorkflowInputError(
            "private vLLM endpoint does not report the pinned Qwen3-VL model identity"
        )


def run() -> None:
    """Run static checks, stage input, then execute real upstream inference.

    Args:
        None.

    Returns:
        None.

    Raises:
        WorkflowInputError: If declared input, endpoint, or output contracts fail.
        subprocess.CalledProcessError: If the real runtime command exits nonzero.
    """

    # Validate every value before obtaining user data. The immutable OCI layer
    # scanner is the payload-absence gate; bootstrap-integrity only checks the
    # installed wrapper and never traverses mutable input or cache directories.
    source = _input_uri()
    prompt = _prompt()
    address = _private_llm_address()
    port = _llm_port()
    model_name = _llm_name()
    output = Path(_required("NPA_SMOKE_OUTPUT_DIR"))
    if not output.is_dir():
        raise WorkflowInputError("NPA_SMOKE_OUTPUT_DIR must be an existing directory")
    image = _required("BYOF_IMAGE")
    stage_root = Path(os.environ.get("NPA_HY_WORLD_INPUT_STAGE_DIR", "/workspace"))
    if not stage_root.is_absolute() or not stage_root.is_dir():
        raise WorkflowInputError(
            "NPA_HY_WORLD_INPUT_STAGE_DIR must be an existing absolute directory"
        )

    environment = {
        **os.environ,
        "HY_WORLD_PROMPT_B64": base64.b64encode(prompt.encode("utf-8")).decode("ascii"),
        "HY_WORLD_SCENE_DIR": str(output / "scene"),
        "HY_WORLD_RESULT_DIR": str(output / "result"),
        "HY_WORLD_CONTAINER_IMAGE": image,
        "NPA_HY_WORLD_LLM_ADDR": address,
        "NPA_HY_WORLD_LLM_PORT": port,
        "NPA_HY_WORLD_LLM_NAME": model_name,
    }
    _run_runtime("terms", environment)
    _run_runtime("health", environment)
    _run_runtime("bootstrap-integrity", environment)
    _check_vllm_endpoint(address, port, model_name)

    with tempfile.TemporaryDirectory(
        prefix="npa-hy-world-input-", dir=stage_root
    ) as directory:
        input_path = Path(directory) / "input.png"
        _download_image(source, input_path)
        _validate_image(input_path)
        environment["HY_WORLD_INPUT_IMAGE"] = str(input_path)
        _run_runtime("run-image-to-world", environment)
    for artifact in (SMOKE_ARTIFACT, *REPORT_ARTIFACTS):
        if not (output / artifact).is_file():
            raise WorkflowInputError(f"real HY-World runtime did not write {artifact}")


def main() -> int:
    """Run the stage adapter and translate expected failures into exit codes.

    Args:
        None.

    Returns:
        The documented configuration or runtime process exit code.

    Raises:
        None.
    """
    try:
        run()
    except WorkflowInputError as exc:
        print(f"npa-hy-world: {exc}", file=sys.stderr)
        return EX_CONFIG
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode or 1)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the container stage
    raise SystemExit(main())
