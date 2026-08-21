"""Shared genuine Alpamayo 2 Super inference implementation.

The CLI, SDK, FastAPI service, and npa.workflow toolRef call this module. The
public image contains Apache-2.0 inference source and runtime, but no model
weights, PhysicalAI-AV data, credentials, or populated Hugging Face cache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable
from urllib.parse import urlparse

from npa.clients.storage import StorageClient

DEFAULT_MODEL_ID = "nvidia/Alpamayo2-Super"
DEFAULT_MODEL_REVISION = "00554695e729a6ff0b6281fd2c81b18d06e33dbe"
DEFAULT_DATASET_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
DEFAULT_DATASET_REVISION = "b719eea7f0a63619ef51ec7f54178af0937ef050"
DEFAULT_MANIFEST = "/opt/alpamayo2/examples/validation_samples.json"
ARTIFACT_SCHEMA = "npa.workbench.alpamayo2_super.inference.v1"


class Alpamayo2SuperError(RuntimeError):
    """Raised when an inference request or upstream execution fails."""


@dataclass(frozen=True)
class Alpamayo2SuperRequest:
    """A reproducible single-sample expert-trajectory request."""

    output_path: str
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    dataset_revision: str = DEFAULT_DATASET_REVISION
    manifest: str = DEFAULT_MANIFEST
    sample_index: int = 0
    diffusion_steps: int = 10
    seed: int = 42
    figure_style: str = "blog"
    require_camera_projection: bool = True
    run_id: str = ""
    runtime_image: str = ""
    dry_run: bool = False


def _validate(request: Alpamayo2SuperRequest) -> None:
    if not request.output_path.strip():
        raise Alpamayo2SuperError("output_path is required")
    if request.sample_index < 0:
        raise Alpamayo2SuperError("sample_index must be non-negative")
    if request.diffusion_steps < 1:
        raise Alpamayo2SuperError("diffusion_steps must be positive")
    if request.figure_style not in {"blog", "compact"}:
        raise Alpamayo2SuperError("figure_style must be blog or compact")
    if (
        request.model_id == DEFAULT_MODEL_ID
        and request.model_revision != DEFAULT_MODEL_REVISION
    ):
        raise Alpamayo2SuperError(
            "the public Alpamayo2-Super model must use the repository-pinned revision"
        )
    parsed = urlparse(request.output_path)
    if parsed.scheme and parsed.scheme != "s3":
        raise Alpamayo2SuperError("output_path must be a local path or s3:// prefix")
    if parsed.scheme == "s3" and (not parsed.netloc or not parsed.path.strip("/")):
        raise Alpamayo2SuperError("output_path must include an S3 bucket and prefix")


def build_inference_argv(
    request: Alpamayo2SuperRequest, *, local_output: Path
) -> list[str]:
    """Build upstream's real inference-smoke argv."""

    argv = [
        os.environ.get("ALPAMAYO2_SUPER_PYTHON", "/opt/alpamayo2/.venv/bin/python"),
        "-m",
        "alpamayo2_super.inference_smoke",
        "--model-id",
        request.model_id,
        "--manifest",
        request.manifest,
        "--sample-index",
        str(request.sample_index),
        "--diffusion-steps",
        str(request.diffusion_steps),
        "--seed",
        str(request.seed),
        "--figure-style",
        request.figure_style,
        "--save-viz",
        str(local_output / "trajectory.png"),
        "--save-json",
        str(local_output / "trajectory.json"),
    ]
    if request.require_camera_projection:
        argv.append("--require-camera-projection")
    return argv


def _runtime_env(request: Alpamayo2SuperRequest) -> dict[str, str]:
    env = dict(os.environ)
    cache = env.get("HF_HOME", "/workspace/.cache/huggingface")
    env.update(
        {
            "HF_HOME": cache,
            "HF_HUB_CACHE": env.get("HF_HUB_CACHE", f"{cache}/hub"),
            "HF_HUB_ETAG_TIMEOUT": env.get("HF_HUB_ETAG_TIMEOUT", "30"),
            "HF_HUB_DOWNLOAD_TIMEOUT": env.get("HF_HUB_DOWNLOAD_TIMEOUT", "300"),
            "HF_HUB_REVISION": request.model_revision,
            "PYTORCH_CUDA_ALLOC_CONF": env.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            ),
            "ALPAMAYO2_SUPER_MODEL_REVISION": request.model_revision,
            "ALPAMAYO2_SUPER_DATASET_REVISION": request.dataset_revision,
        }
    )
    return env


def _publish(local_dir: Path, output_path: str) -> dict[str, str]:
    if output_path.startswith("s3://"):
        base = output_path.rstrip("/") + "/"
        client = StorageClient.from_environment()
        return {
            path.name: client.upload_file(str(path), base + path.name)
            for path in sorted(local_dir.iterdir())
            if path.is_file()
        }
    target = Path(output_path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    published: dict[str, str] = {}
    for source in sorted(local_dir.iterdir()):
        if source.is_file():
            destination = target / source.name
            destination.write_bytes(source.read_bytes())
            published[source.name] = str(destination)
    return published


def _resolve_model_snapshot(request: Alpamayo2SuperRequest) -> str:
    """Resolve the public model to an exact-revision local snapshot."""

    if request.model_id != DEFAULT_MODEL_ID:
        return request.model_id
    code = (
        "from huggingface_hub import snapshot_download; "
        "print(snapshot_download(repo_id="
        + repr(request.model_id)
        + ", revision="
        + repr(request.model_revision)
        + ", token=True if __import__('os').environ.get('HF_TOKEN') else None))"
    )
    completed = subprocess.run(
        [
            os.environ.get("ALPAMAYO2_SUPER_PYTHON", "/opt/alpamayo2/.venv/bin/python"),
            "-c",
            code,
        ],
        env=_runtime_env(request),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout or "").splitlines()[-30:])
        raise Alpamayo2SuperError(
            f"failed to fetch pinned Alpamayo2-Super snapshot ({completed.returncode}):\n{tail}"
        )
    resolved = (completed.stdout or "").splitlines()[-1].strip()
    if not resolved or not Path(resolved).is_dir():
        raise Alpamayo2SuperError("Hugging Face did not return a local model snapshot")
    return resolved


def run_inference(
    request: Alpamayo2SuperRequest,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    model_resolver: Callable[[Alpamayo2SuperRequest], str] = _resolve_model_snapshot,
) -> dict[str, Any]:
    """Run upstream inference and publish its JSON/PNG plus NPA provenance."""

    _validate(request)
    with tempfile.TemporaryDirectory(prefix="npa-alpamayo2-super-") as scratch:
        local_dir = Path(scratch)
        execution_request = request
        if not request.dry_run:
            execution_request = replace(request, model_id=model_resolver(request))
        argv = build_inference_argv(execution_request, local_output=local_dir)
        base = {
            "schema": ARTIFACT_SCHEMA,
            "model": {"id": request.model_id, "revision": request.model_revision},
            "dataset": {
                "id": DEFAULT_DATASET_REPO,
                "revision": request.dataset_revision,
                "operator_runtime_fetch": True,
            },
            "request": asdict(request),
            "runtime": {
                "image": request.runtime_image or os.environ.get("NPA_TASK_IMAGE", ""),
                "weights_baked": False,
                "dataset_baked": False,
                "cache_tier": "node-local-ephemeral",
            },
            "argv": argv,
        }
        if request.dry_run:
            return {**base, "status": "dry_run", "artifacts": {}}
        completed = runner(
            argv,
            cwd="/opt/alpamayo2",
            env=_runtime_env(request),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            tail = "\n".join((completed.stdout or "").splitlines()[-40:])
            raise Alpamayo2SuperError(
                f"upstream Alpamayo2-Super inference failed ({completed.returncode}):\n{tail}"
            )
        expected = (local_dir / "trajectory.json", local_dir / "trajectory.png")
        missing = [
            path.name
            for path in expected
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            raise Alpamayo2SuperError(
                "upstream inference returned success without required artifacts: "
                + ", ".join(missing)
            )
        result_path = local_dir / "result.json"
        result_path.write_text(
            json.dumps({**base, "status": "ok"}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts = _publish(local_dir, request.output_path)
        return {**base, "status": "ok", "artifacts": artifacts}
