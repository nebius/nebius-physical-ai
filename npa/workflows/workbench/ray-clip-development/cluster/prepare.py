# Prepare pinned application dependencies and public CLIP weights per SkyPilot pod.
"""Keep application setup separate from SkyPilot's management environment."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

APPLICATION_ENVIRONMENT = Path("/tmp/ray-clip-env")
MODEL_DIRECTORY = Path("/tmp/npa-clip-model")
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
WEIGHTS_SHA256 = "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"


def _prepare_application_environment() -> str:
    """Install pinned Python packages while retaining the image's CUDA wheels."""
    # The image has pip but no ensurepip; it must remain the pip launcher.
    subprocess.run([
        sys.executable, "-m", "venv", "--system-site-packages", "--without-pip",
        str(APPLICATION_ENVIRONMENT),
    ], check=True)
    application_python = str(APPLICATION_ENVIRONMENT / "bin/python")
    requirements = Path(__file__).with_name("requirements.txt")
    subprocess.run([
        sys.executable, "-m", "pip", "--python", application_python, "install",
        "-r", str(requirements),
    ], check=True)
    return application_python


def _download_and_verify_model(application_python: str) -> str:
    """Fetch the public snapshot without credentials and verify its weight bytes."""
    download = f"""
from huggingface_hub import snapshot_download
snapshot_download(
    'openai/clip-vit-base-patch32', revision={MODEL_REVISION!r},
    local_dir={str(MODEL_DIRECTORY)!r}, token=False,
    allow_patterns=[
        'config.json', 'preprocessor_config.json', 'tokenizer_config.json',
        'vocab.json', 'merges.txt', 'special_tokens_map.json', 'pytorch_model.bin',
    ],
)
"""
    subprocess.run([application_python, "-c", download], check=True)
    weights = MODEL_DIRECTORY / "pytorch_model.bin"
    weights_digest = hashlib.sha256()
    with weights.open("rb") as stream:
        while True:
            content = stream.read(1024 * 1024)
            if not content:
                break
            weights_digest.update(content)
    digest = weights_digest.hexdigest()
    if digest != WEIGHTS_SHA256:
        raise RuntimeError("CLIP weights differ from the pinned public snapshot")
    return digest


def _inspect_cuda_environment(application_python: str) -> dict:
    """Require real CUDA and the application Ray pin before writing readiness."""
    inspection = """
import importlib.metadata
import json
import sys
import torch
import ray
assert torch.cuda.is_available(), 'Real CUDA GPU required'
assert ray.__version__ == '2.58.0'
packages = ['ray', 'lancedb', 'pyarrow', 'numpy', 'pillow', 'transformers']
versions = {}
for package in packages:
    versions[package] = importlib.metadata.version(package)
versions.update(
    python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0), gpu_count=torch.cuda.device_count(),
)
print(json.dumps(versions))
"""
    result = subprocess.check_output([application_python, "-c", inspection], text=True)
    return json.loads(result)


def _write_preparation_receipt(started: float, dependencies_ready: float,
                               model_ready: float, weights_hash: str,
                               application_python: str) -> None:
    """Record setup boundaries and exact installed versions outside source delivery."""
    versions = _inspect_cuda_environment(application_python)
    freeze = subprocess.check_output([application_python, "-m", "pip", "freeze"], text=True)
    finished = time.time()
    receipt = {
        "started_at_unix": started,
        "dependencies_ready_at_unix": dependencies_ready,
        "model_ready_at_unix": model_ready,
        "finished_at_unix": finished,
        "dependency_preparation_seconds": dependencies_ready - started,
        "model_fetch_and_verify_seconds": model_ready - dependencies_ready,
        "cuda_environment_inspection_seconds": finished - model_ready,
        "model_revision": MODEL_REVISION,
        "weights_sha256": weights_hash,
        "versions": versions,
        "dependency_freeze": freeze.splitlines(),
    }
    destination = Path("/tmp/ray-clip-preparation.json")
    destination.write_text(json.dumps(receipt, indent=2) + "\n")


def main() -> None:
    """Prepare a SkyPilot pod for native Ray Jobs without building an image.

    Args:
        None.
    Returns:
        None; writes a private preparation receipt and readiness message.
    Raises:
        subprocess.CalledProcessError: Dependency, model or CUDA preflight fails.
        RuntimeError: Downloaded weights differ from the pinned snapshot.
        OSError: A local environment, model file or receipt cannot be accessed.
    """
    started = time.time()
    application_python = _prepare_application_environment()
    dependencies_ready = time.time()
    weights_hash = _download_and_verify_model(application_python)
    model_ready = time.time()
    _write_preparation_receipt(
        started, dependencies_ready, model_ready, weights_hash, application_python,
    )
    print("CLIP environment prepared; CUDA verified. Application source comes from Ray Jobs.")


if __name__ == "__main__":
    main()
