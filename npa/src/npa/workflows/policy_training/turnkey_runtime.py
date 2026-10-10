"""Prepare the pinned native LeRobot environment and durable checkpoint recovery."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys


NATIVE_PINS = {
    "lerobot": "0.6.0",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "transformers": "5.5.4",
    "requests": "2.32.5",
    "torchcodec": "0.11.0",
}


def native_runtime(root: Path) -> None:
    """Use the qualified native package versions and require CUDA plus NVIDIA EGL.

    Args:
        root: Private worker workspace.
    Returns:
        None.
    Raises:
        RuntimeError: CUDA or the NVIDIA renderer is unavailable.
        subprocess.CalledProcessError: Runtime installation fails.
    """
    versions = {key: _version(key) for key in NATIVE_PINS}
    if any(versions[key].split("+")[0] != value for key, value in NATIVE_PINS.items()):
        _install(root)
    # Import loads the decoder's native libraries: version metadata alone cannot
    # prove its compiled ABI matches torch or the image's FFmpeg shared libraries.
    importlib.import_module("torchcodec.decoders")
    import torch
    from .public_vla import _environment

    if not torch.cuda.is_available():
        raise RuntimeError("public policy stages require a real CUDA GPU")
    _environment(root)
    import json

    if json.loads((root / "rendering.json").read_text())["backend"] != "nvidia-egl":
        raise RuntimeError(
            "public policy proof requires NVIDIA EGL, not software rendering"
        )


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _install(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runtime-install.log").open("a") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "cmake==3.31.6",
                "wheel==0.45.1",
                "pybind11==3.0.2",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
        requirements = [
            f"{key}=={value}" for key, value in NATIVE_PINS.items() if key != "lerobot"
        ]
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-build-isolation",
                "lerobot[smolvla,libero]==0.6.0",
                *requirements,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_native(command: list[str], log: Path, environment: dict) -> None:
    """Run a real native component while retaining its private diagnostics.

    Args:
        command: Native process arguments.
        log: Private output log.
        environment: Exact worker environment.
    Returns:
        None.
    Raises:
        subprocess.CalledProcessError: Native execution fails.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        subprocess.run(
            command,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def publish_recovery(checkpoint: Path, step: int, uri: str) -> None:
    """Publish complete recovery state before atomically replacing the latest pointer.

    Args:
        checkpoint: Native model, optimizer, scheduler, RNG and sampler directory.
        step: Completed optimizer update.
        uri: Attempt-specific private recovery prefix.
    Returns:
        None.
    Raises:
        OSError: Recovery cannot be durably uploaded and verified.
    """
    from .public_vla_export import file_sha256
    from .turnkey_store import publish
    from npa.workbench.dataset.storage import write_json_uri

    shutil.copytree(
        Path(os.environ["NPA_VLA_EVIDENCE"]),
        checkpoint / "npa-evidence",
        dirs_exist_ok=True,
    )
    destination = uri.rstrip("/") + f"/{step}/"
    publish(checkpoint, destination)
    write_json_uri(
        uri.rstrip("/") + "/latest.json",
        {
            "step": step,
            "uri": destination,
            "weights_sha256": file_sha256(
                checkpoint / "pretrained_model/model.safetensors"
            ),
            "recipe_sha256": os.environ["NPA_VLA_RECIPE_SHA256"],
            "selection_sha256": os.environ["NPA_VLA_SELECTION_SHA256"],
        },
    )


def partial_recovery(
    uri: str, target: Path, recipe_sha256: str, selection_sha256: str
) -> Path | None:
    """Adopt only a complete checksum-verified checkpoint from this exact attempt.

    Args:
        uri: Attempt-specific recovery prefix.
        target: Private local staging directory.
        recipe_sha256: Exact immutable recipe identity.
        selection_sha256: Exact selected episodes and fitted normalization identity.
    Returns:
        Complete checkpoint, or None when no checkpoint pointer exists.
    Raises:
        ValueError: Recovery identity is inconsistent.
        OSError: Existing recovery cannot be verified.
    """
    from .public_vla_export import file_sha256
    from .turnkey_store import materialize

    pointer = _recovery_pointer(uri)
    if pointer is None:
        return None
    if (
        pointer["recipe_sha256"] != recipe_sha256
        or pointer["selection_sha256"] != selection_sha256
    ):
        raise ValueError("interrupted checkpoint recipe or training selection changed")
    root = materialize(pointer["uri"], target)
    if (
        file_sha256(root / "pretrained_model/model.safetensors")
        != pointer["weights_sha256"]
    ):
        raise ValueError("recovery model identity changed")
    return root


def _recovery_pointer(uri):
    from botocore.exceptions import ClientError
    from npa.workbench.dataset.storage import read_json_uri

    try:
        return read_json_uri(uri.rstrip("/") + "/latest.json")
    except ClientError as exc:
        if exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
            return None
        raise
    except FileNotFoundError:
        return None
