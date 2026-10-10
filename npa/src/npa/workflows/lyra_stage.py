"""Fetch isolated Lyra dependencies and publish verified reconstruction artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import requests

from npa.workflows.lerobot_transfer_data import file_sha256, materialize, publish
from npa.workflows.lyra_reconstruction import (
    MODEL_FILE,
    MODEL_REVISION,
    SOURCE_REVISION,
)

CHECKPOINT_SHA256 = "d26380a2d2ecceb6c7ed8ccdb6c53d2664259132ddc946c6c189cf29151c8042"
GSPLAT_REVISION = "0b4dddf04cb687367602c01196913cde6a743d70"


def _fetch_checkpoint(destination):
    url = (
        f"https://huggingface.co/nvidia/Lyra-2.0/resolve/{MODEL_REVISION}/{MODEL_FILE}"
    )
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with destination.open("xb") as stream:
            for chunk in response.iter_content(8 * 1024 * 1024):
                stream.write(chunk)
    if file_sha256(destination) != CHECKPOINT_SHA256:
        raise ValueError("Lyra checkpoint failed the upstream LFS SHA-256 check")


def _source(destination):
    subprocess.run(
        ["git", "clone", "https://github.com/nv-tlabs/lyra.git", str(destination)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", SOURCE_REVISION],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "submodule",
            "update",
            "--init",
            "--depth",
            "1",
            "Lyra-2/lyra_2/_src/inference/depth_anything_3",
        ],
        check=True,
    )


def _torch_environment(workspace):
    interpreter = os.environ.get("NPA_LYRA_BASE_PYTHON", "/opt/npa/sim/venv/bin/python")
    environment = workspace / "environment"
    subprocess.run(
        [interpreter, "-m", "venv", str(environment)],
        check=True,
    )
    python = str(environment / "bin/python")
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
            "torch==2.11.0",
            "torchvision==0.26.0",
        ],
        check=True,
    )
    return python


def _environment(workspace, source):
    python = _torch_environment(workspace)
    constraints = workspace / "constraints.txt"
    constraints.write_text(
        "numpy==1.26.4\nopencv-python==4.11.0.86\ntorch==2.11.0\ntorchvision==0.26.0\n"
    )
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "-c",
            str(constraints),
            str(source / "Lyra-2/lyra_2/_src/inference/depth_anything_3"),
            "loguru==0.7.3",
            "ninja==1.13.0",
            "jaxtyping==0.3.2",
            "beartype==0.21.0",
            "addict==2.4.0",
            "setuptools==81.0.0",
            "wheel==0.47.0",
        ],
        check=True,
    )
    _install_gsplat(python)
    _verify_environment(python)
    return python


def _install_gsplat(python):
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            f"git+https://github.com/nerfstudio-project/gsplat.git@{GSPLAT_REVISION}",
        ],
        check=True,
    )


def _verify_environment(python):
    subprocess.run(
        [
            python,
            "-c",
            "from depth_anything_3.api import DepthAnything3; "
            "import torch; import gsplat; "
            "assert torch.cuda.is_available(), 'Lyra requires a CUDA GPU'; "
            "value = torch.ones(1, device='cuda') + 1; "
            "assert value.item() == 2; "
            "print('Lyra imports and CUDA kernel verified:', "
            "torch.cuda.get_device_name())",
        ],
        check=True,
    )


def _reconstruct(args, workspace):
    source = workspace / "upstream"
    _source(source)
    python = _environment(workspace, source)
    checkpoint = workspace / "model.pt"
    _fetch_checkpoint(checkpoint)
    incoming = materialize(args.input_path, workspace / "input")
    output = workspace / "output"
    _run_inference(args, python, source, checkpoint, incoming, output)
    _publish_reconstruction(args, incoming, output)


def _run_inference(args, python, source, checkpoint, incoming, output):
    environment = dict(os.environ, CUDA_HOME="/usr/local/cuda")
    subprocess.run(
        [
            python,
            "-m",
            "npa.workflows.lyra_reconstruction",
            "--source-path",
            str(source),
            "--checkpoint-path",
            str(checkpoint),
            "--input-path",
            str(incoming / "capture.mp4"),
            "--output-path",
            str(output),
            "--views",
            str(args.views),
            "--resolution",
            str(args.resolution),
        ],
        env=environment,
        check=True,
    )


def _publish_reconstruction(args, incoming, output):
    for name in (
        "capture.mp4",
        "capture.json",
        "measured-depth.npz",
        "ATTRIBUTION.txt",
    ):
        if (incoming / name).is_file():
            shutil.copy2(incoming / name, output / name)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "npa.workflows.lyra_reconstruction_demo",
            "--input-path",
            str(output),
            "--output-path",
            str(output / "index.html"),
        ],
        check=True,
    )
    publish(output, args.output_path)


def main():
    """Run a real reconstruction stage with runtime-fetched private model bytes.

    Args:
        None; reads the command line.
    Returns:
        None after artifact publication.
    Raises:
        ValueError: Input or model integrity fails.
        subprocess.CalledProcessError: Upstream installation or execution fails.
        OSError: Files cannot be read or written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--views", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=640)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="lyra-reconstruction-") as temporary:
        _reconstruct(args, Path(temporary))


if __name__ == "__main__":
    main()
