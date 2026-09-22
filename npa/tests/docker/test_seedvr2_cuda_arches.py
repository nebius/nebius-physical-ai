"""Exercise matching build targets and fail closed on missing native kernels."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/seedvr2"
HELPER = IMAGE / "cuda_arches.py"
SCANNER = ROOT / "npa/scripts/measure_extension_arches.py"


@pytest.fixture
def arches():
    spec = importlib.util.spec_from_file_location("seedvr_cuda_arches", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("flash", "torch", "expected"),
    [
        ("90", "9.0", ["sm_90"]),
        ("90;100", "9.0;10.0", ["sm_90", "sm_100"]),
        ("100;90", "9.0;10.0", ["sm_90", "sm_100"]),
    ],
)
def test_matching_targets_are_order_independent(arches, flash, torch, expected):
    assert arches.exact_arches(flash, torch) == expected


@pytest.mark.parametrize(
    ("flash", "torch"),
    [
        ("90;100", "9.0"),
        ("90", "9.0;10.0"),
        ("90;90", "9.0"),
        ("90", "9.0;9.0"),
        ("", ""),
        ("090", "9.0"),
        ("90;", "9.0"),
        ("90; 100", "9.0;10.0"),
        ("90", "9"),
        ("90", "9.0+PTX"),
        ("90", "9.00"),
        ("90a", "9.0"),
        ("90;$(touch /tmp/not-executed)", "9.0"),
    ],
)
def test_ambiguous_or_mismatched_targets_are_rejected(arches, flash, torch):
    with pytest.raises(ValueError):
        arches.exact_arches(flash, torch)


def _arguments(flash, torch):
    result = subprocess.run(
        [sys.executable, str(HELPER), "--flash-arches", flash, "--torch-arches", torch],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.split()


def _fatbin(architectures):
    entries = b""
    for architecture in architectures:
        header = bytearray(64)
        struct.pack_into("<HHIQ", header, 0, 2, 0x101, 64, 8)
        struct.pack_into("<I", header, 28, architecture)
        entries += bytes(header) + b"payload!"
    return struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(entries)) + entries


@pytest.mark.parametrize(
    ("native_arches", "requested_flash", "requested_torch", "expected_exit"),
    [
        ([90], "90", "9.0", 0),
        ([90, 100], "90;100", "9.0;10.0", 0),
        ([90], "90;100", "9.0;10.0", 1),
        ([100], "90;100", "9.0;10.0", 1),
        ([90, 100, 120], "90;100", "9.0;10.0", 1),
    ],
)
def test_real_scanner_enforces_generated_exact_arches(
    tmp_path, native_arches, requested_flash, requested_torch, expected_exit
):
    binary = tmp_path / "extension.so"
    binary.write_bytes(_fatbin(native_arches))
    result = subprocess.run(
        [
            sys.executable,
            str(SCANNER),
            str(binary),
            "--min-size-mb",
            "0",
            *_arguments(requested_flash, requested_torch),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected_exit, result.stderr


def test_build_refuses_mismatch_before_any_git_or_docker_operation(tmp_path):
    tools = tmp_path / "bin"
    tools.mkdir()
    sentinel = tmp_path / "unexpected-operation"
    for name in ("git", "docker"):
        path = tools / name
        path.write_text(f"#!/bin/sh\ntouch '{sentinel}'\nexit 99\n")
        path.chmod(0o755)
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"])
    env["NPA_PYTHON_BIN"] = sys.executable
    result = subprocess.run(
        ["bash", str(IMAGE / "build.sh"), "--flash-attn-cuda-archs", "90;100"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "same architectures" in result.stderr
    assert not sentinel.exists()


def test_dockerfile_and_builder_forward_both_architecture_inputs():
    dockerfile = (IMAGE / "Dockerfile").read_text()
    build = (IMAGE / "build.sh").read_text()
    assert dockerfile.count("ARG TORCH_CUDA_ARCH_LIST=9.0") == 2
    assert dockerfile.count("TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}") == 2
    assert (
        dockerfile.count('npa.build.torch.cuda-arch-list="${TORCH_CUDA_ARCH_LIST}"')
        == 2
    )
    assert '--build-arg "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"' in build
    assert "--torch-cuda-arch-list)" in build
    assert dockerfile.count('--skip-no-fatbin "${ARCH_ARGS[@]}" --json') == 2
