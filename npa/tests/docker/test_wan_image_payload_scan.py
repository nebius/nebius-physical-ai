from __future__ import annotations

import io
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "scan_image_wan_payload.py"
_SPEC = importlib.util.spec_from_file_location("scan_image_wan_payload", _SCRIPT)
assert _SPEC and _SPEC.loader
scanner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scanner
_SPEC.loader.exec_module(scanner)


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def test_clean_oss_rootfs_and_runtime_fetch_plumbing_pass(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "clean.tar",
        {
            "opt/byof/LICENSE.txt": b"Apache License 2.0",
            "opt/npa/wan2-2/runtime-requirements.txt": b"torch==2.7.1\n",
            "usr/local/bin/wan-runtime": b"NVIDIA terms; runtime fetch only\n",
        },
    )
    assert (
        scanner.scan(
            rootfs, {"history": [{"created_by": "COPY runtime-requirements.txt /opt/"}]}
        )
        == []
    )


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        (
            "opt/venv/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas.so.12",
            "nvidia_python_distribution",
        ),
        (
            "opt/venv/lib/python3.10/site-packages/easydict-1.13.dist-info/METADATA",
            "historical_lgpl_python_distribution",
        ),
        (
            "opt/venv/lib/python3.10/site-packages/soxr-1.1.0.dist-info/METADATA",
            "historical_lgpl_python_distribution",
        ),
        (
            "opt/venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2",
            "bundled_ffmpeg_executable",
        ),
        ("usr/local/lib/libcudnn.so.9", "cuda_library"),
        ("usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1", "cuda_library"),
        ("usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0", "cuda_library"),
        ("usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0", "cuda_library"),
        ("usr/lib/x86_64-linux-gnu/libnvcuvid.so.1", "cuda_library"),
        ("usr/lib/x86_64-linux-gnu/libnvoptix.so.1", "cuda_library"),
        (
            "opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so",
            "cuda_library",
        ),
        (
            "opt/venv/lib/python3.10/site-packages/torch/lib/libc10_cuda.so",
            "cuda_library",
        ),
        ("usr/lib/x86_64-linux-gnu/libcufile.so.0", "cuda_library"),
        ("usr/local/cuda/bin/ptxas", "cuda_tool"),
        ("usr/local/cuda/include/cuda.h", "cuda_tool"),
        ("usr/local/cuda-12.8/include/cuda.h", "cuda_tool"),
        ("opt/cuda/include/nccl.h", "cuda_tool"),
        ("usr/include/cuda_runtime_api.h", "cuda_tool"),
        ("usr/bin/nvidia-smi", "cuda_tool"),
        ("usr/lib/x86_64-linux-gnu/libnvToolsExt.so.1", "cuda_library"),
        ("usr/lib/x86_64-linux-gnu/libnppc.so.12", "cuda_library"),
        (
            "opt/venv/lib/python3.10/site-packages/nvidia_cuda_runtime_cu12-12.8.dist-info/METADATA",
            "nvidia_python_distribution",
        ),
        ("opt/byof/model.safetensors", "checkpoint_or_weight"),
        ("opt/byof/pytorch_model-00001-of-00002.bin", "checkpoint_or_weight"),
        ("root/.cache/huggingface/hub/model.bin", "checkpoint_or_weight"),
        ("root/.aws/credentials", "credential_file"),
        ("tmp/wheelhouse/download.whl", "package_cache"),
        ("etc/ssh/ssh_host_rsa_key", "credential_file"),
    ],
)
def test_forbidden_path_mutations_fail(tmp_path: Path, name: str, kind: str) -> None:
    findings = scanner.scan(_tar(tmp_path / "bad.tar", {name: b"payload"}), {})
    assert kind in {item.kind for item in findings}


@pytest.mark.parametrize(
    "created_by",
    [
        "RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1",
        "RUN pip install nvidia-cublas-cu12==12.8.3.14",
        "RUN wan-runtime ensure",
        "FROM nvidia/cuda:12.8.1-runtime",
        "ENV NPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS=YES",
    ],
)
def test_forbidden_history_mutations_fail(tmp_path: Path, created_by: str) -> None:
    rootfs = _tar(tmp_path / "history.tar", {"opt/byof/LICENSE.txt": b"Apache-2.0"})
    assert scanner.scan(rootfs, {"history": [{"created_by": created_by}]})


def test_negative_inventory_assertion_is_not_a_cuda_install(tmp_path: Path) -> None:
    rootfs = _tar(tmp_path / "history.tar", {"opt/byof/LICENSE.txt": b"Apache-2.0"})
    created_by = (
        "RUN python -m pip install --index-url "
        "https://download.pytorch.org/whl/cpu torch==2.7.1+cpu "
        "&& ! grep -Eiq '^nvidia-|cu(da|dnn|blas)|nccl' inventory.txt"
    )
    assert scanner.scan(rootfs, {"history": [{"created_by": created_by}]}) == []


def test_secret_content_mutation_fails(tmp_path: Path) -> None:
    rootfs = _tar(tmp_path / "secret.tar", {"tmp/config": b"AKIAABCDEFGHIJKLMNOP"})
    findings = scanner.scan(rootfs, {})
    assert {item.kind for item in findings} == {"credential_content"}


def test_large_secret_content_and_chunk_boundary_mutations_fail(tmp_path: Path) -> None:
    prefix = b"x" * (2 * 1024 * 1024 - 8)
    rootfs = _tar(
        tmp_path / "large-secret.tar",
        {"opt/application/large.dat": prefix + b"AKIAABCDEFGHIJKLMNOP"},
    )
    findings = scanner.scan(rootfs, {})
    assert {item.kind for item in findings} == {"credential_content"}


def test_deleted_forbidden_bytes_in_a_lower_layer_still_fail(tmp_path: Path) -> None:
    lower = _tar(
        tmp_path / "lower.tar",
        {
            "opt/venv/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.9": b"vendor"
        },
    )
    upper = _tar(
        tmp_path / "upper.tar",
        {
            "opt/venv/lib/python3.10/site-packages/nvidia/cudnn/lib/.wh.libcudnn.so.9": b""
        },
    )
    findings = scanner._scan_tars([lower, upper], {})
    assert "cuda_library" in {item.kind for item in findings}


def test_renamed_elf_with_cuda_dependency_fails(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "elf.tar",
        {"opt/application/libinnocent.so": b"\x7fELF\x00libcuda.so.1\x00"},
    )
    findings = scanner.scan(rootfs, {})
    assert "cuda_elf_dependency" in {item.kind for item in findings}


@pytest.mark.parametrize(
    "dependency",
    [b"libtorch_cuda.so", b"libc10_cuda.so", b"libnvToolsExt.so.1"],
)
def test_renamed_elf_with_embedded_cuda_dependency_fails(
    tmp_path: Path, dependency: bytes
) -> None:
    rootfs = _tar(
        tmp_path / "elf-dependency.tar",
        {"opt/application/librenamed.so": b"\x7fELF\x00" + dependency + b"\x00"},
    )
    findings = scanner.scan(rootfs, {})
    assert "cuda_elf_dependency" in {item.kind for item in findings}


def test_cli_emits_machine_readable_failure(tmp_path: Path, capsys) -> None:
    rootfs = _tar(tmp_path / "bad.tar", {"models/weights.ckpt": b"x"})
    assert scanner.main(["--rootfs-tar", str(rootfs)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "fail"
