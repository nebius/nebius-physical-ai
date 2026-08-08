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
        ("opt/byof/model.safetensors", "checkpoint_or_weight"),
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
        "RUN wan-runtime ensure",
        "FROM nvidia/cuda:12.8.1-runtime",
        "ENV NPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS=YES",
    ],
)
def test_forbidden_history_mutations_fail(tmp_path: Path, created_by: str) -> None:
    rootfs = _tar(tmp_path / "history.tar", {"opt/byof/LICENSE.txt": b"Apache-2.0"})
    assert scanner.scan(rootfs, {"history": [{"created_by": created_by}]})


def test_secret_content_mutation_fails(tmp_path: Path) -> None:
    rootfs = _tar(tmp_path / "secret.tar", {"tmp/config": b"AKIAABCDEFGHIJKLMNOP"})
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


def test_cli_emits_machine_readable_failure(tmp_path: Path, capsys) -> None:
    rootfs = _tar(tmp_path / "bad.tar", {"models/weights.ckpt": b"x"})
    assert scanner.main(["--rootfs-tar", str(rootfs)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "fail"
