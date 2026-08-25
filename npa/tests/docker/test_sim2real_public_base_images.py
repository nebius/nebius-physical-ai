"""Keep public Sim2Real development builds independent of local image tags."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKBENCH = ROOT / "npa" / "docker" / "workbench"


def _default_base(relative: str) -> str:
    text = (WORKBENCH / relative).read_text(encoding="utf-8")
    match = re.search(r"^ARG BASE_IMAGE=(\S+)$", text, re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_sim2real_gpu_overlays_use_immutable_public_bases() -> None:
    for dockerfile in (
        "sim2real-envgen/Dockerfile",
        "cosmos3-reason/Dockerfile",
    ):
        base = _default_base(dockerfile)
        assert base.startswith(
            "ghcr.io/nebius/nebius-physical-ai/"
        ), dockerfile
        assert re.search(r"@sha256:[0-9a-f]{64}$", base), dockerfile


def test_cosmos_reason_replaces_parent_npa_metadata_before_pip_check() -> None:
    text = (WORKBENCH / "cosmos3-reason/Dockerfile").read_text(encoding="utf-8")
    assert text.index("python -m pip uninstall -y npa") < text.index(
        "python -m pip check"
    )


def test_envgen_removes_unrelated_nonredistributable_parent_binary() -> None:
    text = (WORKBENCH / "sim2real-envgen/Dockerfile").read_text(encoding="utf-8")
    installer = (WORKBENCH / "common/install_workflow_runtime_prereqs.sh").read_text(
        encoding="utf-8"
    )
    assert "pip uninstall -y transformers imageio-ffmpeg" in text
    assert "moviepy imageio-ffmpeg" not in text
    assert "imageio_ffmpeg-0.6.0.tar.gz#sha256=" in text
    assert "IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg" in text
    assert "imageio_ffmpeg/binaries/ffmpeg*" in text
    assert "  ffmpeg \\" in installer
    assert "FROM ${BASE_IMAGE} AS sanitized" in text
    assert "FROM scratch AS runtime" in text
    assert "COPY --from=sanitized / /" in text
    assert text.index("FROM scratch AS runtime") < text.index(
        'LABEL npa.tool="envgen"'
    )
    assert (
        'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"'
        in text
    )
    for runtime_contract in (
        "NVIDIA_VISIBLE_DEVICES=all",
        "NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility",
        "CUDA_HOME=/usr/local/cuda",
        "NPA_GENESIS_HOME=/opt/genesis",
        "MUJOCO_GL=egl",
        "IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg",
        "NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}",
    ):
        assert runtime_contract in text


def test_genesis_workflow_runtime_upgrades_fixed_kernel_headers() -> None:
    installer = (WORKBENCH / "common/install_workflow_runtime_prereqs.sh").read_text(
        encoding="utf-8"
    )
    assert "linux-libc-dev=5.15.0-190.200" in installer
    for relative in (
        "sim2real-envgen/Dockerfile",
        "sim2real-eval/Dockerfile",
    ):
        text = (WORKBENCH / relative).read_text(encoding="utf-8")
        assert "ARG UBUNTU_SNAPSHOT=20260820T000000Z" in text, relative


def test_isaac_runtime_uses_system_ffmpeg_without_wheel_bundled_binary() -> None:
    installer = (WORKBENCH / "common/install_isaac_runtime_base.sh").read_text(
        encoding="utf-8"
    )
    dockerfile = (WORKBENCH / "isaac-lab/Dockerfile").read_text(encoding="utf-8")
    assert "  ffmpeg \\" in installer
    assert "--no-binary imageio-ffmpeg" in installer
    assert "imageio_ffmpeg/binaries/ffmpeg*" in installer
    assert "IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg" in dockerfile
