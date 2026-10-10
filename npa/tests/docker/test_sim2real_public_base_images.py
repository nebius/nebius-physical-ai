"""Keep public Sim2Real development builds independent of local image tags."""

from __future__ import annotations

import re
from pathlib import Path

from packaging.version import Version


ROOT = Path(__file__).resolve().parents[3]
WORKBENCH = ROOT / "npa" / "docker" / "workbench"


def test_generic_torch_images_require_patched_versions_and_complete_dependency_checks():
    for relative, floors in (
        (
            "base/cuda13-blackwell/Dockerfile",
            {
                "TORCH_VERSION": "2.13.0",
                "TORCHVISION_VERSION": "0.28.0",
                "TORCHAUDIO_VERSION": "2.11.0",
            },
        ),
        ("cosmos-curate/Dockerfile", {"TORCH_VERSION": "2.13.0"}),
    ):
        text = (WORKBENCH / relative).read_text()
        for variable, minimum in floors.items():
            pin = re.search(rf"^ARG {variable}=(\S+)$", text, re.MULTILINE)
            assert pin and Version(pin.group(1)) >= Version(minimum), relative
        assert text.index("python -m pip check") > text.rindex(
            "python -m pip install"
        ), relative


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
        assert base.startswith("ghcr.io/nebius/nebius-physical-ai/"), dockerfile
        assert re.search(r"@sha256:[0-9a-f]{64}$", base), dockerfile


def test_cosmos_reason_replaces_parent_npa_metadata_before_pip_check() -> None:
    text = (WORKBENCH / "cosmos3-reason/Dockerfile").read_text(encoding="utf-8")
    assert text.index("python -m pip uninstall -y npa") < text.index(
        "python -m pip check"
    )


def test_sim2real_cpu_images_install_security_fixed_perl() -> None:
    for relative in ("sim2real-control/Dockerfile", "rerun-viewer/Dockerfile"):
        text = (WORKBENCH / relative).read_text(encoding="utf-8")
        assert "ARG DEBIAN_SNAPSHOT=20261002T000000Z" in text, relative
        assert "ARG MIN_PERL_BASE_VERSION=5.40.1-6+deb13u1" in text, relative
        assert '"perl-base=${MIN_PERL_BASE_VERSION}"' in text, relative


def test_envgen_clears_inherited_source_before_copying_exact_revision() -> None:
    text = (WORKBENCH / "sim2real-envgen/Dockerfile").read_text(encoding="utf-8")
    removal = text.index("rm -rf /opt/npa/src /opt/npa/workflows")
    source_copy = text.index("COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa")
    catalog_copy = text.index("COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows")
    assert text.index("test ! -L /opt/npa") < removal < source_copy < catalog_copy
    assert catalog_copy < text.index("FROM scratch AS runtime")


def test_cpu_images_update_system_and_viewer_bootstrap_dependencies() -> None:
    lock = (WORKBENCH / "common/sim2real-cpu-build-requirements.txt").read_text()
    assert "pip==26.2.1" in lock
    assert "setuptools==84.0.0" in lock
    assert "wheel==0.48.0" in lock
    assert "msgpack==1.2.1" in lock
    assert "urllib3==2.8.0" in lock
    for relative in ("sim2real-control/Dockerfile", "rerun-viewer/Dockerfile"):
        text = (WORKBENCH / relative).read_text()
        bootstrap = text.index(
            "python -m pip install --no-cache-dir --no-deps --upgrade"
        )
        assert bootstrap < text.index("USER ubuntu")
        assert "python /opt/npa/update_pip_vendor.py" in text
    viewer = (WORKBENCH / "rerun-viewer/Dockerfile").read_text()
    assert viewer.count("python /opt/npa/update_pip_vendor.py") == 2


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
    assert text.index("FROM scratch AS runtime") < text.index('LABEL npa.tool="envgen"')
    assert 'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"' in text
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


def test_envgen_removes_optional_forbidden_and_vulnerable_parent_tools() -> None:
    text = (WORKBENCH / "sim2real-envgen/Dockerfile").read_text(encoding="utf-8")
    sanitizer = (WORKBENCH / "common/sanitize_sim2real_envgen_parent.py").read_text(
        encoding="utf-8"
    )

    assert "pip uninstall -y transformers imageio-ffmpeg wandb tetgen" in text
    assert "sanitize-sim2real-envgen-parent.py" in text
    assert "COPY --chmod=0644 docker/workbench/common/envgen_compat/tetgen.py " in text
    assert "/opt/npa/compat/tetgen.py" in text
    assert "chmod 0755 /opt/npa/compat" in text
    assert "chmod 0644 /opt/npa/compat/tetgen.py" in text
    assert "PYTHONPATH=/opt/npa/compat:/opt/npa/src" in text
    assert '("genesis-world", "tetgen")' in sanitizer
    assert '("lerobot", "wandb")' in sanitizer
    assert "len(filtered) != len(lines) - 1" in sanitizer
    assert "rm -rf /opt/nvidia/nsight-compute" in text
    assert 'names.isdisjoint({"tetgen", "wandb"})' in text
    assert "test ! -e /opt/nvidia/nsight-compute" in text

    compat = (WORKBENCH / "common/envgen_compat/tetgen.py").read_text(encoding="utf-8")
    assert "class TetGen:" in compat
    assert "raise RuntimeError(" in compat


def test_genesis_workflow_runtime_upgrades_fixed_kernel_headers() -> None:
    installer = (WORKBENCH / "common/install_workflow_runtime_prereqs.sh").read_text(
        encoding="utf-8"
    )
    snapshot_config = (WORKBENCH / "common/configure_ubuntu_snapshot.sh").read_text(
        encoding="utf-8"
    )
    assert "snapshot.ubuntu.com/ubuntu/${snapshot}" in snapshot_config
    assert "ubuntu:22.04" in snapshot_config
    assert "ubuntu:24.04" in snapshot_config
    assert "configure-ubuntu-snapshot" in installer
    assert 'linux_libc_dev_version="5.15.0-190.200"' in installer
    assert 'linux_libc_dev_version="6.8.0-139.139"' in installer
    assert '"linux-libc-dev=${linux_libc_dev_version}"' in installer
    for relative in (
        "sim2real-envgen/Dockerfile",
        "sim2real-eval/Dockerfile",
    ):
        text = (WORKBENCH / relative).read_text(encoding="utf-8")
        snapshot = (
            "20261002T000000Z"
            if relative == "sim2real-envgen/Dockerfile"
            else "20260820T000000Z"
        )
        assert f"ARG UBUNTU_SNAPSHOT={snapshot}" in text, relative
        if relative == "sim2real-envgen/Dockerfile":
            assert "ARG LINUX_LIBC_DEV_VERSION=5.15.0-198.208" in text
            assert '"${UBUNTU_SNAPSHOT}" "${LINUX_LIBC_DEV_VERSION}"' in text
        assert "configure_ubuntu_snapshot.sh" in text, relative


def test_genesis_workflow_images_replace_vulnerable_parent_gitpython() -> None:
    requirements = (WORKBENCH / "common/sim2real-genesis-requirements.txt").read_text()
    pin = re.search(r"^GitPython==(\S+)$", requirements, re.MULTILINE)
    assert pin and Version(pin.group(1)) >= Version("3.1.62")
    for relative in (
        "sim2real-envgen/Dockerfile",
        "sim2real-eval/Dockerfile",
        "lerobot-vlm-rl/Dockerfile",
    ):
        text = (WORKBENCH / relative).read_text()
        install = text.index("-r /opt/npa/sim2real-genesis-requirements.txt")
        assert install < text.index("python -m pip check"), relative
        assert not re.search(r"GitPython\s*(?:@|==)", text, re.IGNORECASE), relative
        if relative == "sim2real-envgen/Dockerfile":
            assert install < text.index("FROM scratch AS runtime")
            assert f'm.version("GitPython") == "{pin.group(1)}"' in text


def test_isaac_runtime_uses_system_ffmpeg_without_wheel_bundled_binary() -> None:
    installer = (WORKBENCH / "common/install_isaac_runtime_base.sh").read_text(
        encoding="utf-8"
    )
    dockerfile = (WORKBENCH / "isaac-lab/Dockerfile").read_text(encoding="utf-8")
    assert "  ffmpeg \\" in installer
    assert "--no-binary imageio-ffmpeg" in installer
    assert "imageio_ffmpeg/binaries/ffmpeg*" in installer
    assert "IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg" in dockerfile
    assert "rm -rf /opt/nvidia/nsight-compute" in dockerfile
    assert "test ! -e /opt/nvidia/nsight-compute" in dockerfile
