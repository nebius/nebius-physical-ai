"""Protect MJLab's licensed runtime boundary and immutable build inputs."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/mjlab"


def test_cudnn_sdk_is_removed_in_the_install_layer():
    text = (IMAGE / "Dockerfile").read_text()
    instructions = re.split(r"(?m)^(?=[A-Z]+ )", text)
    install = next(line for line in instructions if "RUN pip install" in line)
    assert "--require-hashes" in install
    assert "python3 /opt/filter_cudnn_runtime.py" in install
    assert "imageio_ffmpeg/binaries/ffmpeg" in install
    assert "nvidia-cudnn-cu13==9.20.0.48" in (IMAGE / "requirements.lock").read_text()


def test_snapshot_build_carries_every_copied_dependency():
    dockerfile = (IMAGE / "Dockerfile").read_text()
    build = (IMAGE / "build.sh").read_text()
    for path in (
        "docker/workbench/curobo/filter_cudnn_runtime.py",
        "docker/workbench/open3d/notices/mcap-LICENSE.txt",
        "docker/workbench/common/workflow_runtime_entrypoint.sh",
    ):
        assert path in dockerfile
        assert "npa/" + path in build


def test_notices_are_bound_to_the_installed_versions():
    text = (IMAGE / "Dockerfile").read_text()
    assert 'm.version("mcap") == "1.4.0"' in text
    assert "nvidia-nvshmem-cu13==3.4.5" in (IMAGE / "requirements.lock").read_text()
    assert "43a87c0ff94ce3196011ff75e17fbee96933c9e1d511557659ece8a326f95e8f" in text
