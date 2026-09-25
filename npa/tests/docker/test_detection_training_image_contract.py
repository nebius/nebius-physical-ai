"""Clean-build guards for the detection-training public image."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa/docker/workbench/detection-training/Dockerfile"


def test_detection_training_uses_an_immutable_ubuntu_snapshot() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG UBUNTU_SNAPSHOT=20260820T000000Z" in text
    assert "ARG LINUX_LIBC_DEV_VERSION=6.8.0-138.138" in text
    assert "configure_ubuntu_snapshot.sh" in text
    assert 'configure-ubuntu-snapshot "${UBUNTU_SNAPSHOT}"' in text
    assert "archive.ubuntu.com" not in text
    assert "security.ubuntu.com" not in text


def test_detection_training_checks_its_dependency_overlay() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert text.index("pip install --no-cache-dir") < text.index("python -m pip check")
