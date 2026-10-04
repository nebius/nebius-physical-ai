"""Static contract checks for the operator-private OpenWAM image recipe."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_openwam_uses_system_ffmpeg_not_wheel_bundled_executables() -> None:
    dockerfile = (
        ROOT / "npa" / "docker" / "workbench" / "openwam" / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert "ffmpeg git git-lfs" in dockerfile
    assert "IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg" in dockerfile
    assert dockerfile.count("*/imageio_ffmpeg/binaries/ffmpeg*' -delete") == 2
    assert 'imageio_ffmpeg.get_ffmpeg_exe() == "/usr/bin/ffmpeg"' in dockerfile
    assert "UV_CACHE_DIR=/tmp/openwam-uv-cache" in dockerfile
    assert "rm -rf /tmp/openwam-uv-cache" in dockerfile


def test_openwam_recipe_retains_runtime_fetch_and_private_quarantine() -> None:
    dockerfile = (
        ROOT / "npa" / "docker" / "workbench" / "openwam" / "Dockerfile"
    ).read_text(encoding="utf-8")
    contract = (
        ROOT / "npa" / "docker" / "workbench" / "packaging-contract.yaml"
    ).read_text(encoding="utf-8")

    assert 'npa.weights="operator-runtime-fetch"' in dockerfile
    assert 'npa.dataset="operator-runtime-fetch"' in dockerfile
    assert 'org.nebius.npa.redistribution="unvalidated-operator-private"' in dockerfile
    assert "openwam:\n    dockerfile: openwam/Dockerfile" in contract
    assert (
        "redistribution: unvalidated"
        in contract.split("  openwam:", 1)[1].split("  libero:", 1)[0]
    )
