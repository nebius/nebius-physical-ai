"""Static contract checks for the operator-private OpenWAM image recipe."""

from __future__ import annotations

import json
from pathlib import Path

from npa.deploy import images


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
    assert "COPY --chmod=0755 src/npa /opt/npa/src/npa" in dockerfile
    assert (
        "sudo -u ubuntu -H /opt/openwam-venv/bin/python -m npa.workflows.openwam_pipeline"
        in dockerfile
    )


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


def test_openwam_datacenter_manifest_does_not_claim_a_device_result() -> None:
    manifest = json.loads(
        (ROOT / "npa" / "docker" / "workbench" / "blackwell-dc-images.json").read_text(
            encoding="utf-8"
        )
    )
    entry = next(item for item in manifest["images"] if item["name"] == "npa-openwam")

    assert entry["dockerfile"] == "openwam/Dockerfile"
    assert entry["verdict"] == "unknown"
    assert entry["validation"] == "pending-gpu"
    assert entry["redistribution"] == "unvalidated"
    assert (
        "RTX evidence must never be treated as B200/B300 equivalence" in entry["notes"]
    )


def test_openwam_is_private_neutral_until_a_public_release_is_accepted() -> None:
    """A private candidate does not authorize a public or generic image route."""

    assert "openwam" in images.NEUTRAL_UNBUILT_CANDIDATE_TOOLS
    assert images.supported_tool_version("openwam") == "operator-private-unbuilt"
    assert not images.is_publicly_redistributable("openwam")
    assert "openwam" not in images.CONTAINER_IMAGE_NAMES
