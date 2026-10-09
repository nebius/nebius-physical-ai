"""Keep Genesis dependency correction inside the original installation layer."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
GENESIS = ROOT / "npa/docker/workbench/genesis"
VARIANTS = (
    ("Dockerfile", "0.25.2", "/opt/genesis/venv/lib/python3.10/site-packages"),
    ("Dockerfile.sm120", "0.26.0", "/opt/npa/venv/lib/python3.11/site-packages"),
)
CORRECTION = "/opt/genesis/remove-scikit-image-recipe.py"


def _instructions(text):
    return [
        (match.group(1), match.group(2))
        for line in text.replace("\\\n", " ").splitlines()
        if (match := re.match(r"^([A-Z]+)\s+(.*)$", line))
    ]


def _assert_installation_contract(text, version, site_packages):
    instructions = _instructions(text)
    install_index, installation = next(
        (index, body)
        for index, (kind, body) in enumerate(instructions)
        if kind == "RUN" and '"genesis-world==' in body
    )
    assert any(
        kind == "ENV" and re.search(r"\bPIP_NO_CACHE_DIR=1\b", body)
        for kind, body in instructions[:install_index]
    )
    assert any(
        kind == "COPY"
        and "docker/workbench/curobo/remove_scikit_image_recipe.py" in body
        and CORRECTION in body
        for kind, body in instructions[:install_index]
    )
    assert f'"scikit-image=={version}"' in installation
    assert f"python {CORRECTION}" in installation
    assert f"--site-packages {site_packages}" in installation
    assert installation.rfind("pip install") < installation.index(
        f"python {CORRECTION}"
    )
    assert "dependency-source-correction.json" in installation


@pytest.mark.parametrize("dockerfile,version,site_packages", VARIANTS)
def test_genesis_ffmpeg_contract_avoids_bundled_wheel_executables(
    dockerfile, version, site_packages
):
    text = (GENESIS / dockerfile).read_text()
    instructions = _instructions(text)
    index, installation = next(
        (index, body)
        for index, (kind, body) in enumerate(instructions)
        if kind == "RUN" and '"genesis-world==' in body
    )
    assert any(
        kind == "ENV" and "IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg" in body
        for kind, body in instructions[:index]
    )
    assert "pip install --no-binary imageio-ffmpeg" in installation
    assert '"imageio-ffmpeg==0.6.0"' in installation
    assert "imageio_ffmpeg/binaries/ffmpeg*' -print -quit" in installation
    assert 'imageio_ffmpeg.get_ffmpeg_exe() == "/usr/bin/ffmpeg"' in installation


@pytest.mark.parametrize("dockerfile,version,site_packages", VARIANTS)
def test_genesis_corrects_reviewed_source_before_the_installation_layer_commits(
    dockerfile, version, site_packages
):
    _assert_installation_contract(
        (GENESIS / dockerfile).read_text(), version, site_packages
    )


@pytest.mark.parametrize("dockerfile,version,site_packages", VARIANTS)
def test_correction_in_a_child_layer_is_rejected(dockerfile, version, site_packages):
    text = (GENESIS / dockerfile).read_text()
    delayed = text.replace(
        f"    && python {CORRECTION}", f"\nRUN python {CORRECTION}"
    ).replace(
        f"    && /opt/genesis/venv/bin/python {CORRECTION}",
        f"\nRUN /opt/genesis/venv/bin/python {CORRECTION}",
    )
    assert delayed != text
    with pytest.raises(AssertionError):
        _assert_installation_contract(delayed, version, site_packages)


@pytest.mark.parametrize("dockerfile,version,site_packages", VARIANTS)
def test_dependency_wheel_caches_cannot_survive_the_correction(
    dockerfile, version, site_packages
):
    text = (
        (GENESIS / dockerfile)
        .read_text()
        .replace("PIP_NO_CACHE_DIR=1", "PIP_NO_CACHE_DIR=0")
    )
    with pytest.raises(AssertionError):
        _assert_installation_contract(text, version, site_packages)


@pytest.mark.parametrize("dockerfile,version,site_packages", VARIANTS)
def test_a_different_dependency_version_is_rejected(dockerfile, version, site_packages):
    other = "0.26.0" if version == "0.25.2" else "0.25.2"
    text = (
        (GENESIS / dockerfile)
        .read_text()
        .replace(f'"scikit-image=={version}"', f'"scikit-image=={other}"')
    )
    with pytest.raises(AssertionError):
        _assert_installation_contract(text, version, site_packages)
