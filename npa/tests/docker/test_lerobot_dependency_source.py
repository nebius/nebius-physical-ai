"""Neither LeRobot version may retain corrected dependency bytes in an ancestor."""

from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa/docker/workbench/lerobot/Dockerfile"
CORRECTION = "/opt/lerobot/remove-scikit-image-recipe.py"


def _installation(text):
    instructions = [
        (match.group(1), match.group(2))
        for line in text.replace("\\\n", " ").splitlines()
        if (match := re.match(r"^([A-Z]+)\s+(.*)$", line))
    ]
    index, body = next(
        (index, body)
        for index, (kind, body) in enumerate(instructions)
        if kind == "RUN" and '"lerobot[' in body
    )
    assert any(
        kind == "ENV" and "PIP_NO_CACHE_DIR=1" in value
        for kind, value in instructions[:index]
    )
    assert any(
        kind == "COPY"
        and "docker/workbench/curobo/remove_scikit_image_recipe.py" in value
        and CORRECTION in value
        for kind, value in instructions[:index]
    )
    branches, correction = body.split(f"python {CORRECTION}", 1)
    assert branches.rfind("fi") > branches.rfind("pip install")
    assert branches.count('"scikit-image==0.26.0"') == 2
    assert (
        "--site-packages /opt/lerobot/venv/lib/python3.12/site-packages" in correction
    )
    assert "dependency-source-correction.json" in correction
    return body


def test_both_versions_correct_the_reviewed_source_in_the_original_install_layer():
    _installation(DOCKERFILE.read_text())


@pytest.mark.parametrize(
    "unsafe_change", ["later-layer", "wheel-cache", "unreviewed-version"]
)
def test_unsafe_dependency_installation_is_rejected(unsafe_change):
    text = DOCKERFILE.read_text()
    if unsafe_change == "later-layer":
        text = text.replace(
            f"    && /opt/lerobot/venv/bin/python {CORRECTION}",
            f"\nRUN /opt/lerobot/venv/bin/python {CORRECTION}",
        )
    elif unsafe_change == "wheel-cache":
        text = text.replace("PIP_NO_CACHE_DIR=1", "PIP_NO_CACHE_DIR=0")
    else:
        text = text.replace('"scikit-image==0.26.0"', '"scikit-image==0.27.0"')
    with pytest.raises((AssertionError, ValueError)):
        _installation(text)
