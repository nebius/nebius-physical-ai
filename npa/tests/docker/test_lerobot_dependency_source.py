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
    optional_branch, default_branch = body.split(
        'elif [ "${LEROBOT_VERSION}" = "0.5.1" ]; then', 1
    )
    default_branch = default_branch.split("else", 1)[0]
    assert optional_branch.count('"scikit-image==0.26.0"') == 0
    assert default_branch.count('"scikit-image==0.26.0"') == 1
    assert "pip uninstall -y wandb" not in optional_branch
    assert "pip uninstall -y wandb" not in default_branch
    assert f"python {CORRECTION}" in optional_branch
    assert f"python {CORRECTION}" in default_branch
    assert (
        "--site-packages /opt/lerobot/venv/lib/python3.12/site-packages"
        in default_branch
    )
    assert (
        "--site-packages /opt/lerobot/venv/lib/python3.12/site-packages"
        in optional_branch
    )
    assert "dependency-source-correction.json" in optional_branch
    assert "dependency-source-correction.json" in default_branch
    return body


def test_both_versions_correct_the_reviewed_source_in_the_original_install_layer():
    text = DOCKERFILE.read_text()
    _installation(text)
    assert "NPA_LEROBOT_INTEGRATION_PROFILE" not in text
    assert "/opt/lerobot/source-integration.json" in text


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
