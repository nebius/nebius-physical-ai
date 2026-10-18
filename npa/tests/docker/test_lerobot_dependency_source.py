"""Keep the original optional LeRobot dependency recipe out of image layers."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa/docker/workbench/lerobot/Dockerfile"


def test_optional_recipe_is_corrected_in_the_installing_layer() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    install = text.index("RUN python3.12 -m venv")
    next_instruction = text.index("\nENV PATH", install)
    correction = text.index(
        "/opt/lerobot/venv/bin/python /opt/lerobot/remove-scikit-image-recipe.py"
    )
    assert install < correction < next_instruction
    scope = text.rindex(
        '&& if [ "${LEROBOT_VERSION}" = "0.6.0" ]; then', install, correction
    )
    block = text[scope : text.index("; \\\n       fi", correction)]
    assert "--site-packages /opt/lerobot/venv/lib/python3.12/site-packages" in block
    assert "> /opt/lerobot/dependency-source-correction.json" in block
    assert text.index("PIP_NO_CACHE_DIR=1") < install
    assert (
        text.index(
            "COPY --chown=ubuntu:ubuntu docker/workbench/curobo/remove_scikit_image_recipe.py"
        )
        < install
    )
