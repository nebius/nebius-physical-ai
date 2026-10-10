"""The shared dependency correction must precede the flattened Envgen layer.

Native sanitizer behavior is covered in test_curobo_dependency_source.py.
"""

from pathlib import Path

import pytest


DOCKERFILE = (
    Path(__file__).resolve().parents[2] / "docker/workbench/sim2real-envgen/Dockerfile"
)


def _require_pre_layer_sanitization(text):
    intermediate, final = text.split("FROM scratch AS runtime", 1)
    assert "FROM ${BASE_IMAGE} AS sanitized" in intermediate
    assert "python /usr/local/lib/npa/remove-scikit-image-recipe.py" in intermediate
    assert "COPY --from=sanitized / /" in final
    assert "FROM " not in final
    assert "COPY --from=sanitized / /" not in intermediate
    assert "remove-scikit-image-recipe.py" not in final
    assert "COPY --from=" not in final.replace("COPY --from=sanitized / /", "")


def test_example_is_removed_before_building_the_scratch_delivery_layer():
    _require_pre_layer_sanitization(DOCKERFILE.read_text())


@pytest.mark.parametrize(
    "case", ("parent-retained", "later-deletion", "extra-ancestor-copy")
)
def test_recipe_must_not_claim_a_later_deletion_clears_ancestor_bytes(case):
    text = DOCKERFILE.read_text()
    command = "python /usr/local/lib/npa/remove-scikit-image-recipe.py"
    if case == "parent-retained":
        text = text.replace("FROM scratch AS runtime", "FROM sanitized AS runtime")
    elif case == "later-deletion":
        text = text.replace(command, "true") + "\nRUN " + command + "\n"
    else:
        text += "\nCOPY --from=sanitized /tmp /inherited-cache\n"
    with pytest.raises((AssertionError, ValueError)):
        _require_pre_layer_sanitization(text)
