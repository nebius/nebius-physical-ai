"""EnvGen must isolate its fresh closure before flattening the delivery layer."""

from pathlib import Path

import pytest


DOCKERFILE = (
    Path(__file__).resolve().parents[2] / "docker/workbench/sim2real-envgen/Dockerfile"
)


def _require_pre_layer_isolation(text):
    intermediate, final = text.split("FROM scratch AS runtime", 1)
    assert "FROM ${BASE_IMAGE} AS sanitized" in intermediate
    # The dedicated role does not install scikit-image.  It must remove the
    # complete parent NPA tree before copying current source, rather than
    # retaining an ancestor interpreter and trying to clean its bytes later.
    command = "rm -rf /opt/npa\n"
    assert command in intermediate
    assert intermediate.index(command) < intermediate.index(
        "COPY --chown=ubuntu:ubuntu pyproject.toml"
    )
    assert "remove-scikit-image-recipe.py" not in intermediate
    assert "scikit-image" not in intermediate
    assert "COPY --from=sanitized / /" in final
    assert "FROM " not in final
    assert "COPY --from=sanitized / /" not in intermediate
    assert "remove-scikit-image-recipe.py" not in final
    assert "COPY --from=" not in final.replace("COPY --from=sanitized / /", "")


def test_parent_npa_tree_is_removed_before_building_the_scratch_delivery_layer():
    _require_pre_layer_isolation(DOCKERFILE.read_text())


@pytest.mark.parametrize(
    "case", ("parent-retained", "later-deletion", "extra-ancestor-copy")
)
def test_recipe_must_not_claim_a_later_deletion_clears_ancestor_bytes(case):
    text = DOCKERFILE.read_text()
    if case == "parent-retained":
        text = text.replace("rm -rf /opt/npa", "rm -rf /opt/npa/src")
    elif case == "later-deletion":
        text = text.replace("rm -rf /opt/npa", "true")
        text = text.replace(
            "FROM scratch AS runtime", "RUN rm -rf /opt/npa\n\nFROM scratch AS runtime"
        )
    else:
        text += "\nCOPY --from=sanitized /tmp /inherited-cache\n"
    with pytest.raises((AssertionError, ValueError)):
        _require_pre_layer_isolation(text)
