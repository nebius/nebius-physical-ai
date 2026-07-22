"""Guard the sim2real no-registry image-pin fallbacks against drift.

``sim2real/constants.py`` carries no-registry fallback tags that must stay in
sync with the canonical ``[tool.npa.supported-tools]`` pins in ``pyproject.toml``
(mirrored by ``deploy/images.py``). The tag-audit script only matches
fully-qualified ``npa-<tool>:<tag>`` references, so it cannot catch drift in
these bare constants — this test closes that gap.
"""

from __future__ import annotations

import pytest

from npa.deploy.images import supported_tool_version
from npa.workflows.sim2real import constants


@pytest.mark.parametrize(
    ("constant_name", "tool"),
    [
        ("DEFAULT_ENVGEN_TAG", "envgen"),
        ("DEFAULT_REFERENCE_POLICY_TAG", "reference-policy"),
        ("DEFAULT_TRAINER_TAG", "lerobot-vlm-rl"),
        ("DEFAULT_EVAL_TAG", "loop-eval"),
    ],
)
def test_sim2real_constant_matches_supported_tool_version(
    constant_name: str, tool: str
) -> None:
    constant_value = getattr(constants, constant_name)
    assert constant_value == supported_tool_version(tool), (
        f"{constant_name}={constant_value!r} drifted from canonical "
        f"{tool}={supported_tool_version(tool)!r} (pyproject supported-tools)"
    )
