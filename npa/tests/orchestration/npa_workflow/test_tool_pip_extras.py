"""Per-toolRef npa extras in the rendered SkyPilot ``setup:`` block.

A stage that runs on SkyPilot's default image gets only the base ``npa`` install, so a
tool with optional dependencies (SONIC needs torch/onnx) fails at import time. The
renderer installs the matching ``npa[<extra>]`` from the same source tree it installed
npa from.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    assert_no_unresolved_placeholders,
    default_npa_setup,
    render_pip_extra_setup,
    render_setup_for_tool,
    render_skypilot_yaml,
    tool_pip_extra,
)
from npa.orchestration.npa_workflow.spec import load_spec

REPO_ROOT = Path(__file__).resolve().parents[4]
SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"


@pytest.mark.parametrize(
    ("tool_ref", "expected"),
    [
        ("workbench.sonic.export", "sonic"),
        ("workbench.sonic.train", "sonic"),
        ("workbench.sonic", "sonic"),
        ("workbench.token_factory.caption", ""),
        ("workbench.mjlab.evaluate", ""),
        ("", ""),
    ],
)
def test_tool_pip_extra_longest_prefix_match(tool_ref: str, expected: str) -> None:
    assert tool_pip_extra(tool_ref) == expected


def test_extra_setup_installs_from_the_recorded_source_root() -> None:
    script = render_pip_extra_setup("sonic")

    assert "/tmp/npa-src-root" in script
    assert "/opt/nebius-physical-ai/npa" in script
    assert "npa_pip_install -e" in script
    assert "[sonic]" in script


def test_extra_setup_uses_no_braced_expansion() -> None:
    """Regression: a ``${var}`` in setup trips the rendered-YAML placeholder guard.

    The first live submit of ``sonic-export.yaml`` failed instantly with
    "rendered SkyPilot YAML still contains unresolved placeholders: ${npa_src_root}".
    """

    assert_no_unresolved_placeholders(render_pip_extra_setup("sonic"))
    assert "${" not in render_pip_extra_setup("sonic")


def test_no_extra_renders_nothing() -> None:
    assert render_pip_extra_setup("") == ""


def test_default_setup_records_the_source_root_for_both_install_paths() -> None:
    setup = default_npa_setup()

    assert "npa_record_src_root /opt/nebius-physical-ai/npa" in setup
    assert "npa_record_src_root /tmp/npa-src" in setup


def test_setup_for_a_sonic_tool_ref_includes_the_extra() -> None:
    setup = render_setup_for_tool(
        "workbench.sonic.export", config={}, options=SkypilotRenderOptions()
    )

    assert "[sonic]" in setup
    # The base install still comes first; the extra layers on top of it.
    assert setup.index("npa_pip_install") < setup.index("[sonic]")


def test_setup_for_an_unrelated_tool_ref_has_no_extra() -> None:
    setup = render_setup_for_tool(
        "workbench.token_factory.caption", config={}, options=SkypilotRenderOptions()
    )

    assert "[sonic]" not in setup


def test_setup_installs_declarative_allowlisted_viz_extra() -> None:
    setup = render_setup_for_tool(
        "workbench.byof.repo",
        config={"solution_name": "any-solution", "pip_extra": "viz"},
        options=SkypilotRenderOptions(),
    )

    assert "npa[viz]" in setup
    assert "npa_pip_install -e" in setup
    assert "viz @" not in setup


def test_setup_for_other_byof_does_not_install_viz_extra() -> None:
    setup = render_setup_for_tool(
        "workbench.byof.repo",
        config={"solution_name": "open-dreamer"},
        options=SkypilotRenderOptions(),
    )

    assert "npa[viz]" not in setup


def test_declarative_extra_rejects_untrusted_package_strings() -> None:
    with pytest.raises(NpaWorkflowError, match="is not allowed"):
        render_setup_for_tool(
            "workbench.byof.repo",
            config={"pip_extra": "viz @ https://example.invalid/payload.whl"},
            options=SkypilotRenderOptions(),
        )


@pytest.mark.parametrize(
    "spec_name", ["sonic-export.yaml", "sonic-eval.yaml", "sonic-train.yaml"]
)
def test_shipped_sonic_specs_render_with_the_extra_and_no_placeholders(
    spec_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end-to-end guard the live failure would have caught offline."""

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")
    spec = load_spec(SPECS / spec_name)
    plan = build_plan(spec, run_id="render-check")

    # `image_overrides={"*": ""}` is the offline equivalent of the live harness's
    # `--image none`: it keeps the renderer from resolving a registry image (which
    # would need real Nebius credentials in a unit test) and is exactly the path where
    # the npa extra matters, since there is no baked image to provide torch.
    yaml_text = render_skypilot_yaml(
        spec,
        plan,
        run_id="render-check",
        options=SkypilotRenderOptions(image_overrides={"*": ""}),
    )

    assert "[sonic]" in yaml_text
    assert_no_unresolved_placeholders(yaml_text)
