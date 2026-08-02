"""Token Factory workflow specs: zero-GPU, real CLI surface, key check in setup.

These replace the raw-SkyPilot shape assertions for the four retired
`token-factory-*` / `vlm-eval-token-factory` templates. The equivalent contract on the
`npa.workflow` side is:

* the stage's resource profile requests **no accelerator** (hosted inference);
* the ``toolRef`` argv invokes the real CLI command and its real flags;
* the renderer's ``setup:`` fails fast when ``NEBIUS_TOKEN_FACTORY_KEY`` is absent,
  which is what the templates' inline ``if [[ -z ... ]]`` guard did;
* nothing serves a model locally (no vLLM install).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_setup_for_tool,
)
from npa.orchestration.npa_workflow.spec import load_spec

ROOT = Path(__file__).resolve().parents[3]
SPECS = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"


def _only_step(spec_name: str):
    spec = load_spec(SPECS / spec_name)
    steps = build_plan(spec, run_id="probe").steps
    assert len(steps) == 1, f"{spec_name} should be a single-stage spec"
    return spec, steps[0]


def _profile(spec, step) -> dict:
    return spec.resources[step.resources]


@pytest.mark.parametrize(
    ("spec_name", "command", "flags"),
    [
        (
            "token-factory-caption.yaml",
            "npa workbench token-factory caption",
            ("--input-path", "--output-path", "--model", "--max-images", "--max-tokens"),
        ),
        (
            "token-factory-generate.yaml",
            "npa workbench token-factory generate",
            ("--input-path", "--output-path", "--model", "--max-tokens"),
        ),
        (
            "token-factory-cosmos-reason.yaml",
            "npa workbench token-factory reason",
            ("--input-path", "--output-path"),
        ),
        (
            "vlm-eval-token-factory.yaml",
            "npa workbench vlm-eval run",
            ("--input-path", "--output-path", "--backend"),
        ),
    ],
)
def test_token_factory_specs_are_cpu_only_and_run_the_real_cli(
    spec_name: str, command: str, flags: tuple[str, ...]
) -> None:
    spec, step = _only_step(spec_name)
    argv = " ".join(step.argv)
    profile = _profile(spec, step)

    assert profile["cloud"] == "kubernetes"
    assert "accelerators" not in profile, f"{spec_name} is hosted inference; it needs no GPU"
    assert command in argv
    for flag in flags:
        assert flag in argv, f"{spec_name} argv is missing {flag}"


@pytest.mark.parametrize(
    "spec_name",
    [
        "token-factory-caption.yaml",
        "token-factory-generate.yaml",
        "token-factory-cosmos-reason.yaml",
    ],
)
def test_token_factory_setup_requires_the_api_key_and_serves_nothing(spec_name: str) -> None:
    spec, step = _only_step(spec_name)

    setup = render_setup_for_tool(
        step.tool_ref, config=spec.config, options=SkypilotRenderOptions()
    )

    assert "NEBIUS_TOKEN_FACTORY_KEY is required" in setup
    assert "vllm" not in setup


#: Flags the retired templates passed that the toolRef argv does NOT carry, so the
#: tool's own defaults apply. Pinned so the gap is visible and can only shrink; see the
#: `spec_gap` discussion in npa/src/npa/guardrails/three_tier.py.
TEMPLATE_ONLY_FLAGS = {
    "token-factory-caption.yaml": ("--instruction",),
    "token-factory-generate.yaml": ("--system-prompt", "--max-prompts"),
    "token-factory-cosmos-reason.yaml": ("--model", "--task", "--max-images"),
    "vlm-eval-token-factory.yaml": (
        "--task",
        "--model",
        "--api-key-env",
        "--frame-selection",
        "--max-frames",
        "--success-threshold",
        "--timeout-s",
    ),
}


@pytest.mark.parametrize("spec_name", sorted(TEMPLATE_ONLY_FLAGS))
def test_template_only_flags_are_still_absent_from_the_spec_surface(spec_name: str) -> None:
    """Pin the capability delta against the retired templates.

    A spec author cannot set these today; the tool's CLI defaults apply instead. When a
    flag is wired into the toolRef argv, delete it from this list deliberately.
    """

    _, step = _only_step(spec_name)
    argv = set(step.argv)

    still_missing = tuple(flag for flag in TEMPLATE_ONLY_FLAGS[spec_name] if flag not in argv)
    assert still_missing == TEMPLATE_ONLY_FLAGS[spec_name], (
        f"{spec_name}: the toolRef argv now carries "
        f"{sorted(set(TEMPLATE_ONLY_FLAGS[spec_name]) - set(still_missing))}; "
        "remove them from TEMPLATE_ONLY_FLAGS"
    )


def test_cosmos_reason_spec_reaches_the_hosted_reasoner_by_cli_default() -> None:
    """`--model` is not in this toolRef's argv, so the CLI default must be the reasoner."""

    from npa.clients.token_factory import DEFAULT_REASONER_MODEL

    _, step = _only_step("token-factory-cosmos-reason.yaml")

    assert "--model" not in step.argv
    assert DEFAULT_REASONER_MODEL == "nvidia/Cosmos3-Super-Reasoner"


def test_vlm_eval_token_factory_spec_uses_the_hosted_api_backend() -> None:
    """The `api` backend is what makes this the always-runnable VLM eval reference.

    `vlm-eval-single.yaml` asks for `self-hosted`, and nothing in that spec starts a
    vLLM server, so it fails live with `Connection refused` (pre-existing gap).
    """

    spec, step = _only_step("vlm-eval-token-factory.yaml")
    argv = step.argv

    assert spec.config["vlm_backend"] == "api"
    assert argv[argv.index("--backend") + 1] == "api"

    setup = render_setup_for_tool(
        step.tool_ref, config=spec.config, options=SkypilotRenderOptions()
    )
    # The renderer only injects a vLLM install for the self-hosted backend.
    assert "vllm" not in setup
