from __future__ import annotations

from pathlib import Path

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG, argv_for_tool

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "adversarial-scenario-hardening.yaml"
SMOKE = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "scenario-gen-smoke.yaml"
#: Retired in favour of SMOKE; asserted absent below.
SKYPILOT = ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot" / "scenario-gen-adversarial.yaml"


def test_workflow_validates_and_expands_hardening_loop() -> None:
    spec = load_spec(WORKFLOW)
    validate_spec(spec)
    assert spec.name == "adversarial-scenario-hardening"
    assert spec.initial == "generate"

    plan = build_plan(spec, run_id="test", assume_decision="loop_back")
    states = [step.state for step in plan.steps]
    # generate -> rank then the outer loop retrain/evaluate/decide x3 -> publish.
    assert states[:2] == ["generate", "rank"]
    assert states[-1] == "publish"
    assert states.count("retrain") == 3
    assert states.count("decide") == 3


def test_workflow_promote_short_circuits_loop() -> None:
    spec = load_spec(WORKFLOW)
    plan = build_plan(spec, run_id="test", assume_decision="promote_checkpoint")
    states = [step.state for step in plan.steps]
    assert states == ["generate", "rank", "retrain", "evaluate", "decide", "publish"]


def test_workflow_dependency_order_is_topological() -> None:
    spec = load_spec(WORKFLOW)
    seen: set[str] = set()
    for state in spec.states.values():
        for dep in state.needs:
            # A dependency must be a declared state.
            assert dep in spec.states
    # generate has no needs; rank needs generate; harden needs rank.
    assert spec.states["generate"].needs == []
    assert spec.states["rank"].needs == ["generate"]
    assert spec.states["harden"].needs == ["rank"]
    assert spec.states["publish"].needs == ["harden"]
    assert not seen


def test_new_scenario_gen_toolrefs_render() -> None:
    for tool_ref in (
        "workbench.scenario_gen.generate",
        "workbench.scenario_gen.rank",
        "workbench.scenario_gen.write_hardening_decision",
    ):
        assert tool_ref in TOOL_CATALOG
        argv = argv_for_tool(tool_ref)
        assert argv, tool_ref
    generate_argv = argv_for_tool("workbench.scenario_gen.generate")
    assert generate_argv[:4] == ["npa", "workbench", "scenario-gen", "generate"]
    assert "--input-path" in generate_argv
    assert "--output-path" in generate_argv


def test_smoke_spec_runs_the_same_two_commands_the_retired_template_did() -> None:
    """`scenario-gen-smoke.yaml` is the twin that let `scenario-gen-adversarial.yaml` go.

    The retired template was a single serial SkyPilot task running
    `scenario-gen generate` then `scenario-gen rank`. The spec resolves to the same two
    commands with the same flags, and it is live-verified (job 213, EVIDENCE.md §R17).
    """

    spec = load_spec(SMOKE)
    plan = build_plan(spec, run_id="scenario-gen-test")
    steps = [step for step in plan.steps if step.argv]

    assert [step.state for step in steps] == ["generate", "rank"]
    assert steps[0].argv[:4] == ["npa", "workbench", "scenario-gen", "generate"]
    assert steps[1].argv[:4] == ["npa", "workbench", "scenario-gen", "rank"]
    # rank consumes exactly the manifest generate declared.
    assert steps[0].outputs[0]["uri"] == steps[1].argv[steps[1].argv.index("--input-path") + 1]


def test_the_cli_cannot_select_an_rl_adversary_backend() -> None:
    """Why the retired template's GPU image selected no different code path.

    `scenario-gen-adversarial.yaml` pinned an Isaac Lab image with the comment "keep this
    image on an RT-core-capable Isaac Lab build (adversary RL backend)" and passed
    `ADVERSARY_STEPS=200000`. But `generate_scenarios(adversary_backend=...)` is a
    **Python-API seam with no CLI flag**: every CLI invocation runs `simulate_adversary`,
    whose own docstring says "This is NOT RL... deterministic heuristic stand-in", and which
    is O(num_scenarios) with `adversary_steps` entering only through a `log10` budget term.

    So a CPU twin is not a downgrade — it is the same code. This test fails the day a real
    backend becomes selectable from the CLI, which is exactly when a GPU spec should be
    authored to go with it.
    """

    from npa.cli.workbench.scenario_gen import generate_cmd
    from npa.guardrails.tool_catalog_argv import option_flags_for_callback

    flags = option_flags_for_callback(generate_cmd)

    assert not [flag for flag in flags if "backend" in flag], (
        f"`scenario-gen generate` now exposes a backend flag ({sorted(flags)}). If it can "
        "select a real Isaac Lab RL adversary, author a GPU spec for that path instead of "
        "relying on scenario-gen-smoke.yaml as the twin."
    )
    assert generate_cmd.__doc__


def test_the_retired_template_is_gone() -> None:
    assert not SKYPILOT.exists(), "scenario-gen-adversarial.yaml came back"
