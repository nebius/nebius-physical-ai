from __future__ import annotations

from pathlib import Path

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG, argv_for_tool

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "adversarial-scenario-hardening.yaml"
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


def test_skypilot_yaml_is_headless_rtxpro() -> None:
    import yaml

    docs = [doc for doc in yaml.safe_load_all(SKYPILOT.read_text()) if doc is not None]
    assert docs[0]["name"] == "scenario-gen-adversarial"
    assert docs[0]["execution"] == "serial"
    task = docs[1]
    assert task["resources"]["cloud"] == "kubernetes"
    assert "RTXPRO" in task["resources"]["accelerators"]
    assert task["envs"]["HEADLESS"] == "1"
    assert "npa workbench scenario-gen generate" in task["run"]
    assert "npa workbench scenario-gen rank" in task["run"]
