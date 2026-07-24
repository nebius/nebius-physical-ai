from __future__ import annotations

from pathlib import Path

import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG, argv_for_tool

ROOT = Path(__file__).resolve().parents[3]
HARDENING = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "hardening-with-insights.yaml"
SMOKE = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "insights-smoke.yaml"
SKYPILOT = ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot" / "insights-aggregate.yaml"


def test_hardening_with_insights_validates_and_appends_insights_stages() -> None:
    spec = load_spec(HARDENING)
    validate_spec(spec)
    assert spec.name == "hardening-with-insights"
    assert spec.initial == "generate"

    promote = [step.state for step in build_plan(spec, run_id="t", assume_decision="promote_checkpoint").steps]
    assert promote == ["generate", "rank", "retrain", "evaluate", "decide", "publish", "aggregate", "dashboard"]

    loop = [step.state for step in build_plan(spec, run_id="t", assume_decision="loop_back").steps]
    assert loop[:2] == ["generate", "rank"]
    assert loop.count("retrain") == 3
    assert loop[-2:] == ["aggregate", "dashboard"]


def test_hardening_dependency_order_carries_lineage() -> None:
    spec = load_spec(HARDENING)
    assert spec.states["aggregate"].needs == ["publish"]
    assert spec.states["dashboard"].needs == ["aggregate"]
    assert spec.states["dashboard"].terminal is True
    for name in ("aggregate", "dashboard"):
        assert spec.states[name].inputs, name


def test_insights_smoke_validates_and_is_cpu_only() -> None:
    spec = load_spec(SMOKE)
    validate_spec(spec)
    assert spec.name == "insights-smoke"
    states = [step.state for step in build_plan(spec, run_id="t").steps]
    assert states == ["ingest", "compare", "dashboard"]
    for state in spec.states.values():
        assert state.resources == "cpu"


def test_new_insights_toolrefs_render() -> None:
    for tool_ref in (
        "workbench.insights.record",
        "workbench.insights.ingest_run",
        "workbench.insights.compare",
        "workbench.insights.dashboard",
    ):
        assert tool_ref in TOOL_CATALOG
        assert argv_for_tool(tool_ref)
    ingest_argv = argv_for_tool("workbench.insights.ingest_run")
    assert ingest_argv[:4] == ["npa", "workbench", "insights", "ingest-run"]
    assert "--input-path" in ingest_argv
    assert "--output-path" in ingest_argv


def test_skypilot_yaml_is_cpu_and_headless() -> None:
    docs = [doc for doc in yaml.safe_load_all(SKYPILOT.read_text()) if doc is not None]
    assert docs[0]["name"] == "insights-aggregate"
    assert docs[0]["execution"] == "serial"
    task = docs[1]
    assert task["resources"]["cloud"] == "kubernetes"
    assert "accelerators" not in task["resources"]
    assert task["envs"]["HEADLESS"] == "1"
    assert "npa workbench insights ingest-run" in task["run"]
    assert "npa workbench insights dashboard" in task["run"]
