from __future__ import annotations

from pathlib import Path

import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG, argv_for_tool

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "dataset-ingest-curate.yaml"
SKYPILOT = ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "dataset-ingest-curate.yaml"


def test_workflow_validates_and_gates_on_quality() -> None:
    spec = load_spec(WORKFLOW)
    validate_spec(spec)
    assert spec.name == "dataset-ingest-curate"
    assert spec.initial == "ingest"

    accept = [step.state for step in build_plan(spec, run_id="t", assume_decision="promote_checkpoint").steps]
    assert accept == ["ingest", "validate", "quality-gate", "curate", "register"]

    reject = [step.state for step in build_plan(spec, run_id="t", assume_decision="loop_back").steps]
    assert reject == ["ingest", "validate", "quality-gate", "reject"]


def test_workflow_dependency_order_carries_lineage() -> None:
    spec = load_spec(WORKFLOW)
    assert spec.states["ingest"].needs == []
    assert spec.states["validate"].needs == ["ingest"]
    assert spec.states["quality-gate"].needs == ["validate"]
    assert spec.states["curate"].needs == ["quality-gate"]
    assert spec.states["register"].needs == ["curate"]
    # Every non-initial stage declares an input carrying an upstream manifest.
    for name in ("validate", "quality-gate", "curate", "register"):
        assert spec.states[name].inputs, name


def test_new_dataset_toolrefs_render() -> None:
    for tool_ref in (
        "workbench.dataset.ingest",
        "workbench.dataset.validate",
        "workbench.dataset.curate",
        "workbench.dataset.query",
        "workbench.dataset.write_quality_decision",
        "workbench.dataset.report_rejection",
    ):
        assert tool_ref in TOOL_CATALOG
        assert argv_for_tool(tool_ref)
    ingest_argv = argv_for_tool("workbench.dataset.ingest")
    assert ingest_argv[:4] == ["npa", "workbench", "dataset", "ingest"]
    assert "--input-path" in ingest_argv
    assert "--output-path" in ingest_argv


def test_skypilot_yaml_has_cpu_and_gpu_stages() -> None:
    docs = [doc for doc in yaml.safe_load_all(SKYPILOT.read_text()) if doc is not None]
    assert docs[0]["name"] == "dataset-ingest-curate"
    assert docs[0]["execution"] == "serial"
    cpu_task = docs[1]
    gpu_task = docs[2]
    assert cpu_task["resources"]["cloud"] == "kubernetes"
    assert "accelerators" not in cpu_task["resources"]
    assert gpu_task["resources"]["accelerators"] == "H100:1"
    assert "npa workbench dataset ingest" in cpu_task["run"]
    assert "npa workbench dataset curate" in cpu_task["run"]
