from __future__ import annotations

from pathlib import Path

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG, argv_for_tool

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "dataset-ingest-curate.yaml"
# The raw template is retired; the spec is the surface (EVIDENCE.md §R41).
SKYPILOT = ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot" / "dataset-ingest-curate.yaml"


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


def test_the_spec_runs_ingest_and_curate_and_registers_against_the_service() -> None:
    """`dataset-ingest-curate.yaml` (raw) is retired; its spec ran all five stages live.

    Job 317 finished with the `register` stage reading 12 records back out of the in-cluster
    LanceDB service that `ingest` had written to (EVIDENCE.md §R41).
    """

    from npa.orchestration.npa_workflow.spec import load_spec

    spec = load_spec(WORKFLOW)

    # The default plan takes the gate's `reject` branch, so assert on the spec's states rather
    # than one traversal of them.
    assert {"ingest", "validate", "quality-gate", "curate", "register"} <= set(spec.states)
    # CPU throughout: the pipeline moves metadata, it does not render.
    for profile in spec.resources.values():
        assert "accelerators" not in profile, profile

    endpoint = spec.config["lancedb_endpoint"]
    ingest = _resolved_argv(spec, "ingest")
    register = _resolved_argv(spec, "register")
    # Index on the way in, query the same table on the way out — the round trip job 317 proved.
    assert ingest[ingest.index("--lancedb-endpoint") + 1] == endpoint
    assert register[register.index("--lancedb-endpoint") + 1] == endpoint
    assert register[register.index("--lance-table") + 1] == spec.config["dataset_id"]


def _resolved_argv(spec, state: str) -> list[str]:
    """Resolve one state's toolRef argv against the spec's config."""

    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    tool_ref = spec.states[state].tool_ref
    resolved: list[str] = []
    for part in TOOL_CATALOG[tool_ref].argv_template:
        text = str(part)
        for key, value in spec.config.items():
            text = text.replace("{{config.%s}}" % key, str(value))
        resolved.append(text)
    return resolved


def test_the_retired_dataset_template_is_gone() -> None:
    assert not SKYPILOT.exists(), "dataset-ingest-curate.yaml came back"
