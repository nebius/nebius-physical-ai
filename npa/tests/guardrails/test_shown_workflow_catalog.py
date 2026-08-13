"""Guardrail: the shown workbench workflow catalog is npa.workflow-only.

SkyPilot stays the execution engine, but the raw SkyPilot workflow catalog is
retired. This guardrail keeps the shown catalog from regressing: no re-created
``skypilot/`` catalog directory and every shown spec must be a declarative
``npa.workflow`` spec.
"""

from __future__ import annotations

from pathlib import Path

from npa.orchestration.npa_workflow.blueprints import iter_npa_workflow_specs
from npa.orchestration.npa_workflow.catalog import (
    PUBLIC_REUSABLE_TOOLREFS,
    TOOL_CATALOG,
)
from npa.orchestration.npa_workflow.detect import detect_submit_format
from npa.orchestration.npa_workflow.spec import load_spec

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKBENCH = REPO_ROOT / "npa" / "workflows" / "workbench"
NPA_WORKFLOWS = WORKBENCH / "npa-workflows"


def test_skypilot_catalog_dir_is_not_reintroduced() -> None:
    forbidden = WORKBENCH / "skypilot"
    assert not forbidden.exists(), (
        "The raw SkyPilot task catalog must not live in the shown workbench "
        "catalog. Author npa.workflow specs in npa/workflows/workbench/"
        "npa-workflows/ instead; raw SkyPilot YAML belongs only in guarded "
        "tool-specific examples or test fixtures."
    )


def test_no_raw_skypilot_task_yaml_in_shown_catalog() -> None:
    # Covers the catalog directory plus any promoted top-level blueprint spec
    # (e.g. npa/workflows/physical-ai-data-factory.yaml).
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in iter_npa_workflow_specs()
        if detect_submit_format(path) == "skypilot"
    ]
    assert not offenders, (
        "Shown catalog specs must be npa.workflow (apiVersion "
        "npa.workflow/v0.0.1), not raw SkyPilot task YAMLs:\n" + "\n".join(offenders)
    )


def test_shown_catalog_has_npa_workflow_specs() -> None:
    specs = iter_npa_workflow_specs()
    assert specs, "expected npa.workflow specs under the shown catalog"
    assert all(detect_submit_format(path) == "npa.workflow" for path in specs)


def test_tool_catalog_is_reachable_or_explicitly_public_reusable() -> None:
    reachable = {
        state.tool_ref
        for path in iter_npa_workflow_specs()
        for state in load_spec(path).states.values()
        if state.tool_ref
    }
    reusable = set(PUBLIC_REUSABLE_TOOLREFS)
    assert not (reachable & reusable), "consumed entries are reachable, not reusable-only"
    assert set(TOOL_CATALOG) == reachable | reusable
    assert all(PUBLIC_REUSABLE_TOOLREFS.values())
