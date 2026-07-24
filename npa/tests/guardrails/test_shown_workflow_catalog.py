"""Guardrail: the shown workbench workflow catalog is npa.workflow-only.

SkyPilot stays the execution engine, but the raw SkyPilot task templates were
moved out of the shown catalog (``npa/workflows/workbench/``) into internal
package resources (``npa/src/npa/workflows/skypilot/``). This guardrail keeps
the catalog from regressing: no re-created ``skypilot/`` catalog directory and
every shown spec must be a declarative ``npa.workflow`` spec.
"""

from __future__ import annotations

from pathlib import Path

from npa.orchestration.npa_workflow.detect import detect_submit_format

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKBENCH = REPO_ROOT / "npa" / "workflows" / "workbench"
NPA_WORKFLOWS = WORKBENCH / "npa-workflows"


def test_skypilot_catalog_dir_is_not_reintroduced() -> None:
    forbidden = WORKBENCH / "skypilot"
    assert not forbidden.exists(), (
        "The raw SkyPilot task catalog must not live in the shown workbench "
        "catalog. Internal SkyPilot task templates belong under "
        "npa/src/npa/workflows/skypilot/; author npa.workflow specs in "
        "npa/workflows/workbench/npa-workflows/ instead."
    )


def test_no_raw_skypilot_task_yaml_in_shown_catalog() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted(NPA_WORKFLOWS.glob("*.yaml"))
        if detect_submit_format(path) == "skypilot"
    ]
    assert not offenders, (
        "Shown catalog specs must be npa.workflow (apiVersion "
        "npa.workflow/v0.0.1), not raw SkyPilot task YAMLs:\n" + "\n".join(offenders)
    )


def test_shown_catalog_has_npa_workflow_specs() -> None:
    specs = sorted(NPA_WORKFLOWS.glob("*.yaml"))
    assert specs, "expected npa.workflow specs under the shown catalog"
    assert all(detect_submit_format(path) == "npa.workflow" for path in specs)
