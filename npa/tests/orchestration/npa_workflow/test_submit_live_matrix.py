"""Unit coverage for the live npa.workflow submit matrix (no cluster)."""

from __future__ import annotations

from pathlib import Path

from npa.orchestration.npa_workflow.submit_matrix import (
    SUBMIT_LIVE_MATRIX,
    selected_submit_cases,
)

SPECS_DIR = (
    Path(__file__).resolve().parents[4]
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
)


def test_submit_live_matrix_specs_exist() -> None:
    missing = [case.spec for case in SUBMIT_LIVE_MATRIX if not (SPECS_DIR / case.spec).is_file()]
    assert not missing, f"matrix references missing specs: {missing}"


def test_selected_submit_cases_tier_filter(monkeypatch) -> None:
    monkeypatch.setenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS", "cpu")
    monkeypatch.delenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS", raising=False)
    cases = selected_submit_cases()
    assert cases
    assert all(case.tier == "cpu" for case in cases)


def test_selected_submit_cases_spec_filter(monkeypatch) -> None:
    monkeypatch.setenv("NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS", "cpu,gpu,multi")
    monkeypatch.setenv(
        "NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS",
        "vlm-eval-single.yaml,token-factory-caption.yaml",
    )
    cases = selected_submit_cases()
    assert {case.spec for case in cases} == {
        "vlm-eval-single.yaml",
        "token-factory-caption.yaml",
    }


def test_submit_live_matrix_has_cpu_gpu_and_multi() -> None:
    tiers = {case.tier for case in SUBMIT_LIVE_MATRIX}
    assert tiers == {"cpu", "gpu", "multi"}
    assert any(case.plan_only for case in SUBMIT_LIVE_MATRIX)
    assert any(not case.plan_only and case.tier == "gpu" for case in SUBMIT_LIVE_MATRIX)
