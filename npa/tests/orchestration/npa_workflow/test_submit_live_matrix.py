"""Unit coverage for the live npa.workflow submit matrix (no cluster)."""

from __future__ import annotations

import importlib.util
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


def _load_live_helpers():
    path = Path(__file__).resolve().parents[2] / "e2e" / "npa_workflow_live_helpers.py"
    spec = importlib.util.spec_from_file_location("npa_workflow_live_helpers", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_force_accelerators_on_cpu_profiles() -> None:
    helpers = _load_live_helpers()
    src = """resources:
  cpu:
    cloud: kubernetes
    cpus: 4
    memory: 16Gi
  gpu:
    cloud: kubernetes
    accelerators: H100:1
"""
    out = helpers._force_accelerators_on_cpu_profiles(src, "L40S:1")
    assert "accelerators: L40S:1" in out
    assert out.count("accelerators: L40S:1") == 1
    assert "accelerators: H100:1" in out
    assert "cloud: kubernetes" in out


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
