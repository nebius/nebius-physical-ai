"""Keep retained-run auditing strict about intentional quality rejection."""

from pathlib import Path

import pytest


@pytest.fixture
def terminal_audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    monkeypatch.delenv("NPA_PAIDF_APPEARANCE_CASES", raising=False)
    from tests.e2e.test_paidf_appearance_recipe_live import _audit_terminal

    return _audit_terminal


@pytest.fixture
def rejected_runtime():
    states = [
        "generate-variants",
        "evaluate",
        "quality-gate",
        "quality-disposition",
        "visualize-quality-evidence",
        "quality-route",
    ]
    waves = [{"states": [state], "status": "succeeded"} for state in states]
    waves.append({"states": ["reject-quality"], "status": "failed"})
    return {"status": "failed", "waves": waves}


def test_intentional_rejection_is_auditable(terminal_audit, rejected_runtime):
    terminal_audit(rejected_runtime, False)


@pytest.mark.parametrize(
    "stage", ["generate-variants", "evaluate", "visualize-quality-evidence"]
)
def test_prior_stage_failure_is_not_quality_rejection(
    terminal_audit, rejected_runtime, stage
):
    for wave in rejected_runtime["waves"]:
        if wave["states"] == [stage]:
            wave["status"] = "failed"
    with pytest.raises(AssertionError):
        terminal_audit(rejected_runtime, False)


def test_missing_review_evidence_is_not_auditable(terminal_audit, rejected_runtime):
    rejected_runtime["waves"] = [
        wave
        for wave in rejected_runtime["waves"]
        if wave["states"] != ["visualize-quality-evidence"]
    ]
    with pytest.raises(AssertionError):
        terminal_audit(rejected_runtime, False)


def test_other_terminal_failure_is_not_auditable(terminal_audit, rejected_runtime):
    rejected_runtime["waves"][-1]["states"] = ["annotate-augmented"]
    with pytest.raises(AssertionError):
        terminal_audit(rejected_runtime, False)


def test_rejected_outputs_must_not_reach_annotation(terminal_audit, rejected_runtime):
    rejected_runtime["waves"].insert(
        -1, {"states": ["annotate-augmented"], "status": "succeeded"}
    )
    with pytest.raises(AssertionError):
        terminal_audit(rejected_runtime, False)
