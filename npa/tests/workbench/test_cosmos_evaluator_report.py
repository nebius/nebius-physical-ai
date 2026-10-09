"""Contract tests for read-only Cosmos Evaluator report inspection."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from npa.sdk import workbench as sdk_workbench
from npa.workbench.cosmos_evaluator.report import (
    CosmosEvaluatorReportError,
    inspect_evaluator_report,
    summarize_evaluator_report,
)


def _diagnostic(*, score: float, passed: bool, checks: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "score": score,
        "passed": passed,
        "threshold": 0.8,
        "total_frames": 12,
    }
    if checks:
        result.update(
            total_checks=2,
            passed_checks=2,
            failed_checks=0,
            checks=[{"error": None, "question": "synthetic question"}],
        )
    return result


@pytest.fixture
def real_shaped_report() -> dict[str, Any]:
    """Return a synthetic artifact shaped like the current evaluator output."""
    return {
        "schema": "npa.cosmos_evaluator.report.v1",
        "status": "completed",
        "score": 0.9,
        "passed": True,
        "threshold": 0.8,
        "clip_count": 1,
        "passed_clips": 1,
        "temporal_mode": "advisory",
        "appearance_mode": "advisory",
        "augment_uri": "s3://example-bucket/source/",
        "result_uri": "s3://example-bucket/grade/cosmos_evaluator.json",
        "metadata": {"prompt": "synthetic prompt", "source_uri": "excluded"},
        "multiview_assessment": {"claim": "not projected"},
        "clips": [
            {
                "clip_id": "variant-001",
                "status": "completed",
                "score": 0.9,
                "passed": True,
                "input_conditioned": True,
                "temporal_enforced": False,
                "appearance_enforced": False,
                "skipped": [],
                "attribute_verification": _diagnostic(
                    score=1.0, passed=True, checks=True
                ),
                "hallucination": _diagnostic(score=0.8, passed=True),
                "temporal_consistency": _diagnostic(score=0.7, passed=False),
                "appearance_fidelity": _diagnostic(score=0.6, passed=False),
            }
        ],
    }


def test_summary_preserves_reported_gate_and_bounds_projection(real_shaped_report) -> None:
    summary = summarize_evaluator_report(real_shaped_report)

    assert summary["reported_gate"] == {"score": 0.9, "passed": True, "threshold": 0.8}
    assert summary["evaluation_state"] == "graded"
    assert summary["score_is_calibrated_confidence"] is False
    assert summary["multiview_assessment"] == {
        "presented": True,
        "status": "not_evaluated",
        "supported": False,
    }
    variant = summary["variants"][0]
    assert variant["diagnostics"]["attributes"]["counts"]["total_checks"] == 2
    assert variant["diagnostics"]["temporal"]["enforcement"] == "advisory"
    rendered = json.dumps(summary)
    assert "synthetic prompt" not in rendered
    assert "s3://example-bucket" not in rendered
    assert "synthetic question" not in rendered


def test_partial_degraded_report_exposes_unavailable_and_error_states(
    real_shaped_report,
) -> None:
    report = copy.deepcopy(real_shaped_report)
    report.update(status="degraded", score=0.0, passed=False, passed_clips=0)
    clip = report["clips"][0]
    clip.update(status="degraded", score=0.0, passed=False)
    clip["attribute_verification"] = {"status": "unavailable"}
    clip["hallucination"] = {"status": "error"}
    clip.pop("temporal_consistency")
    clip["appearance_fidelity"] = None
    clip["skipped"] = ["appearance fidelity needs a source clip"]

    summary = summarize_evaluator_report(report)

    diagnostics = summary["variants"][0]["diagnostics"]
    assert diagnostics["attributes"]["status"] == "unavailable"
    assert diagnostics["hallucination"]["status"] == "error"
    assert diagnostics["temporal"]["status"] == "not_evaluated"
    assert diagnostics["appearance"]["status"] == "skipped"
    assert summary["reported_gate"]["passed"] is False
    assert summary["evidence_complete"] is False


def test_absent_legacy_diagnostics_are_explicitly_not_evaluated(real_shaped_report) -> None:
    report = copy.deepcopy(real_shaped_report)
    clip = report["clips"][0]
    clip.pop("temporal_consistency")
    clip.pop("appearance_fidelity")

    summary = summarize_evaluator_report(report)

    diagnostics = summary["variants"][0]["diagnostics"]
    assert diagnostics["temporal"]["status"] == "not_evaluated"
    assert diagnostics["appearance"]["status"] == "not_evaluated"
    assert summary["evidence_complete"] is True


def test_missing_enforcement_facts_are_unverified_not_advisory(real_shaped_report) -> None:
    report = copy.deepcopy(real_shaped_report)
    report["temporal_mode"] = "required"
    report["appearance_mode"] = "required"
    clip = report["clips"][0]
    clip.pop("input_conditioned")
    clip.pop("temporal_enforced")
    clip.pop("appearance_enforced")
    clip.pop("hallucination")

    summary = summarize_evaluator_report(report)

    diagnostics = summary["variants"][0]["diagnostics"]
    assert diagnostics["hallucination"] == {
        "enforcement": "unverified",
        "status": "not_evaluated",
    }
    assert diagnostics["temporal"]["enforcement"] == "unverified"
    assert diagnostics["appearance"]["enforcement"] == "unverified"
    assert summary["reported_gate"]["passed"] is True
    assert summary["evaluation_state"] == "incomplete"
    assert summary["evidence_complete"] is False


def test_required_root_modes_derive_and_validate_companion_enforcement(
    real_shaped_report,
) -> None:
    contradictory = copy.deepcopy(real_shaped_report)
    contradictory["temporal_mode"] = "required"
    with pytest.raises(CosmosEvaluatorReportError, match="contradicts the required"):
        summarize_evaluator_report(contradictory)

    missing = copy.deepcopy(real_shaped_report)
    missing["temporal_mode"] = "required"
    missing["clips"][0].pop("temporal_enforced")
    missing["clips"][0].pop("temporal_consistency")
    with pytest.raises(CosmosEvaluatorReportError, match="missing required"):
        summarize_evaluator_report(missing)


@pytest.mark.parametrize(
    ("mode", "flag", "field"),
    [
        ("temporal_mode", "temporal_enforced", "temporal_consistency"),
        ("appearance_mode", "appearance_enforced", "appearance_fidelity"),
    ],
)
def test_required_root_modes_apply_when_clip_flags_are_absent(
    real_shaped_report, mode: str, flag: str, field: str
) -> None:
    report = copy.deepcopy(real_shaped_report)
    report[mode] = "required"
    report["clips"][0].pop(flag)
    report["clips"][0][field].update(score=0.8, passed=True)

    summary = summarize_evaluator_report(report)

    diagnostic = "temporal" if mode == "temporal_mode" else "appearance"
    assert summary["variants"][0]["diagnostics"][diagnostic]["enforcement"] == "required"


def test_empty_report_is_ungraded_and_incomplete(real_shaped_report) -> None:
    report = copy.deepcopy(real_shaped_report)
    report.update(score=0.0, passed=False, clip_count=0, passed_clips=0, clips=[])

    summary = summarize_evaluator_report(report)

    assert summary["evaluation_state"] == "ungraded"
    assert summary["evidence_complete"] is False
    assert summary["variants"] == []


def test_reader_uses_one_exact_s3_object(real_shaped_report, tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps(real_shaped_report), encoding="utf-8")
    storage = _ExactStorage(source)

    summary = inspect_evaluator_report(
        "s3://example-bucket/reports/evaluator.json", storage=storage
    )

    assert storage.calls == ["s3://example-bucket/reports/evaluator.json"]
    assert summary["reported_gate"]["score"] == 0.9


def test_sdk_namespace_discovers_report_reader(real_shaped_report, tmp_path: Path) -> None:
    path = _write_report(tmp_path, real_shaped_report)

    assert "cosmos_evaluator" in sdk_workbench.__all__
    summary = sdk_workbench.cosmos_evaluator.report(input_path=str(path))

    assert summary["schema"] == "npa.cosmos_evaluator.report.v1"


@pytest.mark.parametrize("value", [True, float("inf"), 1.1])
def test_reader_rejects_nonfinite_out_of_range_and_boolean_scores(
    real_shaped_report, value: Any
) -> None:
    report = copy.deepcopy(real_shaped_report)
    report["score"] = value

    with pytest.raises(CosmosEvaluatorReportError, match="score"):
        summarize_evaluator_report(report)


def test_reader_rejects_duplicate_ids_count_mismatches_and_bad_dispositions(
    real_shaped_report,
) -> None:
    duplicate = copy.deepcopy(real_shaped_report)
    duplicate["clips"].append(copy.deepcopy(duplicate["clips"][0]))
    duplicate["clip_count"] = 2
    duplicate["passed_clips"] = 2
    with pytest.raises(CosmosEvaluatorReportError, match="duplicate"):
        summarize_evaluator_report(duplicate)

    mismatched = copy.deepcopy(real_shaped_report)
    mismatched["clip_count"] = 2
    with pytest.raises(CosmosEvaluatorReportError, match="clip_count"):
        summarize_evaluator_report(mismatched)

    passed_mismatch = copy.deepcopy(real_shaped_report)
    passed_mismatch["passed_clips"] = 0
    with pytest.raises(CosmosEvaluatorReportError, match="passed_clips"):
        summarize_evaluator_report(passed_mismatch)

    for status in ("degraded", "error", "failed"):
        root = copy.deepcopy(real_shaped_report)
        root["status"] = status
        with pytest.raises(CosmosEvaluatorReportError, match="non-completed"):
            summarize_evaluator_report(root)

        clip = copy.deepcopy(real_shaped_report)
        clip["clips"][0]["status"] = status
        with pytest.raises(CosmosEvaluatorReportError, match="non-completed"):
            summarize_evaluator_report(clip)


def test_reader_rejects_missing_required_evidence_and_unsafe_statuses(
    real_shaped_report,
) -> None:
    missing = copy.deepcopy(real_shaped_report)
    missing["clips"][0]["attribute_verification"] = None
    with pytest.raises(CosmosEvaluatorReportError, match="missing required"):
        summarize_evaluator_report(missing)

    unknown = copy.deepcopy(real_shaped_report)
    unknown["clips"][0]["status"] = "in_progress"
    with pytest.raises(CosmosEvaluatorReportError, match="unsupported status"):
        summarize_evaluator_report(unknown)

    unknown_diagnostic = copy.deepcopy(real_shaped_report)
    unknown_diagnostic["clips"][0]["hallucination"]["status"] = "in_progress"
    with pytest.raises(CosmosEvaluatorReportError, match="unsupported status"):
        summarize_evaluator_report(unknown_diagnostic)

    unsafe = copy.deepcopy(real_shaped_report)
    unsafe["clips"][0]["clip_id"] = "variant\n001"
    with pytest.raises(CosmosEvaluatorReportError, match="safe non-empty"):
        summarize_evaluator_report(unsafe)

    unsafe_status = copy.deepcopy(real_shaped_report)
    unsafe_status["clips"][0]["status"] = "completed\n"
    with pytest.raises(CosmosEvaluatorReportError, match="safe non-empty"):
        summarize_evaluator_report(unsafe_status)


def test_reader_rejects_passing_clips_with_failed_required_diagnostics(
    real_shaped_report,
) -> None:
    failed_attribute = copy.deepcopy(real_shaped_report)
    failed_attribute["clips"][0]["attribute_verification"]["passed"] = False
    with pytest.raises(CosmosEvaluatorReportError, match="failed required"):
        summarize_evaluator_report(failed_attribute)

    failed_temporal = copy.deepcopy(real_shaped_report)
    failed_temporal["temporal_mode"] = "required"
    failed_temporal["clips"][0]["temporal_enforced"] = True
    with pytest.raises(CosmosEvaluatorReportError, match="failed required"):
        summarize_evaluator_report(failed_temporal)

    failed_appearance = copy.deepcopy(real_shaped_report)
    failed_appearance["appearance_mode"] = "required"
    failed_appearance["clips"][0]["appearance_enforced"] = True
    with pytest.raises(CosmosEvaluatorReportError, match="failed required"):
        summarize_evaluator_report(failed_appearance)


def test_reader_rejects_diagnostic_passes_below_their_threshold(real_shaped_report) -> None:
    report = copy.deepcopy(real_shaped_report)
    diagnostic = report["clips"][0]["hallucination"]
    diagnostic.update(score=0.7, passed=True, threshold=0.8)

    with pytest.raises(CosmosEvaluatorReportError, match="declared threshold"):
        summarize_evaluator_report(report)

    clip_below_gate = copy.deepcopy(real_shaped_report)
    clip_below_gate["clips"][0]["score"] = 0.7
    with pytest.raises(CosmosEvaluatorReportError, match="report threshold"):
        summarize_evaluator_report(clip_below_gate)


def test_reader_rejects_duplicate_keys_nan_and_non_mapping(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")
    with pytest.raises(CosmosEvaluatorReportError, match="valid JSON"):
        inspect_evaluator_report(str(duplicate))

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"score":NaN}', encoding="utf-8")
    with pytest.raises(CosmosEvaluatorReportError, match="valid JSON"):
        inspect_evaluator_report(str(nonfinite))

    with pytest.raises(CosmosEvaluatorReportError, match="JSON object"):
        summarize_evaluator_report([])  # type: ignore[arg-type]

    invalid_utf8 = tmp_path / "invalid-utf8.json"
    invalid_utf8.write_bytes(b"\x80")
    with pytest.raises(CosmosEvaluatorReportError, match="could not read"):
        inspect_evaluator_report(str(invalid_utf8))


def test_reader_imports_and_executes_when_jsonschema_is_blocked(
    real_shaped_report, tmp_path: Path
) -> None:
    report = _write_report(tmp_path, real_shaped_report)
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(_JSONSCHEMA_BLOCKER, encoding="utf-8")
    source_root = Path(__file__).resolve().parents[2] / "src"
    completed = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_CHECK, str(report)],
        check=False,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": f"{tmp_path}:{source_root}"},
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "npa.cosmos_evaluator.report.v1"


class _ExactStorage:
    """Storage double that permits one exact object read."""

    def __init__(self, source: Path) -> None:
        self.calls: list[str] = []
        self.source = source

    def download_file(self, uri: str, destination: str) -> str:
        self.calls.append(uri)
        shutil.copyfile(self.source, destination)
        return destination


def _write_report(tmp_path: Path, report: dict[str, Any]) -> Path:
    path = tmp_path / "cosmos_evaluator.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


_JSONSCHEMA_BLOCKER = """
import builtins
_original_import = builtins.__import__
def _blocked(name, *args, **kwargs):
    if name == 'jsonschema' or name.startswith('jsonschema.'):
        raise ModuleNotFoundError('jsonschema is unavailable')
    return _original_import(name, *args, **kwargs)
builtins.__import__ = _blocked
"""
_SUBPROCESS_CHECK = """
import sys
from npa.workbench.cosmos_evaluator.report import inspect_evaluator_report
print(inspect_evaluator_report(sys.argv[1])['schema'])
"""
