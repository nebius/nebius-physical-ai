"""Regression coverage for the Sylvest paired checkpoint comparison workflow."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from npa.workflows import sylvest_oft_mixdata as workflow


def _rollouts(checkpoint: str, successes: list[int]) -> dict[str, object]:
    categories = ["Camera Viewpoints", "Robot Initial States", "Camera Viewpoints"]
    episodes = [
        {
            "case_id": f"task-{index}::seed=0",
            "task_name": f"task-{index}",
            "category": categories[index],
            "difficulty_level": 1,
            "seed": 0,
            "success": success,
        }
        for index, success in enumerate(successes)
    ]
    return {
        "schema": workflow.WORKFLOW_SCHEMA,
        "checkpoint": {"id": checkpoint, "revision": "pinned"},
        "protocol_sha256": "protocol-hash",
        "episodes": episodes,
    }


def test_pairing_requires_identical_case_identity() -> None:
    """Reject a candidate result that is not a one-to-one paired comparison.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    baseline = _rollouts("baseline", [0, 1, 0])
    candidate = _rollouts("candidate", [1, 0, 1])
    candidate["episodes"].pop()

    try:
        workflow._paired_comparison(baseline, candidate)
    except workflow.SylvestComparisonError as exc:
        assert "different paired cases" in str(exc)
    else:  # pragma: no cover - documents the failure boundary
        raise AssertionError("unpaired rollouts were accepted")


def test_compare_and_report_emit_measured_artifacts(tmp_path: Path) -> None:
    """Produce a comparison and RRD only from supplied paired rollout evidence.

    Args:
        tmp_path: Pytest-managed temporary directory.

    Returns:
        None.

    Raises:
        None.
    """

    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(_rollouts("baseline", [0, 1, 0])))
    candidate_path.write_text(json.dumps(_rollouts("candidate", [1, 0, 1])))
    comparison_dir = tmp_path / "comparison"

    assert workflow.main(
        [
            "compare",
            "--baseline-uri",
            str(baseline_path),
            "--candidate-uri",
            str(candidate_path),
            "--output-path",
            str(comparison_dir),
        ]
    ) == 0

    comparison_path = comparison_dir / "comparison.json"
    comparison = json.loads(comparison_path.read_text())
    assert comparison["overall"]["mean_delta_success"] == 1 / 3
    assert comparison["overall"]["candidate_only_successes"] == 2
    assert comparison["overall"]["baseline_only_successes"] == 1

    report_dir = tmp_path / "report"
    assert workflow.main(
        [
            "report",
            "--comparison-uri",
            str(comparison_path),
            "--run-id",
            "test-sylvest-comparison",
            "--output-path",
            str(report_dir),
        ]
    ) == 0
    assert (report_dir / "report.json").is_file()
    assert (report_dir / "comparison.rrd").stat().st_size > 0
    checksums = json.loads((report_dir / "checksums.json").read_text())
    assert set(checksums["files"]) == {"comparison.rrd", "report.json"}
    rerun = Path(sys.executable).with_name("rerun")
    verified = subprocess.run(
        [str(rerun), "rrd", "verify", str(report_dir / "comparison.rrd")],
        capture_output=True,
        check=False,
        text=True,
    )
    assert verified.returncode == 0, verified.stderr
    inspected = subprocess.run(
        [str(rerun), "rrd", "print", str(report_dir / "comparison.rrd")],
        capture_output=True,
        check=False,
        text=True,
    )
    assert inspected.returncode == 0, inspected.stderr
    assert "metrics/delta_success" in inspected.stdout


def test_unlicensed_libero_plus_source_is_refused(tmp_path: Path) -> None:
    """Do not turn an upstream source-license gap into an NPA consent flag."""

    with pytest.raises(workflow.SylvestComparisonError, match="no license file"):
        workflow._require_libero_plus_license(tmp_path)


def test_workflow_has_five_connected_substantive_stages() -> None:
    """Keep all five actual data, rollout, metric, and visualization stages wired.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    root = Path(__file__).resolve().parents[3]
    spec = yaml.safe_load(
        (root / "workflows/testing/sylvest-oft-mixdata-libero-plus-comparison.yaml").read_text()
    )
    states = spec["states"]
    assert list(states) == ["prepare", "baseline_rollouts", "candidate_rollouts", "compare", "report"]
    assert states["prepare"]["next"] == "baseline_rollouts"
    assert states["baseline_rollouts"]["next"] == "candidate_rollouts"
    assert states["candidate_rollouts"]["next"] == "compare"
    assert states["compare"]["next"] == "report"
    assert states["report"]["terminal"] is True
    assert spec["config"]["task_suite"] == "libero_spatial"
    assert all("npa.workflows.sylvest_oft_mixdata" in states[name]["run"]["shell"] for name in states)
    assert len(states["compare"]["inputs"]) == 2
    assert any(output["uri"].endswith("comparison.rrd") for output in states["report"]["outputs"])
