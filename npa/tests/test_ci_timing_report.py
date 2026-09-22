"""Verify timing reports distinguish runner waiting, execution, and missing results."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ci_timing_report as timing  # noqa: E402


def _run() -> dict:
    return {
        "id": 123,
        "run_attempt": 2,
        "event": "merge_group",
        "conclusion": "failure",
        "head_sha": "a" * 40,
        "created_at": "2026-09-18T10:00:00Z",
        "updated_at": "2026-09-18T10:10:00Z",
        "unrelated_private_field": "omit",
    }


def _job(identifier: int = 1) -> dict:
    return {
        "id": identifier,
        "name": "pytest",
        "conclusion": "success",
        "created_at": "2026-09-18T10:00:00Z",
        "started_at": "2026-09-18T10:02:00Z",
        "completed_at": "2026-09-18T10:05:00Z",
        "steps": [
            {
                "name": "Install npa",
                "conclusion": "success",
                "started_at": "2026-09-18T10:02:00Z",
                "completed_at": "2026-09-18T10:02:45Z",
            },
            {
                "name": "Run pytest coverage shard",
                "conclusion": "success",
                "started_at": "2026-09-18T10:02:45Z",
                "completed_at": "2026-09-18T10:05:00Z",
            },
        ],
    }


def test_report_separates_waiting_setup_and_parallel_execution() -> None:
    """Calculate durations without double-counting setup or summing wall latency.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Metadata produces misleading latency metrics.
    """
    report = timing._report(_run(), [{"jobs": [_job()]}, {"jobs": [_job(2)]}])
    assert report["wall_seconds"] == 600
    assert report["total_job_execution_seconds"] == 360
    assert report["median_runner_wait_seconds"] == 120
    assert report["jobs"][0]["setup_seconds"] == 45
    assert report["jobs"][0]["execution_seconds"] == 180
    assert "unrelated_private_field" not in report


def test_live_report_ends_at_required_gate_and_excludes_itself() -> None:
    """Measure completed validation while the reporting job is still running.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Reporter overhead changes the required-gate measurement.
    """
    run = _run()
    run["conclusion"] = None
    gate = _job(2)
    gate.update(name="security-regression", completed_at="2026-09-18T10:06:00Z")
    observer = _job(3)
    observer.update(name="ci-timing-report", conclusion=None, completed_at=None)
    report = timing._report(run, [{"jobs": [_job(), gate, observer]}])
    assert report["conclusion"] == "success"
    assert report["wall_seconds"] == 360
    assert [job["id"] for job in report["jobs"]] == [1, 2]


@pytest.mark.parametrize("conclusion", ["skipped", "cancelled"])
def test_unstarted_jobs_are_not_counted_as_zero_wait(conclusion: str) -> None:
    """Retain missing timestamps and skip GitHub's synthetic skipped-job times.

    Args:
        conclusion: Terminal result without a runner execution.
    Returns:
        None.
    Raises:
        AssertionError: Unstarted jobs distort the runner-wait summary.
    """
    job = _job(2)
    job.update(conclusion=conclusion, started_at=None, completed_at=None, steps=[])
    report = timing._report(_run(), [{"jobs": [_job(), job]}])
    assert report["median_runner_wait_seconds"] == 120
    assert report["jobs"][1]["execution_seconds"] is None


def test_skipped_jobs_with_synthetic_timestamps_are_excluded() -> None:
    """Ignore skipped jobs even when GitHub supplies start and finish values.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: A skipped check is reported as real execution.
    """
    job = _job()
    job["conclusion"] = "skipped"
    report = timing._report(_run(), [{"jobs": [job]}])
    assert report["median_runner_wait_seconds"] is None
    assert report["total_job_execution_seconds"] == 0
    assert report["jobs"][0]["setup_seconds"] is None


def test_cancelled_before_allocation_has_no_execution_time() -> None:
    """Ignore synthetic start timestamps for jobs cancelled while queued.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Runner waiting is mislabeled as test execution.
    """
    job = _job()
    job.update(conclusion="cancelled", runner_id=0, steps=[])
    report = timing._report(_run(), [{"jobs": [job]}])
    assert report["jobs"][0]["runner_wait_seconds"] is None
    assert report["jobs"][0]["execution_seconds"] is None
    assert report["jobs"][0]["setup_seconds"] is None


@pytest.mark.parametrize("end", [None, "2026-09-18T09:59:59Z"])
def test_invalid_intervals_are_unknown(end: str | None) -> None:
    """Distinguish missing or negative timestamps from actual zero-duration jobs.

    Args:
        end: Missing or chronologically invalid end timestamp.
    Returns:
        None.
    Raises:
        AssertionError: An unavailable duration is shown as valid.
    """
    assert timing._seconds("2026-09-18T10:00:00Z", end) is None
    assert timing._seconds("2026-09-18T10:00:00Z", "2026-09-18T10:00:00Z") == 0


def test_cli_writes_portable_reports_and_escapes_metadata(tmp_path: Path) -> None:
    """Exercise the reporting entrypoint and keep job names inside table cells.

    Args:
        tmp_path: Isolated metadata and output directory.
    Returns:
        None.
    Raises:
        AssertionError: The CLI loses pages or permits raw HTML in its summary.
    """
    job = _job()
    job["name"] = "<script>name</script>|second\nline"
    (tmp_path / "run.json").write_text(json.dumps(_run()))
    (tmp_path / "jobs.json").write_text(json.dumps([{"jobs": [job]}]))
    subprocess.run(
        [
            sys.executable,
            timing.__file__,
            "--run",
            str(tmp_path / "run.json"),
            "--jobs",
            str(tmp_path / "jobs.json"),
            "--output-directory",
            str(tmp_path),
        ],
        check=True,
    )
    report = json.loads((tmp_path / "timing.json").read_text())
    markdown = (tmp_path / "timing.md").read_text()
    assert report["attempt"] == 2
    assert "&lt;script&gt;name&lt;/script&gt;&#124;second line" in markdown
    assert "<script>" not in markdown
