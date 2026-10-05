"""Verify startup diagnosis requires GitHub evidence and keeps annotation text private."""

from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ci_runner_startup as startup  # noqa: E402


_REPOSITORY = "example/workbench"
_JOB = {
    "name": "gitleaks",
    "status": "completed",
    "conclusion": "cancelled",
    "runner_id": 0,
    "steps": [],
    "check_run_url": "https://api.github.com/repos/example/workbench/check-runs/21",
}
_ANNOTATION = {"annotation_level": "failure", "message": startup._ACQUISITION_FAILURE}


def test_diagnosis_preserves_input_and_never_copies_arbitrary_annotation_text():
    """Only the known GitHub failure is retained as a boolean diagnosis."""
    endpoints = []

    def read(endpoint):
        endpoints.append(endpoint)
        return [_ANNOTATION, {"annotation_level": "notice", "message": "private-text"}]

    jobs = startup.runner_startup_jobs(_REPOSITORY, [_JOB], read)
    assert jobs == [{**_JOB, "runner_startup_failure": True}]
    assert "runner_startup_failure" not in _JOB
    assert endpoints == [
        "repos/example/workbench/check-runs/21/annotations?per_page=100"
    ]
    assert "private-text" not in str(jobs)


@pytest.mark.parametrize(
    "change",
    [
        {"status": "in_progress"},
        {"conclusion": "success"},
        {"conclusion": "skipped"},
        {"runner_id": 7},
        {"runner_id": None},
        {"steps": [{"name": "test", "conclusion": "failure"}]},
        {"steps": None},
    ],
)
def test_executed_or_incomplete_jobs_cannot_be_classified(change):
    """A cancellation alone does not establish an infrastructure failure."""

    def read(endpoint):
        pytest.fail("Annotations must not be requested for an executed job")

    jobs = startup.runner_startup_jobs(_REPOSITORY, [{**_JOB, **change}], read)
    assert not jobs[0]["runner_startup_failure"]


@pytest.mark.parametrize(
    "annotations",
    [
        [],
        [{"annotation_level": "notice", "message": startup._ACQUISITION_FAILURE}],
        [{"annotation_level": "failure", "message": "unknown-failure"}],
        [_ANNOTATION, {"annotation_level": "failure", "message": "unknown-failure"}],
    ],
)
def test_unknown_or_mixed_failures_cannot_authorize_recovery(annotations):
    """Other failure messages prevent a runner-only classification."""
    jobs = startup.runner_startup_jobs(_REPOSITORY, [_JOB], lambda path: annotations)
    assert not jobs[0]["runner_startup_failure"]


@pytest.mark.parametrize(
    "url",
    [
        None,
        "https://api.github.com/repos/other/workbench/check-runs/21",
        "https://example.invalid/repos/example/workbench/check-runs/21",
        "https://api.github.com/repos/example/workbench/check-runs/21?injected=1",
    ],
)
def test_annotation_endpoints_are_bound_to_the_selected_repository(url):
    """Never follow a foreign or malformed URL from candidate job metadata."""

    def read(endpoint):
        pytest.fail("An untrusted annotation URL must not be followed")

    jobs = startup.runner_startup_jobs(
        _REPOSITORY, [{**_JOB, "check_run_url": url}], read
    )
    assert not jobs[0]["runner_startup_failure"]
    assert jobs[0]["runner_startup_unavailable"]


def test_unreadable_annotations_do_not_leak_error_text():
    """An API error leaves diagnosis unavailable rather than inventing a cause."""

    def read(endpoint):
        raise subprocess.CalledProcessError(1, "gh", stderr="private-error")

    jobs = startup.runner_startup_jobs(_REPOSITORY, [_JOB], read)
    assert jobs[0]["runner_startup_unavailable"]
    assert not jobs[0]["runner_startup_failure"]
    assert "private-error" not in str(jobs)
