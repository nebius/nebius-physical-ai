"""Identify unstarted CI jobs using GitHub's runner-acquisition annotations."""

from collections.abc import Callable
import re
import subprocess


_ACQUISITION_FAILURE = (
    "The job was not acquired by Runner of type hosted even after multiple attempts"
)
_UNSUCCESSFUL = {"failure", "cancelled", "timed_out", "startup_failure"}


def runner_startup_jobs(
    repository: str, jobs: list[dict], read_annotations: Callable[[str], list[dict]]
) -> list[dict]:
    """Annotate confirmed runner failures without exposing arbitrary annotation text.

    Args:
        repository: GitHub owner/repository containing the validation run.
        jobs: Job metadata for one exact validation attempt.
        read_annotations: Paginated reader for GitHub check-run annotations.
    Returns:
        Copied jobs marked with confirmed startup failures or unavailable metadata.
    Raises:
        None. Unreadable annotations leave the failure unclassified.
    """
    annotated = []
    for original in jobs:
        job = {**original, "runner_startup_failure": False}
        if _never_started(job):
            _read_startup_failure(repository, job, read_annotations)
        annotated.append(job)
    return annotated


def _never_started(job: dict) -> bool:
    return (
        job.get("status") == "completed"
        and job.get("conclusion") in _UNSUCCESSFUL
        and job.get("runner_id") == 0
        and job.get("steps") == []
    )


def _read_startup_failure(repository: str, job: dict, read_annotations: Callable):
    match = re.fullmatch(
        rf"https://api\.github\.com/repos/{re.escape(repository)}/check-runs/([0-9]+)",
        job.get("check_run_url") or "",
    )
    if match is None:
        job["runner_startup_unavailable"] = True
        return
    try:
        annotations = read_annotations(
            f"repos/{repository}/check-runs/{match[1]}/annotations?per_page=100"
        )
    except (subprocess.CalledProcessError, ValueError):
        job["runner_startup_unavailable"] = True
        return
    failures = [
        item.get("message")
        for item in annotations
        if item.get("annotation_level") == "failure"
    ]
    job["runner_startup_failure"] = bool(failures) and all(
        message == _ACQUISITION_FAILURE for message in failures
    )
