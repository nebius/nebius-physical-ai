"""Exercise guarded full-attempt recovery, stale PR rejection, and metadata races."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ci_recover_validation as recovery  # noqa: E402


_REPOSITORY = "example/workbench"
_PULL = {
    "number": 604,
    "state": "open",
    "head": {"sha": "a" * 40},
    "base": {"ref": "main", "sha": "b" * 40},
}
_RUN = {
    "id": 12,
    "run_attempt": 2,
    "status": "completed",
    "conclusion": "failure",
    "event": "pull_request",
    "path": recovery._WORKFLOW,
    "head_sha": _PULL["head"]["sha"],
    "repository": {"full_name": _REPOSITORY},
    "pull_requests": [{"number": 604, "base": _PULL["base"]}],
}
_STARTUP = {
    "name": "gitleaks",
    "status": "completed",
    "conclusion": "cancelled",
    "runner_id": 0,
    "steps": [],
    "check_run_url": "https://api.github.com/repos/example/workbench/check-runs/21",
}
_GATE = {
    "name": "security-regression",
    "status": "completed",
    "conclusion": "failure",
    "runner_id": 8,
    "steps": [
        {"name": "Set up job", "conclusion": "success"},
        {
            "name": "Require every candidate gate and security audit",
            "conclusion": "failure",
        },
    ],
}
_ANNOTATIONS = [
    {
        "annotation_level": "failure",
        "message": "The job was not acquired by Runner of type hosted even after multiple attempts",
    }
]


def _metadata(monkeypatch, *, pull=None, runs=None, jobs=None, annotations=None):
    writes, reads = [], []

    def api(endpoint, *, method="GET"):
        if method == "POST":
            writes.append(endpoint)
            return None
        reads.append(endpoint)
        assert endpoint == "repos/example/workbench/pulls/604"
        return deepcopy(pull or _PULL)

    def pages(endpoint, key=None):
        reads.append(endpoint)
        if key == "workflow_runs":
            return deepcopy([_RUN] if runs is None else runs)
        if key == "jobs":
            return deepcopy([_STARTUP, _GATE] if jobs is None else jobs)
        return deepcopy(_ANNOTATIONS if annotations is None else annotations)

    monkeypatch.setattr(recovery, "_api", api)
    monkeypatch.setattr(recovery, "_pages", pages)
    return writes, reads


def test_default_diagnosis_reads_the_exact_attempt_without_writing(monkeypatch):
    """Read-only diagnosis identifies failures in the current attempt."""
    writes, reads = _metadata(monkeypatch)
    result = recovery._recover(_REPOSITORY, 604, False)
    assert result["runner_startup_failures"] == ["gitleaks"]
    assert result["run_id"] == 12 and result["run_attempt"] == 2
    assert not result["rerun_requested"] and writes == []
    assert (
        "repos/example/workbench/actions/runs/12/attempts/2/jobs?per_page=100" in reads
    )


def test_recovery_rechecks_current_identity_then_reruns_all_jobs(monkeypatch):
    """A full attempt preserves the merge queue's complete-evidence requirement."""
    writes, reads = _metadata(monkeypatch)
    result = recovery._recover(_REPOSITORY, 604, True)
    assert result["rerun_requested"]
    assert writes == ["repos/example/workbench/actions/runs/12/rerun"]
    assert reads.count("repos/example/workbench/pulls/604") == 2


@pytest.mark.parametrize(
    "change",
    [
        {"head_sha": "c" * 40},
        {"event": "merge_group"},
        {"path": ".github/workflows/another.yml"},
        {"repository": {"full_name": "other/repository"}},
        {"pull_requests": []},
        {"pull_requests": [{"number": 605}]},
    ],
)
def test_unrelated_validation_cannot_be_recovered(monkeypatch, change):
    """Require exact repository, workflow, PR head, and PR association."""
    writes, _ = _metadata(monkeypatch, runs=[{**_RUN, **change}])
    with pytest.raises(ValueError, match="No validation run matches"):
        recovery._recover(_REPOSITORY, 604, True)
    assert writes == []


@pytest.mark.parametrize(
    "pull, message",
    [
        ({**_PULL, "state": "closed"}, "no longer open"),
        ({**_PULL, "base": {"ref": "another", "sha": "b" * 40}}, "does not target"),
        ({**_PULL, "base": {"ref": "main", "sha": "c" * 40}}, "tested base changed"),
    ],
)
def test_closed_prs_and_changed_bases_require_fresh_validation(
    monkeypatch, pull, message
):
    """Rerunning an old merge cannot validate a new main tree."""
    writes, _ = _metadata(monkeypatch, pull=pull)
    with pytest.raises(ValueError, match=message):
        recovery._recover(_REPOSITORY, 604, True)
    assert writes == []


@pytest.mark.parametrize(
    "change",
    [
        {"status": "queued"},
        {"status": "in_progress"},
        {"conclusion": "success"},
        {"conclusion": "timed_out"},
    ],
)
def test_newest_run_supersedes_an_older_recoverable_failure(monkeypatch, change):
    """Never retry the old attempt when a newer result already exists."""
    writes, _ = _metadata(monkeypatch, runs=[_RUN, {**_RUN, "id": 13, **change}])
    with pytest.raises(ValueError):
        recovery._recover(_REPOSITORY, 604, True)
    assert writes == []


@pytest.mark.parametrize(
    "component",
    [
        {**_STARTUP, "name": "shard", "runner_id": 8},
        {**_STARTUP, "name": "shard", "check_run_url": None},
        {**_GATE, "name": "security-scanners"},
        {
            **_GATE,
            "steps": [{"name": "Publish validation identity", "conclusion": "failure"}],
        },
        {**_GATE, "conclusion": "cancelled"},
        {**_GATE, "status": "in_progress"},
    ],
)
def test_mixed_unknown_executed_and_cancelled_failures_block_recovery(
    monkeypatch, component
):
    """Startup evidence cannot mask a failed test or another unknown cause."""
    writes, _ = _metadata(monkeypatch, jobs=[_STARTUP, component])
    with pytest.raises(ValueError, match="another or unknown reason"):
        recovery._recover(_REPOSITORY, 604, True)
    assert writes == []


def test_missing_startup_annotations_refuse_recovery(monkeypatch):
    """Absence of execution does not by itself prove a runner failure."""
    writes, _ = _metadata(monkeypatch, annotations=[])
    with pytest.raises(ValueError, match="No confirmed"):
        recovery._recover(_REPOSITORY, 604, True)
    assert writes == []


@pytest.mark.parametrize("change", ["head", "base", "id", "attempt", "status"])
def test_identity_races_abort_before_the_rerun_request(monkeypatch, change):
    """Refresh the PR and latest attempt immediately before requesting recovery."""
    writes, _ = _metadata(monkeypatch)
    calls = []

    def latest(repository, number):
        pull, run = deepcopy(_PULL), deepcopy(_RUN)
        if calls:
            if change in {"head", "base"}:
                pull[change]["sha"] = "c" * 40
            else:
                key = {"id": "id", "attempt": "run_attempt", "status": "status"}[change]
                run[key] = "queued" if change == "status" else run[key] + 1
        calls.append(True)
        return pull, run

    monkeypatch.setattr(recovery, "_latest_validation", latest)
    with pytest.raises(ValueError, match="changed during diagnosis"):
        recovery._recover(_REPOSITORY, 604, True)
    assert writes == []


@pytest.mark.parametrize(
    "script", ["ci_recover_validation.py", "merge_queue_report.py"]
)
def test_isolated_python_can_load_the_shared_classifier(script):
    """Trusted automation must retain its Python -I entrypoint."""
    path = Path(__file__).resolve().parents[1] / "scripts" / script
    result = subprocess.run(
        [sys.executable, "-I", str(path), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_cli_api_errors_refuse_recovery_without_private_stderr(monkeypatch, capsys):
    """Metadata failures expose a concise refusal rather than the raw API error."""
    monkeypatch.setattr(
        sys, "argv", ["recover", "--repository", _REPOSITORY, "--pr", "604", "--rerun"]
    )

    def recover(*args):
        raise subprocess.CalledProcessError(1, "gh", stderr="private-api-error")

    monkeypatch.setattr(recovery, "_recover", recover)
    with pytest.raises(SystemExit) as error:
        recovery._main()
    assert error.value.code == 1
    output = capsys.readouterr().err
    assert "could not be verified" in output and "private-api-error" not in output
