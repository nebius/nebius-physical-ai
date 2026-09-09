"""Test the native Ray Jobs two-revision source-sync qualification harness."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


EXAMPLE = Path(__file__).parents[2] / "workflows/workbench/ray-clip-development"


@pytest.fixture
def fast_sync():
    """Import the standalone harness by path.

    Args:
        None.
    Returns:
        Imported fast-sync module.
    Raises:
        ImportError: The harness cannot be loaded.
    """
    specification = importlib.util.spec_from_file_location("ray_fast_sync_test", EXAMPLE / "fast_sync.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class _Status(str):
    """Provide the Ray JobStatus behavior used by the harness."""

    def is_terminal(self) -> bool:
        """Return whether this synthetic Job status is terminal.

        Args:
            None.
        Returns:
            True for terminal states.
        Raises:
            None.
        """
        return self in {"SUCCEEDED", "FAILED", "STOPPED"}


class _JobsClient:
    """Observe source bytes at the same boundary as submit_job."""

    instances: list["_JobsClient"] = []
    stale_second_revision = False

    def __init__(self, address: str):
        """Record the explicit address without making a network request."""
        self.address = address
        self.jobs: dict[str, dict[str, str]] = {}
        self.stopped: list[str] = []
        self.first_result: dict[str, str] | None = None
        self.instances.append(self)

    def submit_job(self, *, submission_id: str, entrypoint: str, runtime_env: dict[str, str]) -> None:
        """Capture the current working-directory module as a remote observation."""
        source = Path(runtime_env["working_dir"]) / "editable_value.py"
        assignment = source.read_text(encoding="utf-8").partition("=")[2].strip()
        result = {
            "ray_version": "2.58.0",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "value": ast.literal_eval(assignment),
        }
        if self.stale_second_revision and self.first_result is not None:
            result = self.first_result
        self.first_result = self.first_result or result
        self.jobs[submission_id] = {
            "status": _Status("SUCCEEDED"),
            "logs": "NPA_RAY_FAST_SYNC_RESULT " + json.dumps(result, sort_keys=True),
            "entrypoint": entrypoint,
        }

    def get_job_status(self, submission_id: str) -> _Status:
        """Return the selected exact Job status."""
        return self.jobs[submission_id]["status"]

    def get_job_logs(self, submission_id: str) -> str:
        """Return the selected exact Job logs."""
        return self.jobs[submission_id]["logs"]

    def stop_job(self, submission_id: str) -> bool:
        """Stop the selected exact Job."""
        self.jobs[submission_id]["status"] = _Status("STOPPED")
        self.stopped.append(submission_id)
        return True


def _install_ray_client(monkeypatch, *, stale: bool = False) -> None:
    """Install a hermetic Ray 2.58 Jobs client module.

    Args:
        monkeypatch: Pytest module-state fixture.
        stale: Whether the second Job should observe cached baseline bytes.
    Returns:
        None.
    Raises:
        None.
    """
    _JobsClient.instances = []
    _JobsClient.stale_second_revision = stale
    ray = ModuleType("ray")
    ray.__version__ = "2.58.0"
    jobs = ModuleType("ray.job_submission")
    jobs.JobSubmissionClient = _JobsClient
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.job_submission", jobs)


def test_qualification_submits_two_working_directories_and_observes_edit(
    fast_sync, tmp_path, monkeypatch
):
    """Prove the harness checks exact remote bytes and an observable change.

    Args:
        fast_sync: Imported qualification harness.
        tmp_path: Pytest-owned evidence parent.
        monkeypatch: Pytest module-state fixture.
    Returns:
        None.
    Raises:
        AssertionError: The two-revision native Jobs contract changes.
    """
    _install_ray_client(monkeypatch)
    evidence = tmp_path / "private-evidence"
    summary = fast_sync.qualify("http://ray.invalid", evidence, run_token="unit")
    client = _JobsClient.instances[0]
    assert summary["status"] == "passed"
    assert [row["value"] for row in summary["revisions"]] == ["before", "after"]
    assert len({row["source_sha256"] for row in summary["revisions"]}) == 2
    assert set(client.jobs) == {"npa-fast-sync-unit-baseline", "npa-fast-sync-unit-changed"}
    assert all(job["entrypoint"] == "python sync_probe.py" for job in client.jobs.values())
    assert all("submission_id" not in row for row in summary["revisions"])
    assert evidence.stat().st_mode & 0o077 == 0
    assert all(path.stat().st_mode & 0o077 == 0 for path in evidence.iterdir())
    private = json.loads((evidence / "result.json").read_text(encoding="utf-8"))
    assert private["submission_ids"] == list(client.jobs)


def test_qualification_rejects_stale_second_package_after_exact_cleanup(
    fast_sync, tmp_path, monkeypatch
):
    """Reject package-cache staleness while retaining terminal cleanup evidence.

    Args:
        fast_sync: Imported qualification harness.
        tmp_path: Pytest-owned evidence parent.
        monkeypatch: Pytest module-state fixture.
    Returns:
        None.
    Raises:
        AssertionError: Stale remote source passes or cleanup evidence is lost.
    """
    _install_ray_client(monkeypatch, stale=True)
    evidence = tmp_path / "private-evidence"
    with pytest.raises(ValueError, match="submitted source bytes"):
        fast_sync.qualify("http://ray.invalid", evidence, run_token="stale")
    cleanup = json.loads((evidence / "cleanup.json").read_text(encoding="utf-8"))
    assert [row["submission_id"] for row in cleanup] == [
        "npa-fast-sync-stale-baseline",
        "npa-fast-sync-stale-changed",
    ]
    assert all(row["status"] == "SUCCEEDED" for row in cleanup)


def test_cleanup_stops_only_owned_nonterminal_jobs(fast_sync):
    """Stop exact supplied IDs without fuzzy discovery.

    Args:
        fast_sync: Imported qualification harness.
    Returns:
        None.
    Raises:
        AssertionError: Cleanup touches an unrelated Job or fails to verify stop.
    """
    client = _JobsClient("http://ray.invalid")
    client.jobs = {
        "owned-running": {"status": _Status("RUNNING"), "logs": "", "entrypoint": ""},
        "owned-done": {"status": _Status("SUCCEEDED"), "logs": "", "entrypoint": ""},
        "unrelated": {"status": _Status("RUNNING"), "logs": "", "entrypoint": ""},
    }
    cleanup = fast_sync._stop_nonterminal_jobs(client, ["owned-running", "owned-done"])
    assert client.stopped == ["owned-running"]
    assert cleanup == [
        {"submission_id": "owned-running", "status": "STOPPED"},
        {"submission_id": "owned-done", "status": "SUCCEEDED"},
    ]
    assert client.jobs["unrelated"]["status"] == "RUNNING"


def test_existing_evidence_directory_fails_before_connecting(fast_sync, tmp_path, monkeypatch):
    """Refuse evidence overwrite before constructing a Jobs client.

    Args:
        fast_sync: Imported qualification harness.
        tmp_path: Existing directory supplied as an unsafe destination.
        monkeypatch: Pytest module-state fixture.
    Returns:
        None.
    Raises:
        AssertionError: Existing evidence can be replaced or a client is created.
    """
    _install_ray_client(monkeypatch)
    with pytest.raises(FileExistsError):
        fast_sync.qualify("http://ray.invalid", tmp_path, run_token="existing")
    assert _JobsClient.instances == []


def test_result_parser_rejects_missing_duplicate_and_invalid_markers(fast_sync):
    """Reject logs that cannot bind one exact remote observation.

    Args:
        fast_sync: Imported qualification harness.
    Returns:
        None.
    Raises:
        AssertionError: Ambiguous or malformed logs are accepted.
    """
    valid = json.dumps({"ray_version": "2.58.0", "source_sha256": "digest", "value": "before"})
    with pytest.raises(ValueError, match="found 0"):
        fast_sync._parse_result("ordinary log")
    with pytest.raises(ValueError, match="found 2"):
        fast_sync._parse_result(f"{fast_sync.RESULT_MARKER}{valid}\n{fast_sync.RESULT_MARKER}{valid}")
    with pytest.raises(ValueError, match="invalid schema"):
        fast_sync._parse_result(f'{fast_sync.RESULT_MARKER}{json.dumps({"value": "before"})}')
