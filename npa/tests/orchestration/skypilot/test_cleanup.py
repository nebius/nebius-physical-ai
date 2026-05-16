from __future__ import annotations

import inspect
import subprocess

import pytest

from npa.orchestration.skypilot import cleanup as cleanup_module
from npa.orchestration.skypilot.cleanup import (
    CleanupResult,
    cleanup_all_for_run,
    cluster_name_patterns_for_run,
    sky_down,
    skypilot_workflow,
)
from npa.orchestration.skypilot import controller as controller_module
from npa.orchestration.skypilot import resources as resources_module


def test_sky_down_constructs_expected_subprocess_invocation(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs["env"]["HOME"] == str(tmp_path / "home")
        return subprocess.CompletedProcess(cmd, 0, stdout="down\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = sky_down("cluster-a", isolated_config_dir=tmp_path)

    assert calls == [["sky", "down", "--yes", "cluster-a"]]
    assert result.resources_removed == ["cluster-a"]
    assert result.errors == []


def test_context_manager_calls_cleanup_on_normal_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_cleanup(run_id, **kwargs):
        calls.append(run_id)
        return CleanupResult(resources_removed=["done"])

    monkeypatch.setattr(cleanup_module, "cleanup_all_for_run", fake_cleanup)

    with skypilot_workflow(run_id="run-123"):
        pass

    assert calls == ["run-123"]


def test_context_manager_calls_cleanup_on_exception_and_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_cleanup(run_id, **kwargs):
        calls.append(run_id)
        return CleanupResult(resources_removed=["done"])

    monkeypatch.setattr(cleanup_module, "cleanup_all_for_run", fake_cleanup)

    with pytest.raises(RuntimeError, match="boom"):
        with skypilot_workflow(run_id="run-456"):
            raise RuntimeError("boom")

    assert calls == ["run-456"]


def test_cleanup_all_for_run_matches_run_id_patterns(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_matching_jobs(run_id, **kwargs):
        return [{"job_id": "7", "name": f"{run_id}-job", "status": "RUNNING"}]

    def fake_cancel(job_id, **kwargs):
        calls.append(f"cancel:{job_id}")
        return CleanupResult(resources_removed=[f"job:{job_id}"])

    def fake_down(cluster_name, **kwargs):
        calls.append(f"down:{cluster_name}")
        return CleanupResult(resources_removed=[cluster_name])

    monkeypatch.setattr(cleanup_module, "_matching_jobs", fake_matching_jobs)
    monkeypatch.setattr(cleanup_module, "_cancel_job", fake_cancel)
    monkeypatch.setattr(cleanup_module, "sky_down", fake_down)

    result = cleanup_all_for_run("w9skypilot-integration-bootstrap-20260516T011706Z")

    assert "cancel:7" in calls
    assert any(call.startswith("down:") and "20260516t011706z" in call for call in calls)
    assert "sky-jobs-controller-*" in result.resources_removed
    assert cluster_name_patterns_for_run("abcDEF_123")[0] == "abcdef-123*"


def test_no_code_path_sets_autostop_down_true() -> None:
    sources = "\n".join(
        [
            inspect.getsource(resources_module),
            inspect.getsource(controller_module),
            inspect.getsource(cleanup_module),
        ]
    )

    assert '"down": True' not in sources
    assert "'down': True" not in sources
    assert "down: true" not in sources.lower()
