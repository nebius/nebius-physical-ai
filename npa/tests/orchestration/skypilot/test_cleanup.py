from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest

from npa.orchestration.skypilot import cleanup as cleanup_module
from npa.orchestration.skypilot.cleanup import (
    CleanupResult,
    cleanup_all_for_run,
    cleanup_jobs_controller,
    cluster_name_patterns_for_run,
    sky_down,
    skypilot_workflow,
)
from npa.orchestration.skypilot import controller as controller_module
from npa.orchestration.skypilot import resources as resources_module


def _fake_sky(tmp_path: Path) -> Path:
    sky = tmp_path / "sky"
    sky.write_text("#!/bin/sh\n", encoding="utf-8")
    sky.chmod(0o755)
    return sky


def test_sky_down_constructs_expected_subprocess_invocation(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    sky_bin = _fake_sky(tmp_path)
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs["env"]["HOME"] == str(tmp_path / "home")
        return subprocess.CompletedProcess(cmd, 0, stdout="down\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = sky_down("cluster-a", isolated_config_dir=tmp_path, sky_bin=sky_bin)

    assert calls == [[str(sky_bin), "down", "--yes", "cluster-a"]]
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


def test_cleanup_all_for_run_matches_run_id_patterns(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    sky_bin = _fake_sky(tmp_path)
    calls: list[str] = []

    def fake_matching_jobs(run_id, **kwargs):
        return [{"job_id": "7", "name": f"{run_id}-job", "status": "RUNNING"}]

    def fake_cancel(job_id, **kwargs):
        calls.append(f"cancel:{job_id}")
        return CleanupResult(resources_removed=[f"job:{job_id}"])

    def fake_down(cluster_name, **kwargs):
        calls.append(f"down:{cluster_name}")
        return CleanupResult(resources_removed=[cluster_name])

    def fake_cleanup_jobs_controller(**kwargs):
        return CleanupResult(resources_removed=["sky-jobs-controller-abc123"])

    monkeypatch.setattr(cleanup_module, "_matching_jobs", fake_matching_jobs)
    monkeypatch.setattr(cleanup_module, "_cancel_job", fake_cancel)
    monkeypatch.setattr(cleanup_module, "sky_down", fake_down)
    monkeypatch.setattr(cleanup_module, "cleanup_jobs_controller", fake_cleanup_jobs_controller)

    result = cleanup_all_for_run("w9skypilot-integration-bootstrap-20260516T011706Z", sky_bin=sky_bin)

    assert "cancel:7" in calls
    assert any(call.startswith("down:") and "20260516t011706z" in call for call in calls)
    assert "sky-jobs-controller-abc123" in result.resources_removed
    assert cluster_name_patterns_for_run("abcDEF_123")[0] == "abcdef-123*"


def test_cleanup_jobs_controller_discovers_exact_name_and_confirms_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    sky_bin = _fake_sky(tmp_path)
    calls: list[tuple[list[str], str | None]] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("input")))
        if cmd[1] == "status":
            stdout = '[{"name": "sky-jobs-controller-abc123", "status": "UP"}]'
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="down\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = cleanup_jobs_controller(isolated_config_dir=tmp_path, sky_bin=sky_bin)

    assert calls[0][0] == [str(sky_bin), "status", "--refresh", "--output", "json"]
    assert calls[0][1] is None
    assert calls[1] == ([str(sky_bin), "down", "--yes", "sky-jobs-controller-abc123"], "delete\n")
    assert result.resources_removed == ["sky-jobs-controller-abc123"]
    assert result.errors == []


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
