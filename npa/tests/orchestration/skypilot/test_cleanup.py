from __future__ import annotations

from fnmatch import fnmatchcase
import inspect
import subprocess
import threading
from pathlib import Path

import pytest

from npa.orchestration.skypilot import cleanup as cleanup_module
from npa.orchestration.skypilot.cleanup import (
    CleanupResult,
    InvalidRunIdError,
    cleanup_all_for_run,
    cleanup_jobs_controller,
    cluster_name_patterns_for_run,
    run_tag,
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
    run_id = "w9skypilot-integration-bootstrap-20260516T011706Z"

    def fake_matching_jobs(run_id, **kwargs):
        return [{"job_id": "7", "name": f"{run_id}-job", "status": "RUNNING"}]

    def fake_cancel(job_id, **kwargs):
        calls.append(f"cancel:{job_id}")
        return CleanupResult(resources_removed=[f"job:{job_id}"])

    def fake_down(cluster_name, **kwargs):
        calls.append(f"down:{cluster_name}")
        return CleanupResult(resources_removed=[cluster_name])

    def fake_cleanup_jobs_controller(**kwargs):
        raise AssertionError("cleanup_all_for_run must not tear down the shared controller by default")

    monkeypatch.setattr(cleanup_module, "_matching_jobs", fake_matching_jobs)
    monkeypatch.setattr(cleanup_module, "_cancel_job", fake_cancel)
    monkeypatch.setattr(cleanup_module, "sky_down", fake_down)
    monkeypatch.setattr(cleanup_module, "cleanup_jobs_controller", fake_cleanup_jobs_controller)

    result = cleanup_all_for_run(run_id, sky_bin=sky_bin)

    assert "cancel:7" in calls
    assert any(call.startswith("down:") and "20260516t011706z" in call for call in calls)
    assert not any(call.startswith("down:*") for call in calls)
    assert "sky-jobs-controller-abc123" not in result.resources_removed
    assert cluster_name_patterns_for_run(run_id)[0] == run_tag(run_id)


def test_cluster_name_patterns_for_run_rejects_short_run_id() -> None:
    with pytest.raises(InvalidRunIdError, match="at least 12 characters"):
        cluster_name_patterns_for_run("run-1234567")


@pytest.mark.parametrize(
    "unsafe",
    [
        "safe-run-123*",
        "safe-run-123?",
        "safe-run-[123",
        "safe-run-]123",
        "safe-run-{123",
        "safe-run-}123",
    ],
)
def test_cluster_name_patterns_for_run_rejects_glob_metachars(unsafe: str) -> None:
    with pytest.raises(InvalidRunIdError, match="ASCII letters"):
        cluster_name_patterns_for_run(unsafe)


@pytest.mark.parametrize("unsafe", ["safe-run 123", "safe-run-123$", "safe-run-123`", "safe-run-123;"])
def test_cluster_name_patterns_for_run_rejects_shell_special_chars(unsafe: str) -> None:
    with pytest.raises(InvalidRunIdError, match="ASCII letters"):
        cluster_name_patterns_for_run(unsafe)


def test_cluster_name_patterns_for_run_accepts_bootstrap_convention() -> None:
    run_id = "w9skypilot-bootstrap-converge-20260516T125841Z"

    patterns = cluster_name_patterns_for_run(run_id)

    assert run_tag(run_id) in patterns
    assert f"{run_tag(run_id)}-*" in patterns


def test_cluster_name_patterns_for_run_uses_boundary_aware_pattern() -> None:
    run_id = "w9skypilot-bootstrap-converge-20260516T125841Z"
    tag = run_tag(run_id)

    patterns = cluster_name_patterns_for_run(run_id)

    assert f"{tag}-*" in patterns
    assert f"*{tag}*" not in patterns
    assert f"{tag}*" not in patterns
    assert not any(pattern.startswith("*") for pattern in patterns)


def test_cluster_name_patterns_for_run_excludes_substring_collision() -> None:
    run_id = "w9skypilot-bootstrap-converge-20260516T125841Z"
    tag = run_tag(run_id)
    patterns = cluster_name_patterns_for_run(run_id)

    intended_cluster = f"{tag}-stage-1"
    unrelated_cluster = f"production-{tag}-stage-1"

    assert any(fnmatchcase(intended_cluster, pattern) for pattern in patterns)
    assert not any(fnmatchcase(unrelated_cluster, pattern) for pattern in patterns)


def test_cleanup_all_for_run_rejects_invalid_run_id_before_cleanup_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("cleanup operations should not run for an invalid run_id")

    monkeypatch.setattr(cleanup_module, "_matching_jobs", fail)
    monkeypatch.setattr(cleanup_module, "_cancel_job", fail)
    monkeypatch.setattr(cleanup_module, "sky_down", fail)
    monkeypatch.setattr(cleanup_module, "cleanup_jobs_controller", fail)

    with pytest.raises(InvalidRunIdError):
        cleanup_all_for_run("abc")


def test_cleanup_all_for_run_does_not_touch_controller_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    controller_calls: list[str] = []

    monkeypatch.setattr(cleanup_module, "_matching_jobs", lambda run_id, **kwargs: [])
    monkeypatch.setattr(
        cleanup_module,
        "sky_down",
        lambda cluster_name, **kwargs: CleanupResult(resources_removed=[cluster_name]),
    )
    monkeypatch.setattr(
        cleanup_module,
        "cleanup_jobs_controller",
        lambda **kwargs: controller_calls.append("controller") or CleanupResult(resources_removed=["controller"]),
    )

    cleanup_all_for_run("w9skypilot-controller-default-20260516T151040Z")

    assert controller_calls == []


def test_cleanup_all_for_run_touches_controller_when_explicitly_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    controller_calls: list[str] = []

    monkeypatch.setattr(cleanup_module, "_matching_jobs", lambda run_id, **kwargs: [])
    monkeypatch.setattr(
        cleanup_module,
        "sky_down",
        lambda cluster_name, **kwargs: CleanupResult(resources_removed=[cluster_name]),
    )
    monkeypatch.setattr(
        cleanup_module,
        "cleanup_jobs_controller",
        lambda **kwargs: controller_calls.append("controller")
        or CleanupResult(resources_removed=["sky-jobs-controller-abc123"]),
    )

    result = cleanup_all_for_run("w9skypilot-controller-optin-20260516T151040Z", also_teardown_controller=True)

    assert controller_calls == ["controller"]
    assert "sky-jobs-controller-abc123" in result.resources_removed


def test_concurrent_cleanup_safety(monkeypatch: pytest.MonkeyPatch) -> None:
    down_calls: list[str] = []
    errors: list[BaseException] = []

    monkeypatch.setattr(cleanup_module, "_matching_jobs", lambda run_id, **kwargs: [])
    monkeypatch.setattr(
        cleanup_module,
        "sky_down",
        lambda cluster_name, **kwargs: down_calls.append(cluster_name)
        or CleanupResult(resources_removed=[cluster_name]),
    )
    monkeypatch.setattr(
        cleanup_module,
        "cleanup_jobs_controller",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("controller cleanup should not run")),
    )

    def run_cleanup(run_id: str) -> None:
        try:
            cleanup_all_for_run(run_id)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=run_cleanup, args=("w9skypilot-concurrent-a-20260516T151040Z",)),
        threading.Thread(target=run_cleanup, args=("w9skypilot-concurrent-b-20260516T151040Z",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert down_calls


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


def test_cleanup_jobs_controller_verifies_kubernetes_controller_pod_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    sky_bin = _fake_sky(tmp_path)
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == str(sky_bin) and cmd[1] == "status":
            stdout = '[{"name": "sky-jobs-controller-k8s", "status": "UP", "infra": "Kubernetes"}]'
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        if cmd[0] == str(sky_bin):
            return subprocess.CompletedProcess(cmd, 0, stdout="down\n", stderr="")
        if cmd[:4] == ["kubectl", "get", "pods", "--all-namespaces"]:
            return subprocess.CompletedProcess(cmd, 0, stdout='{"items": []}', stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(cleanup_module.shutil, "which", lambda name: "/usr/bin/kubectl" if name == "kubectl" else None)

    result = cleanup_jobs_controller(isolated_config_dir=tmp_path, sky_bin=sky_bin)

    assert ["kubectl", "get", "pods", "--all-namespaces", "-o", "json"] in calls
    assert result.resources_removed == ["sky-jobs-controller-k8s"]
    assert result.errors == []


def test_cleanup_jobs_controller_deletes_lingering_kubernetes_controller_pod(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    sky_bin = _fake_sky(tmp_path)

    def fake_run(cmd, **kwargs):
        if cmd[0] == str(sky_bin) and cmd[1] == "status":
            stdout = '[{"name": "sky-jobs-controller-k8s", "status": "UP", "infra": "Kubernetes"}]'
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        if cmd[0] == str(sky_bin):
            return subprocess.CompletedProcess(cmd, 0, stdout="down\n", stderr="")
        if cmd[:4] == ["kubectl", "get", "pods", "--all-namespaces"]:
            stdout = (
                '{"items": [{"metadata": {"namespace": "default", '
                '"name": "sky-jobs-controller-k8s-ray-head", '
                '"labels": {"ray.io/cluster": "sky-jobs-controller-k8s"}}}]}'
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        if cmd[:4] == ["kubectl", "delete", "pod", "-n"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="deleted\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(cleanup_module.shutil, "which", lambda name: "/usr/bin/kubectl" if name == "kubectl" else None)

    result = cleanup_jobs_controller(isolated_config_dir=tmp_path, sky_bin=sky_bin)

    assert "sky-jobs-controller-k8s" in result.resources_removed
    assert "k8s-pod:default/sky-jobs-controller-k8s-ray-head" in result.resources_removed
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
