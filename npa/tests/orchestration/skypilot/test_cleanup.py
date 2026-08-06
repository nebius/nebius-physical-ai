from __future__ import annotations

from fnmatch import fnmatchcase
import inspect
import subprocess
import threading
from pathlib import Path

import pytest

from npa.orchestration.skypilot import _bin as bin_module
from npa.orchestration.skypilot import cleanup as cleanup_module
from npa.orchestration.skypilot.cleanup import (
    CleanupResult,
    InvalidRunIdError,
    cleanup_all_for_run,
    cleanup_launched_workflow,
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


@pytest.fixture(autouse=True)
def _skip_version_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cleanup_module, "ensure_skypilot_version", lambda sky_bin: Path(sky_bin))
    monkeypatch.setattr(bin_module, "CONFIG_PATH", tmp_path / "missing-config.yaml")
    monkeypatch.delenv("NPA_SKYPILOT_BIN", raising=False)
    monkeypatch.delenv("SKYPILOT_GLOBAL_CONFIG", raising=False)
    monkeypatch.delenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR", raising=False)


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


def test_cleanup_launched_workflow_uses_exact_job_and_keeps_controller(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sky_bin = _fake_sky(tmp_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        cleanup_module,
        "_cancel_job",
        lambda job_id, **_kwargs: calls.append(("cancel", job_id))
        or CleanupResult(resources_removed=[f"job:{job_id}"]),
    )
    monkeypatch.setattr(
        cleanup_module,
        "wait_for_jobs_terminal",
        lambda job_ids, **_kwargs: (True, []),
    )
    monkeypatch.setattr(
        cleanup_module,
        "_verify_managed_job_convergence",
        lambda *args, **kwargs: "terminal",
    )
    monkeypatch.setattr(
        cleanup_module,
        "sky_down",
        lambda cluster, **_kwargs: calls.append(("down", cluster))
        or CleanupResult(resources_removed=[cluster]),
    )
    monkeypatch.setattr(
        cleanup_module,
        "cleanup_jobs_controller",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("per-run cancel must keep the shared controller")
        ),
    )

    result = cleanup_launched_workflow(
        "73", "ordinary-run", cluster="ordinary-cluster", sky_bin=sky_bin
    )

    assert result.ok
    assert calls == [("cancel", "73"), ("down", "ordinary-cluster")]
    assert result.resources_removed == ["job:73", "ordinary-cluster"]


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


def test_cleanup_all_for_run_with_no_matching_jobs_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    cancel_calls: list[str] = []

    monkeypatch.setattr(cleanup_module, "_matching_jobs", lambda run_id, **kwargs: [])
    monkeypatch.setattr(
        cleanup_module,
        "_cancel_job",
        lambda job_id, **kwargs: cancel_calls.append(job_id) or CleanupResult(errors=["unexpected"]),
    )
    monkeypatch.setattr(
        cleanup_module,
        "sky_down",
        lambda cluster_name, **kwargs: CleanupResult(resources_removed=[cluster_name]),
    )
    monkeypatch.setattr(
        cleanup_module,
        "cleanup_jobs_controller",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("controller cleanup should not run")),
    )

    result = cleanup_all_for_run("w9skypilot-no-jobs-20260516T151040Z")

    assert result.ok
    assert cancel_calls == []
    assert result.resources_removed


def test_cleanup_all_for_run_logs_cancel_failure_and_continues(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    sky_bin = _fake_sky(tmp_path)

    monkeypatch.setattr(
        cleanup_module,
        "_matching_jobs",
        lambda run_id, **kwargs: [{"job_id": "99", "name": f"{run_id}-task", "status": "RUNNING"}],
    )
    monkeypatch.setattr(
        cleanup_module,
        "sky_down",
        lambda cluster_name, **kwargs: CleanupResult(resources_removed=[cluster_name]),
    )

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="cancel refused")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = cleanup_all_for_run("w9skypilot-cancel-fail-20260516T151040Z", sky_bin=sky_bin)

    assert any("cancel refused" in error for error in result.errors)
    assert result.resources_removed


def test_cleanup_all_for_run_controller_opt_out_does_not_status_controller(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    sky_bin = _fake_sky(tmp_path)
    calls: list[list[str]] = []

    monkeypatch.setattr(cleanup_module, "_matching_jobs", lambda run_id, **kwargs: [])

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert cmd[1] != "status"
        return subprocess.CompletedProcess(cmd, 0, stdout="down\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = cleanup_all_for_run("w9skypilot-controller-optout-20260516T151040Z", sky_bin=sky_bin)

    assert result.errors == []
    assert calls
    assert all(cmd[1] == "down" for cmd in calls)


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


def test_jobs_controller_inventory_discovers_exact_name_without_refresh(
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

    clusters, error = cleanup_module._jobs_controller_clusters(
        isolated_config_dir=tmp_path,
        config_path=None,
        sky_bin=sky_bin,
        refresh=False,
    )

    assert calls[0][0] == [str(sky_bin), "status", "--output", "json"]
    assert calls[0][1] is None
    assert [item["name"] for item in clusters] == ["sky-jobs-controller-abc123"]
    assert error == ""


def test_exact_context_controller_pod_inventory_can_prove_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    calls: list[list[str]] = []
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("{}\n", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:5] == ["kubectl", "--context", "verified", "get", "pods"]:
            return subprocess.CompletedProcess(cmd, 0, stdout='{"items": []}', stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    pods, error = cleanup_module._kubernetes_controller_pods(
        kubeconfig=kubeconfig, context="verified"
    )

    assert calls == [
        [
            "kubectl",
            "--context",
            "verified",
            "get",
            "pods",
            "--all-namespaces",
            "-o",
            "json",
        ]
    ]
    assert pods == []
    assert error == ""


def test_controller_pod_inventory_never_directly_deletes_a_lingering_pod(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("{}\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:5] == ["kubectl", "--context", "verified", "get", "pods"]:
            stdout = (
                '{"items": [{"metadata": {"namespace": "default", '
                '"name": "sky-jobs-controller-k8s-ray-head", '
                '"labels": {"ray.io/cluster": "sky-jobs-controller-k8s"}}}]}'
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    pods, error = cleanup_module._kubernetes_controller_pods(
        kubeconfig=kubeconfig, context="verified"
    )

    assert pods == [
        (
            "default",
            "sky-jobs-controller-k8s-ray-head",
            "sky-jobs-controller-k8s",
        )
    ]
    assert error == ""
    assert not any("delete" in call for call in calls)


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


# --- `sky jobs cancel` -> `sky down` race -------------------------------------
#
# `sky jobs cancel` returns once cancellation is *scheduled*. The controller keeps
# reporting the job as CANCELLING, and `sky down` on the controller refuses while
# any managed job is non-terminal, so cancelling and immediately tearing down fails
# with an error telling the operator to do what they just did.

_IN_PROGRESS_ERROR = (
    "sky.exceptions.NotSupportedError: In-progress managed jobs found. "
    "To avoid resource leakage, cancel all jobs first: sky jobs cancel -a"
)


def _queue(*jobs: dict[str, object]) -> str:
    import json as _json

    return _json.dumps(list(jobs))


def test_cleanup_all_for_run_waits_for_cancelled_jobs_before_tearing_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    sky_bin = _fake_sky(tmp_path)
    calls: list[list[str]] = []
    queue_reads = {"n": 0}

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1:3] == ["jobs", "queue"]:
            queue_reads["n"] += 1
            # First read finds the live job; after cancel it lingers as
            # CANCELLING, then finally reports CANCELLED.
            status = {1: "RUNNING", 2: "CANCELLING"}.get(queue_reads["n"], "CANCELLED")
            return subprocess.CompletedProcess(
                cmd, 0, stdout=_queue({"job_id": 7, "name": "run-abc123456789", "status": status}), stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(cleanup_module.time, "sleep", lambda _seconds: None)

    result = cleanup_all_for_run("run-abc123456789", isolated_config_dir=tmp_path, sky_bin=sky_bin)

    verbs = [cmd[1:3] for cmd in calls]
    cancel_at = verbs.index(["jobs", "cancel"])
    down_at = next(i for i, cmd in enumerate(calls) if cmd[1] == "down")
    # The queue is re-read between cancel and down until the job is terminal.
    assert ["jobs", "queue"] in verbs[cancel_at:down_at]
    assert queue_reads["n"] >= 3
    assert result.errors == []


def test_cleanup_all_for_run_reports_a_job_that_never_finishes_cancelling(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    sky_bin = _fake_sky(tmp_path)

    def fake_run(cmd, **kwargs):
        if cmd[1:3] == ["jobs", "queue"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=_queue({"job_id": 7, "name": "run-abc123456789", "status": "CANCELLING"}), stderr=""
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(cleanup_module.time, "sleep", lambda _seconds: None)

    result = cleanup_all_for_run(
        "run-abc123456789", isolated_config_dir=tmp_path, sky_bin=sky_bin, job_drain_timeout=0
    )

    assert any("still non-terminal" in error for error in result.errors)
    assert any("7" in error for error in result.errors)


def test_controller_teardown_retries_after_the_in_progress_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    sky_bin = _fake_sky(tmp_path)
    downs = {"n": 0}
    queue_reads = {"n": 0}

    def fake_run(cmd, **kwargs):
        if cmd[1] == "status":
            return subprocess.CompletedProcess(
                cmd, 0, stdout='[{"name": "sky-jobs-controller-abc123", "status": "UP"}]', stderr=""
            )
        if cmd[1:3] == ["jobs", "queue"]:
            queue_reads["n"] += 1
            status = "CANCELLING" if queue_reads["n"] == 1 else "CANCELLED"
            return subprocess.CompletedProcess(
                cmd, 0, stdout=_queue({"job_id": 2, "status": status}), stderr=""
            )
        if cmd[1] == "down":
            downs["n"] += 1
            # A job that finished cancelling between the poll and this call still
            # trips the guard the first time.
            if downs["n"] == 1:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=_IN_PROGRESS_ERROR)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(cleanup_module.time, "sleep", lambda _seconds: None)

    result = cleanup_module._down_jobs_controller(
        "sky-jobs-controller-abc123",
        isolated_config_dir=tmp_path,
        config_path=None,
        sky_bin=sky_bin,
    )

    assert downs["n"] == 2
    assert result.resources_removed == ["sky-jobs-controller-abc123"]
    assert result.errors == []


def test_controller_teardown_explains_a_job_that_will_not_drain(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    sky_bin = _fake_sky(tmp_path)

    def fake_run(cmd, **kwargs):
        if cmd[1] == "status":
            return subprocess.CompletedProcess(
                cmd, 0, stdout='[{"name": "sky-jobs-controller-abc123", "status": "UP"}]', stderr=""
            )
        if cmd[1:3] == ["jobs", "queue"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=_queue({"job_id": 2, "status": "RUNNING"}), stderr=""
            )
        if cmd[1] == "down":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=_IN_PROGRESS_ERROR)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(cleanup_module.time, "sleep", lambda _seconds: None)

    result = cleanup_module._down_jobs_controller(
        "sky-jobs-controller-abc123",
        isolated_config_dir=tmp_path,
        config_path=None,
        sky_bin=sky_bin,
        job_drain_timeout=0,
    )

    assert result.resources_removed == []
    joined = " ".join(result.errors)
    assert "refuses while managed job(s) 2" in joined
    assert "sky jobs cancel -a" in joined


def test_a_readable_queue_that_is_already_terminal_does_not_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    sky_bin = _fake_sky(tmp_path)
    slept: list[float] = []

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout=_queue({"job_id": 7, "status": "SUCCEEDED"}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    drained, still_running = cleanup_module.wait_for_jobs_terminal(
        ["7"], isolated_config_dir=tmp_path, sky_bin=sky_bin, sleep=slept.append
    )

    assert drained is True
    assert still_running == []
    assert slept == []


def test_an_unreadable_queue_does_not_stall_teardown(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # Cleanup is best-effort; a broken controller must not hold teardown hostage.
    sky_bin = _fake_sky(tmp_path)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="controller unreachable")

    monkeypatch.setattr(subprocess, "run", fake_run)

    drained, still_running = cleanup_module.wait_for_jobs_terminal(
        ["7"], isolated_config_dir=tmp_path, sky_bin=sky_bin, sleep=lambda _s: None
    )

    assert drained is True
    assert still_running == []


def test_a_job_group_is_terminal_only_when_every_task_is(tmp_path) -> None:
    statuses = cleanup_module._job_statuses(
        [
            {"job_id": 3, "task_id": 0, "status": "SUCCEEDED"},
            {"job_id": 3, "task_id": 1, "status": "RUNNING"},
        ]
    )

    assert statuses["3"] == "RUNNING"


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        (_IN_PROGRESS_ERROR, True),
        ("In progress managed jobs found", True),
        ("cluster not found", False),
        ("", False),
    ],
)
def test_in_progress_guard_is_recognized(detail: str, expected: bool) -> None:
    assert cleanup_module.looks_like_in_progress_jobs_error(detail) is expected
