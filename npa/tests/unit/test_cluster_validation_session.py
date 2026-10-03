"""Exercise owned cluster-validation identity, recovery, and session lifetimes."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading

import pytest

from npa.cli.cluster import terraform_lifecycle as lifecycle
from npa.orchestration.skypilot import (
    _bin,
    cluster_validation,
    k8s_gpu_catalog,
    local_api,
)


_HOST_GUARD = local_api._require_linux_host


@pytest.fixture
def api(tmp_path, monkeypatch):
    selected = tmp_path / "selected-kubeconfig"
    selected.write_text("apiVersion: v1\ncontexts: []\n")
    monkeypatch.setattr(_bin, "CONFIG_PATH", tmp_path / "npa" / "config.yaml")
    monkeypatch.setattr(local_api, "_require_linux_host", lambda: None)
    starts, stops, commands = [], [], []
    monkeypatch.setattr(
        local_api, "ensure_isolated_api", lambda **kwargs: starts.append(kwargs)
    )
    monkeypatch.setattr(local_api, "stop_isolated_api", stops.append)
    monkeypatch.setattr(lifecycle, "_require_bin", lambda value: value)
    monkeypatch.setattr(k8s_gpu_catalog, "resolve_sky_bin", lambda value: Path(value))

    def command(argv, **kwargs):
        commands.append((argv, kwargs))
        if argv[1] == "launch":
            session = cluster_validation.current_validation_session()
            assert (
                json.loads((session.scope / "session.json").read_text())["phase"]
                == "smoke_pending"
            )
        output = "Kubernetes: enabled [compute]\n"
        if argv[1] == "show-gpus":
            output = (
                "Context: exact-context\nGPU REQUESTABLE_QTY_PER_NODE\nRTXPRO6000 1\n"
            )
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(lifecycle, "_run_stream", command)
    monkeypatch.setattr(lifecycle, "_wait_for_sky_down", lambda *_args, **_kwargs: None)
    return selected, starts, stops, commands, command


def _smoke(api, **kwargs):
    lifecycle._run_skypilot_smoke(
        api[0],
        "exact-context",
        "validation",
        "RTXPRO6000:1",
        sky_bin=kwargs.pop("sky_bin", "/selected/python312/bin/sky"),
        **kwargs,
    )


def test_default_validation_owns_one_endpoint_through_check_discovery_and_cleanup(
    api, monkeypatch
):
    selected, starts, stops, commands, command = api
    monkeypatch.delenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR", raising=False)
    ambient = dict(os.environ)
    with cluster_validation.cluster_validation_session(
        selected, "exact-context"
    ) as session:
        lifecycle._check_skypilot_kubernetes(
            selected, "exact-context", sky_bin="/selected/python312/bin/sky"
        )
        k8s_gpu_catalog.discover_kubernetes_gpu_catalog(
            context="exact-context",
            kubeconfig=selected,
            sky_bin="/selected/python312/bin/sky",
            runner=command,
        )
        _smoke(api, credentials_checked=True)
        assert not stops
    assert dict(os.environ) == ambient
    assert stops == [session.scope]
    assert [argv[1] for argv, _kwargs in commands] == [
        "check",
        "show-gpus",
        "launch",
        "down",
    ]
    environments = [kwargs["env"] for _argv, kwargs in commands]
    endpoints = {env["SKYPILOT_API_SERVER_ENDPOINT"] for env in environments}
    assert len(endpoints) == 1 and "http://127.0.0.1:46580" not in endpoints
    assert all(
        start["sky_executable"] == "/selected/python312/bin/sky" for start in starts
    )
    assert all(start["cwd"] == str(session.scope) for start in starts)
    assert all(Path(kwargs["cwd"]) == session.scope for _argv, kwargs in commands)
    assert session.scope.is_dir() and session.scope != selected.parent
    assert all(Path(env["KUBECONFIG"]) == selected for env in environments)
    assert all(
        (Path(env["HOME"]) / ".kube/config").resolve() == selected
        for env in environments
    )


def test_explicit_unowned_api_is_refused_before_client_or_server_start(
    api, monkeypatch
):
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "http://127.0.0.1:46580")
    with pytest.raises(
        local_api.IsolatedApiError, match="different configured API endpoint"
    ):
        _smoke(api)
    assert not api[1] and not api[2] and not api[3]


def test_direct_smoke_cannot_reuse_credentials_check_from_a_closed_session(api):
    lifecycle._check_skypilot_kubernetes(
        api[0], "exact-context", sky_bin="/selected/sky"
    )
    first = api[2][0]
    _smoke(api, credentials_checked=True)
    assert [argv[1] for argv, _kwargs in api[3]] == ["check", "check", "launch", "down"]
    assert api[2][-1] != first


@pytest.mark.parametrize("check_fails", [False, True])
def test_direct_check_releases_no_launch_api(api, monkeypatch, check_fails):
    if check_fails:
        monkeypatch.setattr(
            lifecycle,
            "_run_stream",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("check failed")),
        )
        with pytest.raises(RuntimeError, match="check failed"):
            lifecycle._check_skypilot_kubernetes(
                api[0], "exact-context", sky_bin="/selected/sky"
            )
    else:
        lifecycle._check_skypilot_kubernetes(
            api[0], "exact-context", sky_bin="/selected/sky"
        )
    assert len(api[2]) == 1
    assert json.loads((api[2][0] / "session.json").read_text())["phase"] == "complete"


@pytest.mark.parametrize("failure", ["launch", "down", "absence", "launch-and-down"])
def test_smoke_cleanup_requires_exact_absence_and_preserves_primary_failure(
    api, monkeypatch, failure
):
    command = api[4]

    def fail(argv, **kwargs):
        if argv[1] in failure.split("-and-"):
            raise RuntimeError(argv[1] + " failed")
        return command(argv, **kwargs)

    monkeypatch.setattr(lifecycle, "_run_stream", fail)
    if failure == "absence":
        monkeypatch.setattr(
            lifecycle,
            "_wait_for_sky_down",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("absence unverified")),
        )
    with pytest.raises(RuntimeError, match=failure.split("-and-")[0]):
        _smoke(api)
    scope = api[1][0]["isolated_dir"]
    record = json.loads((scope / "session.json").read_text())
    if failure == "launch":
        assert api[2] == [scope] and record["phase"] == "complete"
    else:
        assert not api[2] and record["phase"] == "smoke_pending"
        assert record["smoke_name"] == "validation-sky-smoke"


def test_pending_recovery_cleans_exact_smoke_before_readiness_and_reuses_identity(
    api, monkeypatch
):
    monkeypatch.setenv("SKYPILOT_USER_ID", "ambient-operator")
    with pytest.raises(RuntimeError, match="interrupted"):
        with cluster_validation.cluster_validation_session(
            api[0], "exact-context"
        ) as session:
            lifecycle._check_skypilot_kubernetes(
                api[0], "exact-context", sky_bin="/selected/sky"
            )
            session.begin_smoke("validation-sky-smoke")
            raise RuntimeError("interrupted")
    assert not api[2]
    with pytest.raises(local_api.IsolatedApiError, match="removal remains unverified"):
        lifecycle._check_skypilot_kubernetes(
            api[0], "exact-context", sky_bin="/selected/sky"
        )
    assert all(argv[1] == "check" for argv, _kwargs in api[3])

    def readiness(*_args, **_kwargs):
        assert api[3][-1][0][1] == "down"
        assert not cluster_validation.current_validation_session().pending_smoke
        return {}

    monkeypatch.setattr(k8s_gpu_catalog, "wait_for_kubernetes_accelerators", readiness)
    lifecycle._validate_skypilot_readiness(
        api[0], "exact-context", "validation", "RTXPRO6000:1", sky_bin="/selected/sky"
    )
    assert {start["isolated_dir"] for start in api[1]} == {session.scope}
    identities = {start["environment"]["SKYPILOT_USER_ID"] for start in api[1]}
    assert len(identities) == 1 and "ambient-operator" not in identities
    assert api[2] == [session.scope]
    assert [argv[1] for argv, _kwargs in api[3]][-4:] == [
        "check",
        "down",
        "launch",
        "down",
    ]


def test_completed_session_rotates_scope_for_new_binary_and_credentials(
    api, monkeypatch
):
    _smoke(api)
    first = api[1][0]["isolated_dir"]
    monkeypatch.setenv("AWS_PROFILE", "rotated-profile")
    _smoke(api, sky_bin="/fresh/python312/bin/sky")
    second = api[1][-1]["isolated_dir"]
    assert first != second
    assert api[2] == [first, second]
    assert (first / "session.json").is_file()
    assert api[1][-1]["environment"]["AWS_PROFILE"] == "rotated-profile"
    assert api[1][-1]["sky_executable"] == "/fresh/python312/bin/sky"


def test_explicit_root_preserves_pre_session_api_and_uses_owned_child(
    api, monkeypatch, tmp_path
):
    root = tmp_path / "explicit"
    monkeypatch.setenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR", str(root))
    identity = hashlib.sha256(
        f"{api[0].resolve()}\0exact-context".encode()
    ).hexdigest()[:24]
    legacy = root / "cluster-validation" / identity / "local-api"
    legacy.mkdir(parents=True)
    marker = legacy / "daemon.json"
    marker.write_text('{"legacy": true}')
    _smoke(api)
    scope = api[1][0]["isolated_dir"]
    assert root in scope.parents and scope.parent == legacy.parent
    assert marker.read_text() == '{"legacy": true}'
    assert legacy.parent not in api[2]


def test_same_target_sessions_serialize_threads_and_nesting_reuses_scope(api):
    entered, release, attempted = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    order = []

    def first():
        with cluster_validation.cluster_validation_session(
            api[0], "exact-context"
        ) as outer:
            with cluster_validation.cluster_validation_session(
                api[0], "exact-context"
            ) as inner:
                assert inner is outer
            order.append("first")
            entered.set()
            assert release.wait(5)

    def second():
        assert entered.wait(5)
        attempted.set()
        with cluster_validation.cluster_validation_session(api[0], "exact-context"):
            order.append("second")

    with ThreadPoolExecutor(2) as pool:
        a, b = pool.submit(first), pool.submit(second)
        assert attempted.wait(5)
        assert order == ["first"]
        lock_path = next(_bin.CONFIG_PATH.parent.rglob("session.lock"))
        with lock_path.open() as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        release.set()
        a.result(timeout=5)
        b.result(timeout=5)
    assert order == ["first", "second"]


def test_unsupported_host_fails_before_session_state(api, monkeypatch):
    monkeypatch.setattr(local_api, "_require_linux_host", _HOST_GUARD)
    monkeypatch.setattr(local_api.sys, "platform", "darwin")
    with pytest.raises(local_api.IsolatedApiError, match="Linux operator host"):
        _smoke(api)
    assert not list(_bin.CONFIG_PATH.parent.rglob("current-session.json"))
    assert not api[1] and not api[3]


def test_api_stop_failure_reports_recovery_without_publishing_source_text(
    api, monkeypatch
):
    def stop(_scope):
        raise RuntimeError("private-provider-text")

    monkeypatch.setattr(local_api, "stop_isolated_api", stop)
    with pytest.raises(local_api.IsolatedApiError, match="Private session:") as caught:
        lifecycle._check_skypilot_kubernetes(
            api[0], "exact-context", sky_bin="/selected/sky"
        )
    assert "private-provider-text" not in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)
    scope = api[1][0]["isolated_dir"]
    assert json.loads((scope / "session.json").read_text())["phase"] == "checking"
    attempts = len(api[1])
    with pytest.raises(
        local_api.IsolatedApiError, match="Private session:"
    ) as recovery:
        lifecycle._check_skypilot_kubernetes(
            api[0], "exact-context", sky_bin="/selected/sky"
        )
    assert "private-provider-text" not in str(recovery.value)
    assert len(api[1]) == attempts
    assert json.loads((scope / "session.json").read_text())["phase"] == "checking"


def test_cleanup_preserves_primary_exception_without_python311_add_note(
    api, monkeypatch
):
    class LegacyFailure(RuntimeError):
        add_note = None

    def fail(argv, **kwargs):
        if argv[1] == "launch":
            raise LegacyFailure("original launch failed")
        if argv[1] == "down":
            raise RuntimeError("cleanup failed")
        return api[4](argv, **kwargs)

    monkeypatch.setattr(lifecycle, "_run_stream", fail)
    with pytest.raises(LegacyFailure, match="original launch failed"):
        _smoke(api)
    assert not api[2]


def test_independent_context_sessions_keep_concurrent_api_identities_separate(
    api, monkeypatch
):
    monkeypatch.setenv("SKYPILOT_USER_ID", "ambient-operator")
    ready = threading.Barrier(2)

    def validate(context):
        with cluster_validation.cluster_validation_session(api[0], context) as session:
            _, env, _ = lifecycle._check_skypilot_kubernetes(
                api[0], context, sky_bin="/selected/sky"
            )
            ready.wait(timeout=5)
            return (
                session.scope,
                env["SKYPILOT_API_SERVER_ENDPOINT"],
                env["SKYPILOT_USER_ID"],
            )

    with ThreadPoolExecutor(2) as pool:
        a, b = (
            pool.submit(validate, "first-context"),
            pool.submit(validate, "second-context"),
        )
        first, second = a.result(timeout=5), b.result(timeout=5)
    assert all(left != right for left, right in zip(first, second))
    assert set(api[2]) == {first[0], second[0]}


def test_pending_smoke_name_mismatch_never_cancels_or_launches(api):
    with pytest.raises(RuntimeError, match="interrupted"):
        with cluster_validation.cluster_validation_session(
            api[0], "exact-context"
        ) as session:
            session.begin_smoke("original-sky-smoke")
            raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="original cluster command"):
        _smoke(api)
    assert not api[2]
    assert all(argv[1] == "check" for argv, _kwargs in api[3])
