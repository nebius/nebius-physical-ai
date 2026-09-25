"""Prove targeted CLI checks never reach an ambient SkyPilot API."""

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli import skypilot as skypilot_cli
from npa.cli.main import app
from npa.orchestration.skypilot import (
    _bin,
    cluster_validation,
    k8s_gpu_catalog,
    local_api,
)


def _selected_kubeconfig(tmp_path):
    kubeconfig = tmp_path / "selected-kubeconfig"
    kubeconfig.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "v1",
                "current-context": "selected-context",
                "contexts": [
                    {
                        "name": "selected-context",
                        "context": {
                            "cluster": "selected-cluster",
                            "user": "selected-user",
                        },
                    }
                ],
                "clusters": [
                    {
                        "name": "selected-cluster",
                        "cluster": {"server": "https://kubernetes.invalid"},
                    }
                ],
                "users": [{"name": "selected-user", "user": {"token": "fixture"}}],
            }
        )
    )
    return kubeconfig


@pytest.fixture
def owned_check(tmp_path, monkeypatch):
    kubeconfig = _selected_kubeconfig(tmp_path)
    sky = tmp_path / "selected-venv/bin/sky"
    sky.parent.mkdir(parents=True)
    sky.write_text("#!/bin/sh\nexit 0\n")
    sky.chmod(0o755)
    monkeypatch.setenv("KUBECONFIG", str(kubeconfig))
    monkeypatch.setenv("NPA_SKYPILOT_BIN", str(sky))
    monkeypatch.setattr(_bin, "CONFIG_PATH", tmp_path / "npa/config.yaml")
    monkeypatch.setattr(local_api, "_require_linux_host", lambda: None)
    monkeypatch.setattr(
        skypilot_cli,
        "inspect_venv",
        lambda _: SimpleNamespace(
            installed=True,
            kubernetes_compatible=True,
            sky_bin=sky,
            path=sky.parent.parent,
        ),
    )
    monkeypatch.setattr(_bin, "ensure_skypilot_version", lambda value=None: sky)
    starts, stops, calls, inventories = [], [], [], []
    monkeypatch.setattr(
        local_api, "ensure_isolated_api", lambda **kwargs: starts.append(kwargs)
    )
    monkeypatch.setattr(local_api, "stop_isolated_api", stops.append)
    monkeypatch.setattr(
        "npa.controller_ownership.verify_recorded_controller_owner", lambda: None
    )

    def inventory(**kwargs):
        inventories.append(kwargs)
        return k8s_gpu_catalog.KubernetesGpuInventory(
            "selected-context", 1, 1, 1, 1, ("RTXPRO6000",), {}
        )

    def execute(argv, **kwargs):
        calls.append((argv, kwargs))
        output = "Kubernetes: enabled [compute]\n"
        if "show-gpus" in argv:
            output = "Context: selected-context\nGPU  REQUESTABLE_QTY_PER_NODE\nRTXPRO6000  1\n"
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(k8s_gpu_catalog, "discover_kubernetes_gpu_inventory", inventory)
    monkeypatch.setattr(skypilot_cli, "_run_no_raise", execute)
    monkeypatch.setattr(k8s_gpu_catalog.subprocess, "run", execute)
    return SimpleNamespace(
        kubeconfig=kubeconfig,
        sky=sky,
        starts=starts,
        stops=stops,
        calls=calls,
        inventories=inventories,
        execute=execute,
        root=tmp_path,
    )


def _invoke(check, command, *extra):
    if command == "verify":
        argv = [
            "skypilot",
            "verify",
            "--cluster",
            "selected-context",
            "--kubeconfig",
            str(check.kubeconfig),
            "--output-format",
            "json",
        ]
    else:
        argv = [
            "workbench",
            "workflow",
            "gpus",
            "--context",
            "selected-context",
            "--json",
        ]
    return CliRunner().invoke(app, [*argv, *extra])


@pytest.mark.parametrize("command", ["verify", "gpus"])
def test_targeted_cli_owns_endpoint_and_closes_its_check_only_session(
    owned_check, command
):
    check = owned_check
    before = dict(os.environ)
    result = _invoke(check, command)
    assert result.exit_code == 0, result.output
    json.loads(result.stdout)
    assert check.starts, "Targeted command reached SkyPilot without an owned API"
    assert len(check.stops) == 1
    scope = check.stops[0]
    assert json.loads((scope / "session.json").read_text())["phase"] == "complete"
    assert all(start["sky_executable"] == str(check.sky) for start in check.starts)
    for argv, kwargs in check.calls:
        assert kwargs["env"]["SKYPILOT_API_SERVER_ENDPOINT"].startswith(
            "http://127.0.0.1:"
        )
        assert Path(kwargs["env"]["HOME"]) == scope / "home"
        assert Path(kwargs["cwd"]) == scope
    assert dict(os.environ) == before
    if command == "gpus":
        assert check.inventories == [
            {"context": "selected-context", "kubeconfig": check.kubeconfig}
        ]


@pytest.mark.parametrize("command", ["verify", "gpus"])
@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "malformed",
        "missing-context",
        "duplicate-context",
        "missing-cluster",
        "duplicate-cluster",
        "missing-user",
        "duplicate-user",
        "empty-server",
    ],
)
def test_invalid_target_fails_before_inventory_session_or_api(
    owned_check, command, invalid
):
    check = owned_check
    document = yaml.safe_load(check.kubeconfig.read_text())
    if invalid == "missing":
        check.kubeconfig.unlink()
    elif invalid == "malformed":
        check.kubeconfig.write_text("[invalid:")
    else:
        section = invalid.split("-")[-1] + "s"
        if invalid.startswith("missing"):
            document[section] = []
        elif invalid.startswith("duplicate"):
            document[section] *= 2
        else:
            document["clusters"][0]["cluster"]["server"] = ""
        check.kubeconfig.write_text(yaml.safe_dump(document))
    result = _invoke(check, command)
    assert result.exit_code != 0
    json.loads(result.stdout)
    assert not check.starts and not check.calls and not check.inventories
    assert not (check.root / "npa/cluster-validation").exists()


def test_multiple_kubeconfig_files_are_not_treated_as_one_path(
    owned_check, monkeypatch
):
    monkeypatch.setenv(
        "KUBECONFIG",
        str(owned_check.kubeconfig) + os.pathsep + str(owned_check.root / "second"),
    )
    result = _invoke(owned_check, "gpus")
    assert result.exit_code != 0
    assert "single KUBECONFIG" in json.loads(result.stdout)["skypilot_error"]
    assert (
        not owned_check.starts and not owned_check.calls and not owned_check.inventories
    )


@pytest.mark.parametrize("command", ["verify", "gpus"])
def test_explicit_target_wins_over_unrelated_current_context(owned_check, command):
    document = yaml.safe_load(owned_check.kubeconfig.read_text())
    document["current-context"] = "unrelated-context"
    owned_check.kubeconfig.write_text(yaml.safe_dump(document))
    result = _invoke(owned_check, command)
    assert result.exit_code == 0, result.output
    assert all(
        'kubernetes.allowed_contexts=["selected-context"]' in argv
        for argv, _ in owned_check.calls
    )


def test_explicit_discovery_root_precedes_environment_without_mutating_it(
    owned_check, monkeypatch
):
    monkeypatch.setenv(
        "NPA_SKYPILOT_ISOLATED_CONFIG_DIR", str(owned_check.root / "ambient-state")
    )
    explicit = owned_check.root / "selected-state"
    before = dict(os.environ)
    result = _invoke(owned_check, "gpus", "--isolated-config-dir", str(explicit))
    assert result.exit_code == 0, result.output
    assert owned_check.starts and all(
        start["isolated_dir"].is_relative_to(explicit) for start in owned_check.starts
    )
    assert not (owned_check.root / "ambient-state").exists()
    assert dict(os.environ) == before


@pytest.mark.parametrize("command", ["verify", "gpus"])
def test_unowned_endpoint_is_refused_before_any_cli_or_inventory_call(
    owned_check, monkeypatch, command
):
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "http://127.0.0.1:46580")
    result = _invoke(owned_check, command)
    assert result.exit_code != 0
    assert "different configured API endpoint" in result.output
    assert (
        not owned_check.starts and not owned_check.calls and not owned_check.inventories
    )


@pytest.mark.parametrize("command", ["verify", "gpus"])
def test_pending_smoke_is_preserved_without_checking_or_stopping_it(
    owned_check, command
):
    with pytest.raises(local_api.IsolatedApiError, match="removal remains unverified"):
        with cluster_validation.cluster_validation_session(
            owned_check.kubeconfig, "selected-context"
        ) as session:
            k8s_gpu_catalog.kubernetes_sky_environment(
                context="selected-context",
                kubeconfig=owned_check.kubeconfig,
                sky_executable=str(owned_check.sky),
            )
            session.begin_smoke("validation-sky-smoke")
    before = {
        path: path.read_bytes() for path in session.scope.rglob("*") if path.is_file()
    }
    owned_check.starts.clear()
    result = _invoke(owned_check, command)
    assert result.exit_code != 0
    assert "requires recovery before standalone" in result.output
    assert (
        not owned_check.starts
        and not owned_check.stops
        and not owned_check.calls
        and not owned_check.inventories
    )
    assert before == {
        path: path.read_bytes() for path in session.scope.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("command", ["verify", "gpus"])
def test_nested_targeted_check_leaves_owned_session_to_outer_caller(
    owned_check, command
):
    with cluster_validation.cluster_validation_session(
        owned_check.kubeconfig, "selected-context"
    ) as session:
        result = _invoke(owned_check, command)
        assert result.exit_code == 0, result.output
        assert not owned_check.stops
        assert all(
            start["isolated_dir"] == session.scope for start in owned_check.starts
        )
    assert owned_check.stops == [session.scope]


def test_nested_explicit_root_disagreement_prevents_calls(owned_check):
    with cluster_validation.cluster_validation_session(
        owned_check.kubeconfig, "selected-context"
    ):
        result = _invoke(
            owned_check,
            "gpus",
            "--isolated-config-dir",
            str(owned_check.root / "other-state"),
        )
        assert result.exit_code != 0
        assert "different isolated state root" in result.output
        assert (
            not owned_check.starts
            and not owned_check.calls
            and not owned_check.inventories
        )


@pytest.mark.parametrize("command", ["verify", "gpus"])
@pytest.mark.parametrize(
    "operation_fails,cleanup_fails", [(True, False), (False, True), (True, True)]
)
def test_primary_failure_and_single_json_survive_owned_cleanup_uncertainty(
    owned_check,
    monkeypatch,
    command,
    operation_fails,
    cleanup_fails,
):
    check = owned_check
    original = check.execute

    def execute(argv, **kwargs):
        if operation_fails:
            check.calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 9, "", "selected check denied")
        return original(argv, **kwargs)

    def stop(scope):
        check.stops.append(scope)
        if cleanup_fails:
            raise local_api.IsolatedApiError("stop verification unavailable")

    monkeypatch.setattr(skypilot_cli, "_run_no_raise", execute)
    monkeypatch.setattr(k8s_gpu_catalog.subprocess, "run", execute)
    monkeypatch.setattr(local_api, "stop_isolated_api", stop)
    result = _invoke(check, command)
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    if operation_fails:
        assert "selected check denied" in json.dumps(payload)
    else:
        assert "cleanup is incomplete" in json.dumps(payload)
    assert len(check.stops) == 1
    phase = json.loads((check.stops[0] / "session.json").read_text())["phase"]
    assert phase == ("checking" if cleanup_fails else "complete")


def test_inventory_failure_preserves_environment_and_closes_owned_api(
    owned_check, monkeypatch
):
    monkeypatch.setattr(
        "npa.cluster.state.kubeconfig_file", lambda _: owned_check.kubeconfig
    )

    def inventory(**kwargs):
        raise RuntimeError("inventory unavailable")

    monkeypatch.setattr(k8s_gpu_catalog, "discover_kubernetes_gpu_inventory", inventory)
    before = dict(os.environ)
    result = _invoke(owned_check, "gpus", "--cluster", "selected-context")
    assert result.exit_code != 0
    assert "inventory unavailable" in json.loads(result.stdout)["skypilot_error"]
    assert dict(os.environ) == before
    assert len(owned_check.stops) == 1 and not owned_check.calls


def _project(monkeypatch, identifier="selected-project"):
    monkeypatch.setattr(
        "npa.clients.config.resolve_environment",
        lambda _: SimpleNamespace(project_id="selected-project"),
    )
    monkeypatch.setattr(
        "npa.cluster.state.load_cluster_state",
        lambda _: SimpleNamespace(project_id=identifier),
    )


def test_explicit_project_is_bound_without_consulting_shared_controller(
    owned_check, monkeypatch
):
    _project(monkeypatch)
    monkeypatch.setenv("NPA_SKYPILOT_PROJECT", "unrelated-project")
    monkeypatch.setattr(
        "npa.controller_ownership.verify_controller_owner",
        lambda *_: pytest.fail("shared owner consulted"),
    )
    monkeypatch.setattr(
        "npa.controller_ownership.verify_recorded_controller_owner",
        lambda: pytest.fail("shared owner consulted"),
    )
    result = _invoke(owned_check, "gpus", "--project", "selected-alias")
    assert result.exit_code == 0, result.output
    assert all(
        start["environment"]["NPA_SKYPILOT_PROJECT"] == "selected-alias"
        for start in owned_check.starts
    )
    assert (
        json.loads((owned_check.stops[0] / "session.json").read_text())["project_alias"]
        == "selected-alias"
    )


def test_project_mismatch_fails_before_session_or_inventory(owned_check, monkeypatch):
    _project(monkeypatch, "another-project")
    result = _invoke(owned_check, "gpus", "--project", "selected-alias")
    assert result.exit_code != 0
    assert "does not match" in result.output
    assert (
        not owned_check.starts and not owned_check.inventories and not owned_check.calls
    )
    assert not (owned_check.root / "npa/cluster-validation").exists()


def test_nested_project_disagreement_cannot_rebind_api(owned_check, monkeypatch):
    _project(monkeypatch)
    with cluster_validation.cluster_validation_session(
        owned_check.kubeconfig, "selected-context", project_alias="original-alias"
    ):
        result = _invoke(owned_check, "gpus", "--project", "selected-alias")
        assert result.exit_code != 0
        assert "different selected project" in result.output
        assert (
            not owned_check.starts
            and not owned_check.inventories
            and not owned_check.calls
        )


@pytest.mark.parametrize(
    "user",
    [
        {},
        {"exec": {"command": "credential-helper"}},
        {"client-certificate": "cert", "client-key": "key"},
    ],
)
def test_target_validation_does_not_restrict_existing_authentication_forms(
    owned_check, user
):
    document = yaml.safe_load(owned_check.kubeconfig.read_text())
    document["users"][0]["user"] = user
    owned_check.kubeconfig.write_text(yaml.safe_dump(document))
    assert cluster_validation.resolve_validation_target(
        owned_check.kubeconfig, "selected-context"
    ) == (owned_check.kubeconfig, "selected-context")


def test_text_discovery_fails_when_nonempty_catalog_cleanup_is_unverified(
    owned_check, monkeypatch
):
    def stop(_scope):
        raise local_api.IsolatedApiError("stop verification unavailable")

    monkeypatch.setattr(local_api, "stop_isolated_api", stop)
    result = CliRunner().invoke(
        app, ["workbench", "workflow", "gpus", "--context", "selected-context"]
    )
    assert owned_check.calls and "show-gpus" in owned_check.calls[-1][0]
    assert result.exit_code == 2
    assert "cleanup is incomplete" in result.output
    assert "export NPA_WORKFLOW_GPU_ACCELERATOR" not in result.output


def test_catalog_enable_check_and_retry_share_one_owned_endpoint(
    owned_check, monkeypatch
):
    calls = []

    def execute(argv, **kwargs):
        calls.append((argv, kwargs))
        output = (
            "Kubernetes is not enabled"
            if len(calls) == 1
            else (
                "Kubernetes: enabled"
                if "check" in argv
                else "Context: selected-context\nGPU  REQUESTABLE_QTY_PER_NODE\nRTXPRO6000  1\n"
            )
        )
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(k8s_gpu_catalog.subprocess, "run", execute)
    result = _invoke(owned_check, "gpus")
    assert result.exit_code == 0, result.output
    assert [argv[1] for argv, _ in calls] == ["show-gpus", "check", "show-gpus"]
    assert (
        len({kwargs["env"]["SKYPILOT_API_SERVER_ENDPOINT"] for _, kwargs in calls}) == 1
    )
    assert len({kwargs["cwd"] for _, kwargs in calls}) == 1
    assert len(owned_check.stops) == 1


def test_discovery_version_mismatch_cannot_start_api_or_read_inventory(
    owned_check, monkeypatch
):
    def wrong_version(_executable):
        raise _bin.SkyPilotVersionError("SkyPilot version mismatch")

    monkeypatch.setattr(_bin, "ensure_skypilot_version", wrong_version)
    result = _invoke(owned_check, "gpus")
    assert result.exit_code != 0 and "version mismatch" in result.output
    assert (
        not owned_check.starts and not owned_check.inventories and not owned_check.calls
    )


def test_concurrent_explicit_roots_do_not_share_api_or_mutate_configuration(
    owned_check, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from npa.cli.workbench import workflow
    import threading

    rendezvous = threading.Barrier(2)
    before = dict(os.environ)

    def execute(argv, **kwargs):
        rendezvous.wait(timeout=5)
        return owned_check.execute(argv, **kwargs)

    monkeypatch.setattr(k8s_gpu_catalog.subprocess, "run", execute)
    roots = [owned_check.root / "first-state", owned_check.root / "second-state"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda root: workflow._inspect_workflow_gpus(
                    "", "", "selected-context", str(owned_check.sky), root
                ),
                roots,
            )
        )
    assert all(not result[2] and not result[1].is_empty for result in results)
    assert len(set(owned_check.stops)) == 2
    endpoints = {
        kwargs["env"]["SKYPILOT_API_SERVER_ENDPOINT"] for _, kwargs in owned_check.calls
    }
    assert len(endpoints) == 2
    assert dict(os.environ) == before


def test_durable_pending_project_binding_cannot_be_overridden_by_environment(
    owned_check, monkeypatch
):
    monkeypatch.setenv("NPA_SKYPILOT_PROJECT", "original-alias")
    with pytest.raises(local_api.IsolatedApiError, match="removal remains unverified"):
        with cluster_validation.cluster_validation_session(
            owned_check.kubeconfig, "selected-context"
        ) as session:
            session.begin_smoke("validation-sky-smoke")
    before = (session.scope / "session.json").read_bytes()
    monkeypatch.setenv("NPA_SKYPILOT_PROJECT", "changed-alias")
    with pytest.raises(local_api.IsolatedApiError, match="different selected project"):
        with cluster_validation.cluster_validation_session(
            owned_check.kubeconfig, "selected-context"
        ):
            pytest.fail("pending project mismatch was accepted")
    assert (session.scope / "session.json").read_bytes() == before
    assert not owned_check.starts and not owned_check.stops


def test_legacy_pending_session_without_project_field_retains_its_recovery_path(
    owned_check, monkeypatch
):
    monkeypatch.setenv("NPA_SKYPILOT_PROJECT", "original-alias")
    with pytest.raises(local_api.IsolatedApiError, match="removal remains unverified"):
        with cluster_validation.cluster_validation_session(
            owned_check.kubeconfig, "selected-context"
        ) as session:
            session.begin_smoke("validation-sky-smoke")
    record = json.loads((session.scope / "session.json").read_text())
    del record["project_alias"]
    (session.scope / "session.json").write_text(json.dumps(record))
    with cluster_validation.cluster_validation_session(
        owned_check.kubeconfig, "selected-context"
    ) as recovered:
        assert recovered.scope == session.scope
        recovered.smoke_removed()
    assert (
        json.loads((session.scope / "session.json").read_text())["phase"] == "complete"
    )
