"""Tests for the Intrinsic Core workbench toolRef (issue #816).

Honest surface: ``preflight`` (host + runtime probes), ``icon-status``
(read-only ICON status), and ``world-probe`` (read-only digital-twin
reachability) are real and tested. Mutating operations (``inctl world
reset``, ``asset install``, ``icon clear-faults``) are not implemented
and are not exposed.

All subprocess/network interaction is mocked; no live infrastructure.
"""

from __future__ import annotations

import importlib
import socket
import subprocess

import pytest
import typer
from typer.testing import CliRunner

import npa.workbench
from npa.cli.workbench.intrinsic import (
    app,
    preflight_cmd,
)
from npa.workbench import intrinsic as intrinsic_workbench
from npa.workbench.intrinsic import SUPPORTED_ROS_DISTRO

runner = CliRunner()


def _nested_app():
    """Minimal npa > workbench nesting mirroring the real CLI dispatch."""
    top = typer.Typer(name="npa", no_args_is_help=True)
    wb = typer.Typer(name="workbench", help="wb", no_args_is_help=True)
    top.add_typer(wb, name="workbench")
    wb.add_typer(app, name="intrinsic")
    return top


def _completed(argv, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


class _FakeEnv:
    """Canned healthy host: Ubuntu 24.04, lyrical, k3s, inctl, kubectl, GPU."""

    def __init__(self, monkeypatch):
        self.seen_argv: list[list[str]] = []
        monkeypatch.setattr(
            "npa.workbench.intrinsic._read_os_release",
            lambda: {"VERSION_ID": "24.04"},
        )
        monkeypatch.setenv("ROS_DISTRO", "lyrical")
        monkeypatch.setattr(
            "npa.workbench.intrinsic.shutil.which", lambda name: f"/usr/bin/{name}"
        )
        monkeypatch.setattr(
            "npa.workbench.intrinsic.socket.create_connection",
            lambda *a, **k: _DummySocket(),
        )

        def fake_run(argv, **kwargs):
            self.seen_argv.append(list(argv))
            if argv[:2] == ["systemctl", "is-active"]:
                return _completed(argv, stdout="active\n")
            if argv[0] == "kubectl":
                return _completed(
                    argv,
                    stdout='{"items": [{"metadata": {"name": "svc"}, '
                    '"status": {"phase": "Running"}}]}',
                )
            if argv[:3] == ["inctl", "service", "state"]:
                return _completed(argv, stdout='[{"name": "world", "state": "OK"}]')
            if argv[:3] == ["inctl", "icon", "status"]:
                return _completed(argv, stdout="state: OPERATIONAL\nfaults: none\n")
            return _completed(argv, stdout="")

        monkeypatch.setattr("npa.workbench.intrinsic.subprocess.run", fake_run)


class _DummySocket:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_cli_app_exposes_three_read_only_commands():
    assert isinstance(app, typer.Typer)
    command_names = {c.name for c in app.registered_commands}
    assert command_names == {"preflight", "icon-status", "world-probe"}


def test_app_help_is_honest():
    result = runner.invoke(_nested_app(), ["workbench", "intrinsic", "--help"])
    assert result.exit_code == 0, result.output
    assert "preflight" in result.output
    assert "not exposed" in result.output


def test_supported_distro_is_lyrical():
    assert SUPPORTED_ROS_DISTRO == "lyrical"


def test_workbench_surface_is_first_class():
    """The SDK surface exposes the real functions directly.

    New tools must not route through ``npa._sdk.make_cli_wrapper``: the
    workbench package is the primary surface and the CLI is a thin client.
    """
    module = importlib.import_module("npa.workbench.intrinsic")
    assert callable(module.preflight)
    assert callable(module.icon_status)
    assert callable(module.world_probe)
    assert module.TOOLREF == "workbench.intrinsic"
    assert module.SUPPORTED_ROS_DISTRO == "lyrical"
    assert not hasattr(module.preflight, "__npa_cli_module__")
    assert set(module.__all__) == {
        "TOOLREF",
        "SUPPORTED_ROS_DISTRO",
        "ADDRESS_ENV",
        "DEFAULT_ADDRESS",
        "resolve_address",
        "preflight",
        "icon_status",
        "world_probe",
    }


def test_workbench_lazy_namespace():
    module = npa.workbench.intrinsic
    assert callable(module.preflight)
    assert not hasattr(module, "world_reset")


def test_resolve_address_prefers_explicit_then_env(monkeypatch):
    assert intrinsic_workbench.resolve_address("h:1") == "h:1"
    monkeypatch.setenv("INTRINSIC_ADDRESS", "env-host:17080")
    assert intrinsic_workbench.resolve_address() == "env-host:17080"
    monkeypatch.delenv("INTRINSIC_ADDRESS")
    assert intrinsic_workbench.resolve_address() == intrinsic_workbench.DEFAULT_ADDRESS


def test_preflight_ok_when_healthy(monkeypatch):
    _FakeEnv(monkeypatch)
    payload = intrinsic_workbench.preflight()
    assert payload["ok"] is True
    assert payload["supported_distro"] == "lyrical"
    names = {c["name"] for c in payload["checks"]}
    assert {
        "ubuntu",
        "ros-distro",
        "k3s",
        "inctl",
        "gpu",
        "runtime-pods",
        "ingress",
        "service-state",
    } <= names


def test_preflight_warns_on_2204_without_gpu(monkeypatch):
    env = _FakeEnv(monkeypatch)
    monkeypatch.setattr(
        "npa.workbench.intrinsic._read_os_release",
        lambda: {"VERSION_ID": "22.04"},
    )
    monkeypatch.setattr(
        "npa.workbench.intrinsic.shutil.which",
        lambda name: None if name == "nvidia-smi" else f"/usr/bin/{name}",
    )
    payload = intrinsic_workbench.preflight()
    assert payload["ok"] is True
    assert len(payload["warnings"]) == 2
    assert any("22.04" in w for w in payload["warnings"])
    assert any("nvidia-smi" in w for w in payload["warnings"])
    assert env.seen_argv  # runtime probes still ran


def test_preflight_fails_fast_without_inctl(monkeypatch):
    _FakeEnv(monkeypatch)
    monkeypatch.setattr("npa.workbench.intrinsic.shutil.which", lambda name: None)
    payload = intrinsic_workbench.preflight()
    assert payload["ok"] is False
    assert "inctl" in payload["detail"]


def test_preflight_detects_wrong_distro(monkeypatch):
    _FakeEnv(monkeypatch)
    monkeypatch.setenv("ROS_DISTRO", "humble")
    payload = intrinsic_workbench.preflight()
    assert payload["ok"] is False
    assert "humble" in payload["detail"]
    assert "lyrical" in payload["detail"]


def test_preflight_fails_when_distro_unset(monkeypatch):
    _FakeEnv(monkeypatch)
    monkeypatch.delenv("ROS_DISTRO", raising=False)
    payload = intrinsic_workbench.preflight()
    assert payload["ok"] is False


def test_preflight_detects_errored_service(monkeypatch):
    env = _FakeEnv(monkeypatch)

    def fake_run2(argv, **kwargs):
        env.seen_argv.append(list(argv))
        if argv[:2] == ["systemctl", "is-active"]:
            return _completed(argv, stdout="active\n")
        if argv[:3] == ["inctl", "service", "state"]:
            return _completed(argv, stdout='[{"state": "STATE_CODE_ERROR"}]')
        if argv[0] == "kubectl":
            return _completed(argv, stdout='{"items": []}')
        return _completed(argv, stdout="")

    monkeypatch.setattr("npa.workbench.intrinsic.subprocess.run", fake_run2)
    payload = intrinsic_workbench.preflight()
    assert payload["ok"] is False
    assert "service-state" in payload["detail"]


def test_preflight_skips_kubectl_gracefully(monkeypatch):
    _FakeEnv(monkeypatch)
    monkeypatch.setattr(
        "npa.workbench.intrinsic.shutil.which",
        lambda name: None if name == "kubectl" else f"/usr/bin/{name}",
    )
    payload = intrinsic_workbench.preflight()
    assert payload["ok"] is True
    pods = next(c for c in payload["checks"] if c["name"] == "runtime-pods")
    assert pods["skipped"] is True


def test_preflight_ingress_failure(monkeypatch):
    _FakeEnv(monkeypatch)

    def boom(*a, **k):
        raise OSError("refused")

    monkeypatch.setattr("npa.workbench.intrinsic.socket.create_connection", boom)
    payload = intrinsic_workbench.preflight()
    assert payload["ok"] is False
    assert "ingress" in payload["detail"]


def test_preflight_cli_ok(monkeypatch):
    _FakeEnv(monkeypatch)
    result = runner.invoke(app, ["preflight"])
    assert result.exit_code == 0, result.output
    assert "preflight OK" in result.output


def test_preflight_cli_reports_failure(monkeypatch):
    _FakeEnv(monkeypatch)
    monkeypatch.setattr("npa.workbench.intrinsic.shutil.which", lambda name: None)
    result = runner.invoke(app, ["preflight"])
    assert result.exit_code == 3
    assert "preflight FAILED" in result.output
    assert "intrinsic-core" in result.output


def test_preflight_cmd_raises_exit_3(monkeypatch):
    monkeypatch.setattr(
        intrinsic_workbench,
        "preflight",
        lambda address=None: {
            "ok": False,
            "detail": "k3s: nope",
            "checks": [],
            "warnings": [],
        },
    )
    with pytest.raises(typer.Exit) as exc_info:
        preflight_cmd()
    assert exc_info.value.exit_code == 3


def test_icon_status_ok(monkeypatch):
    _FakeEnv(monkeypatch)
    payload = intrinsic_workbench.icon_status()
    assert payload["ok"] is True
    assert "OPERATIONAL" in payload["detail"]
    assert payload["instance"] == "icon"


def test_icon_status_no_inctl(monkeypatch):
    monkeypatch.setattr("npa.workbench.intrinsic.shutil.which", lambda name: None)
    payload = intrinsic_workbench.icon_status()
    assert payload["ok"] is False


def test_icon_status_cli_failure(monkeypatch):
    monkeypatch.setattr(
        intrinsic_workbench,
        "icon_status",
        lambda address=None, instance_name="icon": {
            "ok": False,
            "detail": "boom",
            "instance": instance_name,
            "address": "x",
            "excerpt": "",
        },
    )
    result = runner.invoke(app, ["icon-status"])
    assert result.exit_code == 3
    assert "ICON status FAILED" in result.output


def test_world_probe_ok_and_read_only(monkeypatch):
    env = _FakeEnv(monkeypatch)
    payload = intrinsic_workbench.world_probe()
    assert payload["ok"] is True
    assert payload["world_service_found"] is True
    for argv in env.seen_argv:
        joined = " ".join(argv).lower()
        assert "reset" not in joined
        assert "install" not in joined
        assert "delete" not in joined


def test_world_probe_no_world_service(monkeypatch):
    env = _FakeEnv(monkeypatch)

    def fake_run(argv, **kwargs):
        env.seen_argv.append(list(argv))
        if argv[:3] == ["inctl", "service", "state"]:
            return _completed(argv, stdout='[{"name": "robot", "state": "OK"}]')
        return _completed(argv, stdout="")

    monkeypatch.setattr("npa.workbench.intrinsic.subprocess.run", fake_run)
    payload = intrinsic_workbench.world_probe()
    assert payload["ok"] is False
    assert payload["world_service_found"] is False


def test_world_probe_ingress_down(monkeypatch):
    _FakeEnv(monkeypatch)

    def boom(*a, **k):
        raise socket.error("refused")

    monkeypatch.setattr("npa.workbench.intrinsic.socket.create_connection", boom)
    payload = intrinsic_workbench.world_probe()
    assert payload["ok"] is False


def test_toolref_argv_is_exact():
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    assert TOOL_CATALOG["workbench.intrinsic.preflight"].argv_template == [
        "npa",
        "workbench",
        "intrinsic",
        "preflight",
    ]
    assert TOOL_CATALOG["workbench.intrinsic.icon_status"].argv_template == [
        "npa",
        "workbench",
        "intrinsic",
        "icon-status",
    ]
    assert TOOL_CATALOG["workbench.intrinsic.world_probe"].argv_template == [
        "npa",
        "workbench",
        "intrinsic",
        "world-probe",
    ]
