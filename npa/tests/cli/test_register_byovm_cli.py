from __future__ import annotations

from dataclasses import dataclass

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.network import EnsureIngressResult, InstanceNetworkContext, NetworkIngressError


runner = CliRunner()


@dataclass(frozen=True)
class RegisterCase:
    tool: str
    port: int


TOOL_CASES = [
    RegisterCase("cosmos", 8081),
    RegisterCase("groot", 8082),
    RegisterCase("fiftyone", 5151),
]


def _context() -> InstanceNetworkContext:
    return InstanceNetworkContext(
        instance_id="computeinstance-test",
        project_id="project-test",
        public_ip="203.0.113.10/32",
        security_group_ids=("vpcsecuritygroup-first", "vpcsecuritygroup-second"),
    )


def _result(case: RegisterCase) -> EnsureIngressResult:
    return EnsureIngressResult(
        instance_id="computeinstance-test",
        project_id="project-test",
        public_ip="203.0.113.10/32",
        ports=(case.port,),
        source="0.0.0.0/0",
        tool=case.tool,
        security_groups=(),
    )


def _patch_register_context(mocker, *, workbenches: dict | None = None):
    mocker.patch("npa.cli.ingress.resolve_instance_network_context", return_value=_context())
    mocker.patch(
        "npa.cli.ingress.list_projects",
        return_value={"proj": {"workbenches": workbenches or {}}},
    )
    mocker.patch("npa.cli.ingress.default_project_name", return_value="proj")
    return mocker.patch("npa.cli.ingress.write_config")


@pytest.mark.parametrize("case", TOOL_CASES, ids=lambda case: case.tool)
def test_register_byovm_writes_alias_and_ensures_ingress(mocker, case: RegisterCase) -> None:
    write_config = _patch_register_context(mocker)
    ensure = mocker.patch("npa.cli.ingress.ensure_ingress", return_value=_result(case))

    result = runner.invoke(
        app,
        [
            "workbench",
            case.tool,
            "register-byovm",
            "--alias",
            "demo",
            "--instance-id",
            "computeinstance-test",
        ],
    )

    assert result.exit_code == 0
    assert f"Registered {case.tool} BYOVM alias 'demo' in project 'proj'." in result.output
    assert f"Network ingress confirmed for port {case.port}" in result.output
    ensure.assert_called_once_with(
        vm_id="computeinstance-test",
        ports=(case.port,),
        source="0.0.0.0/0",
        tool=case.tool,
    )
    alias_config = write_config.call_args.args[0]["projects"]["proj"]["workbenches"]["demo"]
    assert alias_config["alias"] == "demo"
    assert alias_config["endpoint"] == f"http://203.0.113.10:{case.port}"
    assert alias_config["runtime"] == "byovm"
    assert alias_config["workbench_type"] == case.tool
    assert alias_config["service_port"] == case.port
    assert alias_config["instance_id"] == "computeinstance-test"
    assert alias_config["project_id"] == "project-test"
    assert alias_config["security_group_id"] == "vpcsecuritygroup-first"
    assert alias_config["ssh"]["host"] == "203.0.113.10"
    if case.tool == "fiftyone":
        assert alias_config["app_port"] == 5151


def test_register_byovm_instance_get_failure_does_not_write_alias(mocker) -> None:
    mocker.patch(
        "npa.cli.ingress.resolve_instance_network_context",
        side_effect=NetworkIngressError("Could not fetch VM computeinstance-missing"),
    )
    write_config = mocker.patch("npa.cli.ingress.write_config")
    ensure = mocker.patch("npa.cli.ingress.ensure_ingress")

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "register-byovm",
            "--alias",
            "demo",
            "--instance-id",
            "computeinstance-missing",
        ],
    )

    assert result.exit_code == 1
    assert "Could not fetch VM computeinstance-missing" in result.output
    write_config.assert_not_called()
    ensure.assert_not_called()


def test_register_byovm_existing_alias_overwrites_registration_fields_with_warning(mocker) -> None:
    existing = {
        "endpoint": "http://old.example:9999",
        "runtime": "vm",
        "ssh": {
            "host": "old.example",
            "user": "robot",
            "key_path": "~/.ssh/robot",
        },
        "storage": {
            "checkpoint_bucket": "s3://bucket/checkpoints/",
            "endpoint_url": "https://storage.example",
        },
    }
    write_config = _patch_register_context(mocker, workbenches={"demo": existing})
    mocker.patch("npa.cli.ingress.ensure_ingress", return_value=_result(RegisterCase("groot", 8082)))

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "register-byovm",
            "--alias",
            "demo",
            "--instance-id",
            "computeinstance-test",
        ],
    )

    assert result.exit_code == 0
    assert "Warning: alias 'demo' already exists; overwriting BYOVM registration fields." in result.output
    alias_config = write_config.call_args.args[0]["projects"]["proj"]["workbenches"]["demo"]
    assert alias_config["endpoint"] == "http://203.0.113.10:8082"
    assert alias_config["runtime"] == "byovm"
    assert alias_config["project_id"] == "project-test"
    assert alias_config["security_group_id"] == "vpcsecuritygroup-first"
    assert alias_config["ssh"] == {
        "host": "203.0.113.10",
        "user": "robot",
        "key_path": "~/.ssh/robot",
    }
    assert alias_config["storage"] == existing["storage"]
