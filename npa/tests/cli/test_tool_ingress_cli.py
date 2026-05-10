from __future__ import annotations

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.network import EnsureIngressResult


runner = CliRunner()


TOOL_CASES = [
    ("cosmos", 8081),
    ("groot", 8082),
    ("fiftyone", 5151),
]


def _result(tool: str, port: int) -> EnsureIngressResult:
    return EnsureIngressResult(
        instance_id="computeinstance-test",
        project_id="project-test",
        public_ip="203.0.113.10/32",
        ports=(port,),
        source="0.0.0.0/0",
        tool=tool,
        security_groups=(),
    )


def _patch_projects(mocker, workbenches: dict):
    mocker.patch(
        "npa.cli.ingress.list_projects",
        return_value={"proj": {"workbenches": workbenches}},
    )
    mocker.patch("npa.cli.ingress.default_project_name", return_value="proj")
    mocker.patch("npa.cli.ingress.default_workbench_name", return_value="demo")


@pytest.mark.parametrize(("tool", "port"), TOOL_CASES)
def test_tool_ensure_ingress_calls_primitive_with_instance_and_port(mocker, tool: str, port: int) -> None:
    _patch_projects(mocker, {"demo": {"instance_id": "computeinstance-test"}})
    ensure = mocker.patch("npa.cli.ingress.ensure_ingress", return_value=_result(tool, port))

    result = runner.invoke(app, ["workbench", tool, "ensure-ingress", "-n", "demo"])

    assert result.exit_code == 0
    assert f"ingress already covered for port {port}" in result.output
    ensure.assert_called_once_with(
        vm_id="computeinstance-test",
        ports=(port,),
        source="0.0.0.0/0",
        tool=tool,
    )


@pytest.mark.parametrize(("tool", "port"), TOOL_CASES)
def test_tool_ensure_ingress_missing_alias_is_clean(mocker, tool: str, port: int) -> None:
    _patch_projects(mocker, {"other": {"instance_id": "computeinstance-test"}})
    ensure = mocker.patch("npa.cli.ingress.ensure_ingress")

    result = runner.invoke(app, ["workbench", tool, "ensure-ingress", "-n", "demo"])

    assert result.exit_code == 1
    assert "Workbench 'demo' not found" in result.output
    ensure.assert_not_called()


@pytest.mark.parametrize(("tool", "port"), TOOL_CASES)
def test_tool_ensure_ingress_missing_instance_id_has_remediation(mocker, tool: str, port: int) -> None:
    _patch_projects(mocker, {"demo": {"endpoint": "http://203.0.113.10"}})
    ensure = mocker.patch("npa.cli.ingress.ensure_ingress")

    result = runner.invoke(app, ["workbench", tool, "ensure-ingress", "-n", "demo"])

    assert result.exit_code == 1
    normalized = " ".join(result.output.split())
    assert (
        f"alias 'demo' has no instance_id; re-register with "
        f"'npa workbench {tool} register-byovm'"
    ) in normalized
    ensure.assert_not_called()
