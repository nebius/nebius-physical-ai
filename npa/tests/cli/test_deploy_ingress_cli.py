from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.credentials import CredentialsConfig
from npa.clients.network import EnsureIngressResult, NetworkIngressError


runner = CliRunner()
TERRAFORM_PLAN_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "terraform_plans"


@dataclass(frozen=True)
class DeployIngressCase:
    tool: str
    module: str
    port: int
    command_port_args: tuple[str, ...]


TOOL_CASES = [
    DeployIngressCase(
        tool="cosmos",
        module="npa.cli.cosmos",
        port=8081,
        command_port_args=("--server-port", "8081"),
    ),
    DeployIngressCase(
        tool="groot",
        module="npa.cli.groot",
        port=8082,
        command_port_args=("--server-port", "8082"),
    ),
    DeployIngressCase(
        tool="fiftyone",
        module="npa.cli.fiftyone",
        port=5151,
        command_port_args=(),
    ),
]


def _ingress_result(case: DeployIngressCase) -> EnsureIngressResult:
    return EnsureIngressResult(
        instance_id="computeinstance-test",
        project_id="project-test",
        public_ip="203.0.113.10/32",
        ports=(case.port,),
        source="0.0.0.0/0",
        tool=case.tool,
        security_groups=(),
    )


def _patch_successful_deploy(mocker, case: DeployIngressCase, *, instance_id: str | None) -> None:
    module = case.module
    tf_outputs = {
        "vm_ip": "203.0.113.10",
        "ssh_user": "ubuntu",
        "ssh_key_path": "~/.ssh/id",
        "storage_bucket": "bucket",
        "storage_endpoint": "https://storage.example",
    }
    if instance_id is not None:
        tf_outputs["instance_id"] = instance_id

    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")
    ssh.run_or_raise.return_value = (0, "", "")

    mocker.patch(f"{module}.provisioner.init")
    mocker.patch(
        f"{module}.provisioner.plan",
        return_value=(TERRAFORM_PLAN_FIXTURES / "fresh_create.txt").read_text(),
    )
    mocker.patch(f"{module}.provisioner.apply", return_value=tf_outputs)
    mocker.patch(f"{module}.resolve_environment", return_value=None)
    mocker.patch(f"{module}.list_projects", return_value={})
    mocker.patch(f"{module}.write_config")
    mocker.patch(f"{module}.update_workbench_app_status")
    mocker.patch(f"{module}.write_manifest")
    mocker.patch(f"{module}.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.ingress.list_projects", return_value={})

    if case.tool == "cosmos":
        mocker.patch(f"{module}.resolve_credentials", return_value=SimpleNamespace(hf_token="", tokens={}))
        mocker.patch(f"{module}.health_check_auto", return_value=(True, ""))
    elif case.tool == "groot":
        mocker.patch(f"{module}.resolve_credentials", return_value=CredentialsConfig(tokens={}))
        mocker.patch(f"{module}.health_check_auto", return_value=(True, ""))
        mocker.patch(f"{module}.write_remote_docker_env_file")
    else:
        mocker.patch(f"{module}.resolve_credentials", return_value=SimpleNamespace(tokens={}))
        mocker.patch(f"{module}._run_fiftyone_command", return_value=(0, "", ""))
        mocker.patch(f"{module}._app_health_check", return_value=True)


def _deploy_args(case: DeployIngressCase, tmp_path: Path) -> list[str]:
    args = [
        "workbench",
        case.tool,
        "-p",
        "proj",
        "-n",
        "demo",
        "deploy",
        "--project-id",
        "project",
        "--tenant-id",
        "tenant",
        "--region",
        "eu-north1",
        "--tf-dir",
        str(tmp_path),
        "--no-verify-env",
        *case.command_port_args,
    ]
    if case.tool == "cosmos":
        args.extend([
            "--gpu-type",
            "gpu-h100-sxm",
            "--gpu-preset",
            "1gpu-16vcpu-200gb",
            "--skip-model-check",
        ])
    elif case.tool == "groot":
        args.append("--skip-model-check")
    return args


@pytest.mark.parametrize("case", TOOL_CASES, ids=lambda case: case.tool)
def test_deploy_success_ensures_ingress_for_tool_port(
    tmp_path: Path,
    mocker,
    case: DeployIngressCase,
) -> None:
    _patch_successful_deploy(mocker, case, instance_id="computeinstance-test")
    ensure = mocker.patch("npa.cli.ingress.ensure_ingress", return_value=_ingress_result(case))

    result = runner.invoke(app, _deploy_args(case, tmp_path))

    assert result.exit_code == 0
    assert "Deploy complete" in result.output
    assert f"Network ingress confirmed for port {case.port}" in result.output
    ensure.assert_called_once_with(
        vm_id="computeinstance-test",
        ports=(case.port,),
        source="0.0.0.0/0",
        tool=case.tool,
    )


@pytest.mark.parametrize("case", TOOL_CASES, ids=lambda case: case.tool)
def test_deploy_ingress_failure_warns_and_still_succeeds(
    tmp_path: Path,
    mocker,
    case: DeployIngressCase,
) -> None:
    _patch_successful_deploy(mocker, case, instance_id="computeinstance-test")
    mocker.patch(
        "npa.cli.ingress.ensure_ingress",
        side_effect=NetworkIngressError("permission denied"),
    )

    result = runner.invoke(app, _deploy_args(case, tmp_path))

    assert result.exit_code == 0
    assert "Deploy complete" in result.output
    assert f"Warning: could not ensure network ingress for port {case.port}: permission denied." in result.output
    assert f"npa workbench {case.tool} ensure-ingress -n demo" in result.output


@pytest.mark.parametrize("case", TOOL_CASES, ids=lambda case: case.tool)
def test_deploy_skips_ingress_when_instance_id_unavailable(
    tmp_path: Path,
    mocker,
    case: DeployIngressCase,
) -> None:
    _patch_successful_deploy(mocker, case, instance_id=None)
    ensure = mocker.patch("npa.cli.ingress.ensure_ingress")

    result = runner.invoke(app, _deploy_args(case, tmp_path))

    assert result.exit_code == 0
    assert "Deploy complete" in result.output
    assert f"Debug: skipping network ingress for port {case.port}: instance_id unavailable." in result.output
    ensure.assert_not_called()
