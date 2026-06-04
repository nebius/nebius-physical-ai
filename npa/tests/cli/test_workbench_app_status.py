from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients import config, credentials
from npa.clients.credentials import CredentialsConfig
from npa.clients.ssh import SSHError


runner = CliRunner()
TERRAFORM_PLAN_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "terraform_plans"


@pytest.fixture()
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg_path = tmp_path / ".npa" / "config.yaml"
    credentials_path = tmp_path / ".npa" / "credentials.yaml"
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", credentials_path)
    return cfg_path


def _write_failed_cosmos_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "default_project": "proj",
                "default_workbench": "cosmos",
                "projects": {
                    "proj": {
                        "project_id": "project",
                        "tenant_id": "tenant",
                        "region": "eu-north1",
                        "workbenches": {
                            "cosmos": {
                                "endpoint": "http://10.0.0.7:8080",
                                "gpu_platform": "gpu-h100-sxm",
                                "gpu_preset": "1gpu-16vcpu-200gb",
                                "tf_instance_name": "cosmos-proj-cosmos",
                                "workbench_type": "cosmos",
                                "app_status": "install_failed",
                                "model": "nvidia/Cosmos-Test",
                                "backend": "basic",
                                "ssh": {
                                    "host": "10.0.0.7",
                                    "user": "ubuntu",
                                    "key_path": "~/.ssh/id",
                                },
                                "storage": {
                                    "checkpoint_bucket": "s3://bucket/checkpoints/",
                                    "endpoint_url": "https://storage.example",
                                },
                            },
                        },
                    },
                },
            },
            sort_keys=False,
        )
    )


def test_cosmos_registers_after_terraform_before_app_install_and_marks_failure(
    tmp_path: Path,
    mocker,
) -> None:
    events: list[tuple[str, str]] = []
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")
    ssh.run_or_raise.side_effect = SSHError("install boom")

    mocker.patch(
        "npa.cli.workbench.load_credentials",
        return_value=CredentialsConfig(tokens={"HF_TOKEN": "hf-test"}),
    )
    mocker.patch("npa.cli.cosmos.provisioner.init")
    mocker.patch(
        "npa.cli.cosmos.provisioner.plan",
        return_value=(TERRAFORM_PLAN_FIXTURES / "fresh_create.txt").read_text(),
    )
    mocker.patch(
        "npa.cli.cosmos.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.7",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.cosmos.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.cosmos.resolve_environment", return_value=None)
    mocker.patch(
        "npa.cli.cosmos.resolve_credentials",
        return_value=CredentialsConfig(tokens={"HF_TOKEN": "hf-test"}),
    )
    validate_hf_access = mocker.patch(
        "npa.cli.cosmos.validate_hf_access",
        return_value=mocker.MagicMock(ok=True, error=""),
    )
    mocker.patch("npa.cli.cosmos.list_projects", return_value={})
    mocker.patch(
        "npa.cli.cosmos.write_config",
        side_effect=lambda data: events.append(
            ("write", data["projects"]["proj"]["workbenches"]["cosmos"]["app_status"])
        ),
    )
    mocker.patch(
        "npa.cli.cosmos.update_workbench_app_status",
        side_effect=lambda _project, _name, status: events.append(("status", status)),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "-p",
            "proj",
            "-n",
            "cosmos",
            "deploy",
            "--project-id",
            "project",
            "--tenant-id",
            "tenant",
            "--region",
            "eu-north1",
            "--tf-dir",
            str(tmp_path),
            "--gpu-type",
            "gpu-h100-sxm",
            "--gpu-preset",
            "1gpu-16vcpu-200gb",
        ],
    )

    assert result.exit_code == 1
    assert "Cosmos installation failed: install boom" in result.output
    validate_hf_access.assert_called_once()
    assert events == [
        ("write", "provisioned"),
        ("status", "installing"),
        ("status", "install_failed"),
    ]


def test_cosmos_skip_infra_retries_install_failed_workbench(
    isolated_config: Path,
    tmp_path: Path,
    mocker,
) -> None:
    _write_failed_cosmos_config(isolated_config)
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")
    ssh.run_or_raise.return_value = (0, "COSMOS_ENV_SMOKE_OK", "")

    mocker.patch(
        "npa.cli.workbench.load_credentials",
        return_value=CredentialsConfig(tokens={"HF_TOKEN": "hf-test"}),
    )
    mocker.patch(
        "npa.cli.cosmos.resolve_credentials",
        return_value=CredentialsConfig(tokens={"HF_TOKEN": "hf-test"}),
    )
    mocker.patch("npa.cli.cosmos.SSHClient", return_value=ssh)
    mocker.patch(
        "npa.cli.cosmos.validate_hf_access",
        return_value=mocker.MagicMock(ok=True, error=""),
    )
    mocker.patch("npa.cli.cosmos.health_check_auto", return_value=(True, ""))
    mocker.patch("npa.cli.cosmos.write_manifest")
    mocker.patch("npa.cli.cosmos.provisioner.working_dir_path", return_value=tmp_path / "missing")
    apply = mocker.patch("npa.cli.cosmos.provisioner.apply")

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "-p",
            "proj",
            "-n",
            "cosmos",
            "deploy",
            "--skip-infra",
            "--gpu-type",
            "gpu-h100-sxm",
            "--gpu-preset",
            "1gpu-16vcpu-200gb",
            "--model",
            "nvidia/Cosmos-Test",
        ],
    )

    assert result.exit_code == 0
    apply.assert_not_called()
    assert ssh.run_or_raise.called
    data = yaml.safe_load(isolated_config.read_text())
    wb_cfg = data["projects"]["proj"]["workbenches"]["cosmos"]
    assert wb_cfg["app_status"] == "healthy"


def test_cosmos_destroy_removes_install_failed_workbench(
    isolated_config: Path,
    tmp_path: Path,
    mocker,
) -> None:
    _write_failed_cosmos_config(isolated_config)
    destroy = mocker.patch("npa.cli.cosmos.provisioner.destroy")
    mocker.patch(
        "npa.cli.workbench.load_credentials",
        return_value=CredentialsConfig(tokens={"HF_TOKEN": "hf-test"}),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "-p",
            "proj",
            "-n",
            "cosmos",
            "deploy",
            "--destroy",
            "--tf-dir",
            str(tmp_path),
            "--gpu-type",
            "gpu-h100-sxm",
            "--gpu-preset",
            "1gpu-16vcpu-200gb",
        ],
    )

    assert result.exit_code == 0
    destroy.assert_called_once()
    data = yaml.safe_load(isolated_config.read_text())
    assert "proj" not in data.get("projects", {})
