from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from click.utils import strip_ansi
import httpx
import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.fiftyone import (
    DEFAULT_CPU_IMAGE_FAMILY,
    DEFAULT_CPU_PLATFORM,
    DEFAULT_CPU_PRESET,
    FIFTYONE_AUTO_PUBLIC_HEALTH_RETRIES,
    FIFTYONE_HEALTH_BACKOFF_SEC,
    FIFTYONE_VERSION,
)
from npa.cli.main import app
from npa.clients import config as config_module
from npa.clients import credentials as credentials_module
from npa.clients.config import (
    EnvironmentConfig,
    SSHConfig,
    StorageConfig,
    TerraformStateConfig,
    WorkbenchConfig,
)
from npa.clients.ssh import SSHError


runner = CliRunner()


def _cfg(app_status: str = "") -> WorkbenchConfig:
    return WorkbenchConfig(
        endpoint="http://fiftyone.example:5151",
        ssh=SSHConfig(host="10.0.0.10", user="ubuntu", key_path="~/.ssh/id"),
        storage=StorageConfig(checkpoint_bucket="", endpoint_url=""),
        app_status=app_status,
    )


@contextmanager
def _active_endpoint(url: str):
    yield SimpleNamespace(url=url)


@pytest.mark.parametrize(
    "command",
    [
        "deploy",
        "launch",
        "load-dataset",
        "restart",
        "status",
        "system-info",
        "datasets",
        "list",
    ],
)
def test_fiftyone_command_help(command: str) -> None:
    result = runner.invoke(app, ["workbench", "fiftyone", command, "--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_fiftyone_registered_under_workbench() -> None:
    result = runner.invoke(app, ["workbench", "--help"])

    assert result.exit_code == 0
    assert "fiftyone" in result.output


def test_fiftyone_load_dataset_help_includes_format_flag() -> None:
    result = runner.invoke(app, ["workbench", "fiftyone", "load-dataset", "--help"])
    output = strip_ansi(result.output)

    assert result.exit_code == 0
    assert "--format" in output
    assert "lerobot" in output


def test_fiftyone_deploy_defaults_to_cpu_without_gpu_flags(tmp_path: Path, mocker) -> None:
    init = mocker.patch("npa.cli.fiftyone.provisioner.init")
    apply = mocker.patch(
        "npa.cli.fiftyone.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.20",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.fiftyone.resolve_environment", return_value=None)
    mocker.patch("npa.cli.fiftyone.list_projects", return_value={})
    write_config = mocker.patch("npa.cli.fiftyone.write_config")

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            "proj",
            "-n",
            "curate",
            "deploy",
            "--project-id",
            "project",
            "--tenant-id",
            "tenant",
            "--region",
            "eu-north1",
            "--tf-dir",
            str(tmp_path),
            "--skip-app",
        ],
    )

    assert result.exit_code == 0
    assert "Deploy complete" in result.output
    init.assert_called_once_with(tf_dir=str(tmp_path), backend_config=None)
    apply.assert_called_once()
    tf_vars = apply.call_args.kwargs["tf_vars"]
    assert tf_vars["gpu_platform"] == DEFAULT_CPU_PLATFORM
    assert tf_vars["gpu_preset"] == DEFAULT_CPU_PRESET
    assert tf_vars["image_family"] == DEFAULT_CPU_IMAGE_FAMILY
    assert tf_vars["server_port"] == "5151"
    assert tf_vars["enable_preemptible"] == "false"
    assert tf_vars["workbench_type"] == "fiftyone"
    assert tf_vars["fiftyone_version"] == FIFTYONE_VERSION
    assert tf_vars["instance_name"] == "fiftyone-proj-curate"
    assert "boot_disk_size_gb" not in tf_vars
    config_data = write_config.call_args.args[0]
    wb_cfg = config_data["projects"]["proj"]["workbenches"]["curate"]
    assert wb_cfg["endpoint"] == "http://10.0.0.20:5151"
    assert wb_cfg["workbench_type"] == "fiftyone"
    assert wb_cfg["app_status"] == "provisioned"


def test_fiftyone_deploy_accepts_cpu_override_flags(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.fiftyone.provisioner.init")
    apply = mocker.patch(
        "npa.cli.fiftyone.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.25",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.fiftyone.resolve_environment", return_value=None)
    mocker.patch("npa.cli.fiftyone.list_projects", return_value={})
    mocker.patch("npa.cli.fiftyone.write_config")

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            "proj",
            "-n",
            "curate-cpu",
            "deploy",
            "--project-id",
            "project",
            "--tenant-id",
            "tenant",
            "--region",
            "eu-north1",
            "--tf-dir",
            str(tmp_path),
            "--skip-app",
            "--cpu-type",
            "cpu-e2",
            "--cpu-preset",
            "8vcpu-32gb",
        ],
    )

    assert result.exit_code == 0
    tf_vars = apply.call_args.kwargs["tf_vars"]
    assert tf_vars["gpu_platform"] == "cpu-e2"
    assert tf_vars["gpu_preset"] == "8vcpu-32gb"
    assert tf_vars["image_family"] == DEFAULT_CPU_IMAGE_FAMILY
    assert tf_vars["enable_preemptible"] == "false"


def test_fiftyone_deploy_accepts_gpu_flags_and_installs_app(tmp_path: Path, mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.side_effect = [
        (0, "connected", ""),
        (0, "FIFTYONE_ENV_SMOKE_OK\nNPA_FIFTYONE_APP_READY", ""),
    ]

    mocker.patch("npa.cli.fiftyone.provisioner.init")
    apply = mocker.patch(
        "npa.cli.fiftyone.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.21",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.fiftyone.resolve_environment", return_value=None)
    mocker.patch("npa.cli.fiftyone.list_projects", return_value={})
    mocker.patch("npa.cli.fiftyone.write_config")
    update_status = mocker.patch("npa.cli.fiftyone.update_workbench_app_status")
    mocker.patch("npa.cli.fiftyone.write_manifest")
    health = mocker.patch("npa.cli.fiftyone._app_health_check", return_value=True)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            "proj",
            "-n",
            "curate-gpu",
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
            "gpu-l40s-a",
            "--gpu-preset",
            "1gpu-40vcpu-160gb",
        ],
    )

    assert result.exit_code == 0
    tf_vars = apply.call_args.kwargs["tf_vars"]
    assert tf_vars["gpu_platform"] == "gpu-l40s-a"
    assert tf_vars["gpu_preset"] == "1gpu-40vcpu-160gb"
    assert "image_family" not in tf_vars
    assert tf_vars["enable_preemptible"] == "true"
    assert tf_vars["workbench_type"] == "fiftyone"
    assert tf_vars["fiftyone_version"] == FIFTYONE_VERSION
    install_cmd = ssh.run.call_args_list[1].args[0]
    assert "python3 -m venv /opt/fiftyone/venv" in install_cmd
    assert f'/opt/fiftyone/venv/bin/python -m pip install "fiftyone=={FIFTYONE_VERSION}"' in install_cmd
    assert "pyarrow pillow" in install_cmd
    assert "FIFTYONE_DEFAULT_APP_ADDRESS=0.0.0.0" in install_cmd
    assert "FIFTYONE_DEFAULT_APP_PORT=5151" in install_cmd
    assert 'sudo chown "$USER:$USER" /etc/npa-fiftyone/env' in install_cmd
    assert "lerobot[pusht" not in install_cmd
    assert "Installing LeRobot" not in install_cmd
    assert "TimeoutStopSec=15" in install_cmd
    assert "seq 1 120" in install_cmd
    assert update_status.call_args_list[0].args == ("proj", "curate-gpu", "installing")
    assert update_status.call_args_list[1].args == ("proj", "curate-gpu", "provisioned")
    assert update_status.call_args_list[-1].args == ("proj", "curate-gpu", "healthy")
    health.assert_called_once_with("http://10.0.0.21:5151")


def test_fiftyone_deploy_runtime_container_starts_image(tmp_path: Path, mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")

    mocker.patch("npa.cli.fiftyone.provisioner.init")
    apply = mocker.patch(
        "npa.cli.fiftyone.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.27",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.fiftyone.resolve_environment", return_value=None)
    mocker.patch("npa.cli.fiftyone.list_projects", return_value={})
    write_config = mocker.patch("npa.cli.fiftyone.write_config")
    update_status = mocker.patch("npa.cli.fiftyone.update_workbench_app_status")
    mocker.patch("npa.cli.fiftyone.write_manifest")
    mocker.patch("npa.cli.fiftyone._app_health_check", return_value=True)
    deploy_container = mocker.patch("npa.deploy.configurator.deploy_workbench_container")
    mocker.patch("npa.deploy.configurator.write_remote_docker_env_file")
    mocker.patch("npa.deploy.configurator.write_remote_text_file")

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            "proj",
            "-n",
            "curate-container",
            "deploy",
            "--project-id",
            "project",
            "--tenant-id",
            "tenant",
            "--region",
            "eu-north1",
            "--tf-dir",
            str(tmp_path),
            "--runtime",
            "container",
        ],
    )

    assert result.exit_code == 0
    tf_vars = apply.call_args.kwargs["tf_vars"]
    assert tf_vars["workbench_type"] == "lerobot-container"
    assert tf_vars["boot_disk_size_gb"] == "250"
    assert tf_vars["image_family"] == DEFAULT_CPU_IMAGE_FAMILY
    deploy_container.assert_called_once()
    assert deploy_container.call_args.kwargs["container_name"] == "npa-fiftyone"
    assert deploy_container.call_args.kwargs["image_ref"].endswith("/npa-fiftyone:1.15.0")
    assert deploy_container.call_args.kwargs["gpu"] is False
    wb_cfg = write_config.call_args_list[0].args[0]["projects"]["proj"]["workbenches"]["curate-container"]
    assert wb_cfg["runtime"] == "container"
    assert update_status.call_args_list[0].args == ("proj", "curate-container", "installing")
    assert update_status.call_args_list[-1].args == ("proj", "curate-container", "healthy")


def test_fiftyone_byovm_auto_health_uses_short_public_retry_budget(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")
    ssh.run_or_raise.side_effect = [
        (0, "connected", ""),
        (0, "NVIDIA H200\n", ""),
    ]
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.fiftyone.resolve_environment", return_value=None)
    mocker.patch("npa.cli.fiftyone.resolve_credentials", return_value=SimpleNamespace(tokens={}))
    mocker.patch("npa.cli.fiftyone.list_projects", return_value={})
    mocker.patch("npa.cli.fiftyone.write_config")
    mocker.patch("npa.cli.fiftyone.update_workbench_app_status")
    mocker.patch("npa.deploy.configurator.deploy_workbench_container")
    mocker.patch("npa.deploy.configurator.write_remote_docker_env_file")
    mocker.patch("npa.deploy.configurator.write_remote_text_file")
    mocker.patch("npa.cli.fiftyone.write_manifest")
    public_health = mocker.patch("npa.cli.fiftyone._app_health_check", return_value=False)
    ssh_health = mocker.patch("npa.cli.fiftyone.health_check_ssh", return_value=True)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            "proj",
            "-n",
            "curate-byovm",
            "deploy",
            "--runtime",
            "byovm",
            "--host",
            "203.0.113.20",
            "--ssh-key",
            "~/.ssh/byovm",
            "--region",
            "eu-north1",
            "--tf-var",
            "s3_bucket=lerobot-bucket",
            "--tf-var",
            "s3_endpoint=https://storage.example",
        ],
    )

    assert result.exit_code == 0
    public_health.assert_called_once_with(
        "http://203.0.113.20:5151",
        retries=FIFTYONE_AUTO_PUBLIC_HEALTH_RETRIES,
        backoff=FIFTYONE_HEALTH_BACKOFF_SEC,
    )
    ssh_health.assert_called_once()


def test_fiftyone_byovm_skip_infra_reuses_saved_config_and_preserves_status(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")
    ssh.run_or_raise.side_effect = [
        (0, "connected", ""),
        (0, "NVIDIA H200\n", ""),
    ]
    saved_cfg = _cfg(app_status="healthy")
    saved_cfg.runtime = "byovm"
    saved_cfg.endpoint_strategy = "ssh"
    saved_cfg.service_port = 5151
    saved_cfg.storage = StorageConfig(
        checkpoint_bucket="s3://saved-bucket/checkpoints/",
        endpoint_url="https://saved-storage.example",
    )

    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=saved_cfg)
    resolve_byovm = mocker.patch("npa.cli.fiftyone.resolve_byovm_target")
    mocker.patch("npa.cli.fiftyone.resolve_environment", return_value=None)
    mocker.patch("npa.cli.fiftyone.resolve_credentials", return_value=SimpleNamespace(tokens={}))
    mocker.patch("npa.cli.fiftyone.list_projects", return_value={"proj": {}})
    mocker.patch(
        "npa.clients.config.resolve_project_storage",
        return_value=StorageConfig(
            checkpoint_bucket="",
            endpoint_url="",
            aws_access_key_id="",
            aws_secret_access_key="",
        ),
    )
    write_config = mocker.patch("npa.cli.fiftyone.write_config")
    mocker.patch("npa.cli.fiftyone.update_workbench_app_status")
    mocker.patch("npa.deploy.configurator.deploy_workbench_container")
    mocker.patch("npa.deploy.configurator.write_remote_docker_env_file")
    mocker.patch("npa.deploy.configurator.write_remote_text_file")
    mocker.patch("npa.cli.fiftyone.write_manifest")
    mocker.patch("npa.cli.fiftyone._app_health_check", return_value=False)
    mocker.patch("npa.cli.fiftyone.health_check_ssh", return_value=True)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            "proj",
            "-n",
            "curate-byovm",
            "deploy",
            "--runtime",
            "byovm",
            "--skip-infra",
            "--region",
            "eu-north1",
        ],
    )

    assert result.exit_code == 0
    resolve_byovm.assert_not_called()
    wb_cfg = write_config.call_args_list[0].args[0]["projects"]["proj"]["workbenches"]["curate-byovm"]
    assert wb_cfg["ssh"] == {
        "host": "10.0.0.10",
        "user": "ubuntu",
        "key_path": "~/.ssh/id",
    }
    assert wb_cfg["storage"]["checkpoint_bucket"] == "s3://saved-bucket/checkpoints/"
    assert wb_cfg["endpoint_strategy"] == "ssh"
    assert wb_cfg["app_status"] == "healthy"


def test_fiftyone_deploy_writes_config_before_readiness_and_warns_on_timeout(
    tmp_path: Path,
    mocker,
) -> None:
    events: list[tuple[str, str]] = []
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")
    ssh.run_or_raise.return_value = (0, "FIFTYONE_ENV_SMOKE_OK", "")

    mocker.patch("npa.cli.fiftyone.provisioner.init")
    mocker.patch(
        "npa.cli.fiftyone.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.22",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.fiftyone.resolve_environment", return_value=None)
    mocker.patch("npa.cli.fiftyone.list_projects", return_value={})
    mocker.patch(
        "npa.cli.fiftyone.write_config",
        side_effect=lambda data: events.append(
            ("write", data["projects"]["proj"]["workbenches"]["curate"]["app_status"])
        ),
    )
    mocker.patch(
        "npa.cli.fiftyone.update_workbench_app_status",
        side_effect=lambda _project, _name, status: events.append(("status", status)),
    )
    mocker.patch("npa.cli.fiftyone.write_manifest")
    mocker.patch(
        "npa.cli.fiftyone._app_health_check",
        side_effect=lambda _endpoint: events.append(("health", "timeout")) or False,
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            "proj",
            "-n",
            "curate",
            "deploy",
            "--project-id",
            "project",
            "--tenant-id",
            "tenant",
            "--region",
            "eu-north1",
            "--tf-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Warning" in result.output
    assert "Deploy complete" in result.output
    assert events.index(("write", "provisioned")) < events.index(("health", "timeout"))
    assert ("status", "installing") in events
    assert ("status", "provisioned") in events
    assert ("status", "install_failed") not in events
    assert ("status", "healthy") not in events


def test_fiftyone_deploy_accepts_ready_marker_when_ssh_exits_nonzero(
    tmp_path: Path,
    mocker,
) -> None:
    ssh = mocker.MagicMock()
    ssh.run.side_effect = [
        (0, "connected", ""),
        (1, "FIFTYONE_ENV_SMOKE_OK\nNPA_FIFTYONE_APP_READY\n", "late systemd status"),
    ]

    mocker.patch("npa.cli.fiftyone.provisioner.init")
    mocker.patch(
        "npa.cli.fiftyone.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.23",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.fiftyone.resolve_environment", return_value=None)
    mocker.patch("npa.cli.fiftyone.list_projects", return_value={})
    mocker.patch("npa.cli.fiftyone.write_config")
    update_status = mocker.patch("npa.cli.fiftyone.update_workbench_app_status")
    mocker.patch("npa.cli.fiftyone.write_manifest")
    mocker.patch("npa.cli.fiftyone._app_health_check", return_value=True)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            "proj",
            "-n",
            "curate",
            "deploy",
            "--project-id",
            "project",
            "--tenant-id",
            "tenant",
            "--region",
            "eu-north1",
            "--tf-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "FiftyOne installation failed" not in result.output
    assert update_status.call_args_list[-1].args == ("proj", "curate", "healthy")


def test_fiftyone_deploy_fails_nonzero_ssh_without_ready_marker(
    tmp_path: Path,
    mocker,
) -> None:
    ssh = mocker.MagicMock()
    ssh.run.side_effect = [
        (0, "connected", ""),
        (1, "FIFTYONE_ENV_SMOKE_OK\n", "install boom"),
    ]

    mocker.patch("npa.cli.fiftyone.provisioner.init")
    mocker.patch(
        "npa.cli.fiftyone.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.24",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.fiftyone.resolve_environment", return_value=None)
    mocker.patch("npa.cli.fiftyone.list_projects", return_value={})
    mocker.patch("npa.cli.fiftyone.write_config")
    update_status = mocker.patch("npa.cli.fiftyone.update_workbench_app_status")

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            "proj",
            "-n",
            "curate",
            "deploy",
            "--project-id",
            "project",
            "--tenant-id",
            "tenant",
            "--region",
            "eu-north1",
            "--tf-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "FiftyOne installation failed" in result.output
    assert update_status.call_args_list[-1].args == ("proj", "curate", "install_failed")


def test_fiftyone_deploy_rejects_partial_gpu_selection(tmp_path: Path, mocker) -> None:
    apply = mocker.patch("npa.cli.fiftyone.provisioner.apply")

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
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
            "gpu-l40s-a",
        ],
    )

    assert result.exit_code == 1
    assert "Missing --gpu-preset" in result.output
    apply.assert_not_called()


def test_fiftyone_launch_builds_remote_command_and_url(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "NPA_FIFTYONE_APP_READY", "")
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg())
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        ["workbench", "fiftyone", "launch", "--port", "6161"],
    )

    assert result.exit_code == 0
    assert "http://fiftyone.example:6161" in result.output
    cmd = ssh.run.call_args.args[0]
    assert "test -x /opt/fiftyone/venv/bin/python" in cmd
    assert "FIFTYONE_DEFAULT_APP_PORT=6161" in cmd
    assert "sudo systemctl enable npa-fiftyone-app" in cmd
    assert "http://127.0.0.1:6161/" in cmd
    assert "TimeoutStopSec=15" in cmd


def test_fiftyone_launch_accepts_ready_marker_when_ssh_exits_nonzero(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (1, "FiftyOne already running\nNPA_FIFTYONE_APP_READY\n", "")
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg())
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "fiftyone", "launch"])

    assert result.exit_code == 0
    assert "http://fiftyone.example:5151" in result.output


def test_fiftyone_launch_adds_polling_for_ssh_endpoint_strategy(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "NPA_FIFTYONE_APP_READY", "")
    cfg = _cfg()
    cfg.endpoint_strategy = "ssh"
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=cfg)
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "fiftyone", "launch"])

    assert result.exit_code == 0
    assert "http://127.0.0.1:5151?polling=true" in result.output


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "s3://bucket/images",
            [
                "download_s3(SOURCE)",
                "boto3.client",
                "download_file",
                "dataset.add_samples",
                "fo.Sample",
                "_refresh_fiftyone_collection_stats(dataset)",
                "stale estimatedDocumentCount",
            ],
        ),
        ("Voxel51/VisDrone2019-DET", ["load_from_hub(SOURCE, name=NAME)", "source_type = \"huggingface\""]),
        (
            "https://huggingface.co/datasets/Voxel51/VisDrone2019-DET",
            ["load_from_hub(SOURCE, name=NAME)", "source_type = \"huggingface\""],
        ),
    ],
)
def test_fiftyone_load_dataset_builds_source_specific_command(
    source: str,
    expected: list[str],
    mocker,
) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, '{"status": "loaded"}', "")
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg())
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "load-dataset",
            "--name",
            "curated",
            "--input-path",
            source,
        ],
    )

    assert result.exit_code == 0
    cmd = ssh.run.call_args.args[0]
    assert 'NAME = "curated"' in cmd
    assert f'SOURCE = "{source}"' in cmd
    assert 'FORMAT = "auto"' in cmd
    assert "FIFTYONE_DATASET_NAME=curated" in cmd
    assert 'sudo chown "$USER:$USER" /etc/npa-fiftyone/env' in cmd
    assert "sudo systemctl restart npa-fiftyone-app" in cmd
    assert "NPA_FIFTYONE_APP_READY" in cmd
    for snippet in expected:
        assert snippet in cmd


def test_fiftyone_load_dataset_video_format_uses_video_loader(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, '{"status": "loaded", "samples": 1}', "")
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg())
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "load-dataset",
            "--name",
            "videos",
            "--input-path",
            "s3://bucket/cosmos/out.mp4",
            "--format",
            "video",
        ],
    )

    assert result.exit_code == 0
    cmd = ssh.run.call_args.args[0]
    assert 'FORMAT = "video"' in cmd
    assert "VIDEO_EXTENSIONS" in cmd
    assert "dataset.add_samples" in cmd
    assert "fo.Sample" in cmd
    assert "fo.Dataset.from_videos" not in cmd
    assert "fo.types.VideoDirectory" not in cmd


def test_fiftyone_load_dataset_container_runtime_execs_app_container(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, '{"status": "loaded", "samples": 5}', "")
    cfg = _cfg()
    cfg.runtime = "container"
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=cfg)
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "load-dataset",
            "--name",
            "curated",
            "--input-path",
            "s3://bucket/images",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    cmd = ssh.run.call_args.args[0]
    assert "sudo docker exec -i npa-fiftyone bash -lc" in cmd
    assert "dataset.add_samples" in cmd
    assert "_refresh_fiftyone_collection_stats(dataset)" in cmd
    assert "sudo docker stop npa-fiftyone" not in cmd
    assert "sudo docker run --rm" not in cmd


def test_fiftyone_load_dataset_rejects_vm_local_cosmos_output_at_cli_boundary(mocker) -> None:
    resolve_ssh = mocker.patch("npa.cli.fiftyone.resolve_ssh_config")
    ssh_cls = mocker.patch("npa.cli.fiftyone.SSHClient")

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "load-dataset",
            "--name",
            "videos",
            "--input-path",
            "/opt/cosmos-data/outputs/cosmos.mp4",
            "--format",
            "video",
        ],
    )

    assert result.exit_code == 1
    assert (
        "FiftyOne load-dataset expects an S3 URI or a Hugging Face Hub dataset. "
        "VM-local paths are not supported. If you generated this with cosmos infer, "
        "pass the same s3:// URI you used for --output-path."
    ) in result.output
    assert "HFValidationError" not in result.output
    resolve_ssh.assert_not_called()
    ssh_cls.assert_not_called()


@pytest.mark.parametrize(
    "source",
    [
        "s3://bucket/lerobot-dataset",
        "lerobot/pusht",
    ],
)
def test_fiftyone_load_dataset_lerobot_format_uses_remote_importer(
    source: str,
    mocker,
) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (
        0,
        json.dumps({
            "status": "loaded",
            "format": "lerobot",
            "samples": 2,
            "metadata_fields": ["episode_index", "frame_index"],
        }),
        "",
    )
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg())
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "load-dataset",
            "--name",
            "curated",
            "--input-path",
            source,
            "--format",
            "lerobot",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["format"] == "lerobot"
    cmd = ssh.run.call_args.args[0]
    assert 'NAME = "curated"' in cmd
    assert f'SOURCE = "{source}"' in cmd
    assert 'FORMAT = "lerobot"' in cmd
    assert "npa_fiftyone_lerobot_importer.py" in cmd
    assert "def import_lerobot_dataset(" in cmd
    assert "stale estimatedDocumentCount" in cmd
    assert "import_lerobot_dataset(NAME, SOURCE, DATASETS_DIR)" in cmd


def test_fiftyone_load_dataset_accepts_ready_marker_when_ssh_exits_nonzero(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (
        1,
        '{"status": "loaded", "name": "curated"}\nNPA_FIFTYONE_APP_READY\n',
        "late restart warning",
    )
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg())
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "load-dataset",
            "--name",
            "curated",
            "--input-path",
            "s3://bucket/images",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "loaded"
    assert payload["name"] == "curated"


def test_fiftyone_load_dataset_suppresses_transient_curl_errors_on_success(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (
        0,
        '{"status": "loaded", "samples": 5}\nNPA_FIFTYONE_APP_READY\n',
        "curl: (7) Failed to connect to 127.0.0.1 port 5151: Couldn't connect to server\n",
    )
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg())
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "load-dataset",
            "--name",
            "curated",
            "--input-path",
            "s3://bucket/fiftyone-ranked/",
        ],
    )

    assert result.exit_code == 0
    assert '"status": "loaded"' in result.output
    assert "curl: (7)" not in result.output


def test_fiftyone_load_dataset_fails_nonzero_ssh_without_ready_marker(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (1, '{"status": "loaded"}\n', "restart failed")
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg())
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "load-dataset",
            "--name",
            "curated",
            "--input-path",
            "s3://bucket/images",
        ],
    )

    assert result.exit_code == 1
    assert "SSH error" in result.output
    assert "restart failed" in result.output


def test_fiftyone_load_dataset_restart_timeout_is_warning_not_failure(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, '{"status": "loaded"}', "")
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg())
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "load-dataset",
            "--name",
            "curated",
            "--input-path",
            "s3://bucket/images",
        ],
    )

    assert result.exit_code == 0
    cmd = ssh.run.call_args.args[0]
    assert "restart readiness timeout" in cmd
    assert "exit 0" in cmd


def test_fiftyone_restart_restarts_systemd_service(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "NPA_FIFTYONE_APP_READY", "")
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg())
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)
    update_status = mocker.patch("npa.cli.fiftyone.update_workbench_app_status")

    result = runner.invoke(
        app,
        ["workbench", "fiftyone", "-p", "proj", "-n", "curate", "restart"],
    )

    assert result.exit_code == 0
    assert "status: restarted" in result.output
    cmd = ssh.run.call_args.args[0]
    assert "sudo systemctl restart npa-fiftyone-app" in cmd
    assert "http://127.0.0.1:5151/" in cmd
    update_status.assert_called_once_with("proj", "curate", "healthy")


def test_fiftyone_restart_restarts_container_runtime(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "NPA_FIFTYONE_APP_READY", "")
    cfg = _cfg()
    cfg.runtime = "container"
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=cfg)
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.fiftyone.update_workbench_app_status")

    result = runner.invoke(app, ["workbench", "fiftyone", "restart"])

    assert result.exit_code == 0
    cmd = ssh.run.call_args.args[0]
    assert "sudo docker restart npa-fiftyone" in cmd


def test_fiftyone_datasets_list_queries_graphql(mocker) -> None:
    response = mocker.MagicMock(status_code=200)
    response.json.return_value = {
        "data": {
            "datasets": {
                "total": 1,
                "edges": [
                    {
                        "node": {
                            "name": "demo_cosmos_ranked",
                            "persistent": True,
                            "mediaType": "image",
                            "estimatedSampleCount": 5,
                        }
                    }
                ],
            }
        }
    }
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg())
    post = mocker.patch("npa.cli.fiftyone.httpx.post", return_value=response)

    result = runner.invoke(
        app,
        ["workbench", "fiftyone", "datasets", "list", "--output", "json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["total"] == 1
    assert payload["datasets"][0]["name"] == "demo_cosmos_ranked"
    assert payload["datasets"][0]["samples"] == 5
    post.assert_called_once()
    assert post.call_args.args[0] == "http://fiftyone.example:5151/graphql"
    assert post.call_args.kwargs["json"]["variables"] == {"first": 100, "search": ""}


def test_fiftyone_status_checks_app_port_url(mocker) -> None:
    response = mocker.MagicMock(status_code=200)
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg())
    get = mocker.patch("npa.cli.fiftyone.httpx.get", return_value=response)

    result = runner.invoke(
        app,
        ["workbench", "fiftyone", "status", "--port", "6161"],
    )

    assert result.exit_code == 0
    assert "server: up" in result.output
    assert "http://fiftyone.example:6161" in result.output
    get.assert_called_once_with("http://fiftyone.example:6161", timeout=5.0)


def test_fiftyone_status_uses_recorded_ssh_endpoint_strategy(mocker) -> None:
    cfg = _cfg()
    cfg.endpoint_strategy = "ssh"
    cfg.service_port = 5151
    response = mocker.MagicMock(status_code=200)
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=cfg)
    endpoint = mocker.patch(
        "npa.cli.fiftyone.service_endpoint",
        return_value=_active_endpoint("http://127.0.0.1:15151"),
    )
    get = mocker.patch("npa.cli.fiftyone.httpx.get", return_value=response)

    result = runner.invoke(app, ["workbench", "fiftyone", "status"])

    assert result.exit_code == 0
    endpoint.assert_called_once_with(
        cfg,
        default_port=5151,
        endpoint="http://fiftyone.example:5151",
        service_port=5151,
    )
    get.assert_called_once_with("http://127.0.0.1:15151", timeout=5.0)


def test_fiftyone_status_self_heals_legacy_byovm_alias(
    tmp_path: Path,
    monkeypatch,
    mocker,
) -> None:
    cfg_path = tmp_path / ".npa" / "config.yaml"
    credentials_path = tmp_path / ".npa" / "credentials.yaml"
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", credentials_path)
    for env_var in config_module.ENV_MAP.values():
        monkeypatch.delenv(env_var, raising=False)
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(yaml.safe_dump({
        "projects": {
            "proj": {
                "workbenches": {
                    "curate": {
                        "endpoint": "http://66.201.4.1:5151",
                        "runtime": "byovm",
                        "app_port": 5151,
                        "ssh": {
                            "host": "66.201.4.1",
                            "user": "ubuntu",
                            "key_path": "~/.ssh/h200",
                        },
                    },
                },
            },
        },
    }))

    response = mocker.MagicMock(status_code=200)
    get = mocker.patch("npa.cli.fiftyone.httpx.get", return_value=response)
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "", "")
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)
    process = SimpleNamespace(
        poll=lambda: None,
        terminate=mocker.MagicMock(),
        wait=mocker.MagicMock(),
        stderr=SimpleNamespace(read=lambda: ""),
    )
    mocker.patch("npa.clients.endpoint.subprocess.Popen", return_value=process)
    mocker.patch("npa.clients.endpoint._tcp_open", return_value=False)
    mocker.patch("npa.clients.endpoint._free_local_port", side_effect=[15151, 15152])
    mocker.patch("npa.clients.endpoint._wait_for_local_port")

    first = runner.invoke(app, ["workbench", "fiftyone", "-p", "proj", "-n", "curate", "status"])

    assert first.exit_code == 0
    get.assert_called_with("http://127.0.0.1:15151", timeout=5.0)
    saved = yaml.safe_load(cfg_path.read_text())
    wb = saved["projects"]["proj"]["workbenches"]["curate"]
    assert wb["endpoint_strategy"] == "ssh"
    assert wb["service_port"] == 5151

    public_probe = mocker.patch("npa.clients.endpoint._public_endpoint_open")
    second = runner.invoke(app, ["workbench", "fiftyone", "-p", "proj", "-n", "curate", "status"])

    assert second.exit_code == 0
    get.assert_called_with("http://127.0.0.1:15152", timeout=5.0)
    public_probe.assert_not_called()


def test_fiftyone_status_reports_http_error(mocker) -> None:
    response = mocker.MagicMock(status_code=503)
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg(app_status="provisioning"))
    mocker.patch("npa.cli.fiftyone.httpx.get", return_value=response)

    result = runner.invoke(app, ["workbench", "fiftyone", "status"])

    assert result.exit_code == 1
    assert "app_status: unreachable" in result.output
    assert "returned HTTP 503" in result.output


def test_fiftyone_status_reports_provisioning_when_unreachable(mocker) -> None:
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg(app_status="provisioning"))
    mocker.patch("npa.cli.fiftyone.httpx.get", side_effect=httpx.ConnectError("down"))

    result = runner.invoke(app, ["workbench", "fiftyone", "status"])

    assert result.exit_code == 1
    assert "app_status: unreachable" in result.output
    assert "Cannot reach FiftyOne app" in result.output


def test_fiftyone_launch_maps_ssh_error(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (1, "", "ssh failed")
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg())
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "fiftyone", "launch"])

    assert result.exit_code == 1
    assert "ssh failed" in result.output


def test_fiftyone_load_dataset_accepts_deprecated_source_alias(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, '{"status": "loaded"}', "")
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg())
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "load-dataset",
            "--name",
            "curated",
            "--source",
            "s3://bucket/images",
        ],
    )

    assert result.exit_code == 0
    assert 'SOURCE = "s3://bucket/images"' in ssh.run.call_args.args[0]


def test_fiftyone_list_filters_to_fiftyone_workbenches(mocker) -> None:
    mocker.patch("npa.cli.fiftyone.default_project_name", return_value="proj")
    mocker.patch("npa.cli.fiftyone.default_workbench_name", return_value="curate")
    mocker.patch(
        "npa.cli.fiftyone.list_projects",
        return_value={
            "proj": {
                "region": "eu-north1",
                "workbenches": {
                    "curate": {
                        "workbench_type": "fiftyone",
                        "gpu_platform": DEFAULT_CPU_PLATFORM,
                        "endpoint": "http://fiftyone.example:5151",
                        "app_status": "provisioning",
                    },
                    "cosmos": {
                        "workbench_type": "cosmos",
                        "endpoint": "http://cosmos:8080",
                    },
                    "train": {
                        "workbench_type": "lerobot",
                        "endpoint": "http://train:8080",
                    },
                },
            }
        },
    )

    result = runner.invoke(app, ["workbench", "fiftyone", "list"])

    assert result.exit_code == 0
    assert "curate" in result.output
    assert "app_status=provisioning" in result.output
    assert "cosmos" not in result.output
    assert "train" not in result.output


def test_fiftyone_destroy_removes_provisioning_workbench(tmp_path: Path, mocker) -> None:
    destroy = mocker.patch("npa.cli.fiftyone.provisioner.destroy")
    remove = mocker.patch("npa.cli.fiftyone.remove_workbench_config")
    mocker.patch("npa.cli.fiftyone.resolve_environment", return_value=None)
    cfg = _cfg(app_status="provisioning")
    cfg.tf_instance_name = "fiftyone-proj-curate-existing"
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=cfg)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            "proj",
            "-n",
            "curate",
            "deploy",
            "--project-id",
            "project",
            "--tenant-id",
            "tenant",
            "--region",
            "eu-north1",
            "--tf-dir",
            str(tmp_path),
            "--destroy",
        ],
    )

    assert result.exit_code == 0
    tf_vars = destroy.call_args.kwargs["tf_vars"]
    assert tf_vars["instance_name"] == "fiftyone-proj-curate-existing"
    assert tf_vars["workbench_type"] == "fiftyone"
    remove.assert_called_once_with("proj", "curate")


def test_fiftyone_destroy_reuses_saved_terraform_state_credentials(mocker) -> None:
    prepare = mocker.patch(
        "npa.cli.fiftyone.provisioner.prepare_working_dir",
        return_value=Path("/tmp/npa-state-workdir"),
    )
    init = mocker.patch("npa.cli.fiftyone.provisioner.init")
    destroy = mocker.patch("npa.cli.fiftyone.provisioner.destroy")
    mocker.patch("npa.cli.fiftyone.provisioner.cleanup_working_dir")
    mocker.patch("npa.cli.fiftyone.remove_workbench_config")
    mocker.patch(
        "npa.cli.fiftyone.resolve_environment",
        return_value=EnvironmentConfig(
            project_id="project",
            tenant_id="tenant",
            region="eu-north1",
        ),
    )
    cfg = _cfg(app_status="provisioning")
    cfg.tf_instance_name = "fiftyone-proj-curate"
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=cfg)
    mocker.patch(
        "npa.cli.fiftyone.resolve_terraform_state",
        return_value=TerraformStateConfig(
            bucket="saved-bucket",
            endpoint="https://saved-storage.example",
            access_key="saved-access",
            secret_key="saved-secret",
        ),
    )
    bootstrap = mocker.patch(
        "npa.clients.nebius.bootstrap_environment",
    )
    mocker.patch("npa.clients.nebius.get_iam_token", return_value="iam")
    mocker.patch("npa.clients.nebius.ensure_service_account", return_value="sa")
    mocker.patch("npa.cli.fiftyone.write_config")

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            "proj",
            "-n",
            "curate",
            "deploy",
            "--destroy",
        ],
    )

    assert result.exit_code == 0
    bootstrap.assert_not_called()
    prepare.assert_called_once_with(
        "proj",
        "curate",
        bucket="saved-bucket",
        region="eu-north1",
        endpoint="https://saved-storage.example",
    )
    init.assert_called_once_with(
        tf_dir="/tmp/npa-state-workdir",
        backend_config={"access_key": "saved-access", "secret_key": "saved-secret"},
    )
    tf_vars = destroy.call_args.kwargs["tf_vars"]
    assert tf_vars["s3_bucket"] == "saved-bucket"
    assert tf_vars["s3_endpoint"] == "https://saved-storage.example"
    assert tf_vars["nebius_api_key"] == "saved-access"
    assert tf_vars["nebius_secret_key"] == "saved-secret"
    assert tf_vars["iam_token"] == "iam"
    assert tf_vars["service_account_id"] == "sa"


def test_fiftyone_system_info_prints_ssh_output(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (0, "cpu info", "")
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg())
    mocker.patch("npa.cli.fiftyone.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "fiftyone", "system-info"])

    assert result.exit_code == 0
    assert "cpu info" in result.output
    cmd = ssh.run_or_raise.call_args.args[0]
    assert "nvidia-smi" in cmd
    assert "lscpu" in cmd
    assert "free -h" in cmd
    assert "lsblk" in cmd
