from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from npa.cli.fiftyone import (
    DEFAULT_CPU_IMAGE_FAMILY,
    DEFAULT_CPU_PLATFORM,
    DEFAULT_CPU_PRESET,
    FIFTYONE_VERSION,
)
from npa.cli.main import app
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


@pytest.mark.parametrize(
    "command",
    [
        "deploy",
        "launch",
        "load-dataset",
        "status",
        "system-info",
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

    assert result.exit_code == 0
    assert "--format" in result.output
    assert "lerobot" in result.output


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
    mocker.patch("npa.deploy.configurator.write_remote_env_file")
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
    wb_cfg = write_config.call_args.args[0]["projects"]["proj"]["workbenches"]["curate-container"]
    assert wb_cfg["runtime"] == "container"
    assert update_status.call_args_list[0].args == ("proj", "curate-container", "installing")
    assert update_status.call_args_list[-1].args == ("proj", "curate-container", "healthy")


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


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("s3://bucket/images", ["download_s3(SOURCE)", "boto3.client", "download_file"]),
        ("/data/images", ["elif Path(SOURCE).exists()", "fo.Dataset.from_dir", 'SOURCE = "/data/images"']),
        ("Voxel51/VisDrone2019-DET", ["load_from_hub(SOURCE, name=NAME)", "source_type = \"huggingface\""]),
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


@pytest.mark.parametrize(
    "source",
    [
        "/data/lerobot",
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
            "/data/images",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "loaded"
    assert payload["name"] == "curated"


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
            "/data/images",
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
            "/data/images",
        ],
    )

    assert result.exit_code == 0
    cmd = ssh.run.call_args.args[0]
    assert "restart readiness timeout" in cmd
    assert "exit 0" in cmd


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


def test_fiftyone_status_reports_http_error(mocker) -> None:
    response = mocker.MagicMock(status_code=503)
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg(app_status="provisioning"))
    mocker.patch("npa.cli.fiftyone.httpx.get", return_value=response)

    result = runner.invoke(app, ["workbench", "fiftyone", "status"])

    assert result.exit_code == 1
    assert "app_status: provisioning" in result.output
    assert "returned HTTP 503" in result.output


def test_fiftyone_status_reports_provisioning_when_unreachable(mocker) -> None:
    mocker.patch("npa.cli.fiftyone.resolve_ssh_config", return_value=_cfg(app_status="provisioning"))
    mocker.patch("npa.cli.fiftyone.httpx.get", side_effect=httpx.ConnectError("down"))

    result = runner.invoke(app, ["workbench", "fiftyone", "status"])

    assert result.exit_code == 1
    assert "app_status: provisioning" in result.output
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
            "/data/images",
        ],
    )

    assert result.exit_code == 0
    assert 'SOURCE = "/data/images"' in ssh.run.call_args.args[0]


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
