from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import SSHConfig, StorageConfig, WorkbenchConfig
from npa.clients.ssh import SSHError


runner = CliRunner()


def _ssh_cfg() -> WorkbenchConfig:
    return WorkbenchConfig(
        endpoint="",
        ssh=SSHConfig(host="isaac", user="ubuntu", key_path="~/.ssh/id"),
        storage=StorageConfig(checkpoint_bucket="", endpoint_url=""),
    )


@pytest.mark.parametrize(
    "command",
    [
        "deploy",
        "status",
        "system-info",
        "train",
        "eval",
        "list",
    ],
)
def test_isaac_lab_command_help(command: str) -> None:
    result = runner.invoke(app, ["workbench", "isaac-lab", command, "--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_isaac_lab_registered_under_workbench() -> None:
    result = runner.invoke(app, ["workbench", "--help"])

    assert result.exit_code == 0
    assert "isaac-lab" in result.output


def test_isaac_lab_deploy_requires_gpu_selection(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "-p",
            "proj",
            "-n",
            "isaac",
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
    assert "GPU selection is required" in result.output
    assert "L40S" in result.output
    assert "H100/H200" in result.output


def test_isaac_lab_deploy_installs_expected_package(tmp_path: Path, mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")
    ssh.run_or_raise.return_value = (0, "ISAAC_LAB_ENV_SMOKE_OK", "")

    init = mocker.patch("npa.cli.isaac_lab.provisioner.init")
    apply = mocker.patch(
        "npa.cli.isaac_lab.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.5",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.isaac_lab.resolve_environment", return_value=None)
    write_config = mocker.patch("npa.cli.isaac_lab.write_config")
    update_status = mocker.patch("npa.cli.isaac_lab.update_workbench_app_status")
    mocker.patch("npa.cli.isaac_lab.write_manifest")
    mocker.patch("npa.cli.isaac_lab.list_projects", return_value={})

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "-p",
            "proj",
            "-n",
            "isaac",
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

    assert result.exit_code == 0
    assert "Deploy complete" in result.output
    init.assert_called_once_with(tf_dir=str(tmp_path), backend_config=None)
    apply.assert_called_once()
    tf_vars = apply.call_args.kwargs["tf_vars"]
    assert tf_vars["gpu_platform"] == "gpu-h100-sxm"
    assert tf_vars["gpu_preset"] == "1gpu-16vcpu-200gb"
    assert "boot_disk_size_gb" not in tf_vars

    install_cmd = ssh.run_or_raise.call_args.args[0]
    assert "python3.11 -m venv /opt/isaac-lab/venv" in install_cmd
    assert (
        '/opt/isaac-lab/venv/bin/python -m pip install "isaaclab[isaacsim,all]==2.3.2.post1" '
        "--extra-index-url https://pypi.nvidia.com"
    ) in install_cmd
    assert "ISAAC_LAB_ENV_SMOKE_OK" in install_cmd
    write_config.assert_called()
    wb_cfg = write_config.call_args.args[0]["projects"]["proj"]["workbenches"]["isaac"]
    assert wb_cfg["app_status"] == "provisioned"
    assert update_status.call_args_list[0].args == ("proj", "isaac", "installing")
    assert update_status.call_args_list[-1].args == ("proj", "isaac", "healthy")


def test_isaac_lab_deploy_runtime_container_starts_image(tmp_path: Path, mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")

    mocker.patch("npa.cli.isaac_lab.provisioner.init")
    apply = mocker.patch(
        "npa.cli.isaac_lab.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.35",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.isaac_lab.resolve_environment", return_value=None)
    write_config = mocker.patch("npa.cli.isaac_lab.write_config")
    update_status = mocker.patch("npa.cli.isaac_lab.update_workbench_app_status")
    mocker.patch("npa.cli.isaac_lab.write_manifest")
    mocker.patch("npa.cli.isaac_lab.list_projects", return_value={})
    deploy_container = mocker.patch("npa.deploy.configurator.deploy_workbench_container")
    mocker.patch("npa.deploy.configurator.write_remote_docker_env_file")

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "-p",
            "proj",
            "-n",
            "isaac-container",
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
            "--runtime",
            "container",
        ],
    )

    assert result.exit_code == 0
    tf_vars = apply.call_args.kwargs["tf_vars"]
    assert tf_vars["workbench_type"] == "lerobot-container"
    assert tf_vars["boot_disk_size_gb"] == "250"
    deploy_container.assert_called_once()
    assert deploy_container.call_args.kwargs["container_name"] == "npa-isaac-lab"
    assert deploy_container.call_args.kwargs["image_ref"].endswith("/npa-isaac-lab:2.3.2.post1")
    wb_cfg = write_config.call_args.args[0]["projects"]["proj"]["workbenches"]["isaac-container"]
    assert wb_cfg["runtime"] == "container"
    assert update_status.call_args_list[0].args == ("proj", "isaac-container", "installing")
    assert update_status.call_args_list[-1].args == ("proj", "isaac-container", "healthy")


def test_isaac_lab_train_builds_remote_command(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "", "")
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "train",
            "--task",
            "Isaac-Reach-Franka-v0",
            "--num-envs",
            "64",
            "--steps",
            "25",
            "--output-dir",
            "/tmp/isaac-out",
        ],
    )

    assert result.exit_code == 0
    cmd = ssh.run.call_args.args[0]
    assert "source /opt/isaac-lab/venv/bin/activate" in cmd
    assert "ISAACLAB_PKG=/opt/isaac-lab/venv/lib/python3.11/site-packages/isaaclab" in cmd
    assert "$ISAACLAB_PKG/source/isaaclab_tasks" in cmd
    assert "from isaaclab.app import AppLauncher" in cmd
    assert "import isaaclab_tasks" in cmd
    assert "parse_env_cfg" in cmd
    assert "Isaac-Reach-Franka-v0" in cmd
    assert "num_envs = 64" in cmd
    assert "steps = 25" in cmd
    assert "/tmp/isaac-out" in cmd
    assert "ISAAC_LAB_ENV_CREATE_COMPLETE" in cmd
    assert "ISAAC_LAB_ENV_RESET_COMPLETE" in cmd
    assert "npa_isaac_lab_random_policy_checkpoint.json" in cmd
    assert "checkpoint_path" in cmd
    assert "ISAAC_LAB_TRAIN_COMPLETE" in cmd


def test_isaac_lab_train_container_uses_docker_exec(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "", "")
    cfg = _ssh_cfg()
    cfg.runtime = "container"
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=cfg)
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "train",
            "--task",
            "Isaac-Reach-Franka-v0",
            "--steps",
            "1",
            "--output-dir",
            "/opt/isaac-lab/runs/container-test",
        ],
    )

    assert result.exit_code == 0
    cmd = ssh.run.call_args.args[0]
    assert "sudo docker exec npa-isaac-lab" in cmd
    assert "/isaac-sim/python.sh" in cmd
    assert "import json\nimport time" in cmd
    assert "/opt/isaac-lab/runs/container-test" in cmd


def test_isaac_lab_eval_builds_remote_command(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "", "")
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "eval",
            "--task",
            "Isaac-Reach-Franka-v0",
            "--checkpoint",
            "/opt/isaac-lab/runs/model.pt",
            "--num-episodes",
            "3",
            "--output-dir",
            "/tmp/isaac-eval",
        ],
    )

    assert result.exit_code == 0
    cmd = ssh.run.call_args.args[0]
    assert "source /opt/isaac-lab/venv/bin/activate" in cmd
    assert "from isaaclab.app import AppLauncher" in cmd
    assert "import isaaclab_tasks" in cmd
    assert "parse_env_cfg" in cmd
    assert "Isaac-Reach-Franka-v0" in cmd
    assert "checkpoint_path = Path(" in cmd
    assert "/opt/isaac-lab/runs/model.pt" in cmd
    assert "num_episodes = 3" in cmd
    assert "/tmp/isaac-eval" in cmd
    assert "ISAAC_LAB_EVAL_EPISODE" in cmd
    assert "ISAAC_LAB_EVAL_COMPLETE" in cmd


def test_isaac_lab_public_path_options_reject_local_paths(mocker) -> None:
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    ssh_cls = mocker.patch("npa.cli.isaac_lab.SSHClient")

    train = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "train",
            "--task",
            "Isaac-Reach-Franka-v0",
            "--output-path",
            "/tmp/isaac-out",
        ],
    )
    eval_result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "eval",
            "--task",
            "Isaac-Reach-Franka-v0",
            "--input-path",
            "/tmp/model.pt",
            "--output-path",
            "s3://bucket/eval/",
        ],
    )

    assert train.exit_code == 1
    assert "Isaac Lab train --output-path expects an S3 URI" in train.output
    assert eval_result.exit_code == 1
    assert "Isaac Lab eval --input-path expects an S3 URI" in eval_result.output
    ssh_cls.assert_not_called()


def test_isaac_lab_train_accepts_deprecated_output_dir_alias(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "", "")
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "train",
            "--task",
            "Isaac-Reach-Franka-v0",
            "--output-dir",
            "/tmp/old-isaac-out",
        ],
    )

    assert result.exit_code == 0
    assert "/tmp/old-isaac-out" in ssh.run.call_args.args[0]


def test_isaac_lab_eval_accepts_deprecated_checkpoint_and_output_dir_aliases(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "", "")
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "eval",
            "--task",
            "Isaac-Reach-Franka-v0",
            "--checkpoint",
            "/tmp/old-checkpoint.json",
            "--output-dir",
            "/tmp/old-isaac-eval",
        ],
    )

    assert result.exit_code == 0
    cmd = ssh.run.call_args.args[0]
    assert "/tmp/old-checkpoint.json" in cmd
    assert "/tmp/old-isaac-eval" in cmd


def test_isaac_lab_status_prints_ssh_output(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (0, "venv: present\nno isaac lab processes", "")
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "isaac-lab", "status"])

    assert result.exit_code == 0
    assert "venv: present" in result.output
    cmd = ssh.run_or_raise.call_args.args[0]
    assert "test -x /opt/isaac-lab/venv/bin/python" in cmd
    assert "ps -eo pid=,comm=,args=" in cmd
    assert "$2 !~ /^(bash|sh|zsh|ps|awk)$/" in cmd


def test_isaac_lab_status_maps_ssh_error(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.side_effect = SSHError("ssh failed")
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "isaac-lab", "status"])

    assert result.exit_code == 1
    assert "ssh failed" in result.output


def test_isaac_lab_system_info_prints_ssh_output(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (0, "gpu info", "")
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "isaac-lab", "system-info"])

    assert result.exit_code == 0
    assert "gpu info" in result.output
    cmd = ssh.run_or_raise.call_args.args[0]
    assert "nvidia-smi" in cmd
    assert "lscpu" in cmd
    assert "free -h" in cmd
    assert "lsblk" in cmd


def test_isaac_lab_system_info_maps_ssh_error(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.side_effect = SSHError("ssh failed")
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "isaac-lab", "system-info"])

    assert result.exit_code == 1
    assert "ssh failed" in result.output


def test_isaac_lab_list_filters_to_isaac_lab_workbenches(mocker) -> None:
    mocker.patch("npa.cli.isaac_lab.default_project_name", return_value="proj")
    mocker.patch("npa.cli.isaac_lab.default_workbench_name", return_value="isaac")
    mocker.patch(
        "npa.cli.isaac_lab.list_projects",
        return_value={
            "proj": {
                "region": "eu-north1",
                "workbenches": {
                    "isaac": {
                        "workbench_type": "isaac-lab",
                        "gpu_platform": "gpu-l40s-a",
                        "ssh": {"host": "isaac"},
                    },
                    "sim": {
                        "workbench_type": "genesis",
                        "gpu_platform": "gpu-l40s-a",
                        "ssh": {"host": "sim"},
                    },
                    "train": {
                        "workbench_type": "lerobot",
                        "endpoint": "http://train:8080",
                    },
                },
            }
        },
    )

    result = runner.invoke(app, ["workbench", "isaac-lab", "list"])

    assert result.exit_code == 0
    assert "isaac" in result.output
    assert "sim" not in result.output
    assert "train" not in result.output


def test_isaac_lab_list_no_projects_message(mocker) -> None:
    mocker.patch("npa.cli.isaac_lab.default_project_name", return_value="default")
    mocker.patch("npa.cli.isaac_lab.default_workbench_name", return_value="default")
    mocker.patch("npa.cli.isaac_lab.list_projects", return_value={})

    result = runner.invoke(app, ["workbench", "isaac-lab", "list"])

    assert result.exit_code == 0
    assert "No projects configured" in result.output
