from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from botocore.exceptions import ClientError
import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import SSHConfig, StorageConfig, WorkbenchConfig
from npa.clients.serverless import EndpointNotFoundError
from npa.clients.ssh import SSHError


runner = CliRunner()
TERRAFORM_PLAN_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "terraform_plans"


@pytest.fixture(autouse=True)
def _terraform_plan_allows_apply(mocker):
    mocker.patch.dict("os.environ", {"ACCEPT_EULA": "Y"})
    mocker.patch(
        "npa.cli.isaac_lab.provisioner.plan",
        return_value=(TERRAFORM_PLAN_FIXTURES / "fresh_create.txt").read_text(),
    )


def _ssh_cfg() -> WorkbenchConfig:
    return WorkbenchConfig(
        endpoint="",
        ssh=SSHConfig(host="isaac", user="ubuntu", key_path="~/.ssh/id"),
        storage=StorageConfig(checkpoint_bucket="", endpoint_url=""),
    )


def _access_denied(message: str = "AccessDenied") -> ClientError:
    return ClientError(
        {"Error": {"Code": "AccessDenied", "Message": message}},
        "PutObject",
    )


@pytest.mark.parametrize(
    "command",
    [
        "deploy",
        "status",
        "system-info",
        "train",
        "eval",
        "export-lerobot",
        "export-onnx",
        "list",
        "cleanup-partial",
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
    assert "RTX Pro 6000" in result.output
    assert "H100/H200" in result.output


def test_isaac_lab_deploy_defaults_to_reproducible_container(tmp_path: Path, mocker) -> None:
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
    mocker.patch("npa.cli.isaac_lab.resolve_container_registry", return_value="registry.example")
    mocker.patch(
        "npa.cli.isaac_lab.container_image_for_tool",
        return_value="registry.example/npa-isaac-lab:3.0.0b2.post1",
    )
    write_env = mocker.patch("npa.deploy.configurator.write_remote_docker_env_file")
    deploy_container = mocker.patch("npa.deploy.configurator.deploy_workbench_container")

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
            "gpu-l40s-a",
            "--gpu-preset",
            "1gpu-40vcpu-160gb",
        ],
    )

    assert result.exit_code == 0
    assert "Deploy complete" in result.output
    init.assert_called_once_with(tf_dir=str(tmp_path), backend_config=None)
    apply.assert_called_once()
    tf_vars = apply.call_args.kwargs["tf_vars"]
    assert tf_vars["gpu_platform"] == "gpu-l40s-a"
    assert tf_vars["gpu_preset"] == "1gpu-40vcpu-160gb"
    assert tf_vars["boot_disk_size_gb"] == "250"

    write_env.assert_called_once()
    deploy_container.assert_called_once()
    assert (
        deploy_container.call_args.kwargs["image_ref"]
        == "registry.example/npa-isaac-lab:3.0.0b2.post1"
    )
    write_config.assert_called()
    wb_cfg = write_config.call_args.args[0]["projects"]["proj"]["workbenches"]["isaac"]
    assert wb_cfg["app_status"] == "provisioned"
    assert update_status.call_args_list[0].args == ("proj", "isaac", "installing")
    assert update_status.call_args_list[-1].args == ("proj", "isaac", "healthy")


@pytest.mark.parametrize("dry_run", [False, True])
def test_isaac_lab_deploy_rejects_native_vm_install_consistently(
    dry_run: bool,
) -> None:
    args = [
        "workbench",
        "isaac-lab",
        "deploy",
        "--runtime",
        "vm",
        "--gpu-type",
        "gpu-l40s-a",
        "--gpu-preset",
        "1gpu-40vcpu-160gb",
    ]
    if dry_run:
        args.append("--dry-run")
    result = runner.invoke(
        app,
        args,
    )

    assert result.exit_code == 1
    assert "Native VM installation is not supported for Isaac Lab 3" in result.output
    assert "--runtime container" in result.output
    assert "Would install" not in result.output


def _isaac_existing_config() -> dict:
    return {
        "projects": {
            "proj": {
                "workbenches": {
                    "isaac": {
                        "ssh": {
                            "host": "10.0.0.5",
                            "user": "ubuntu",
                            "key_path": "~/.ssh/id",
                        },
                        "storage": {
                            "checkpoint_bucket": "s3://bucket/checkpoints/",
                            "endpoint_url": "https://storage.example",
                        },
                    },
                    "byovm": {
                        "runtime": "byovm",
                        "ssh": {
                            "host": "10.0.0.6",
                            "user": "ubuntu",
                            "key_path": "~/.ssh/id",
                        },
                        "storage": {
                            "checkpoint_bucket": "s3://bucket/checkpoints/",
                            "endpoint_url": "https://storage.example",
                        },
                    },
                }
            }
        }
    }


def test_isaac_lab_deploy_existing_alias_no_replace_skips_terraform(mocker) -> None:
    mocker.patch("npa.cli.isaac_lab.resolve_environment", return_value=None)
    mocker.patch("npa.cli.isaac_lab.alias_has_terraform_state", return_value=True)
    mocker.patch("npa.cli.isaac_lab.workbench_is_byovm", return_value=False)
    mocker.patch("npa.clients.config._load_yaml", return_value=_isaac_existing_config())
    mocker.patch("npa.cli.isaac_lab.write_config")
    mocker.patch("npa.cli.isaac_lab.list_projects", return_value={})
    init = mocker.patch("npa.cli.isaac_lab.provisioner.init")
    apply = mocker.patch("npa.cli.isaac_lab.provisioner.apply")

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
            "--gpu-type",
            "gpu-l40s-a",
            "--gpu-preset",
            "1gpu-40vcpu-160gb",
            "--skip-app",
        ],
    )

    assert result.exit_code == 0
    assert "updating in place without Terraform" in result.output
    init.assert_not_called()
    apply.assert_not_called()


def test_isaac_lab_deploy_existing_alias_with_replace_prompts_confirmation(mocker) -> None:
    mocker.patch("npa.cli.isaac_lab.resolve_environment", return_value=None)
    mocker.patch("npa.cli.isaac_lab.alias_has_terraform_state", return_value=True)
    mocker.patch("npa.cli.isaac_lab.workbench_is_byovm", return_value=False)
    mocker.patch("npa.cli.isaac_lab.typer.confirm", return_value=False)
    init = mocker.patch("npa.cli.isaac_lab.provisioner.init")
    apply = mocker.patch("npa.cli.isaac_lab.provisioner.apply")

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
            "--gpu-type",
            "gpu-l40s-a",
            "--gpu-preset",
            "1gpu-40vcpu-160gb",
            "--replace",
        ],
    )

    assert result.exit_code == 1
    assert "Aborted" in result.output
    init.assert_not_called()
    apply.assert_not_called()


def test_isaac_lab_deploy_existing_alias_with_replace_and_yes_runs_terraform(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.isaac_lab.resolve_environment", return_value=None)
    mocker.patch("npa.cli.isaac_lab.alias_has_terraform_state", return_value=True)
    mocker.patch("npa.cli.isaac_lab.workbench_is_byovm", return_value=False)
    confirm = mocker.patch("npa.cli.isaac_lab.typer.confirm")
    mocker.patch("npa.cli.isaac_lab.provisioner.init")
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
    mocker.patch("npa.cli.isaac_lab.write_config")
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
            "--replace",
            "--yes",
            "--tf-dir",
            str(tmp_path),
            "--gpu-type",
            "gpu-l40s-a",
            "--gpu-preset",
            "1gpu-40vcpu-160gb",
            "--skip-app",
        ],
    )

    assert result.exit_code == 0
    confirm.assert_not_called()
    apply.assert_called_once()


def test_isaac_lab_deploy_replacement_plan_without_replace_aborts(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.isaac_lab.resolve_environment", return_value=None)
    mocker.patch("npa.cli.isaac_lab.alias_has_terraform_state", return_value=False)
    mocker.patch("npa.cli.isaac_lab.workbench_is_byovm", return_value=False)
    mocker.patch("npa.cli.isaac_lab.provisioner.init")
    mocker.patch(
        "npa.cli.isaac_lab.provisioner.plan",
        return_value=(TERRAFORM_PLAN_FIXTURES / "gpu_type_change_full_replace.txt").read_text(),
    )
    apply = mocker.patch("npa.cli.isaac_lab.provisioner.apply")

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
            "--tf-dir",
            str(tmp_path),
            "--gpu-type",
            "gpu-l40s-a",
            "--gpu-preset",
            "1gpu-40vcpu-160gb",
            "--skip-app",
        ],
    )

    assert result.exit_code == 1
    assert "would replace or destroy managed infrastructure" in result.output
    assert "nebius_compute_v1_instance.workbench" in result.output
    apply.assert_not_called()


def test_isaac_lab_deploy_fresh_alias_runs_terraform(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.isaac_lab.resolve_environment", return_value=None)
    mocker.patch("npa.cli.isaac_lab.alias_has_terraform_state", return_value=False)
    mocker.patch("npa.cli.isaac_lab.workbench_is_byovm", return_value=False)
    mocker.patch("npa.cli.isaac_lab.provisioner.init")
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
    mocker.patch("npa.cli.isaac_lab.write_config")
    mocker.patch("npa.cli.isaac_lab.list_projects", return_value={})

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "-p",
            "proj",
            "-n",
            "new",
            "deploy",
            "--tf-dir",
            str(tmp_path),
            "--gpu-type",
            "gpu-l40s-a",
            "--gpu-preset",
            "1gpu-40vcpu-160gb",
            "--skip-app",
        ],
    )

    assert result.exit_code == 0
    apply.assert_called_once()


def test_isaac_lab_deploy_byovm_alias_skips_terraform(mocker) -> None:
    mocker.patch("npa.cli.isaac_lab.resolve_environment", return_value=None)
    mocker.patch("npa.cli.isaac_lab.alias_has_terraform_state", return_value=False)
    mocker.patch("npa.cli.isaac_lab.workbench_is_byovm", return_value=True)
    mocker.patch("npa.clients.config._load_yaml", return_value=_isaac_existing_config())
    mocker.patch("npa.cli.isaac_lab.write_config")
    mocker.patch("npa.cli.isaac_lab.list_projects", return_value={})
    init = mocker.patch("npa.cli.isaac_lab.provisioner.init")
    apply = mocker.patch("npa.cli.isaac_lab.provisioner.apply")

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "-p",
            "proj",
            "-n",
            "byovm",
            "deploy",
            "--gpu-type",
            "gpu-l40s-a",
            "--gpu-preset",
            "1gpu-40vcpu-160gb",
            "--skip-app",
        ],
    )

    assert result.exit_code == 0
    init.assert_not_called()
    apply.assert_not_called()


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
    assert deploy_container.call_args.kwargs["image_ref"].endswith("/npa-isaac-lab:3.0.0b2.post1")
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
    assert "ISAACLAB_PKG=/opt/isaac-lab/venv/lib/python3.12/site-packages/isaaclab" in cmd
    assert "$ISAACLAB_PKG/source/isaaclab_tasks" in cmd
    assert "scripts/reinforcement_learning/rsl_rl/train.py" in cmd
    assert "--task \"$TASK\"" in cmd
    assert "--num_envs \"$NUM_ENVS\"" in cmd
    assert "--max_iterations \"$MAX_ITERATIONS\"" in cmd
    assert "--visualizer none" in cmd
    assert "Refusing to generate or run a compatibility trainer" in cmd
    assert "agent.save_interval=1" in cmd
    assert "Isaac-Reach-Franka-v0" in cmd
    assert "NUM_ENVS=64" in cmd
    assert "MAX_ITERATIONS=25" in cmd
    assert "/tmp/isaac-out" in cmd
    assert "ISAAC_LAB_RSL_RL_COMMAND" in cmd
    assert "npa_isaac_lab_checkpoint.pt" in cmd
    assert "npa_isaac_lab_checkpoint_manifest.json" in cmd
    assert "checkpoint_path" in cmd
    assert "ISAAC_LAB_TRAIN_COMPLETE" in cmd


def test_isaac_lab_train_accepts_success_summary_with_nonzero_ssh_status(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (
        1,
        'ISAAC_LAB_TRAIN_COMPLETE\n{"status": "success", "checkpoint_count": 1}\n',
        "",
    )
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "train",
            "--task",
            "Isaac-Cartpole-v0",
            "--steps",
            "1",
            "--output-dir",
            "/tmp/isaac-out",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "success"
    assert payload["exit_code"] == 0
    assert payload["ssh_exit_code"] == 1


def test_isaac_lab_train_falls_back_to_remote_env_upload(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (
        1,
        'ISAAC_LAB_TRAIN_COMPLETE\n{"status": "success", "checkpoint_count": 1}\n',
        "",
    )
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)
    local_upload = mocker.patch(
        "npa.cli.isaac_lab._upload_remote_directory_to_s3",
        side_effect=RuntimeError("AccessDenied"),
    )
    remote_upload = mocker.patch(
        "npa.cli.isaac_lab._upload_existing_remote_directory_via_remote_env",
        return_value="s3://bucket/isaac/",
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "train",
            "--task",
            "Isaac-Cartpole-v0",
            "--steps",
            "1",
            "--output-path",
            "s3://bucket/isaac/",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "success"
    assert payload["upload_mode"] == "remote-env"
    assert payload["local_upload_error"] == "AccessDenied"
    local_upload.assert_called_once()
    remote_upload.assert_called_once()


def _mock_isaac_serverless_env(mocker):
    mocker.patch("npa.cli.isaac_lab.resolve_environment", return_value=SimpleNamespace(project_id="project-1"))
    mocker.patch(
        "npa.cli.isaac_lab.resolve_project_storage",
        return_value=SimpleNamespace(
            checkpoint_bucket="",
            endpoint_url="https://s3.example",
            aws_access_key_id="AKIA",
            aws_secret_access_key="SECRET",
        ),
    )
    mocker.patch("npa.cli.isaac_lab.resolve_container_registry", return_value="registry.example")
    mocker.patch("npa.cli.isaac_lab.container_image_for_tool", return_value="registry.example/npa-isaac-lab:smoke")
    return mocker.patch("npa.cli.isaac_lab.resolve_subnet", return_value="vpcsubnet-auto")


def test_isaac_lab_serverless_requires_output_path(mocker) -> None:
    _mock_isaac_serverless_env(mocker)

    result = runner.invoke(
        app,
        [
            "workbench", "isaac-lab", "-p", "proj", "-n", "isaac", "train",
            "--runtime", "serverless", "--task", "Isaac-Reach-Franka-v0",
        ],
    )

    assert result.exit_code == 1
    assert "requires --output-path" in result.output


def test_isaac_lab_serverless_requires_rt_cores_gpu_type(mocker) -> None:
    _mock_isaac_serverless_env(mocker)
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    client.create_job.return_value = SimpleNamespace(id="job-1", name="isaac-job", status="running", output_uris=())
    mocker.patch("npa.cli.isaac_lab.ServerlessClient", return_value=client)

    result = runner.invoke(
        app,
        [
            "workbench", "isaac-lab", "-p", "proj", "-n", "isaac", "train",
            "--runtime", "serverless", "--task", "Isaac-Reach-Franka-v0",
            "--output-path", "s3://bucket/isaac/", "--submit-only",
            "--gpu-type", "l40s", "--job-name", "isaac-job", "--output-format", "json",
        ],
    )

    assert result.exit_code == 0
    kwargs = client.create_job.call_args.kwargs
    assert kwargs["gpu_type"] == "gpu-l40s-a"
    assert kwargs["preset"] == "1gpu-40vcpu-160gb"


def test_isaac_lab_serverless_rejects_non_rt_gpu_type(mocker) -> None:
    _mock_isaac_serverless_env(mocker)
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    client.create_job.return_value = SimpleNamespace(id="job-1", name="isaac-job", status="running", output_uris=())
    mocker.patch("npa.cli.isaac_lab.ServerlessClient", return_value=client)

    result = runner.invoke(
        app,
        [
            "workbench", "isaac-lab", "-p", "proj", "-n", "isaac", "train",
            "--runtime", "serverless", "--task", "Isaac-Reach-Franka-v0",
            "--output-path", "s3://bucket/isaac/", "--submit-only",
            "--gpu-type", "h200", "--job-name", "isaac-job",
        ],
    )

    assert result.exit_code == 1
    assert "requires RT-core GPUs" in result.output
    client.create_job.assert_not_called()


def test_isaac_lab_serverless_uses_shared_env_builder(mocker) -> None:
    resolver = _mock_isaac_serverless_env(mocker)
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    client.create_job.return_value = SimpleNamespace(id="job-1", name="isaac-job", status="running", output_uris=())
    mocker.patch("npa.cli.isaac_lab.ServerlessClient", return_value=client)

    result = runner.invoke(
        app,
        [
            "workbench", "isaac-lab", "-p", "proj", "-n", "isaac", "train",
            "--runtime", "serverless", "--task", "Isaac-Reach-Franka-v0",
            "--output-path", "s3://bucket/isaac/", "--submit-only",
            "--job-name", "isaac-job", "--output-format", "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["job_id"] == "job-1"
    kwargs = client.create_job.call_args.kwargs
    assert kwargs["env"]["NPA_OUTPUT_PATH"] == "s3://bucket/isaac/"
    assert kwargs["env"]["HF_HOME"] == "/tmp/hf_home"
    assert kwargs["extra_env"]["AWS_ACCESS_KEY_ID"] == "AKIA"
    assert kwargs["extra_env"]["AWS_SECRET_ACCESS_KEY"] == "SECRET"
    resolver.assert_called_once_with(project_id="project-1", explicit_subnet_id="")


def test_isaac_lab_serverless_uploads_output_dir(mocker) -> None:
    _mock_isaac_serverless_env(mocker)
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    client.create_job.return_value = SimpleNamespace(id="job-1", name="isaac-job", status="running", output_uris=())
    mocker.patch("npa.cli.isaac_lab.ServerlessClient", return_value=client)

    result = runner.invoke(
        app,
        [
            "workbench", "isaac-lab", "-p", "proj", "-n", "isaac", "train",
            "--runtime", "serverless", "--task", "Isaac-Reach-Franka-v0",
            "--output-path", "s3://bucket/isaac/", "--submit-only",
            "--job-name", "isaac-job",
        ],
    )

    assert result.exit_code == 0
    command = client.create_job.call_args.kwargs["command"]
    assert "PYUPLOAD" in command
    assert "scripts/reinforcement_learning/rsl_rl/train.py" in command
    assert "--max_iterations \"$MAX_ITERATIONS\"" in command
    assert "agent.save_interval=1" in command
    assert "npa_isaac_lab_train_summary.json" in command
    assert "npa_isaac_lab_checkpoint_manifest.json" in command


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
    assert "scripts/reinforcement_learning/rsl_rl/train.py" in cmd
    assert "--max_iterations \"$MAX_ITERATIONS\"" in cmd
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
            "--max-steps-per-episode",
            "123",
            "--seed",
            "99",
            "--success-metric",
            "survival",
            "--min-success-rate",
            "0.75",
            "--video",
            "--video-length",
            "120",
            "--video-fps",
            "15",
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
    assert "NPA_ISAAC_EVAL_CHECKPOINT" in cmd
    assert "/opt/isaac-lab/runs/model.pt" in cmd
    assert "NPA_ISAAC_EVAL_EPISODES" in cmd
    assert "NPA_ISAAC_EVAL_MAX_STEPS" in cmd
    assert "NPA_ISAAC_EVAL_SEED" in cmd
    assert "NPA_ISAAC_EVAL_SUCCESS_METRIC" in cmd
    assert "NPA_ISAAC_EVAL_MIN_SUCCESS_RATE" in cmd
    assert "NPA_ISAAC_EVAL_VIDEO" in cmd
    assert "NPA_ISAAC_EVAL_VIDEO_LENGTH" in cmd
    assert "NPA_ISAAC_EVAL_VIDEO_FPS" in cmd
    assert "/tmp/isaac-eval" in cmd
    assert "npa.isaac_lab.eval.v1" in cmd
    assert "ISAAC_LAB_EVAL_POLICY_LOAD_FAILED" not in cmd
    assert "random fallback" not in cmd
    assert "ISAAC_LAB_EVAL_EPISODE" in cmd
    assert "ISAAC_LAB_EVAL_COMPLETE" in cmd
    assert "ISAAC_LAB_EVAL_STATUS_VERIFIED" in cmd


def test_isaac_lab_eval_script_has_exact_runtime_options() -> None:
    from npa.cli.isaac_lab import IsaacLabEvalMetric, _build_eval_script

    script = _build_eval_script(
        "Isaac-Reach-Franka-v0",
        "/opt/isaac-lab/runs/model.pt",
        3,
        "/tmp/isaac-eval",
        success_metric=IsaacLabEvalMetric.survival,
        min_success_rate=0.75,
        max_steps_per_episode=123,
        seed=99,
        capture_video=True,
        video_length=120,
        video_fps=15,
    )

    expected = {
        "NPA_ISAAC_EVAL_EPISODES": "3",
        "NPA_ISAAC_EVAL_MAX_STEPS": "123",
        "NPA_ISAAC_EVAL_SEED": "99",
        "NPA_ISAAC_EVAL_SUCCESS_METRIC": "survival",
        "NPA_ISAAC_EVAL_MIN_SUCCESS_RATE": "0.75",
        "NPA_ISAAC_EVAL_VIDEO": "1",
        "NPA_ISAAC_EVAL_VIDEO_LENGTH": "120",
        "NPA_ISAAC_EVAL_VIDEO_FPS": "15",
    }
    for name, value in expected.items():
        assert f"os.environ[{name!r}] = {value!r}" in script


def test_isaac_lab_eval_inline_script_compiles() -> None:
    from npa.cli.isaac_lab import IsaacLabEvalMetric, _build_eval_script

    script = _build_eval_script(
        "Isaac-Velocity-Flat-Anymal-C-v0",
        "/tmp/model.pt",
        2,
        "/tmp/eval",
        success_metric=IsaacLabEvalMetric.survival,
        success_distance_m=0.05,
        min_success_rate=0.8,
        max_steps_per_episode=300,
        seed=123,
    )

    compile(script, "<isaac-lab-eval>", "exec")


@pytest.mark.parametrize(
    ("status", "policy_loaded", "passed", "expected_exit"),
    [("success", True, False, 0), ("failed", False, False, 1)],
)
def test_isaac_lab_eval_status_check_enforces_summary(
    tmp_path: Path,
    status: str,
    policy_loaded: bool,
    passed: bool,
    expected_exit: int,
) -> None:
    from npa.cli.isaac_lab import _build_eval_status_check

    (tmp_path / "npa_isaac_lab_eval_summary.json").write_text(
        json.dumps(
            {
                "status": status,
                "policy_loaded": policy_loaded,
                "passed": passed,
                "success_rate": 0.25,
            }
        )
    )

    completed = subprocess.run(
        ["bash", "-lc", _build_eval_status_check(str(tmp_path))],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == expected_exit
    from npa.cli.isaac_lab import _extract_eval_verdict

    assert _extract_eval_verdict(completed.stdout) == {
        "eval_status": status,
        "passed": passed,
        "policy_loaded": policy_loaded,
        "success_rate": 0.25,
    }


def test_isaac_lab_eval_json_includes_structured_verdict(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (
        0,
        'ISAAC_LAB_EVAL_RESULT_JSON {"eval_status":"success",'
        '"passed":false,"policy_loaded":true,"success_rate":0.25}\n'
        "ISAAC_LAB_EVAL_STATUS_VERIFIED passed=False success_rate=0.25\n",
        "",
    )
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
            "/tmp/model.pt",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["eval_status"] == "success"
    assert payload["policy_loaded"] is True
    assert payload["success_rate"] == 0.25
    assert payload["passed"] is False


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--max-steps-per-episode", "0"], "must be positive"),
        (["--success-distance-m", "0"], "must be positive"),
        (["--min-success-rate", "1.1"], "must be between 0 and 1"),
    ],
)
def test_isaac_lab_eval_rejects_invalid_metric_options(args, message, mocker) -> None:
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    ssh_cls = mocker.patch("npa.cli.isaac_lab.SSHClient")

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "eval",
            "--task",
            "Isaac-Reach-Franka-v0",
            "--checkpoint",
            "/tmp/model.pt",
            *args,
        ],
    )

    assert result.exit_code == 1
    assert message in result.output
    ssh_cls.assert_not_called()


def test_isaac_lab_eval_uploads_failure_summary_directory(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (1, "ISAAC_LAB_EVAL_FAILED", "checkpoint load failed")
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)
    upload = mocker.patch(
        "npa.cli.isaac_lab._upload_remote_directory_to_s3",
        return_value="s3://bucket/eval/",
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "eval",
            "--task",
            "Isaac-Reach-Franka-v0",
            "--checkpoint",
            "/tmp/model.pt",
            "--output-path",
            "s3://bucket/eval/",
        ],
    )

    assert result.exit_code == 1
    upload.assert_called_once()


def test_isaac_lab_s3_checkpoint_prefers_portable_stable_weights(mocker) -> None:
    from npa.cli import isaac_lab

    ssh = mocker.MagicMock()
    storage = mocker.MagicMock()

    def download_path(_uri: str, destination: str) -> None:
        root = Path(destination)
        root.mkdir(parents=True)
        (root / "npa_isaac_lab_checkpoint_manifest.json").write_text(
            json.dumps({"stable_checkpoint_path": "/stale/training/vm/model_1.pt"})
        )
        (root / "npa_isaac_lab_checkpoint.pt").write_bytes(b"weights")

    storage.download_path.side_effect = download_path
    mocker.patch("npa.cli.isaac_lab._storage_client", return_value=storage)

    remote = isaac_lab._prepare_remote_input_path(
        ssh,
        _ssh_cfg(),
        "s3://bucket/isaac-lab/train/",
    )

    assert remote.endswith("/npa_isaac_lab_checkpoint.pt")
    uploaded_local, uploaded_remote = ssh.upload_file.call_args.args
    assert Path(uploaded_local).name == "npa_isaac_lab_checkpoint.pt"
    assert uploaded_remote == remote


def test_isaac_lab_s3_checkpoint_prefers_model_latest_over_numbered(mocker) -> None:
    from npa.cli import isaac_lab

    ssh = mocker.MagicMock()
    storage = mocker.MagicMock()

    def download_path(_uri: str, destination: str) -> None:
        root = Path(destination)
        root.mkdir(parents=True)
        (root / "npa_isaac_lab_checkpoint_manifest.json").write_text(
            json.dumps({"checkpoint_path": "/stale/training/vm/model_500.pt"})
        )
        (root / "model_latest.pt").write_bytes(b"latest")
        (root / "model_500.pt").write_bytes(b"500")
        (root / "model_1500.pt").write_bytes(b"1500")

    storage.download_path.side_effect = download_path
    mocker.patch("npa.cli.isaac_lab._storage_client", return_value=storage)

    remote = isaac_lab._prepare_remote_input_path(
        ssh,
        _ssh_cfg(),
        "s3://bucket/isaac-lab/train/",
    )

    assert remote.endswith("/model_latest.pt")


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


def test_isaac_lab_export_lerobot_runs_remote_rollout_and_uploads(tmp_path: Path, mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "ISAAC_LAB_EXPORT_LEROBOT_COMPLETE", "")
    cfg = _ssh_cfg()
    cfg.runtime = "container"
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=cfg)
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.isaac_lab._download_remote_directory", return_value=tmp_path / "raw")
    storage = mocker.MagicMock()
    storage.upload_directory.return_value = "s3://bucket/isaac-lab/g1/"
    mocker.patch("npa.cli.isaac_lab._storage_client", return_value=storage)
    converted = tmp_path / "converted"
    converted.mkdir()
    convert = mocker.patch("npa.adapter.isaac_lab_lerobot.convert", return_value=converted)

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "export-lerobot",
            "--task",
            "Isaac-Velocity-Flat-G1-v0",
            "--num-episodes",
            "2",
            "--steps-per-episode",
            "4",
            "--output-path",
            "s3://bucket/isaac-lab/g1/",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0
    cmd = ssh.run.call_args.args[0]
    assert "sudo docker exec npa-isaac-lab" in cmd
    assert "/isaac-sim/python.sh" in cmd
    assert "ISAAC_LAB_EXPORT_LEROBOT_START" in cmd
    assert "Isaac-Velocity-Flat-G1-v0" in cmd
    assert "num_episodes = 2" in cmd
    assert "steps_per_episode = 4" in cmd
    convert.assert_called_once()
    assert convert.call_args.kwargs["fps"] == 50
    assert convert.call_args.kwargs["include_placeholder_video"] is True
    storage.upload_directory.assert_called_once_with(str(converted), "s3://bucket/isaac-lab/g1/")
    assert "s3://bucket/isaac-lab/g1/" in result.output


def test_isaac_lab_export_lerobot_falls_back_to_remote_env_upload(
    tmp_path: Path, mocker
) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "ISAAC_LAB_EXPORT_LEROBOT_COMPLETE", "")
    ssh.run_or_raise.return_value = (0, "npa_remote_s3_upload_done files=2", "")
    cfg = _ssh_cfg()
    cfg.runtime = "container"
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=cfg)
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.isaac_lab._download_remote_directory", return_value=tmp_path / "raw")
    storage = mocker.MagicMock()
    storage.upload_directory.side_effect = _access_denied("AccessDenied")
    mocker.patch("npa.cli.isaac_lab._storage_client", return_value=storage)
    converted = tmp_path / "converted"
    converted.mkdir()
    (converted / "meta.json").write_text("{}")
    mocker.patch("npa.adapter.isaac_lab_lerobot.convert", return_value=converted)

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "export-lerobot",
            "--task",
            "Isaac-Velocity-Flat-G1-v0",
            "--output-path",
            "s3://bucket/isaac-lab/g1/",
            "--output-format",
            "json",
            "--allow-host-creds",
        ],
    )

    assert result.exit_code == 0
    assert '"upload_mode": "remote-env"' in result.output
    assert "AccessDenied" in result.output
    ssh.upload_directory.assert_called_once()
    assert ssh.upload_directory.call_args.args[0] == str(converted)
    remote_upload_cmd = ssh.run_or_raise.call_args_list[-1].args[0]
    assert "source /etc/npa-isaac-lab/env" not in remote_upload_cmd
    assert ". /etc/npa-isaac-lab/env" in remote_upload_cmd
    assert "AWS_ACCESS_KEY_ID" in remote_upload_cmd
    assert "AccessDenied" not in remote_upload_cmd


def test_isaac_lab_export_lerobot_rejects_non_s3_output(mocker) -> None:
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    ssh_cls = mocker.patch("npa.cli.isaac_lab.SSHClient")

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "export-lerobot",
            "--task",
            "Isaac-Velocity-Flat-G1-v0",
            "--output-path",
            "/tmp/out",
        ],
    )

    assert result.exit_code == 1
    assert "Isaac Lab export-lerobot --output-path expects an S3 URI" in result.output
    ssh_cls.assert_not_called()


def test_isaac_lab_export_lerobot_maps_remote_failure(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (23, "", "task failed")
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "export-lerobot",
            "--task",
            "Isaac-Velocity-Flat-G1-v0",
            "--output-path",
            "s3://bucket/isaac-lab/g1/",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 1
    assert '"status": "failed"' in result.output
    assert "task failed" in result.output


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


def _onnx_export_result(out_dir: Path) -> dict:
    onnx = out_dir / "policy.onnx"
    contract = out_dir / "policy_contract.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.write_bytes(b"ONNX")
    contract.write_text("{}")
    return {
        "status": "success",
        "onnx_path": str(onnx),
        "contract_path": str(contract),
        "obs_dim": 36,
        "act_dim": 8,
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
        "opset": 17,
        "input_name": "obs",
        "output_name": "action",
        "normalization": "none",
        "isaac_task": "Isaac-Lift-Cube-Franka-v0",
        "checkpoint": {},
    }


def test_isaac_lab_export_onnx_local_to_local(tmp_path: Path, mocker) -> None:
    ckpt = tmp_path / "model_975.pt"
    ckpt.write_bytes(b"x")
    out_dir = tmp_path / "out"

    def fake_export(checkpoint_path, *, out_dir, **kwargs):
        assert checkpoint_path == str(ckpt)
        return _onnx_export_result(Path(out_dir))

    export = mocker.patch(
        "npa.workflows.sim2real.policy_export.export_policy_onnx",
        side_effect=fake_export,
    )
    # Local-only export must NOT touch SSH/storage config.
    ssh_cfg = mocker.patch("npa.cli.isaac_lab._get_ssh_config")

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "export-onnx",
            "--input-path",
            str(ckpt),
            "--output-path",
            str(out_dir),
            "--task",
            "Isaac-Lift-Cube-Franka-v0",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "success"
    assert payload["obs_dim"] == 36
    assert payload["act_dim"] == 8
    assert payload["onnx_path"].endswith("policy.onnx")
    export.assert_called_once()
    assert export.call_args.kwargs["isaac_task"] == "Isaac-Lift-Cube-Franka-v0"
    ssh_cfg.assert_not_called()


def test_isaac_lab_export_onnx_s3_roundtrip_uploads(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.isaac_lab._get_ssh_config", return_value=_ssh_cfg())
    storage = mocker.MagicMock()
    storage.upload_file.side_effect = lambda local, uri: uri
    mocker.patch("npa.cli.isaac_lab._storage_client", return_value=storage)

    def fake_export(checkpoint_path, *, out_dir, **kwargs):
        # The s3 checkpoint was staged to a local temp file first.
        assert Path(checkpoint_path).name == "model.pt"
        return _onnx_export_result(Path(out_dir))

    mocker.patch(
        "npa.workflows.sim2real.policy_export.export_policy_onnx",
        side_effect=fake_export,
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "export-onnx",
            "--input-path",
            "s3://bucket/run/model_975.pt",
            "--output-path",
            "s3://bucket/run/onnx/",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["upload_status"] == "ok"
    storage.download_path.assert_called_once()
    assert storage.upload_file.call_count == 2
    uploaded = {c.args[1] for c in storage.upload_file.call_args_list}
    assert uploaded == {
        "s3://bucket/run/onnx/policy.onnx",
        "s3://bucket/run/onnx/policy_contract.json",
    }


def test_isaac_lab_export_onnx_upload_failure_exits_nonzero(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.isaac_lab._get_ssh_config", return_value=_ssh_cfg())
    storage = mocker.MagicMock()
    storage.upload_file.side_effect = _access_denied("denied")
    mocker.patch("npa.cli.isaac_lab._storage_client", return_value=storage)
    mocker.patch(
        "npa.workflows.sim2real.policy_export.export_policy_onnx",
        side_effect=lambda checkpoint_path, *, out_dir, **kw: _onnx_export_result(Path(out_dir)),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "export-onnx",
            "--input-path",
            "s3://bucket/run/model_975.pt",
            "--output-path",
            "s3://bucket/run/onnx/",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["upload_status"] == "failed"


def test_isaac_lab_export_onnx_export_error_exits_nonzero(tmp_path: Path, mocker) -> None:
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")
    from npa.workflows.sim2real.policy_export import PolicyExportError

    mocker.patch(
        "npa.workflows.sim2real.policy_export.export_policy_onnx",
        side_effect=PolicyExportError("bad checkpoint"),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "export-onnx",
            "--input-path",
            str(ckpt),
            "--output-path",
            str(tmp_path / "out"),
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert "bad checkpoint" in payload["export_error"]


def test_isaac_lab_export_onnx_rejects_bad_opset() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "export-onnx",
            "--input-path",
            "s3://bucket/run/model.pt",
            "--output-path",
            "s3://bucket/run/onnx/",
            "--opset",
            "0",
        ],
    )
    assert result.exit_code == 1
    assert "--opset must be positive" in result.output


def test_isaac_lab_export_onnx_rejects_missing_local_checkpoint(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "export-onnx",
            "--input-path",
            str(tmp_path / "nope.pt"),
            "--output-path",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1
    assert "local checkpoint not found" in result.output


def test_isaac_lab_list_tasks_parses_remote_registry(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (
        0,
        'ISAAC_LAB_LIST_TASKS_JSON {"tasks": ["Isaac-Lift-Cube-Franka-v0", "Isaac-Reach-Franka-v0", "Isaac-Velocity-Flat-G1-v0"], "count": 3}\n',
        "",
    )
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "isaac-lab", "list-tasks"])

    assert result.exit_code == 0
    cmd = ssh.run.call_args.args[0]
    assert "import isaaclab_tasks" in cmd
    assert "gym.registry" in cmd
    assert "Isaac-Lift-Cube-Franka-v0" in result.output
    assert "(3 tasks)" in result.output


def test_isaac_lab_list_tasks_contains_filter_and_json(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (
        0,
        'ISAAC_LAB_LIST_TASKS_JSON {"tasks": ["Isaac-Lift-Cube-Franka-v0", "Isaac-Velocity-Flat-G1-v0"], "count": 2}\n',
        "",
    )
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        ["workbench", "isaac-lab", "list-tasks", "--contains", "franka", "--output-format", "json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["tasks"] == ["Isaac-Lift-Cube-Franka-v0"]
    assert payload["count"] == 1


def test_isaac_lab_list_tasks_fails_cleanly_without_registry(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (3, "", "ModuleNotFoundError: isaaclab_tasks")
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "isaac-lab", "list-tasks"])

    assert result.exit_code != 0
    assert "Failed to list Isaac Lab tasks" in result.output


def test_isaac_lab_train_export_trajectories_runs_second_remote_script(mocker) -> None:
    ssh = mocker.MagicMock()
    # Training call, then the trajectory-export call. The export must emit its
    # completion marker for the CLI to treat it as a real success (Isaac Sim's
    # kit app can exit 0 even when the rollout raised).
    ssh.run.side_effect = [
        (0, "", ""),
        (0, "ISAAC_LAB_TRAJ_EXPORT_COMPLETE\n", ""),
    ]
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
            "--steps",
            "5",
            "--output-dir",
            "/tmp/isaac-out",
            "--export-trajectories",
            "--export-episodes",
            "2",
            "--export-steps-per-episode",
            "10",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert ssh.run.call_count == 2
    traj_cmd = ssh.run.call_args_list[1].args[0]
    assert "ISAAC_LAB_TRAJ_EXPORT_START" in traj_cmd
    assert "npa_isaac_lab_checkpoint.pt" in traj_cmd
    assert "/tmp/isaac-out/trajectories" in traj_cmd
    assert "RslRlVecEnvWrapper(env, clip_actions=clip_actions)" in traj_cmd
    assert "handle_deprecated_rsl_rl_cfg" in traj_cmd
    assert 'metadata.version("rsl-rl-lib")' in traj_cmd
    assert traj_cmd.index("import importlib.metadata as metadata") < traj_cmd.index(
        'metadata.version("rsl-rl-lib")'
    )
    assert "device=runner_device" in traj_cmd
    assert "random fallback" not in traj_cmd
    assert "trained-policy checkpoint load failed" in traj_cmd
    assert "ISAAC_LAB_TRAJ_EXPORT_FAILED" in traj_cmd
    assert "capture_rgb = True" in traj_cmd
    assert "enable_cameras=capture_rgb" in traj_cmd
    assert '"--portable-root /tmp/npa-isaac-kit "' in traj_cmd
    assert '"--/structuredLog/enable=false "' in traj_cmd
    assert '"--/telemetry/enableAnonymousData=false "' in traj_cmd
    assert 'rtx_settings.set_float("/rtx/dataWindowNDC/0", 0.0)' in traj_cmd
    assert 'rtx_settings.set_float("/rtx/dataWindowNDC/3", 1.0)' in traj_cmd
    assert 'rtx_settings.set_bool("/rtx/dataWindow/fitOutputToDataWindow", False)' in traj_cmd
    assert "TiledCameraCfg(" in traj_cmd
    assert 'prim_path="{ENV_REGEX_NS}/NpaRolloutCamera"' in traj_cmd
    assert 'task == "Isaac-Cartpole-v0"' in traj_cmd
    assert "return (0.0, -5.0, 3.0), (0.0, 0.0, 3.0)" in traj_cmd
    assert "camera.set_world_poses_from_view(eyes=eye, targets=target)" in traj_cmd
    assert 'camera = render_env.unwrapped.scene["npa_rollout_camera"]' in traj_cmd
    assert traj_cmd.index("frame = _rgb_frame(render_env)") < traj_cmd.index(
        "obs, _rewards, done, _info = _step_env"
    )
    assert 'np.save(episode_dir / "rgb.npy"' in traj_cmd
    assert "RGB content validation failed" in traj_cmd
    assert "RGB center framing validation failed" in traj_cmd
    assert "RGB motion validation failed" in traj_cmd
    assert '"renderer": "isaac_sim_tiled_camera_rtx"' in traj_cmd
    assert '"rgb_content_frame_count": total_rgb_content_frames' in traj_cmd
    assert '"rgb_center_content_frame_count": total_rgb_center_content_frames' in traj_cmd
    assert '"rgb_motion_pair_count": total_rgb_motion_pairs' in traj_cmd
    assert '"checkpoint_sha256": hashlib.sha256' in traj_cmd
    payload = json.loads(result.output)
    assert payload["trajectory_export"] == "success"
    assert payload["trajectories_dir"] == "/tmp/isaac-out/trajectories"
    assert payload["trajectory_rgb_requested"] is True


def test_isaac_lab_train_can_explicitly_disable_rgb_trajectory_capture(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.side_effect = [
        (0, "", ""),
        (0, "ISAAC_LAB_TRAJ_EXPORT_COMPLETE\n", ""),
    ]
    mocker.patch("npa.cli.isaac_lab.resolve_ssh_config", return_value=_ssh_cfg())
    mocker.patch("npa.cli.isaac_lab.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "train",
            "--task",
            "Isaac-Cartpole-v0",
            "--steps",
            "1",
            "--output-dir",
            "/tmp/isaac-out",
            "--export-trajectories",
            "--no-export-rgb",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    traj_cmd = ssh.run.call_args_list[1].args[0]
    assert "capture_rgb = False" in traj_cmd
    assert "enable_cameras=capture_rgb" in traj_cmd
    assert json.loads(result.output)["trajectory_rgb_requested"] is False


def test_isaac_lab_train_export_trajectories_marks_masked_failure(mocker) -> None:
    """A zero exit without the completion marker must not be reported as success.

    Isaac Sim's kit app can exit 0 even when the Python rollout raised (e.g. the
    env.step tuple mismatch that produced an empty trajectories dir), so the CLI
    must key success off the ISAAC_LAB_TRAJ_EXPORT_COMPLETE marker, not the exit
    code alone.
    """
    ssh = mocker.MagicMock()
    ssh.run.side_effect = [
        (0, "", ""),
        (
            1,
            "ISAAC_LAB_TRAJ_EXPORT_START ...\nPOLICY_LOAD_FAILURE\n",
            "runtime warning",
        ),
    ]
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
            "--steps",
            "5",
            "--output-dir",
            "/tmp/isaac-out",
            "--export-trajectories",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["trajectory_export"] == "failed"
    assert "POLICY_LOAD_FAILURE" in payload["trajectory_export_error"]
    assert "runtime warning" in payload["trajectory_export_error"]


def test_isaac_lab_train_without_export_flag_runs_single_command(mocker) -> None:
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
            "--steps",
            "5",
            "--output-dir",
            "/tmp/isaac-out",
        ],
    )

    assert result.exit_code == 0
    assert ssh.run.call_count == 1


def test_isaac_lab_train_export_trajectories_rejected_on_serverless(mocker) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "isaac-lab",
            "train",
            "--task",
            "Isaac-Reach-Franka-v0",
            "--runtime",
            "serverless",
            "--export-trajectories",
        ],
    )
    assert result.exit_code != 0
    assert "only supported on the VM runtime" in result.output
