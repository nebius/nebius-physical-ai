from __future__ import annotations

import json
import shlex
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli.groot import (
    COSMOS_REASON_REVISION,
    DEFAULT_MODEL,
    GROOT_CONTAINER_ENV_FILE,
    GROOT_CONTAINER_NAME,
    GROOT_DATA_MOUNT,
    GROOT_RUNTIME_VERSION,
    GROOT_RELEASE,
    GROOT_VENV,
    ISAAC_LAB_VERSION,
    _build_download_command,
    _build_finetune_command,
    _build_infer_command,
    _build_install_command,
    _build_offline_eval_command,
    _build_reload_env_command,
    _storage_env_tokens,
)
from npa.cli.main import app
from npa.clients.config import SSHConfig, StorageConfig, WorkbenchConfig
from npa.clients.credentials import CredentialsConfig
from npa.clients.http import ServerError
from npa.clients.serverless import EndpointNotFoundError
from npa.clients.ssh import SSHError


runner = CliRunner()
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _cfg(app_status: str = "", *, runtime: str = "vm", hf_token: str = "") -> WorkbenchConfig:
    return WorkbenchConfig(
        endpoint="http://groot:8080",
        ssh=SSHConfig(host="groot", user="ubuntu", key_path="~/.ssh/id"),
        storage=StorageConfig(
            checkpoint_bucket="s3://bucket/checkpoints/",
            endpoint_url="https://storage.example",
            aws_access_key_id="key",
            aws_secret_access_key="secret",
        ),
        hf_token=hf_token,
        app_status=app_status,
        runtime=runtime,
    )


@contextmanager
def _active_endpoint(url: str):
    yield SimpleNamespace(url=url)


@pytest.mark.parametrize(
    "command",
    [
        "deploy",
        "download",
        "finetune",
        "eval",
        "serve",
        "reload-env",
        "infer",
        "convert",
        "status",
        "system-info",
        "list",
        "cleanup-partial",
    ],
)
def test_groot_command_help(command: str) -> None:
    result = runner.invoke(app, ["workbench", "groot", command, "--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_groot_registered_under_workbench() -> None:
    result = runner.invoke(app, ["workbench", "--help"])

    assert result.exit_code == 0
    assert "groot" in result.output


def test_groot_list_filters_to_groot_workbenches(mocker) -> None:
    mocker.patch("npa.cli.groot.default_project_name", return_value="proj")
    mocker.patch("npa.cli.groot.default_workbench_name", return_value="groot")
    mocker.patch(
        "npa.cli.groot.list_projects",
        return_value={
            "proj": {
                "region": "eu-north1",
                "workbenches": {
                    "groot": {
                        "workbench_type": "groot",
                        "gpu_platform": "gpu-h100-sxm",
                        "endpoint": "http://groot:8080",
                        "app_status": "healthy",
                    },
                    "sim": {
                        "workbench_type": "isaac-lab",
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

    result = runner.invoke(app, ["workbench", "groot", "list"])

    assert result.exit_code == 0
    assert "groot" in result.output
    assert "app_status=healthy" in result.output
    assert "sim" not in result.output
    assert "train" not in result.output


def test_groot_deploy_dry_run_defaults_to_l40s(mocker) -> None:
    mocker.patch("npa.cli.groot.resolve_environment", return_value=None)
    mocker.patch("npa.cli.groot.list_projects", return_value={})
    init = mocker.patch("npa.cli.groot.provisioner.init")
    apply = mocker.patch("npa.cli.groot.provisioner.apply")

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "-p",
            "proj",
            "-n",
            "groot",
            "deploy",
            "--project-id",
            "project",
            "--tenant-id",
            "tenant",
            "--region",
            "eu-north1",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Deploy complete" in result.output
    assert "gpu-l40s-a" in result.output
    assert "http://<pending>:8080" in result.output
    init.assert_not_called()
    apply.assert_not_called()


def test_groot_deploy_passes_workbench_type_to_provisioner(tmp_path: Path, mocker) -> None:
    init = mocker.patch("npa.cli.groot.provisioner.init")
    apply = mocker.patch(
        "npa.cli.groot.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.9",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.groot.resolve_environment", return_value=None)
    mocker.patch("npa.cli.groot.list_projects", return_value={})
    write_config = mocker.patch("npa.cli.groot.write_config")

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "-p",
            "proj",
            "-n",
            "groot",
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
    init.assert_called_once_with(tf_dir=str(tmp_path), backend_config=None)
    apply.assert_called_once()
    tf_vars = apply.call_args.kwargs["tf_vars"]
    assert tf_vars["gpu_platform"] == "gpu-l40s-a"
    assert tf_vars["gpu_preset"] == "1gpu-40vcpu-160gb"
    assert tf_vars["data_disk_size_gb"] == "200"
    assert tf_vars["instance_name"] == "groot-proj-groot"
    assert tf_vars["workbench_type"] == "groot"
    wb_cfg = write_config.call_args.args[0]["projects"]["proj"]["workbenches"]["groot"]
    assert wb_cfg["workbench_type"] == "groot"
    assert wb_cfg["data_disk_size_gb"] == 200
    assert wb_cfg["data_mount"] == GROOT_DATA_MOUNT
    assert wb_cfg["model"] == DEFAULT_MODEL
    assert wb_cfg["app_status"] == "provisioned"


def test_groot_deploy_passes_custom_data_disk_size(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.groot.provisioner.init")
    apply = mocker.patch(
        "npa.cli.groot.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.9",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.groot.resolve_environment", return_value=None)
    mocker.patch("npa.cli.groot.list_projects", return_value={})
    mocker.patch("npa.cli.groot.write_config")

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "-p",
            "proj",
            "-n",
            "groot",
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
            "--data-disk-size",
            "384",
        ],
    )

    assert result.exit_code == 0
    assert apply.call_args.kwargs["tf_vars"]["data_disk_size_gb"] == "384"


def test_groot_deploy_existing_alias_no_replace_uses_idempotent_path(mocker) -> None:
    mocker.patch("npa.cli.groot.resolve_environment", return_value=None)
    mocker.patch("npa.cli.groot.alias_has_terraform_state", return_value=True)
    mocker.patch("npa.cli.groot.workbench_is_byovm", return_value=False)
    update_existing = mocker.patch("npa.cli.groot._update_existing_deployment")
    init = mocker.patch("npa.cli.groot.provisioner.init")
    apply = mocker.patch("npa.cli.groot.provisioner.apply")

    result = runner.invoke(app, ["workbench", "groot", "-p", "proj", "-n", "groot", "deploy"])

    assert result.exit_code == 0
    update_existing.assert_called_once()
    assert update_existing.call_args.kwargs["project"] == "proj"
    assert update_existing.call_args.kwargs["name"] == "groot"
    init.assert_not_called()
    apply.assert_not_called()


def test_groot_deploy_existing_alias_with_replace_prompts_confirmation(mocker) -> None:
    mocker.patch("npa.cli.groot.resolve_environment", return_value=None)
    mocker.patch("npa.cli.groot.alias_has_terraform_state", return_value=True)
    mocker.patch("npa.cli.groot.workbench_is_byovm", return_value=False)
    mocker.patch("npa.cli.groot.typer.confirm", return_value=False)
    init = mocker.patch("npa.cli.groot.provisioner.init")
    apply = mocker.patch("npa.cli.groot.provisioner.apply")

    result = runner.invoke(app, ["workbench", "groot", "-p", "proj", "-n", "groot", "deploy", "--replace"])

    assert result.exit_code == 1
    assert "Aborted" in result.output
    init.assert_not_called()
    apply.assert_not_called()


def test_groot_deploy_existing_alias_with_replace_and_yes_skips_prompt(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.groot.resolve_environment", return_value=None)
    mocker.patch("npa.cli.groot.alias_has_terraform_state", return_value=True)
    mocker.patch("npa.cli.groot.workbench_is_byovm", return_value=False)
    confirm = mocker.patch("npa.cli.groot.typer.confirm")
    mocker.patch("npa.cli.groot.provisioner.init")
    apply = mocker.patch(
        "npa.cli.groot.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.9",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.groot.write_config")
    mocker.patch("npa.cli.groot.list_projects", return_value={})

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "-p",
            "proj",
            "-n",
            "groot",
            "deploy",
            "--replace",
            "--yes",
            "--tf-dir",
            str(tmp_path),
            "--skip-app",
        ],
    )

    assert result.exit_code == 0
    confirm.assert_not_called()
    apply.assert_called_once()


def test_groot_deploy_fresh_alias_runs_terraform(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.groot.resolve_environment", return_value=None)
    mocker.patch("npa.cli.groot.alias_has_terraform_state", return_value=False)
    mocker.patch("npa.cli.groot.workbench_is_byovm", return_value=False)
    mocker.patch("npa.cli.groot.provisioner.init")
    apply = mocker.patch(
        "npa.cli.groot.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.9",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.groot.write_config")
    mocker.patch("npa.cli.groot.list_projects", return_value={})

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "-p",
            "proj",
            "-n",
            "new",
            "deploy",
            "--tf-dir",
            str(tmp_path),
            "--skip-app",
        ],
    )

    assert result.exit_code == 0
    apply.assert_called_once()


def test_groot_deploy_byovm_alias_skips_terraform_regardless(mocker) -> None:
    mocker.patch("npa.cli.groot.resolve_environment", return_value=None)
    mocker.patch("npa.cli.groot.alias_has_terraform_state", return_value=False)
    mocker.patch("npa.cli.groot.workbench_is_byovm", return_value=True)
    update_existing = mocker.patch("npa.cli.groot._update_existing_deployment")
    init = mocker.patch("npa.cli.groot.provisioner.init")
    apply = mocker.patch("npa.cli.groot.provisioner.apply")

    result = runner.invoke(app, ["workbench", "groot", "-p", "proj", "-n", "byovm", "deploy"])

    assert result.exit_code == 0
    update_existing.assert_called_once()
    init.assert_not_called()
    apply.assert_not_called()


def test_groot_deploy_container_runtime_starts_container(tmp_path: Path, mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected\n", "")
    mocker.patch("npa.cli.groot.provisioner.init")
    apply = mocker.patch(
        "npa.cli.groot.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.9",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.groot.resolve_environment", return_value=None)
    mocker.patch("npa.cli.groot.resolve_credentials", return_value=CredentialsConfig(tokens={"NGC_API_KEY": "nvapi"}))
    mocker.patch("npa.cli.groot.list_projects", return_value={})
    write_config = mocker.patch("npa.cli.groot.write_config")
    update_status = mocker.patch("npa.cli.groot.update_workbench_app_status")
    mocker.patch("npa.cli.groot.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.groot.health_check_auto", return_value=(True, ""))
    write_env = mocker.patch("npa.cli.groot.write_remote_docker_env_file")
    mocker.patch("npa.cli.groot.write_manifest")
    deploy_container = mocker.patch("npa.deploy.configurator.deploy_workbench_container")

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "-p",
            "proj",
            "-n",
            "groot-container",
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
    assert tf_vars["workbench_type"] == "groot-container"
    assert tf_vars["data_disk_size_gb"] == "200"
    assert tf_vars["boot_disk_size_gb"] == "250"
    deploy_container.assert_called_once()
    assert deploy_container.call_args.kwargs["container_name"] == GROOT_CONTAINER_NAME
    assert deploy_container.call_args.kwargs["env_file"] == GROOT_CONTAINER_ENV_FILE
    assert deploy_container.call_args.kwargs["image_ref"].endswith(f"/npa-groot:{GROOT_RUNTIME_VERSION}")
    assert f"{GROOT_DATA_MOUNT}:{GROOT_DATA_MOUNT}" in deploy_container.call_args.kwargs["volumes"]
    assert GROOT_DATA_MOUNT in " ".join(deploy_container.call_args.kwargs["work_dirs"])
    write_env.assert_called_once()
    env = write_env.call_args.args[2]
    assert env["GROOT_MODEL_DIR"] == f"{GROOT_DATA_MOUNT}/models"
    assert env["HF_HOME"] == f"{GROOT_DATA_MOUNT}/hf_cache"
    wb_cfg = write_config.call_args_list[0].args[0]["projects"]["proj"]["workbenches"]["groot-container"]
    assert wb_cfg["runtime"] == "container"
    assert wb_cfg["data_mount"] == GROOT_DATA_MOUNT
    assert update_status.call_args_list[0].args == ("proj", "groot-container", "installing")
    assert update_status.call_args_list[-1].args == ("proj", "groot-container", "healthy")


def test_groot_byovm_deploy_skips_terraform_and_records_detected_gpus(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.side_effect = [
        (0, "connected\n", ""),
        (0, "NVIDIA L40S\nNVIDIA L40S\n", ""),
    ]
    ssh_cls = mocker.patch("npa.cli.groot.SSHClient", return_value=ssh)
    init = mocker.patch("npa.cli.groot.provisioner.init")
    apply = mocker.patch("npa.cli.groot.provisioner.apply")
    mocker.patch("npa.cli.groot.resolve_environment", return_value=None)
    mocker.patch("npa.cli.groot.resolve_credentials", return_value=CredentialsConfig())
    mocker.patch("npa.cli.groot.list_projects", return_value={})
    write_config = mocker.patch("npa.cli.groot.write_config")

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "-p",
            "proj",
            "-n",
            "groot",
            "deploy",
            "--runtime",
            "byovm",
            "--host",
            "203.0.113.10",
            "--ssh-key",
            "~/.ssh/byovm",
            "--gpu-count",
            "1",
            "--skip-app",
            "--tf-var",
            "s3_bucket=bucket",
        ],
    )

    assert result.exit_code == 0
    init.assert_not_called()
    apply.assert_not_called()
    ssh_cls.assert_called_once()
    wb_cfg = write_config.call_args.args[0]["projects"]["proj"]["workbenches"]["groot"]
    assert wb_cfg["runtime"] == "byovm"
    assert wb_cfg["ssh"] == {
        "host": "203.0.113.10",
        "user": "ubuntu",
        "key_path": "~/.ssh/byovm",
    }
    assert wb_cfg["gpu_platform"] == "NVIDIA L40S"
    assert wb_cfg["gpu_count"] == 1
    assert wb_cfg["detected_gpu_count"] == 2
    assert wb_cfg["cuda_visible_devices"] == "0"


def test_groot_byovm_deploy_injects_s3_credentials_into_env(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")
    ssh.run_or_raise.side_effect = [
        (0, "connected\n", ""),
        (0, "NVIDIA H200\n", ""),
        (0, "GR00T_ENV_SMOKE_OK\nISAAC_LAB_ENV_SMOKE_OK\n", ""),
    ]
    mocker.patch("npa.cli.groot.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.groot.provisioner.init")
    mocker.patch("npa.cli.groot.provisioner.apply")
    mocker.patch("npa.cli.groot.resolve_environment", return_value=None)
    mocker.patch(
        "npa.cli.groot.resolve_credentials",
        return_value=CredentialsConfig(tokens={"HF_TOKEN": "hf-token"}),
    )
    mocker.patch("npa.cli.groot.list_projects", return_value={})
    mocker.patch("npa.cli.groot.write_config")
    mocker.patch("npa.cli.groot.update_workbench_app_status")
    audit = mocker.patch("npa.cli.groot.audit_remote_env", return_value=[])
    mocker.patch("npa.cli.groot.health_check_auto", return_value=(True, ""))
    mocker.patch("npa.cli.groot.write_manifest")

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "-p",
            "proj",
            "-n",
            "groot",
            "deploy",
            "--runtime",
            "byovm",
            "--host",
            "203.0.113.10",
            "--ssh-key",
            "~/.ssh/byovm",
            "--region",
            "eu-north1",
            "--tf-var",
            "s3_bucket=lerobot-bucket",
            "--tf-var",
            "s3_endpoint=https://storage.example",
            "--tf-var",
            "nebius_api_key=key",
            "--tf-var",
            "nebius_secret_key=secret",
            "--skip-model-check",
            "--verify-env",
        ],
    )

    assert result.exit_code == 0
    install_cmd = next(
        call.args[0]
        for call in ssh.run_or_raise.call_args_list
        if "/etc/npa-groot-server/env" in call.args[0]
    )
    assert "HF_TOKEN=hf-token" in install_cmd
    assert "AWS_ACCESS_KEY_ID=key" in install_cmd
    assert "AWS_SECRET_ACCESS_KEY=secret" in install_cmd
    assert "AWS_ENDPOINT_URL=https://storage.example" in install_cmd
    assert "NEBIUS_S3_ENDPOINT=https://storage.example" in install_cmd
    assert "NEBIUS_S3_BUCKET=lerobot-bucket" in install_cmd
    assert "NGC_API_KEY=" not in install_cmd
    audit.assert_called_once()
    audited_keys = audit.call_args.args[2]
    for key in (
        "HF_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ENDPOINT_URL",
        "NEBIUS_S3_ENDPOINT",
        "NEBIUS_S3_BUCKET",
    ):
        assert key in audited_keys
    assert "Warning: NGC credentials not configured" in result.output


def test_groot_byovm_deploy_injects_ngc_credentials_into_env(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")
    ssh.run_or_raise.side_effect = [
        (0, "connected\n", ""),
        (0, "NVIDIA H200\n", ""),
        (0, "GR00T_ENV_SMOKE_OK\nISAAC_LAB_ENV_SMOKE_OK\n", ""),
    ]
    mocker.patch("npa.cli.groot.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.groot.provisioner.init")
    mocker.patch("npa.cli.groot.provisioner.apply")
    mocker.patch("npa.cli.groot.resolve_environment", return_value=None)
    mocker.patch(
        "npa.cli.groot.resolve_credentials",
        return_value=CredentialsConfig(
            tokens={
                "HF_TOKEN": "hf-token",
                "NGC_API_KEY": "nvapi-file",
                "NGC_ORG": "org-file",
                "NGC_TEAM": "team-file",
            }
        ),
    )
    mocker.patch("npa.cli.groot.list_projects", return_value={})
    mocker.patch("npa.cli.groot.write_config")
    mocker.patch("npa.cli.groot.update_workbench_app_status")
    audit = mocker.patch("npa.cli.groot.audit_remote_env", return_value=[])
    mocker.patch("npa.cli.groot.health_check_auto", return_value=(True, ""))
    mocker.patch("npa.cli.groot.write_manifest")

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "-p",
            "proj",
            "-n",
            "groot",
            "deploy",
            "--runtime",
            "byovm",
            "--host",
            "203.0.113.10",
            "--ssh-key",
            "~/.ssh/byovm",
            "--region",
            "eu-north1",
            "--skip-model-check",
            "--verify-env",
        ],
    )

    assert result.exit_code == 0
    install_cmd = next(
        call.args[0]
        for call in ssh.run_or_raise.call_args_list
        if "/etc/npa-groot-server/env" in call.args[0]
    )
    assert "NGC_API_KEY=nvapi-file" in install_cmd
    assert "NGC_ORG=org-file" in install_cmd
    assert "NGC_TEAM=team-file" in install_cmd
    audited_keys = audit.call_args.args[2]
    assert audited_keys["NGC_API_KEY"] == "nvapi-file"
    assert audited_keys["NGC_ORG"] == "org-file"
    assert audited_keys["NGC_TEAM"] == "team-file"
    assert "Credential audit: NGC credentials merged and written." in result.output
    assert "nvapi-file" not in result.output


def test_groot_deploy_rejects_invalid_data_disk_size() -> None:
    result = runner.invoke(app, ["workbench", "groot", "deploy", "--data-disk-size", "0"])

    assert result.exit_code == 1
    assert "--data-disk-size must be positive" in result.output


def test_groot_install_command_installs_gr00t_and_isaac_lab() -> None:
    cmd = _build_install_command(8080)

    assert "git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git" in cmd
    assert "git -C /opt/groot/Isaac-GR00T checkout 3df8b3825d67f755e69141446f4315f281b9b7e6" in cmd
    assert "expected Isaac-GR00T ref 3df8b3825d67f755e69141446f4315f281b9b7e6" in cmd
    assert "expected gr00t 0.1.0" in cmd
    assert "GROOT_RUNTIME_PIN_PATCH_OK " in cmd
    assert "config.model.model_revision" in cmd
    assert "uv sync --python 3.10" in cmd
    assert "ngccli_linux.zip" in cmd
    assert 'export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"' in cmd
    assert "GR00T_ENV_SMOKE_OK" in cmd
    assert "isaaclab[isaacsim,all]==2.3.2.post1" in cmd
    assert "ISAAC_LAB_ENV_SMOKE_OK" in cmd
    assert "OMNI_KIT_ACCEPT_EULA=YES" in cmd
    assert f"HF_HOME={GROOT_DATA_MOUNT}/hf_cache" in cmd
    assert f'sudo chown -R "$USER:$USER" /opt/groot/ {GROOT_DATA_MOUNT}/' in cmd
    assert "npa-groot-server" in cmd
    assert "gr00t.policy.gr00t_policy import Gr00tPolicy" in cmd


def test_groot_container_dockerfile_pins_runtime_versions() -> None:
    dockerfile = (PACKAGE_ROOT / "docker/groot/Dockerfile").read_text()
    build_script = (PACKAGE_ROOT / "docker/groot/build.sh").read_text()

    assert f"ARG GROOT_RUNTIME_VERSION={GROOT_RUNTIME_VERSION}" in dockerfile
    assert "ARG GROOT_REPO_REF=3df8b3825d67f755e69141446f4315f281b9b7e6" in dockerfile
    assert f"ARG ISAAC_LAB_VERSION={ISAAC_LAB_VERSION}" in dockerfile
    assert f"ARG COSMOS_REASON_REVISION={COSMOS_REASON_REVISION}" in dockerfile
    assert "git -C \"${GROOT_REPO}\" checkout \"${GROOT_REPO_REF}\"" in dockerfile
    assert "isaaclab[isaacsim,all]==${ISAAC_LAB_VERSION}" in dockerfile
    assert "GROOT_MODEL_DIR=/opt/groot-data/models" in dockerfile
    assert "huggingface-cli download nvidia/GR00T-N1.7-3B" not in dockerfile
    assert "--platform linux/amd64" in build_script


def test_groot_install_command_accepts_byovm_gpu_env() -> None:
    cmd = _build_install_command(
        8080,
        env_fields={
            "CUDA_VISIBLE_DEVICES": "0,1",
            "NPA_GPU_COUNT": "2",
        },
    )

    assert "CUDA_VISIBLE_DEVICES=0,1" in cmd
    assert "NPA_GPU_COUNT=2" in cmd


def test_groot_download_command_uses_huggingface_for_current_public_model() -> None:
    cmd = _build_download_command(DEFAULT_MODEL, "/models/groot")

    assert "uv run huggingface-cli download nvidia/GR00T-N1.7-3B" in cmd
    assert "--revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495" in cmd
    assert 'if [ hf = "ngc" ]' in cmd
    assert "NPA_GROOT_DOWNLOAD_COMPLETE" in cmd


def test_groot_download_command_supports_ngc_refs() -> None:
    cmd = _build_download_command("ngc://nvidia/gr00t-n1:1", "s3://bucket/models/groot/")

    assert "ngc registry model download-version nvidia/gr00t-n1:1" in cmd
    assert "apikey = $NGC_API_KEY" in cmd
    assert "org = %s" in cmd
    assert "team = %s" in cmd
    assert "upload_file" in cmd
    assert f"{GROOT_VENV}/bin/python -c" in cmd


def test_groot_download_passes_cli_ngc_token_to_ssh(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (0, "done", "")
    ssh_cls = mocker.patch("npa.cli.groot.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.groot.resolve_ssh_config", return_value=_cfg())
    mocker.patch("npa.cli.groot.resolve_credentials", return_value=CredentialsConfig())

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "download",
            "--model",
            "ngc://nvidia/gr00t-n1:1",
            "--ngc-api-key",
            "nvapi-test",
        ],
    )

    assert result.exit_code == 0
    assert "source: ngc" in result.output
    ssh_cfg = ssh_cls.call_args.args[0]
    assert ssh_cfg.tokens["NGC_API_KEY"] == "nvapi-test"
    assert ssh_cfg.tokens["AWS_ACCESS_KEY_ID"] == "key"
    assert ssh_cfg.tokens["AWS_SECRET_ACCESS_KEY"] == "secret"
    assert ssh_cfg.tokens["NEBIUS_S3_ENDPOINT"] == "https://storage.example"


def test_groot_storage_env_tokens_include_s3_credentials() -> None:
    tokens = _storage_env_tokens(_cfg())

    assert tokens == {
        "AWS_ENDPOINT_URL": "https://storage.example",
        "NEBIUS_S3_ENDPOINT": "https://storage.example",
        "AWS_ACCESS_KEY_ID": "key",
        "AWS_SECRET_ACCESS_KEY": "secret",
    }


def test_groot_reload_env_command_updates_credentials_without_embedding_secret() -> None:
    cmd = _build_reload_env_command(("NGC_API_KEY", "NGC_ORG"), port=8082)

    assert "/etc/npa-groot-server/env" in cmd
    assert "NGC_API_KEY=\"${NGC_API_KEY:-}\"" in cmd
    assert "NGC_ORG=\"${NGC_ORG:-}\"" in cmd
    assert "npa-groot-server" in cmd
    assert "NPA_GROOT_RELOAD_ENV_COMPLETE" in cmd
    assert "nvapi" not in cmd


def test_groot_reload_env_syncs_shared_credentials_and_preserves_loaded_model(mocker) -> None:
    cfg = _cfg(runtime="vm", hf_token="hf-token")
    cfg.service_port = 8082
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (
        0,
        "updated_keys=HF_TOKEN,HUGGING_FACE_HUB_TOKEN,NGC_API_KEY\n"
        "NPA_GROOT_RELOAD_ENV_COMPLETE env_path=/etc/npa-groot-server/env mode=systemd\n",
        "",
    )
    http = mocker.MagicMock()
    http.health.return_value = {
        "status": "ok",
        "loaded": True,
        "loaded_model": DEFAULT_MODEL,
        "embodiment_tag": "REAL_G1",
    }
    http._request.return_value = {"status": "serving"}
    mocker.patch("npa.cli.groot.resolve_config", return_value=cfg)
    mocker.patch(
        "npa.cli.groot.resolve_credentials",
        return_value=CredentialsConfig(
            tokens={
                "HF_TOKEN": "hf-token",
                "NGC_API_KEY": "nvapi-file",
            }
        ),
    )
    ssh_cls = mocker.patch("npa.cli.groot.SSHClient", return_value=ssh)
    mocker.patch(
        "npa.cli.groot.service_endpoint",
        side_effect=lambda *args, **kwargs: _active_endpoint("http://127.0.0.1:19082"),
    )
    mocker.patch("npa.cli.groot.HTTPClient", return_value=http)

    result = runner.invoke(app, ["workbench", "groot", "reload-env", "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "reloaded"
    assert payload["env_path"] == "/etc/npa-groot-server/env"
    assert payload["mode"] == "systemd"
    assert payload["restarted"] is True
    assert "NGC_API_KEY" in payload["updated_keys"]
    assert payload["served"]["model"] == DEFAULT_MODEL
    assert payload["served"]["embodiment_tag"] == "REAL_G1"
    cmd = ssh.run_or_raise.call_args.args[0]
    assert "NGC_API_KEY=\"${NGC_API_KEY:-}\"" in cmd
    assert "nvapi-file" not in cmd
    ssh_tokens = ssh_cls.call_args.args[0].tokens
    assert ssh_tokens["NGC_API_KEY"] == "nvapi-file"
    assert ssh_tokens["HUGGING_FACE_HUB_TOKEN"] == "hf-token"
    http._request.assert_called_once_with(
        "POST",
        "/serve",
        json={"model_path": DEFAULT_MODEL, "embodiment_tag": "REAL_G1", "device": "cuda"},
        timeout=600.0,
    )


def test_groot_apply_env_update_helper_used_by_deploy_and_reload_env(mocker) -> None:
    cfg = _cfg(runtime="vm", hf_token="hf-token")
    cfg.service_port = 8082
    mocker.patch("npa.cli.groot.resolve_environment", return_value=None)
    mocker.patch("npa.cli.groot.alias_has_terraform_state", return_value=True)
    mocker.patch("npa.cli.groot.workbench_is_byovm", return_value=False)
    mocker.patch("npa.cli.groot.resolve_config", return_value=cfg)
    mocker.patch(
        "npa.cli.groot.resolve_credentials",
        return_value=CredentialsConfig(tokens={"HF_TOKEN": "hf-token"}),
    )
    apply_env_update = mocker.patch(
        "npa.cli.groot._apply_env_update",
        return_value={
            "updated_keys": ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"],
            "env_path": "/etc/npa-groot-server/env",
            "mode": "systemd",
            "restarted": True,
            "port": 8082,
        },
    )

    deploy_result = runner.invoke(app, ["workbench", "groot", "-p", "proj", "-n", "groot", "deploy"])
    reload_result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "-p",
            "proj",
            "-n",
            "groot",
            "reload-env",
            "--no-preserve-loaded",
        ],
    )

    assert deploy_result.exit_code == 0
    assert reload_result.exit_code == 0
    assert apply_env_update.call_count == 2


def test_groot_reload_env_dry_run_does_not_apply(mocker) -> None:
    cfg = _cfg(runtime="vm", hf_token="hf-token")
    cfg.service_port = 8082
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (
        0,
        "NPA_GROOT_ENV_READ env_path=/etc/npa-groot-server/env mode=systemd\n"
        "HF_TOKEN=old-token\n",
        "",
    )
    mocker.patch("npa.cli.groot.resolve_config", return_value=cfg)
    mocker.patch(
        "npa.cli.groot.resolve_credentials",
        return_value=CredentialsConfig(tokens={"HF_TOKEN": "hf-token"}),
    )
    mocker.patch("npa.cli.groot.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "groot", "reload-env", "--dry-run"])

    assert result.exit_code == 0
    assert "Dry run" in result.output
    assert "No changes applied" in result.output
    command = ssh.run_or_raise.call_args.args[0]
    assert "NPA_GROOT_ENV_READ" in command
    assert "NPA_GROOT_RELOAD_ENV_COMPLETE" not in command


def test_groot_reload_env_dry_run_shows_changes(mocker) -> None:
    cfg = _cfg(runtime="vm", hf_token="hf-token")
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (
        0,
        "NPA_GROOT_ENV_READ env_path=/etc/npa-groot-server/env mode=systemd\n"
        "HF_TOKEN=old-token\n",
        "",
    )
    mocker.patch("npa.cli.groot.resolve_config", return_value=cfg)
    mocker.patch(
        "npa.cli.groot.resolve_credentials",
        return_value=CredentialsConfig(tokens={"HF_TOKEN": "hf-token"}),
    )
    mocker.patch("npa.cli.groot.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "groot", "reload-env", "--dry-run"])

    assert result.exit_code == 0
    assert "--- current" in result.output
    assert "+++ proposed" in result.output
    assert "-HF_TOKEN=old-" in result.output
    assert "+HF_TOKEN=hf-t" in result.output


def test_groot_reload_env_requires_shared_credentials(mocker) -> None:
    mocker.patch("npa.cli.groot.resolve_config", return_value=_cfg())
    mocker.patch("npa.cli.groot.resolve_credentials", return_value=CredentialsConfig())
    ssh_cls = mocker.patch("npa.cli.groot.SSHClient")

    result = runner.invoke(app, ["workbench", "groot", "reload-env"])

    assert result.exit_code == 1
    assert "No shared credentials found" in result.output
    ssh_cls.assert_not_called()


def test_groot_finetune_s3_paths_build_pytorch_command(mocker) -> None:
    mocker.patch("npa.cli.groot.time.time", return_value=1234.0)
    cmd = _build_finetune_command(
        input_path="s3://bucket/datasets/train/",
        output_path="s3://bucket/checkpoints/groot/",
        base_model=DEFAULT_MODEL,
        robot_embodiment="g1",
        num_gpus=2,
        config="s3://bucket/configs/groot.yaml",
        endpoint_url="https://storage.example",
        max_steps=2,
        global_batch_size=1,
        dataloader_num_workers=0,
        save_steps=1,
        save_total_limit=1,
        save_only_model=True,
    )

    assert "uv run torchrun --nproc_per_node=2" in cmd
    assert "gr00t/experiment/launch_finetune.py" in cmd
    assert "huggingface-cli download nvidia/GR00T-N1.7-3B --revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495" in cmd
    assert f"--base-model-path {GROOT_DATA_MOUNT}/models/nvidia--GR00T-N1.7-3B" in cmd
    assert f"--dataset-path {GROOT_DATA_MOUNT}/data_cache/bucket_datasets_train" in cmd
    assert "--embodiment-tag UNITREE_G1" in cmd
    assert f"modality_config_path={GROOT_DATA_MOUNT}/config_cache/groot.yaml" in cmd
    assert '"${modality_config_arg[@]}"' in cmd
    assert "meta/npa_groot_modality_config.py" in cmd
    assert "--max-steps 2" in cmd
    assert "--global-batch-size 1" in cmd
    assert "--dataloader-num-workers 0" in cmd
    assert "--save-steps 1" in cmd
    assert "--save-total-limit 1" in cmd
    assert "--save-only-model" in cmd
    assert "upload_file" in cmd


def test_groot_finetune_runs_ssh_command(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "NPA_GROOT_FINETUNE_COMPLETE", "")
    mocker.patch("npa.cli.groot.resolve_ssh_config", return_value=_cfg())
    ssh_cls = mocker.patch("npa.cli.groot.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "finetune",
            "--input-path",
            "s3://bucket/datasets/train/",
            "--output-path",
            "s3://bucket/checkpoints/groot/",
            "--robot-embodiment",
            "g1",
            "--num-gpus",
            "1",
            "--max-steps",
            "1",
            "--global-batch-size",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "status: success" in result.output
    assert "robot_embodiment: UNITREE_G1" in result.output
    assert "launch_finetune.py" in ssh.run.call_args.args[0]
    assert ssh_cls.call_args.args[0].tokens["AWS_ACCESS_KEY_ID"] == "key"


def test_groot_finetune_container_runtime_execs_inside_container(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "NPA_GROOT_FINETUNE_COMPLETE", "")
    mocker.patch("npa.cli.groot.resolve_ssh_config", return_value=_cfg(runtime="container"))
    mocker.patch("npa.cli.groot.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "finetune",
            "--input-path",
            "s3://bucket/datasets/train/",
            "--output-path",
            "s3://bucket/checkpoints/groot/",
            "--max-steps",
            "1",
            "--global-batch-size",
            "1",
        ],
    )

    assert result.exit_code == 0
    cmd = ssh.run.call_args.args[0]
    assert "docker exec" in cmd
    assert GROOT_CONTAINER_NAME in cmd
    assert "-e AWS_ACCESS_KEY_ID" in cmd
    assert "launch_finetune.py" in cmd
    assert f"{GROOT_DATA_MOUNT}/data_cache" in cmd


def test_groot_eval_offline_requires_dataset_path(mocker) -> None:
    mocker.patch("npa.cli.groot.resolve_ssh_config", return_value=_cfg())

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "eval",
            "--input-path",
            "s3://bucket/checkpoints/groot/",
            "--output-path",
            "s3://bucket/evals/groot/",
        ],
    )

    assert result.exit_code == 1
    assert "Offline eval requires --dataset-path" in result.output


def test_groot_eval_offline_builds_open_loop_eval_command(mocker) -> None:
    mocker.patch("npa.cli.groot.time.time", return_value=1234.0)
    cmd = _build_offline_eval_command(
        checkpoint_path="s3://bucket/checkpoints/groot/",
        dataset_path="s3://bucket/datasets/heldout/",
        output_path="s3://bucket/evals/groot/",
        robot_embodiment="LIBERO_PANDA",
        endpoint_url="https://storage.example",
    )

    remote_script = shlex.split(cmd)[2]
    assert f"mkdir -p {GROOT_DATA_MOUNT}/outputs/offline-eval-1234" in remote_script
    assert f"eval_plot_path={GROOT_DATA_MOUNT}/outputs/offline-eval-1234/traj_0.jpeg" in remote_script
    assert f"eval_log_path={GROOT_DATA_MOUNT}/outputs/offline-eval-1234/open_loop_eval.log" in remote_script
    assert f"{GROOT_VENV}/bin/python gr00t/eval/open_loop_eval.py" in remote_script
    assert f"{GROOT_VENV}/bin/python - <<'PY'" in remote_script
    assert "\npython - <<'PY'" not in remote_script
    assert f"--dataset-path {GROOT_DATA_MOUNT}/eval_data_cache/bucket_datasets_heldout" in remote_script
    assert f"--model-path {GROOT_DATA_MOUNT}/checkpoint_cache/bucket_checkpoints_groot" in remote_script
    assert "--embodiment-tag LIBERO_PANDA" in remote_script
    assert '--save-plot-path "$eval_plot_path"' in remote_script
    assert "npa_groot_eval_results.json" in remote_script
    assert "Average MSE across all trajs" in remote_script
    assert "upload_file" in remote_script


def test_groot_eval_sim_writes_request_without_ssh(tmp_path: Path, mocker) -> None:
    resolve_ssh = mocker.patch("npa.cli.groot.resolve_ssh_config", return_value=_cfg())
    ssh_cls = mocker.patch("npa.cli.groot.SSHClient")
    mocker.patch("npa.cli.groot.resolve_config", return_value=_cfg())
    captured_request: dict[str, object] = {}
    store = mocker.MagicMock()

    def upload_request(local_path: str, dest: str) -> str:
        captured_request.update(json.loads(Path(local_path).read_text()))
        captured_request["uploaded_to"] = dest
        return dest

    store.upload_file.side_effect = upload_request
    mocker.patch("npa.clients.storage.StorageClient.from_environment", return_value=store)

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "-p",
            "proj",
            "eval",
            "--sim",
            "--isaac-lab-workbench",
            "isaac",
            "--input-path",
            "s3://bucket/checkpoints/groot/",
            "--output-path",
            "s3://bucket/sim-eval/",
            "--num-episodes",
            "3",
            "--robot-embodiment",
            "g1",
        ],
    )

    assert result.exit_code == 0
    data = captured_request
    assert data["type"] == "npa_groot_sim_eval_request_v1"
    assert data["checkpoint_path"] == "s3://bucket/checkpoints/groot/"
    assert data["isaac_lab_workbench"] == "isaac"
    assert data["robot_embodiment"] == "UNITREE_G1"
    assert "does not install or bundle Isaac Lab" in data["note"]
    assert data["uploaded_to"] == "s3://bucket/sim-eval/groot_sim_eval_request.json"
    resolve_ssh.assert_called_once()
    ssh_cls.assert_not_called()


def test_groot_serve_restarts_server_with_model(mocker) -> None:
    ssh = mocker.MagicMock()
    http = mocker.MagicMock()
    http._request.return_value = {"status": "serving"}
    mocker.patch("npa.cli.groot.resolve_config", return_value=_cfg())
    mocker.patch("npa.cli.groot.SSHClient", return_value=ssh)
    http_cls = mocker.patch("npa.cli.groot.HTTPClient", return_value=http)

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "serve",
            "--model",
            DEFAULT_MODEL,
            "--robot-embodiment",
            "g1",
            "--port",
            "9090",
        ],
    )

    assert result.exit_code == 0
    ssh.run_or_raise.assert_not_called()
    http_cls.assert_called_once_with("http://groot:8080", timeout=600.0, retries=1)
    assert http._request.call_args.args[:2] == ("POST", "/serve")
    assert http._request.call_args.kwargs["json"] == {
        "model_path": "nvidia/GR00T-N1.7-3B",
        "embodiment_tag": "UNITREE_G1",
        "device": "cuda",
    }


def test_groot_serve_s3_checkpoint_downloads_first(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (0, "ok", "")
    http = mocker.MagicMock()
    http._request.return_value = {"status": "serving"}
    mocker.patch("npa.cli.groot.resolve_config", return_value=_cfg())
    mocker.patch("npa.cli.groot.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.groot.HTTPClient", return_value=http)

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "serve",
            "--input-path",
            "s3://bucket/checkpoints/groot/",
        ],
    )

    assert result.exit_code == 0
    assert ssh.run_or_raise.call_count == 1
    assert "download_file" in ssh.run_or_raise.call_args_list[0].args[0]
    assert http._request.call_args.kwargs["json"]["model_path"].startswith("/opt/groot-data/checkpoint_cache/")


def test_groot_serve_requires_one_model_source(mocker) -> None:
    mocker.patch("npa.cli.groot.resolve_config", return_value=_cfg())

    result = runner.invoke(app, ["workbench", "groot", "serve"])

    assert result.exit_code == 1
    assert "Provide exactly one of --input-path or --model" in result.output


def test_groot_serve_dry_run_prints_pending_without_ssh(mocker) -> None:
    ssh_cls = mocker.patch("npa.cli.groot.SSHClient")

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "serve",
            "--model",
            DEFAULT_MODEL,
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "status: pending" in result.output
    assert "Would ask the running GR00T server" in result.output
    ssh_cls.assert_not_called()


def test_groot_serve_fails_fast_on_gated_model_error(mocker) -> None:
    http = mocker.MagicMock()
    http._request.side_effect = ServerError(
        "Client error 401: Access to model nvidia/Cosmos-Reason2-2B is restricted"
    )
    mocker.patch("npa.cli.groot.resolve_config", return_value=_cfg())
    mocker.patch("npa.cli.groot.SSHClient", return_value=mocker.MagicMock())
    mocker.patch("npa.cli.groot.HTTPClient", return_value=http)

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "serve",
            "--model",
            DEFAULT_MODEL,
            "--timeout",
            "30",
        ],
    )

    assert result.exit_code == 1
    output = " ".join(result.output.split())
    assert "Model load failed" in result.output
    assert "Access to model nvidia/Cosmos-Reason2-2B is restricted" in output
    assert "Request access at https://huggingface.co/nvidia/GR00T-N1.7-3B" in output
    http._request.assert_called_once()


def test_build_groot_infer_command_runs_standalone_and_uploads_results() -> None:
    cmd = _build_infer_command(
        checkpoint_path="s3://bucket/checkpoints/groot/",
        dataset_path="s3://bucket/data/groot/",
        output_path="s3://bucket/results/infer/",
        embodiment_tag="new",
        inference_mode="pytorch",
        endpoint_url="https://storage.example",
        steps=8,
        action_horizon=4,
        trt_engine_path="./engines",
    )

    assert f"{GROOT_VENV}/bin/python - <<" in cmd
    assert "standalone_inference_script.py" in cmd
    assert "npa_groot_infer_results.json" in cmd
    assert "predicted_actions.npz" in cmd
    assert "npa_s3_download_done" in cmd
    assert "npa_s3_upload_done" in cmd
    assert "inference_mode" in cmd
    assert "pytorch" in cmd
    assert "EmbodimentTag.resolve" in cmd
    assert "NEW_EMBODIMENT" in cmd


def test_groot_infer_runs_remote_batch_inference(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "NPA_GROOT_INFER_COMPLETE\n", "")
    mocker.patch("npa.cli.groot.resolve_ssh_config", return_value=_cfg())
    ssh_cls = mocker.patch("npa.cli.groot.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "infer",
            "--input-path",
            "s3://bucket/checkpoint/",
            "--dataset-path",
            "s3://bucket/dataset/",
            "--output-path",
            "s3://bucket/infer/",
            "--embodiment-tag",
            "new",
            "--inference-mode",
            "pytorch",
            "--steps",
            "8",
            "--action-horizon",
            "4",
        ],
    )

    assert result.exit_code == 0
    ssh_cls.assert_called_once()
    cmd = ssh.run.call_args.args[0]
    assert "standalone_inference_script.py" in cmd
    assert "s3://bucket/checkpoint/" in cmd
    assert "s3://bucket/dataset/" in cmd
    assert "status: success" in result.output
    assert "inference_mode: pytorch" in result.output


def test_groot_infer_rejects_conflicting_cross_project_flags() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "infer",
            "--input-path",
            "s3://bucket/checkpoint/",
            "--dataset-path",
            "s3://bucket/dataset/",
            "--output-path",
            "s3://bucket/infer/",
            "--source-project",
            "project-a",
            "--target-project",
            "project-b",
        ],
    )

    assert result.exit_code == 1
    assert "single credential" in result.output
    assert "NOVEL_ISSUE_E6_AUTH_SCOPE" in result.output
    assert "demo stage" in result.output


@pytest.mark.parametrize(
    ("project_flags", "expected_project"),
    [
        (["--source-project", "project-a"], "project-a"),
        (["--target-project", "project-b"], "project-b"),
        (
            ["--source-project", "project-a", "--target-project", "project-a"],
            "project-a",
        ),
    ],
)
def test_groot_infer_accepts_single_remote_storage_project(
    mocker, project_flags: list[str], expected_project: str
) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "NPA_GROOT_INFER_COMPLETE\n", "")
    mocker.patch("npa.cli.groot.resolve_ssh_config", return_value=_cfg())
    ssh_cls = mocker.patch("npa.cli.groot.SSHClient", return_value=ssh)
    storage_env = mocker.patch(
        "npa.cli.groot.storage_env_for_project",
        return_value={"AWS_ACCESS_KEY_ID": "scoped-key"},
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "infer",
            "--input-path",
            "s3://bucket/checkpoint/",
            "--dataset-path",
            "s3://bucket/dataset/",
            "--output-path",
            "s3://bucket/infer/",
            *project_flags,
        ],
    )

    assert result.exit_code == 0
    storage_env.assert_called_once_with(expected_project)
    ssh_config = ssh_cls.call_args.args[0]
    assert ssh_config.tokens["AWS_ACCESS_KEY_ID"] == "scoped-key"


def test_groot_infer_rejects_invalid_steps() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "infer",
            "--input-path",
            "s3://bucket/checkpoint/",
            "--dataset-path",
            "s3://bucket/dataset/",
            "--output-path",
            "s3://bucket/infer/",
            "--steps",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "--steps must be positive" in result.output


def _mock_groot_serverless_env(mocker) -> None:
    mocker.patch("npa.cli.groot.resolve_environment", return_value=SimpleNamespace(project_id="project-1"))
    mocker.patch(
        "npa.cli.groot.resolve_project_storage",
        return_value=SimpleNamespace(
            checkpoint_bucket="",
            endpoint_url="https://s3.example",
            aws_access_key_id="AKIA",
            aws_secret_access_key="SECRET",
        ),
    )
    mocker.patch("npa.cli.groot.resolve_container_registry", return_value="registry.example")
    mocker.patch("npa.cli.groot.container_image_for_tool", return_value="registry.example/npa-groot:smoke")
    mocker.patch("npa.cli.groot._serverless_subnet_id", return_value="vpcsubnet-auto")


def test_groot_serverless_requires_output_path() -> None:
    result = runner.invoke(
        app,
        [
            "workbench", "groot", "infer",
            "--runtime", "serverless",
            "--input-path", "s3://bucket/checkpoint/",
            "--dataset-path", "s3://bucket/dataset/",
            "--output-path", "file:///tmp/out",
        ],
    )

    assert result.exit_code == 1
    assert "expects an S3 URI" in result.output


def test_groot_serverless_uses_shared_env_builder(mocker) -> None:
    _mock_groot_serverless_env(mocker)
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    client.create_job.return_value = SimpleNamespace(id="job-1", name="groot-job", status="running", output_uris=())
    mocker.patch("npa.cli.groot.ServerlessClient", return_value=client)

    result = runner.invoke(
        app,
        [
            "workbench", "groot", "infer",
            "--runtime", "serverless",
            "--input-path", "s3://bucket/checkpoint/",
            "--dataset-path", "s3://bucket/dataset/",
            "--output-path", "s3://bucket/groot/",
            "--submit-only", "--job-name", "groot-job", "--output", "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["job_id"] == "job-1"
    kwargs = client.create_job.call_args.kwargs
    assert kwargs["env"]["NPA_OUTPUT_PATH"] == "s3://bucket/groot/"
    assert kwargs["env"]["HF_HOME"] == "/tmp/hf_home"
    assert kwargs["extra_env"]["AWS_ACCESS_KEY_ID"] == "AKIA"
    assert kwargs["extra_env"]["AWS_SECRET_ACCESS_KEY"] == "SECRET"


def test_groot_serverless_with_model_variant_arg(mocker) -> None:
    _mock_groot_serverless_env(mocker)
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    client.create_job.return_value = SimpleNamespace(id="job-1", name="groot-job", status="running", output_uris=())
    mocker.patch("npa.cli.groot.ServerlessClient", return_value=client)

    result = runner.invoke(
        app,
        [
            "workbench", "groot", "infer",
            "--runtime", "serverless",
            "--input-path", "s3://bucket/checkpoint/",
            "--dataset-path", "s3://bucket/dataset/",
            "--output-path", "s3://bucket/groot/",
            "--model-variant", "nvidia/GR00T-N1.7-3B",
            "--submit-only", "--job-name", "groot-job",
        ],
    )

    assert result.exit_code == 0
    command = client.create_job.call_args.kwargs["command"]
    assert "nvidia/GR00T-N1.7-3B" in command
    assert "PYUPLOAD" in command


def test_groot_convert_dispatches_lerobot_to_groot(tmp_path: Path, mocker) -> None:
    input_dir = tmp_path / "lerobot"
    converted_dir = tmp_path / "groot"
    input_dir.mkdir()
    converted_dir.mkdir()
    storage = mocker.MagicMock()
    storage.download_directory.return_value = str(input_dir)
    storage.upload_directory.return_value = "s3://bucket/groot/"
    mocker.patch("npa.cli.groot._storage_client_for_project_or_environment", return_value=storage)
    convert_mock = mocker.patch(
        "npa.adapter.groot.lerobot_to_groot",
        return_value=converted_dir,
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "convert",
            "--input-path",
            "s3://bucket/lerobot/",
            "--output-path",
            "s3://bucket/groot/",
            "--direction",
            "lerobot-to-groot",
            "--robot-embodiment",
            "new",
        ],
    )

    assert result.exit_code == 0
    assert convert_mock.call_args.args[0] == input_dir
    assert convert_mock.call_args.kwargs["robot_embodiment"] == "NEW_EMBODIMENT"
    storage.upload_directory.assert_called_once_with(str(converted_dir), "s3://bucket/groot/")
    assert "status: converted" in result.output


def test_groot_convert_accepts_real_g1_embodiment_tag_alias(tmp_path: Path, mocker) -> None:
    input_dir = tmp_path / "lerobot"
    converted_dir = tmp_path / "groot"
    input_dir.mkdir()
    converted_dir.mkdir()
    storage = mocker.MagicMock()
    storage.download_directory.return_value = str(input_dir)
    storage.upload_directory.return_value = "s3://bucket/groot/"
    mocker.patch("npa.cli.groot._storage_client_for_project_or_environment", return_value=storage)
    convert_mock = mocker.patch(
        "npa.adapter.groot.lerobot_to_groot",
        return_value=converted_dir,
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "convert",
            "--input-path",
            "s3://bucket/lerobot/",
            "--output-path",
            "s3://bucket/groot/",
            "--embodiment-tag",
            "REAL_G1",
        ],
    )

    assert result.exit_code == 0
    assert convert_mock.call_args.kwargs["robot_embodiment"] == "REAL_G1"
    storage.upload_directory.assert_called_once_with(str(converted_dir), "s3://bucket/groot/")


def test_groot_convert_dispatches_groot_to_lerobot(tmp_path: Path, mocker) -> None:
    input_dir = tmp_path / "groot"
    converted_dir = tmp_path / "lerobot"
    input_dir.mkdir()
    converted_dir.mkdir()
    storage = mocker.MagicMock()
    storage.download_directory.return_value = str(input_dir)
    storage.upload_directory.return_value = "s3://bucket/lerobot/"
    mocker.patch("npa.cli.groot._storage_client_for_project_or_environment", return_value=storage)
    convert_mock = mocker.patch(
        "npa.adapter.groot.groot_to_lerobot",
        return_value=converted_dir,
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "convert",
            "--input-path",
            "s3://bucket/groot/",
            "--output-path",
            "s3://bucket/lerobot/",
            "--direction",
            "groot-to-lerobot",
        ],
    )

    assert result.exit_code == 0
    assert convert_mock.call_args.args[0] == input_dir
    storage.upload_directory.assert_called_once_with(str(converted_dir), "s3://bucket/lerobot/")


def test_groot_convert_s3_does_not_require_deployed_workbench(tmp_path: Path, mocker) -> None:
    input_dir = tmp_path / "downloaded"
    input_dir.mkdir()
    converted_dir = tmp_path / "converted"
    converted_dir.mkdir()
    storage = mocker.MagicMock()
    storage.download_directory.return_value = str(input_dir)
    storage.upload_directory.return_value = "s3://bucket/out/"
    mocker.patch("npa.cli.groot._storage_client_for_project_or_environment", return_value=storage)
    resolve_config = mocker.patch("npa.cli.groot.resolve_config")
    convert_mock = mocker.patch(
        "npa.adapter.groot.lerobot_to_groot",
        return_value=converted_dir,
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "groot",
            "-p",
            "proj",
            "-n",
            "not-deployed-yet",
            "convert",
            "--input-path",
            "s3://bucket/in/",
            "--output-path",
            "s3://bucket/out/",
        ],
    )

    assert result.exit_code == 0
    resolve_config.assert_not_called()
    storage.download_directory.assert_called_once()
    storage.upload_directory.assert_called_once_with(str(converted_dir), "s3://bucket/out/")
    convert_mock.assert_called_once()


def test_groot_status_checks_health_endpoint(mocker) -> None:
    http = mocker.MagicMock()
    http.health.return_value = {"status": "ok", "groot_version": "0.1.0"}
    mocker.patch("npa.cli.groot.resolve_config", return_value=_cfg("healthy"))
    http_cls = mocker.patch("npa.cli.groot.HTTPClient", return_value=http)

    result = runner.invoke(app, ["workbench", "groot", "status"])

    assert result.exit_code == 0
    assert "server: up" in result.output
    assert "groot_version: 0.1.0" in result.output
    http_cls.assert_called_once_with("http://groot:8080", timeout=10.0, retries=1)


def test_groot_status_uses_recorded_ssh_endpoint_strategy(mocker) -> None:
    cfg = _cfg("healthy")
    cfg.endpoint_strategy = "ssh"
    cfg.service_port = 8082
    http = mocker.MagicMock()
    http.health.return_value = {"status": "ok", "groot_version": "0.1.0"}
    mocker.patch("npa.cli.groot.resolve_config", return_value=cfg)
    endpoint = mocker.patch(
        "npa.cli.groot.service_endpoint",
        return_value=_active_endpoint("http://127.0.0.1:19082"),
    )
    http_cls = mocker.patch("npa.cli.groot.HTTPClient", return_value=http)

    result = runner.invoke(app, ["workbench", "groot", "status"])

    assert result.exit_code == 0
    endpoint.assert_called_once_with(cfg, default_port=8080)
    http_cls.assert_called_once_with("http://127.0.0.1:19082", timeout=10.0, retries=1)


def test_groot_status_reports_readiness_blockers(mocker) -> None:
    http = mocker.MagicMock()
    http.health.return_value = {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "loaded": False,
        "ngc_credentials_configured": False,
    }
    mocker.patch("npa.cli.groot.resolve_config", return_value=_cfg(hf_token="hf-token"))
    mocker.patch("npa.cli.groot.HTTPClient", return_value=http)

    result = runner.invoke(app, ["workbench", "groot", "status", "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["app_status"] == "degraded"
    assert payload["reason"] == "model not loaded"
    assert payload["readiness"]["hf_token_present"] is True
    assert payload["readiness"]["ngc_credentials_configured"] is False
    assert payload["readiness"]["model_loaded"] is False
    assert payload["readiness"]["ready"] is False
    assert "NGC credentials not configured" in payload["readiness"]["blockers"]
    assert f"Model {DEFAULT_MODEL} not loaded" in payload["readiness"]["blockers"]


def test_groot_status_maps_server_error(mocker) -> None:
    http = mocker.MagicMock()
    http.health.side_effect = ServerError("down")
    mocker.patch("npa.cli.groot.resolve_config", return_value=_cfg("install_failed"))
    mocker.patch("npa.cli.groot.HTTPClient", return_value=http)

    result = runner.invoke(app, ["workbench", "groot", "status"])

    assert result.exit_code == 1
    assert "app_status: unreachable" in result.output
    assert "Cannot reach GR00T endpoint" in result.output


def test_groot_system_info_prints_ssh_output(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (0, "gr00t_version: 0.1.0\nngc_credentials_configured: True", "")
    mocker.patch("npa.cli.groot.resolve_ssh_config", return_value=_cfg())
    mocker.patch("npa.cli.groot.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "groot", "system-info"])

    assert result.exit_code == 0
    assert "gr00t_version: 0.1.0" in result.output
    cmd = ssh.run_or_raise.call_args.args[0]
    assert "nvidia-smi" in cmd
    assert "lscpu" in cmd
    assert "metadata.version('gr00t')" in cmd
    assert "ngc_credentials_configured" in cmd
    assert "metadata.version('isaaclab')" in cmd


def test_groot_release_constant_documented() -> None:
    assert GROOT_RELEASE == "n1.7"
