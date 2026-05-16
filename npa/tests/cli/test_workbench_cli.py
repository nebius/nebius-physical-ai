from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients import config as config_module
from npa.clients import credentials
from npa.clients.config import DEFAULT_CONTAINER_REGISTRY, SSHConfig, StorageConfig, WorkbenchConfig
from npa.clients.http import ServerError
from npa.clients.ssh import SSHError


runner = CliRunner()


def _cfg(*, storage: bool = False) -> WorkbenchConfig:
    return WorkbenchConfig(
        endpoint="http://vm:8080",
        ssh=SSHConfig(host="vm", user="ubuntu", key_path="~/.ssh/id"),
        storage=StorageConfig(
            checkpoint_bucket="s3://bucket/checkpoints/" if storage else "",
            endpoint_url="https://storage.example" if storage else "",
            aws_access_key_id="key",
            aws_secret_access_key="secret",
        ),
        hf_token="hf-token",
    )


@pytest.mark.parametrize(
    "command",
    [
        "list",
        "status",
        "train",
        "eval",
        "serve",
        "infer",
        "list-checkpoints",
        "deploy",
        "system-info",
        "benchmark",
        "profile-train",
        "train-student",
    ],
)
def test_lerobot_command_help(command: str) -> None:
    result = runner.invoke(app, ["workbench", "lerobot", command, "--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_lerobot_list_filters_to_lerobot_workbenches(mocker) -> None:
    mocker.patch("npa.clients.config.default_project_name", return_value="proj")
    mocker.patch("npa.clients.config.default_workbench_name", return_value="train")
    mocker.patch(
        "npa.clients.config.list_projects",
        return_value={
            "proj": {
                "region": "eu-north1",
                "workbenches": {
                    "train": {
                        "workbench_type": "lerobot",
                        "gpu_platform": "gpu-h100",
                        "endpoint": "http://train:8080",
                    },
                    "sim": {
                        "workbench_type": "genesis",
                        "gpu_platform": "gpu-l40s",
                        "ssh": {"host": "sim"},
                    },
                },
            }
        },
    )

    result = runner.invoke(app, ["workbench", "lerobot", "list"])

    assert result.exit_code == 0
    assert "train" in result.output
    assert "sim" not in result.output


def test_lerobot_list_no_projects_message(mocker) -> None:
    mocker.patch("npa.clients.config.default_project_name", return_value="default")
    mocker.patch("npa.clients.config.default_workbench_name", return_value="default")
    mocker.patch("npa.clients.config.list_projects", return_value={})

    result = runner.invoke(app, ["workbench", "lerobot", "list"])

    assert result.exit_code == 0
    assert "No projects configured" in result.output


def test_lerobot_status_uses_http_client(mocker) -> None:
    http = mocker.MagicMock()
    http.status.return_value = {"policy_server": {"running": False}, "jobs": []}
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.http.HTTPClient", return_value=http)

    result = runner.invoke(app, ["workbench", "lerobot", "status"])

    assert result.exit_code == 0
    assert "server: up" in result.output
    http.status.assert_called_once()


def test_lerobot_status_maps_server_error(mocker) -> None:
    http = mocker.MagicMock()
    http.status.side_effect = ServerError("down")
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.http.HTTPClient", return_value=http)

    result = runner.invoke(app, ["workbench", "lerobot", "status"])

    assert result.exit_code == 1
    assert "Cannot reach server" in result.output


def test_lerobot_train_runs_ssh_command(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "", "")
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "train",
            "--policy-type",
            "act",
            "--dataset",
            "user/ds",
            "--job-name",
            "job",
            "--steps",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "status: success" in result.output
    assert "lerobot-train" in ssh.run.call_args.args[0]


def test_lerobot_train_rejects_local_input_output_paths(mocker) -> None:
    resolve_config = mocker.patch("npa.cli.workbench.lerobot.resolve_config")
    ssh_cls = mocker.patch("npa.clients.ssh.SSHClient")

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "train",
            "--policy-type",
            "act",
            "--input-path",
            "/datasets/pick-place",
            "--job-name",
            "job",
            "--steps",
            "1",
            "--output-path",
            "/runs/student",
        ],
    )

    assert result.exit_code == 1
    assert "LeRobot train --input-path expects an S3 URI" in result.output
    assert "S3 handoff contract" in result.output
    resolve_config.assert_not_called()
    ssh_cls.assert_not_called()


def test_lerobot_train_s3_input_and_output_syncs(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "npa_s3_upload_done", "")
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "train",
            "--policy-type",
            "act",
            "--input-path",
            "s3://bucket/datasets/pick-place/",
            "--job-name",
            "job",
            "--steps",
            "1",
            "--output-path",
            "s3://bucket/checkpoints/job/",
        ],
    )

    assert result.exit_code == 0
    train_cmd = ssh.run.call_args_list[0].args[0]
    upload_cmd = ssh.run.call_args_list[1].args[0]
    assert "download_file" in train_cmd
    assert "--dataset.root=/opt/lerobot/dataset_cache/bucket_datasets_pick-place" in train_cmd
    assert "upload_file" in upload_cmd
    assert "output_path: s3://bucket/checkpoints/job/" in result.output


def test_lerobot_train_rejects_bad_num_workers(mocker) -> None:
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "train",
            "--policy-type",
            "act",
            "--dataset",
            "user/ds",
            "--job-name",
            "job",
            "--num-workers",
            "-2",
        ],
    )

    assert result.exit_code == 1
    assert "num-workers must be -1" in result.output


def test_lerobot_eval_parses_eval_json(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (
        0,
        'logs\n{"overall": {"pc_success": 0.5, "avg_sum_reward": 2, "n_episodes": 4, "eval_s": 1.2}}\n',
        "",
    )
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "eval",
            "--input-path",
            "repo/model",
            "--env",
            "aloha",
        ],
    )

    assert result.exit_code == 0
    assert "pc_success: 0.5" in result.output


def test_lerobot_eval_uses_input_and_output_path(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (
        0,
        '{"overall": {"pc_success": 1.0, "avg_sum_reward": 3, "n_episodes": 2, "eval_s": 0.5}}\n',
        "",
    )
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "eval",
            "--input-path",
            "repo/model",
            "--env",
            "aloha",
            "--output-path",
            "s3://bucket/eval-results/",
        ],
    )

    assert result.exit_code == 0
    cmd = ssh.run.call_args_list[0].args[0]
    assert "--policy.path=repo/model" in cmd
    assert "--output_dir=/tmp/npa-eval-" in cmd
    assert "upload_file" in ssh.run.call_args_list[1].args[0]
    assert "output_path: s3://bucket/eval-results/" in result.output


def test_lerobot_eval_s3_input_and_output_syncs(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (
        0,
        '{"overall": {"pc_success": 1.0, "avg_sum_reward": 3, "n_episodes": 2, "eval_s": 0.5}}\n',
        "",
    )
    ssh.run_or_raise.return_value = (0, "npa_s3_download_done", "")
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "eval",
            "--input-path",
            "s3://bucket/checkpoints/job/",
            "--env",
            "aloha",
            "--output-path",
            "s3://bucket/evals/job/",
        ],
    )

    assert result.exit_code == 0
    assert "download_file" in ssh.run_or_raise.call_args.args[0]
    assert "upload_file" in ssh.run.call_args_list[1].args[0]
    assert "output_path: s3://bucket/evals/job/" in result.output


def test_lerobot_eval_nonzero_exits(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (1, "", "eval failed")
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        ["workbench", "lerobot", "eval", "--input-path", "repo/model", "--env", "aloha"],
    )

    assert result.exit_code == 1
    assert "status: failed" in result.output


def test_lerobot_serve_loads_checkpoint(mocker) -> None:
    http = mocker.MagicMock()
    http.serve.return_value = {"policy_class": "ACTPolicy", "device": "cpu"}
    http.wait_healthy.return_value = True
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.http.HTTPClient", return_value=http)

    result = runner.invoke(
        app,
        ["workbench", "lerobot", "serve", "--input-path", "repo/model"],
    )

    assert result.exit_code == 0
    assert "ACTPolicy" in result.output
    http.serve.assert_called_once_with("repo/model", env_type=None, env_task=None)


def test_lerobot_serve_accepts_deprecated_checkpoint_alias(mocker) -> None:
    http = mocker.MagicMock()
    http.serve.return_value = {"policy_class": "ACTPolicy", "device": "cpu"}
    http.wait_healthy.return_value = True
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.http.HTTPClient", return_value=http)

    result = runner.invoke(
        app,
        ["workbench", "lerobot", "serve", "--checkpoint", "repo/model"],
    )

    assert result.exit_code == 0
    http.serve.assert_called_once_with("repo/model", env_type=None, env_task=None)


def test_lerobot_serve_health_timeout_errors(mocker) -> None:
    http = mocker.MagicMock()
    http.serve.return_value = {}
    http.wait_healthy.return_value = False
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.http.HTTPClient", return_value=http)

    result = runner.invoke(
        app,
        ["workbench", "lerobot", "serve", "--input-path", "repo/model"],
    )

    assert result.exit_code == 1
    assert "did not become healthy" in result.output


def test_lerobot_infer_posts_observation(
    tmp_path: Path, mocker
) -> None:
    obs = tmp_path / "obs.json"
    obs.write_text(json.dumps({"observation.state": [1.0]}))
    http = mocker.MagicMock()
    http.infer.return_value = {"actions": [0.1, 0.2]}
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.http.HTTPClient", return_value=http)

    result = runner.invoke(
        app,
        ["workbench", "lerobot", "infer", "--observation", str(obs)],
    )

    assert result.exit_code == 0
    assert "actions" in result.output
    http.infer.assert_called_once_with({"observation.state": [1.0]})


def test_lerobot_infer_writes_output_path(tmp_path: Path, mocker) -> None:
    obs = tmp_path / "obs.json"
    obs.write_text(json.dumps({"observation.state": [1.0]}))
    output_path = "s3://bucket/infer/infer-response.json"
    http = mocker.MagicMock()
    http.infer.return_value = {"actions": [0.1, 0.2]}
    store = mocker.MagicMock()
    store.upload_file.return_value = output_path
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.http.HTTPClient", return_value=http)
    mocker.patch("npa.clients.storage.StorageClient.from_environment", return_value=store)

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "infer",
            "--observation",
            str(obs),
            "--output-path",
            output_path,
        ],
    )

    assert result.exit_code == 0
    assert f"output_path: {output_path}" in result.output
    store.upload_file.assert_called_once()


def test_lerobot_infer_missing_observation_errors(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())

    result = runner.invoke(
        app,
        ["workbench", "lerobot", "infer", "--observation", str(tmp_path / "missing.json")],
    )

    assert result.exit_code == 1
    assert "Observation file not found" in result.output


def test_lerobot_list_checkpoints_lists_vm_and_storage(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (
        0,
        "/opt/lerobot/checkpoints/job/checkpoints/last/pretrained_model\n",
        "",
    )
    store = mocker.MagicMock()
    store.list_checkpoints.return_value = [{"name": "s3-job", "uri": "s3://bucket/checkpoints/s3-job/"}]
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg(storage=True))
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)
    mocker.patch("npa.clients.storage.StorageClient", return_value=store)

    result = runner.invoke(app, ["workbench", "lerobot", "list-checkpoints"])

    assert result.exit_code == 0
    assert "job" in result.output
    assert "s3-job" in result.output


def test_lerobot_list_checkpoints_config_error_exits(mocker) -> None:
    from npa.clients.config import ConfigError

    mocker.patch(
        "npa.cli.workbench.lerobot.resolve_config",
        side_effect=ConfigError("missing config"),
    )

    result = runner.invoke(app, ["workbench", "lerobot", "list-checkpoints"])

    assert result.exit_code == 1
    assert "missing config" in result.output


def test_lerobot_deploy_dry_run_avoids_infra(mocker) -> None:
    mocker.patch("npa.clients.config.resolve_environment", return_value=None)
    mocker.patch("npa.clients.config.list_projects", return_value={})
    mocker.patch(
        "npa.cli.workbench.lerobot.resolve_container_registry",
        return_value=DEFAULT_CONTAINER_REGISTRY,
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "-p",
            "proj",
            "-n",
            "wb",
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
    assert "http://<pending>:8080" in result.output


def test_lerobot_deploy_runtime_vm_preserves_existing_behavior(tmp_path: Path, mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")

    init = mocker.patch("npa.deploy.provisioner.init")
    apply = mocker.patch(
        "npa.deploy.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.8",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.clients.config.resolve_environment", return_value=None)
    mocker.patch("npa.clients.config.list_projects", return_value={})
    write_config = mocker.patch("npa.clients.config.write_config")
    mocker.patch("npa.cli.workbench.lerobot.update_workbench_app_status")
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)
    install_lerobot = mocker.patch("npa.deploy.configurator.install_lerobot", return_value=True)
    deploy_server = mocker.patch("npa.deploy.configurator.deploy_server")
    deploy_container = mocker.patch("npa.deploy.configurator.deploy_lerobot_container")
    mocker.patch("npa.deploy.configurator.health_check", return_value=True)
    mocker.patch("npa.deploy.configurator.write_manifest")
    mocker.patch(
        "npa.cli.workbench.lerobot.resolve_container_registry",
        return_value=DEFAULT_CONTAINER_REGISTRY,
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "-p",
            "proj",
            "-n",
            "wb",
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
            "vm",
        ],
    )

    assert result.exit_code == 0
    init.assert_called_once_with(tf_dir=str(tmp_path), backend_config=None)
    apply.assert_called_once()
    tf_vars = apply.call_args.kwargs["tf_vars"]
    assert tf_vars["workbench_type"] == "lerobot"
    assert tf_vars["lerobot_version"] == "0.5.1"
    assert "boot_disk_size_gb" not in tf_vars
    install_lerobot.assert_called_once_with(ssh)
    deploy_server.assert_called_once()
    deploy_container.assert_not_called()
    wb_cfg = write_config.call_args.args[0]["projects"]["proj"]["workbenches"]["wb"]
    assert wb_cfg["runtime"] == "vm"


def test_lerobot_deploy_runtime_container_uses_default_registry(tmp_path: Path, mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")
    ssh.run_or_raise.return_value = (0, "true", "")

    apply = mocker.patch(
        "npa.deploy.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.8",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.deploy.provisioner.init")
    mocker.patch("npa.clients.config.resolve_environment", return_value=None)
    mocker.patch("npa.clients.config.list_projects", return_value={})
    write_config = mocker.patch("npa.clients.config.write_config")
    mocker.patch("npa.cli.workbench.lerobot.update_workbench_app_status")
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)
    install_lerobot = mocker.patch("npa.deploy.configurator.install_lerobot")
    deploy_server = mocker.patch("npa.deploy.configurator.deploy_server")
    mocker.patch("npa.deploy.configurator.health_check", return_value=True)
    mocker.patch("npa.deploy.configurator.write_manifest")
    mocker.patch(
        "npa.cli.workbench.lerobot.resolve_container_registry",
        return_value=DEFAULT_CONTAINER_REGISTRY,
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "-p",
            "proj",
            "-n",
            "wb",
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
    install_lerobot.assert_not_called()
    deploy_server.assert_not_called()

    image = f"{DEFAULT_CONTAINER_REGISTRY}/npa-lerobot:0.5.1"
    remote_commands = "\n".join(call.args[0] for call in ssh.run_or_raise.call_args_list)
    assert f"docker pull {image}" in remote_commands
    assert "docker run -d --gpus all --ipc=host --network host" in remote_commands
    assert image in remote_commands
    wb_cfg = write_config.call_args.args[0]["projects"]["proj"]["workbenches"]["wb"]
    assert wb_cfg["runtime"] == "container"


def test_lerobot_deploy_disk_size_overrides_vm_default(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.deploy.provisioner.init")
    apply = mocker.patch(
        "npa.deploy.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.9",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.clients.config.resolve_environment", return_value=None)
    mocker.patch("npa.clients.config.list_projects", return_value={})
    mocker.patch("npa.clients.config.write_config")

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "-p",
            "proj",
            "-n",
            "wb",
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
            "--disk-size",
            "384",
        ],
    )

    assert result.exit_code == 0
    assert apply.call_args.kwargs["tf_vars"]["boot_disk_size_gb"] == "384"


def test_lerobot_deploy_runtime_container_respects_registry_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker,
) -> None:
    cfg_path = tmp_path / ".npa" / "config.yaml"
    creds_path = tmp_path / ".npa" / "credentials.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "proj",
                "projects": {
                    "proj": {
                        "project_id": "project",
                        "tenant_id": "tenant",
                        "region": "eu-north1",
                        "container_registry": "registry.example/private",
                        "workbenches": {},
                    }
                },
            },
            sort_keys=False,
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", creds_path)

    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")
    ssh.run_or_raise.return_value = (0, "true", "")

    mocker.patch("npa.deploy.provisioner.init")
    apply = mocker.patch(
        "npa.deploy.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.8",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.clients.config.write_config")
    mocker.patch("npa.cli.workbench.lerobot.update_workbench_app_status")
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)
    mocker.patch("npa.deploy.configurator.health_check", return_value=True)
    mocker.patch("npa.deploy.configurator.write_manifest")

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "-p",
            "proj",
            "-n",
            "wb",
            "deploy",
            "--tf-dir",
            str(tmp_path),
            "--runtime",
            "container",
            "--disk-size",
            "384",
        ],
    )

    assert result.exit_code == 0
    assert apply.call_args.kwargs["tf_vars"]["boot_disk_size_gb"] == "384"
    remote_commands = "\n".join(call.args[0] for call in ssh.run_or_raise.call_args_list)
    assert "docker pull registry.example/private/npa-lerobot:0.5.1" in remote_commands


def test_lerobot_deploy_rejects_invalid_tf_var() -> None:
    result = runner.invoke(
        app,
        ["workbench", "lerobot", "deploy", "--tf-var", "not-key-value"],
    )

    assert result.exit_code == 1
    assert "Invalid --tf-var format" in result.output


def test_lerobot_system_info_prints_ssh_output(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (0, "gpu info", "")
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "lerobot", "system-info"])

    assert result.exit_code == 0
    assert "gpu info" in result.output


def test_lerobot_system_info_maps_ssh_error(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.side_effect = SSHError("ssh failed")
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "lerobot", "system-info"])

    assert result.exit_code == 1
    assert "ssh failed" in result.output


def test_lerobot_benchmark_runs_single_spec(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (0, "4", "")
    ssh.run.return_value = (0, "", "")
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "benchmark",
            "--run",
            "act:user/ds:1",
            "--num-workers",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Benchmark complete" in result.output


def test_lerobot_benchmark_rejects_bad_run_spec(mocker) -> None:
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=mocker.MagicMock())

    result = runner.invoke(
        app,
        ["workbench", "lerobot", "benchmark", "--run", "bad", "--num-workers", "1"],
    )

    assert result.exit_code == 1
    assert "Invalid --run format" in result.output


def test_lerobot_profile_train_runs_single_spec(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (0, "4", "")
    ssh.run.return_value = (0, "", "")
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "profile-train",
            "--run",
            "act:user/ds:1",
            "--num-workers",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Profile complete" in result.output


def test_lerobot_profile_train_rejects_bad_run_spec(mocker) -> None:
    mocker.patch("npa.cli.workbench.lerobot.resolve_config", return_value=_cfg())
    mocker.patch("npa.clients.ssh.SSHClient", return_value=mocker.MagicMock())

    result = runner.invoke(
        app,
        ["workbench", "lerobot", "profile-train", "--run", "bad"],
    )

    assert result.exit_code == 1
    assert "Invalid --run format" in result.output


def test_lerobot_train_student_dispatches(tmp_path: Path, mocker) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text("{}")
    train_mock = mocker.patch(
        "npa.lerobot.train_student.train_student",
        return_value={
            "status": "success",
            "checkpoint_path": "/tmp/checkpoint",
            "output_dir": "/tmp/out",
        },
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "train-student",
            "--dataset",
            str(dataset),
            "--epochs",
            "1",
            "--output-dir",
            str(tmp_path / "student"),
        ],
    )

    assert result.exit_code == 0
    assert "Student training complete" in result.output
    train_mock.assert_called_once()


def test_lerobot_train_student_uses_input_and_output_path(
    tmp_path: Path, mocker
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text("{}")
    output_uri = "s3://bucket/student/"
    storage = mocker.MagicMock()
    storage.download_directory.return_value = str(dataset)
    storage.upload_directory.return_value = output_uri
    mocker.patch("npa.clients.storage.StorageClient.from_environment", return_value=storage)
    train_mock = mocker.patch(
        "npa.lerobot.train_student.train_student",
        return_value={
            "status": "success",
            "checkpoint_path": str(tmp_path / "student" / "checkpoints" / "last" / "pretrained_model"),
            "output_dir": str(tmp_path / "student"),
        },
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "train-student",
            "--input-path",
            "s3://bucket/dataset/",
            "--epochs",
            "1",
            "--output-path",
            output_uri,
        ],
    )

    assert result.exit_code == 0
    train_mock.assert_called_once()
    assert train_mock.call_args.kwargs["dataset_path"] == dataset
    storage.upload_directory.assert_called_once()


def test_lerobot_train_student_missing_dataset_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "train-student",
            "--dataset",
            str(tmp_path / "missing"),
        ],
    )

    assert result.exit_code == 1
    assert "Dataset not found" in result.output
