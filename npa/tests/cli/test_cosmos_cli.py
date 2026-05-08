from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.cosmos import (
    COSMOS_FLASH_ATTN_VERSION,
    COSMOS_FLASH_ATTN_WHEEL_URL,
    COSMOS_NATTEN_VERSION,
    COSMOS_NATTEN_WHEEL_URL,
    COSMOS_PEFT_MIN_VERSION,
    COSMOS_PIP_EXTRA_INDEX_URL,
    COSMOS_TORCH_VERSION,
    COSMOS_TORCHVISION_VERSION,
    COSMOS_VERSION,
    _build_install_command,
)
from npa.clients.config import SSHConfig, StorageConfig, WorkbenchConfig
from npa.clients.http import ServerError
from npa.clients.ssh import SSHError


runner = CliRunner()


def _cfg(app_status: str = "") -> WorkbenchConfig:
    return WorkbenchConfig(
        endpoint="http://cosmos:8080",
        ssh=SSHConfig(host="cosmos", user="ubuntu", key_path="~/.ssh/id"),
        storage=StorageConfig(checkpoint_bucket="", endpoint_url=""),
        app_status=app_status,
    )


@pytest.mark.parametrize(
    "command",
    [
        "deploy",
        "serve",
        "infer",
        "finetune",
        "optimize",
        "status",
        "system-info",
        "list",
    ],
)
def test_cosmos_command_help(command: str) -> None:
    result = runner.invoke(app, ["workbench", "cosmos", command, "--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cosmos_registered_under_workbench() -> None:
    result = runner.invoke(app, ["workbench", "--help"])

    assert result.exit_code == 0
    assert "cosmos" in result.output


def test_cosmos_placeholder_help_describes_roadmap() -> None:
    finetune = runner.invoke(app, ["workbench", "cosmos", "finetune", "--help"])
    optimize = runner.invoke(app, ["workbench", "cosmos", "optimize", "--help"])

    assert finetune.exit_code == 0
    assert "LoRA" in finetune.output
    assert "full fine-tuning" in finetune.output
    assert "custom" in finetune.output
    assert optimize.exit_code == 0
    assert "TensorRT" in optimize.output
    assert "quantization" in optimize.output


def test_cosmos_backend_help_describes_choices() -> None:
    deploy = runner.invoke(app, ["workbench", "cosmos", "deploy", "--help"])
    serve = runner.invoke(app, ["workbench", "cosmos", "serve", "--help"])

    assert deploy.exit_code == 0
    assert "basic" in deploy.output
    assert "NIM" in deploy.output
    assert "Triton" in deploy.output
    assert "TensorRT" in deploy.output
    assert serve.exit_code == 0
    assert "basic" in serve.output
    assert "NIM" in serve.output
    assert "Triton" in serve.output
    assert "TensorRT" in serve.output


@pytest.mark.parametrize("command", ["finetune", "optimize"])
def test_cosmos_placeholders_exit_not_implemented(command: str) -> None:
    result = runner.invoke(app, ["workbench", "cosmos", command])

    assert result.exit_code == 1
    assert result.output.strip() == "not yet implemented"


@pytest.mark.parametrize("backend", ["nim", "triton"])
def test_cosmos_deploy_rejects_unimplemented_backends(
    backend: str,
    tmp_path: Path,
    mocker,
) -> None:
    apply = mocker.patch("npa.cli.cosmos.provisioner.apply")

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "deploy",
            "--backend",
            backend,
            "--gpu-type",
            "gpu-h100-sxm",
            "--gpu-preset",
            "1gpu-16vcpu-200gb",
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
    assert result.output.strip() == "NIM/Triton backend is not yet implemented"
    apply.assert_not_called()


@pytest.mark.parametrize("backend", ["nim", "triton"])
def test_cosmos_serve_rejects_unimplemented_backends(backend: str, mocker) -> None:
    resolve_config = mocker.patch("npa.cli.cosmos.resolve_config")

    result = runner.invoke(
        app,
        ["workbench", "cosmos", "serve", "--backend", backend],
    )

    assert result.exit_code == 1
    assert result.output.strip() == "NIM/Triton backend is not yet implemented"
    resolve_config.assert_not_called()


def test_cosmos_deploy_requires_gpu_selection(tmp_path: Path) -> None:
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
        ],
    )

    assert result.exit_code == 1
    assert "GPU selection is required" in result.output
    assert "7B Text2World" in result.output


def test_cosmos_deploy_passes_gpu_selection_to_provisioner(tmp_path: Path, mocker) -> None:
    init = mocker.patch("npa.cli.cosmos.provisioner.init")
    apply = mocker.patch(
        "npa.cli.cosmos.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.7",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.cosmos.resolve_environment", return_value=None)
    mocker.patch("npa.cli.cosmos.list_projects", return_value={})
    write_config = mocker.patch("npa.cli.cosmos.write_config")

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
            "--skip-app",
        ],
    )

    assert result.exit_code == 0
    assert "Deploy complete" in result.output
    init.assert_called_once_with(tf_dir=str(tmp_path), backend_config=None)
    apply.assert_called_once()
    tf_vars = apply.call_args.kwargs["tf_vars"]
    assert tf_vars["gpu_platform"] == "gpu-h100-sxm"
    assert tf_vars["gpu_preset"] == "1gpu-16vcpu-200gb"
    assert tf_vars["instance_name"] == "cosmos-proj-cosmos"
    assert tf_vars["workbench_type"] == "cosmos"
    assert "boot_disk_size_gb" not in tf_vars
    write_config.assert_called_once()
    wb_cfg = write_config.call_args.args[0]["projects"]["proj"]["workbenches"]["cosmos"]
    assert wb_cfg["app_status"] == "provisioned"


def test_cosmos_deploy_runtime_container_starts_image(tmp_path: Path, mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")
    ssh.run_or_raise.return_value = (0, "", "")

    mocker.patch("npa.cli.cosmos.provisioner.init")
    apply = mocker.patch(
        "npa.cli.cosmos.provisioner.apply",
        return_value={
            "vm_ip": "10.0.0.36",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.cosmos.SSHClient", return_value=ssh)
    mocker.patch("npa.cli.cosmos.resolve_environment", return_value=None)
    mocker.patch("npa.cli.cosmos.resolve_credentials", return_value=SimpleNamespace(hf_token="hf-token", tokens={}))
    mocker.patch("npa.cli.cosmos.list_projects", return_value={})
    write_config = mocker.patch("npa.cli.cosmos.write_config")
    update_status = mocker.patch("npa.cli.cosmos.update_workbench_app_status")
    deploy_container = mocker.patch("npa.deploy.configurator.deploy_workbench_container")
    mocker.patch("npa.deploy.configurator.write_remote_env_file")
    mocker.patch("npa.cli.cosmos.health_check", return_value=True)
    mocker.patch("npa.cli.cosmos.write_manifest")

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "-p",
            "proj",
            "-n",
            "cosmos-container",
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
            "--runtime",
            "container",
        ],
    )

    assert result.exit_code == 0
    tf_vars = apply.call_args.kwargs["tf_vars"]
    assert tf_vars["workbench_type"] == "cosmos"
    assert tf_vars["boot_disk_size_gb"] == "250"
    deploy_container.assert_called_once()
    assert deploy_container.call_args.kwargs["container_name"] == "npa-cosmos"
    assert deploy_container.call_args.kwargs["image_ref"].endswith("/npa-cosmos:1.0.9")
    wb_cfg = write_config.call_args.args[0]["projects"]["proj"]["workbenches"]["cosmos-container"]
    assert wb_cfg["runtime"] == "container"
    assert update_status.call_args_list[0].args == ("proj", "cosmos-container", "installing")
    assert update_status.call_args_list[-1].args == ("proj", "cosmos-container", "healthy")


def test_cosmos_install_command_installs_torch_before_flash_attn_and_cosmos() -> None:
    cmd = _build_install_command("nvidia/Cosmos-Test", 8080)

    torch_install = (
        f'/opt/cosmos/venv/bin/python -m pip install "torch=={COSMOS_TORCH_VERSION}" '
        f'"torchvision=={COSMOS_TORCHVISION_VERSION}" '
        f"--extra-index-url {COSMOS_PIP_EXTRA_INDEX_URL}"
    )
    flash_attn_install = (
        '/opt/cosmos/venv/bin/python -m pip install --no-deps "$flash_attn_wheel"'
    )
    flash_attn_wheel = f'flash_attn_wheel="/tmp/flash_attn-{COSMOS_FLASH_ATTN_VERSION}-cp310-cp310-linux_x86_64.whl"'
    flash_attn_download = f'curl -L -o "$flash_attn_wheel" "{COSMOS_FLASH_ATTN_WHEEL_URL}"'
    natten_wheel = f'natten_wheel="/tmp/natten-{COSMOS_NATTEN_VERSION}-cp310-cp310-linux_x86_64.whl"'
    natten_download = f'curl -L -o "$natten_wheel" "{COSMOS_NATTEN_WHEEL_URL}"'
    natten_install = '/opt/cosmos/venv/bin/python -m pip install --no-deps "$natten_wheel"'
    cosmos_install = (
        f'/opt/cosmos/venv/bin/python -m pip install "cosmos-predict2[cu126]=={COSMOS_VERSION}" '
        f"--extra-index-url {COSMOS_PIP_EXTRA_INDEX_URL}"
    )
    server_extras_install = (
        f'/opt/cosmos/venv/bin/python -m pip install "diffusers>=0.34.0" '
        f'"peft>={COSMOS_PEFT_MIN_VERSION}"'
    )
    guardrail_install = "/opt/cosmos/venv/bin/python -m pip install --no-deps cosmos_guardrail"

    assert torch_install in cmd
    assert flash_attn_wheel in cmd
    assert flash_attn_download in cmd
    assert flash_attn_install in cmd
    assert natten_wheel in cmd
    assert natten_download in cmd
    assert natten_install in cmd
    assert cosmos_install in cmd
    assert server_extras_install in cmd
    assert guardrail_install in cmd
    assert (
        cmd.index(torch_install)
        < cmd.index(flash_attn_wheel)
        < cmd.index(flash_attn_download)
        < cmd.index(flash_attn_install)
        < cmd.index(natten_wheel)
        < cmd.index(natten_download)
        < cmd.index(natten_install)
        < cmd.index(cosmos_install)
        < cmd.index(server_extras_install)
        < cmd.index(guardrail_install)
    )


def test_cosmos_install_command_uses_data_disk_for_models_and_cache() -> None:
    cmd = _build_install_command("nvidia/Cosmos-Test", 8080)

    assert "/opt/cosmos-data/models" in cmd
    assert "/opt/cosmos-data/hf_cache" in cmd
    assert "/opt/cosmos-data/outputs" in cmd
    assert "export HF_HOME=/opt/cosmos-data/hf_cache" in cmd
    assert "export HUGGINGFACE_HUB_CACHE=/opt/cosmos-data/hf_cache" in cmd
    assert "COSMOS_DISABLE_SAFETY=1" in cmd
    assert "load_kwargs[\"safety_checker\"] = _NoOpSafetyChecker()" in cmd
    assert "HF_TOKEN=%s" in cmd
    assert "sudo tee -a /etc/npa-cosmos-server/env >/dev/null" in cmd
    assert "--local-dir /opt/cosmos-data/models/nvidia--Cosmos-Test" in cmd


def test_cosmos_serve_builds_remote_restart_command(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (0, "started", "")
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=_cfg())
    mocker.patch("npa.cli.cosmos.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "serve",
            "--model",
            "nvidia/Cosmos-Test",
            "--port",
            "9090",
        ],
    )

    assert result.exit_code == 0
    cmd = ssh.run_or_raise.call_args.args[0]
    assert "COSMOS_MODEL_ID=nvidia/Cosmos-Test" in cmd
    assert "COSMOS_SERVER_PORT=9090" in cmd
    assert "COSMOS_DISABLE_SAFETY=1" in cmd
    assert "/opt/cosmos/server.py" in cmd
    assert "from fastapi import FastAPI, HTTPException" in cmd
    assert '@app.get("/jobs/{job_id}")' in cmd
    assert "HF_TOKEN=%s" in cmd
    assert "sudo tee -a /etc/npa-cosmos-server/env >/dev/null" in cmd
    assert "sudo systemctl restart npa-cosmos-server" in cmd


def test_cosmos_serve_maps_ssh_error(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.side_effect = SSHError("ssh failed")
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=_cfg())
    mocker.patch("npa.cli.cosmos.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "cosmos", "serve"])

    assert result.exit_code == 1
    assert "ssh failed" in result.output


def test_cosmos_infer_posts_prompt_and_input(tmp_path: Path, mocker) -> None:
    source = tmp_path / "input.jpg"
    source.write_bytes(b"image-bytes")
    output = tmp_path / "result.mp4"

    http = mocker.MagicMock()
    http.infer.return_value = {"job_id": "job-1", "status": "running"}
    http.job_status.side_effect = [
        {"job_id": "job-1", "status": "running"},
        {
            "job_id": "job-1",
            "status": "completed",
            "model": "nvidia/Cosmos-Test",
            "output_path": "/opt/cosmos/outputs/out.mp4",
        },
    ]
    ssh = mocker.MagicMock()
    ssh.download_file.return_value = str(output)
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=_cfg())
    http_cls = mocker.patch("npa.cli.cosmos.HTTPClient", return_value=http)
    ssh_cls = mocker.patch("npa.cli.cosmos.SSHClient", return_value=ssh)
    sleep = mocker.patch("npa.cli.cosmos.time.sleep")

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "infer",
            "--prompt",
            "robot arm moving a cube",
            "--input-path",
            str(source),
            "--output-path",
            str(output),
            "--poll-interval",
            "0",
        ],
    )

    assert result.exit_code == 0
    http_cls.assert_called_once_with("http://cosmos:8080", timeout=30.0, retries=1)
    payload = http.infer.call_args.args[0]
    assert http.infer.call_args.kwargs == {"timeout": 30.0}
    assert payload["prompt"] == "robot arm moving a cube"
    assert payload["input"]["filename"] == "input.jpg"
    assert payload["input"]["content_base64"] == base64.b64encode(b"image-bytes").decode("ascii")
    assert http.job_status.call_count == 2
    ssh_cls.assert_called_once()
    ssh.download_file.assert_called_once_with("/opt/cosmos/outputs/out.mp4", str(output))
    sleep.assert_not_called()
    assert "job_id: job-1" in result.output
    assert "job_status: completed" in result.output
    assert f"downloaded_to: {output}" in result.output


def test_cosmos_infer_s3_input_and_output(tmp_path: Path, mocker) -> None:
    downloaded = tmp_path / "downloaded.jpg"
    downloaded.write_bytes(b"image-bytes")
    store = mocker.MagicMock()
    store.download_path.return_value = str(downloaded)
    store.upload_file.return_value = "s3://bucket/results/out.mp4"
    mocker.patch("npa.clients.storage.StorageClient.from_environment", return_value=store)
    http = mocker.MagicMock()
    http.infer.return_value = {"job_id": "job-1", "status": "running"}
    http.job_status.return_value = {
        "job_id": "job-1",
        "status": "completed",
        "output_path": "/opt/cosmos/outputs/out.mp4",
    }
    ssh = mocker.MagicMock()
    ssh.download_file.return_value = str(tmp_path / "out.mp4")
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=_cfg())
    mocker.patch("npa.cli.cosmos.HTTPClient", return_value=http)
    mocker.patch("npa.cli.cosmos.SSHClient", return_value=ssh)

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "infer",
            "--prompt",
            "robot arm moving a cube",
            "--input-path",
            "s3://bucket/inputs/input.jpg",
            "--output-path",
            "s3://bucket/results/out.mp4",
        ],
    )

    assert result.exit_code == 0
    store.download_path.assert_called_once()
    store.upload_file.assert_called_once()
    ssh.download_file.assert_called_once()
    assert "saved_to: s3://bucket/results/out.mp4" in result.output


def test_cosmos_infer_times_out_while_polling(mocker) -> None:
    http = mocker.MagicMock()
    http.infer.return_value = {"job_id": "job-1", "status": "running"}
    http.job_status.return_value = {"job_id": "job-1", "status": "running"}
    times = iter([0.0, 0.0, 2.0, 2.0])
    mocker.patch("npa.cli.cosmos.time.monotonic", side_effect=lambda: next(times))
    mocker.patch("npa.cli.cosmos.time.sleep")
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=_cfg())
    mocker.patch("npa.cli.cosmos.HTTPClient", return_value=http)

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "infer",
            "--prompt",
            "robot arm moving a cube",
            "--timeout",
            "1",
            "--poll-interval",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "Inference timed out waiting for job job-1" in result.output


def test_cosmos_infer_reports_server_side_failure(mocker) -> None:
    http = mocker.MagicMock()
    http.infer.return_value = {"job_id": "job-1", "status": "running"}
    http.job_status.return_value = {"job_id": "job-1", "status": "failed", "error": "generation exploded"}
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=_cfg())
    mocker.patch("npa.cli.cosmos.HTTPClient", return_value=http)

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "infer",
            "--prompt",
            "robot arm moving a cube",
        ],
    )

    assert result.exit_code == 1
    assert "Inference job failed: generation exploded" in result.output


def test_cosmos_infer_requires_prompt_or_input(mocker) -> None:
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=_cfg())

    result = runner.invoke(app, ["workbench", "cosmos", "infer"])

    assert result.exit_code == 1
    assert "Provide --prompt, --input, or both" in result.output


def test_cosmos_status_checks_health_endpoint(mocker) -> None:
    http = mocker.MagicMock()
    http.health.return_value = {"status": "ok", "model": "nvidia/Cosmos-Test"}
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=_cfg())
    http_cls = mocker.patch("npa.cli.cosmos.HTTPClient", return_value=http)

    result = runner.invoke(app, ["workbench", "cosmos", "status"])

    assert result.exit_code == 0
    assert "server: up" in result.output
    http_cls.assert_called_once_with("http://cosmos:8080", timeout=10.0, retries=1)
    http.health.assert_called_once()


def test_cosmos_status_maps_server_error(mocker) -> None:
    http = mocker.MagicMock()
    http.health.side_effect = ServerError("down")
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=_cfg("install_failed"))
    mocker.patch("npa.cli.cosmos.HTTPClient", return_value=http)

    result = runner.invoke(app, ["workbench", "cosmos", "status"])

    assert result.exit_code == 1
    assert "app_status: install_failed" in result.output
    assert "Cannot reach Cosmos endpoint" in result.output
    assert "http://cosmos:8080/health" in result.output


def test_cosmos_list_filters_to_cosmos_workbenches(mocker) -> None:
    mocker.patch("npa.cli.cosmos.default_project_name", return_value="proj")
    mocker.patch("npa.cli.cosmos.default_workbench_name", return_value="cosmos")
    mocker.patch(
        "npa.cli.cosmos.list_projects",
        return_value={
            "proj": {
                "region": "eu-north1",
                "workbenches": {
                    "cosmos": {
                        "workbench_type": "cosmos",
                        "gpu_platform": "gpu-h100-sxm",
                        "endpoint": "http://cosmos:8080",
                        "app_status": "install_failed",
                    },
                    "sim": {
                        "workbench_type": "genesis",
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

    result = runner.invoke(app, ["workbench", "cosmos", "list"])

    assert result.exit_code == 0
    assert "cosmos" in result.output
    assert "app_status=install_failed" in result.output
    assert "sim" not in result.output
    assert "train" not in result.output


def test_cosmos_system_info_prints_ssh_output(mocker) -> None:
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (0, "gpu info", "")
    mocker.patch("npa.cli.cosmos.resolve_ssh_config", return_value=_cfg())
    mocker.patch("npa.cli.cosmos.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "cosmos", "system-info"])

    assert result.exit_code == 0
    assert "gpu info" in result.output
    cmd = ssh.run_or_raise.call_args.args[0]
    assert "nvidia-smi" in cmd
    assert "lscpu" in cmd
    assert "free -h" in cmd
    assert "lsblk" in cmd
