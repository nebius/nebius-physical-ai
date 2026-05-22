from __future__ import annotations

import base64
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from botocore.exceptions import ClientError
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
    _build_reload_env_command,
    _build_install_command,
    _cosmos_train_smoke_command,
    _download_remote_output,
    _save_inference_output,
    _serverless_job_env,
    _serverless_train_output_path,
)
from npa.clients import config as config_module
from npa.clients import credentials as credentials_module
from npa.clients.config import (
    SSHConfig,
    ServerlessConfig,
    ServerlessJobConfig,
    StorageConfig,
    WorkbenchConfig,
)
from npa.clients.credentials import CredentialsConfig
from npa.clients.http import ServerError
from npa.clients.serverless import (
    AuthError,
    EndpointInfo,
    EndpointNotFoundError,
    EndpointStatus,
    JobInfo,
    NotEnoughResourcesError,
)
from npa.clients.ssh import SSHError


runner = CliRunner()
TERRAFORM_PLAN_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "terraform_plans"


@pytest.fixture(autouse=True)
def _terraform_plan_allows_apply(mocker):
    mocker.patch(
        "npa.cli.cosmos.provisioner.plan",
        return_value=(TERRAFORM_PLAN_FIXTURES / "fresh_create.txt").read_text(),
    )


def _cfg(
    app_status: str = "",
    *,
    hf_token: str = "",
    runtime: str = "vm",
    serverless: ServerlessConfig | None = None,
) -> WorkbenchConfig:
    return WorkbenchConfig(
        endpoint="http://cosmos:8080",
        ssh=SSHConfig(host="cosmos", user="ubuntu", key_path="~/.ssh/id"),
        storage=StorageConfig(checkpoint_bucket="", endpoint_url=""),
        hf_token=hf_token,
        app_status=app_status,
        runtime=runtime,
        serverless=serverless or ServerlessConfig(),
    )


def _serverless_cfg() -> WorkbenchConfig:
    cfg = _cfg(
        runtime="serverless",
        serverless=ServerlessConfig(
            resource_type="endpoint",
            endpoint_id="endpoint-1",
            endpoint_name="npa-cosmos-proj-cosmos",
            project_id="project-1",
            url="https://cosmos.example",
            image="registry/cosmos:cuda12",
            platform="gpu-h200-sxm",
            preset="1gpu-16vcpu-200gb",
            container_port=8080,
        ),
    )
    cfg.endpoint = "https://cosmos.example"
    cfg.project = "proj"
    cfg.name = "cosmos"
    return cfg


@contextmanager
def _active_endpoint(url: str):
    yield SimpleNamespace(url=url)


def _access_denied(message: str = "AccessDenied") -> ClientError:
    return ClientError(
        {"Error": {"Code": "AccessDenied", "Message": message}},
        "PutObject",
    )


@pytest.mark.parametrize(
    "command",
    [
        "deploy",
        "reload-env",
        "serve",
        "infer",
        "train",
        "finetune",
        "optimize",
        "status",
        "system-info",
        "list",
        "cleanup-partial",
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


def _mock_train_env(mocker):
    mocker.patch("npa.cli.cosmos.resolve_environment", return_value=SimpleNamespace(project_id="project-1"))
    mocker.patch(
        "npa.cli.cosmos.resolve_project_storage",
        return_value=SimpleNamespace(
            checkpoint_bucket="s3://bucket/checkpoints",
            endpoint_url="https://s3.example",
            aws_access_key_id="AKIA",
            aws_secret_access_key="SECRET",
        ),
    )
    mocker.patch("npa.cli.cosmos.resolve_container_registry", return_value="registry.example")
    mocker.patch("npa.cli.cosmos.container_image_for_tool", return_value="registry.example/npa-cosmos:smoke")
    mocker.patch("npa.cli.cosmos._serverless_hf_env", return_value={"HF_TOKEN": "PLACEHOLDER_HF_TOKEN"})
    return mocker.patch("npa.cli.cosmos.resolve_subnet", return_value="vpcsubnet-auto")


def test_cosmos_train_serverless_submit_only_creates_job(mocker) -> None:
    resolver = _mock_train_env(mocker)
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    client.create_job.return_value = SimpleNamespace(id="job-1", name="npa-e2e-jobs-test", status="running", output_uris=())
    mocker.patch("npa.cli.cosmos.ServerlessClient", return_value=client)

    result = runner.invoke(
        app,
        [
            "workbench", "cosmos", "-p", "proj", "-n", "cosmos", "train",
            "--runtime", "serverless", "--smoke", "--submit-only",
            "--job-name", "npa-e2e-jobs-test", "--output-format", "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["job_id"] == "job-1"
    kwargs = client.create_job.call_args.kwargs
    assert kwargs["project_id"] == "project-1"
    assert kwargs["output_path"] == "s3://bucket/checkpoints/jobs/npa-e2e-jobs-test/"
    assert kwargs["subnet_id"] == "vpcsubnet-auto"
    assert kwargs["env"]["COSMOS_TRAIN_SMOKE"] == "1"
    assert kwargs["env"]["NPA_OUTPUT_PATH"] == "s3://bucket/checkpoints/jobs/npa-e2e-jobs-test/"
    assert kwargs["env"]["HF_HOME"] == "/tmp/hf_home"
    assert kwargs["extra_env"]["HF_TOKEN"] == "PLACEHOLDER_HF_TOKEN"
    assert kwargs["extra_env"]["AWS_ACCESS_KEY_ID"] == "AKIA"
    assert kwargs["extra_env"]["AWS_SECRET_ACCESS_KEY"] == "SECRET"
    assert "NPA_COSMOS_TRAIN_SMOKE_DONE" in kwargs["command"]
    assert "PYUPLOAD" in kwargs["command"]
    resolver.assert_called_once_with(project_id="project-1", explicit_subnet_id="")
    client.poll_job.assert_not_called()


def test_cosmos_train_serverless_maps_gpu_alias(mocker) -> None:
    _mock_train_env(mocker)
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    client.create_job.return_value = SimpleNamespace(id="job-1", name="npa-e2e-jobs-test", status="running", output_uris=())
    mocker.patch("npa.cli.cosmos.ServerlessClient", return_value=client)

    result = runner.invoke(
        app,
        [
            "workbench", "cosmos", "-p", "proj", "-n", "cosmos", "train",
            "--runtime", "serverless", "--smoke", "--submit-only",
            "--gpu-type", "l40s",
            "--job-name", "npa-e2e-jobs-test", "--output-format", "json",
        ],
    )

    assert result.exit_code == 0
    kwargs = client.create_job.call_args.kwargs
    assert kwargs["gpu_type"] == "gpu-l40s-a"
    assert kwargs["preset"] == "1gpu-40vcpu-160gb"


def test_cosmos_serverless_job_env_uses_shared_builder(mocker) -> None:
    mocker.patch(
        "npa.cli.cosmos.resolve_project_storage",
        return_value=SimpleNamespace(
            endpoint_url="https://s3.example",
            aws_access_key_id="AKIA",
            aws_secret_access_key="SECRET",
        ),
    )
    mocker.patch("npa.cli.cosmos._serverless_hf_env", return_value={"HF_TOKEN": "PLACEHOLDER_HF_TOKEN"})

    env, extra_env = _serverless_job_env("proj", require_hf=True, output_path="s3://bucket/out/")

    assert env["NPA_OUTPUT_PATH"] == "s3://bucket/out/"
    assert env["NPA_REQUIRE_HF"] == "1"
    assert env["HF_HOME"] == "/tmp/hf_home"
    assert env["AWS_ENDPOINT_URL"] == "https://s3.example"
    assert extra_env["HF_TOKEN"] == "PLACEHOLDER_HF_TOKEN"
    assert extra_env["AWS_ACCESS_KEY_ID"] == "AKIA"
    assert extra_env["AWS_SECRET_ACCESS_KEY"] == "SECRET"


def test_cosmos_train_smoke_command_uses_shared_upload_helper() -> None:
    command = _cosmos_train_smoke_command(0)

    assert "boto3" in command
    assert "PYUPLOAD" in command
    assert "NPA_COSMOS_TRAIN_SMOKE_DONE" in command


def test_cosmos_train_serverless_rejects_bad_output_path(mocker) -> None:
    _mock_train_env(mocker)
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    mocker.patch("npa.cli.cosmos.ServerlessClient", return_value=client)

    result = runner.invoke(
        app,
        [
            "workbench", "cosmos", "-p", "proj", "-n", "cosmos", "train",
            "--runtime", "serverless", "--smoke", "--submit-only",
            "--job-name", "npa-e2e-jobs-test",
            "--output-path", "file:///tmp/out",
        ],
    )

    assert result.exit_code == 1
    assert "expects an S3 URI" in result.output
    client.create_job.assert_not_called()


def test_serverless_train_output_path_returns_s3_uri(mocker) -> None:
    mocker.patch(
        "npa.cli.cosmos.resolve_project_storage",
        return_value=SimpleNamespace(checkpoint_bucket="bucket", endpoint_url=""),
    )

    result = _serverless_train_output_path("proj", "job-1", "")
    parsed = urlparse(result)

    assert result.startswith("s3://")
    assert parsed.scheme == "s3"
    assert parsed.netloc == "bucket"
    assert result == "s3://bucket/jobs/job-1/"

    mocker.patch(
        "npa.cli.cosmos.resolve_project_storage",
        return_value=SimpleNamespace(checkpoint_bucket="s3://bucket/checkpoints/", endpoint_url=""),
    )
    assert _serverless_train_output_path("proj", "job-1", "") == "s3://bucket/checkpoints/jobs/job-1/"


def test_cosmos_train_serverless_sync_waits_and_status_cancel_dispatch(mocker) -> None:
    _mock_train_env(mocker)
    client = mocker.Mock()
    client.get_job.side_effect = [
        EndpointNotFoundError("missing"),
        SimpleNamespace(id="job-1", name="npa-e2e-jobs-test", status="running", output_uris=()),
    ]
    client.create_job.return_value = SimpleNamespace(id="job-1", name="npa-e2e-jobs-test", status="running", output_uris=())
    client.poll_job.return_value = SimpleNamespace(id="job-1", name="npa-e2e-jobs-test", status="succeeded", output_uris=())
    client.cancel_job.return_value = SimpleNamespace(id="job-1", name="npa-e2e-jobs-test", status="cancelled")
    client.classify_queue_state.return_value = "running"
    mocker.patch("npa.cli.cosmos.ServerlessClient", return_value=client)

    result = runner.invoke(app, ["workbench", "cosmos", "-p", "proj", "-n", "cosmos", "train", "--runtime", "serverless", "--smoke"])
    assert result.exit_code == 0
    client.poll_job.assert_called_once_with("job-1", "project-1", interval_s=30.0, ceiling_s=2400.0)

    status = runner.invoke(app, ["workbench", "cosmos", "-p", "proj", "train", "--runtime", "serverless", "status", "job-1", "--output-format", "json"])
    assert status.exit_code == 0
    assert json.loads(status.output)["status"] == "running"

    cancel = runner.invoke(app, ["workbench", "cosmos", "-p", "proj", "train", "--runtime", "serverless", "cancel", "job-1", "--output-format", "json"])
    assert cancel.exit_code == 0
    assert json.loads(cancel.output)["status"] == "cancelled"


def test_cosmos_train_serverless_ner_error_formatted_for_user(mocker) -> None:
    _mock_train_env(mocker)
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    client.create_job.side_effect = NotEnoughResourcesError(
        "capacity blocked",
        project_id="project-1",
        platform="gpu-h200-sxm",
        preset="8gpu-128vcpu-1600gb",
        gpu_count=8,
        suggested_alternatives=["Retry in a few minutes"],
    )
    mocker.patch("npa.cli.cosmos.ServerlessClient", return_value=client)

    result = runner.invoke(
        app,
        [
            "workbench", "cosmos", "-p", "proj", "-n", "cosmos", "train",
            "--runtime", "serverless", "--smoke", "--submit-only",
            "--job-name", "npa-e2e-jobs-test",
        ],
    )

    assert result.exit_code == 1
    assert "Not enough resources" in result.output
    assert "Retry in a few minutes" in result.output
    assert "Traceback" not in result.output


def test_cosmos_train_serverless_ner_error_json_mode(mocker) -> None:
    _mock_train_env(mocker)
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    client.create_job.side_effect = NotEnoughResourcesError(
        "capacity blocked",
        project_id="project-1",
        platform="gpu-h200-sxm",
        suggested_alternatives=["Retry in a few minutes"],
    )
    mocker.patch("npa.cli.cosmos.ServerlessClient", return_value=client)

    result = runner.invoke(
        app,
        [
            "workbench", "cosmos", "-p", "proj", "-n", "cosmos", "train",
            "--runtime", "serverless", "--smoke", "--submit-only",
            "--job-name", "npa-e2e-jobs-test", "--output-format", "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "NotEnoughResources"
    assert payload["project_id"] == "project-1"
    assert payload["platform"] == "gpu-h200-sxm"


def test_cosmos_train_serverless_auth_error_shows_hint(mocker) -> None:
    _mock_train_env(mocker)
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    client.create_job.side_effect = AuthError("permission denied")
    mocker.patch("npa.cli.cosmos.ServerlessClient", return_value=client)

    result = runner.invoke(
        app,
        [
            "workbench", "cosmos", "-p", "proj", "-n", "cosmos", "train",
            "--runtime", "serverless", "--smoke", "--submit-only",
            "--job-name", "npa-e2e-jobs-test",
        ],
    )

    assert result.exit_code == 1
    assert "Nebius authentication failed" in result.output
    assert "nebius profile create" in result.output


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


def test_cosmos_deploy_help_lists_serverless_runtime() -> None:
    result = runner.invoke(app, ["workbench", "cosmos", "deploy", "--help"])

    assert result.exit_code == 0
    assert "serverless" in result.output


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


def test_cosmos_deploy_existing_alias_no_replace_skips_terraform(mocker) -> None:
    mocker.patch("npa.cli.cosmos.resolve_environment", return_value=None)
    mocker.patch("npa.cli.cosmos.alias_has_terraform_state", return_value=True)
    mocker.patch("npa.cli.cosmos.workbench_is_byovm", return_value=False)
    mocker.patch(
        "npa.cli.cosmos._read_existing_outputs",
        return_value={
            "vm_ip": "10.0.0.7",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.cosmos.write_config")
    mocker.patch("npa.cli.cosmos.list_projects", return_value={})
    init = mocker.patch("npa.cli.cosmos.provisioner.init")
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
            "--gpu-type",
            "gpu-h100-sxm",
            "--gpu-preset",
            "1gpu-16vcpu-200gb",
            "--skip-app",
        ],
    )

    assert result.exit_code == 0
    assert "updating in place without Terraform" in result.output
    init.assert_not_called()
    apply.assert_not_called()


def test_cosmos_deploy_existing_alias_with_replace_prompts_confirmation(mocker) -> None:
    mocker.patch("npa.cli.cosmos.resolve_environment", return_value=None)
    mocker.patch("npa.cli.cosmos.alias_has_terraform_state", return_value=True)
    mocker.patch("npa.cli.cosmos.workbench_is_byovm", return_value=False)
    mocker.patch("npa.cli.cosmos.typer.confirm", return_value=False)
    init = mocker.patch("npa.cli.cosmos.provisioner.init")
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
            "--gpu-type",
            "gpu-h100-sxm",
            "--gpu-preset",
            "1gpu-16vcpu-200gb",
            "--replace",
        ],
    )

    assert result.exit_code == 1
    assert "Aborted" in result.output
    init.assert_not_called()
    apply.assert_not_called()


def test_cosmos_deploy_existing_alias_with_replace_and_yes_runs_terraform(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.cosmos.resolve_environment", return_value=None)
    mocker.patch("npa.cli.cosmos.alias_has_terraform_state", return_value=True)
    mocker.patch("npa.cli.cosmos.workbench_is_byovm", return_value=False)
    confirm = mocker.patch("npa.cli.cosmos.typer.confirm")
    mocker.patch("npa.cli.cosmos.provisioner.init")
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
    mocker.patch("npa.cli.cosmos.write_config")
    mocker.patch("npa.cli.cosmos.list_projects", return_value={})

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
            "--replace",
            "--yes",
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
    confirm.assert_not_called()
    apply.assert_called_once()


def test_cosmos_deploy_replacement_plan_without_replace_aborts(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.cosmos.resolve_environment", return_value=None)
    mocker.patch("npa.cli.cosmos.alias_has_terraform_state", return_value=False)
    mocker.patch("npa.cli.cosmos.workbench_is_byovm", return_value=False)
    mocker.patch("npa.cli.cosmos.provisioner.init")
    mocker.patch(
        "npa.cli.cosmos.provisioner.plan",
        return_value=(TERRAFORM_PLAN_FIXTURES / "gpu_type_change_full_replace.txt").read_text(),
    )
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
            "--tf-dir",
            str(tmp_path),
            "--gpu-type",
            "gpu-h100-sxm",
            "--gpu-preset",
            "1gpu-16vcpu-200gb",
            "--skip-app",
        ],
    )

    assert result.exit_code == 1
    assert "would replace or destroy managed infrastructure" in result.output
    assert "nebius_compute_v1_instance.workbench" in result.output
    apply.assert_not_called()


def test_cosmos_deploy_fresh_alias_runs_terraform(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.cosmos.resolve_environment", return_value=None)
    mocker.patch("npa.cli.cosmos.alias_has_terraform_state", return_value=False)
    mocker.patch("npa.cli.cosmos.workbench_is_byovm", return_value=False)
    mocker.patch("npa.cli.cosmos.provisioner.init")
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
    mocker.patch("npa.cli.cosmos.write_config")
    mocker.patch("npa.cli.cosmos.list_projects", return_value={})

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "-p",
            "proj",
            "-n",
            "new",
            "deploy",
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
    apply.assert_called_once()


def test_cosmos_deploy_byovm_alias_skips_terraform(mocker) -> None:
    mocker.patch("npa.cli.cosmos.resolve_environment", return_value=None)
    mocker.patch("npa.cli.cosmos.alias_has_terraform_state", return_value=False)
    mocker.patch("npa.cli.cosmos.workbench_is_byovm", return_value=True)
    mocker.patch(
        "npa.cli.cosmos._read_existing_outputs",
        return_value={
            "vm_ip": "10.0.0.7",
            "ssh_user": "ubuntu",
            "ssh_key_path": "~/.ssh/id",
            "storage_bucket": "bucket",
            "storage_endpoint": "https://storage.example",
        },
    )
    mocker.patch("npa.cli.cosmos.write_config")
    mocker.patch("npa.cli.cosmos.list_projects", return_value={})
    init = mocker.patch("npa.cli.cosmos.provisioner.init")
    apply = mocker.patch("npa.cli.cosmos.provisioner.apply")

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "-p",
            "proj",
            "-n",
            "byovm",
            "deploy",
            "--gpu-type",
            "gpu-h100-sxm",
            "--gpu-preset",
            "1gpu-16vcpu-200gb",
            "--skip-app",
        ],
    )

    assert result.exit_code == 0
    init.assert_not_called()
    apply.assert_not_called()


def test_cosmos_reload_env_command_updates_credentials_without_embedding_secret() -> None:
    cmd = _build_reload_env_command(("HF_TOKEN", "AWS_ACCESS_KEY_ID"), port=8081)

    assert "/etc/npa-cosmos-server/env" in cmd
    assert "HF_TOKEN=\"${HF_TOKEN:-}\"" in cmd
    assert "AWS_ACCESS_KEY_ID=\"${AWS_ACCESS_KEY_ID:-}\"" in cmd
    assert "npa-cosmos-server" in cmd
    assert "NPA_COSMOS_RELOAD_ENV_COMPLETE" in cmd
    assert "PLACEHOLDER_HF_TOKEN" not in cmd


def test_cosmos_reload_env_writes_env_via_ssh(mocker) -> None:
    cfg = _cfg(hf_token="PLACEHOLDER_HF_TOKEN")
    cfg.service_port = 8081
    cfg.storage = StorageConfig(
        checkpoint_bucket="s3://bucket/checkpoints/",
        endpoint_url="https://storage.example",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
    )
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (
        0,
        "updated_keys=AWS_ACCESS_KEY_ID,HF_TOKEN\n"
        "NPA_COSMOS_RELOAD_ENV_COMPLETE env_path=/etc/npa-cosmos-server/env mode=systemd\n",
        "",
    )
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=cfg)
    mocker.patch(
        "npa.cli.cosmos.resolve_credentials",
        return_value=CredentialsConfig(tokens={"HF_TOKEN": "PLACEHOLDER_HF_TOKEN"}),
    )
    ssh_cls = mocker.patch("npa.cli.cosmos.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "cosmos", "reload-env", "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "reloaded"
    assert payload["env_path"] == "/etc/npa-cosmos-server/env"
    assert payload["mode"] == "systemd"
    assert payload["restarted"] is True
    assert "HF_TOKEN" in payload["updated_keys"]
    cmd = ssh.run_or_raise.call_args.args[0]
    assert "HF_TOKEN=\"${HF_TOKEN:-}\"" in cmd
    assert "PLACEHOLDER_HF_TOKEN" not in cmd
    ssh_tokens = ssh_cls.call_args.args[0].tokens
    assert ssh_tokens["HF_TOKEN"] == "PLACEHOLDER_HF_TOKEN"
    assert ssh_tokens["AWS_ACCESS_KEY_ID"] == "key"


def test_cosmos_reload_env_fails_clean_on_missing_credentials(mocker) -> None:
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=_cfg())
    mocker.patch("npa.cli.cosmos.resolve_credentials", return_value=CredentialsConfig())
    ssh_cls = mocker.patch("npa.cli.cosmos.SSHClient")

    result = runner.invoke(app, ["workbench", "cosmos", "reload-env"])

    assert result.exit_code == 1
    assert "No shared credentials found" in result.output
    ssh_cls.assert_not_called()


def test_cosmos_reload_env_propagates_ssh_failure(mocker) -> None:
    cfg = _cfg(hf_token="PLACEHOLDER_HF_TOKEN")
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=cfg)
    mocker.patch(
        "npa.cli.cosmos.resolve_credentials",
        return_value=CredentialsConfig(tokens={"HF_TOKEN": "PLACEHOLDER_HF_TOKEN"}),
    )
    ssh = mocker.MagicMock()
    ssh.run_or_raise.side_effect = SSHError("transport down")
    mocker.patch("npa.cli.cosmos.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "cosmos", "reload-env"])

    assert result.exit_code == 1
    assert "Cosmos env reload failed: transport down" in result.output


def test_cosmos_reload_env_dry_run_does_not_apply(mocker) -> None:
    cfg = _cfg(hf_token="PLACEHOLDER_HF_TOKEN")
    cfg.service_port = 8081
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (
        0,
        "NPA_COSMOS_ENV_READ env_path=/etc/npa-cosmos-server/env mode=systemd\n"
        "HF_TOKEN=old-token\n",
        "",
    )
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=cfg)
    mocker.patch(
        "npa.cli.cosmos.resolve_credentials",
        return_value=CredentialsConfig(tokens={"HF_TOKEN": "PLACEHOLDER_HF_TOKEN"}),
    )
    mocker.patch("npa.cli.cosmos.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "cosmos", "reload-env", "--dry-run"])

    assert result.exit_code == 0
    assert "Dry run" in result.output
    assert "No changes applied" in result.output
    command = ssh.run_or_raise.call_args.args[0]
    assert "NPA_COSMOS_ENV_READ" in command
    assert "NPA_COSMOS_RELOAD_ENV_COMPLETE" not in command


def test_cosmos_reload_env_dry_run_shows_changes(mocker) -> None:
    cfg = _cfg(hf_token="PLACEHOLDER_HF_TOKEN")
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (
        0,
        "NPA_COSMOS_ENV_READ env_path=/etc/npa-cosmos-server/env mode=systemd\n"
        "HF_TOKEN=old-token\n",
        "",
    )
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=cfg)
    mocker.patch(
        "npa.cli.cosmos.resolve_credentials",
        return_value=CredentialsConfig(tokens={"HF_TOKEN": "PLACEHOLDER_HF_TOKEN"}),
    )
    mocker.patch("npa.cli.cosmos.SSHClient", return_value=ssh)

    result = runner.invoke(app, ["workbench", "cosmos", "reload-env", "--dry-run"])

    assert result.exit_code == 0
    assert "--- current" in result.output
    assert "+++ proposed" in result.output
    assert "-HF_TOKEN=old-" in result.output
    assert "+HF_TOKEN=PLAC" in result.output


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
    mocker.patch("npa.cli.cosmos.resolve_credentials", return_value=SimpleNamespace(hf_token="PLACEHOLDER_HF_TOKEN", tokens={}))
    mocker.patch("npa.cli.cosmos.list_projects", return_value={})
    write_config = mocker.patch("npa.cli.cosmos.write_config")
    update_status = mocker.patch("npa.cli.cosmos.update_workbench_app_status")
    deploy_container = mocker.patch("npa.deploy.configurator.deploy_workbench_container")
    mocker.patch("npa.deploy.configurator.write_remote_docker_env_file")
    validate = mocker.patch("npa.cli.cosmos.validate_hf_access", return_value=SimpleNamespace(ok=True, error=""))
    mocker.patch("npa.cli.cosmos.health_check_auto", return_value=(True, ""))
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
    assert "HF access ok: nvidia/Cosmos-1.0-Diffusion-7B-Text2World" in result.output
    validate.assert_called_once()
    tf_vars = apply.call_args.kwargs["tf_vars"]
    assert tf_vars["workbench_type"] == "cosmos"
    assert tf_vars["boot_disk_size_gb"] == "250"
    deploy_container.assert_called_once()
    assert deploy_container.call_args.kwargs["container_name"] == "npa-cosmos"
    assert deploy_container.call_args.kwargs["image_ref"].endswith("/npa-cosmos:1.0.9")
    wb_cfg = write_config.call_args_list[0].args[0]["projects"]["proj"]["workbenches"]["cosmos-container"]
    assert wb_cfg["runtime"] == "container"
    assert update_status.call_args_list[0].args == ("proj", "cosmos-container", "installing")
    assert update_status.call_args_list[-1].args == ("proj", "cosmos-container", "healthy")


def test_cosmos_deploy_serverless_creates_endpoint_and_persists_config(mocker) -> None:
    client = mocker.MagicMock()
    client.create_endpoint.return_value = EndpointInfo(
        id="endpoint-1",
        name="npa-cosmos-proj-cosmos",
        project_id="project-1",
        status=EndpointStatus.CREATING,
        url="https://cosmos.example",
    )
    mocker.patch("npa.cli.cosmos.ServerlessClient", return_value=client)
    mocker.patch(
        "npa.cli.cosmos.resolve_environment",
        return_value=SimpleNamespace(project_id="project-1", tenant_id="", region="eu-north1"),
    )
    mocker.patch("npa.cli.cosmos.workbench_entry", return_value={})
    mocker.patch("npa.cli.cosmos.list_projects", return_value={"proj": {}})
    update_serverless = mocker.patch("npa.cli.cosmos.update_workbench_serverless_endpoint")
    write_config = mocker.patch("npa.cli.cosmos.write_config")
    model_check = mocker.patch("npa.cli.cosmos.validate_hf_access")
    provisioner_apply = mocker.patch("npa.cli.cosmos.provisioner.apply")

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
            "--runtime",
            "serverless",
            "--image",
            "registry/cosmos:cuda12",
            "--platform",
            "gpu-h200-sxm",
            "--preset",
            "1gpu-16vcpu-200gb",
            "--container-port",
            "8080",
            "--subnet-id",
            "vpcsubnet-1",
            "--env",
            "MODE=smoke",
            "--volume",
            "s3://bucket:/data:rw",
        ],
    )

    assert result.exit_code == 0
    spec = client.create_endpoint.call_args.args[0]
    assert spec.name == "npa-cosmos-proj-cosmos"
    assert spec.project_id == "project-1"
    assert spec.image == "registry/cosmos:cuda12"
    assert spec.platform == "gpu-h200-sxm"
    assert spec.preset == "1gpu-16vcpu-200gb"
    assert spec.container_ports == [8080]
    assert spec.subnet_id == "vpcsubnet-1"
    assert spec.env["COSMOS_MODEL_ID"] == "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
    assert spec.env["MODE"] == "smoke"
    assert spec.volumes == ["s3://bucket:/data:rw"]
    update_serverless.assert_called_once()
    write_config.assert_called_once()
    model_check.assert_not_called()
    provisioner_apply.assert_not_called()


def test_cosmos_deploy_serverless_waits_when_requested(mocker) -> None:
    client = mocker.MagicMock()
    client.create_endpoint.return_value = EndpointInfo(
        id="endpoint-1",
        name="npa-cosmos-proj-cosmos",
        project_id="project-1",
        status=EndpointStatus.CREATING,
        url="",
    )
    client.wait_for_running.return_value = EndpointInfo(
        id="endpoint-1",
        name="npa-cosmos-proj-cosmos",
        project_id="project-1",
        status=EndpointStatus.RUNNING,
        url="https://cosmos.example",
    )
    mocker.patch("npa.cli.cosmos.ServerlessClient", return_value=client)
    mocker.patch(
        "npa.cli.cosmos.resolve_environment",
        return_value=SimpleNamespace(project_id="project-1", tenant_id="", region="eu-north1"),
    )
    mocker.patch("npa.cli.cosmos.workbench_entry", return_value={})
    mocker.patch("npa.cli.cosmos.list_projects", return_value={"proj": {}})
    mocker.patch("npa.cli.cosmos.update_workbench_serverless_endpoint")
    update_status = mocker.patch("npa.cli.cosmos.update_workbench_app_status")
    mocker.patch("npa.cli.cosmos.write_config")
    resolver = mocker.patch("npa.cli.cosmos.resolve_subnet", return_value="vpcsubnet-auto")

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
            "--runtime",
            "serverless",
            "--platform",
            "gpu-h200-sxm",
            "--preset",
            "1gpu-16vcpu-200gb",
            "--wait",
        ],
    )

    assert result.exit_code == 0
    client.wait_for_running.assert_called_once_with("project-1", "endpoint-1")
    update_status.assert_called_once_with("proj", "cosmos", "healthy")
    resolver.assert_called_once_with(project_id="project-1", explicit_subnet_id="")


def test_cosmos_deploy_serverless_dry_run_does_not_create_endpoint(mocker) -> None:
    client_cls = mocker.patch("npa.cli.cosmos.ServerlessClient")
    mocker.patch(
        "npa.cli.cosmos.resolve_environment",
        return_value=SimpleNamespace(project_id="project-1", tenant_id="", region="eu-north1"),
    )
    mocker.patch("npa.cli.cosmos.workbench_entry", return_value={})
    resolver = mocker.patch("npa.cli.cosmos.resolve_subnet", return_value="vpcsubnet-auto")

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
            "--runtime",
            "serverless",
            "--platform",
            "gpu-h200-sxm",
            "--preset",
            "1gpu-16vcpu-200gb",
            "--dry-run",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert '"status": "dry_run"' in result.output
    assert '"runtime": "serverless"' in result.output
    assert '"subnet_id": "vpcsubnet-auto"' in result.output
    resolver.assert_called_once_with(project_id="project-1", explicit_subnet_id="")
    client_cls.assert_not_called()


def test_cosmos_deploy_serverless_rejects_invalid_env(mocker) -> None:
    mocker.patch(
        "npa.cli.cosmos.resolve_environment",
        return_value=SimpleNamespace(project_id="project-1", tenant_id="", region="eu-north1"),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "deploy",
            "--runtime",
            "serverless",
            "--platform",
            "gpu-h200-sxm",
            "--preset",
            "1gpu-16vcpu-200gb",
            "--env",
            "BROKEN",
        ],
    )

    assert result.exit_code == 1
    assert "Invalid --env format" in result.output


def test_cosmos_deploy_serverless_existing_alias_requires_replace(mocker) -> None:
    mocker.patch(
        "npa.cli.cosmos.resolve_environment",
        return_value=SimpleNamespace(project_id="project-1", tenant_id="", region="eu-north1"),
    )
    mocker.patch("npa.cli.cosmos.workbench_entry", return_value={"runtime": "serverless"})

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
            "--runtime",
            "serverless",
            "--platform",
            "gpu-h200-sxm",
            "--preset",
            "1gpu-16vcpu-200gb",
        ],
    )

    assert result.exit_code == 1
    assert "Use --replace" in result.output


def test_cosmos_deploy_serverless_replace_deletes_existing_endpoint(mocker) -> None:
    old_cfg = _serverless_cfg()
    client = mocker.MagicMock()
    client.create_endpoint.return_value = EndpointInfo(
        id="endpoint-2",
        name="npa-cosmos-proj-cosmos",
        project_id="project-1",
        status=EndpointStatus.CREATING,
        url="https://new.example",
    )
    mocker.patch("npa.cli.cosmos.ServerlessClient", return_value=client)
    mocker.patch(
        "npa.cli.cosmos.resolve_environment",
        return_value=SimpleNamespace(project_id="project-1", tenant_id="", region="eu-north1"),
    )
    mocker.patch("npa.cli.cosmos.workbench_entry", return_value={"runtime": "serverless"})
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=old_cfg)
    mocker.patch("npa.cli.cosmos.list_projects", return_value={"proj": {}})
    mocker.patch("npa.cli.cosmos.update_workbench_serverless_endpoint")
    mocker.patch("npa.cli.cosmos.write_config")
    resolver = mocker.patch("npa.cli.cosmos.resolve_subnet", return_value="vpcsubnet-auto")

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
            "--runtime",
            "serverless",
            "--platform",
            "gpu-h200-sxm",
            "--preset",
            "1gpu-16vcpu-200gb",
            "--replace",
            "--yes",
        ],
    )

    assert result.exit_code == 0
    client.delete_endpoint.assert_called_once_with("project-1", "endpoint-1")
    resolver.assert_called_once_with(project_id="project-1", explicit_subnet_id="")


def test_cosmos_deploy_destroy_serverless_deletes_endpoint_and_config(mocker) -> None:
    cfg = _serverless_cfg()
    client = mocker.MagicMock()
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=cfg)
    mocker.patch("npa.cli.cosmos.ServerlessClient", return_value=client)
    remove = mocker.patch("npa.cli.cosmos.remove_workbench_config")

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
            "--runtime",
            "serverless",
            "--destroy",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    client.delete_endpoint.assert_called_once_with("project-1", "endpoint-1")
    remove.assert_called_once_with("proj", "cosmos")
    assert '"status": "deleted"' in result.output


def test_cosmos_teardown_serverless_deletes_endpoint_and_config(mocker) -> None:
    cfg = _serverless_cfg()
    client = mocker.MagicMock()
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=cfg)
    mocker.patch("npa.cli.cosmos.ServerlessClient", return_value=client)
    remove = mocker.patch("npa.cli.cosmos.remove_workbench_config")

    result = runner.invoke(
        app,
        ["workbench", "cosmos", "teardown", "--yes"],
    )

    assert result.exit_code == 0
    client.delete_endpoint.assert_called_once_with("project-1", "endpoint-1")
    remove.assert_called_once_with("proj", "cosmos")
    assert "config_removed: True" in result.output


def test_cosmos_teardown_serverless_dry_run_does_not_delete(mocker) -> None:
    cfg = _serverless_cfg()
    client_cls = mocker.patch("npa.cli.cosmos.ServerlessClient")
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=cfg)
    remove = mocker.patch("npa.cli.cosmos.remove_workbench_config")

    result = runner.invoke(
        app,
        ["workbench", "cosmos", "teardown", "--dry-run"],
    )

    assert result.exit_code == 0
    assert "status: dry_run" in result.output
    client_cls.assert_not_called()
    remove.assert_not_called()


def test_cosmos_teardown_rejects_vm_alias(mocker) -> None:
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=_cfg())

    result = runner.invoke(app, ["workbench", "cosmos", "teardown", "--yes"])

    assert result.exit_code == 1
    assert "serverless aliases" in result.output


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


def test_cosmos_serve_serverless_prewarms_health_only(mocker) -> None:
    cfg = _serverless_cfg()
    http = mocker.MagicMock()
    http.health.return_value = {"status": "ok", "model": "baked-model"}
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=cfg)
    endpoint = mocker.patch(
        "npa.cli.cosmos.service_endpoint",
        return_value=_active_endpoint("https://cosmos.example"),
    )
    http_cls = mocker.patch("npa.cli.cosmos.HTTPClient", return_value=http)
    ssh_cls = mocker.patch("npa.cli.cosmos.SSHClient")

    result = runner.invoke(app, ["workbench", "cosmos", "serve"])

    assert result.exit_code == 0
    assert "status: prewarmed" in result.output
    endpoint.assert_called_once_with(cfg, default_port=8080, service_port=8080)
    http_cls.assert_called_once_with("https://cosmos.example", timeout=120.0, retries=1)
    http.health.assert_called_once()
    assert not http.serve_model.called
    ssh_cls.assert_not_called()


def test_cosmos_infer_posts_prompt_and_input(tmp_path: Path, mocker) -> None:
    source = tmp_path / "input.jpg"
    source.write_bytes(b"image-bytes")
    output_uri = "s3://bucket/results/result.mp4"
    store = mocker.MagicMock()
    store.upload_file.return_value = output_uri
    mocker.patch("npa.clients.storage.StorageClient.from_environment", return_value=store)

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
    ssh.download_file.return_value = str(tmp_path / "result.mp4")
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
            output_uri,
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
    assert ssh.download_file.call_args.args[0] == "/opt/cosmos/outputs/out.mp4"
    store.upload_file.assert_called_once()
    sleep.assert_not_called()
    assert "job_id: job-1" in result.output


def test_cosmos_infer_serverless_uses_saved_endpoint_without_ssh(mocker) -> None:
    http = mocker.MagicMock()
    http.infer.return_value = {"job_id": "job-1", "status": "running"}
    http.job_status.return_value = {
        "job_id": "job-1",
        "status": "completed",
        "result": "ok",
    }
    cfg = _serverless_cfg()
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=cfg)
    endpoint = mocker.patch(
        "npa.cli.cosmos.service_endpoint",
        return_value=_active_endpoint("https://cosmos.example"),
    )
    http_cls = mocker.patch("npa.cli.cosmos.HTTPClient", return_value=http)
    ssh_cls = mocker.patch("npa.cli.cosmos.SSHClient")

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "infer",
            "--prompt",
            "robot arm",
            "--poll-interval",
            "0",
        ],
    )

    assert result.exit_code == 0
    endpoint.assert_called_once_with(cfg, default_port=8080)
    http_cls.assert_called_once_with("https://cosmos.example", timeout=30.0, retries=1)
    ssh_cls.assert_not_called()
    assert "Generating... (status: completed)" in result.output
    assert "Generation complete in" in result.output


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


def test_cosmos_infer_falls_back_to_remote_env_upload_on_local_access_denied(
    tmp_path: Path,
    mocker,
) -> None:
    output_uri = "s3://bucket/results/out.mp4"
    store = mocker.MagicMock()
    store.upload_file.side_effect = _access_denied("AccessDenied")
    mocker.patch("npa.clients.storage.StorageClient.from_environment", return_value=store)
    http = mocker.MagicMock()
    http.infer.return_value = {"job_id": "job-1", "status": "running"}
    http.job_status.return_value = {
        "job_id": "job-1",
        "status": "completed",
        "output_path": "/opt/cosmos-data/outputs/out.mp4",
    }
    ssh = mocker.MagicMock()
    ssh.download_file.return_value = str(tmp_path / "out.mp4")
    ssh.run_or_raise.return_value = (0, "npa_remote_s3_upload_done", "")
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
            "humanoid carrying a box",
            "--output-path",
            output_uri,
            "--output-format",
            "json",
            "--allow-host-creds",
        ],
    )

    assert result.exit_code == 0
    assert ssh.download_file.call_args.args[0] == "/opt/cosmos-data/outputs/out.mp4"
    store.upload_file.assert_called_once()
    remote_upload_cmd = ssh.run_or_raise.call_args.args[0]
    assert ". /etc/npa-cosmos-server/env" in remote_upload_cmd
    assert "AWS_ACCESS_KEY_ID" in remote_upload_cmd
    assert "AccessDenied" not in remote_upload_cmd
    assert f'"saved_to": "{output_uri}"' in result.output
    assert '"upload_mode": "remote"' in result.output
    assert "AccessDenied" in result.output


def test_cosmos_upload_logging_records_local_and_remote_modes(
    mocker,
) -> None:
    output_uri = "s3://bucket/results/out.mp4"
    cfg = _cfg()
    temp_dirs = []
    store = mocker.MagicMock()
    store.upload_file.side_effect = [
        _access_denied("AccessDenied: local base64 upload denied"),
        output_uri,
        _access_denied("AccessDenied: local remote-output upload denied"),
        output_uri,
    ]
    mocker.patch("npa.cli.cosmos._storage_client_for_config", return_value=store)
    ssh = mocker.MagicMock()
    ssh.run_or_raise.return_value = (0, "npa_remote_s3_upload_done", "")
    mocker.patch("npa.cli.cosmos.SSHClient", return_value=ssh)

    try:
        base64_result = {"video_base64": base64.b64encode(b"video").decode("ascii")}
        saved_to = _save_inference_output(
            base64_result,
            output_uri,
            cfg,
            temp_dirs,
            allow_host_creds=True,
        )
        assert saved_to == output_uri
        assert base64_result["upload_mode"] == "remote"
        assert "AccessDenied: local base64 upload denied" in base64_result["local_upload_error"]

        base64_success = {"video_base64": base64.b64encode(b"video").decode("ascii")}
        saved_to = _save_inference_output(base64_success, output_uri, cfg, temp_dirs)
        assert saved_to == output_uri
        assert base64_success["upload_mode"] == "local"
        assert "local_upload_error" not in base64_success

        remote_result: dict[str, str] = {}
        saved_to = _download_remote_output(
            "/opt/cosmos-data/outputs/out.mp4",
            output_uri,
            cfg,
            temp_dirs,
            result=remote_result,
            allow_host_creds=True,
        )
        assert saved_to == output_uri
        assert remote_result["upload_mode"] == "remote"
        assert "AccessDenied: local remote-output upload denied" in remote_result["local_upload_error"]

        remote_success: dict[str, str] = {}
        saved_to = _download_remote_output(
            "/opt/cosmos-data/outputs/out-2.mp4",
            output_uri,
            cfg,
            temp_dirs,
            result=remote_success,
        )
        assert saved_to == output_uri
        assert remote_success["upload_mode"] == "local"
        assert "local_upload_error" not in remote_success
    finally:
        for tmp in temp_dirs:
            tmp.cleanup()


def test_cosmos_infer_rejects_local_output_path_before_config(mocker) -> None:
    resolve_config = mocker.patch("npa.cli.cosmos.resolve_config")

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "infer",
            "--prompt",
            "robot arm moving a cube",
            "--output-path",
            "/tmp/out.mp4",
        ],
    )

    assert result.exit_code == 1
    assert "Cosmos infer --output-path expects an S3 URI" in result.output
    assert "S3 handoff contract" in result.output
    resolve_config.assert_not_called()


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


def test_cosmos_status_serverless_reports_nebius_and_health(mocker) -> None:
    cfg = _serverless_cfg()
    client = mocker.MagicMock()
    client.get_endpoint.return_value = EndpointInfo(
        id="endpoint-1",
        name="npa-cosmos-proj-cosmos",
        project_id="project-1",
        status=EndpointStatus.RUNNING,
        url="https://cosmos.example",
    )
    http = mocker.MagicMock()
    http.health.return_value = {"status": "ok", "model": "baked-model"}
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=cfg)
    mocker.patch("npa.cli.cosmos.ServerlessClient", return_value=client)
    http_cls = mocker.patch("npa.cli.cosmos.HTTPClient", return_value=http)
    ssh_cls = mocker.patch("npa.cli.cosmos.SSHClient")

    result = runner.invoke(app, ["workbench", "cosmos", "status"])

    assert result.exit_code == 0
    assert "runtime: serverless" in result.output
    assert "serverless_status: running" in result.output
    client.get_endpoint.assert_called_once_with("project-1", "endpoint-1")
    http_cls.assert_called_once_with("https://cosmos.example", timeout=10.0, retries=1)
    ssh_cls.assert_not_called()


def test_cosmos_status_serverless_keeps_resource_status_when_health_fails(mocker) -> None:
    cfg = _serverless_cfg()
    client = mocker.MagicMock()
    client.get_endpoint.return_value = EndpointInfo(
        id="endpoint-1",
        name="npa-cosmos-proj-cosmos",
        project_id="project-1",
        status=EndpointStatus.RUNNING,
        url="https://cosmos.example",
    )
    http = mocker.MagicMock()
    http.health.side_effect = ServerError("connection refused")
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=cfg)
    mocker.patch("npa.cli.cosmos.ServerlessClient", return_value=client)
    mocker.patch("npa.cli.cosmos.HTTPClient", return_value=http)

    result = runner.invoke(app, ["workbench", "cosmos", "status"])

    assert result.exit_code == 0
    assert "server: down" in result.output
    assert "health_error: connection refused" in result.output


def test_cosmos_status_shows_waiting_for_capacity_with_hint(mocker) -> None:
    cfg = _serverless_cfg()
    cfg.serverless_job = ServerlessJobConfig(
        job_id="job-1",
        job_name="train-1",
        project_id="project-1",
        gpu_type="gpu-h200-sxm",
        gpu_count=8,
    )
    client = mocker.MagicMock()
    client.get_job.return_value = JobInfo(
        id="job-1",
        name="train-1",
        project_id="project-1",
        status="queued",
        queued_for_seconds=492,
    )
    client.classify_queue_state.return_value = "waiting_for_capacity"
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=cfg)
    mocker.patch("npa.cli.cosmos.ServerlessClient", return_value=client)

    result = runner.invoke(app, ["workbench", "cosmos", "status"])

    assert result.exit_code == 0
    assert "status: waiting_for_capacity" in result.output
    assert "queue_state_classification: capacity" in result.output
    assert "Platform may be at capacity" in result.output


def test_cosmos_status_json_includes_queue_state_classification(mocker) -> None:
    cfg = _serverless_cfg()
    cfg.serverless_job = ServerlessJobConfig(
        job_id="job-1",
        job_name="train-1",
        project_id="project-1",
        gpu_type="gpu-h200-sxm",
        gpu_count=8,
    )
    client = mocker.MagicMock()
    client.get_job.return_value = JobInfo(
        id="job-1",
        name="train-1",
        project_id="project-1",
        status="queued",
        queued_for_seconds=492,
    )
    client.classify_queue_state.return_value = "waiting_for_capacity"
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=cfg)
    mocker.patch("npa.cli.cosmos.ServerlessClient", return_value=client)

    result = runner.invoke(app, ["workbench", "cosmos", "status", "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "waiting_for_capacity"
    assert payload["queue_state_classification"] == "capacity"
    assert payload["queued_for_seconds"] == 492
    assert payload["platform"] == "gpu-h200-sxm"


def test_cosmos_status_uses_recorded_ssh_endpoint_strategy(mocker) -> None:
    cfg = _cfg()
    cfg.endpoint_strategy = "ssh_fallback"
    cfg.service_port = 8081
    http = mocker.MagicMock()
    http.health.return_value = {"status": "ok", "model": "nvidia/Cosmos-Test"}
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=cfg)
    endpoint = mocker.patch(
        "npa.cli.cosmos.service_endpoint",
        return_value=_active_endpoint("http://127.0.0.1:19081"),
    )
    http_cls = mocker.patch("npa.cli.cosmos.HTTPClient", return_value=http)

    result = runner.invoke(app, ["workbench", "cosmos", "status"])

    assert result.exit_code == 0
    endpoint.assert_called_once_with(cfg, default_port=8080)
    http_cls.assert_called_once_with("http://127.0.0.1:19081", timeout=10.0, retries=1)


def test_cosmos_byovm_deploy_fallback_then_status_uses_ssh_strategy(
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

    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "connected", "")
    ssh.run_or_raise.side_effect = [
        (0, "connected", ""),
        (0, "NVIDIA H200\nNVIDIA H200\n", ""),
        (0, "downloaded", ""),
        (0, "restarted", ""),
    ]
    mocker.patch("npa.cli.cosmos.SSHClient", return_value=ssh)
    mocker.patch(
        "npa.cli.cosmos.resolve_credentials",
        return_value=CredentialsConfig(tokens={"HF_TOKEN": "PLACEHOLDER_HF_TOKEN"}),
    )
    mocker.patch("npa.deploy.configurator.deploy_workbench_container")
    mocker.patch("npa.deploy.configurator.write_remote_docker_env_file")
    mocker.patch("npa.cli.cosmos.write_manifest")
    mocker.patch(
        "npa.cli.cosmos.health_check_auto",
        return_value=(
            True,
            "Public port 8081 unreachable; service healthy via SSH on 203.0.113.10.",
        ),
    )

    deploy = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "-p",
            "proj",
            "-n",
            "cosmos",
            "deploy",
            "--runtime",
            "byovm",
            "--host",
            "203.0.113.10",
            "--ssh-key",
            "~/.ssh/byovm",
            "--region",
            "eu-north1",
            "--gpu-type",
            "gpu-h200-sxm",
            "--gpu-preset",
            "8gpu-160vcpu-1792gb",
            "--server-port",
            "8081",
            "--skip-model-check",
        ],
    )

    assert deploy.exit_code == 0
    resolved = config_module.resolve_config(project="proj", name="cosmos")
    assert resolved.endpoint_strategy == "ssh_fallback"
    assert resolved.service_port == 8081

    http = mocker.MagicMock()
    http.health.return_value = {"status": "ok", "model": "nvidia/Cosmos-Test"}
    endpoint = mocker.patch(
        "npa.cli.cosmos.service_endpoint",
        return_value=_active_endpoint("http://127.0.0.1:19081"),
    )
    http_cls = mocker.patch("npa.cli.cosmos.HTTPClient", return_value=http)

    status = runner.invoke(app, ["workbench", "cosmos", "-p", "proj", "-n", "cosmos", "status"])

    assert status.exit_code == 0
    endpoint.assert_called_once()
    assert endpoint.call_args.args[0].endpoint_strategy == "ssh_fallback"
    http_cls.assert_called_once_with("http://127.0.0.1:19081", timeout=10.0, retries=1)


def test_cosmos_status_reports_degraded_when_model_not_loaded(mocker) -> None:
    http = mocker.MagicMock()
    http.health.return_value = {
        "status": "ok",
        "model": "nvidia/Cosmos-Test",
        "loaded": False,
    }
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=_cfg(hf_token="PLACEHOLDER_HF_TOKEN"))
    mocker.patch("npa.cli.cosmos.HTTPClient", return_value=http)

    result = runner.invoke(app, ["workbench", "cosmos", "status", "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["app_status"] == "degraded"
    assert payload["reason"] == "model not loaded"
    assert payload["readiness"]["hf_token_present"] is True
    assert payload["readiness"]["model_loaded"] is False
    assert payload["readiness"]["ready"] is False
    assert "Model nvidia/Cosmos-Test not loaded" in payload["readiness"]["blockers"]


def test_cosmos_status_maps_server_error(mocker) -> None:
    http = mocker.MagicMock()
    http.health.side_effect = ServerError("down")
    mocker.patch("npa.cli.cosmos.resolve_config", return_value=_cfg("install_failed"))
    mocker.patch("npa.cli.cosmos.HTTPClient", return_value=http)

    result = runner.invoke(app, ["workbench", "cosmos", "status"])

    assert result.exit_code == 1
    assert "app_status: unreachable" in result.output
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
