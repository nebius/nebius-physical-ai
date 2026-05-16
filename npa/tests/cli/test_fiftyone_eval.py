from __future__ import annotations

import json
from types import SimpleNamespace

from click.utils import strip_ansi
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import StorageConfig
from npa.clients.serverless import EndpointNotFoundError


runner = CliRunner()


def _mock_fiftyone_serverless_env(mocker):
    mocker.patch("npa.cli.fiftyone.resolve_environment", return_value=SimpleNamespace(project_id="project-1"))
    mocker.patch(
        "npa.cli.fiftyone.resolve_project_storage",
        return_value=StorageConfig(
            checkpoint_bucket="",
            endpoint_url="https://s3.example",
            aws_access_key_id="AKIA",
            aws_secret_access_key="SECRET",
        ),
    )
    mocker.patch("npa.cli.fiftyone.resolve_container_registry", return_value="registry.example")
    mocker.patch("npa.cli.fiftyone.container_image_for_tool", return_value="registry.example/npa-fiftyone:smoke")
    return mocker.patch("npa.cli.fiftyone.resolve_subnet", return_value="vpcsubnet-auto")


def _mock_serverless_client(mocker, *, poll_status: str | None = None):
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    client.create_job.return_value = SimpleNamespace(id="job-1", name="fiftyone-eval-job", status="running", output_uris=())
    if poll_status is not None:
        client.poll_job.return_value = SimpleNamespace(
            id="job-1",
            name="fiftyone-eval-job",
            status=poll_status,
            output_uris=(),
        )
    mocker.patch("npa.cli.fiftyone.ServerlessClient", return_value=client)
    return client


def test_fiftyone_eval_help_documents_serverless_flags() -> None:
    result = runner.invoke(app, ["workbench", "fiftyone", "eval", "--help"])
    output = strip_ansi(result.output)

    assert result.exit_code == 0
    for flag in (
        "--runtime",
        "--project-id",
        "--gpu-type",
        "--region",
        "--checkpoint-path",
        "--predictions-path",
        "--output-path",
        "--subnet-id",
        "--job-name",
        "--timeout-minutes",
        "--submit-only",
    ):
        assert flag in output


def test_fiftyone_eval_h100_builds_serverless_job_spec(mocker) -> None:
    resolver = _mock_fiftyone_serverless_env(mocker)
    client = _mock_serverless_client(mocker)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            "proj",
            "-n",
            "eval",
            "eval",
            "--runtime",
            "serverless",
            "--project-id",
            "project-1",
            "--checkpoint-path",
            "s3://bucket/checkpoint/",
            "--predictions-path",
            "s3://bucket/predictions/",
            "--output-path",
            "s3://bucket/eval/",
            "--submit-only",
            "--job-name",
            "fiftyone-eval-job",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["job_id"] == "job-1"
    assert payload["gpu_type"] == "gpu-h100-sxm"
    assert payload["gpu_preset"] == "1gpu-16vcpu-200gb"
    assert payload["region"] == "eu-north1"
    resolver.assert_called_once_with(project_id="project-1", explicit_subnet_id="")
    kwargs = client.create_job.call_args.kwargs
    assert kwargs["image"] == "registry.example/npa-fiftyone:smoke"
    assert kwargs["gpu_type"] == "gpu-h100-sxm"
    assert kwargs["gpu_count"] == 1
    assert kwargs["preset"] == "1gpu-16vcpu-200gb"
    assert kwargs["subnet_id"] == "vpcsubnet-auto"
    assert kwargs["timeout"] == "30m"
    assert kwargs["env"]["NPA_OUTPUT_PATH"] == "s3://bucket/eval/"
    assert kwargs["env"]["NPA_REGION"] == "eu-north1"
    assert kwargs["env"]["NPA_STAGE_INPUT_PATH"] == "s3://bucket/checkpoint/"
    assert kwargs["env"]["NPA_PREDICTIONS_PATH"] == "s3://bucket/predictions/"
    assert kwargs["extra_env"]["AWS_ACCESS_KEY_ID"] == "AKIA"
    assert kwargs["extra_env"]["AWS_SECRET_ACCESS_KEY"] == "SECRET"
    assert "npa_fiftyone_eval_curation.json" in kwargs["command"]
    assert "eval_summary.json" in kwargs["command"]
    assert "config.json" in kwargs["command"]
    assert "model.safetensors" in kwargs["command"]


def test_fiftyone_eval_rtx6000_defaults_to_us_central1(mocker) -> None:
    _mock_fiftyone_serverless_env(mocker)
    client = _mock_serverless_client(mocker)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            "proj",
            "-n",
            "eval",
            "eval",
            "--runtime",
            "serverless",
            "--project-id",
            "project-1",
            "--gpu-type",
            "rtx6000",
            "--checkpoint-path",
            "s3://bucket/checkpoint/",
            "--output-path",
            "s3://bucket/eval/",
            "--submit-only",
            "--job-name",
            "fiftyone-eval-job",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["gpu_type"] == "gpu-rtx6000"
    assert payload["gpu_preset"] == "1gpu-24vcpu-218gb"
    assert payload["region"] == "us-central1"
    kwargs = client.create_job.call_args.kwargs
    assert kwargs["gpu_type"] == "gpu-rtx6000"
    assert kwargs["preset"] == "1gpu-24vcpu-218gb"
    assert kwargs["env"]["NPA_REGION"] == "us-central1"


def test_fiftyone_eval_rejects_l40s_routing(mocker) -> None:
    _mock_fiftyone_serverless_env(mocker)
    client = _mock_serverless_client(mocker)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "eval",
            "--runtime",
            "serverless",
            "--project-id",
            "project-1",
            "--gpu-type",
            "l40s",
            "--checkpoint-path",
            "s3://bucket/checkpoint/",
            "--output-path",
            "s3://bucket/eval/",
        ],
    )

    assert result.exit_code == 1
    assert "L40S-family routing is intentionally excluded" in result.output
    client.create_job.assert_not_called()


def test_fiftyone_eval_rejects_non_serverless_runtime(mocker) -> None:
    _mock_fiftyone_serverless_env(mocker)
    client = _mock_serverless_client(mocker)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "eval",
            "--runtime",
            "vm",
            "--checkpoint-path",
            "s3://bucket/checkpoint/",
            "--output-path",
            "s3://bucket/eval/",
        ],
    )

    assert result.exit_code == 1
    assert "supports only --runtime serverless" in result.output
    client.create_job.assert_not_called()


def test_fiftyone_eval_rejects_non_s3_checkpoint_path(mocker) -> None:
    _mock_fiftyone_serverless_env(mocker)
    client = _mock_serverless_client(mocker)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "eval",
            "--runtime",
            "serverless",
            "--checkpoint-path",
            "/tmp/checkpoint",
            "--output-path",
            "s3://bucket/eval/",
        ],
    )

    assert result.exit_code == 1
    assert "--checkpoint-path must be an s3:// URI" in result.output
    client.create_job.assert_not_called()


def test_fiftyone_eval_rejects_non_s3_predictions_path(mocker) -> None:
    _mock_fiftyone_serverless_env(mocker)
    client = _mock_serverless_client(mocker)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "eval",
            "--runtime",
            "serverless",
            "--checkpoint-path",
            "s3://bucket/checkpoint/",
            "--predictions-path",
            "/tmp/predictions",
            "--output-path",
            "s3://bucket/eval/",
        ],
    )

    assert result.exit_code == 1
    assert "--predictions-path must be an s3:// URI" in result.output
    client.create_job.assert_not_called()


def test_fiftyone_eval_returns_nonzero_for_failed_polled_job(mocker) -> None:
    _mock_fiftyone_serverless_env(mocker)
    client = _mock_serverless_client(mocker, poll_status="failed")

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            "proj",
            "-n",
            "eval",
            "eval",
            "--runtime",
            "serverless",
            "--project-id",
            "project-1",
            "--checkpoint-path",
            "s3://bucket/checkpoint/",
            "--output-path",
            "s3://bucket/eval/",
            "--job-name",
            "fiftyone-eval-job",
        ],
    )

    assert result.exit_code == 1
    client.poll_job.assert_called_once()
