from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.serverless import EndpointNotFoundError
from npa.deploy.images import container_image_for_tool


runner = CliRunner()


def _json_output(raw: str) -> dict:
    start = raw.find("{")
    assert start >= 0, raw
    return json.loads(raw[start:])


def _mock_sonic_serverless(mocker) -> object:
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    client.create_job.return_value = SimpleNamespace(id="job-1", name="sonic-job", status="running")
    mocker.patch("npa.cli.workbench.sonic.train.ServerlessClient", return_value=client)
    mocker.patch("npa.cli.workbench.sonic.train.resolve_project_id", return_value="project-1")
    client.subnet_resolver = mocker.patch(
        "npa.cli.workbench.sonic.train.resolve_subnet",
        return_value="vpcsubnet-auto",
    )
    mocker.patch("npa.cli.workbench.sonic.train.sonic_image", return_value="registry.example/npa-sonic:0.1.0")
    mocker.patch(
        "npa.cli.workbench.sonic.train.serverless_job_env",
        return_value=({"NPA_OUTPUT_PATH": "s3://bucket/sonic/"}, {}),
    )
    return client


def test_sonic_registered_under_workbench() -> None:
    result = runner.invoke(app, ["workbench", "--help"])

    assert result.exit_code == 0
    assert "sonic" in result.output


@pytest.mark.parametrize("command", ["deploy", "train", "serve", "status", "list"])
def test_sonic_command_help(command: str) -> None:
    result = runner.invoke(app, ["workbench", "sonic", command, "--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_sonic_deploy_runtime_validation() -> None:
    result = runner.invoke(app, ["workbench", "sonic", "deploy", "--runtime", "invalid"])

    assert result.exit_code != 0
    assert "invalid" in result.output.lower()


def test_sonic_deploy_requires_output_path() -> None:
    result = runner.invoke(app, ["workbench", "sonic", "deploy", "--runtime", "serverless"])

    assert result.exit_code == 1
    assert "requires --output-path" in result.output


def test_sonic_train_serverless_requires_project_id(mocker) -> None:
    mocker.patch("npa.cli.workbench.sonic.helpers.resolve_environment", return_value=None)

    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "-p",
            "proj",
            "-n",
            "sonic",
            "train",
            "--runtime",
            "serverless",
            "--output-path",
            "s3://bucket/sonic/",
            "--submit-only",
        ],
    )

    assert result.exit_code == 1
    assert "requires --project-id" in result.output


def test_sonic_train_default_embodiment_is_unitree_g1(mocker) -> None:
    client = _mock_sonic_serverless(mocker)

    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "-p",
            "proj",
            "-n",
            "sonic",
            "train",
            "--runtime",
            "serverless",
            "--project-id",
            "project-1",
            "--output-path",
            "s3://bucket/sonic/",
            "--submit-only",
            "--job-name",
            "sonic-job",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = _json_output(result.output)
    assert payload["job_id"] == "job-1"
    assert payload["embodiment"] == "UNITREE_G1_SONIC"
    command = client.create_job.call_args.kwargs["command"]
    assert "UNITREE_G1_SONIC" in command
    client.subnet_resolver.assert_called_once_with(project_id="project-1", explicit_subnet_id="")


def test_sonic_train_validates_gpu_type() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "train",
            "--runtime",
            "serverless",
            "--project-id",
            "project-1",
            "--output-path",
            "s3://bucket/sonic/",
            "--gpu-type",
            "not-a-gpu",
        ],
    )

    assert result.exit_code == 1
    assert "Unknown GPU type" in result.output


@pytest.mark.smoke
@pytest.mark.skipif(os.environ.get("NPA_TEST_SONIC_SMOKE") != "1", reason="set NPA_TEST_SONIC_SMOKE=1")
def test_sonic_train_smoke_marker() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "train",
            "--runtime",
            "container",
            "--steps",
            "1",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert _json_output(result.output)["sample_data"] is True


def test_sonic_serve_endpoint_format() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "serve",
            "--runtime",
            "container",
            "--mode",
            "sim",
            "--input-type",
            "keyboard",
            "--smoke",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert _json_output(result.output)["endpoint"] == "tcp://127.0.0.1:5556"


def test_sonic_status_endpoint_required() -> None:
    result = runner.invoke(app, ["workbench", "sonic", "status", "--runtime", "serverless"])

    assert result.exit_code == 1
    assert "requires --project-id" in result.output


def test_sonic_list_returns_models() -> None:
    result = runner.invoke(app, ["workbench", "sonic", "list", "--output", "json"])

    assert result.exit_code == 0
    payload = _json_output(result.output)
    assert payload["models"][0]["repo"] == "nvidia/GEAR-SONIC"
    assert "model_encoder.onnx" in payload["models"][0]["artifacts"]


def test_sonic_hf_artifact_manifest() -> None:
    from npa.cli.workbench.sonic.helpers import EXPECTED_HF_ARTIFACTS

    assert set(EXPECTED_HF_ARTIFACTS) == {
        "model_encoder.onnx",
        "model_decoder.onnx",
        "observation_config.yaml",
        "planner_sonic.onnx",
    }


def test_sonic_container_image_name_resolves() -> None:
    assert container_image_for_tool("sonic", registry="registry.example", tag="0.1.0") == (
        "registry.example/npa-sonic:0.1.0"
    )
