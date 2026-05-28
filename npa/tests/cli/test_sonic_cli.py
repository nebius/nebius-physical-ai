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
    assert client.create_job.call_args.kwargs["gpu_type"] == "gpu-h100-sxm"
    assert client.create_job.call_args.kwargs["preset"] == "1gpu-16vcpu-200gb"
    client.subnet_resolver.assert_called_once_with(project_id="project-1", explicit_subnet_id="")


def test_sonic_train_explicit_h100_has_no_availability_warning(mocker) -> None:
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
            "--gpu-type",
            "h100",
        ],
    )

    assert result.exit_code == 0
    assert "L40S on-demand availability" not in result.output
    assert client.create_job.call_args.kwargs["gpu_type"] == "gpu-h100-sxm"
    assert client.create_job.call_args.kwargs["preset"] == "1gpu-16vcpu-200gb"


def test_sonic_train_explicit_l40s_warns_about_availability(mocker) -> None:
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
            "--gpu-type",
            "l40s",
        ],
    )

    assert result.exit_code == 0
    assert "L40S on-demand availability is effectively zero" in result.output
    assert client.create_job.call_args.kwargs["gpu_type"] == "gpu-l40s-a"
    assert client.create_job.call_args.kwargs["preset"] == "1gpu-40vcpu-160gb"


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


# ── status.py coverage gaps ───────────────────────────────────────────────


def test_sonic_status_serverless_requires_job_id_or_name(mocker) -> None:
    mocker.patch(
        "npa.cli.workbench.sonic.helpers.default_workbench_name", return_value=""
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "-p",
            "proj",
            "status",
            "--runtime",
            "serverless",
            "--project-id",
            "project-1",
        ],
    )

    assert result.exit_code == 1
    assert "requires --job-id or --name" in result.output


def test_sonic_status_serverless_reports_running_job(mocker) -> None:
    client = mocker.Mock()
    client.get_job.return_value = SimpleNamespace(
        id="job-1", name="sonic-job", status="running"
    )
    mocker.patch(
        "npa.cli.workbench.sonic.status.ServerlessClient", return_value=client
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "status",
            "--runtime",
            "serverless",
            "--project-id",
            "project-1",
            "--job-id",
            "job-1",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = _json_output(result.output)
    assert payload["status"] == "running"
    assert payload["job_id"] == "job-1"
    assert payload["runtime"] == "serverless"


def test_sonic_status_serverless_returns_not_found(mocker) -> None:
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    mocker.patch(
        "npa.cli.workbench.sonic.status.ServerlessClient", return_value=client
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "status",
            "--runtime",
            "serverless",
            "--project-id",
            "project-1",
            "--job-id",
            "ghost-job",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = _json_output(result.output)
    assert payload["status"] == "not_found"
    assert payload["job"] == "ghost-job"


def test_sonic_status_serverless_surfaces_client_error(mocker) -> None:
    from npa.clients.serverless import ServerlessClientError

    client = mocker.Mock()
    client.get_job.side_effect = ServerlessClientError("api down")
    mocker.patch(
        "npa.cli.workbench.sonic.status.ServerlessClient", return_value=client
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "status",
            "--runtime",
            "serverless",
            "--project-id",
            "project-1",
            "--job-id",
            "job-1",
        ],
    )

    assert result.exit_code == 1
    assert "Serverless Job lookup failed" in result.output


def test_sonic_status_configured_vm_reports_workbench_state(mocker) -> None:
    mocker.patch(
        "npa.cli.workbench.sonic.status.list_projects",
        return_value={
            "proj": {
                "workbenches": {
                    "sonic-wb": {
                        "tool": "sonic",
                        "runtime": "vm",
                        "mode": "sim",
                        "checkpoint_source": "hf",
                        "checkpoint_path": "nvidia/GEAR-SONIC:sonic_release/last.pt",
                        "zmq_port": 5556,
                        "port": 5557,
                        "build_state": "ready",
                        "last_smoke_status": "passed",
                        "app_status": "running",
                    }
                }
            }
        },
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "-p",
            "proj",
            "-n",
            "sonic-wb",
            "status",
            "--runtime",
            "vm",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = _json_output(result.output)
    assert payload["project"] == "proj"
    assert payload["workbench"] == "sonic-wb"
    assert payload["runtime"] == "vm"
    assert payload["mode"] == "sim"
    assert payload["ports"] == {"zmq": 5556, "debug": 5557}
    assert payload["build_state"] == "ready"
    assert payload["app_status"] == "running"


def test_sonic_status_configured_rejects_unknown_workbench(mocker) -> None:
    mocker.patch(
        "npa.cli.workbench.sonic.status.list_projects",
        return_value={"proj": {"workbenches": {}}},
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "-p",
            "proj",
            "-n",
            "missing",
            "status",
            "--runtime",
            "vm",
        ],
    )

    assert result.exit_code == 1
    assert "must reference a configured SONIC workbench" in result.output


# ── sonic deploy: additional coverage for serverless validation + plan ──


def test_sonic_deploy_rejects_invalid_output_path() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "deploy",
            "--runtime",
            "serverless",
            "--output-path",
            "not-an-s3-uri",
        ],
    )
    assert result.exit_code == 1
    assert "output" in result.output.lower() or "s3" in result.output.lower()


def test_sonic_deploy_vm_emits_plan_dict() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "-p",
            "proj",
            "-n",
            "sonic-wb",
            "deploy",
            "--runtime",
            "vm",
            "--mode",
            "sim",
            "--default",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _json_output(result.output)
    assert payload["status"] == "planned"
    assert payload["runtime"] == "vm"
    assert payload["mode"] == "sim"
    assert payload["default"] is True
    assert payload["next"].startswith("npa workbench sonic serve")
    assert payload["port"] == 5557
    assert payload["zmq_port"] == 5556


def test_sonic_deploy_serverless_with_valid_output_path() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "-p",
            "proj",
            "-n",
            "sonic-wb",
            "deploy",
            "--runtime",
            "serverless",
            "--output-path",
            "s3://bucket/sonic/",
            "--dry-run",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _json_output(result.output)
    assert payload["runtime"] == "serverless"
    assert payload["output_path"] == "s3://bucket/sonic/"


# ── sonic serve: additional coverage ─────────────────────────────────────


def test_sonic_serve_requires_zmq_host_for_zmq_input() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "serve",
            "--input-type",
            "zmq",
            "--zmq-host",
            "",
        ],
    )
    assert result.exit_code == 1
    assert "--zmq-host is required" in result.output


def test_sonic_serve_serverless_requires_output_path() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "serve",
            "--runtime",
            "serverless",
        ],
    )
    assert result.exit_code == 1
    assert "requires --output-path" in result.output


def test_sonic_serve_serverless_rejects_bad_output_path() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "serve",
            "--runtime",
            "serverless",
            "--output-path",
            "not-s3",
        ],
    )
    assert result.exit_code == 1


def test_sonic_serve_emits_endpoint_and_container_command() -> None:
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
            "--zmq-host",
            "10.0.0.5",
            "--zmq-port",
            "5560",
            "--realtime-debug-port",
            "5570",
            "--smoke",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _json_output(result.output)
    assert payload["status"] == "smoke-ready"
    assert payload["endpoint"] == "tcp://10.0.0.5:5560"
    assert "5560:5560" in payload["container_command"]
    assert payload["realtime_debug_port"] == 5570
