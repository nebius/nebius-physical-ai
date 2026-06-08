from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.serverless import EndpointNotFoundError
from npa.deploy.images import container_image_for_tool, sonic_image_variant_for_gpu


runner = CliRunner()
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _json_output(raw: str) -> dict:
    start = raw.find("{")
    assert start >= 0, raw
    return json.loads(raw[start:])


def _mock_sonic_serverless(mocker) -> object:
    client = mocker.Mock()
    client.get_job.side_effect = EndpointNotFoundError("missing")
    client.create_job.return_value = SimpleNamespace(
        id="job-1", name="sonic-job", status="running"
    )
    mocker.patch("npa.cli.workbench.sonic.train.ServerlessClient", return_value=client)
    mocker.patch(
        "npa.cli.workbench.sonic.train.resolve_project_id", return_value="project-1"
    )
    client.subnet_resolver = mocker.patch(
        "npa.cli.workbench.sonic.train.resolve_subnet",
        return_value="vpcsubnet-auto",
    )
    mocker.patch(
        "npa.cli.workbench.sonic.train.sonic_image",
        return_value="registry.example/npa-sonic:0.1.2",
    )
    mocker.patch(
        "npa.cli.workbench.sonic.train.serverless_job_env",
        return_value=({"NPA_OUTPUT_PATH": "s3://bucket/sonic/"}, {}),
    )
    return client


def test_sonic_registered_under_workbench() -> None:
    result = runner.invoke(app, ["workbench", "--help"])

    assert result.exit_code == 0
    assert "sonic" in result.output


@pytest.mark.parametrize(
    "command", ["deploy", "train", "export", "eval", "serve", "status", "list"]
)
def test_sonic_command_help(command: str) -> None:
    result = runner.invoke(app, ["workbench", "sonic", command, "--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_sonic_export_help_documents_defaults() -> None:
    result = runner.invoke(app, ["workbench", "sonic", "export", "--help"])

    assert result.exit_code == 0
    assert "--checkpoint" in result.output
    assert "--output" in result.output
    assert "[default: 17]" in result.output
    assert "[default: dynamic]" in result.output
    assert "[default: baked]" in result.output
    assert "[default: sidecar]" in result.output


def test_sonic_eval_help_documents_backend_and_container_contract() -> None:
    result = runner.invoke(app, ["workbench", "sonic", "eval", "--help"])

    assert result.exit_code == 0
    assert "--onnx" in result.output
    assert "--metadata" in result.output
    assert "--sidecar" in result.output
    assert "--backend" in result.output
    assert "[default: reference]" in result.output
    assert "--container-image" in result.output
    assert "--container-gpus" in result.output
    assert "--container-driver-" in result.output
    assert "--container-vulkan-" in result.output
    assert "--container-render-" in result.output
    assert "--container-policy-" in result.output
    assert "--container-metadat" in result.output
    assert "--container-output-" in result.output


def test_sonic_export_cli_maps_flags_to_sdk(mocker, tmp_path) -> None:
    from npa.workbench.sonic import SonicExportResult

    checkpoint = tmp_path / "policy.pt"
    out = tmp_path / "policy.onnx"
    obs_spec = tmp_path / "obs.yaml"
    action_spec = tmp_path / "action.yaml"
    config = tmp_path / "config.yaml"
    checkpoint.write_bytes(b"checkpoint")
    obs_spec.write_text("dim: 4\n", encoding="utf-8")
    action_spec.write_text("dim: 2\n", encoding="utf-8")
    config.write_text("control_dt: 0.02\n", encoding="utf-8")
    export = mocker.patch(
        "npa.cli.workbench.sonic.export.export_onnx",
        return_value=SonicExportResult(
            status="exported",
            checkpoint=str(checkpoint),
            onnx_path=str(out),
            metadata_path=str(out.with_suffix(".metadata.json")),
            opset=18,
            axes="static",
            normalize="sidecar",
            metadata="embedded",
            input_name="obs",
            output_name="action",
            obs_dim=4,
            action_dim=2,
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "export",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(out),
            "--opset",
            "18",
            "--axes",
            "static",
            "--normalize",
            "sidecar",
            "--metadata",
            "embedded",
            "--obs-spec",
            str(obs_spec),
            "--action-spec",
            str(action_spec),
            "--config",
            str(config),
            "--verify",
            "--parity-atol",
            "0.0002",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = _json_output(result.output)
    assert payload["status"] == "exported"
    export.assert_called_once_with(
        checkpoint=str(checkpoint),
        output=str(out),
        opset=18,
        axes="static",
        normalize="sidecar",
        metadata="embedded",
        obs_spec=str(obs_spec),
        action_spec=str(action_spec),
        config=str(config),
        verify=True,
        parity_atol=0.0002,
    )


def test_sonic_eval_cli_maps_flags_to_sdk(mocker, tmp_path) -> None:
    onnx = tmp_path / "policy.onnx"
    metadata = tmp_path / "policy.metadata.json"
    output_path = tmp_path / "result.json"
    onnx.write_bytes(b"onnx")
    metadata.write_text("{}", encoding="utf-8")
    evaluate = mocker.patch(
        "npa.cli.workbench.sonic.eval.evaluate_onnx_policy",
        return_value={
            "format": "npa_sonic_eval_result_v1",
            "status": "completed",
            "backend": "container",
            "mode": "container",
            "smoke_level": False,
            "result_uri": str(output_path),
            "policy": {},
            "eval": {},
            "metrics": {},
            "episodes": [],
            "warnings": [],
        },
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "eval",
            "--onnx",
            str(onnx),
            "--metadata",
            str(metadata),
            "--backend",
            "container",
            "--episodes",
            "4",
            "--env",
            "locomotion-smoke",
            "--container-image",
            "registry.example/sonic-eval:latest",
            "--container-runtime",
            "podman",
            "--container-gpus",
            "all",
            "--container-driver-capabilities",
            "graphics,compute,utility,display",
            "--container-vulkan-icd",
            "/etc/vulkan/icd.d/nvidia_icd.json",
            "--container-glx-vendor",
            "nvidia",
            "--container-device",
            "/dev/dri/card0",
            "--container-device",
            "/dev/dri/renderD128",
            "--container-render-frames",
            "12",
            "--container-policy-path",
            "/eval/in/policy.onnx",
            "--container-metadata-path",
            "/eval/in/policy.metadata.json",
            "--container-output-path",
            "/eval/out/result.json",
            "--container-arg",
            "--verbose",
            "--output",
            str(output_path),
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = _json_output(result.output)
    assert payload["backend"] == "container"
    evaluate.assert_called_once_with(
        onnx=str(onnx),
        metadata=str(metadata),
        backend="container",
        episodes=4,
        env="locomotion-smoke",
        output=str(output_path),
        container_image="registry.example/sonic-eval:latest",
        container_runtime="podman",
        container_gpus="all",
        container_driver_capabilities="graphics,compute,utility,display",
        container_vulkan_icd="/etc/vulkan/icd.d/nvidia_icd.json",
        container_glx_vendor="nvidia",
        container_device=["/dev/dri/card0", "/dev/dri/renderD128"],
        container_render_frames=12,
        container_policy_path="/eval/in/policy.onnx",
        container_metadata_path="/eval/in/policy.metadata.json",
        container_output_path="/eval/out/result.json",
        container_args=["--verbose"],
    )


def test_sonic_deploy_runtime_validation() -> None:
    result = runner.invoke(
        app, ["workbench", "sonic", "deploy", "--runtime", "invalid"]
    )

    assert result.exit_code != 0
    assert "invalid" in result.output.lower()


def test_sonic_deploy_requires_output_path() -> None:
    result = runner.invoke(
        app, ["workbench", "sonic", "deploy", "--runtime", "serverless"]
    )

    assert result.exit_code == 1
    assert "requires --output-path" in result.output


def test_sonic_train_serverless_requires_project_id(mocker) -> None:
    mocker.patch(
        "npa.cli.workbench.sonic.helpers.resolve_environment", return_value=None
    )

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
    assert client.create_job.call_args.kwargs["gpu_type"] == "gpu-l40s-a"
    assert client.create_job.call_args.kwargs["preset"] == "1gpu-40vcpu-160gb"
    client.subnet_resolver.assert_called_once_with(
        project_id="project-1", explicit_subnet_id=""
    )


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


def test_sonic_train_explicit_l40s_uses_l40s_manifest_default(mocker) -> None:
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
    assert "L40S on-demand availability" not in result.output
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
@pytest.mark.skipif(
    os.environ.get("NPA_TEST_SONIC_SMOKE") != "1", reason="set NPA_TEST_SONIC_SMOKE=1"
)
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
    result = runner.invoke(
        app, ["workbench", "sonic", "status", "--runtime", "serverless"]
    )

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
    assert container_image_for_tool(
        "sonic", registry="registry.example", tag="0.1.2"
    ) == ("registry.example/npa-sonic:0.1.2")
    assert container_image_for_tool(
        "sonic", registry="registry.example", gpu_target="gpu-rtx6000"
    ) == ("registry.example/npa-sonic:0.1.2-k8s-runtime")
    assert sonic_image_variant_for_gpu("NVIDIA RTX PRO 6000 Blackwell") == (
        "sonic-k8s-host-mounted"
    )


def test_sonic_container_build_script_uses_supported_version() -> None:
    dockerfile = (PACKAGE_ROOT / "docker/workbench/sonic/Dockerfile").read_text()
    build_script = (PACKAGE_ROOT / "docker/workbench/sonic/build.sh").read_text()

    assert "ARG SONIC_VERSION=0.1.2" in dockerfile
    assert "ARG INSTALL_NVIDIA_DRIVER_USERSPACE=1" in dockerfile
    assert "ARG NPA_DRIVER_PROVISIONING=baked" in dockerfile
    assert 'npa.version="${SONIC_VERSION}"' in dockerfile
    assert 'npa.driver_provisioning="${NPA_DRIVER_PROVISIONING}"' in dockerfile
    assert "COPY docker/workbench/sonic/requirements.txt" in dockerfile
    assert "COPY docker/workbench/sonic/entrypoint.sh" in dockerfile
    assert 'git clone --filter=blob:none --no-checkout "${SONIC_REPO_URL}"' in dockerfile
    assert "git sparse-checkout set" in dockerfile
    assert '"/gear_sonic/**"' in dockerfile
    assert 'rm -rf "${SONIC_HOME}/.git"' in dockerfile
    assert 'data["tool"]["npa"]["supported-tools"]["sonic"]' in build_script
    assert "--platform linux/amd64" in build_script
    assert "--variant" in build_script
    assert "--push" in build_script
    assert "NPA_BUILDX_BUILDER" in build_script
    assert "--driver docker-container" in build_script
    assert 'docker buildx build --builder "$BUILDX_BUILDER"' in build_script
    assert 'docker build "${BUILD_ARGS[@]}" -t "$LOCAL_IMAGE"' in build_script
    assert "npa-sonic:${IMAGE_TAG}" in build_script
