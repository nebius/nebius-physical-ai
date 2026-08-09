"""Tests for the in-repo sim2real runbook -> Kubernetes Job materializer."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workflows.sim2real.materialize import (
    Sim2RealMaterializeError,
    default_runbook_path,
    materialize_k8s_job,
)

RUNBOOK = default_runbook_path()
IMAGE = "cr.eu-north1.nebius.cloud/test-registry/npa-lerobot-vlm-rl:0.1.1"


def _materialize(**kwargs):
    kwargs.setdefault("image", IMAGE)
    kwargs.setdefault("run_id", "unit-run")
    return materialize_k8s_job(RUNBOOK, **kwargs)


def test_default_runbook_path_is_the_committed_runbook() -> None:
    assert RUNBOOK.is_file()
    assert RUNBOOK.name == "sim2real.yaml"
    assert RUNBOOK.parent.name == "workflows"
    assert (RUNBOOK.parent / "physical-ai-data-factory.yaml").is_file()


def test_placeholder_image_is_rejected_with_actionable_error() -> None:
    with pytest.raises(Sim2RealMaterializeError, match="placeholder image"):
        materialize_k8s_job(RUNBOOK, run_id="unit-run")


def test_manifest_is_a_runnable_job() -> None:
    job = _materialize()
    manifest = job.manifest
    assert manifest["apiVersion"] == "batch/v1"
    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["name"] == "sim2real-unit-run"
    pod = manifest["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["image"] == IMAGE
    assert container["command"][0] == "/bin/bash"
    # The image contains exact source and dependencies before it reaches a GPU.
    assert "NPA_SIM2REAL_SOURCE_TARBALL_URI" not in container["command"][2]
    assert "pip install" not in container["command"][2]
    assert "runtime_attestation" in container["command"][2]
    assert "npa.workflows.sim2real run" in container["command"][2]
    assert pod["restartPolicy"] == "Never"
    assert manifest["spec"]["backoffLimit"] == 3
    assert manifest["spec"]["podFailurePolicy"]["rules"] == [
        {
            "action": "Ignore",
            "onPodConditions": [{"type": "DisruptionTarget", "status": "True"}],
        },
        {
            "action": "FailJob",
            "onExitCodes": {"operator": "NotIn", "values": [0]},
        },
    ]


def test_runbook_driver_is_cpu_only_and_preserves_sibling_gpu_config() -> None:
    job = _materialize()
    pod = job.manifest["spec"]["template"]["spec"]
    limits = pod["containers"][0]["resources"]["limits"]
    assert "nvidia.com/gpu" not in limits
    assert limits["cpu"] == "16"
    assert limits["memory"] == "64Gi"
    assert "nodeSelector" not in pod
    env = {item["name"]: item["value"] for item in pod["containers"][0]["env"]}
    assert env["NPA_SIM2REAL_K8S_GPU_PRODUCT"].startswith("NVIDIA-RTX-PRO")
    assert pod["serviceAccountName"]
    assert {entry["name"] for entry in pod["imagePullSecrets"]}
    assert pod["volumes"] == [
        {
            "name": "registry-config",
            "secret": {
                "secretName": "npa-nebius-registry",
                "items": [{"key": ".dockerconfigjson", "path": "config.json"}],
                "defaultMode": 0o400,
            },
        }
    ]
    assert pod["containers"][0]["volumeMounts"] == [
        {
            "name": "registry-config",
            "mountPath": "/var/run/npa/registry",
            "readOnly": True,
        }
    ]
    assert env["DOCKER_CONFIG"] == "/var/run/npa/registry"
    assert "activeDeadlineSeconds" not in job.manifest["spec"]
    labels = job.manifest["metadata"]["labels"]
    assert labels["sim2real.local/run-id"] == "sim2real-unit-run"
    assert job.manifest["spec"]["template"]["metadata"]["labels"] == labels


def test_explicit_positive_timeout_adds_job_deadline() -> None:
    job = _materialize(env_overrides={"NPA_SIM2REAL_K8S_JOB_TIMEOUT_S": "3600"})
    assert job.manifest["spec"]["activeDeadlineSeconds"] == 3600
    container = job.manifest["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["NPA_SIM2REAL_K8S_JOB_TIMEOUT_S"] == "3600"
    assert (
        '--k8s-job-timeout-s "${NPA_SIM2REAL_K8S_JOB_TIMEOUT_S:-0}"'
        in container["command"][2]
    )


def test_registry_config_secret_must_also_be_a_pull_secret() -> None:
    with pytest.raises(Sim2RealMaterializeError, match="must also be listed"):
        _materialize(
            env_overrides={
                "NPA_SIM2REAL_K8S_REGISTRY_CONFIG_SECRET": "runtime-registry",
                "NPA_SIM2REAL_K8S_IMAGE_PULL_SECRETS": "another-secret",
            }
        )


def test_envs_carry_no_unexpanded_variables_and_overrides_win() -> None:
    job = _materialize(env_overrides={"NPA_SIM2REAL_BUCKET": "my-real-bucket"})
    env = {
        item["name"]: item["value"]
        for item in job.manifest["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert all("${" not in value for value in env.values())
    assert env["NPA_SIM2REAL_BUCKET"] == "my-real-bucket"
    assert env["NPA_SIM2REAL_RUN_ID"] == "unit-run"
    secret_sources = {
        ref["secretRef"]["name"]
        for ref in job.manifest["spec"]["template"]["spec"]["containers"][0]["envFrom"]
    }
    assert secret_sources


def test_skip_setup_omits_bootstrap() -> None:
    job = _materialize(include_setup=False)
    script = job.manifest["spec"]["template"]["spec"]["containers"][0]["command"][2]
    assert "pip install" not in script
    assert "npa.workflows.sim2real run" in script


def test_job_name_is_dns1123_sanitized_and_bounded() -> None:
    job = _materialize(run_id="My_Run/With Bad*Chars-" + "x" * 80)
    name = job.manifest["metadata"]["name"]
    assert len(name) <= 63
    assert name.startswith("sim2real-my-run-with-bad-chars")


def test_cli_materialize_writes_applyable_manifest(tmp_path: Path) -> None:
    out = tmp_path / "job.yaml"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "workbench",
            "sim2real",
            "materialize",
            "--run-id",
            "cli-run",
            "--image",
            IMAGE,
            "--env",
            "NPA_SIM2REAL_BUCKET=my-real-bucket",
            "--namespace",
            "robots",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = yaml.safe_load(out.read_text())
    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["namespace"] == "robots"
    assert "kubectl apply -f" in result.output


def test_cli_materialize_rejects_malformed_env() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["workbench", "sim2real", "materialize", "--image", IMAGE, "--env", "NOVALUE"],
    )
    assert result.exit_code != 0
