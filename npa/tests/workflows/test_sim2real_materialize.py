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
    assert RUNBOOK.name == "runbook.yaml"


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
    # setup + run travel together so the pod bootstraps npa before the loop.
    assert "pip install -e ./npa" in container["command"][2]
    assert "npa.workflows.sim2real run" in container["command"][2]
    assert pod["restartPolicy"] == "Never"
    assert manifest["spec"]["backoffLimit"] == 0


def test_runbook_resources_map_to_k8s_limits_and_node_selector() -> None:
    job = _materialize()
    pod = job.manifest["spec"]["template"]["spec"]
    limits = pod["containers"][0]["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == 1
    assert limits["cpu"] == "16"
    assert limits["memory"] == "64Gi"
    assert pod["nodeSelector"]["nvidia.com/gpu.product"]
    assert pod["serviceAccountName"]
    assert {entry["name"] for entry in pod["imagePullSecrets"]}
    assert job.manifest["spec"]["activeDeadlineSeconds"] > 0


def test_envs_carry_no_unexpanded_variables_and_overrides_win() -> None:
    job = _materialize(env_overrides={"NPA_SIM2REAL_BUCKET": "my-real-bucket"})
    env = {item["name"]: item["value"] for item in job.manifest["spec"]["template"]["spec"]["containers"][0]["env"]}
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
    assert "pip install -e ./npa" not in script
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
