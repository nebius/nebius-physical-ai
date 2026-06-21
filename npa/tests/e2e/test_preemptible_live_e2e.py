"""Live Nebius validation for Workbench preemptible VM deploys.

Run:

    NPA_PREEMPTIBLE_E2E=1 npa/.venv/bin/python -m pytest npa/tests/e2e/test_preemptible_live_e2e.py -q

Requires IAM/compute permissions to bootstrap a workbench VM, plus a configured
``~/.npa/config.yaml`` project alias (default: ``rtxpro``).
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients import config as config_module
from npa.clients import nebius

runner = CliRunner()


@pytest.fixture(scope="module")
def live_project_alias() -> str:
    if os.environ.get("NPA_PREEMPTIBLE_E2E") != "1":
        pytest.skip("Set NPA_PREEMPTIBLE_E2E=1 to run live preemptible e2e")
    return os.environ.get("NPA_PREEMPTIBLE_E2E_PROJECT", "rtxpro").strip() or "rtxpro"


@pytest.fixture(scope="module")
def live_workbench_name() -> str:
    suffix = os.environ.get("NPA_PREEMPTIBLE_E2E_SUFFIX", str(int(time.time())))
    return f"preempt-e2e-{suffix}"


def _nebius_json(args: list[str]) -> dict:
    raw = subprocess.check_output(["nebius", *args, "--format", "json"], text=True)
    return json.loads(raw) if raw.strip() else {}


def test_live_preemptible_lerobot_deploy_and_destroy(live_project_alias, live_workbench_name) -> None:
    env = config_module.resolve_environment(live_project_alias)
    assert env is not None, f"Unknown project alias {live_project_alias!r}"

    deploy = runner.invoke(
        app,
        [
            "workbench",
            "lerobot",
            "-p",
            live_project_alias,
            "-n",
            live_workbench_name,
            "deploy",
            "--gpu-type",
            os.environ.get("NPA_PREEMPTIBLE_E2E_GPU_TYPE", "gpu-l40s-a"),
            "--gpu-preset",
            os.environ.get("NPA_PREEMPTIBLE_E2E_GPU_PRESET", "1gpu-40vcpu-160gb"),
            "--preemptible",
        ],
        env={**os.environ, "PYTHONPATH": "npa/src"},
    )
    if deploy.exit_code != 0 and "PermissionDenied" in deploy.output:
        pytest.skip("Active profile lacks VPC/compute create permissions for live VM deploy")
    assert deploy.exit_code == 0, deploy.output

    try:
        cfg = config_module.resolve_workbench(live_project_alias, live_workbench_name)
        instance_id = ""
        if cfg is not None:
            instance_id = str(getattr(cfg, "instance_id", "") or "")
        if instance_id:
            payload = _nebius_json(["compute", "instance", "get", "--id", instance_id])
            preemptible = payload.get("spec", {}).get("preemptible") or payload.get("preemptible")
            assert preemptible, f"expected preemptible spec, got: {preemptible!r}"
    finally:
        destroy = runner.invoke(
            app,
            [
                "workbench",
                "lerobot",
                "-p",
                live_project_alias,
                "-n",
                live_workbench_name,
                "deploy",
                "--destroy",
                "--yes",
            ],
        )
        assert destroy.exit_code == 0, destroy.output


def test_live_bootstrap_reuses_restricted_iam_profile() -> None:
    if os.environ.get("NPA_PREEMPTIBLE_E2E") != "1":
        pytest.skip("Set NPA_PREEMPTIBLE_E2E=1 to run live preemptible e2e")

    project_id = nebius.current_project_id()
    tenant_id = nebius.current_tenant_id()
    region = os.environ.get("NPA_PREEMPTIBLE_E2E_REGION", "us-central1")
    if not project_id or not tenant_id:
        pytest.skip("Active Nebius CLI profile must expose parent-id and tenant-id")

    creds = nebius.bootstrap_environment(project_id, tenant_id, region)
    assert creds["nebius_api_key"]
    assert creds["nebius_secret_key"]
    assert creds["service_account_id"].startswith("serviceaccount-")
