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
from pathlib import Path

import pytest

from npa.clients import config as config_module
from npa.clients import nebius


@pytest.fixture(scope="module")
def live_project_alias() -> str:
    if os.environ.get("NPA_PREEMPTIBLE_E2E") != "1":
        pytest.skip("Set NPA_PREEMPTIBLE_E2E=1 to run live preemptible e2e")
    return os.environ.get("NPA_PREEMPTIBLE_E2E_PROJECT", "rtxpro").strip() or "rtxpro"


@pytest.fixture(scope="module")
def live_workbench_name() -> str:
    suffix = os.environ.get("NPA_PREEMPTIBLE_E2E_SUFFIX", str(int(time.time())))
    return f"preempt-e2e-{suffix}"


def _npa_bin() -> str:
    repo_npa = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "npa"
    if repo_npa.is_file():
        return str(repo_npa)
    return "npa"


def _run_npa(args: list[str]) -> subprocess.CompletedProcess[str]:
    # Real subprocess: CliRunner StringIO lacks fileno() and breaks Terraform.
    return subprocess.run(
        [_npa_bin(), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        check=False,
    )


def _nebius_json(args: list[str]) -> dict:
    raw = subprocess.check_output(["nebius", *args, "--format", "json"], text=True)
    return json.loads(raw) if raw.strip() else {}


def test_live_preemptible_lerobot_deploy_and_destroy(live_project_alias, live_workbench_name) -> None:
    env = config_module.resolve_environment(live_project_alias)
    assert env is not None, f"Unknown project alias {live_project_alias!r}"

    # rtxpro / us-central1 exposes gpu-rtx6000 (not L40S); CUDA13 image has driver 580.x.
    gpu_type = os.environ.get("NPA_PREEMPTIBLE_E2E_GPU_TYPE", "gpu-rtx6000")
    gpu_preset = os.environ.get("NPA_PREEMPTIBLE_E2E_GPU_PRESET", "1gpu-24vcpu-218gb")
    image_family = os.environ.get(
        "NPA_PREEMPTIBLE_E2E_IMAGE_FAMILY", "ubuntu24.04-cuda13.0"
    )
    # Infra-only: this test validates preemptible VM create/destroy, not app health.
    deploy_args = [
        "workbench",
        "lerobot",
        "-p",
        live_project_alias,
        "-n",
        live_workbench_name,
        "deploy",
        "--gpu-type",
        gpu_type,
        "--gpu-preset",
        gpu_preset,
        "--preemptible",
        "--skip-app",
        "-v",
        f"image_family={image_family}",
    ]
    try:
        deploy = _run_npa(deploy_args)
        output = deploy.stdout or ""
        if deploy.returncode != 0 and (
            "PermissionDenied" in output
            or "permission denied" in output.lower()
            or "UnsupportedOperation" in output
        ):
            pytest.skip(
                "Active profile lacks VPC/compute create permissions for live VM deploy: "
                f"{output[-400:]}"
            )
        assert deploy.returncode == 0, output

        instance_name = f"lerobot-{live_project_alias}-{live_workbench_name}"
        listed = _nebius_json(
            [
                "compute",
                "instance",
                "list",
                "--parent-id",
                env.project_id,
            ]
        )
        items = listed.get("items", listed if isinstance(listed, list) else [])
        match = next(
            (
                it
                for it in items
                if (it.get("metadata") or {}).get("name") == instance_name
            ),
            None,
        )
        assert match is not None, f"expected instance {instance_name!r} after deploy"
        preemptible = (match.get("spec") or {}).get("preemptible") or match.get(
            "preemptible"
        )
        assert preemptible, f"expected preemptible spec, got: {preemptible!r}"
    finally:
        destroy = _run_npa(
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
            ]
        )
        assert destroy.returncode == 0, destroy.stdout


def test_live_bootstrap_reuses_restricted_iam_profile() -> None:
    if os.environ.get("NPA_PREEMPTIBLE_E2E") != "1":
        pytest.skip("Set NPA_PREEMPTIBLE_E2E=1 to run live preemptible e2e")

    project_id = nebius.current_project_id()
    tenant_id = nebius.current_tenant_id()
    region = os.environ.get("NPA_PREEMPTIBLE_E2E_REGION", "us-central1")
    if not project_id or not tenant_id:
        pytest.skip("Active Nebius CLI profile must expose parent-id and tenant-id")

    try:
        creds = nebius.bootstrap_environment(project_id, tenant_id, region)
    except nebius.NebiusError as exc:
        pytest.skip(f"Bootstrap unavailable for active profile: {exc}")
    if not creds.get("service_account_id"):
        pytest.skip(
            "Bootstrap did not return service_account_id; set NPA_SERVICE_ACCOUNT_ID "
            "or nebius.service_account_id in ~/.npa/credentials.yaml"
        )
    assert creds["nebius_api_key"]
    assert creds["nebius_secret_key"]
    assert str(creds["service_account_id"]).startswith("serviceaccount-")
