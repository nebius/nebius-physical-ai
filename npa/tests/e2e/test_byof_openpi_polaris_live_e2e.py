"""Contract and gated live B200 E2E for OpenPI pi0.5 Polaris serving.

The live test consumes an image already built and pushed by the canonical BYOF
runner, pinned by registry digest. It reads the checked-in npa.workflow smoke
and resource profile unchanged, launches the Kubernetes workload through pinned
SkyPilot, and verifies direct plus upstream WebSocket-served inference from S3.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from npa.clients.config import resolve_container_registry
from npa.clients.project_credentials import storage_env_for_project
from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.skypilot_render import secret_env_hints_for_plan
from npa.workflows.byof.live import (
    resolve_byof_kubernetes_target,
    resolve_byof_profile_path,
    resolve_skypilot_bin,
    skypilot_config_for_project,
)

from .npa_workflow_live_helpers import live_bucket

REPO_ROOT = Path(__file__).resolve().parents[3]
BYOF_RUNNER = REPO_ROOT / "npa" / "scripts" / "run_byof_repo.py"
OPENPI_SPEC = (
    REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "byof-openpi.yaml"
)
EXPECTED_CAPABILITIES = {
    "pi05_droid_jointpos_polaris_checkpoint_download",
    "pi05_droid_jointpos_polaris_direct_infer",
    "pi05_droid_jointpos_polaris_served_infer",
}


def _config() -> dict[str, object]:
    payload = yaml.safe_load(OPENPI_SPEC.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    config = payload.get("config")
    assert isinstance(config, dict)
    return config


def _planned_args(run_id: str) -> dict[str, str | bool]:
    plan = build_plan(load_spec(OPENPI_SPEC), run_id=run_id)
    assert len(plan.steps) == 1
    argv = list(plan.steps[0].argv)
    assert argv[:4] == ["npa", "workbench", "byof", "run"]
    result: dict[str, str | bool] = {}
    index = 4
    while index < len(argv):
        flag = argv[index]
        assert flag.startswith("--"), argv[index:]
        if index + 1 == len(argv) or argv[index + 1].startswith("--"):
            result[flag] = True
            index += 1
        else:
            result[flag] = argv[index + 1]
            index += 2
    return result


def _last_json(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    index = 0
    last: dict[str, object] | None = None
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict):
            last = value
        index = max(end, start + 1)
    assert last is not None, text[-4000:]
    return last


def _s3_client(project: str | None):
    boto3 = pytest.importorskip("boto3")
    from botocore.config import Config

    env = storage_env_for_project(
        project,
        allow_host_creds=True,
        endpoint_url=os.environ.get("NPA_BYOF_S3_ENDPOINT", ""),
    )
    kwargs: dict[str, object] = {
        "endpoint_url": (env.get("AWS_ENDPOINT_URL") or "").strip() or None,
        "config": Config(s3={"addressing_style": "path"}),
        "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-central1"),
    }
    if env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"):
        kwargs["aws_access_key_id"] = env["AWS_ACCESS_KEY_ID"]
        kwargs["aws_secret_access_key"] = env["AWS_SECRET_ACCESS_KEY"]
    return boto3.client("s3", **kwargs)


def _read_json(s3, bucket: str, key: str) -> dict[str, object]:
    value = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    assert isinstance(value, dict)
    return value


def test_openpi_polaris_spec_plans_real_b200_serving() -> None:
    plan = build_plan(load_spec(OPENPI_SPEC), run_id="openpi-polaris-render")
    assert len(plan.steps) == 1
    step = plan.steps[0]
    rendered = " ".join(step.argv)
    config = _config()

    assert step.tool_ref == "workbench.byof.repo"
    assert config["repo_ref"] == "15a9616a00943ada6c20a0f158e3adb39df2ccac"
    assert config["resource_profile_yaml"] == "byof-solution-smoke-openpi-b200-gpu"
    assert "pi05_droid_jointpos_polaris" in rendered
    assert "WebsocketPolicyServer" in rendered
    assert "B200:1" in OPENPI_SPEC.read_text(encoding="utf-8")
    assert secret_env_hints_for_plan(plan.steps) == (
        "NPA_OPENPI_ACCEPT_GEMMA_TERMS",
    )
    profile = resolve_byof_profile_path(str(config["resource_profile_yaml"]))
    profile_text = profile.read_text(encoding="utf-8")
    assert "accelerators: B200:1" in profile_text
    assert 'NVIDIA_DRIVER_CAPABILITIES: "compute,utility"' in profile_text
    assert "NPA_OPENPI_ACCEPT_GEMMA_TERMS" not in profile_text


@pytest.mark.skipif(
    os.environ.get("NPA_INTEGRATION_E2E") != "1"
    or os.environ.get("NPA_BYOF_OPENPI_LIVE_B200") != "1",
    reason=(
        "Set NPA_INTEGRATION_E2E=1 and NPA_BYOF_OPENPI_LIVE_B200=1 to run "
        "the digest-pinned OpenPI Polaris B200 smoke."
    ),
)
@pytest.mark.e2e
def test_openpi_polaris_live_b200_served_inference(e2e_project: str | None) -> None:
    assert os.environ.get("NPA_OPENPI_ACCEPT_GEMMA_TERMS") == "YES", (
        "scoped Gemma terms acceptance must be forwarded for this OpenPI run"
    )
    saved_registry = resolve_container_registry(e2e_project)
    project_registry = os.environ.get(
        "NPA_BYOF_OPENPI_PROJECT_REGISTRY", ""
    ).strip() or saved_registry
    image = os.environ.get("NPA_BYOF_OPENPI_REUSE_IMAGE", "").strip()
    assert project_registry.startswith("cr.") and ".nebius.cloud/" in project_registry
    assert image.startswith(project_registry.rstrip("/") + "/"), image
    assert re.search(r"@sha256:[0-9a-f]{64}$", image), (
        "NPA_BYOF_OPENPI_REUSE_IMAGE must be an immutable project-registry digest"
    )

    run_id = "byof-openpi-polaris-e2e-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    planned = _planned_args(run_id)
    profile = resolve_byof_profile_path(str(planned["--yaml"]))
    bucket = live_bucket(e2e_project)
    output_root = f"s3://{bucket}/oss-solutions/openpi"
    prefix = f"oss-solutions/openpi/{run_id}/"
    cmd = [
        sys.executable,
        str(BYOF_RUNNER),
        "--repo-url",
        str(planned["--repo-url"]),
        "--repo-ref",
        str(planned["--repo-ref"]),
        "--base-profile",
        str(planned["--base-profile"]),
        "--base-image",
        str(planned["--base-image"]),
        "--build-command",
        str(planned["--build-command"]),
        "--project",
        e2e_project or "",
        "--registry",
        project_registry,
        "--image",
        image,
        "--workload",
        str(planned["--workload"]),
        "--yaml",
        str(profile),
        "--smoke-command",
        str(planned["--smoke-command"]),
        "--solution-name",
        str(planned["--solution-name"]),
        "--capability-name",
        str(planned["--capability-name"]),
        "--smoke-artifact-name",
        str(planned["--smoke-artifact-name"]),
        "--run-id",
        run_id,
        "--output-root",
        output_root,
        "--wait-timeout",
        str(planned["--wait-timeout"]),
        "--poll-interval",
        str(planned["--poll-interval"]),
        "--skip-build",
        "--skip-push",
        "--cleanup",
    ]
    config_path = skypilot_config_for_project(e2e_project)
    if config_path:
        cmd.extend(["--config-path", config_path])

    env = dict(os.environ)
    env["NPA_E2E_PROJECT"] = e2e_project or env.get("NPA_E2E_PROJECT", "")
    env["NPA_REGISTRY"] = project_registry
    target = resolve_byof_kubernetes_target(e2e_project)
    if target.kubeconfig:
        env["KUBECONFIG"] = target.kubeconfig
        env["NPA_BYOF_KUBECONFIG"] = target.kubeconfig
    if target.context:
        env["NPA_BYOF_K8S_CONTEXT"] = target.context
    sky_bin = resolve_skypilot_bin()
    if sky_bin:
        env["PATH"] = f"{Path(sky_bin).parent}:{env.get('PATH', '')}"

    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, combined[-16000:]
    runner = _last_json(proc.stdout)
    assert runner["status"] == "ok"
    assert runner["image"] == image
    assert runner["build"] == {"ok": True, "skipped": True}

    s3 = _s3_client(e2e_project)
    summary = _read_json(s3, bucket, prefix + "npa_byof_summary.json")
    artifact = _read_json(
        s3,
        bucket,
        prefix + "openpi_pi05_droid_jointpos_polaris_inference.json",
    )
    assert summary["status"] == "success"
    assert summary["smoke_exit_code"] == 0
    assert summary["image"] == image
    assert artifact["status"] == "passed"
    assert set(artifact["capabilities_exercised"]) == EXPECTED_CAPABILITIES
    assert artifact["checkpoint"]["uri"].endswith("pi05_droid_jointpos_polaris")
    assert artifact["checkpoint"]["weights_baked"] is False
    assert artifact["checkpoint"]["provenance"]["object_count"] > 0
    assert artifact["checkpoint"]["provenance"]["total_size_bytes"] > 1_000_000_000
    assert artifact["terms"] == {
        "forwarded": True,
        "scope": "this_openpi_workload_only",
        "persisted": False,
    }
    assert artifact["runtime"]["visible_device_count"] == 1
    assert "B200" in artifact["runtime"]["device_kind"].upper()
    assert artifact["runtime"]["compute_capability"] in {
        "100",
        "10.0",
        "(10, 0)",
        "[10, 0]",
    }
    for mode in ("direct", "served"):
        response = artifact["response"][mode]
        assert response["ok"] is True
        assert response["action_shape"][0] >= 5
        assert response["action_shape"][1] == 8
        assert response["finite"] is True
        assert response["latency_ms"] > 0
    assert artifact["response"]["served"]["healthz"] == "OK"
    assert artifact["response"]["served"]["server_infer_ms"] > 0
    assert artifact["response"]["served"]["execution_scope"] == "kubernetes_workload"
    assert artifact["response"]["served"]["client_origin"] == "same_pod_loopback"
    assert artifact["response"]["action_semantics"].startswith(
        "joint_position_targets_dims_0_6"
    )
    first_five = artifact["response"]["first_five_targets"]
    assert len(first_five) == 5 and all(len(row) == 8 for row in first_five)
    assert all(math.isfinite(float(value)) for row in first_five for value in row)
    assert artifact["training_evaluation"]["live_validated"] is False
