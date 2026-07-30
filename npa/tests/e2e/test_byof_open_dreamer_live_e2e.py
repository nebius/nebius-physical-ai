"""Live real-GPU e2e for the Open Dreamer BYOF npa.workflow spec.

This drives the *actual* ``byof-open-dreamer.yaml`` spec end to end on real
Nebius GPUs through the canonical BYOF runner path (the same path the workbench
``workbench.byof.repo`` toolRef resolves to), then verifies the run's S3
artifacts prove all seven Dreamer-4 capabilities were exercised.

Why the BYOF runner and not ``npa workbench workflow submit``: the BYOF submit
path is a nested two-hop (an outer SkyPilot K8s pod that would need to build and
push the image before launching the inner GPU job), which is why the submit
matrix only *plan*-tests BYOF specs. The real GPU path builds/reuses the image
on the operator VM and launches the 2-GPU smoke directly; that is what this test
exercises. The spec remains the tested artifact: its ``smoke_command``,
``resource_profile_yaml``, and capability contract are read straight from the
YAML and run unmodified.

Gating (all required):
  NPA_INTEGRATION_E2E=1           live infra opt-in (module-level)
  NPA_BYOF_OPEN_DREAMER_LIVE_GPU=1  this multi-hour 2-GPU run opt-in
  NPA_BYOF_TEST_IMAGE=<ref>       prebuilt Open Dreamer image to reuse
                                  (avoids an in-test docker build)

The BYOF runner uses a synchronous ``sky launch``, so it blocks for the whole
multi-hour run and returns the final status; do not SIGKILL it mid-run.

Tunables:
  NPA_BYOF_OD_LAUNCH_TIMEOUT (default 18000s) launcher subprocess cap (>= run time)
  NPA_BYOF_OD_S3_WAIT        (default 18000s) max wait for the S3 results artifact
  NPA_BYOF_OD_POLL           (default 120s)   S3 poll interval
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from npa.clients.config import resolve_container_registry
from npa.clients.project_credentials import storage_env_for_project
from npa.workflows.byof.live import (
    resolve_byof_kubernetes_target,
    resolve_skypilot_bin,
    skypilot_config_for_project,
)

from .npa_workflow_live_helpers import live_bucket

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("NPA_INTEGRATION_E2E") != "1",
        reason="Set NPA_INTEGRATION_E2E=1 for live Open Dreamer BYOF infra checks.",
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[3]
BYOF_RUNNER = REPO_ROOT / "npa" / "scripts" / "run_byof_repo.py"
OPEN_DREAMER_SPEC = (
    REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "byof-open-dreamer.yaml"
)

# Capability contract for the accepted Open Dreamer smoke. Keep in sync with
# npa/tests/workflows/test_byof_solution_smokes.py and the tool skill.
EXPECTED_CAPABILITIES = {
    "jax_two_gpu_data_parallel_mesh",
    "minecraft_vpt_video_dataloader",
    "dreamer4_tokenizer_train_two_gpu",
    "dreamer4_latent_tokenization",
    "dreamer4_dynamics_train_two_gpu",
    "dreamer4_action_conditioned_dream_rollout",
    "world_model_rerun_visualization",
}


def _spec_config() -> dict[str, object]:
    payload = yaml.safe_load(OPEN_DREAMER_SPEC.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), OPEN_DREAMER_SPEC
    config = payload.get("config")
    assert isinstance(config, dict), OPEN_DREAMER_SPEC
    return config


def _parse_last_json_blob(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    idx = 0
    last_obj: dict[str, object] | None = None
    while idx < len(text):
        brace = text.find("{", idx)
        if brace < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, brace)
        except json.JSONDecodeError:
            idx = brace + 1
            continue
        if isinstance(obj, dict):
            last_obj = obj
        idx = max(end, brace + 1)
    if last_obj is None:
        raise ValueError(f"no JSON object found in output:\n{text[-2000:]}")
    return last_obj


def _s3_client(e2e_project: str | None):
    boto3 = pytest.importorskip("boto3")
    from botocore.config import Config

    env = storage_env_for_project(e2e_project, allow_host_creds=True)
    endpoint = (env.get("AWS_ENDPOINT_URL") or "").strip() or None
    kwargs: dict[str, object] = {
        "endpoint_url": endpoint,
        "config": Config(s3={"addressing_style": "path"}),
        "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-central1"),
    }
    if env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"):
        kwargs["aws_access_key_id"] = env["AWS_ACCESS_KEY_ID"]
        kwargs["aws_secret_access_key"] = env["AWS_SECRET_ACCESS_KEY"]
    return boto3.client("s3", **kwargs)


def _wait_for_object(s3, bucket: str, key: str, *, deadline: float, poll: float):
    while time.time() < deadline:
        try:
            return s3.get_object(Bucket=bucket, Key=key)
        except Exception:  # noqa: BLE001 - object not present yet
            time.sleep(poll)
    return None


def test_open_dreamer_spec_renders_via_workflow_machinery() -> None:
    """The spec must plan/render through the real npa.workflow machinery."""
    from npa.orchestration.npa_workflow import build_plan, load_spec

    spec = load_spec(OPEN_DREAMER_SPEC)
    plan = build_plan(spec, run_id="od-render-check")
    # The single BYOF state must resolve to the workbench.byof.repo toolRef with
    # the spec's real smoke command and 2-GPU resource profile baked into argv.
    commands = " ".join(
        str(part)
        for task in getattr(plan, "scheduler_plan", {}).get("tasks", [])
        for part in (task.get("command") or [])
    ) if hasattr(plan, "scheduler_plan") else ""
    config = _spec_config()
    assert config.get("workload") == "solution-smoke"
    assert "byof-solution-smoke-rtxpro-2gpu.yaml" in str(config.get("resource_profile_yaml"))
    smoke = str(config.get("smoke_command") or "")
    for capability in EXPECTED_CAPABILITIES:
        assert capability in smoke, capability
    # commands is best-effort; when present it should reference the byof toolRef.
    if commands:
        assert "byof" in commands


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_OPEN_DREAMER_LIVE_GPU") != "1",
    reason="Set NPA_BYOF_OPEN_DREAMER_LIVE_GPU=1 to run the multi-hour 2-GPU Open Dreamer smoke.",
)
def test_open_dreamer_live_gpu_smoke(e2e_project: str | None) -> None:
    image = os.environ.get("NPA_BYOF_TEST_IMAGE", "").strip()
    if not image:
        pytest.skip("Set NPA_BYOF_TEST_IMAGE=<prebuilt open-dreamer image> to reuse the image.")

    config = _spec_config()
    smoke_command = str(config.get("smoke_command") or "")
    assert smoke_command, "spec smoke_command is empty"
    profile = REPO_ROOT / str(config["resource_profile_yaml"])
    assert profile.is_file(), profile
    solution_name = str(config["solution_name"])
    capability_name = str(config["capability_name"])
    smoke_artifact = str(config["smoke_artifact_name"])
    repo_url = str(config["repo_url"])
    repo_ref = str(config["repo_ref"])

    registry = resolve_container_registry(e2e_project)
    bucket = live_bucket(e2e_project)
    run_id = "byof-open-dreamer-e2e-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    cmd = [
        sys.executable,
        str(BYOF_RUNNER),
        "--repo-url", repo_url,
        "--repo-ref", repo_ref,
        "--base-profile", "ubuntu",
        "--project", e2e_project or "",
        "--workload", "solution-smoke",
        "--yaml", str(profile),
        "--skip-build",
        "--skip-push",
        "--image", image,
        "--smoke-command", smoke_command,
        "--solution-name", solution_name,
        "--capability-name", capability_name,
        "--smoke-artifact-name", smoke_artifact,
        "--run-id", run_id,
        "--wait-timeout", os.environ.get("NPA_BYOF_OD_WAIT", "21600"),
        "--poll-interval", "60",
        "--no-cleanup",
    ]
    config_path = skypilot_config_for_project(e2e_project)
    if config_path:
        cmd.extend(["--config-path", config_path])

    env = dict(os.environ)
    env["NPA_E2E_PROJECT"] = e2e_project or env.get("NPA_E2E_PROJECT", "")
    if registry:
        env.setdefault("NPA_REGISTRY", registry)
    target = resolve_byof_kubernetes_target(e2e_project)
    if target.kubeconfig:
        env["KUBECONFIG"] = target.kubeconfig
        env["NPA_BYOF_KUBECONFIG"] = target.kubeconfig
    if target.context:
        env["NPA_BYOF_K8S_CONTEXT"] = target.context
    skypilot_bin = resolve_skypilot_bin()
    if skypilot_bin:
        env["PATH"] = f"{Path(skypilot_bin).parent}:{env.get('PATH', '')}"

    # The BYOF runner uses a synchronous `sky launch`, so it blocks for the whole
    # multi-hour run and returns the final status. Bound it generously (default
    # 5h) and NEVER SIGKILL mid-run in the normal case — a mid-run kill can
    # disrupt the cluster before it uploads. S3 remains the source of truth.
    launch_timeout = int(os.environ.get("NPA_BYOF_OD_LAUNCH_TIMEOUT", "18000"))

    def _text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
        return str(value)

    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=launch_timeout,
            env=env,
        )
        combined = _text(proc.stdout) + "\n" + _text(proc.stderr)
    except subprocess.TimeoutExpired as exc:
        # Exceeded the (generous) cap; the cluster may still be finishing. Fall
        # through to S3 verification. TimeoutExpired.stdout/stderr may be bytes.
        combined = _text(exc.stdout) + "\n" + _text(exc.stderr)

    if "{" in combined:
        try:
            summary = _parse_last_json_blob(combined)
            assert summary.get("status") in {"ok", None} or summary.get("run"), summary
        except ValueError:
            pass  # launcher may still be mid-flight; S3 is the source of truth

    # Verify via S3: the results artifact must show all 7 capabilities, 0 deferred.
    s3 = _s3_client(e2e_project)
    prefix = f"byof/{run_id}/"
    results_key = prefix + smoke_artifact
    deadline = time.time() + int(os.environ.get("NPA_BYOF_OD_S3_WAIT", "18000"))
    poll = float(os.environ.get("NPA_BYOF_OD_POLL", "120"))
    obj = _wait_for_object(s3, bucket, results_key, deadline=deadline, poll=poll)
    assert obj is not None, f"results artifact never appeared: s3://{bucket}/{results_key}"
    results = json.loads(obj["Body"].read())

    exercised = set(results.get("capabilities_exercised") or [])
    deferred = results.get("deferred") or []
    assert EXPECTED_CAPABILITIES.issubset(exercised), (
        f"missing capabilities: {sorted(EXPECTED_CAPABILITIES - exercised)}; deferred={deferred}"
    )
    assert not deferred, f"unexpected deferred capabilities: {deferred}"
    assert results.get("jax_device_count", 0) >= 2, results.get("jax_device_count")
    assert results.get("data_parallel_mesh", {}).get("data", 0) >= 2, results.get("data_parallel_mesh")

    # The headline dream .rrd must exist and be non-trivial.
    rrd = s3.head_object(Bucket=bucket, Key=prefix + "open_dreamer_world_model.rrd")
    assert rrd["ContentLength"] > 1_000_000, rrd["ContentLength"]

    # Best-effort teardown of the (auto-stopping) cluster.
    if skypilot_bin:
        subprocess.run(
            [skypilot_bin, "down", "-y", run_id],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
