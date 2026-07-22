from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.exceptions import ProfileNotFound


pytestmark = [pytest.mark.e2e, pytest.mark.gpu]

ROOT = Path(__file__).resolve().parents[3]
RAW_SKY_YAML = ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot" / "sim-to-real-pipeline.yaml"
DEFAULT_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
GPU_CHAIN = ("H100:1", "H200:1", "A100:1")


def test_raw_sky_sim_to_real_pipeline_writes_run_scoped_s3_artifacts(tmp_path: Path) -> None:
    """Exercise the YAML via raw `sky launch`, not the NPA SDK or CLI wrapper."""

    _require_live_mode()
    sky_bin = _sky_bin()
    run_id = f"sim-to-real-raw-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    s3_prefix = f"sim-to-real/{run_id}"
    bucket = os.environ.get("NPA_E2E_SIM_TO_REAL_BUCKET") or os.environ.get("S3_BUCKET")
    if not bucket:
        pytest.skip("NPA_E2E_SIM_TO_REAL_BUCKET or S3_BUCKET must be set for live sim-to-real e2e")
    endpoint = os.environ.get("NPA_E2E_SIM_TO_REAL_S3_ENDPOINT", os.environ.get("S3_ENDPOINT_URL", DEFAULT_ENDPOINT))
    policy_image = _policy_image()
    credentials_env = _s3_credentials_env()
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=credentials_env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=credentials_env["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=credentials_env.get("AWS_SESSION_TOKEN") or None,
    )
    evidence_dir = Path(os.environ.get("NPA_E2E_SIM_TO_REAL_EVIDENCE_DIR", str(tmp_path / "evidence")))
    evidence_dir.mkdir(parents=True, exist_ok=True)
    workdir = _copy_clean_workdir(tmp_path / "workdir")
    yaml_path = workdir / "npa" / "src" / "npa" / "workflows" / "skypilot" / "sim-to-real-pipeline.yaml"
    assert yaml_path.exists()

    attempts: list[dict[str, Any]] = []
    for gpu in _gpu_chain():
        cluster = _cluster_name(run_id, gpu)
        stdout_path = evidence_dir / f"{cluster}.stdout.txt"
        stderr_path = evidence_dir / f"{cluster}.stderr.txt"
        cmd = _sky_launch_command(
            sky_bin=sky_bin,
            yaml_path=yaml_path,
            workdir=workdir,
            cluster=cluster,
            run_id=run_id,
            bucket=bucket,
            endpoint=endpoint,
            s3_prefix=s3_prefix,
            policy_image=policy_image,
            gpu=gpu,
            has_session_token="AWS_SESSION_TOKEN" in credentials_env,
        )
        env = {**os.environ, **credentials_env}
        attempt = {
            "cluster": cluster,
            "gpu": gpu,
            "command": _redact_command(cmd),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        try:
            result = subprocess.run(
                cmd,
                cwd=workdir,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=int(os.environ.get("NPA_E2E_SIM_TO_REAL_TIMEOUT_SECONDS", "14400")),
                check=False,
            )
            stdout_path.write_text(result.stdout, encoding="utf-8")
            stderr_path.write_text(result.stderr, encoding="utf-8")
            attempt["returncode"] = result.returncode
            if result.returncode == 0:
                artifacts = _assert_s3_artifacts(s3_client, bucket=bucket, prefix=s3_prefix)
                attempt["artifacts"] = artifacts
                attempt["status"] = "passed"
                attempts.append(attempt)
                _write_evidence(evidence_dir, run_id=run_id, bucket=bucket, endpoint=endpoint, attempts=attempts)
                return
            attempt["status"] = "failed"
        finally:
            _sky_down_and_poll(sky_bin, cluster, evidence_dir=evidence_dir)
        attempts.append(attempt)

    _write_evidence(evidence_dir, run_id=run_id, bucket=bucket, endpoint=endpoint, attempts=attempts)
    pytest.fail(f"raw SkyPilot sim-to-real launch failed on all GPU tiers; evidence={evidence_dir}")


def _require_live_mode() -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    if os.environ.get("NPA_DRY_RUN") in {"1", "true", "TRUE"} or os.environ.get("DRY_RUN") in {"1", "true", "TRUE"}:
        pytest.skip("live sim-to-real raw SkyPilot e2e requires writes")


def _sky_bin() -> str:
    sky_bin = os.environ.get("NPA_SKYPILOT_BIN", "/home/ubuntu/.npa/skypilot-venv/bin/sky")
    if not Path(sky_bin).exists():
        pytest.skip(f"SkyPilot binary not found: {sky_bin}")
    return sky_bin


def _s3_credentials_env() -> dict[str, str]:
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        env = {
            "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
            "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
        }
        if os.environ.get("AWS_SESSION_TOKEN"):
            env["AWS_SESSION_TOKEN"] = os.environ["AWS_SESSION_TOKEN"]
        return env

    profile = os.environ.get("AWS_PROFILE", "nebius")
    try:
        credentials = boto3.Session(profile_name=profile).get_credentials()
    except ProfileNotFound:
        pytest.skip(f"AWS profile is not configured: {profile}")
    if credentials is None:
        pytest.skip(f"AWS profile has no credentials: {profile}")
    frozen = credentials.get_frozen_credentials()
    env = {
        "AWS_ACCESS_KEY_ID": frozen.access_key,
        "AWS_SECRET_ACCESS_KEY": frozen.secret_key,
    }
    if frozen.token:
        env["AWS_SESSION_TOKEN"] = frozen.token
    return env


def _policy_image() -> str:
    explicit = os.environ.get("NPA_E2E_SIM_TO_REAL_POLICY_IMAGE") or os.environ.get("POLICY_IMAGE")
    if explicit:
        return explicit
    registry = os.environ.get("NPA_REGISTRY")
    if registry:
        return f"{registry.rstrip('/')}/npa-lerobot-policy:0.1.1"
    registry_id = os.environ.get("NPA_REGISTRY_ID")
    if registry_id:
        return f"cr.eu-north1.nebius.cloud/{registry_id}/npa-lerobot-policy:0.1.1"
    return "npa-lerobot-policy:0.1.1"


def _gpu_chain() -> tuple[str, ...]:
    configured = os.environ.get("NPA_E2E_SIM_TO_REAL_GPU_CHAIN", "")
    if not configured.strip():
        return GPU_CHAIN
    return tuple(gpu.strip() for gpu in configured.split(",") if gpu.strip())


def _copy_clean_workdir(target: Path) -> Path:
    shutil.copytree(
        ROOT,
        target,
        ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", "*.pyc"),
    )
    return target


def _sky_launch_command(
    *,
    sky_bin: str,
    yaml_path: Path,
    workdir: Path,
    cluster: str,
    run_id: str,
    bucket: str,
    endpoint: str,
    s3_prefix: str,
    policy_image: str,
    gpu: str,
    has_session_token: bool,
) -> list[str]:
    cmd = [
        sky_bin,
        "launch",
        "--yes",
        "--cluster",
        cluster,
        "--name",
        cluster,
        "--workdir",
        str(workdir),
        "--infra",
        os.environ.get("NPA_E2E_SIM_TO_REAL_INFRA", "nebius/eu-north1"),
        "--gpus",
        gpu,
        "--env",
        f"NPA_SIM_TO_REAL_RUN_ID={run_id}",
        "--env",
        f"S3_ENDPOINT_URL={endpoint}",
        "--env",
        f"NEBIUS_S3_ENDPOINT={endpoint}",
        "--env",
        f"AWS_ENDPOINT_URL={endpoint}",
        "--env",
        f"S3_BUCKET={bucket}",
        "--env",
        f"NPA_S3_BUCKET={bucket}",
        "--env",
        f"S3_PREFIX={s3_prefix}",
        "--env",
        f"PIPELINE_ROOT_URI=s3://{bucket}/{s3_prefix}/",
        "--env",
        f"INPUT_DATA_URI=s3://{bucket}/datasets/lerobot-pusht/",
        "--env",
        f"LEROBOT_DATASET_URI=s3://{bucket}/datasets/lerobot-pusht/",
        "--env",
        f"RAW_ENVS_URI=s3://{bucket}/{s3_prefix}/raw-envs/",
        "--env",
        f"TRAIN_ENVS_URI=s3://{bucket}/{s3_prefix}/splits/train/",
        "--env",
        f"HELDOUT_ENVS_URI=s3://{bucket}/{s3_prefix}/splits/heldout/",
        "--env",
        f"POLICY_IMAGE={policy_image}",
        "--env",
        f"CHECKPOINT_URI=s3://{bucket}/{s3_prefix}/checkpoints/policy/",
        "--env",
        f"RERUN_RRD_PATH=s3://{bucket}/{s3_prefix}/viz/{run_id}.rrd",
        "--env",
        f"GPU={gpu}",
        "--env",
        "TRAIN_STEPS=2000",
        "--env",
        "TRAIN_STEP_BUDGET=6000",
        "--env",
        "MAX_TRAINING_ITERATIONS=3",
        "--env",
        "EVAL_EPISODES=10",
        "--env",
        "EVAL_BACKEND=pusht",
        "--env",
        "FEEDBACK_SOURCE=rollout",
        "--secret",
        "AWS_ACCESS_KEY_ID",
        "--secret",
        "AWS_SECRET_ACCESS_KEY",
        str(yaml_path),
    ]
    if has_session_token:
        cmd[-1:-1] = ["--secret", "AWS_SESSION_TOKEN"]
    image_id = os.environ.get("NPA_E2E_SIM_TO_REAL_IMAGE_ID")
    if image_id:
        cmd[-1:-1] = ["--image-id", image_id]
    return cmd


def _assert_s3_artifacts(client: Any, *, bucket: str, prefix: str) -> dict[str, str]:
    keys = {
        "roundtrip": f"{prefix}/health/s3-roundtrip.json",
        "dataset_summary": f"{prefix}/datasets/lerobot-summary.json",
        "split": f"{prefix}/splits/train/episode-split.json",
        "checkpoint_manifest": f"{prefix}/checkpoints/policy/policy-checkpoint-manifest.json",
        "report": f"{prefix}/reports/sim-to-real-report.json",
        "rrd": f"{prefix}/viz/{prefix.rsplit('/', 1)[-1]}.rrd",
    }
    for key in keys.values():
        client.head_object(Bucket=bucket, Key=key)
    checkpoint_page = client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/checkpoints/policy/")
    checkpoint_keys = [item["Key"] for item in checkpoint_page.get("Contents", [])]
    assert any(key.endswith("model.safetensors") or key.endswith("pytorch_model.bin") for key in checkpoint_keys)
    report = json.loads(client.get_object(Bucket=bucket, Key=keys["report"])["Body"].read().decode("utf-8"))
    assert report["run_id"].startswith("sim-to-real-raw-")
    assert 0.0 <= float(report["feedback"]["score"]) <= 1.0
    assert report["artifacts"]["root"] == f"s3://{bucket}/{prefix}/"
    component_tiers = {component["name"]: component["tier"] for component in report["components"]}
    assert component_tiers["nebius_s3"] == "WORKS"
    assert component_tiers["s3_artifact_upload"] == "WORKS"
    assert component_tiers["s3_real_artifact_assertions"] == "WORKS"
    return keys


def _sky_down_and_poll(sky_bin: str, cluster: str, *, evidence_dir: Path) -> None:
    down = subprocess.run(
        [sky_bin, "down", "--yes", cluster],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=int(os.environ.get("NPA_E2E_SIM_TO_REAL_TEARDOWN_TIMEOUT_SECONDS", "900")),
        check=False,
    )
    (evidence_dir / f"{cluster}.down.stdout.txt").write_text(down.stdout, encoding="utf-8")
    (evidence_dir / f"{cluster}.down.stderr.txt").write_text(down.stderr, encoding="utf-8")
    deadline = time.monotonic() + int(os.environ.get("NPA_E2E_SIM_TO_REAL_TEARDOWN_POLL_TIMEOUT_SECONDS", "1200"))
    while time.monotonic() < deadline:
        status = subprocess.run(
            [sky_bin, "status", "--refresh"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
            check=False,
        )
        (evidence_dir / f"{cluster}.status-after-down.txt").write_text(status.stdout + status.stderr, encoding="utf-8")
        if cluster not in status.stdout:
            return
        time.sleep(float(os.environ.get("NPA_E2E_SIM_TO_REAL_TEARDOWN_POLL_SECONDS", "30")))
    pytest.fail(f"SkyPilot cluster still present after teardown timeout: {cluster}")


def _cluster_name(run_id: str, gpu: str) -> str:
    return (run_id.replace("sim-to-real-", "s2r-") + "-" + gpu.lower().replace(":", "")).replace("_", "-")[:63]


def _redact_command(cmd: list[str]) -> list[str]:
    return ["<sky>" if part.endswith("/sky") else part for part in cmd]


def _write_evidence(
    evidence_dir: Path,
    *,
    run_id: str,
    bucket: str,
    endpoint: str,
    attempts: list[dict[str, Any]],
) -> None:
    evidence = {
        "run_id": run_id,
        "s3_prefix": f"s3://{bucket}/sim-to-real/{run_id}/",
        "s3_endpoint": endpoint,
        "attempts": attempts,
    }
    (evidence_dir / "raw-sky-evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
