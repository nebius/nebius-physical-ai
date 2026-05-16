"""SkyPilot managed-jobs e2e replay for the NPA wrapper.

This uses a dedicated `e2e_skypilot` marker instead of `e2e_serverless` because
it launches SkyPilot managed jobs backed by MK8s pods, not Nebius Serverless AI
Jobs. The test is skip-by-default and requires both `NPA_INTEGRATION_E2E=1` and
`NPA_E2E_SKYPILOT=1`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import boto3
import pytest
import yaml
from botocore.exceptions import ClientError, ProfileNotFound

from npa.cluster.api import MK8sClient
from npa.orchestration.skypilot._bin import resolve_sky_bin
from npa.orchestration.skypilot import cleanup_all_for_run, submit_workflow, workflow_status
from npa.orchestration.skypilot.cleanup import run_tag, sky_environment


pytestmark = pytest.mark.e2e_skypilot

CLUSTER_NAME = "npa-workbench-eu-north1"
BUCKET = "YOUR_S3_BUCKET"
S3_PREFIX_ROOT = "skypilot-bootstrap-converge"
S3_ENDPOINT = "https://storage.eu-north1.nebius.cloud:443"
POLL_INTERVAL_SECONDS = 30
MAX_WAIT_SECONDS = 1800
TERMINAL_STATUSES = {
    "SUCCEEDED",
    "CANCELLED",
    "FAILED",
    "FAILED_SETUP",
    "FAILED_PRECHECKS",
    "FAILED_NO_RESOURCE",
    "FAILED_CONTROLLER",
}


@pytest.fixture(autouse=True)
def _require_skypilot_e2e() -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    if os.environ.get("NPA_E2E_SKYPILOT") != "1":
        pytest.skip("NPA_E2E_SKYPILOT not set")


def test_three_stage_dag_replays_through_npa_wrapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = os.environ.get("NPA_SKYPILOT_TEST_RUN_ID") or f"w9skypilot-bootstrap-converge-{uuid.uuid4().hex[:8]}"
    tag = run_tag(run_id)
    sky_bin = str(resolve_sky_bin(os.environ.get("NPA_SKYPILOT_SKY_BIN") or os.environ.get("NPA_SKYPILOT_BIN")))
    submit_timeout = int(os.environ.get("NPA_SKYPILOT_SUBMIT_TIMEOUT_SECONDS", "7200"))
    evidence_dir = Path(os.environ.get("NPA_SKYPILOT_EVIDENCE_DIR", str(tmp_path / "evidence")))
    evidence_dir.mkdir(parents=True, exist_ok=True)

    isolated_root = tmp_path / "sky"
    home = isolated_root / "home"
    _copy_operator_auth(home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SKY_RUNTIME_DIR", str(isolated_root / "sky-runtime"))
    _bootstrap_nebius_token(home, evidence_dir)

    kubeconfig, project_id = _derive_fresh_kubeconfig(home, evidence_dir)
    monkeypatch.setenv("KUBECONFIG", str(kubeconfig))

    base_config = evidence_dir / "skypilot-base-config.yaml"
    _write_base_skypilot_config(base_config, home=home, project_id=project_id)
    monkeypatch.setenv("SKYPILOT_GLOBAL_CONFIG", str(base_config))
    _capture_command(
        [sky_bin, "api", "stop"],
        evidence_dir / "sky-api-stop-before-check.txt",
        env=sky_environment(isolated_root),
        timeout=120,
    )
    check = _capture_command(
        [sky_bin, "check", "--config", str(base_config), "nebius", "kubernetes"],
        evidence_dir / "sky-check.txt",
        env=sky_environment(isolated_root),
        timeout=300,
    )
    assert check.returncode == 0, check.stderr or check.stdout

    s3_prefix = f"{S3_PREFIX_ROOT}/{run_id}"
    yaml_path = _write_three_stage_yaml(tmp_path, run_id=run_id, tag=tag, s3_prefix=s3_prefix)
    shutil.copy2(yaml_path, evidence_dir / "three-stage-wrapper.yaml")

    s3_client = _s3_client()
    _delete_s3_prefix(s3_client, s3_prefix)

    result = submit_workflow(
        yaml_path,
        run_id,
        isolated_config_dir=isolated_root,
        sky_bin=sky_bin,
        timeout=submit_timeout,
    )
    (evidence_dir / "submit-result.json").write_text(json.dumps(result.__dict__, indent=2, sort_keys=True) + "\n")
    assert result.status == "SUBMITTED", result.error or result.stderr
    assert result.job_id

    config_path = Path(result.log_paths["config"])
    final = _wait_for_job(
        result.job_id,
        isolated_root=isolated_root,
        config_path=config_path,
        sky_bin=sky_bin,
        evidence_dir=evidence_dir,
    )
    assert final.status == "SUCCEEDED", final.stderr or final.stdout

    for stage in ("stage-1", "stage-2", "stage-3"):
        _capture_command(
            [sky_bin, "jobs", "logs", "--config", str(config_path), result.job_id, f"{tag}-{stage}"],
            evidence_dir / f"sky-jobs-logs-{stage}.txt",
            env=os.environ.copy(),
            timeout=300,
        )

    marker = s3_client.get_object(Bucket=BUCKET, Key=f"{s3_prefix}/final-marker.json")["Body"].read()
    marker_data = json.loads(marker.decode("utf-8"))
    assert marker_data["chain"] == "stage1->stage2->stage3"
    assert marker_data["verified"] is True

    _delete_s3_prefix(s3_client, s3_prefix)
    cleanup = cleanup_all_for_run(
        run_id,
        isolated_config_dir=isolated_root,
        config_path=config_path,
        sky_bin=sky_bin,
    )
    (evidence_dir / "cleanup-result.json").write_text(json.dumps(cleanup.__dict__, indent=2, sort_keys=True) + "\n")

    _capture_command(
        [sky_bin, "jobs", "queue", "--config", str(config_path), "--skip-finished", "--output", "json"],
        evidence_dir / "sky-jobs-queue-after-cleanup.json",
        env=os.environ.copy(),
        timeout=180,
    )
    _capture_command(
        [sky_bin, "status", "--config", str(config_path), "--refresh"],
        evidence_dir / "sky-status-after-cleanup.txt",
        env=os.environ.copy(),
        timeout=300,
    )
    _capture_command(
        ["kubectl", "get", "pods", "-A", "-o", "name"],
        evidence_dir / "k8s-pods-after-cleanup.txt",
        env=os.environ.copy(),
        timeout=180,
    )

    assert not _s3_prefix_exists(s3_client, s3_prefix)
    assert tag not in (evidence_dir / "sky-jobs-queue-after-cleanup.json").read_text()
    assert tag not in (evidence_dir / "sky-status-after-cleanup.txt").read_text()
    assert tag not in (evidence_dir / "k8s-pods-after-cleanup.txt").read_text()


def _copy_operator_auth(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    for directory in (".npa", ".nebius", ".aws"):
        src = Path.home() / directory
        dst = home / directory
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst, symlinks=True)


def _bootstrap_nebius_token(home: Path, evidence_dir: Path) -> None:
    nebius_dir = home / ".nebius"
    nebius_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["nebius", "iam", "get-access-token"],
        env=_python_env(home),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    (evidence_dir / "nebius-token-bootstrap-redacted.json").write_text(
        json.dumps(
            {
                "cmd": ["nebius", "iam", "get-access-token"],
                "returncode": result.returncode,
                "stdout": "<redacted>" if result.stdout else "",
                "stderr": result.stderr,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    token_path = nebius_dir / "NEBIUS_IAM_TOKEN.txt"
    token_path.write_text(result.stdout.strip() + "\n", encoding="utf-8")
    token_path.chmod(0o600)


def _derive_fresh_kubeconfig(home: Path, evidence_dir: Path) -> tuple[Path, str]:
    status_path = evidence_dir / "npa-cluster-status.json"
    result = _capture_command(
        [
            sys.executable,
            "-c",
            "from npa.cli.main import app; app()",
            "cluster",
            "status",
            "--name",
            CLUSTER_NAME,
            "--format",
            "json",
        ],
        status_path,
        env=_python_env(home),
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    assert rows and rows[0]["cluster_id"], rows
    cluster_id = rows[0]["cluster_id"]
    project_id = rows[0]["project_id"]

    kubeconfig = home / ".npa" / "clusters" / CLUSTER_NAME / "kubeconfig"
    MK8sClient(timeout=180, poll_interval=10.0).get_kubeconfig(
        cluster_id,
        kubeconfig,
        context_name=CLUSTER_NAME,
        external=True,
    )
    assert kubeconfig.exists()
    return kubeconfig, project_id


def _write_base_skypilot_config(path: Path, *, home: Path, project_id: str) -> None:
    tenant_id = _tenant_id(home)
    if not tenant_id:
        pytest.fail("Unable to resolve Nebius tenant ID for SkyPilot base config")
    data = {
        "nebius": {
            "tenant_id": tenant_id,
            "region_configs": {"eu-north1": {"project_id": project_id}},
        },
        "kubernetes": {"allowed_contexts": [CLUSTER_NAME]},
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _tenant_id(home: Path) -> str:
    if os.environ.get("NEBIUS_TENANT_ID"):
        return os.environ["NEBIUS_TENANT_ID"].strip()
    tenant_file = home / ".nebius" / "NEBIUS_TENANT_ID.txt"
    if tenant_file.exists():
        return tenant_file.read_text().strip()
    config_file = home / ".nebius" / "config.yaml"
    if not config_file.exists():
        return ""
    data = yaml.safe_load(config_file.read_text()) or {}
    for key in ("tenant_id", "tenant-id", "tenant"):
        value = data.get(key) if isinstance(data, dict) else ""
        if value:
            return str(value).strip()
    return ""


def _write_three_stage_yaml(tmp_path: Path, *, run_id: str, tag: str, s3_prefix: str) -> Path:
    docs = [
        {"name": run_id, "execution": "serial"},
        _stage_doc(f"{tag}-stage-1", "1", s3_prefix),
        _stage_doc(f"{tag}-stage-2", "2", s3_prefix),
        _stage_doc(f"{tag}-stage-3", "3", s3_prefix),
    ]
    path = tmp_path / "three-stage.yaml"
    path.write_text(yaml.safe_dump_all(docs, sort_keys=False), encoding="utf-8")
    return path


def _stage_doc(name: str, stage: str, s3_prefix: str) -> dict[str, object]:
    return {
        "name": name,
        "resources": {"cloud": "kubernetes", "cpus": 1, "memory": 1},
        "setup": "pip install boto3\n",
        "envs": {
            "AWS_PROFILE": "nebius",
            "NEBIUS_S3_ENDPOINT": S3_ENDPOINT,
            "S3_BUCKET": BUCKET,
            "S3_PREFIX": s3_prefix,
            "STAGE": stage,
        },
        "run": _stage_run_script(),
    }


def _stage_run_script() -> str:
    return """timeout 300 python - <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ProfileNotFound

bucket = os.environ["S3_BUCKET"]
prefix = os.environ["S3_PREFIX"].strip("/")
stage = os.environ["STAGE"]
endpoint = os.environ["NEBIUS_S3_ENDPOINT"]
try:
    session = boto3.session.Session(profile_name=os.environ.get("AWS_PROFILE", "nebius"))
except ProfileNotFound:
    session = boto3.session.Session()
client = session.client("s3", endpoint_url=endpoint)

if stage == "1":
    chain = "stage1"
    client.put_object(Bucket=bucket, Key=f"{prefix}/stage1.txt", Body=chain.encode())
elif stage == "2":
    previous = client.get_object(Bucket=bucket, Key=f"{prefix}/stage1.txt")["Body"].read().decode().strip()
    if previous != "stage1":
        print(f"unexpected stage1 marker: {previous!r}", file=sys.stderr)
        raise SystemExit(1)
    chain = f"{previous}->stage2"
    client.put_object(Bucket=bucket, Key=f"{prefix}/stage2.txt", Body=chain.encode())
else:
    previous = client.get_object(Bucket=bucket, Key=f"{prefix}/stage2.txt")["Body"].read().decode().strip()
    if previous != "stage1->stage2":
        print(f"unexpected stage2 marker: {previous!r}", file=sys.stderr)
        raise SystemExit(1)
    chain = f"{previous}->stage3"
    client.put_object(Bucket=bucket, Key=f"{prefix}/final.txt", Body=chain.encode())
    marker = {"chain": chain, "verified": True, "completed_at": datetime.now(timezone.utc).isoformat()}
    client.put_object(Bucket=bucket, Key=f"{prefix}/final-marker.json", Body=json.dumps(marker).encode())
    print(f"W9_SKYPILOT_DAG_SENTINEL_CHAIN={chain}")

print(json.dumps({"stage": stage, "chain": chain}))
PY
"""


def _wait_for_job(
    job_id: str,
    *,
    isolated_root: Path,
    config_path: Path,
    sky_bin: str,
    evidence_dir: Path,
):
    deadline = time.time() + MAX_WAIT_SECONDS
    last = None
    while time.time() < deadline:
        last = workflow_status(
            job_id,
            isolated_config_dir=isolated_root,
            config_path=config_path,
            sky_bin=sky_bin,
        )
        (evidence_dir / "last-status.json").write_text(json.dumps(last.__dict__, indent=2, sort_keys=True) + "\n")
        if last.status in TERMINAL_STATUSES:
            return last
        time.sleep(POLL_INTERVAL_SECONDS)
    pytest.fail(f"SkyPilot job {job_id} did not finish within {MAX_WAIT_SECONDS}s; last={last}")


def _s3_client():
    try:
        session = boto3.session.Session(profile_name=os.environ.get("AWS_PROFILE", "nebius"))
    except ProfileNotFound:
        session = boto3.session.Session()
    return session.client("s3", endpoint_url=S3_ENDPOINT)


def _delete_s3_prefix(client, prefix: str) -> None:
    while True:
        response = client.list_objects_v2(Bucket=BUCKET, Prefix=prefix.rstrip("/") + "/")
        objects = [{"Key": item["Key"]} for item in response.get("Contents", [])]
        if objects:
            client.delete_objects(Bucket=BUCKET, Delete={"Objects": objects})
        if not response.get("IsTruncated"):
            return


def _s3_prefix_exists(client, prefix: str) -> bool:
    try:
        response = client.list_objects_v2(Bucket=BUCKET, Prefix=prefix.rstrip("/") + "/", MaxKeys=1)
    except ClientError:
        return False
    return bool(response.get("Contents"))


def _capture_command(
    cmd: list[str],
    output_path: Path,
    *,
    env: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    output_path.write_text(
        json.dumps({"cmd": cmd, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return result


def _python_env(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    repo_src = Path(__file__).resolve().parents[3] / "src"
    env["PYTHONPATH"] = str(repo_src) + os.pathsep + env.get("PYTHONPATH", "")
    return env
