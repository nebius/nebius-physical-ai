from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import pytest


@dataclass(frozen=True)
class BYOVMTarget:
    host: str
    ssh_key: str
    gpu_count: int
    ssh_user: str = "ubuntu"
    project: str = "multi-gpu-byovm"


@dataclass
class CLIResult:
    returncode: int
    stdout: str
    stderr: str = ""
    gpu_snapshots: list[list[int]] | None = None


def _redact_secrets(text: str) -> str:
    patterns = (
        r"(--tf-var\s+nebius_api_key=)\S+",
        r"(--tf-var\s+nebius_secret_key=)\S+",
    )
    redacted = text
    for pattern in patterns:
        redacted = re.sub(pattern, r"\1<redacted>", redacted)
    return redacted


def _required_env() -> dict[str, str]:
    names = (
        "NPA_TEST_BYOVM_HOST",
        "NPA_TEST_BYOVM_SSH_KEY",
        "NPA_TEST_BYOVM_GPU_COUNT",
    )
    return {name: os.environ.get(name, "") for name in names}


@pytest.fixture(scope="session")
def byovm_target() -> BYOVMTarget:
    env = _required_env()
    missing = [name for name, value in env.items() if not value]
    if missing:
        pytest.skip("BYOVM multi-GPU tests require " + ", ".join(missing))

    try:
        gpu_count = int(env["NPA_TEST_BYOVM_GPU_COUNT"])
    except ValueError:
        pytest.skip("NPA_TEST_BYOVM_GPU_COUNT must be an integer")
    if gpu_count < 2:
        pytest.skip("BYOVM multi-GPU tests require at least two GPUs")

    return BYOVMTarget(
        host=env["NPA_TEST_BYOVM_HOST"],
        ssh_key=env["NPA_TEST_BYOVM_SSH_KEY"],
        gpu_count=gpu_count,
        ssh_user=os.environ.get("NPA_TEST_BYOVM_SSH_USER", "ubuntu"),
        project=os.environ.get("NPA_TEST_BYOVM_PROJECT", "multi-gpu-byovm"),
    )


@pytest.fixture(scope="session")
def npa_base_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "src"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    return env


@pytest.fixture(scope="session")
def s3_prefix() -> str:
    prefix = os.environ.get("NPA_TEST_BYOVM_S3_PREFIX") or os.environ.get("NPA_CHECKPOINT_BUCKET")
    if not prefix:
        pytest.skip("Set NPA_TEST_BYOVM_S3_PREFIX or NPA_CHECKPOINT_BUCKET for S3 assertions")
    if not prefix.startswith("s3://"):
        pytest.skip("S3 prefix must be an s3:// URI")
    return prefix.rstrip("/") + "/multi-gpu/" + uuid.uuid4().hex + "/"


@pytest.fixture
def unique_name(request: pytest.FixtureRequest) -> str:
    return request.node.name.replace("[", "-").replace("]", "").replace("/", "-") + "-" + uuid.uuid4().hex[:8]


@pytest.fixture
def run_npa(npa_base_env: dict[str, str]):
    def _run(args: Iterable[str], *, timeout: int = 600, check: bool = True) -> CLIResult:
        proc = subprocess.run(
            [sys.executable, "-c", "from npa.cli.main import app; app()", *args],
            env=npa_base_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        result = CLIResult(proc.returncode, proc.stdout, proc.stderr)
        if check and proc.returncode != 0:
            command = _redact_secrets("npa " + " ".join(args))
            raise AssertionError(
                f"{command} failed with {proc.returncode}\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return result

    return _run


def npa_args(tool: str, target: BYOVMTarget, name: str) -> list[str]:
    return ["workbench", tool, "-p", target.project, "-n", name]


def deploy_byovm_args(tool: str, target: BYOVMTarget, name: str, gpu_count: int) -> list[str]:
    return [
        *npa_args(tool, target, name),
        "deploy",
        "--runtime",
        "byovm",
        "--host",
        target.host,
        "--ssh-key",
        target.ssh_key,
        "--ssh-user",
        target.ssh_user,
        "--gpu-count",
        str(gpu_count),
        "--tf-var",
        f"s3_bucket={_s3_bucket_name()}",
        "--tf-var",
        f"s3_endpoint={os.environ.get('AWS_ENDPOINT_URL', '')}",
        "--tf-var",
        f"nebius_api_key={os.environ.get('AWS_ACCESS_KEY_ID', '')}",
        "--tf-var",
        f"nebius_secret_key={os.environ.get('AWS_SECRET_ACCESS_KEY', '')}",
    ]


def cleanup_workbench(run_npa, tool: str, target: BYOVMTarget, name: str) -> None:
    run_npa(
        [
            *npa_args(tool, target, name),
            "deploy",
            "--runtime",
            "byovm",
            "--destroy",
        ],
        timeout=120,
        check=False,
    )


def run_with_gpu_poll(
    args: list[str],
    *,
    target: BYOVMTarget,
    env: dict[str, str],
    timeout: int = 1800,
    poll_interval: float = 5.0,
) -> CLIResult:
    snapshots: list[list[int]] = []
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as out:
        proc = subprocess.Popen(
            [sys.executable, "-c", "from npa.cli.main import app; app()", *args],
            env=env,
            text=True,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + timeout
        while proc.poll() is None:
            if time.monotonic() > deadline:
                proc.kill()
                raise TimeoutError(f"npa {' '.join(args)} timed out after {timeout}s")
            snapshot = poll_gpu_utilization(target)
            if snapshot:
                snapshots.append(snapshot)
                print(f"GPU utilization snapshot: {snapshot}", flush=True)
            time.sleep(poll_interval)
        out.seek(0)
        stdout = out.read()
    return CLIResult(proc.returncode or 0, stdout, gpu_snapshots=snapshots)


def poll_gpu_utilization(target: BYOVMTarget) -> list[int]:
    ssh = shutil.which("ssh")
    if not ssh:
        return []
    proc = subprocess.run(
        [
            ssh,
            "-i",
            target.ssh_key,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            f"{target.ssh_user}@{target.host}",
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )
    if proc.returncode != 0:
        return []
    values: list[int] = []
    for line in proc.stdout.splitlines():
        match = re.search(r"\d+", line)
        if match:
            values.append(int(match.group(0)))
    return values


def query_gpu_names(target: BYOVMTarget) -> list[str]:
    ssh = shutil.which("ssh")
    if not ssh:
        return []
    proc = subprocess.run(
        [
            ssh,
            "-i",
            target.ssh_key,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            f"{target.ssh_user}@{target.host}",
            "nvidia-smi --query-gpu=name --format=csv,noheader",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def assert_visible_gpus_used(snapshots: list[list[int]] | None, expected_count: int) -> None:
    assert snapshots, "no nvidia-smi utilization snapshots were captured"
    maxima = [0] * expected_count
    for snapshot in snapshots:
        for idx, value in enumerate(snapshot[:expected_count]):
            maxima[idx] = max(maxima[idx], value)
    assert all(value > 0 for value in maxima), f"expected all GPUs to show >0% utilization, got maxima={maxima}"


def assert_s3_has_objects(uri: str) -> None:
    import boto3

    parsed = urlparse(uri)
    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
    )
    prefix = parsed.path.lstrip("/")
    resp = client.list_objects_v2(Bucket=parsed.netloc, Prefix=prefix, MaxKeys=10)
    objects = resp.get("Contents", [])
    if not objects and prefix and not prefix.endswith("/"):
        resp = client.list_objects_v2(Bucket=parsed.netloc, Prefix=prefix + "/", MaxKeys=10)
        objects = resp.get("Contents", [])
    assert objects, f"expected S3 objects under {uri}"
    assert all(obj.get("Size", 0) > 0 for obj in objects), f"expected non-empty S3 objects under {uri}"


def parse_loss_values(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"(?:train[_/ ]loss|loss)\D{0,20}([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE):
        values.append(float(match.group(1)))
    return values


def parse_fps_values(text: str) -> list[float]:
    return [float(value.replace(",", "")) for value in re.findall(r"Running at\s+([0-9,.]+)\s+FPS", text)]


def _s3_bucket_name() -> str:
    uri = os.environ.get("NPA_TEST_BYOVM_S3_PREFIX") or os.environ.get("NPA_CHECKPOINT_BUCKET", "")
    if uri.startswith("s3://"):
        return urlparse(uri).netloc
    return uri
