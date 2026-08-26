from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


pytestmark = pytest.mark.smoke


@pytest.fixture
def smoke_env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    npa_home = home / ".npa"
    npa_home.mkdir(parents=True)
    config_path = npa_home / "config.yaml"
    credentials_path = npa_home / "credentials.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "smoke": {
                        "project_id": "project-smoke",
                        "region": "eu-north1",
                        "storage": {
                            "checkpoint_bucket": "s3://bucket/checkpoints/",
                            "endpoint_url": "https://storage.example",
                            "aws_access_key_id": "key",
                            "aws_secret_access_key": "secret",
                        },
                        "workbenches": {},
                    },
                },
                "default_project": "smoke",
            },
            sort_keys=False,
        )
    )
    credentials_path.write_text(yaml.safe_dump({"HF_TOKEN": "hf-smoke"}, sort_keys=False))
    config_path.chmod(0o600)
    credentials_path.chmod(0o600)

    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    state_path = tmp_path / "fake-nebius-state.json"
    log_path = tmp_path / "fake-nebius-calls.jsonl"
    state_path.write_text(json.dumps({"endpoints": {}, "jobs": {}, "next_id": 1}))
    fake = fakebin / "nebius"
    fake.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

state_path = Path(os.environ["NPA_FAKE_NEBIUS_STATE"])
log_path = Path(os.environ["NPA_FAKE_NEBIUS_LOG"])
args = sys.argv[1:]
log_path.write_text(log_path.read_text() + json.dumps(args) + "\\n" if log_path.exists() else json.dumps(args) + "\\n")
state = json.loads(state_path.read_text())

def save():
    state_path.write_text(json.dumps(state))

def option(name, default=""):
    if name not in args:
        return default
    idx = args.index(name)
    return args[idx + 1] if idx + 1 < len(args) else default

def options(name):
    values = []
    i = 0
    while i < len(args):
        if args[i] == name and i + 1 < len(args):
            values.append(args[i + 1])
            i += 2
            continue
        i += 1
    return values

def emit(data):
    print(json.dumps(data))

def endpoint_doc(endpoint):
    return {
        "metadata": {
            "id": endpoint["id"],
            "name": endpoint["name"],
            "parent_id": endpoint["parent_id"],
        },
        "status": {
            "state": endpoint.get("state", "RUNNING"),
            "url": endpoint.get("url", "https://endpoint.example"),
        },
        "spec": endpoint.get("spec", {}),
    }

def job_doc(job, mutate=False):
    if mutate and job["state"] not in {"SUCCEEDED", "FAILED", "CANCELLED"}:
        count = int(job.get("get_count", 0)) + 1
        job["get_count"] = count
        job["state"] = "RUNNING" if count == 1 else "SUCCEEDED"
        save()
    return {
        "metadata": {
            "id": job["id"],
            "name": job["name"],
            "parent_id": job["parent_id"],
            "created_at": "2026-05-13T00:00:00Z",
        },
        "status": {
            "state": job.get("state", "QUEUED"),
            "output_uris": [job.get("output_path", "")],
            "message": "fake log line",
        },
        "spec": job.get("spec", {}),
        "output_path": job.get("output_path", ""),
    }

if args[:3] == ["vpc", "subnet", "list"]:
    emit({"items": [{"metadata": {"id": "vpcsubnet-lerobot", "name": "lerobot-default"}, "status": {"state": "READY"}}]})
    raise SystemExit(0)

if args[:3] == ["ai", "endpoint", "create"]:
    endpoint_id = f"endpoint-smoke-{state['next_id']}"
    state["next_id"] += 1
    endpoint = {
        "id": endpoint_id,
        "name": option("--name"),
        "parent_id": option("--parent-id"),
        "state": "RUNNING",
        "url": "https://endpoint.example",
        "spec": {"image": option("--image"), "platform": option("--platform"), "env": options("--env")},
    }
    state["endpoints"][endpoint_id] = endpoint
    save()
    emit(endpoint_doc(endpoint))
    raise SystemExit(0)

if args[:3] == ["ai", "endpoint", "list"]:
    parent = option("--parent-id")
    emit({"items": [endpoint_doc(e) for e in state["endpoints"].values() if not parent or e["parent_id"] == parent]})
    raise SystemExit(0)

if args[:3] == ["ai", "endpoint", "get"]:
    endpoint_id = option("--id") or (args[3] if len(args) > 3 else "")
    endpoint = state["endpoints"].get(endpoint_id)
    if not endpoint:
        print("endpoint not found", file=sys.stderr)
        raise SystemExit(1)
    emit(endpoint_doc(endpoint))
    raise SystemExit(0)

if args[:3] == ["ai", "endpoint", "delete"]:
    endpoint_id = option("--id") or (args[3] if len(args) > 3 else "")
    state["endpoints"].pop(endpoint_id, None)
    save()
    emit({"status": "deleted", "id": endpoint_id})
    raise SystemExit(0)

if args[:3] in (["ai", "endpoint", "start"], ["ai", "endpoint", "stop"]):
    endpoint_id = option("--id") or (args[3] if len(args) > 3 else "")
    endpoint = state["endpoints"].get(endpoint_id)
    if not endpoint:
        print("endpoint not found", file=sys.stderr)
        raise SystemExit(1)
    endpoint["state"] = "RUNNING" if args[2] == "start" else "STOPPED"
    save()
    emit(endpoint_doc(endpoint))
    raise SystemExit(0)

if args[:3] == ["ai", "endpoint", "logs"]:
    print("fake log line")
    raise SystemExit(0)

if args[:3] == ["ai", "job", "create"]:
    job_id = f"job-smoke-{state['next_id']}"
    state["next_id"] += 1
    env = options("--env")
    output_path = ""
    for item in env:
        if item.startswith("NPA_OUTPUT_PATH="):
            output_path = item.split("=", 1)[1]
    job = {
        "id": job_id,
        "name": option("--name"),
        "parent_id": option("--parent-id"),
        "state": "QUEUED",
        "get_count": 0,
        "output_path": output_path,
        "spec": {
            "image": option("--image"),
            "command": option("--container-command"),
            "platform": option("--platform"),
            "preset": option("--preset"),
            "subnet_id": option("--subnet-id"),
            "env": env,
        },
    }
    state["jobs"][job_id] = job
    save()
    emit(job_doc(job))
    raise SystemExit(0)

if args[:3] == ["ai", "job", "list"]:
    parent = option("--parent-id")
    emit({"items": [job_doc(j) for j in state["jobs"].values() if not parent or j["parent_id"] == parent]})
    raise SystemExit(0)

if args[:3] == ["ai", "job", "get"]:
    job_id = option("--id") or (args[3] if len(args) > 3 else "")
    job = state["jobs"].get(job_id)
    if not job:
        print("job not found", file=sys.stderr)
        raise SystemExit(1)
    emit(job_doc(job, mutate=True))
    raise SystemExit(0)

if args[:3] == ["ai", "job", "get-by-name"]:
    parent = option("--parent-id")
    name = option("--name")
    for job in state["jobs"].values():
        if job["name"] == name and (not parent or job["parent_id"] == parent):
            emit(job_doc(job, mutate=True))
            raise SystemExit(0)
    print("job not found", file=sys.stderr)
    raise SystemExit(1)

if args[:3] == ["ai", "job", "cancel"]:
    job_id = option("--id") or (args[3] if len(args) > 3 else "")
    job = state["jobs"].get(job_id)
    if not job:
        print("job not found", file=sys.stderr)
        raise SystemExit(1)
    job["state"] = "CANCELLED"
    save()
    emit(job_doc(job))
    raise SystemExit(0)

if args[:3] == ["ai", "job", "delete"]:
    job_id = option("--id") or (args[3] if len(args) > 3 else "")
    state["jobs"].pop(job_id, None)
    save()
    emit({"status": "deleted", "id": job_id})
    raise SystemExit(0)

if args[:3] == ["ai", "job", "logs"]:
    print("fake log line")
    raise SystemExit(0)

print("unsupported fake nebius args: " + " ".join(args), file=sys.stderr)
raise SystemExit(2)
"""
    )
    fake.chmod(0o755)

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PATH"] = str(fakebin) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["NPA_FAKE_NEBIUS_STATE"] = str(state_path)
    env["NPA_FAKE_NEBIUS_LOG"] = str(log_path)
    return env


def _run_npa(env: dict[str, str], args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", "from npa.cli.main import app; app()", *args],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )


def _state(env: dict[str, str]) -> dict:
    return json.loads(Path(env["NPA_FAKE_NEBIUS_STATE"]).read_text())


def _train_args(*extra: str, name: str = "smoke-train") -> list[str]:
    return [
        "workbench",
        "lerobot",
        "-p",
        "smoke",
        "-n",
        "lerobot",
        "train",
        "--runtime",
        "serverless",
        "--project-id",
        "project-smoke",
        "--image",
        "registry.example/npa-lerobot:smoke",
        "--subnet-id",
        "vpcsubnet-smoke",
        "--policy-type",
        "act",
        "--dataset",
        "lerobot/pusht",
        "--job-name",
        name,
        "--smoke",
        "--poll-interval",
        "0.1",
        "--wait-timeout",
        "30",
        "--output",
        "json",
        *extra,
    ]


def test_smoke_lerobot_train_serverless_happy_path(smoke_env: dict[str, str]) -> None:
    result = _run_npa(smoke_env, _train_args(name="smoke-happy"))

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "succeeded"
    assert payload["output_path"].startswith("s3://")


def test_smoke_lerobot_train_serverless_submit_only(smoke_env: dict[str, str]) -> None:
    result = _run_npa(smoke_env, _train_args("--submit-only", name="smoke-submit"))

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "submitted"
    assert payload["job_id"] in _state(smoke_env)["jobs"]


def test_smoke_lerobot_train_serverless_cancel(smoke_env: dict[str, str]) -> None:
    submit = _run_npa(smoke_env, _train_args("--submit-only", name="smoke-cancel"))
    assert submit.returncode == 0, submit.stderr + submit.stdout
    job_id = json.loads(submit.stdout)["job_id"]

    cancel = subprocess.run(
        ["nebius", "ai", "job", "cancel", "--id", job_id, "--format", "json"],
        env=smoke_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert cancel.returncode == 0, cancel.stderr
    assert json.loads(cancel.stdout)["status"]["state"] == "CANCELLED"


def test_smoke_lerobot_train_serverless_output_path_is_s3_uri(smoke_env: dict[str, str]) -> None:
    result = _run_npa(smoke_env, _train_args("--submit-only", name="smoke-output"))
    assert result.returncode == 0, result.stderr + result.stdout

    job = next(iter(_state(smoke_env)["jobs"].values()))
    env_values = job["spec"]["env"]
    output_env = [item for item in env_values if item.startswith("NPA_OUTPUT_PATH=")]
    assert output_env
    assert output_env[0].split("=", 1)[1].startswith("s3://")


def test_smoke_lerobot_train_serverless_b300_diffusion_warning(smoke_env: dict[str, str]) -> None:
    args = _train_args("--submit-only", "--gpu-type", "b300", name="smoke-b300")
    args[args.index("--policy-type") + 1] = "diffusion"

    result = _run_npa(smoke_env, args)

    assert result.returncode == 0, result.stderr + result.stdout
    assert "B300 is ~2.5x slower than H200 on Diffusion Policy" in result.stderr + result.stdout
    job = next(iter(_state(smoke_env)["jobs"].values()))
    assert job["spec"]["platform"] == "gpu-b300-sxm"


def test_smoke_lerobot_train_serverless_idempotent_submit(smoke_env: dict[str, str]) -> None:
    first = _run_npa(smoke_env, _train_args("--submit-only", name="smoke-idempotent"))
    second = _run_npa(smoke_env, _train_args("--submit-only", name="smoke-idempotent"))

    assert first.returncode == 0, first.stderr + first.stdout
    assert second.returncode == 0, second.stderr + second.stdout
    assert json.loads(second.stdout)["status"] == "existing"
    assert len(_state(smoke_env)["jobs"]) == 1


def test_smoke_lerobot_train_serverless_s3_input_command(smoke_env: dict[str, str]) -> None:
    result = _run_npa(
        smoke_env,
        _train_args(
            "--submit-only",
            "--input-path",
            "s3://bucket/datasets/pusht/",
            name="smoke-s3-input",
        ),
    )

    assert result.returncode == 0, result.stderr + result.stdout
    job = next(iter(_state(smoke_env)["jobs"].values()))
    assert "download_file" in job["spec"]["command"]
    assert "--dataset.root=/tmp/lerobot_dataset/pusht" in job["spec"]["command"]
