from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest
import yaml


pytestmark = pytest.mark.smoke


class _CosmosHandler(BaseHTTPRequestHandler):
    def _write_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json({"status": "ok", "model": "smoke-model", "loaded": True})
            return
        if self.path.startswith("/jobs/"):
            self._write_json({"job_id": self.path.rsplit("/", 1)[-1], "status": "completed", "result": "ok"})
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or "0")
        if length:
            self.rfile.read(length)
        if self.path == "/infer":
            self._write_json({"job_id": "job-smoke", "status": "running"})
            return
        if self.path == "/serve":
            self._write_json({"status": "serving", "model": "smoke-model"})
            return
        self.send_error(404)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


@pytest.fixture
def cosmos_server() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CosmosHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def smoke_env(tmp_path: Path, cosmos_server: str) -> dict[str, str]:
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
                        "workbenches": {},
                    },
                },
            },
            sort_keys=False,
        )
    )
    credentials_path.write_text("{}\n")
    config_path.chmod(0o600)
    credentials_path.chmod(0o600)

    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    state_path = tmp_path / "fake-nebius-state.json"
    log_path = tmp_path / "fake-nebius-calls.jsonl"
    state_path.write_text(json.dumps({"endpoints": {}, "next_id": 1}))
    fake = fakebin / "nebius"
    fake.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

state_path = Path(os.environ["NPA_FAKE_NEBIUS_STATE"])
log_path = Path(os.environ["NPA_FAKE_NEBIUS_LOG"])
endpoint_url = os.environ["NPA_FAKE_SERVERLESS_URL"]
args = sys.argv[1:]
log_path.write_text(log_path.read_text() + json.dumps(args) + "\\n" if log_path.exists() else json.dumps(args) + "\\n")
state = json.loads(state_path.read_text())

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
            "url": endpoint.get("url", endpoint_url),
        },
        "spec": endpoint.get("spec", {}),
    }

if args[:3] == ["ai", "endpoint", "create"]:
    endpoint_id = f"endpoint-smoke-{state['next_id']}"
    state["next_id"] += 1
    endpoint = {
        "id": endpoint_id,
        "name": option("--name"),
        "parent_id": option("--parent-id"),
        "state": "RUNNING",
        "url": endpoint_url,
        "spec": {
            "image": option("--image"),
            "platform": option("--platform"),
            "preset": option("--preset"),
            "container_ports": options("--container-port"),
            "env": options("--env"),
            "volumes": options("--volume"),
        },
    }
    state["endpoints"][endpoint_id] = endpoint
    state_path.write_text(json.dumps(state))
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
    state_path.write_text(json.dumps(state))
    emit({"status": "deleted", "id": endpoint_id})
    raise SystemExit(0)

if args[:3] in (["ai", "endpoint", "start"], ["ai", "endpoint", "stop"]):
    endpoint_id = option("--id") or (args[3] if len(args) > 3 else "")
    endpoint = state["endpoints"].get(endpoint_id)
    if not endpoint:
        print("endpoint not found", file=sys.stderr)
        raise SystemExit(1)
    endpoint["state"] = "RUNNING" if args[2] == "start" else "STOPPED"
    state_path.write_text(json.dumps(state))
    emit(endpoint_doc(endpoint))
    raise SystemExit(0)

if args[:3] == ["ai", "endpoint", "logs"]:
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
    env["NPA_FAKE_SERVERLESS_URL"] = cosmos_server
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


def _deploy_args(*extra: str) -> list[str]:
    return [
        "workbench",
        "cosmos",
        "-p",
        "smoke",
        "-n",
        "cosmos",
        "deploy",
        "--runtime",
        "serverless",
        "--image",
        "registry.example/npa-cosmos:smoke",
        "--platform",
        "gpu-h200-sxm",
        "--preset",
        "1gpu-16vcpu-200gb",
        "--wait",
        *extra,
    ]


def test_smoke_cosmos_serverless_deploy_status_teardown(smoke_env: dict[str, str]) -> None:
    deploy = _run_npa(smoke_env, _deploy_args())
    assert deploy.returncode == 0, deploy.stderr + deploy.stdout
    assert "runtime: serverless" in deploy.stdout

    status = _run_npa(smoke_env, ["workbench", "cosmos", "-p", "smoke", "-n", "cosmos", "status"])
    assert status.returncode == 0, status.stderr + status.stdout
    assert "server: up" in status.stdout
    assert "serverless_status: running" in status.stdout

    teardown = _run_npa(smoke_env, ["workbench", "cosmos", "-p", "smoke", "-n", "cosmos", "teardown", "--yes"])
    assert teardown.returncode == 0, teardown.stderr + teardown.stdout
    assert "status: deleted" in teardown.stdout


def test_smoke_cosmos_serverless_serve_prewarm(smoke_env: dict[str, str]) -> None:
    assert _run_npa(smoke_env, _deploy_args()).returncode == 0

    serve = _run_npa(smoke_env, ["workbench", "cosmos", "-p", "smoke", "-n", "cosmos", "serve"])

    assert serve.returncode == 0, serve.stderr + serve.stdout
    assert "status: prewarmed" in serve.stdout


def test_smoke_cosmos_serverless_infer(smoke_env: dict[str, str]) -> None:
    assert _run_npa(smoke_env, _deploy_args()).returncode == 0

    infer = _run_npa(
        smoke_env,
        [
            "workbench",
            "cosmos",
            "-p",
            "smoke",
            "-n",
            "cosmos",
            "infer",
            "--prompt",
            "robot arm",
            "--poll-interval",
            "0",
        ],
    )

    assert infer.returncode == 0, infer.stderr + infer.stdout
    assert "job_id: job-smoke" in infer.stdout
    assert "Generation complete" in infer.stdout


def test_smoke_cosmos_serverless_replace(smoke_env: dict[str, str]) -> None:
    assert _run_npa(smoke_env, _deploy_args()).returncode == 0

    replace = _run_npa(smoke_env, _deploy_args("--replace", "--yes"))

    assert replace.returncode == 0, replace.stderr + replace.stdout
    config_path = Path(smoke_env["HOME"]) / ".npa" / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    endpoint_id = config["projects"]["smoke"]["workbenches"]["cosmos"]["serverless"]["endpoint_id"]
    assert endpoint_id == "endpoint-smoke-2"


def test_smoke_cosmos_serverless_dry_run_does_not_call_nebius(smoke_env: dict[str, str]) -> None:
    dry_run = _run_npa(smoke_env, _deploy_args("--dry-run", "--output", "json"))

    assert dry_run.returncode == 0, dry_run.stderr + dry_run.stdout
    assert '"status": "dry_run"' in dry_run.stdout
    log_path = Path(smoke_env["NPA_FAKE_NEBIUS_LOG"])
    assert not log_path.exists() or log_path.read_text() == ""
