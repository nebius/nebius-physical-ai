#!/usr/bin/env bash
# Live verification for NPA agent artifact discovery:
# 1. discover runs, 2. list artifacts, 3. load one artifact.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PROJECT="${NPA_AGENT_PROJECT:-fresh-us-central1}"
NAME="${NPA_AGENT_NAME:-agent}"
NPA_BIN="${NPA_BIN:-$ROOT/npa/.venv/bin/npa}"
PY_BIN="${PY_BIN:-$ROOT/npa/.venv/bin/python}"

AUTH_ENV="${NPA_AGENT_AUTH_ENV:-$HOME/.npa/agents/${PROJECT}/${NAME}/auth.env}"
if [[ ! -f "$AUTH_ENV" ]]; then
  echo "missing auth env: $AUTH_ENV" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$AUTH_ENV"

BASE="$("$NPA_BIN" agent status --project "$PROJECT" --name "$NAME" --json \
  | "$PY_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("public_url","").rstrip("/"))')"
if [[ -z "$BASE" ]]; then
  echo "agent public_url is empty" >&2
  exit 1
fi

export BASE AGENT_USER AGENT_PASSWORD
"$PY_BIN" - <<'PY'
from __future__ import annotations

import base64
import json
import os
import ssl
import time
import urllib.parse
import urllib.request

base = os.environ["BASE"]
auth = base64.b64encode(f"{os.environ['AGENT_USER']}:{os.environ['AGENT_PASSWORD']}".encode()).decode()
ctx = ssl._create_unverified_context()


def request(path: str, *, method: str = "GET", payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Authorization": f"Basic {auth}"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
        return json.loads(resp.read().decode())


def ensure_s3_run() -> str:
    runs = request("/api/artifacts/runs?limit=20")
    if runs.get("ok") and runs.get("runs") and runs.get("source", "s3") != "local":
        return str(runs["runs"][0]["run_id"])
    submit = request("/api/workflows/sim2real/submit", method="POST", payload={})
    run_id = str(submit["run_id"])
    for _ in range(90):
        detail = request(f"/api/workflows/sim2real/runs/{urllib.parse.quote(run_id)}")
        run = detail.get("run", {})
        if run.get("result") in {"completed", "failed"} or run.get("status") in {"completed", "failed"}:
            break
        time.sleep(1)
    assert request(f"/api/workflows/sim2real/runs/{urllib.parse.quote(run_id)}")["run"].get("result") == "completed"
    return run_id


run_id = ensure_s3_run()

# Component 1: discover runs.
runs = request("/api/artifacts/runs?limit=50")
assert runs.get("ok") is True, runs
assert runs.get("source", "s3") != "local", runs
assert not runs.get("s3_error"), runs
assert any(str(item.get("run_id")) == run_id for item in runs.get("runs", [])), runs

# Component 2: list artifacts for a run.
listed = request(f"/api/artifacts/run/{urllib.parse.quote(run_id)}")
assert listed.get("ok") is True, listed
assert listed.get("source", "s3") != "local", listed
assert not listed.get("s3_error"), listed
artifacts = listed.get("artifacts", [])
assert artifacts, listed
preferred = listed.get("preferred") or artifacts[0]
assert preferred.get("key"), preferred

# Component 3: load an artifact.
loaded = request(
    "/api/sim-viz/load-artifact",
    method="POST",
    payload={"run_id": run_id, "key": preferred["key"]},
)
assert loaded.get("ok") is True, loaded
assert loaded.get("source", "s3") != "local", loaded
assert not loaded.get("s3_error"), loaded
assert loaded.get("render"), loaded
sim_viz = loaded.get("sim_viz", {})
assert sim_viz.get("run_id") == run_id, sim_viz

print(json.dumps({
    "ok": True,
    "run_id": run_id,
    "discover_source": runs.get("source", "s3"),
    "artifact_count": len(artifacts),
    "loaded_render": loaded.get("render"),
}, sort_keys=True))
PY
