#!/usr/bin/env bash
# Sync a completed sim2real run from S3 for offline walkthrough (report + Rerun .rrd).
# Credentials read from ~/.npa/credentials.yaml at runtime — never committed.
# Usage: prestage-offline-run.sh <run-id> [local-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${ROOT}/npa/.venv/bin/python"

# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"

if [ ! -x "${PY}" ]; then
  if command -v python3 >/dev/null; then
    python3 -m venv "${ROOT}/npa/.venv"
    "${PY}" -m pip install -U pip -q
    "${PY}" -m pip install -e "${ROOT}/npa" -q
  else
    echo "python3 required to bootstrap npa/.venv" >&2
    exit 1
  fi
fi

RUN_ID="${1:?usage: prestage-offline-run.sh <run-id> [local-dir]}"
LOCAL_DIR="${2:-/tmp/sim2real-prestage/${RUN_ID}}"
PREFIX="${S3_PREFIX:-sim2real-b}"

readarray -t _npa_cfg < <(operator_read_config "${ROOT}")
BUCKET="${S3_BUCKET:-${_npa_cfg[0]:-}}"
ENDPOINT="${S3_ENDPOINT:-${_npa_cfg[1]:-https://storage.eu-north1.nebius.cloud}}"
if [ -z "${BUCKET}" ]; then
  echo "Set storage.bucket in ~/.npa/config.yaml or S3_BUCKET" >&2
  exit 1
fi

mkdir -p "${LOCAL_DIR}"

echo "Syncing s3://${BUCKET}/${PREFIX}/${RUN_ID}/ -> ${LOCAL_DIR}/"

export PRESTAGE_RUN_ID="${RUN_ID}" PRESTAGE_BUCKET="${BUCKET}" PRESTAGE_PREFIX="${PREFIX}"
export PRESTAGE_LOCAL_DIR="${LOCAL_DIR}" PRESTAGE_ENDPOINT="${ENDPOINT}"

"${PY}" - <<'PY'
import json, os, sys, yaml
from pathlib import Path
import boto3
from botocore.config import Config

run_id = os.environ["PRESTAGE_RUN_ID"]
bucket = os.environ["PRESTAGE_BUCKET"]
prefix = f"{os.environ['PRESTAGE_PREFIX']}/{run_id}/"
local_dir = Path(os.environ["PRESTAGE_LOCAL_DIR"])
endpoint = os.environ["PRESTAGE_ENDPOINT"]

creds_path = Path.home() / ".npa" / "credentials.yaml"
if not creds_path.exists():
    print("ERROR: ~/.npa/credentials.yaml required for S3 sync", file=sys.stderr)
    sys.exit(1)
creds = yaml.safe_load(creds_path.read_text())
s = creds.get("storage") or {}
if not s.get("aws_access_key_id") or not s.get("aws_secret_access_key"):
    print("ERROR: storage credentials missing in ~/.npa/credentials.yaml", file=sys.stderr)
    sys.exit(1)
client = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=s["aws_access_key_id"],
    aws_secret_access_key=s["aws_secret_access_key"],
    config=Config(signature_version="s3v4"),
    region_name="eu-north1",
)

keys = [
    "reports/sim2real-report.json",
    "reports/sim2real.rrd",
    "stage_01_trigger/trigger.json",
    "stage_02_assets/external_stub.json",
    "augment/manifest.json",
    "envs/train/manifest.json",
    "envs/heldout/manifest.json",
    "eval/heldout/report.json",
    "outer_loop/decision.json",
    "stage_12_external_validation/external_stub.json",
    "stage_13_retrigger/retrigger.json",
    "state/workflow_state.json",
]

missing = []
for rel in keys:
    key = prefix + rel
    dest = local_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(bucket, key, str(dest))
        print(f"  OK  {rel} ({dest.stat().st_size}b)")
    except Exception as exc:
        missing.append((rel, str(exc)))
        print(f"  MISS {rel}: {exc}")

report_path = local_dir / "reports" / "sim2real-report.json"
rrd_path = local_dir / "reports" / "sim2real.rrd"
if report_path.exists():
    report = json.loads(report_path.read_text())
    comps = {c["name"]: c for c in report.get("components", [])}
    s14 = comps.get("stage_14_rerun_viz", {})
    print()
    print("stage_14_rerun_viz tier:", s14.get("tier", "MISSING"))
    print("visualization status:", report.get("visualization", {}).get("status"))
else:
    print("WARN: report missing — run may be incomplete on S3", file=sys.stderr)

if not rrd_path.exists() or rrd_path.stat().st_size == 0:
    print("ERROR: reports/sim2real.rrd missing — check stage_14 tier on cluster run", file=sys.stderr)
    sys.exit(1)

print()
print("Open locally:")
print(f"  rerun {rrd_path} --web-viewer")
PY
