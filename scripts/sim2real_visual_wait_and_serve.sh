#!/usr/bin/env bash
# Wait for a Sim2Real run to produce reports/sim2real.rrd, then deploy a Rerun
# viewer and keep a local .rrd copy for direct inspection.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/npa/.venv/bin/python"
export PYTHONPATH="${ROOT}/npa/src:${PYTHONPATH:-}"

RUN_ID="${1:-${RUN_ID:-}}"
if [ -z "${RUN_ID}" ]; then
  RUN_ID="$(basename "$(readlink -f /tmp/sim2real-real-success/latest)")"
fi
if [ -z "${RUN_ID}" ]; then
  echo "RUN_ID is required" >&2
  exit 2
fi

OUT_ROOT="${OUT_ROOT:-/tmp/sim2real-real-success}"
OUT="${OUT_ROOT}/${RUN_ID}"
mkdir -p "${OUT}/reports"
LOG="${OUT}/visual-serve.log"
exec > >(tee -a "${LOG}") 2>&1

echo "=== sim2real visual wait-and-serve ==="
echo "run_id=${RUN_ID}"
echo "log=${LOG}"
date -u +"started_at=%Y-%m-%dT%H:%M:%SZ"

for env_file in "${HOME}/.npa/sim2real-operator.env" "${HOME}/.npa/live-e2e.env"; do
  if [ -f "${env_file}" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
done
unset NEBIUS_IAM_TOKEN

CONFIG_JSON="${OUT}/visual-config.json"
"${PY}" - "${CONFIG_JSON}" <<'PY'
import json
import os
from pathlib import Path

from npa.deploy.images import supported_tool_version
from npa.workflows.sim2real.monitor import load_operator_config, resolve_kubeconfig

operator = load_operator_config()
registry = operator.registry.rstrip("/")
payload = {
    "BUCKET": operator.bucket,
    "ENDPOINT": operator.endpoint_url,
    "CTX": operator.k8s_context,
    "KUBECONFIG_PATH": str(resolve_kubeconfig(operator.k8s_context)),
    "RERUN_IMAGE": os.environ.get("RERUN_IMAGE")
    or os.environ.get("NPA_RERUN_VIEWER_IMAGE")
    or f"{registry}/npa-rerun-viewer:{supported_tool_version('rerun-viewer')}",
}
Path(os.sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

eval "$("${PY}" - "${CONFIG_JSON}" <<'PY'
import json
import shlex
import sys
data = json.loads(open(sys.argv[1], encoding="utf-8").read())
for key, value in data.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"
export AWS_ENDPOINT_URL="${ENDPOINT}"
export S3_ENDPOINT_URL="${ENDPOINT}"
export KUBECONFIG="${KUBECONFIG_PATH}"

RRD_URI="s3://${BUCKET}/sim2real-b/${RUN_ID}/reports/sim2real.rrd"
LOCAL_RRD="${OUT}/reports/sim2real.rrd"
echo "rrd_uri=${RRD_URI}"
echo "local_rrd=${LOCAL_RRD}"

deadline_seconds="${VISUAL_DEADLINE_SECONDS:-28800}"
start_epoch="$(date +%s)"
while true; do
  now="$(date +%s)"
  elapsed=$((now - start_epoch))
  echo "visual_wait elapsed=${elapsed}s"
  if "${PY}" -m npa.cli.main workbench sim2real rerun serve \
      --run-id "${RUN_ID}" \
      --kubeconfig "${KUBECONFIG_PATH}" \
      --namespace default \
      --s3-bucket "${BUCKET}" \
      --s3-prefix sim2real-b \
      --s3-endpoint "${ENDPOINT}" \
      --rrd-uri "${RRD_URI}" \
      --rerun-image "${RERUN_IMAGE}" \
      --local-record \
      --local-rrd-path "${LOCAL_RRD}" \
      --output json > "${OUT}/visual-rerun-serve.json" 2>"${OUT}/visual-rerun-serve.err"; then
    cat "${OUT}/visual-rerun-serve.json"
    date -u +"finished_at=%Y-%m-%dT%H:%M:%SZ"
    echo "VISUAL_READY run_id=${RUN_ID} local_rrd=${LOCAL_RRD}"
    exit 0
  fi
  tail -20 "${OUT}/visual-rerun-serve.err" || true
  if [ "${elapsed}" -gt "${deadline_seconds}" ]; then
    echo "visual_timeout=true"
    exit 1
  fi
  sleep "${VISUAL_POLL_SECONDS:-60}"
done
