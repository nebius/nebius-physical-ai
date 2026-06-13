#!/usr/bin/env bash
# Run the full 13-stage sim2real demo on THIS machine only.
#
# No Kubernetes, no GPU cluster, no S3, no credentials required.
# All stages use in-process reference payloads; Rerun .rrd is emitted locally.
#
# Usage:
#   ./ops/private/sim2real-rtxpro/run-local-demo.sh
#   VISUALIZE=0 ./ops/private/sim2real-rtxpro/run-local-demo.sh   # skip Rerun server
#   MODE=staged ./ops/private/sim2real-rtxpro/run-local-demo.sh
#
# After run, opens a local Rerun web viewer (http://127.0.0.1:<port>/).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${ROOT}/npa/.venv/bin/python"
RERUN="${ROOT}/npa/.venv/bin/rerun"
NPA="${ROOT}/npa/.venv/bin/npa"

_bootstrap_venv() {
  if [ -x "${PY}" ]; then
    return 0
  fi
  echo "Creating npa/.venv (first run)..."
  if ! command -v python3 >/dev/null; then
    echo "python3 not found — install Python 3.10+ first." >&2
    exit 1
  fi
  python3 -m venv "${ROOT}/npa/.venv"
  "${ROOT}/npa/.venv/bin/python" -m pip install -U pip -q
  "${ROOT}/npa/.venv/bin/python" -m pip install -e "${ROOT}/npa" -q
  echo "Installed npa into ${ROOT}/npa/.venv"
}

_bootstrap_venv

RUN_ID="${RUN_ID:-local-demo-$(date -u +%Y%m%dT%H%M%Sz | tr '[:upper:]' '[:lower:]')}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/sim2real-local/${RUN_ID}}"
MODE="${MODE:-full-loop}"
LOCAL_ONLY="${LOCAL_ONLY:-1}"
VISUALIZE="${VISUALIZE:-1}"

INNER_ITERATIONS="${INNER_ITERATIONS:-1}"
OUTER_ITERATIONS="${OUTER_ITERATIONS:-2}"
ROLLOUT_COUNT="${ROLLOUT_COUNT:-2}"
HELDOUT_ENV_COUNT="${HELDOUT_ENV_COUNT:-4}"
STEPS_PER_ROLLOUT="${STEPS_PER_ROLLOUT:-3}"
THRESHOLD="${SUCCESS_THRESHOLD:-0.45}"
ENV_COUNT="${NPA_ENV_COUNT:-0}"
TRAIN_FRACTION="${NPA_TRAIN_FRACTION:-0.8}"

LOG_DIR="/tmp/sim2real-local"
LOG="${LOG_DIR}/${RUN_ID}.log"
RERUN_LOG="${LOG_DIR}/rerun-${RUN_ID}.log"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# --- Local-only guard: never inherit cluster/S3 env from env.local ---
if [ "${LOCAL_ONLY}" = "1" ]; then
  unset NPA_SIM2REAL_BUCKET S3_BUCKET AWS_ENDPOINT_URL S3_ENDPOINT_URL UPLOAD 2>/dev/null || true
  unset KUBECONFIG NPA_SIM2REAL_K8S_CONTEXT 2>/dev/null || true
fi

if ! "${PY}" -c "import rerun" 2>/dev/null; then
  echo "rerun-sdk missing — install with: ${PY} -m pip install -e ${ROOT}/npa" >&2
  exit 1
fi

common_args=(
  --run-id "${RUN_ID}"
  --output-dir "${OUTPUT_DIR}"
  --inner-iterations "${INNER_ITERATIONS}"
  --outer-iterations "${OUTER_ITERATIONS}"
  --rollout-count "${ROLLOUT_COUNT}"
  --heldout-env-count "${HELDOUT_ENV_COUNT}"
  --steps-per-rollout "${STEPS_PER_ROLLOUT}"
  --threshold "${THRESHOLD}"
  --env-count "${ENV_COUNT}"
  --train-fraction "${TRAIN_FRACTION}"
)

echo "=== Sim2Real LOCAL demo (no cluster, no S3) ===" | tee "${LOG}"
echo "run_id=${RUN_ID} mode=${MODE} output=${OUTPUT_DIR}" | tee -a "${LOG}"
echo "inner=${INNER_ITERATIONS} outer=${OUTER_ITERATIONS} rollouts=${ROLLOUT_COUNT} heldout=${HELDOUT_ENV_COUNT}" | tee -a "${LOG}"

case "${MODE}" in
  full-loop)
    "${PY}" -m npa.workflows.sim2real_loop full-loop "${common_args[@]}" 2>&1 | tee -a "${LOG}"
    ;;
  staged)
    "${PY}" -m npa.workflows.sim2real_loop preamble "${common_args[@]}" 2>&1 | tee -a "${LOG}"
    state_json="${OUTPUT_DIR}/state/workflow_state.json"
    current_quality="$("${PY}" -c "import json; print(json.load(open('${state_json}'))['current_quality'])")"
    for outer in $(seq 1 "${OUTER_ITERATIONS}"); do
      "${PY}" -m npa.workflows.sim2real_loop outer-iteration "${common_args[@]}" \
        --outer-iteration "${outer}" --initial-quality "${current_quality}" 2>&1 | tee -a "${LOG}"
      current_quality="$("${PY}" -c "import json; print(json.load(open('${state_json}'))['current_quality'])")"
      decision="$("${PY}" -c "import json; print(json.load(open('${state_json}'))['final_decision']['decision'])")"
      echo "outer=${outer} quality=${current_quality} decision=${decision}" | tee -a "${LOG}"
      [ "${decision}" = "promote_checkpoint" ] && break
    done
    "${PY}" -m npa.workflows.sim2real_loop finalize "${common_args[@]}" 2>&1 | tee -a "${LOG}"
    ;;
  *)
    echo "Unknown MODE=${MODE} (use full-loop or staged)" >&2
    exit 1
    ;;
esac

REPORT="${OUTPUT_DIR}/reports/sim2real-report.json"
RRD="${OUTPUT_DIR}/reports/sim2real.rrd"

echo "" | tee -a "${LOG}"
echo "=== Summary ===" | tee -a "${LOG}"
"${PY}" - <<PY | tee -a "${LOG}"
import json, sys
from pathlib import Path

report = json.loads(Path("${REPORT}").read_text())
comps = {c["name"]: c for c in report.get("components", [])}
s14 = comps.get("stage_14_rerun_viz", {})
summary = {
    "run_id": report.get("run_id"),
    "status": report.get("status"),
    "decision": report.get("outer_loop", {}).get("latest_decision", {}).get("decision"),
    "success_rate": report.get("outer_loop", {}).get("latest_decision", {}).get("success_rate"),
    "reward_trend": report.get("inner_loop", {}).get("reward_trend"),
    "stage_14_rerun_viz_tier": s14.get("tier"),
    "visualization_status": report.get("visualization", {}).get("status"),
    "local_artifact_dir": report.get("local_artifact_dir"),
}
print(json.dumps(summary, indent=2))
tier = s14.get("tier", "")
if tier != "WORKS":
    print(f"ERROR: stage_14_rerun_viz tier is {tier!r}, expected WORKS", file=sys.stderr)
    sys.exit(1)
PY
if [ "${PIPESTATUS[0]:-0}" -ne 0 ]; then
  echo "Report/tier check failed" >&2
  exit 1
fi

if [ ! -f "${RRD}" ] || [ ! -s "${RRD}" ]; then
  echo "ERROR: ${RRD} missing or empty" | tee -a "${LOG}" >&2
  exit 1
fi
echo "rrd=${RRD} ($(wc -c < "${RRD}") bytes)" | tee -a "${LOG}"

_visualize() {
  local rrd="$1"
  local bind="${RERUN_BIND:-127.0.0.1}"
  local port="${RERUN_WEB_PORT:-}"

  if [ -z "${port}" ]; then
    port="$("${PY}" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"
  fi

  # Native viewer when explicitly requested and DISPLAY available
  if [ "${OPEN_RERUN:-0}" = "1" ] && { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; }; then
    echo "Opening native Rerun viewer..." | tee -a "${LOG}"
    exec "${RERUN}" "${rrd}"
  fi

  pkill -f "rerun.*${rrd}" 2>/dev/null || true
  nohup "${RERUN}" "${rrd}" --web-viewer --web-viewer-port "${port}" --bind "${bind}" \
    > "${RERUN_LOG}" 2>&1 &
  local pid=$!
  local url=""
  for _ in $(seq 1 20); do
    url="$(grep -oE 'Hosting a web-viewer at http[^[:space:]]+' "${RERUN_LOG}" 2>/dev/null \
      | sed 's/Hosting a web-viewer at //' | head -1 || true)"
    [ -n "${url}" ] && break
    sleep 0.5
  done
  if [ -z "${url}" ]; then
    url="http://${bind}:${port}/"
  fi

  echo "" | tee -a "${LOG}"
  echo "=== Rerun visualization (local web viewer) ===" | tee -a "${LOG}"
  echo "  ${url}" | tee -a "${LOG}"
  echo "  pid=${pid} log=${RERUN_LOG}" | tee -a "${LOG}"
  echo "" | tee -a "${LOG}"
  echo "Walkthrough in the viewer (~30 s):" | tee -a "${LOG}"
  echo "  1. rollouts/ → camera frames + critique text overlays" | tee -a "${LOG}"
  echo "  2. signal/reward → per-step RL signal timeseries" | tee -a "${LOG}"
  echo "  3. heldout/scores → held-out env success scores" | tee -a "${LOG}"
  echo "" | tee -a "${LOG}"
  echo "Stop viewer: kill ${pid}  (or: pkill -f 'rerun.*${rrd}')" | tee -a "${LOG}"
  echo "Artifacts: ${OUTPUT_DIR}" | tee -a "${LOG}"
  echo "log=${LOG}" | tee -a "${LOG}"
}

if [ "${VISUALIZE}" = "1" ]; then
  _visualize "${RRD}"
else
  echo "" | tee -a "${LOG}"
  echo "VISUALIZE=0 — open manually:" | tee -a "${LOG}"
  echo "  ${RERUN} ${RRD} --web-viewer" | tee -a "${LOG}"
  echo "log=${LOG}" | tee -a "${LOG}"
fi
