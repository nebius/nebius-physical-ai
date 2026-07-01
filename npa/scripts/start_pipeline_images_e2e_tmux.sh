#!/usr/bin/env bash
# Real-infra e2e for renamed pipeline images (envgen, reference-policy, loop-eval).
#
# Runs build (optional), serverless capability e2e via pytest, and an optional
# cursor-agent patch loop on failure.
#
# Usage:
#   ./npa/scripts/start_pipeline_images_e2e_tmux.sh
#   ./npa/scripts/start_pipeline_images_e2e_tmux.sh --with-build --with-cursor-agent
#   ./npa/scripts/start_pipeline_images_e2e_tmux.sh --skip-build
#
# Attach:  tmux attach -t pipeline-images-e2e
# Logs:    /tmp/pipeline-images-e2e/<run_id>/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${ROOT}/npa/.venv/bin/python"
STATE_DIR="${PIPELINE_E2E_STATE_DIR:-/tmp/pipeline-images-e2e}"
RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)"
LOG_ROOT="${STATE_DIR}/${RUN_ID}"
SESSION="${PIPELINE_E2E_TMUX_SESSION:-pipeline-images-e2e}"
BRANCH="${PIPELINE_E2E_SOURCE_REF:-$(git -C "${ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"
REGISTRY="${REGISTRY:-cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw}"
NEBIUS_REGISTRY_PROFILE="${NEBIUS_REGISTRY_PROFILE:-agent-sa}"
NPA_E2E_PIPELINE_GPU="${NPA_E2E_PIPELINE_GPU:-h200}"

WITH_BUILD=1
WITH_CURSOR=0
SKIP_BUILD=0

usage() {
  sed -n '2,14p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-build) WITH_BUILD=1; shift ;;
    --skip-build) SKIP_BUILD=1; WITH_BUILD=0; shift ;;
    --with-cursor-agent) WITH_CURSOR=1; shift ;;
    --session) SESSION="${2:?}"; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "${LOG_ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Creating venv at ${ROOT}/npa/.venv" >&2
  python3 -m venv "${ROOT}/npa/.venv"
  "${ROOT}/npa/.venv/bin/pip" install -q -e "${ROOT}/npa[dev]"
fi

_resolve_project() {
  "${PYTHON}" - <<'PY'
import os, yaml
from pathlib import Path
for key in ("NPA_E2E_SERVERLESS_PROJECT", "NEBIUS_PROJECT_ID", "NPA_PROJECT_ID"):
    v = os.environ.get(key, "").strip()
    if v:
        print(v)
        raise SystemExit(0)
p = Path.home() / ".npa" / "credentials.yaml"
if p.is_file():
    data = yaml.safe_load(p.read_text()) or {}
    v = str((data.get("nebius") or {}).get("project_id", "")).strip()
    if v:
        print(v)
        raise SystemExit(0)
raise SystemExit("no project id")
PY
}

PROJECT_ID=""
if PROJECT_ID="$(_resolve_project 2>/dev/null)"; then
  export NPA_E2E_SERVERLESS_PROJECT="${PROJECT_ID}"
else
  echo "WARN: no Nebius project id; e2e window will skip/fail until credentials are configured" | tee -a "${LOG_ROOT}/preflight.log"
fi

export NPA_INTEGRATION_E2E=1
export PIPELINE_E2E_LOG_ROOT="${LOG_ROOT}"
export PIPELINE_E2E_SOURCE_REF="${BRANCH}"

TMUX_ENV="cd \"${ROOT}\" && unset NEBIUS_IAM_TOKEN NPA_IAM_TOKEN 2>/dev/null || true && export NPA_INTEGRATION_E2E=1 && export NPA_E2E_SERVERLESS_PROJECT=\"${NPA_E2E_SERVERLESS_PROJECT:-}\" && export NPA_E2E_PIPELINE_GPU=\"${NPA_E2E_PIPELINE_GPU}\" && export NEBIUS_REGISTRY_PROFILE=\"${NEBIUS_REGISTRY_PROFILE}\" && export REGISTRY=\"${REGISTRY}\" && export PIPELINE_E2E_LOG_ROOT=\"${LOG_ROOT}\""

if ! command -v tmux >/dev/null; then
  echo "tmux required" >&2
  exit 1
fi

tmux kill-session -t "${SESSION}" 2>/dev/null || true

tmux new-session -d -s "${SESSION}" -n dashboard \
  "bash -lc '${TMUX_ENV} && echo RUN_ID=${RUN_ID} && echo LOG_ROOT=${LOG_ROOT} && echo BRANCH=${BRANCH} && echo PROJECT=${NPA_E2E_SERVERLESS_PROJECT:-unset} && while true; do clear; date -u; echo; tail -n 40 \"${LOG_ROOT}/e2e.log\" 2>/dev/null || echo \"(e2e pending)\"; echo; tail -n 20 \"${LOG_ROOT}/build.log\" 2>/dev/null || true; sleep 20; done'"

if [[ "${WITH_BUILD}" == "1" && "${SKIP_BUILD}" == "0" ]]; then
  tmux new-window -t "${SESSION}" -n build \
    "bash -lc '${TMUX_ENV} && git fetch origin ${BRANCH} 2>&1 | tee -a \"${LOG_ROOT}/build.log\"; git checkout ${BRANCH} 2>&1 | tee -a \"${LOG_ROOT}/build.log\"; REGISTRY=\"${REGISTRY}\" bash \"${SCRIPT_DIR}/build_golden_eval_images.sh\" envgen reference-policy loop-eval --push 2>&1 | tee -a \"${LOG_ROOT}/build.log\"; echo build_done | tee -a \"${LOG_ROOT}/build.log\"; exec bash'"
fi

tmux new-window -t "${SESSION}" -n e2e \
  "bash -lc '${TMUX_ENV} && git fetch origin ${BRANCH} && git checkout ${BRANCH} && ${PYTHON} -m pytest npa/tests/e2e/test_pipeline_images_serverless_e2e.py -v -m e2e_serverless --tb=short 2>&1 | tee \"${LOG_ROOT}/e2e.log\"; ec=\${PIPESTATUS[0]}; echo e2e_exit=\$ec | tee -a \"${LOG_ROOT}/e2e.log\"; if [[ \$ec -ne 0 ]]; then echo \"${RUN_ID}\" > \"${STATE_DIR}/patch-request-run-id\"; fi; exec bash'"

if [[ "${WITH_CURSOR}" == "1" ]]; then
  export GOLDEN_EVAL_STATE_DIR="${STATE_DIR}/cursor"
  mkdir -p "${GOLDEN_EVAL_STATE_DIR}"
  tmux new-window -t "${SESSION}" -n cursor \
    "bash -lc '${TMUX_ENV} && export GOLDEN_EVAL_STATE_DIR=\"${STATE_DIR}/cursor\" && export GOLDEN_EVAL_SOURCE_REF=\"${BRANCH}\" && bash \"${SCRIPT_DIR}/golden_eval_cursor_patch.sh\" 2>&1 | tee -a \"${LOG_ROOT}/cursor.log\"; exec bash'"
fi

echo "TMUX_SESSION=${SESSION}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "attach: tmux attach -t ${SESSION}"
