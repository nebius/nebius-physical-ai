#!/usr/bin/env bash
# Run Cypress browser checks for the generated NPA agent UI.
#
# Mocked mode:
#   bash npa/scripts/run_agent_cypress.sh --mock
#
# Live mode:
#   bash npa/scripts/run_agent_cypress.sh --live --project <alias> --name agent
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BROWSER_DIR="${ROOT}/npa/tests/browser"
NPA_BIN="${ROOT}/npa/.venv/bin/npa"
PYTHON="${ROOT}/npa/.venv/bin/python"

MODE="mock"
PROJECT="${NPA_AGENT_PROJECT:-us-central1}"
NAME="${NPA_AGENT_NAME:-agent}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--mock|--live] [--project NAME] [--name NAME]

Options:
  --mock            Run mocked browser UI coverage (default)
  --live            Run against a deployed agent using stored npa auth
  --project NAME    NPA project alias for live mode (default: ${PROJECT})
  --name NAME       Agent deployment name for live mode (default: ${NAME})
  --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mock)
      MODE="mock"
      shift
      ;;
    --live)
      MODE="live"
      shift
      ;;
    --project)
      PROJECT="$2"
      shift 2
      ;;
    --name)
      NAME="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "${BROWSER_DIR}/node_modules" ]]; then
  (cd "${BROWSER_DIR}" && npm ci)
fi

if [[ "${MODE}" == "mock" ]]; then
  (cd "${BROWSER_DIR}" && npm run cy:mock)
  exit 0
fi

if [[ ! -x "${NPA_BIN}" || ! -x "${PYTHON}" ]]; then
  echo "Missing npa virtualenv; expected ${NPA_BIN} and ${PYTHON}" >&2
  exit 1
fi

STATUS_JSON="$("${NPA_BIN}" agent status --project "${PROJECT}" --name "${NAME}" --json)"
AGENT_URL="$(NPA_STATUS_JSON="${STATUS_JSON}" "${PYTHON}" - <<'PY'
import json
import os

data = json.loads(os.environ["NPA_STATUS_JSON"])
print((data.get("public_url") or data.get("agent_url") or "").rstrip("/"))
PY
)"
AUTH_ENV="${HOME}/.npa/agents/${PROJECT}/${NAME}/auth.env"
if [[ ! -f "${AUTH_ENV}" ]]; then
  echo "Missing live agent auth env: ${AUTH_ENV}" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${AUTH_ENV}"

if [[ -z "${AGENT_URL}" || -z "${AGENT_USER:-}" || -z "${AGENT_PASSWORD:-}" ]]; then
  echo "Live Cypress requires agent URL plus AGENT_USER/AGENT_PASSWORD" >&2
  exit 1
fi

(
  cd "${BROWSER_DIR}"
  NODE_TLS_REJECT_UNAUTHORIZED=0 \
    CYPRESS_NPA_AGENT_BASE_URL="${AGENT_URL}" \
    CYPRESS_NPA_AGENT_USER="${AGENT_USER}" \
    CYPRESS_NPA_AGENT_PASSWORD="${AGENT_PASSWORD}" \
    npm run cy:live
)
