#!/usr/bin/env bash
# Run Cypress browser checks for the generated NPA agent UI.
#
# Mocked mode:
#   bash npa/scripts/run_agent_cypress.sh --mock
#
# Live mode is an explicit, credential-in-environment opt-in:
#   NPA_AGENT_CYPRESS_LIVE=1 NPA_AGENT_BASE_URL=https://agent.example \
#   NPA_AGENT_USER=... NPA_AGENT_PASSWORD=... \
#     bash npa/scripts/run_agent_cypress.sh --live
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BROWSER_DIR="${ROOT}/npa/tests/browser"

MODE="mock"
LIVE_DESTRUCTIVE="${NPA_AGENT_CYPRESS_LIVE_DESTRUCTIVE:-0}"
LIVE_RUN_ID="${NPA_AGENT_CYPRESS_RUN_ID:-${NPA_AGENT_RUN_ID:-}}"
LIVE_RUN_REF="${NPA_AGENT_CYPRESS_RUN_REF:-${NPA_AGENT_RUN_REF:-}}"
FOXGLOVE_RUN_ID="${NPA_AGENT_CYPRESS_FOXGLOVE_RUN_ID:-${LIVE_RUN_ID}}"
LIVE_ARTIFACT_KEY="${NPA_AGENT_CYPRESS_ARTIFACT_KEY:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--mock|--live]

Options:
  --mock            Run mocked browser UI coverage (default)
  --live            Run against an HTTPS agent using explicit environment credentials
  --destructive     Enable live Cypress Sim2Real submit button test
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
    --destructive)
      LIVE_DESTRUCTIVE=1
      shift
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

if [[ "${NPA_AGENT_CYPRESS_LIVE:-0}" != "1" ]]; then
  echo "Live Cypress is disabled; set NPA_AGENT_CYPRESS_LIVE=1 explicitly." >&2
  exit 1
fi
if [[ -z "${NPA_AGENT_BASE_URL:-}" || -z "${NPA_AGENT_USER:-}" || -z "${NPA_AGENT_PASSWORD:-}" ]]; then
  echo "Live Cypress requires NPA_AGENT_BASE_URL, NPA_AGENT_USER, and NPA_AGENT_PASSWORD." >&2
  exit 1
fi
case "${NPA_AGENT_BASE_URL}" in
  https://*) ;;
  *)
    echo "Live Cypress requires an HTTPS NPA_AGENT_BASE_URL." >&2
    exit 1
    ;;
esac

LIVE_CYPRESS_SCRIPT="cy:live"
if [[ -n "${LIVE_ARTIFACT_KEY}" ]]; then
  if [[ -z "${LIVE_RUN_ID}" ]]; then
    echo "Exact RRD mode requires NPA_AGENT_CYPRESS_RUN_ID with NPA_AGENT_CYPRESS_ARTIFACT_KEY" >&2
    exit 2
  fi
  LIVE_CYPRESS_SCRIPT="cy:live-rrd"
fi
if [[ "${LIVE_CYPRESS_SCRIPT}" == "cy:live" ]]; then
  if [[ -z "${NPA_PLAYWRIGHT_CHROMIUM_EXECUTABLE:-}" || ! -x "${NPA_PLAYWRIGHT_CHROMIUM_EXECUTABLE}" ]]; then
    echo "Live Foxglove Cypress requires NPA_PLAYWRIGHT_CHROMIUM_EXECUTABLE." >&2
    exit 1
  fi
  if [[ -z "${NPA_AGENT_CYPRESS_EVIDENCE_DIR:-}" || "${NPA_AGENT_CYPRESS_EVIDENCE_DIR}" != /* ]]; then
    echo "Live Foxglove Cypress requires an absolute NPA_AGENT_CYPRESS_EVIDENCE_DIR." >&2
    exit 1
  fi
  case "${NPA_AGENT_CYPRESS_EVIDENCE_DIR}" in
    "${ROOT}"|"${ROOT}"/*)
      echo "Live Foxglove evidence must be stored outside the clone." >&2
      exit 1
      ;;
  esac
fi

(
  cd "${BROWSER_DIR}"
  NODE_TLS_REJECT_UNAUTHORIZED=0 \
    CYPRESS_NPA_AGENT_CYPRESS_LIVE=1 \
    CYPRESS_NPA_AGENT_BASE_URL="${NPA_AGENT_BASE_URL}" \
    CYPRESS_NPA_AGENT_USER="${NPA_AGENT_USER}" \
    CYPRESS_NPA_AGENT_PASSWORD="${NPA_AGENT_PASSWORD}" \
    CYPRESS_NPA_AGENT_CYPRESS_LIVE_DESTRUCTIVE="${LIVE_DESTRUCTIVE}" \
    CYPRESS_NPA_AGENT_CYPRESS_RUN_ID="${LIVE_RUN_ID}" \
    CYPRESS_NPA_AGENT_CYPRESS_RUN_REF="${LIVE_RUN_REF}" \
    CYPRESS_NPA_AGENT_CYPRESS_FOXGLOVE_RUN_ID="${FOXGLOVE_RUN_ID}" \
    CYPRESS_NPA_AGENT_CYPRESS_ARTIFACT_KEY="${LIVE_ARTIFACT_KEY}" \
    npm run "${LIVE_CYPRESS_SCRIPT}"
)
