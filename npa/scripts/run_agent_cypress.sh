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
#   NPA_AGENT_CYPRESS_RUN_ID=<run> NPA_AGENT_CYPRESS_ARTIFACT_KEY=<key> \
#     bash npa/scripts/run_agent_cypress.sh --live
#   NPA_AGENT_CYPRESS_LIVE_DESTRUCTIVE=1 bash npa/scripts/run_agent_cypress.sh --live
#   NPA_LEISAAC_RUN_ID=<run> NPA_AGENT_TASK=<task> NPA_AGENT_ENVIRONMENT_ID=<env> \
#     NPA_AGENT_COMPLETED_EPISODES=0 bash npa/scripts/run_agent_cypress.sh --live
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BROWSER_DIR="${ROOT}/npa/tests/browser"

MODE="mock"
LIVE_DESTRUCTIVE="${NPA_AGENT_CYPRESS_LIVE_DESTRUCTIVE:-0}"
LEGACY_RUN_ID="${NPA_AGENT_RUN_ID:-}"
LIVE_RUN_ID="${NPA_AGENT_CYPRESS_RUN_ID:-}"
LIVE_RUN_REF="${NPA_AGENT_CYPRESS_RUN_REF:-${NPA_AGENT_RUN_REF:-}}"
FOXGLOVE_RUN_ID="${NPA_AGENT_CYPRESS_FOXGLOVE_RUN_ID:-${LIVE_RUN_ID}}"
LIVE_LEISAAC_RUN_ID="${NPA_LEISAAC_RUN_ID:-}"
LIVE_TASK="${NPA_AGENT_TASK:-}"
LIVE_ENVIRONMENT_ID="${NPA_AGENT_ENVIRONMENT_ID:-}"
LIVE_COMPLETED_EPISODES="${NPA_AGENT_COMPLETED_EPISODES:-}"
LIVE_ARTIFACT_KEY="${NPA_AGENT_CYPRESS_ARTIFACT_KEY:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--mock|--live]

Options:
  --mock            Run mocked browser UI coverage (default)
  --live            Run against an HTTPS agent using explicit environment credentials
  --destructive     Enable live Cypress Sim2Real submit button test
  --resolve-mode    Print the resolved live npm script without network access
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
    --resolve-mode)
      MODE="resolve"
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

LEISAAC_CONTEXT=0
LIVE_CYPRESS_SCRIPT="cy:live"
if [[ "${MODE}" != "mock" ]]; then
  if [[ -n "${LEGACY_RUN_ID}" && ( -n "${LIVE_RUN_ID}" || -n "${LIVE_LEISAAC_RUN_ID}" ) ]]; then
    echo "NPA_AGENT_RUN_ID is the compatibility selector; do not combine it with NPA_AGENT_CYPRESS_RUN_ID or NPA_LEISAAC_RUN_ID" >&2
    exit 2
  fi
  if [[ -n "${LIVE_RUN_ID}" && -n "${LIVE_LEISAAC_RUN_ID}" ]]; then
    echo "Set only one of NPA_AGENT_CYPRESS_RUN_ID or NPA_LEISAAC_RUN_ID" >&2
    exit 2
  fi

  if [[ -n "${LIVE_TASK}" || -n "${LIVE_ENVIRONMENT_ID}" || -n "${LIVE_COMPLETED_EPISODES}" ]]; then
    LEISAAC_CONTEXT=1
  fi
  if [[ -z "${LIVE_RUN_ID}" && -z "${LIVE_LEISAAC_RUN_ID}" && -n "${LEGACY_RUN_ID}" ]]; then
    if [[ "${LEISAAC_CONTEXT}" == "1" ]]; then
      LIVE_LEISAAC_RUN_ID="${LEGACY_RUN_ID}"
    else
      LIVE_RUN_ID="${LEGACY_RUN_ID}"
    fi
  fi

  if [[ -n "${LIVE_LEISAAC_RUN_ID}" || "${LEISAAC_CONTEXT}" == "1" ]]; then
    if [[ -z "${LIVE_LEISAAC_RUN_ID}" || -z "${LIVE_TASK}" || -z "${LIVE_ENVIRONMENT_ID}" || -z "${LIVE_COMPLETED_EPISODES}" ]]; then
      echo "Live LeIsaac Cypress requires NPA_LEISAAC_RUN_ID (or legacy NPA_AGENT_RUN_ID with full LeIsaac context), NPA_AGENT_TASK, NPA_AGENT_ENVIRONMENT_ID, and NPA_AGENT_COMPLETED_EPISODES" >&2
      exit 2
    fi
    if ! [[ "${LIVE_COMPLETED_EPISODES}" =~ ^[0-9]+$ ]]; then
      echo "NPA_AGENT_COMPLETED_EPISODES must be a non-negative integer" >&2
      exit 2
    fi
    LIVE_CYPRESS_SCRIPT="cy:live-leisaac"
  fi
  if [[ -n "${LIVE_RUN_ID}" || -n "${LIVE_ARTIFACT_KEY}" ]]; then
    if [[ -z "${LIVE_RUN_ID}" || -z "${LIVE_ARTIFACT_KEY}" ]]; then
      echo "Exact RRD mode requires a run selector (NPA_AGENT_CYPRESS_RUN_ID or legacy NPA_AGENT_RUN_ID) and NPA_AGENT_CYPRESS_ARTIFACT_KEY" >&2
      exit 2
    fi
    LIVE_CYPRESS_SCRIPT="cy:live-rrd"
  fi
fi

if [[ "${MODE}" == "resolve" ]]; then
  printf '%s\n' "${LIVE_CYPRESS_SCRIPT}"
  exit 0
fi

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
    CYPRESS_NPA_AGENT_RUN_ID="${LIVE_LEISAAC_RUN_ID}" \
    CYPRESS_NPA_AGENT_TASK="${LIVE_TASK}" \
    CYPRESS_NPA_AGENT_ENVIRONMENT_ID="${LIVE_ENVIRONMENT_ID}" \
    CYPRESS_NPA_AGENT_COMPLETED_EPISODES="${LIVE_COMPLETED_EPISODES}" \
    CYPRESS_NPA_AGENT_CYPRESS_ARTIFACT_KEY="${LIVE_ARTIFACT_KEY}" \
    npm run "${LIVE_CYPRESS_SCRIPT}"
)
