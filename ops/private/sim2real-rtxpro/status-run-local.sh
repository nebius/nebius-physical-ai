#!/usr/bin/env bash
# Local sim2real run status — kubectl + S3 only (no npa CLI version required).
# Usage: status-run-local.sh <run-id> [--watch]
set -euo pipefail

RUN_ID="${1:?usage: status-run-local.sh <run-id> [--watch]}"
WATCH=0
if [[ "${2:-}" == "--watch" ]]; then
  WATCH=1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-env.sh
source "${SCRIPT_DIR}/lib/operator-env.sh"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"
ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
export NPA_SIM2REAL_REPO="${ROOT}"

_npa_cfg=()
while IFS= read -r _line; do
  _npa_cfg+=("${_line}")
done < <(operator_read_config "${ROOT}" 2>/dev/null || true)
BUCKET="${S3_BUCKET:-${_npa_cfg[0]:-lerobot-d87cf691}}"
ENDPOINT="${S3_ENDPOINT:-${_npa_cfg[1]:-https://storage.eu-north1.nebius.cloud}}"
CTX="${KUBECONTEXT:-${_npa_cfg[3]:-npa-rtxpro-mk8s}}"
PREFIX="${S3_PREFIX:-sim2real-b}"
NS="${KUBENS:-default}"

export KUBECONFIG="${KUBECONFIG:-$(operator_kubeconfig_path "${CTX}")}"
operator_export_kubeconfig "${CTX}" "${ROOT}" 2>/dev/null || true

JOB="sim2real-${RUN_ID}"
S3_BASE="s3://${BUCKET}/${PREFIX}/${RUN_ID}"

STAGE_PATHS=(
  "stage_01_trigger/trigger.json"
  "stage_02_assets/assets_manifest.json"
  "augment/cosmos2-transfer-result.json"
  "envs/raw/"
  "eval/heldout/report.json"
  "state/workflow_state.json"
  "reports/sim2real-report.json"
)

print_status() {
  echo "=== sim2real ${RUN_ID} ==="
  echo "k8s_context=${CTX} bucket=${BUCKET}"
  echo "s3=${S3_BASE}/"
  echo ""

  if command -v kubectl >/dev/null 2>&1; then
    echo "--- orchestrator job ${JOB} ---"
    if kubectl --context "${CTX}" get "job/${JOB}" -n "${NS}" >/dev/null 2>&1; then
      kubectl --context "${CTX}" get "job/${JOB}" -n "${NS}" -o wide 2>/dev/null || true
      POD="$(kubectl --context "${CTX}" get pods -n "${NS}" -l "job-name=${JOB}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
      if [[ -n "${POD}" ]]; then
        REASON="$(kubectl --context "${CTX}" get pod "${POD}" -n "${NS}" \
          -o jsonpath='{.status.containerStatuses[0].state.waiting.reason}' 2>/dev/null || true)"
        PHASE="$(kubectl --context "${CTX}" get pod "${POD}" -n "${NS}" \
          -o jsonpath='{.status.phase}' 2>/dev/null || true)"
        echo "pod=${POD} phase=${PHASE} reason=${REASON:-ok}"
      fi
    else
      echo "job/${JOB}: NOT FOUND (deleted or never started)"
    fi
    echo ""
    echo "--- sibling jobs (s2r-*) ---"
    kubectl --context "${CTX}" get jobs -n "${NS}" 2>/dev/null \
      | grep -i "${RUN_ID}" || echo "(none)"
  else
    echo "WARN: kubectl not in PATH — skip cluster section"
  fi

  echo ""
  echo "--- S3 stages ---"
  if command -v aws >/dev/null 2>&1; then
    for path in "${STAGE_PATHS[@]}"; do
      if aws s3 ls "${S3_BASE}/${path}" --endpoint-url "${ENDPOINT}" >/dev/null 2>&1; then
        echo "OK   ${path}"
      else
        echo "---- ${path}"
      fi
    done
  else
    echo "WARN: aws CLI not in PATH — skip S3 section"
  fi
  echo ""
}

if [[ "${WATCH}" == "1" ]]; then
  while true; do
    clear 2>/dev/null || true
    /bin/date -u
    print_status
    /bin/sleep 10
  done
else
  print_status
fi
