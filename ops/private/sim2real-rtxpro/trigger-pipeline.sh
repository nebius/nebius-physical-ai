#!/usr/bin/env bash
# Customer entrypoint: upload LeRobot data to S3, then explicitly trigger sim2real.
#
# No S3 polling — you control when the batch is complete and when the run starts.
#
# Usage:
#   export TRIGGER_DATASET_URI=s3://<bucket>/sim2real-triggers/<batch>/lerobot-<task>/
#   ./ops/private/sim2real-rtxpro/trigger-pipeline.sh
#
# Optional:
#   TRIGGER_DATASET_ID=lerobot/pusht
#   RUN_ID=my-batch-20260613          (default: auto-generated on submit)
#   WAIT=0                            (submit only, no wait for completion)
#   VISUALIZE=0                       (skip Rerun after sync)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-env.sh
source "${SCRIPT_DIR}/lib/operator-env.sh"
# shellcheck source=lib/demo-common.sh
source "${SCRIPT_DIR}/lib/demo-common.sh"
# shellcheck source=lib/trigger-preflight.sh
source "${SCRIPT_DIR}/lib/trigger-preflight.sh"
# shellcheck source=lib/asset-profile-guard.sh
source "${SCRIPT_DIR}/lib/asset-profile-guard.sh"

ROOT="$(demo_common_root)"
TRIGGER_URI="${TRIGGER_DATASET_URI:-${NPA_SIM2REAL_TRIGGER_DATASET_URI:-}}"

if [ -z "${TRIGGER_URI}" ]; then
  cat >&2 <<'EOF'
ERROR: set the uploaded LeRobot dataset path before triggering:

  export TRIGGER_DATASET_URI=s3://<bucket>/sim2real-triggers/<batch-id>/lerobot-<task>/
  ./ops/private/sim2real-rtxpro/trigger-pipeline.sh

Upload the full LeRobot tree first (meta/info.json, data/*.parquet, videos/…).
Trigger only after the batch is complete — the pipeline does not poll S3.
EOF
  exit 1
fi

demo_bootstrap_venv "${ROOT}"

ENDPOINT="$(operator_require_storage_endpoint "${ROOT}")" || exit 1
export AWS_ENDPOINT_URL="${ENDPOINT}" S3_ENDPOINT="${ENDPOINT}" S3_ENDPOINT_URL="${ENDPOINT}"

_cfg=()
while IFS= read -r _line; do
  _cfg+=("${_line}")
done < <(demo_read_storage_config "${ROOT}" 2>/dev/null || true)
BUCKET="${S3_BUCKET:-${_cfg[0]:-}}"
if [ -z "${BUCKET}" ]; then
  echo "ERROR: storage.bucket missing in ~/.npa/config.yaml" >&2
  exit 1
fi

echo "=== Preflight: storage endpoint + credentials ==="
echo "  endpoint: ${ENDPOINT}"
echo "  bucket:   ${BUCKET}"
storage_preflight_write "${BUCKET}" "${ENDPOINT}" "${ROOT}" || exit 1

echo ""
echo "=== Preflight: LeRobot trigger on S3 ==="
echo "  ${TRIGGER_URI}"
trigger_preflight_s3 "${TRIGGER_URI}" "${ENDPOINT}" "${ROOT}"

export NPA_SIM2REAL_TRIGGER_DATASET_URI="${TRIGGER_URI}"
export TRIGGER_DATASET_URI="${TRIGGER_URI}"
if [ -n "${TRIGGER_DATASET_ID:-}" ]; then
  export NPA_SIM2REAL_TRIGGER_DATASET_ID="${TRIGGER_DATASET_ID}"
fi

customer_asset_prepare_for_submit

echo ""
echo "=== Trigger sim2real pipeline on Nebius cluster ==="
unset RUN_ID
export SUBMIT=1
export WAIT="${WAIT:-0}"
exec "${SCRIPT_DIR}/run-demo.sh"
