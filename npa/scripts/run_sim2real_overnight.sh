#!/usr/bin/env bash
# Run the production Sim2Real VLM→RL loop locally with K8s sibling GPU jobs.
# Logs to ~/sim2real-overnight-<run-id>.log — safe to detach from tmux.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${NPA_SIM2REAL_RUN_ID:-sim2real-overnight-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${NPA_SIM2REAL_OUTPUT_DIR:-/tmp/npa-sim2real-${RUN_ID}}"
LOG="${NPA_SIM2REAL_LOG:-${HOME}/sim2real-overnight-${RUN_ID}.log}"
KUBECONFIG="${KUBECONFIG:-${HOME}/.npa/clusters/npa-rtxpro-mk8s/kubeconfig}"

export KUBECONFIG
# Full registry hostname so sibling Jobs pull real workbench images (not local reference).
export NPA_REGISTRY="${NPA_REGISTRY:-cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw}"

eval "$("${ROOT}/npa/.venv/bin/python" <<'PY'
from npa.clients.credentials import load_credentials, storage_endpoint_url

creds = load_credentials()
endpoint = storage_endpoint_url(creds.s3_endpoint) or "https://storage.eu-north1.nebius.cloud"
bucket = (creds.s3_bucket or "").removeprefix("s3://").split("/", 1)[0]
print(f'export AWS_ACCESS_KEY_ID="{creds.s3_access_key_id}"')
print(f'export AWS_SECRET_ACCESS_KEY="{creds.s3_secret_access_key}"')
print(f'export AWS_ENDPOINT_URL="{endpoint}"')
print(f'export S3_ENDPOINT_URL="{endpoint}"')
print(f'export NPA_S3_BUCKET="{bucket}"')
print(f'export NPA_SIM2REAL_BUCKET="{bucket}"')
PY
)"

BUCKET="${NPA_SIM2REAL_BUCKET}"
PREFIX="${NPA_SIM2REAL_PREFIX:-sim2real-b}"
TRIGGER_URI="${NPA_SIM2REAL_TRIGGER_DATASET_URI:-s3://${BUCKET}/sim2real-triggers/${RUN_ID}/lerobot-pusht/}"

EXTRA_ASSET_ARGS=()
if [[ -n "${ASSETS_URI:-}" ]]; then
  EXTRA_ASSET_ARGS+=(--assets-uri "${ASSETS_URI}")
fi
if [[ -n "${SCENE_SPEC_URI:-}" ]]; then
  EXTRA_ASSET_ARGS+=(--scene-spec-uri "${SCENE_SPEC_URI}")
fi

{
  echo "=== Sim2Real overnight run ${RUN_ID} ==="
  echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "output:  ${OUTPUT_DIR}"
  echo "kube:    ${KUBECONFIG}"
  echo "registry: ${NPA_REGISTRY}"
  echo

  "${ROOT}/npa/.venv/bin/python" -m npa.workflows.sim2real run \
    --run-id "${RUN_ID}" \
    --output-dir "${OUTPUT_DIR}" \
    --s3-bucket "${BUCKET}" \
    --s3-prefix "${PREFIX}" \
    --s3-endpoint "${AWS_ENDPOINT_URL}" \
    --trigger-dataset-uri "${TRIGGER_URI}" \
    --trigger-dataset-id "${NPA_SIM2REAL_TRIGGER_DATASET_ID:-lerobot/pusht}" \
    "${EXTRA_ASSET_ARGS[@]}" \
    --inner-iterations "${INNER_ITERATIONS:-2}" \
    --outer-iterations "${OUTER_ITERATIONS:-1}" \
    --rollout-count "${ROLLOUT_COUNT:-3}" \
    --steps-per-rollout "${STEPS_PER_ROLLOUT:-4}" \
    --heldout-env-count "${HELDOUT_ENV_COUNT:-8}" \
    --initial-quality "${INITIAL_QUALITY:-0.38}" \
    --k8s-kubeconfig "${KUBECONFIG}" \
    --upload-artifacts

  echo
  echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee -a "${LOG}"
