#!/usr/bin/env bash
set -euo pipefail

if [[ "${NPA_RUN_GPU_INTEGRATION:-0}" != "1" ]]; then
  echo "skip: set NPA_RUN_GPU_INTEGRATION=1 to run cross-tool GPU pipeline tests"
  exit 0
fi

: "${NPA_INTEGRATION_PROJECT:?missing NPA_INTEGRATION_PROJECT}"
: "${NPA_INTEGRATION_COSMOS_NAME:?missing NPA_INTEGRATION_COSMOS_NAME}"
: "${NPA_INTEGRATION_FIFTYONE_NAME:?missing NPA_INTEGRATION_FIFTYONE_NAME}"
: "${NPA_INTEGRATION_S3_PREFIX:?missing NPA_INTEGRATION_S3_PREFIX}"
: "${NPA_INTEGRATION_REGION:?missing NPA_INTEGRATION_REGION}"
: "${NPA_INTEGRATION_GPU_TYPE:?missing NPA_INTEGRATION_GPU_TYPE}"
: "${NPA_INTEGRATION_GPU_PRESET:?missing NPA_INTEGRATION_GPU_PRESET}"

cosmos_output="${NPA_INTEGRATION_S3_PREFIX%/}/cosmos-fiftyone-smoke.mp4"
runtime="${NPA_INTEGRATION_RUNTIME:-byovm}"
ssh_user="${NPA_INTEGRATION_SSH_USER:-ubuntu}"
cosmos_port="${NPA_INTEGRATION_COSMOS_PORT:-8081}"
fiftyone_port="${NPA_INTEGRATION_FIFTYONE_PORT:-5151}"

deploy_args=(
  --runtime "${runtime}"
  --gpu-type "${NPA_INTEGRATION_GPU_TYPE}"
  --gpu-preset "${NPA_INTEGRATION_GPU_PRESET}"
  --region "${NPA_INTEGRATION_REGION}"
)
if [[ "${runtime}" == "byovm" ]]; then
  : "${NPA_INTEGRATION_HOST:?missing NPA_INTEGRATION_HOST for BYOVM integration}"
  : "${NPA_INTEGRATION_SSH_KEY:?missing NPA_INTEGRATION_SSH_KEY for BYOVM integration}"
  deploy_args+=(--host "${NPA_INTEGRATION_HOST}" --ssh-key "${NPA_INTEGRATION_SSH_KEY}" --ssh-user "${ssh_user}")
fi

if [[ "${NPA_INTEGRATION_SKIP_DEPLOY:-0}" != "1" ]]; then
  echo "seam=cosmos->fiftyone step=cosmos-deploy"
  npa workbench cosmos \
    -p "${NPA_INTEGRATION_PROJECT}" \
    -n "${NPA_INTEGRATION_COSMOS_NAME}" \
    deploy \
    "${deploy_args[@]}" \
    --server-port "${cosmos_port}" \
    --verify-env

  echo "seam=cosmos->fiftyone step=fiftyone-deploy"
  npa workbench fiftyone \
    -p "${NPA_INTEGRATION_PROJECT}" \
    -n "${NPA_INTEGRATION_FIFTYONE_NAME}" \
    deploy \
    "${deploy_args[@]}" \
    --port "${fiftyone_port}" \
    --verify-env
fi

echo "seam=cosmos->fiftyone step=cosmos-infer"
npa workbench cosmos \
  -p "${NPA_INTEGRATION_PROJECT}" \
  -n "${NPA_INTEGRATION_COSMOS_NAME}" \
  infer \
  --prompt "A small robot arm moves one colored block on a table" \
  --output-path "${cosmos_output}" \
  --timeout "${NPA_INTEGRATION_COSMOS_TIMEOUT:-1200}" \
  --poll-interval 10

echo "seam=cosmos->fiftyone step=fiftyone-load"
load_json="$(npa workbench fiftyone \
  -p "${NPA_INTEGRATION_PROJECT}" \
  -n "${NPA_INTEGRATION_FIFTYONE_NAME}" \
  load-dataset \
  --name cosmos_fiftyone_smoke \
  --input-path "${cosmos_output}" \
  --format video \
  --output json)"

LOAD_JSON="${load_json}" python3 - <<'PY'
import json
import os

data = json.loads(os.environ["LOAD_JSON"])
samples = int(data.get("samples", 0))
if samples <= 0:
    raise SystemExit(
        "seam=cosmos->fiftyone failure=format-mismatch "
        f"expected sample count > 0, got {samples}: {data}"
    )
PY
