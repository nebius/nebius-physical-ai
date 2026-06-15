#!/usr/bin/env bash
# Build and optionally push Workbench images needed for golden-eval serverless runs.
#
# Usage:
#   ./npa/scripts/build_golden_eval_images.sh retargeting lancedb
#   ./npa/scripts/build_golden_eval_images.sh --all --push
#   REGISTRY=cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw ./npa/scripts/build_golden_eval_images.sh --all --push
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${NPA_ROOT}/npa/.venv/bin/python"
REGISTRY="${REGISTRY:-cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw}"
PUSH=0
BUILD_ALL=0
TOOLS=()

usage() {
  sed -n '2,8p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --push) PUSH=1; shift ;;
    --all) BUILD_ALL=1; shift ;;
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) TOOLS+=("$1"); shift ;;
  esac
done

if [[ "${BUILD_ALL}" == "1" ]]; then
  TOOLS=(
    retargeting lancedb detection-training fiftyone genesis
    lerobot-policy sim2real-envgen sim2real-reference-policy
    lerobot-vlm-rl sim2real-eval
  )
fi

if [[ ${#TOOLS[@]} -eq 0 ]]; then
  usage >&2
  exit 2
fi

tool_version() {
  local tool="$1"
  TOOL="${tool}" "${PYTHON}" - <<'PY'
import os
from npa.deploy.images import SUPPORTED_TOOL_VERSIONS
print(SUPPORTED_TOOL_VERSIONS[os.environ["TOOL"]])
PY
}

build_simple() {
  local tool="$1"
  local image="$2"
  local dockerfile="$3"
  local tag
  tag="$(tool_version "${tool}")"
  local local_ref="${image}:${tag}"
  local remote_ref="${REGISTRY}/${image}:${tag}"
  echo "=== build ${tool} -> ${local_ref} ==="
  docker build --platform linux/amd64 \
    -f "${NPA_ROOT}/${dockerfile}" \
    -t "${local_ref}" \
    -t "${remote_ref}" \
    "${NPA_ROOT}/npa"
  if [[ "${PUSH}" == "1" ]]; then
    echo "=== push ${remote_ref} ==="
    docker push "${remote_ref}"
  fi
}

build_lancedb() {
  bash "${NPA_ROOT}/npa/docker/workbench/lancedb/build.sh" \
    --registry "${REGISTRY}" \
    $([[ "${PUSH}" == "1" ]] && echo --push)
}

build_sim2real_stack() {
  local genesis_tag
  genesis_tag="$(tool_version genesis)"
  export GENESIS_IMAGE="${REGISTRY}/npa-genesis:${genesis_tag}"
  bash "${NPA_ROOT}/npa/docker/workbench/sim2real-build.sh" \
    --registry "${REGISTRY}" \
    $([[ "${PUSH}" == "1" ]] && echo --push)
}

for tool in "${TOOLS[@]}"; do
  case "${tool}" in
    retargeting)
      build_simple retargeting npa-retargeting docker/workbench/retargeting/Dockerfile
      ;;
    lancedb)
      build_lancedb
      ;;
    detection-training)
      build_simple detection-training npa-detection-training docker/workbench/detection-training/Dockerfile
      ;;
    fiftyone)
      build_simple fiftyone npa-fiftyone docker/workbench/fiftyone/Dockerfile
      ;;
    genesis)
      build_simple genesis npa-genesis docker/workbench/genesis/Dockerfile
      ;;
    lerobot-policy)
      build_simple lerobot-policy npa-lerobot-policy docker/workbench/lerobot-policy/Dockerfile
      ;;
    sim2real-envgen | sim2real-reference-policy | lerobot-vlm-rl | sim2real-eval)
      # Built together; skip duplicates in loop.
      if [[ "${tool}" == "sim2real-envgen" ]]; then
        build_sim2real_stack
      fi
      ;;
    *)
      echo "No build recipe for: ${tool}" >&2
      exit 2
      ;;
  esac
done

echo "Done: ${TOOLS[*]} push=${PUSH}"
