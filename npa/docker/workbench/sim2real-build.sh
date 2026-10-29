#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REGISTRY=""
PUSH=0
ENVGEN_ONLY=0
BASE_IMAGE="${BASE_IMAGE:-ghcr.io/nebius/nebius-physical-ai/npa-cosmos3-reason@sha256:e708aa32b9247eaedf1a7a5b0d82adbe65b95a9af50721b9d09fddbfd508bbc9}"
# EnvGen's canonical Stage 4 code does not import Genesis or LeRobot.  Unless
# an operator deliberately supplies another current-source parent, use the
# freshly built Cosmos3-reason role as the source-attested CUDA/Kubernetes base
# and construct an isolated EnvGen Python closure on top of it.
SIM2REAL_ENVGEN_BASE_IMAGE="${SIM2REAL_ENVGEN_BASE_IMAGE:-}"
VLM_TAG="${VLM_TAG:-cuda13-b300-3.0.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z}"
ENVGEN_TAG="${ENVGEN_TAG:-cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z}"
SIM2REAL_COMPONENT_BASE_IMAGE="${SIM2REAL_COMPONENT_BASE_IMAGE:-npa-envgen:${ENVGEN_TAG}}"
EVAL_TAG="${EVAL_TAG:-cuda13-b300-0.1.3-sm80-sm90-sm100-sm103-sm120-20260803T034152Z}"
VLM_RL_TAG="${VLM_RL_TAG:-cuda13-b300-0.1.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z}"
CONTROL_TAG="${CONTROL_TAG:-0.1.2}"
NPA_SOURCE_SHA="${NPA_SOURCE_SHA:-$(git -C "${NPA_ROOT}/.." rev-parse HEAD)}"
if [[ ! "${NPA_SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: NPA_SOURCE_SHA must be the exact 40-character checkout SHA" >&2
  exit 2
fi

usage() {
  cat <<EOF
Usage: sim2real-build.sh [--registry REGISTRY] [--push] [--envgen-only]

Builds the Sim2Real reference images one at a time:
  npa-sim2real-control:${CONTROL_TAG}
  npa-cosmos3-reason:${VLM_TAG} (skipped when SKIP_COSMOS3_REASON=1)
  npa-envgen:${ENVGEN_TAG}
  npa-reference-policy:${ENVGEN_TAG}
  npa-lerobot-vlm-rl:${VLM_RL_TAG}
  npa-loop-eval:${EVAL_TAG}
  npa-rerun-viewer:${RERUN_VIEWER_TAG:-0.38.1} (skipped when SKIP_RERUN_VIEWER=1)

--envgen-only builds only the current-source npa-cosmos3-reason parent and
npa-envgen.  It is the trusted-development route for EnvGen because its
Dockerfile requires that newly built source-attested parent; it deliberately
does not build controller, policy, signal-adapter, legacy Genesis evaluation,
or viewer roles.

Set BASE_IMAGE to a compatible immutable CUDA 13 / sm80-sm90-sm100-sm103-sm120
parent for the freshly built Cosmos3-reason image. Unless
SIM2REAL_ENVGEN_BASE_IMAGE is set, that freshly built current-source image is
then the EnvGen parent. An explicit SIM2REAL_ENVGEN_BASE_IMAGE must carry the
same source-revision label. EnvGen does not inherit Genesis: its dedicated
Python 3.12 closure excludes the unrelated Genesis/LeRobot dataset stack.
The freshly built Envgen image is the default parent for the reference policy
and VLM-RL signal adapter; set SIM2REAL_COMPONENT_BASE_IMAGE only to validate
another immutable Envgen build. The legacy loop-eval image is not part of the
canonical Isaac workflow and stays skipped unless BUILD_LEGACY_GENESIS_EVAL=1
with a separately qualified Genesis runtime.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --registry)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --registry requires a value" >&2
        exit 2
      fi
      REGISTRY="${2%/}"
      shift 2
      ;;
    --push)
      PUSH=1
      shift
      ;;
    --envgen-only)
      ENVGEN_ONLY=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ "$PUSH" -eq 1 ] && [ -z "$REGISTRY" ]; then
  echo "ERROR: --push requires --registry" >&2
  exit 2
fi

if [ -n "${SKIP_COSMOS3_REASON:-}" ] && [ -z "$SIM2REAL_ENVGEN_BASE_IMAGE" ]; then
  cat >&2 <<EOF
ERROR: SKIP_COSMOS3_REASON requires an explicit SIM2REAL_ENVGEN_BASE_IMAGE.
The EnvGen parent must carry org.opencontainers.image.revision=${NPA_SOURCE_SHA}.
EOF
  exit 2
fi

if [ "$(git -C "${NPA_ROOT}/.." rev-parse HEAD)" != "${NPA_SOURCE_SHA}" ]; then
  echo "ERROR: NPA_SOURCE_SHA must match the build-context checkout HEAD" >&2
  exit 2
fi

build_one() {
  local name="$1"
  local tag="$2"
  local dockerfile="$3"
  local base_arg="$4"
  local local_image="${name}:${tag}"
  local registry_image=""
  local args=(-f "${dockerfile}" --build-arg "${base_arg}" --build-arg "NPA_SOURCE_SHA=${NPA_SOURCE_SHA}" -t "${local_image}")
  if [ -n "$REGISTRY" ]; then
    registry_image="${REGISTRY}/${name}:${tag}"
    args+=(-t "${registry_image}")
  fi
  docker build "${args[@]}" "${NPA_ROOT}"
  echo "Built: ${local_image}"
  if [ -n "$registry_image" ]; then
    echo "Tagged: ${registry_image}"
  fi
  if [ "$PUSH" -eq 1 ]; then
    docker push "${registry_image}"
  fi
}

if [ "$ENVGEN_ONLY" -eq 0 ]; then
  build_one "npa-sim2real-control" "${CONTROL_TAG}" "${SCRIPT_DIR}/sim2real-control/Dockerfile" "SIM2REAL_CONTROL_VERSION=${CONTROL_TAG}"
fi
if [ -n "${SKIP_COSMOS3_REASON:-}" ]; then
  echo "Skipping npa-cosmos3-reason (SKIP_COSMOS3_REASON=${SKIP_COSMOS3_REASON})"
else
  build_one "npa-cosmos3-reason" "${VLM_TAG}" "${SCRIPT_DIR}/cosmos3-reason/Dockerfile" "BASE_IMAGE=${BASE_IMAGE}"
fi
if [ -z "$SIM2REAL_ENVGEN_BASE_IMAGE" ]; then
  SIM2REAL_ENVGEN_BASE_IMAGE="npa-cosmos3-reason:${VLM_TAG}"
fi
PARENT_SOURCE_SHA="$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' "$SIM2REAL_ENVGEN_BASE_IMAGE" 2>/dev/null || true)"
if [ "$PARENT_SOURCE_SHA" != "$NPA_SOURCE_SHA" ]; then
  echo "ERROR: SIM2REAL_ENVGEN_BASE_IMAGE must carry org.opencontainers.image.revision=${NPA_SOURCE_SHA}; found ${PARENT_SOURCE_SHA:-missing}" >&2
  exit 2
fi
build_one "npa-envgen" "${ENVGEN_TAG}" "${SCRIPT_DIR}/sim2real-envgen/Dockerfile" "BASE_IMAGE=${SIM2REAL_ENVGEN_BASE_IMAGE}"
if [ "$ENVGEN_ONLY" -eq 1 ]; then
  exit 0
fi
build_one "npa-reference-policy" "${ENVGEN_TAG}" "${SCRIPT_DIR}/sim2real-reference-policy/Dockerfile" "BASE_IMAGE=npa-envgen:${ENVGEN_TAG}"
build_one "npa-lerobot-vlm-rl" "${VLM_RL_TAG}" "${SCRIPT_DIR}/lerobot-vlm-rl/Dockerfile" "BASE_IMAGE=${SIM2REAL_COMPONENT_BASE_IMAGE}"
if [ "${BUILD_LEGACY_GENESIS_EVAL:-0}" = "1" ]; then
  build_one "npa-loop-eval" "${EVAL_TAG}" "${SCRIPT_DIR}/sim2real-eval/Dockerfile" "BASE_IMAGE=${SIM2REAL_COMPONENT_BASE_IMAGE}"
else
  echo "Skipping npa-loop-eval (BUILD_LEGACY_GENESIS_EVAL=0; not a canonical workflow role)"
fi
if [ -n "${SKIP_RERUN_VIEWER:-}" ]; then
  echo "Skipping npa-rerun-viewer (SKIP_RERUN_VIEWER=${SKIP_RERUN_VIEWER})"
else
  build_one "npa-rerun-viewer" "${RERUN_VIEWER_TAG:-0.38.1}" "${SCRIPT_DIR}/rerun-viewer/Dockerfile" "RERUN_SDK_VERSION=${RERUN_VIEWER_TAG:-0.38.1}"
fi
