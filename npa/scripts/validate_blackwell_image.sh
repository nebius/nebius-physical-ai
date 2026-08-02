#!/usr/bin/env bash
#
# validate_blackwell_image.sh - check a workbench image against a Blackwell target.
#
# Runs npa/docker/workbench/base/cuda13-b300/scripts/check_torch_gpu_arch.py inside the image. Two modes:
#
#   build-host mode (no GPU)  - proves the torch wheel ships SASS for the target
#                               architecture. Enough to catch a cu124/cu126 wheel
#                               that stops at sm_90 before burning a GPU hour.
#   GPU-node mode (--gpu)     - additionally proves get_device_capability() is the
#                               architecture we meant to validate. Validating on
#                               RTX PRO 6000 (sm_120, major 12) does NOT prove
#                               B200/B300 (sm_100/sm_103, major 10).
#
# The wheel-arch check deliberately does not require sm_103: stock cu130 wheels
# ship sm_100 SASS and B300 is covered by 10.0 -> 10.3 forward compatibility.
#
# USAGE
#   validate_blackwell_image.sh <image> [--target b200|b300|rtx6000|hopper]
#                                       [--gpu] [--python PATH] [--json]
#
# EXAMPLES
#   # On the dev VM, before pushing a tag:
#   validate_blackwell_image.sh npa-base:cuda13-b300-sm80-sm90-sm100-sm103-sm120-latest
#
#   # On a real B300 node:
#   validate_blackwell_image.sh "$NPA_REGISTRY/npa-lerobot:0.5.1" --target b300 --gpu
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Piped in over stdin rather than run from /npa inside the image, so this also
# works against images that are not derived from npa-base.
CHECKER="$SCRIPT_DIR/../docker/workbench/base/cuda13-b300/scripts/check_torch_gpu_arch.py"

IMAGE=""
TARGET="b200"
USE_GPU=0
IN_PYTHON="python"
JSON=0

usage() { grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --target) TARGET="${2:?--target requires a value}"; shift 2 ;;
    --gpu) USE_GPU=1; shift ;;
    --python) IN_PYTHON="${2:?--python requires a value}"; shift 2 ;;
    --json) JSON=1; shift ;;
    --help|-h) usage; exit 0 ;;
    -*) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [ -n "$IMAGE" ]; then echo "ERROR: unexpected argument: $1" >&2; exit 2; fi
      IMAGE="$1"; shift ;;
  esac
done

if [ -z "$IMAGE" ]; then
  echo "ERROR: an image reference is required" >&2
  usage >&2
  exit 2
fi

# Per target: the SASS the wheel must carry, and the capability the device must
# report. sm_100 covers B300 by forward compat, so both datacenter targets
# require the same wheel arch but a different device capability.
case "$TARGET" in
  b200)    REQUIRE_ARCH="sm_100"; REQUIRE_CAP="10.0" ;;
  b300)    REQUIRE_ARCH="sm_100"; REQUIRE_CAP="10.3" ;;
  rtx6000) REQUIRE_ARCH="sm_120"; REQUIRE_CAP="12.0" ;;
  hopper)  REQUIRE_ARCH="sm_90";  REQUIRE_CAP="9.0"  ;;
  *) echo "ERROR: unknown --target: $TARGET (b200|b300|rtx6000|hopper)" >&2; exit 2 ;;
esac

ARGS=(--require-arch "$REQUIRE_ARCH")
DOCKER_ARGS=(run --rm -i)

if [ "$USE_GPU" -eq 1 ]; then
  DOCKER_ARGS+=(--gpus all)
  ARGS+=(--require-capability "$REQUIRE_CAP" --require-sass-coverage)
else
  echo "note: no --gpu, checking the wheel arch set only; this does NOT prove the" >&2
  echo "      image runs on $TARGET. Re-run with --gpu on a real node." >&2
fi

if [ "$JSON" -eq 1 ]; then
  ARGS+=(--json)
fi

echo "validating $IMAGE against target=$TARGET (require arch $REQUIRE_ARCH, capability $REQUIRE_CAP)"
docker "${DOCKER_ARGS[@]}" --entrypoint "$IN_PYTHON" "$IMAGE" - "${ARGS[@]}" < "$CHECKER"
