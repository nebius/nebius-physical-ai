#!/usr/bin/env bash
#
# isaac_python.sh - the Isaac Python entrypoint for images that fetch Isaac at run time.
#
# Installed as /opt/npa/bin/isaac-python AND as /isaac-sim/python.sh.
#
# WHY /isaac-sim/python.sh
#   Every caller in this repo already reaches Isaac through that path - seven SkyPilot
#   task templates (`PYTHON_BIN="${ISAAC_LAB_PYTHON:-/isaac-sim/python.sh}"`), the
#   sim2real engine (`PYBIN=/isaac-sim/python.sh`), byo_isaac_{trainer,policy_rollout,
#   eval}, rl_sweep, retargeting, and sonic's entrypoint. Keeping the path means the
#   re-architecture needs no changes at ~30 call sites.
#
#   More importantly it is the only reliable trigger: SkyPilot and Kubernetes pods
#   override the image ENTRYPOINT, so an entrypoint-only bootstrap would silently never
#   run in the real submission path. Every path does, however, invoke this interpreter.
#
#   In the old images this was NVIDIA's launcher inside a baked Isaac Sim install. Here
#   it is our own ~40-line shell script: it makes sure the operator-consented Isaac
#   install exists, then hands over to the interpreter inside it. No NVIDIA bytes ship
#   in the image; see npa/scripts/scan_image_omniverse_payload.py, which asserts exactly
#   that against the built image.
#
# All bootstrap chatter goes to stderr, because callers parse this interpreter's stdout.
#
set -uo pipefail

# Match isaac_bootstrap.sh: unset means accepted by default, while an explicitly
# empty/recognized-negative value remains an opt-out.
ACCEPT_EULA="${ACCEPT_EULA-Y}"

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
BOOTSTRAP="${NPA_ISAAC_BOOTSTRAP:-}"
if [ -z "$BOOTSTRAP" ]; then
  for candidate in \
    "$SCRIPT_DIR/isaac_bootstrap.sh" \
    /opt/npa/docker/workbench/common/isaac_bootstrap.sh \
    /opt/npa/bin/isaac-bootstrap
  do
    [ -x "$candidate" ] && { BOOTSTRAP="$candidate"; break; }
  done
fi
[ -n "$BOOTSTRAP" ] || {
  echo "isaac-python: cannot find isaac_bootstrap.sh; set NPA_ISAAC_BOOTSTRAP" >&2
  exit 70
}

# `ensure` prints the ready cache tree on stdout and everything else on stderr, so
# capturing stdout here keeps the caller's stdout clean.
TREE="$("$BOOTSTRAP" ensure)" || exit $?
PYTHON="$TREE/venv/bin/python"
[ -x "$PYTHON" ] || {
  echo "isaac-python: bootstrap reported ${TREE} but ${PYTHON} is not executable" >&2
  exit 70
}

# `ensure` runs in command substitution, so exports made by the bootstrap cannot reach
# this launcher. A successful ensure has already normalized and validated the public
# value through the single shell parser, so propagate only the internal Kit acknowledgement.
# This never derives or enables privacy consent.
export OMNI_KIT_ACCEPT_EULA=YES

# Isaac Lab's repo layout (scripts/, source/, apps/) lives in the cache because the
# isaaclab wheel ships the library without scripts/, and every SkyPilot Isaac task
# asserts `test -f /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py`.
# Link the read-only pieces in and leave /workspace/isaaclab itself a real writable
# directory, so NPA_ISAAC_LAB_OUTPUT_DIR=/workspace/isaaclab/npa-runs still works when
# the cache volume is mounted read-only.
ISAAC_LAB_WORKDIR="${NPA_ISAAC_LAB_WORKDIR:-/workspace/isaaclab}"
if [ -d "$ISAAC_LAB_WORKDIR" ] && [ -w "$ISAAC_LAB_WORKDIR" ] && [ -d "$TREE/isaaclab-src" ]; then
  for entry in scripts source apps tools; do
    if [ -e "$TREE/isaaclab-src/$entry" ] && [ ! -e "$ISAAC_LAB_WORKDIR/$entry" ]; then
      ln -sfn "$TREE/isaaclab-src/$entry" "$ISAAC_LAB_WORKDIR/$entry" 2>/dev/null || true
    fi
  done
fi

# Kit writes caches, logs and shader compilations; keep them off the (possibly
# read-only) Isaac cache and out of the image, defaulting to per-container scratch.
export OMNI_USER_DIR="${OMNI_USER_DIR:-/tmp/isaac-sim-cache}"
export OMNI_LOG_DIR="${OMNI_LOG_DIR:-/tmp/isaac-sim-cache/logs}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-runtime}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/isaac-sim-cache/xdg-cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/tmp/isaac-sim-cache/xdg-config}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-/tmp/isaac-sim-cache/xdg-data}"
# The pip Isaac distribution carries a Kit portable marker beside its binaries,
# which takes precedence over XDG/OMNI path overrides. Every AppLauncher entrypoint
# in the exact runtime consumes this argument so mutable user, shader, log, and data
# state lands in Pod-local scratch instead of the read-only dependency closure.
export NPA_ISAAC_KIT_ARGS="${NPA_ISAAC_KIT_ARGS:---portable-root /tmp/npa-isaac-kit}"
mkdir -p \
  "$OMNI_USER_DIR" "$OMNI_LOG_DIR" "$XDG_RUNTIME_DIR" \
  "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" \
  /tmp/npa-isaac-kit 2>/dev/null || true
chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true

exec "$PYTHON" "$@"
