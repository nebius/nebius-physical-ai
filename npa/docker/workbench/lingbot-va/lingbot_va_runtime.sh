#!/usr/bin/env bash
# Create one immutable-identity CUDA 13.0 runtime in an operator-owned volume.
# CUDA 13.0 carries the sm_120 wheel required by the selected RTX target. This
# script never asks for consent and never fetches datasets. The public
# LingBot checkpoints are fetched only by the invoked workload.
set -euo pipefail

readonly EX_CONFIG=78
readonly EX_UNAVAILABLE=69
readonly EX_SOFTWARE=70

CACHE_ROOT="${NPA_LINGBOT_VA_RUNTIME_CACHE:-/workspace/.cache/npa/lingbot-va/runtime}"
REQUIREMENTS="${NPA_LINGBOT_VA_RUNTIME_REQUIREMENTS:-/opt/npa/lingbot-va/runtime-requirements.txt}"
OFFLINE="${NPA_LINGBOT_VA_RUNTIME_OFFLINE:-0}"

die() { printf 'lingbot-va-runtime: %s\n' "$*" >&2; exit "$1"; }

stamp() {
  local abi requirement_sha
  abi="$(python3 -c 'import sys,sysconfig; print(f"{sys.version_info.major}.{sys.version_info.minor}-{sysconfig.get_platform()}")')"
  requirement_sha="$(sha256sum "$REQUIREMENTS" | cut -d' ' -f1)"
  printf '%s|%s' "$abi" "$requirement_sha" | sha256sum | cut -c1-16
}

ready() { [[ -x "$1/venv/bin/python" && -f "$1/.complete" ]]; }

verify() {
  # The exact upstream LeRobot 0.3.3 installation deliberately overrides its
  # stale Torch, TorchVision, Transformers, and datasets upper bounds. The
  # pinned datasets 5.0.1 reader is exercised against the native LingBot
  # loader before GPU submission. Its optional Linux physical-input backend
  # also declares evdev and python-xlib, neither of which the offline LIBERO
  # path imports. Keep pip's full integrity check, but reject every mismatch
  # other than these explicitly named compatibility overrides.
  local pip_check unexpected
  if ! pip_check="$("$1/venv/bin/python" -m pip check 2>&1)"; then
    unexpected="$(printf '%s\n' "$pip_check" | grep -Ev '^(lerobot 0\.3\.3 has requirement (datasets|torch|torchvision|transformers).*, but you have |pynput 1\.8\.1 requires (evdev|python-xlib), which is not installed\.)' || true)"
    if [[ -n "$unexpected" ]]; then
      printf 'lingbot-va-runtime: unexpected dependency mismatch: %s\n' "$unexpected" >&2
      return "$EX_SOFTWARE"
    fi
    printf '%s\n' "$pip_check" >&2
  fi
  "$1/venv/bin/python" - <<'PY'
import torch
import datasets
assert torch.__version__.split('+', 1)[0] == '2.13.0', torch.__version__
assert torch.version.cuda == '13.0', torch.version.cuda
assert datasets.__version__ == '5.0.1', datasets.__version__
from torch.nn.attention.flex_attention import flex_attention
import wan_va.modules.model
print(f'torch={torch.__version__} cuda={torch.version.cuda} flex_attention=ready')
PY
}

promote() {
  local candidate="$1" target="$2"
  cp "$REQUIREMENTS" "$candidate/runtime-requirements.txt"
  "$candidate/venv/bin/python" -m pip freeze --all > "$candidate/pip-freeze.txt"
  : > "$candidate/.complete"
  mv "$candidate" "$target"
}

ensure() {
  local id target tmp lock
  id="$(stamp)"
  target="$CACHE_ROOT/$id"
  if [[ "$OFFLINE" == 1 ]]; then
    [[ -L "$CACHE_ROOT/current" && "$(readlink "$CACHE_ROOT/current")" == "$target" && -d "$target" ]] \
      || die "$EX_UNAVAILABLE" "offline cache does not match the reviewed runtime identity"
    ready "$target" || die "$EX_UNAVAILABLE" "offline cache is incomplete"
    verify "$target"
    return
  fi
  mkdir -p "$CACHE_ROOT"
  lock="$CACHE_ROOT/.install.lock"
  exec 9>"$lock"
  flock 9
  # A verification or process interruption can leave a complete candidate
  # before its atomic rename. Re-verify such a cache under this identity lock
  # before discarding it or downloading the multi-gigabyte CUDA stack again.
  if ! ready "$target"; then
    for stale in "$CACHE_ROOT"/."$id".tmp.*; do
      [[ -d "$stale" && -x "$stale/venv/bin/python" ]] || continue
      if PYTHONPATH="/opt/npa-native:/opt/lingbot-va" verify "$stale"; then
        promote "$stale" "$target"
        break
      fi
      rm -rf -- "$stale"
    done
  fi
  if ! ready "$target"; then
    rm -rf -- "$target"
    tmp="$CACHE_ROOT/.${id}.tmp.$$"
    rm -rf -- "$tmp"
    python3 -m venv "$tmp/venv"
    "$tmp/venv/bin/python" -m pip install --upgrade 'pip==25.1.1' 'setuptools==80.9.0' 'wheel==0.45.1'
    # flash-attn's metadata imports torch. Install the CUDA 13.0 sm_120-capable
    # Torch/TorchVision pair first, then resolve the rest of the upstream
    # closure. The selected Flex training and Torch inference paths do not
    # execute Flash Attention, TorchAudio, or Accelerate.
    "$tmp/venv/bin/python" -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu130 \
      'torch==2.13.0+cu130' 'torchvision==0.28.0+cu130'
    "$tmp/venv/bin/python" -m pip install --no-cache-dir -r "$REQUIREMENTS"
    # Upstream explicitly installs this historical LeRobot revision with
    # --no-deps because its package metadata caps Torch below LingBot-VA's
    # required 2.9.0. Its non-Torch dependencies are listed above and resolved
    # before this deliberate compatibility override.
    # The offline workload does not initialize LeRobot's physical-input
    # backends. Install their two metadata-only packages without dependencies:
    # resolving pynput's Linux evdev extra would try to compile against a host
    # kernel header, which is neither portable nor needed here.
    "$tmp/venv/bin/python" -m pip install --no-cache-dir --no-deps \
      'lerobot==0.3.3' 'pynput==1.8.1' 'pyserial==3.5'
    PYTHONPATH="/opt/npa-native:/opt/lingbot-va" verify "$tmp"
    promote "$tmp" "$target"
  fi
  ln -sfn "$target" "$CACHE_ROOT/.current.$$"
  mv -Tf "$CACHE_ROOT/.current.$$" "$CACHE_ROOT/current"
  PYTHONPATH="/opt/npa-native:/opt/lingbot-va" verify "$CACHE_ROOT/current"
}

case "${1:-health}" in
  health)
    test -s /opt/lingbot-va/LICENSE.txt
    test -r "$REQUIREMENTS"
    printf '%s\n' '{"status":"ok","source_ref":"7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb","cuda_runtime":"runtime-fetch"}'
    ;;
  status)
    id="$(stamp)"; target="$CACHE_ROOT/$id"
    if [[ -L "$CACHE_ROOT/current" && "$(readlink "$CACHE_ROOT/current")" == "$target" ]] && ready "$target"; then
      printf '{"status":"ready","cache":"%s"}\n' "$CACHE_ROOT/current"
    else
      printf '{"status":"absent","cache":"%s"}\n' "$CACHE_ROOT/current"
    fi
    ;;
  ensure|warm) ensure ;;
  exec)
    shift
    [[ "$#" -gt 0 ]] || die "$EX_CONFIG" "exec needs a command"
    ensure
    export PATH="$CACHE_ROOT/current/venv/bin:$PATH"
    export PYTHONPATH="/opt/npa-native:/opt/lingbot-va${PYTHONPATH:+:$PYTHONPATH}"
    exec "$@"
    ;;
  *) die "$EX_CONFIG" "use health, status, ensure, warm, or exec" ;;
esac
