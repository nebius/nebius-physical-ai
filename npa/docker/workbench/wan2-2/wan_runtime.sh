#!/usr/bin/env bash
# Runtime delivery for the CUDA-enabled PyTorch stack used by Wan 2.2.
# The public image deliberately contains no NVIDIA CUDA Python distribution.
set -euo pipefail

readonly EX_CONFIG=78
readonly EX_UNAVAILABLE=69
readonly EX_SOFTWARE=70
readonly ACCEPT_ENV=NPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS

CACHE_ROOT="${NPA_WAN_RUNTIME_CACHE:-/workspace/.cache/npa/wan2-2/runtime}"
BASE_PYTHON="${NPA_WAN_BASE_PYTHON:-/opt/wan-base/bin/python}"
REQUIREMENTS="${NPA_WAN_RUNTIME_REQUIREMENTS:-/opt/npa/wan2-2/runtime-requirements.txt}"
OFFLINE="${NPA_WAN_RUNTIME_OFFLINE:-0}"

log() { printf 'wan-runtime: %s\n' "$*" >&2; }
die() { local code="$1"; shift; log "$*"; exit "$code"; }

tmp=""
cleanup_tmp() {
  if [[ -n "$tmp" && -d "$tmp" ]]; then
    rm -rf -- "$tmp"
  fi
}
trap cleanup_tmp EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

accepted() {
  case "$(printf '%s' "${!ACCEPT_ENV:-}" | tr '[:lower:]' '[:upper:]')" in
    YES) return 0 ;;
    *) return 1 ;;
  esac
}

require_acceptance() {
  accepted && return 0
  cat >&2 <<'EOF'
wan-runtime: refusing to download or execute the NVIDIA CUDA Python runtime.

The image contains no NVIDIA CUDA wheels or libraries. The requested operation
would ask PyPI to deliver pinned torch 2.13.0 with its CUDA 13.0 NVIDIA
dependencies into this operator-owned writable cache. Nebius
cannot accept NVIDIA terms for the operator, so acceptance is never baked in.

Review the current terms before proceeding:
  https://docs.nvidia.com/cuda/eula/index.html
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/

Set NPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS=YES to record the operator's explicit
acceptance. Any other value refuses with exit 78. Nothing has been downloaded.
EOF
  exit "$EX_CONFIG"
}

cache_stamp() {
  local requirement_sha abi
  requirement_sha="$(sha256sum "$REQUIREMENTS" | cut -d' ' -f1)"
  abi="$("$BASE_PYTHON" -c 'import sys,sysconfig; print(f"{sys.version_info.major}.{sys.version_info.minor}-{sysconfig.get_platform()}")')"
  printf '%s|%s' "$requirement_sha" "$abi" | sha256sum | cut -c1-16
}

ready_tree() {
  local tree="$1"
  [[ -f "$tree/.complete" && -x "$tree/venv/bin/python" ]]
}

verify_tree() {
  local tree="$1"
  "$tree/venv/bin/python" -m pip check
  "$tree/venv/bin/python" - <<'PY'
import importlib.metadata
import torch

def public_version(value):
    # A wheel-local suffix is harmless, but prefix extensions and post/dev
    # releases are not accepted as the exact reviewed public version.
    return value.split("+", 1)[0]

assert public_version(torch.__version__) == "2.13.0", torch.__version__
assert torch.version.cuda == "13.0", torch.version.cuda
for name, expected in {
    "torchvision": "0.28.0",
    "cuda-toolkit": "13.0.3.0",
    "cuda-bindings": "13.3.1",
    "cuda-pathfinder": "1.6.0",
    "nvidia-cublas": "13.1.1.3",
    "nvidia-cuda-cupti": "13.0.85",
    "nvidia-cuda-nvrtc": "13.0.88",
    "nvidia-cuda-runtime": "13.0.96",
    "nvidia-cudnn-cu13": "9.20.0.48",
    "nvidia-cufft": "12.0.0.61",
    "nvidia-cufile": "1.15.1.6",
    "nvidia-curand": "10.4.0.35",
    "nvidia-cusolver": "12.0.4.66",
    "nvidia-cusparse": "12.6.3.3",
    "nvidia-cusparselt-cu13": "0.8.1",
    "nvidia-nccl-cu13": "2.29.7",
    "nvidia-nvjitlink": "13.3.33",
    "nvidia-nvshmem-cu13": "3.4.5",
    "nvidia-nvtx": "13.0.85",
    "triton": "3.7.1",
}.items():
    observed = importlib.metadata.version(name)
    assert public_version(observed) == expected, (name, observed, expected)
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
PY
}

ensure_runtime() {
  # This gate intentionally precedes mkdir, locks, package probes, and network.
  require_acceptance
  local stamp target lock base_site cache_site
  stamp="$(cache_stamp)"
  target="$CACHE_ROOT/$stamp"
  [[ "$OFFLINE" != 1 ]] || {
    [[ -L "$CACHE_ROOT/current" && "$(readlink "$CACHE_ROOT/current")" == "$target" ]] \
      || die "$EX_UNAVAILABLE" "offline cache does not match the current requirements and Python ABI"
    ready_tree "$target" \
      || die "$EX_UNAVAILABLE" "offline mode requested but no complete current-stamp cache exists"
    verify_tree "$target"
    return
  }

  lock="$CACHE_ROOT/.install.lock"
  mkdir -p "$CACHE_ROOT"
  exec 9>"$lock"
  flock 9
  # The lock covers the whole Wan runtime root, so no live installer can own
  # these partial trees/symlinks. Clean every stale stamp, including remnants
  # from a SIGKILL before a requirements/ABI change.
  find "$CACHE_ROOT" -maxdepth 1 -type d -name ".*.tmp.*" \
    -exec rm -rf -- {} +
  find "$CACHE_ROOT" -maxdepth 1 -type l -name ".current.*" -delete
  if ! ready_tree "$target"; then
    # A killed or failed installer must not consume another multi-gigabyte tree
    # on every retry. The lock makes this exact-stamp cleanup race-free.
    tmp="$CACHE_ROOT/.${stamp}.tmp.$$"
    "$BASE_PYTHON" -m venv --copies "$tmp/venv"
    base_site="$("$BASE_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    cache_site="$("$tmp/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    printf '%s\n' "$base_site" > "$cache_site/_npa_wan_image_site.pth"
    "$tmp/venv/bin/python" -m pip install \
      --disable-pip-version-check --no-cache-dir --ignore-installed --no-deps \
      --require-hashes \
      -r "$REQUIREMENTS"
    verify_tree "$tmp"
    cp "$REQUIREMENTS" "$tmp/runtime-requirements.txt"
    "$tmp/venv/bin/python" -m pip freeze --all > "$tmp/pip-freeze.txt"
    : > "$tmp/.complete"
    rm -rf "$target"
    mv "$tmp" "$target"
    tmp=""
  fi
  ln -sfn "$target" "$CACHE_ROOT/.current.$$"
  mv -Tf "$CACHE_ROOT/.current.$$" "$CACHE_ROOT/current"
  verify_tree "$CACHE_ROOT/current"
}

status() {
  local stamp target
  stamp="$(cache_stamp)"
  target="$CACHE_ROOT/$stamp"
  if [[ -L "$CACHE_ROOT/current" && "$(readlink "$CACHE_ROOT/current")" == "$target" ]] \
    && ready_tree "$target"; then
    printf '{"status":"ready","cache":"%s"}\n' "$CACHE_ROOT/current"
  else
    printf '{"status":"absent","cache":"%s"}\n' "$CACHE_ROOT/current"
  fi
}

ensure_ssh_host_keys() {
  command -v ssh-keygen >/dev/null 2>&1 || return 0
  [[ -s /etc/ssh/ssh_host_ed25519_key ]] && return 0
  sudo -n ssh-keygen -A >/dev/null
}

mode="${1:-ensure}"
case "$mode" in
  ensure|warm)
    ensure_runtime
    ;;
  status)
    status
    ;;
  health)
    [[ -r /opt/byof/LICENSE.txt && -r "$REQUIREMENTS" ]] || exit "$EX_SOFTWARE"
    printf '{"status":"ok","source_ref":"42bf4cfaa384bc21833865abc2f9e6c0e67233dc","cuda_runtime":"runtime-fetch"}\n'
    ;;
  version)
    printf 'wan2.2 source=42bf4cfaa384bc21833865abc2f9e6c0e67233dc torch=2.13.0 cu=13.0 provisioning=runtime-fetch\n'
    ;;
  exec)
    shift
    [[ $# -gt 0 ]] || die "$EX_CONFIG" "exec requires a command"
    ensure_runtime
    ensure_ssh_host_keys
    export PATH="$CACHE_ROOT/current/venv/bin:$PATH"
    export PYTHONPATH="/opt/byof${PYTHONPATH:+:$PYTHONPATH}"
    exec "$@"
    ;;
  *)
    die "$EX_CONFIG" "unknown mode '$mode' (use ensure, warm, status, health, version, or exec)"
    ;;
esac
