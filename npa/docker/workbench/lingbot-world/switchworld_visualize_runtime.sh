#!/usr/bin/env bash
# Runtime delivery for the CPU-only SwitchWorld visualizer. This deliberately
# keeps Rerun's NumPy 2 closure out of the NumPy 1.26 Wan model runtime.
set -euo pipefail

readonly EX_CONFIG=78
readonly EX_UNAVAILABLE=69
readonly EX_SOFTWARE=70

CACHE_ROOT="${NPA_SWITCHWORLD_VISUALIZE_RUNTIME_CACHE:-/workspace/.cache/npa/switchworld/visualize-runtime}"
BASE_PYTHON="${NPA_WAN_BASE_PYTHON:-/opt/wan-base/bin/python}"
REQUIREMENTS="${NPA_SWITCHWORLD_VISUALIZE_REQUIREMENTS:-/opt/npa/switchworld-visualize-runtime-requirements.txt}"
OFFLINE="${NPA_SWITCHWORLD_VISUALIZE_RUNTIME_OFFLINE:-0}"

log() { printf 'switchworld-visualize-runtime: %s\n' "$*" >&2; }
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
import importlib.metadata as md
import av
import boto3
import numpy
import rerun

assert md.version("av") == "17.1.0"
assert md.version("attrs") == "26.1.0"
assert md.version("boto3") == "1.39.11"
assert md.version("botocore") == "1.39.17"
assert int(md.version("numpy").split(".", 1)[0]) >= 2
assert md.version("psutil") == "7.1.1"
assert md.version("pyarrow") == "23.0.1"
assert md.version("rerun-sdk") == "0.38.1"
assert md.version("urllib3") == "2.8.0"
assert av is not None and boto3 is not None and numpy is not None and rerun is not None
PY
  "$tree/venv/bin/python" -m rerun rrd --help >/dev/null
}

ensure_runtime() {
  local stamp target lock
  stamp="$(cache_stamp)"
  target="$CACHE_ROOT/$stamp"
  [[ "$OFFLINE" != 1 ]] || {
    [[ -L "$CACHE_ROOT/current" && "$(readlink "$CACHE_ROOT/current")" == "$target" ]] \
      || die "$EX_UNAVAILABLE" "offline cache does not match the visualization requirements and Python ABI"
    ready_tree "$target" \
      || die "$EX_UNAVAILABLE" "offline mode requested but no complete visualization cache exists"
    verify_tree "$target"
    return
  }

  lock="$CACHE_ROOT/.install.lock"
  mkdir -p "$CACHE_ROOT"
  exec 9>"$lock"
  flock 9
  find "$CACHE_ROOT" -maxdepth 1 -type d -name ".*.tmp.*" -exec rm -rf -- {} +
  find "$CACHE_ROOT" -maxdepth 1 -type l -name ".current.*" -delete
  if ! ready_tree "$target"; then
    tmp="$CACHE_ROOT/.${stamp}.tmp.$$"
    "$BASE_PYTHON" -m venv --copies "$tmp/venv"
    "$tmp/venv/bin/python" -m pip install \
      --disable-pip-version-check --no-cache-dir --ignore-installed --no-deps \
      --require-hashes -r "$REQUIREMENTS"
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
    if [[ -L "$CACHE_ROOT/current" ]] && ready_tree "$(readlink -f "$CACHE_ROOT/current")"; then
      printf '{"status":"ready","cache":"%s"}\n' "$CACHE_ROOT/current"
    else
      printf '{"status":"absent","cache":"%s"}\n' "$CACHE_ROOT/current"
    fi
    ;;
  health)
    [[ -r "$REQUIREMENTS" ]] || exit "$EX_SOFTWARE"
    printf '{"status":"ok","runtime":"switchworld-visualize-isolated"}\n'
    ;;
  exec)
    shift
    [[ $# -gt 0 ]] || die "$EX_CONFIG" "exec requires a command"
    ensure_runtime
    ensure_ssh_host_keys
    export PATH="$CACHE_ROOT/current/venv/bin:$PATH"
    exec "$@"
    ;;
  *)
    die "$EX_CONFIG" "unknown mode '$mode' (use ensure, warm, status, health, or exec)"
    ;;
esac
