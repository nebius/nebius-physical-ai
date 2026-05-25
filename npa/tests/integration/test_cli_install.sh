#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/npa-cli-install.XXXXXX")"
VENV_DIR="$TMP_ROOT/venv"
TMP_HOME="$TMP_ROOT/home"
NPA_BIN="$VENV_DIR/bin/npa"

TOTAL=0
FAILED=0

cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

record_pass() {
  local label="$1"
  TOTAL=$((TOTAL + 1))
  printf 'PASS %s\n' "$label"
}

record_fail() {
  local label="$1"
  local output="${2:-}"
  TOTAL=$((TOTAL + 1))
  FAILED=$((FAILED + 1))
  printf 'FAIL %s\n' "$label"
  if [[ -n "$output" ]]; then
    printf '%s\n' "$output" | sed 's/^/  /'
  fi
}

finish() {
  local passed=$((TOTAL - FAILED))
  printf '\nSummary: %d passed, %d failed, %d total\n' "$passed" "$FAILED" "$TOTAL"
  if [[ "$FAILED" -ne 0 ]]; then
    exit 1
  fi
}

run_setup_check() {
  local label="$1"
  shift
  local output

  output="$("$@" 2>&1)"
  local rc=$?
  if [[ "$rc" -eq 0 ]]; then
    record_pass "$label"
    return 0
  fi

  record_fail "$label" "$output"
  finish
}

run_npa_check() {
  local label="$1"
  shift
  local output

  output="$(HOME="$TMP_HOME" "$NPA_BIN" "$@" 2>&1)"
  local rc=$?
  if [[ "$rc" -eq 0 ]]; then
    record_pass "$label"
  else
    record_fail "$label" "$output"
  fi
}

mkdir -p "$TMP_HOME"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

run_setup_check "create temporary venv" python3 -m venv "$VENV_DIR"
run_setup_check "install npa from local source" bash -c \
  'cd "$1" && "$2/bin/python" -m pip install .' bash "$PROJECT_DIR" "$VENV_DIR"

if [[ -x "$NPA_BIN" ]]; then
  record_pass "npa binary exists in venv"
else
  record_fail "npa binary exists in venv" "Missing executable: $NPA_BIN"
  finish
fi

run_npa_check "npa --help" --help
run_npa_check "npa workbench --help" workbench --help
run_npa_check "npa workbench lerobot --help" workbench lerobot --help
run_npa_check "npa workbench genesis --help" workbench genesis --help
run_npa_check "npa adapter --help" adapter --help
run_npa_check "npa workbench workflow --help" workbench workflow --help
run_npa_check "npa workbench lerobot list" workbench lerobot list

finish
