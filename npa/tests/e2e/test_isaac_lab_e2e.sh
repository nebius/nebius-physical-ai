#!/usr/bin/env bash

set -uo pipefail

SKIP_DEPLOY=0
ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-deploy)
      SKIP_DEPLOY=1
      shift
      ;;
    --)
      shift
      while [ "$#" -gt 0 ]; do
        ARGS+=("$1")
        shift
      done
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [ "${#ARGS[@]}" -ne 4 ]; then
  echo "Usage: $0 [--skip-deploy] <project> <name> <gpu-type> <gpu-preset>" >&2
  exit 2
fi

PROJECT="${ARGS[0]}"
NAME="${ARGS[1]}"
GPU_TYPE="${ARGS[2]}"
GPU_PRESET="${ARGS[3]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLI="$NPA_DIR/.venv/bin/npa"
PYTHON_BIN="$NPA_DIR/.venv/bin/python"
TOOL="isaac-lab"
REMOTE_ROOT="/tmp/npa-e2e-isaac-lab-smoke-$$"

TOTAL=0
FAILED=0
DEPLOY_ATTEMPTED=0
TEARDOWN_DONE=0
LOCAL_TMP=""
SSH_HOST=""
SSH_USER=""
SSH_KEY=""
SSH_TARGET=""
SSH_OPTS=()

record_pass() {
  TOTAL=$((TOTAL + 1))
  printf 'PASS: %s\n' "$1"
}

record_fail() {
  TOTAL=$((TOTAL + 1))
  FAILED=$((FAILED + 1))
  printf 'FAIL: %s\n' "$1"
}

print_summary() {
  local passed=$((TOTAL - FAILED))
  printf 'SUMMARY: %d/%d steps passed\n' "$passed" "$TOTAL"
}

print_command() {
  printf '+'
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

run_step() {
  local label="$1"
  shift
  print_command "$@"
  if "$@"; then
    record_pass "$label"
    return 0
  fi
  record_fail "$label"
  return 1
}

run_npa_step() {
  local label="$1"
  shift
  run_step "$label" "$CLI" workbench "$TOOL" -p "$PROJECT" -n "$NAME" "$@"
}

run_teardown_step() {
  if [ "$TEARDOWN_DONE" -eq 1 ]; then
    return 0
  fi
  TEARDOWN_DONE=1
  run_npa_step "destroy workbench" deploy --destroy --gpu-type "$GPU_TYPE" --gpu-preset "$GPU_PRESET"
}

cleanup() {
  local status=$?
  trap - EXIT
  if [ "$DEPLOY_ATTEMPTED" -eq 1 ] && [ "$TEARDOWN_DONE" -eq 0 ]; then
    run_teardown_step || true
  fi
  if [ -n "$LOCAL_TMP" ]; then
    rm -rf "$LOCAL_TMP"
  fi
  print_summary
  if [ "$FAILED" -ne 0 ] || [ "$status" -ne 0 ]; then
    exit 1
  fi
  exit 0
}
trap cleanup EXIT

slugify() {
  local value
  value="$(printf '%s' "$1" | tr -c 'A-Za-z0-9_-' '-' | sed 's/^-*//;s/-*$//')"
  if [ -z "$value" ]; then
    value="workbench"
  fi
  printf '%s' "$value"
}

validate_rt_core_gpu() {
  local normalized
  normalized="$(printf '%s' "$GPU_TYPE" | tr '[:upper:]_' '[:lower:]-')"
  case "$normalized" in
    *l40s*|*rtx*6000*)
      return 0
      ;;
    *)
      printf 'Isaac Lab e2e requires an RT-core GPU type, such as gpu-l40s-a or gpu-rtx-pro-6000. Got: %s\n' "$GPU_TYPE" >&2
      return 1
      ;;
  esac
}

resolve_ssh() {
  local values
  values="$("$PYTHON_BIN" - "$PROJECT" "$NAME" <<'PY'
import os
import sys

from npa.clients.config import resolve_ssh_config

cfg = resolve_ssh_config(project=sys.argv[1], name=sys.argv[2])
print(cfg.ssh.host)
print(cfg.ssh.user)
print(os.path.expanduser(cfg.ssh.key_path))
PY
)"
  if [ "$?" -ne 0 ]; then
    return 1
  fi

  SSH_HOST="$(printf '%s\n' "$values" | sed -n '1p')"
  SSH_USER="$(printf '%s\n' "$values" | sed -n '2p')"
  SSH_KEY="$(printf '%s\n' "$values" | sed -n '3p')"
  SSH_TARGET="${SSH_USER}@${SSH_HOST}"
  SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)

  [ -n "$SSH_HOST" ] && [ -n "$SSH_USER" ] && [ -n "$SSH_KEY" ]
}

stage_smoke_files() {
  resolve_ssh || return 1

  LOCAL_TMP="$(mktemp -d "${TMPDIR:-/tmp}/npa-isaac-lab-e2e.XXXXXX")"
  mkdir -p "$LOCAL_TMP/npa/smoke" || return 1
  printf '' > "$LOCAL_TMP/npa/__init__.py" || return 1
  cp "$NPA_DIR/pyproject.toml" "$LOCAL_TMP/pyproject.toml" || return 1
  cp "$NPA_DIR/src/npa/smoke/__init__.py" "$LOCAL_TMP/npa/smoke/__init__.py" || return 1
  cp "$NPA_DIR/src/npa/smoke/_versions.py" "$LOCAL_TMP/npa/smoke/_versions.py" || return 1
  cp "$NPA_DIR/src/npa/smoke/test_isaac_lab_env.py" "$LOCAL_TMP/npa/smoke/test_isaac_lab_env.py" || return 1
  cp "$NPA_DIR/src/npa/smoke/test_isaac_lab_functional.py" "$LOCAL_TMP/npa/smoke/test_isaac_lab_functional.py" || return 1

  print_command ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "rm -rf '$REMOTE_ROOT' && mkdir -p '$REMOTE_ROOT'"
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "rm -rf '$REMOTE_ROOT' && mkdir -p '$REMOTE_ROOT'" || return 1

  print_command scp -r -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$LOCAL_TMP/npa" "$LOCAL_TMP/pyproject.toml" "$SSH_TARGET:$REMOTE_ROOT/"
  scp -r -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$LOCAL_TMP/npa" "$LOCAL_TMP/pyproject.toml" "$SSH_TARGET:$REMOTE_ROOT/" || return 1
}

run_remote_script_step() {
  local label="$1"
  local script="$2"
  print_command ssh "${SSH_OPTS[@]}" "$SSH_TARGET" bash -s
  if ssh "${SSH_OPTS[@]}" "$SSH_TARGET" bash -s <<< "$script"; then
    record_pass "$label"
    return 0
  fi
  record_fail "$label"
  return 1
}

run_step "validate RT-core GPU flag" validate_rt_core_gpu || exit 1
run_step "npa CLI exists" test -x "$CLI" || exit 1

if [ "$SKIP_DEPLOY" -eq 1 ]; then
  printf 'SKIP: deploy workbench (--skip-deploy)\n'
else
  DEPLOY_ATTEMPTED=1
  run_npa_step "deploy workbench" deploy --gpu-type "$GPU_TYPE" --gpu-preset "$GPU_PRESET" || exit 1
fi
run_npa_step "system-info exits 0" system-info || exit 1
run_npa_step "status exits 0" status || exit 1
run_step "scp smoke tests to VM" stage_smoke_files || exit 1

run_remote_script_step "run Isaac Lab environment smoke test" "set -euo pipefail
source /opt/isaac-lab/venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=\"\${OMNI_KIT_ACCEPT_EULA:-YES}\"
python - <<'PY'
try:
    import tomllib
except ModuleNotFoundError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"tomli\"])
PY
export PYTHONPATH=\"$REMOTE_ROOT\"
python -m npa.smoke.test_isaac_lab_env" || exit 1

run_remote_script_step "run Isaac Lab functional smoke test" "set -euo pipefail
source /opt/isaac-lab/venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=\"\${OMNI_KIT_ACCEPT_EULA:-YES}\"
python - <<'PY'
try:
    import tomllib
except ModuleNotFoundError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"tomli\"])
PY
export PYTHONPATH=\"$REMOTE_ROOT\"
python -m npa.smoke.test_isaac_lab_functional" || exit 1

RUN_SLUG="$(slugify "$NAME")"
RUN_ROOT="/tmp/npa-e2e-isaac-lab-${RUN_SLUG}-$(date +%Y%m%d%H%M%S)"
CHECKPOINT_PATH="$RUN_ROOT/npa_isaac_lab_checkpoint.pt"

run_npa_step "run short Isaac Lab training job" train --task Isaac-Reach-Franka-v0 --num-envs 64 --steps 1 --output-dir "$RUN_ROOT" || exit 1
run_npa_step "run Isaac Lab eval job" eval --task Isaac-Reach-Franka-v0 --checkpoint "$CHECKPOINT_PATH" --num-episodes 1 --output-dir "$RUN_ROOT/eval" || exit 1
if [ "$SKIP_DEPLOY" -eq 1 ]; then
  printf 'SKIP: destroy workbench (--skip-deploy)\n'
else
  run_teardown_step || exit 1
fi
