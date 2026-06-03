#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${NPA_LIVE_E2E_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"

SOURCED_ENV_FILES=()

source_env_file() {
  local path="$1"
  if [[ ! -r "$path" ]]; then
    return 1
  fi

  set -a
  # shellcheck source=/dev/null
  . "$path"
  set +a
  SOURCED_ENV_FILES+=("$path")
}

if [[ -n "${NPA_LIVE_E2E_ENV_FILE:-}" ]]; then
  if ! source_env_file "$NPA_LIVE_E2E_ENV_FILE"; then
    printf 'Configured NPA_LIVE_E2E_ENV_FILE is not readable: %s\n' "$NPA_LIVE_E2E_ENV_FILE" >&2
    exit 2
  fi
else
  for env_file in \
    "${NPA_CONFIG_HOME:-${HOME}/.npa}/live-e2e.env" \
    "${XDG_CONFIG_HOME:-${HOME}/.config}/npa/live-e2e.env"; do
    source_env_file "$env_file" || true
  done
fi

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${NPA_LIVE_E2E_LOG_DIR:-${HOME}/npa-live-e2e-logs}"
LOG_FILE="${LOG_DIR}/live-e2e-${RUN_STAMP}.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

PYTHON_BIN="${NPA_LIVE_E2E_PYTHON_BIN:-${REPO_ROOT}/npa/.venv/bin/python}"
export NPA_SKYPILOT_BIN="${NPA_SKYPILOT_BIN:-${HOME}/.npa/skypilot-venv/bin/sky}"
export NPA_INTEGRATION_E2E="${NPA_INTEGRATION_E2E:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MARK_EXPR="${NPA_LIVE_E2E_MARK_EXPR:-gpu and e2e}"
PYTEST_TARGET="${NPA_LIVE_E2E_PYTEST_TARGET:-npa/tests/}"
TIMEOUT_SECONDS="${NPA_LIVE_E2E_TIMEOUT_SECONDS:-14400}"
KILL_AFTER_SECONDS="${NPA_LIVE_E2E_KILL_AFTER_SECONDS:-900}"
TEARDOWN_TIMEOUT_SECONDS="${NPA_LIVE_E2E_TEARDOWN_TIMEOUT_SECONDS:-1200}"
TEARDOWN_POLL_SECONDS="${NPA_LIVE_E2E_TEARDOWN_POLL_SECONDS:-30}"
CLUSTER_PREFIXES="${NPA_LIVE_E2E_CLUSTER_PREFIXES:-npa-vlm-live npa-sonic-e2e npa-spine-e2e npa-live-e2e}"
GITHUB_STATUS_CONTEXT="${NPA_LIVE_E2E_GITHUB_CONTEXT:-live-e2e}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
  log "ERROR: $*"
  exit 2
}

notification_url() {
  if [[ -n "${NPA_LIVE_E2E_NOTIFY_URL:-}" ]]; then
    printf '%s\n' "$NPA_LIVE_E2E_NOTIFY_URL"
    return
  fi
  if [[ -n "${NPA_NOTIFY_URL:-}" ]]; then
    printf '%s\n' "$NPA_NOTIFY_URL"
    return
  fi
  if [[ -n "${NOTIFY_URL:-}" ]]; then
    printf '%s\n' "$NOTIFY_URL"
  fi
}

notify_webhook() {
  local state="$1"
  local body="$2"
  local url
  url="$(notification_url || true)"
  if [[ -z "$url" ]]; then
    log "notification endpoint not configured; skipping notification"
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    log "curl not found; skipping notification"
    return 0
  fi

  if ! curl -fsS \
    -H "Title: NPA live e2e ${state}" \
    -H "Priority: default" \
    --data-binary "$body" \
    "$url" >/dev/null; then
    log "notification post failed"
  fi
}

github_status_payload() {
  local state="$1"
  local description="$2"
  local target_url="${3:-}"
  python3 - "$state" "$description" "$target_url" "$GITHUB_STATUS_CONTEXT" <<'PY'
import json
import sys

state, description, target_url, context = sys.argv[1:5]
payload = {
    "state": state,
    "description": description[:140],
    "context": context,
}
if target_url:
    payload["target_url"] = target_url
print(json.dumps(payload, separators=(",", ":")))
PY
}

post_github_status() {
  local state="$1"
  local description="$2"
  local target_url="${3:-${NPA_LIVE_E2E_TARGET_URL:-}}"
  if [[ "${NPA_LIVE_E2E_POST_GITHUB_STATUS:-1}" == "0" ]]; then
    log "GitHub commit status disabled"
    return 0
  fi

  local token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"

  local repo="${GITHUB_REPOSITORY:-${NPA_LIVE_E2E_GITHUB_REPO:-nebius/nebius-physical-ai}}"
  local sha="${NPA_LIVE_E2E_COMMIT_SHA:-}"
  if [[ -z "$sha" ]]; then
    sha="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  fi

  local payload
  payload="$(github_status_payload "$state" "$description" "$target_url")"
  if [[ -n "$token" ]] && command -v curl >/dev/null 2>&1 && curl -fsS \
    -X POST \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${token}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    --data "$payload" \
    "https://api.github.com/repos/${repo}/statuses/${sha}" >/dev/null; then
    return 0
  fi

  if command -v gh >/dev/null 2>&1; then
    if [[ -n "$token" ]]; then
      log "GitHub commit status curl post failed for ${repo}@${sha}; retrying with gh api"
    else
      log "GitHub token env not configured; trying authenticated gh api for ${repo}@${sha}"
    fi
    if post_github_status_with_gh "$payload" "$repo" "$sha" "$token"; then
      return 0
    fi
  fi

  log "GitHub commit status post failed for ${repo}@${sha}"
}

post_github_status_with_gh() {
  local payload="$1"
  local repo="$2"
  local sha="$3"
  local token="${4:-}"
  local cmd=(
    gh api
    --method POST
    -H "Accept: application/vnd.github+json"
    -H "X-GitHub-Api-Version: 2022-11-28"
    "/repos/${repo}/statuses/${sha}"
    --input -
  )

  if [[ -n "$token" ]]; then
    printf '%s' "$payload" | GH_TOKEN="$token" "${cmd[@]}" >/dev/null
  else
    printf '%s' "$payload" | "${cmd[@]}" >/dev/null
  fi
}

sky_status_output() {
  "$NPA_SKYPILOT_BIN" status --refresh 2>&1 || true
}

list_matching_clusters() {
  sky_status_output | awk -v prefixes="$CLUSTER_PREFIXES" '
    BEGIN {
      count = split(prefixes, parts, /[ ,]+/)
    }
    /^[[:space:]]*$/ { next }
    /^Fetching/ { next }
    /^Clusters/ { next }
    /^NAME[[:space:]]/ { next }
    {
      name = $1
      for (i = 1; i <= count; i++) {
        if (parts[i] != "" && index(name, parts[i]) == 1) {
          print name
        }
      }
    }
  ' | sort -u
}

down_matching_clusters() {
  local clusters cluster rc
  clusters="$(list_matching_clusters || true)"
  if [[ -z "$clusters" ]]; then
    log "No matching SkyPilot clusters found"
    return 0
  fi

  rc=0
  while IFS= read -r cluster; do
    [[ -z "$cluster" ]] && continue
    log "Tearing down SkyPilot cluster: ${cluster}"
    if ! "$NPA_SKYPILOT_BIN" down --yes "$cluster"; then
      log "SkyPilot teardown command failed for ${cluster}"
      rc=1
    fi
  done <<< "$clusters"
  return "$rc"
}

verify_no_matching_clusters() {
  local deadline clusters
  deadline=$((SECONDS + TEARDOWN_TIMEOUT_SECONDS))

  while (( SECONDS < deadline )); do
    clusters="$(list_matching_clusters || true)"
    if [[ -z "$clusters" ]]; then
      log "Verified no matching SkyPilot clusters remain"
      return 0
    fi
    log "Waiting for SkyPilot clusters to disappear: $(tr '\n' ' ' <<< "$clusters")"
    down_matching_clusters || true
    sleep "$TEARDOWN_POLL_SECONDS"
  done

  clusters="$(list_matching_clusters || true)"
  if [[ -n "$clusters" ]]; then
    log "SkyPilot clusters still present after teardown timeout:"
    printf '%s\n' "$clusters"
  fi
  return 1
}

run_with_timeout() {
  if command -v timeout >/dev/null 2>&1; then
    timeout --kill-after="${KILL_AFTER_SECONDS}s" "${TIMEOUT_SECONDS}s" "$@"
  else
    log "GNU timeout not found; running without an outer wall-clock cap"
    "$@"
  fi
}

run_pytest_attempts() {
  local candidates_csv rc attempt total gpu
  if [[ -n "${NPA_SONIC_E2E_GPU:-}" ]]; then
    candidates_csv="$NPA_SONIC_E2E_GPU"
  else
    candidates_csv="${NPA_LIVE_E2E_GPU_CANDIDATES:-H100:1,H200:1,A100:1,L40S:1,RTX6000:1}"
  fi

  IFS=',' read -r -a candidates <<< "$candidates_csv"
  total="${#candidates[@]}"
  rc=1
  attempt=1

  for gpu in "${candidates[@]}"; do
    gpu="$(xargs <<< "$gpu")"
    [[ -z "$gpu" ]] && continue
    export NPA_SONIC_E2E_GPU="$gpu"
    export NPA_LIVE_E2E_GPU_ATTEMPT="$gpu"

    log "Running live GPU e2e attempt ${attempt}/${total} with GPU candidate ${gpu}"
    log "Pytest command: ${PYTHON_BIN} -m pytest -m '${MARK_EXPR}' ${PYTEST_TARGET}"

    set +e
    run_with_timeout "$PYTHON_BIN" -m pytest -m "$MARK_EXPR" "$PYTEST_TARGET" -q
    rc=$?
    set -e

    if [[ "$rc" -eq 0 ]]; then
      log "Live GPU e2e passed with GPU candidate ${gpu}"
      return 0
    fi

    log "Live GPU e2e attempt ${attempt}/${total} failed with exit code ${rc}"
    down_matching_clusters || true
    verify_no_matching_clusters || return 1
    attempt=$((attempt + 1))
  done

  return "$rc"
}

finish() {
  local rc="$?"
  local cleanup_rc=0
  local state description
  trap - EXIT INT TERM

  log "Final SkyPilot cleanup starting"
  set +e
  down_matching_clusters
  cleanup_rc=$?
  if ! verify_no_matching_clusters; then
    cleanup_rc=1
  fi
  set -e

  if [[ "$rc" -eq 0 && "$cleanup_rc" -ne 0 ]]; then
    rc=1
  fi

  if [[ "$rc" -eq 0 ]]; then
    state="success"
    description="Live GPU e2e passed"
  else
    state="failure"
    description="Live GPU e2e failed"
  fi

  post_github_status "$state" "$description"
  notify_webhook "$state" "${description}. Log: ${LOG_FILE}"
  log "${description}. Log: ${LOG_FILE}"
  exit "$rc"
}

trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$REPO_ROOT"

log "Starting on-demand live GPU e2e"
log "Repository: ${REPO_ROOT}"
log "Log file: ${LOG_FILE}"
if [[ "${#SOURCED_ENV_FILES[@]}" -gt 0 ]]; then
  log "Loaded local env files: ${SOURCED_ENV_FILES[*]}"
else
  log "No local env file loaded; using current process environment"
fi

[[ -x "$PYTHON_BIN" ]] || die "Python executable is missing or not executable: ${PYTHON_BIN}"
[[ -x "$NPA_SKYPILOT_BIN" ]] || die "SkyPilot executable is missing or not executable: ${NPA_SKYPILOT_BIN}"

post_github_status "pending" "Live GPU e2e running"
notify_webhook "running" "Live GPU e2e started. Log: ${LOG_FILE}"

log "Clearing pre-existing live-e2e SkyPilot clusters before pytest"
down_matching_clusters || true
verify_no_matching_clusters

run_pytest_attempts
