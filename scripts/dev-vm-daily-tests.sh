#!/usr/bin/env bash
#
# dev-vm-daily-tests.sh - run the daily NPA test tiers on the dev/operator VM.
#
# WHY
#   GitHub-hosted runners live outside the operator's Nebius environment, so they
#   cannot reach real infrastructure to run e2e tests (see
#   docs/testing/live-e2e.md). This script inverts the direction: the
#   .github/workflows/dev-vm-daily-tests.yml workflow SSHes INTO the dev VM
#   (which already has ~/.npa credentials, config, and SkyPilot) and runs the
#   tests there. It is also runnable by hand on the dev VM.
#
# ISOLATION
#   The script uses a DEDICATED CI checkout (NPA_CI_REPO_DIR, default
#   ~/npa-ci-daily) with its own venv so it never disturbs the shared dev clone
#   or other agents' worktrees. See npa/scripts/dev_vm_isolated_session.sh for
#   the interactive equivalent.
#
# GUARDRAIL
#   The GPU-spending live path (scripts/live-e2e.sh, `-m "gpu and e2e"`) must
#   never run unattended on a schedule: it provisions real GPU clusters and can
#   leak spend overnight (docs/testing/live-e2e.md). The `live-gpu` tier here is
#   opt-in only and additionally requires NPA_DAILY_ALLOW_LIVE_GPU=1; the GitHub
#   workflow refuses to set that flag for scheduled runs.
#
# USAGE
#   dev-vm-daily-tests.sh [tier] [git-ref]
#     tier     one of: unit | e2e | e2e-daily | gpu-daily | e2e-serverless
#              | mutation-live | live-gpu   (default: unit)
#     git-ref  branch, tag, or sha to test                       (default: main)
#
#   e2e-daily is the scheduled default: every day it runs the >= 4-step
#   comprehensive workflow E2E coverage gate + plans every npa.workflow spec,
#   checks that every workbench image is reachable in the registry, and runs a
#   DIFFERENT rotating shard of the real S3 e2e suite (so the whole suite is
#   covered over NPA_DAILY_E2E_SHARDS days). When NPA_DAILY_ENABLE_GPU=1 it also
#   submits ONE rotating real-GPU workflow E2E (a self-cleaning managed job).
#
#   gpu-daily runs just that one rotating real-GPU workflow submit (a different
#   GPU twin each day). The full `gpu and e2e` sky-cluster suite stays on the
#   manual-only `live-gpu` tier.
#
# ENV
#   NPA_CI_REPO_DIR        dedicated CI checkout dir  (default: $HOME/npa-ci-daily)
#   NPA_CI_REPO_URL        git remote to clone        (default: derived from the
#                          shared dev clone or an existing CI checkout)
#   NPA_REPO               shared dev clone used only to derive the remote URL
#                          (default: $HOME/nebius-physical-ai)
#   NPA_DAILY_LOG_DIR      log directory              (default: $HOME/npa-daily-test-logs)
#   NPA_DAILY_FRESH=1      hard-reset + clean the checkout to a pristine ref
#                          (default: 1; keeps .venv)
#   NPA_DAILY_DETACH=1     re-exec under `setsid` as a separate process group
#                          (default: 1)
#   NPA_DAILY_E2E_SHARDS   how many days to spread the S3 e2e suite over
#                          (default: 7)
#   NPA_REGISTRY        Full registry prefix for the daily all-image reachability
#                          check (unset => that check is skipped with a warning)
#   NPA_DAILY_ENABLE_GPU=1 run one rotating real-GPU workflow submit as part of
#                          the e2e-daily tier (default off). gpu-daily always runs it.
#   NPA_DAILY_AGENT_GPU_E2E=1 replace the rotating case with the agent-confirmed
#                          self-hosted VLM GPU proof (requires a deployed agent).
#   NPA_DAILY_GPU_MAX_WAIT_SECONDS / _POLL_SECONDS / NPA_DAILY_GPU_PYTEST_TIMEOUT
#                          bound the GPU submit wait (defaults 2400 / 30 / 2600)
#   NPA_DAILY_ALLOW_LIVE_GPU=1  required to run the full live-gpu suite tier
#   NPA_DAILY_PIP_EXTRAS   pip extras to install      (default: dev,adapter)
#
set -Eeuo pipefail

# Run the daily suite in its own session/process group (a genuinely separate
# process from the invoking SSH TTY / shared shell), so a dropped SSH channel or
# a shared-shell signal cannot disturb it. Re-exec at most once, and only when
# `setsid --wait` is available (util-linux) so the exit code is still preserved.
if [[ "${NPA_DAILY_DETACHED:-0}" != "1" && "${NPA_DAILY_DETACH:-1}" == "1" ]] \
   && command -v setsid >/dev/null 2>&1 && setsid --help 2>&1 | grep -q -- '--wait'; then
  export NPA_DAILY_DETACHED=1
  exec setsid --wait "$0" "$@"
fi

TEST_TIER="${1:-${NPA_DAILY_TEST_TIER:-unit}}"
GIT_REF="${2:-${NPA_DAILY_GIT_REF:-main}}"

CI_REPO_DIR="${NPA_CI_REPO_DIR:-${HOME}/npa-ci-daily}"
SHARED_REPO_DIR="${NPA_REPO:-${HOME}/nebius-physical-ai}"
PIP_EXTRAS="${NPA_DAILY_PIP_EXTRAS:-dev,adapter}"

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${NPA_DAILY_LOG_DIR:-${HOME}/npa-daily-test-logs}"
LOG_FILE="${LOG_DIR}/daily-${TEST_TIER}-${RUN_STAMP}.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

# Logs go to stderr so they never pollute `$(...)` command substitutions (both
# stdout and stderr are still tee'd to the log file via the exec redirect above).
log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }

redact_url() {
  # Strip any embedded credentials (user:token@ or x-access-token:...@) so a
  # tokenized https remote never lands in the log.
  sed -E 's#://[^/@]+@#://***@#' <<< "$1"
}

resolve_repo_url() {
  if [[ -n "${NPA_CI_REPO_URL:-}" ]]; then
    printf '%s\n' "$NPA_CI_REPO_URL"
    return 0
  fi
  local dir
  for dir in "$SHARED_REPO_DIR" "$CI_REPO_DIR"; do
    if [[ -d "${dir}/.git" ]]; then
      if git -C "$dir" remote get-url origin >/dev/null 2>&1; then
        git -C "$dir" remote get-url origin
        return 0
      fi
    fi
  done
  return 1
}

sync_checkout() {
  local url
  if [[ ! -d "${CI_REPO_DIR}/.git" ]]; then
    url="$(resolve_repo_url)" || die "cannot resolve a git remote; set NPA_CI_REPO_URL"
    log "Cloning $(redact_url "$url") -> ${CI_REPO_DIR}"
    git clone "$url" "$CI_REPO_DIR"
  fi

  log "Fetching origin in ${CI_REPO_DIR}"
  git -C "$CI_REPO_DIR" fetch --prune --tags origin

  log "Checking out ${GIT_REF}"
  if git -C "$CI_REPO_DIR" rev-parse --verify --quiet "origin/${GIT_REF}" >/dev/null; then
    # Branch: pin the local branch to the fetched tip.
    git -C "$CI_REPO_DIR" checkout -f -B "$GIT_REF" "origin/${GIT_REF}"
  else
    # Tag or explicit sha.
    git -C "$CI_REPO_DIR" checkout -f "$GIT_REF"
  fi

  if [[ "${NPA_DAILY_FRESH:-1}" == "1" ]]; then
    # Pristine tree for the day's run: discard any tracked drift and remove
    # untracked build artifacts, but keep the venv so we don't reinstall deps
    # from scratch every day.
    log "Fresh mode: hard-reset + clean (keeping .venv)"
    git -C "$CI_REPO_DIR" reset --hard HEAD
    git -C "$CI_REPO_DIR" clean -ffdx -e .venv
  fi
  log "HEAD is now $(git -C "$CI_REPO_DIR" rev-parse HEAD)"
}

# Sets the global PY to the venv interpreter (an absolute path, required by the
# Makefile which `cd`s into npa/ before invoking $(PYTHON)).
PY=""
ensure_venv() {
  local venv="${CI_REPO_DIR}/.venv"
  if [[ ! -x "${venv}/bin/python" ]]; then
    log "Creating venv at ${venv}"
    python3 -m venv "$venv"
  fi
  "${venv}/bin/python" -m pip install -q --upgrade pip
  log "Installing npa[${PIP_EXTRAS}] (editable)"
  "${venv}/bin/python" -m pip install -q -e "${CI_REPO_DIR}/npa[${PIP_EXTRAS}]"
  PY="${venv}/bin/python"
}

run_unit() {
  local py="$1"
  log "Tier=unit: fast unit suite (no live infra) + ruff lint"
  make -C "$CI_REPO_DIR" test PYTHON="$py"
  # Mirror the CI lint gate scope (.github/workflows/lint.yml: `ruff check
  # src tests`) rather than `make lint`, which scans the whole tree including
  # scripts/, examples/, and docker/ that CI does not gate.
  (cd "${CI_REPO_DIR}/npa" && "$py" -m ruff check src tests)
}

run_e2e() {
  local py="$1"
  run_unit "$py"
  log "Tier=e2e: real-infra S3 e2e suite (self-cleaning, budget-bounded)"
  (
    cd "${CI_REPO_DIR}/npa"
    NPA_INTEGRATION_E2E=1 "$py" -m pytest tests/e2e -m e2e --tb=short
  )
}

run_workflow_coverage_gate() {
  local py="$1"
  local helper="${CI_REPO_DIR}/npa/scripts/daily_workflow_e2e.py"
  log "e2e-daily [1/5]: >= 4-step workflow image-coverage report + regression gate"
  "$py" "$helper" report || true
  log "e2e-daily [1/5]: today's rotating comprehensive plan-set:"
  "$py" "$helper" plan-set | while read -r spec; do log "  plan-set: ${spec}"; done
  "$py" "$helper" check
}

run_agent_eval_gate() {
  local py="$1"
  log "e2e-daily [agent-eval]: defend the committed zero-token scorecard"
  (
    cd "${CI_REPO_DIR}/npa"
    NPA_AGENT_CHAT_LIVE=0 "$py" -m pytest \
      tests/agent_eval/test_agent_eval_scorecard.py \
      tests/cli/test_agent_backend_render.py \
      tests/guardrails/test_agent_no_hardcoded_data.py \
      -q
  )
}

run_workflow_plan_smoke() {
  local py="$1"
  log "e2e-daily [2/5]: validate + plan every npa.workflow spec (all >= 4-step workflows, no GPU)"
  (
    cd "${CI_REPO_DIR}/npa"
    "$py" -m pytest \
      tests/smoke/test_all_workflow_yamls.py \
      tests/smoke/test_npa_workflow_smoke.py \
      tests/orchestration/npa_workflow/test_skypilot_render.py \
      -q --timeout=300
  )
}

run_image_reachability() {
  local py="$1"
  log "e2e-daily [3/5]: resolve + inspect every workbench image in the registry"
  # Resolves all CONTAINER_IMAGE_NAMES to pinned refs and inspects registry
  # presence (needs crane/skopeo/docker on the dev VM; degrades to 'unknown'
  # otherwise). Report-only by default so an intentionally unpushed image does
  # not fail the daily run; set NPA_DAILY_REQUIRE_IMAGES=1 to enforce presence.
  local require_flag=()
  if [[ "${NPA_DAILY_REQUIRE_IMAGES:-0}" == "1" ]]; then
    require_flag=(--require)
  fi
  "$py" "${CI_REPO_DIR}/npa/scripts/daily_workflow_e2e.py" images --inspect "${require_flag[@]}"
}

run_e2e_shard() {
  local py="$1"
  local shards day shard
  shards="${NPA_DAILY_E2E_SHARDS:-7}"
  day="$(date -u +%j)"
  day=$((10#$day))
  shard=$(( day % shards ))
  log "e2e-daily [4/5]: rotating S3 e2e shard ${shard} of ${shards} (day-of-year ${day})"
  (
    cd "${CI_REPO_DIR}/npa"
    local nodes=() collect_rc tmpf
    tmpf="$(mktemp)"
    # `-o addopts=` clears the repo's default `-v` so --collect-only -q emits
    # flat `path::node` ids (not the verbose tree) for deterministic slicing.
    set +e
    NPA_INTEGRATION_E2E=1 "$py" -m pytest tests/e2e -m e2e --collect-only -q -o addopts='' > "$tmpf" 2>&1
    collect_rc=$?
    set -e
    # pytest exit codes: 0 = collected, 5 = no tests collected. Anything else is
    # a real collection error — fail loudly instead of silently skipping.
    if [[ "$collect_rc" -ne 0 && "$collect_rc" -ne 5 ]]; then
      log "e2e collection FAILED (pytest exit ${collect_rc}); not silently skipping:"
      tail -n 20 "$tmpf" >&2
      rm -f "$tmpf"
      return "$collect_rc"
    fi
    mapfile -t nodes < <(grep -E '::' "$tmpf" | sort -u)
    rm -f "$tmpf"
    if [[ "${#nodes[@]}" -eq 0 ]]; then
      log "no S3 e2e tests collected (pytest exit ${collect_rc}); nothing to shard"
      return 0
    fi
    local selected=() i
    for i in "${!nodes[@]}"; do
      if (( i % shards == shard )); then
        selected+=("${nodes[$i]}")
      fi
    done
    log "today's shard selects ${#selected[@]} of ${#nodes[@]} S3 e2e tests"
    if [[ "${#selected[@]}" -eq 0 ]]; then
      log "shard is empty today; nothing to run"
      return 0
    fi
    NPA_INTEGRATION_E2E=1 "$py" -m pytest "${selected[@]}" -m e2e --tb=short
  )
}

run_gpu_daily() {
  local py="$1"
  local day case_line spec driver
  day="$(date -u +%j)"
  day=$((10#$day))
  if [[ "${NPA_DAILY_AGENT_GPU_E2E:-0}" == "1" ]]; then
    spec="vlm-eval-single.yaml"
    driver="agent"
  else
    case_line="$("$py" "${CI_REPO_DIR}/npa/scripts/daily_workflow_e2e.py" gpu-case --day-index "$day" 2>/dev/null)"
    spec="$(printf '%s' "$case_line" | cut -f1 | tr -d '[:space:]')"
    driver="$(printf '%s' "$case_line" | cut -f2 | tr -d '[:space:]')"
  fi
  if [[ -z "$spec" ]]; then
    log "GPU e2e: no real-GPU workflow twin available; skipping"
    return 0
  fi
  log "GPU e2e: today's rotating real-GPU workflow submit = ${spec} (day-of-year ${day}, driver ${driver:-one-shot})"
  # Registry + accelerator remap (e.g. H100:1=RTXPRO6000:1) and SkyPilot creds
  # live in the operator's env files on the dev VM.
  # npa-cloud-env.sh FIRST: it ends by unsetting AWS_ACCESS_KEY_ID /
  # AWS_SECRET_ACCESS_KEY / AWS_ENDPOINT_URL so workbench VM deploys do not
  # inherit region-specific S3 globals. Sourcing it after live-e2e.env wiped the
  # very credentials the submit test requires, and every GPU twin then SKIPPED
  # with "AWS_ACCESS_KEY_ID required for live submit" instead of running.
  # shellcheck source=/dev/null
  [[ -f "${HOME}/bin/npa-cloud-env.sh" ]] && . "${HOME}/bin/npa-cloud-env.sh"
  set -a
  # shellcheck source=/dev/null
  [[ -f "${HOME}/.npa/live-e2e.env" ]] && . "${HOME}/.npa/live-e2e.env"
  set +a
  export NPA_SKYPILOT_BIN="${NPA_SKYPILOT_BIN:-${HOME}/.npa/skypilot-venv/bin/sky}"
  # A twin with a parallel group or a decision-driven loop is only collected by
  # the runtime test; running the one-shot test for it collects nothing and
  # pytest exits 5.
  local node='tests/e2e/test_npa_workflow_submit_live_e2e.py::test_npa_workflow_submit_live_reaches_terminal'
  local runtime_flag=0
  if [[ "$driver" == "agent" ]]; then
    node='tests/e2e/test_agent_gpu_workflow_live_e2e.py::test_agent_confirmation_to_real_gpu_artifact_and_grounded_answer'
  elif [[ "$driver" == "runtime" ]]; then
    node='tests/e2e/test_npa_workflow_submit_live_e2e.py::test_npa_workflow_runtime_live_reaches_terminal'
    runtime_flag=1
  fi
  # One managed job, self-cleaning, cancel-on-timeout so no GPU leaks unattended.
  (
    cd "${CI_REPO_DIR}/npa"
    NPA_INTEGRATION_E2E=1 \
    NPA_E2E_NPA_WORKFLOW_SUBMIT=1 \
    NPA_E2E_NPA_WORKFLOW_RUNTIME="$runtime_flag" \
    NPA_AGENT_GPU_LIVE="${NPA_DAILY_AGENT_GPU_E2E:-0}" \
    NPA_AGENT_LIVE="${NPA_DAILY_AGENT_GPU_E2E:-0}" \
    NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS="gpu,multi" \
    NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS="$spec" \
    NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS="${NPA_DAILY_GPU_MAX_WAIT_SECONDS:-2400}" \
    NPA_E2E_NPA_WORKFLOW_SUBMIT_POLL_SECONDS="${NPA_DAILY_GPU_POLL_SECONDS:-30}" \
    NPA_E2E_NPA_WORKFLOW_SUBMIT_CANCEL_ON_TIMEOUT=1 \
      "$py" -m pytest "$node" \
        -o addopts= -q -s --timeout="${NPA_DAILY_GPU_PYTEST_TIMEOUT:-2600}"
  )
}

run_pr218_safe_contracts() {
  local py="$1"
  log "e2e-daily [PR218-safe]: image, queue, identity, CLI, and teardown contracts"
  (
    cd "${CI_REPO_DIR}/npa"
    "$py" -m pytest \
      tests/guardrails/test_internal_cli_entrypoint.py \
      tests/orchestration/skypilot/test_image_bootstrap_contract.py \
      tests/orchestration/skypilot/test_cleanup.py \
      tests/unit/test_cleanup_identity.py \
      tests/unit/test_controller_preflight_total.py \
      tests/unit/test_pr218_integrated_lifecycle.py \
      -o addopts= -q --tb=short
  )
}

run_mutation_live() {
  local py="$1"
  [[ -n "${NPA_E2E_PROJECT:-}" ]] || die "mutation-live requires NPA_E2E_PROJECT"
  [[ -n "${NPA_E2E_CLUSTER_CONTEXT:-}" ]] || die "mutation-live requires NPA_E2E_CLUSTER_CONTEXT"
  [[ -n "${NPA_E2E_AGENT_NAME:-}" ]] || die "mutation-live requires NPA_E2E_AGENT_NAME"
  [[ -n "${NPA_E2E_CONTROLLER_TRANSACTION_RUN_ID:-}" ]] \
    || die "mutation-live requires NPA_E2E_CONTROLLER_TRANSACTION_RUN_ID"
  log "Tier=mutation-live: explicitly authorized lifecycle/controller/agent mutation"
  # shellcheck source=/dev/null
  [[ -f "${HOME}/bin/npa-cloud-env.sh" ]] && . "${HOME}/bin/npa-cloud-env.sh"
  set -a
  # shellcheck source=/dev/null
  [[ -f "${HOME}/.npa/live-e2e.env" ]] && . "${HOME}/.npa/live-e2e.env"
  set +a
  export NPA_SKYPILOT_BIN="${NPA_SKYPILOT_BIN:-${HOME}/.npa/skypilot-venv/bin/sky}"
  (
    cd "${CI_REPO_DIR}/npa"
    NPA_INTEGRATION_E2E=1 \
    NPA_PR218_LIVE_LIFECYCLE=1 \
    NPA_LIVE_CONTROLLER_LAUNCH_TRANSACTION=1 \
      "$py" -m pytest \
        tests/e2e/test_pr218_lifecycle_live.py \
        tests/e2e/test_controller_launch_transaction_live.py \
        -o addopts= -q -s --tb=short
  )
}

run_e2e_daily() {
  local py="$1"
  log "Tier=e2e-daily: comprehensive >= 4-step workflow coverage + all-image check + rotating S3 e2e subset"
  run_agent_eval_gate "$py"
  run_workflow_coverage_gate "$py"
  run_workflow_plan_smoke "$py"
  run_image_reachability "$py"
  run_pr218_safe_contracts "$py"
  run_e2e_shard "$py"
  # Bounded real-GPU e2e is opt-in on the schedule: one rotating managed-job
  # workflow submit per day when the operator sets NPA_DAILY_ENABLE_GPU=1. The
  # full `gpu and e2e` sky-cluster suite stays on the manual `live-gpu` tier.
  if [[ "${NPA_DAILY_ENABLE_GPU:-0}" == "1" ]]; then
    log "e2e-daily [5/5]: bounded rotating real-GPU workflow submit (NPA_DAILY_ENABLE_GPU=1)"
    run_gpu_daily "$py"
  else
    log "e2e-daily [5/5]: GPU e2e disabled (set NPA_DAILY_ENABLE_GPU=1 to run one rotating real-GPU workflow submit/day)"
  fi
}

run_e2e_serverless() {
  local py="$1"
  [[ -n "${NPA_E2E_SERVERLESS_PROJECT:-}" ]] \
    || die "e2e-serverless tier requires NPA_E2E_SERVERLESS_PROJECT (a sandbox project id)"
  log "Tier=e2e-serverless: serverless endpoint/job e2e against project ${NPA_E2E_SERVERLESS_PROJECT}"
  (
    cd "${CI_REPO_DIR}/npa"
    NPA_INTEGRATION_E2E=1 "$py" -m pytest tests/e2e -m e2e_serverless --tb=short
  )
}

run_live_gpu() {
  [[ "${NPA_DAILY_ALLOW_LIVE_GPU:-0}" == "1" ]] \
    || die "live-gpu tier requires NPA_DAILY_ALLOW_LIVE_GPU=1; it must never run on a schedule (docs/testing/live-e2e.md)"
  log "Tier=live-gpu: delegating to scripts/live-e2e.sh (its own teardown guarantees apply)"
  NPA_LIVE_E2E_REPO_ROOT="$CI_REPO_DIR" bash "${CI_REPO_DIR}/scripts/live-e2e.sh"
}

main() {
  log "Starting daily dev-VM tests: tier=${TEST_TIER} ref=${GIT_REF}"
  log "CI checkout: ${CI_REPO_DIR}"
  log "Log file: ${LOG_FILE}"

  # Validate inputs before any expensive clone/venv work.
  case "$TEST_TIER" in
    unit | e2e | e2e-daily | gpu-daily | e2e-serverless | mutation-live | live-gpu) ;;
    *) die "unknown tier '${TEST_TIER}' (expected: unit | e2e | e2e-daily | gpu-daily | e2e-serverless | mutation-live | live-gpu)" ;;
  esac
  if [[ "$TEST_TIER" == "live-gpu" && "${NPA_DAILY_ALLOW_LIVE_GPU:-0}" != "1" ]]; then
    die "live-gpu tier requires NPA_DAILY_ALLOW_LIVE_GPU=1; it must never run on a schedule (docs/testing/live-e2e.md)"
  fi

  command -v git >/dev/null 2>&1 || die "git not found on dev VM"
  command -v python3 >/dev/null 2>&1 || die "python3 not found on dev VM"
  command -v make >/dev/null 2>&1 || die "make not found on dev VM"

  sync_checkout
  ensure_venv

  # Put the venv on PATH (equivalent to activating it) so subprocesses spawned by
  # the suite that call bare `python3`/`npa`/`ruff` resolve THIS run's install,
  # not a global one that may be absent or stale.
  export VIRTUAL_ENV="${CI_REPO_DIR}/.venv"
  export PATH="${VIRTUAL_ENV}/bin:${PATH}"
  log "PATH primed with ${VIRTUAL_ENV}/bin"

  case "$TEST_TIER" in
    unit) run_unit "$PY" ;;
    e2e) run_e2e "$PY" ;;
    e2e-daily) run_e2e_daily "$PY" ;;
    gpu-daily) run_gpu_daily "$PY" ;;
    e2e-serverless) run_e2e_serverless "$PY" ;;
    mutation-live) run_mutation_live "$PY" ;;
    live-gpu) run_live_gpu ;;
    *) die "unknown tier '${TEST_TIER}' (expected: unit | e2e | e2e-daily | gpu-daily | e2e-serverless | mutation-live | live-gpu)" ;;
  esac

  log "Daily dev-VM tests completed: tier=${TEST_TIER} ref=${GIT_REF}"
}

main "$@"
