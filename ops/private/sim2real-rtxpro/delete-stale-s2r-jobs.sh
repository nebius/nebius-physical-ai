#!/usr/bin/env bash
# Delete completed/failed s2r-* sibling Jobs left behind after sim2real runs.
# Active (Running) jobs and sim2real-* orchestrator jobs are never touched.
set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-$HOME/.npa/clusters/npa-rtxpro-mk8s/kubeconfig}"
CTX="${KUBECONTEXT:-npa-rtxpro-mk8s}"
NS="${KUBENS:-default}"
DRY_RUN=0
KEEP_RUN_IDS=()

usage() {
  cat <<'EOF'
Usage: delete-stale-s2r-jobs.sh [--dry-run] [--keep-run-id RUN_ID]...

Remove s2r-* Jobs that are Complete or Failed. Running/pending jobs are kept.
Pass --keep-run-id to retain siblings for an active orchestrator run.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --keep-run-id)
      KEEP_RUN_IDS+=("$2")
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

slug_run_id() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9-]+/-/g;s/^-+|-+$//g' | cut -c1-63
}

should_keep() {
  local run_label="$1"
  local keep
  for keep in "${KEEP_RUN_IDS[@]}"; do
    [[ "$(slug_run_id "${keep}")" == "${run_label}" ]] && return 0
  done
  return 1
}

mapfile -t rows < <(
  kubectl --context "${CTX}" get jobs -n "${NS}" -o json 2>/dev/null \
    | python3 -c '
import json, sys

data = json.load(sys.stdin)
for item in data.get("items", []):
    name = item["metadata"]["name"]
    if not name.startswith("s2r-"):
        continue
    labels = item.get("metadata", {}).get("labels") or {}
    run_id = labels.get("sim2real.local/run-id", "")
    status = item.get("status") or {}
    active = int(status.get("active") or 0)
    succeeded = int(status.get("succeeded") or 0)
    failed = int(status.get("failed") or 0)
    if active:
        phase = "active"
    elif succeeded:
        phase = "complete"
    elif failed:
        phase = "failed"
    else:
        phase = "pending"
    print(f"{name}\t{run_id}\t{phase}")
'
)

if [[ ${#rows[@]} -eq 0 ]]; then
  echo "No s2r-* jobs in namespace ${NS}."
  exit 0
fi

deleted=0
kept=0
for row in "${rows[@]}"; do
  IFS=$'\t' read -r name run_id phase <<<"${row}"
  if [[ "${phase}" == "active" || "${phase}" == "pending" ]]; then
    echo "KEEP ${name} (${phase}, run-id=${run_id:-unknown})"
    kept=$((kept + 1))
    continue
  fi
  if [[ -n "${run_id}" ]] && should_keep "${run_id}"; then
    echo "KEEP ${name} (--keep-run-id, run-id=${run_id})"
    kept=$((kept + 1))
    continue
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY-RUN delete job/${name} (${phase}, run-id=${run_id:-unknown})"
  else
    echo "DELETE job/${name} (${phase}, run-id=${run_id:-unknown})"
    kubectl --context "${CTX}" delete job -n "${NS}" "${name}" --wait=false
  fi
  deleted=$((deleted + 1))
done

echo "summary kept=${kept} deleted=${deleted} dry_run=${DRY_RUN}"
