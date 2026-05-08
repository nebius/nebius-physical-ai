#!/bin/bash
set -euo pipefail

# ── LeRobot B300 vs H200 Benchmark Runner ─────────────────────────────────
#
# Deploys two non-preemptible GPU VMs and runs a controlled benchmark matrix:
#   3 policies × 2 num_workers settings = 6 training runs per node
#
# Prerequisites:
#   - npa CLI installed (pip install -e ./npa)
#   - nebius CLI installed and authenticated (nebius config init)
#   - SSH key at ~/.ssh/id_ed25519
#
# Usage:
#   bash npa/benchmark_b300_h200.sh          # deploy + benchmark
#   bash npa/benchmark_b300_h200.sh bench    # benchmark only (VMs already deployed)
#   bash npa/benchmark_b300_h200.sh destroy  # tear down both VMs

MODE="${1:-full}"
: "${NPA_PROJECT_ID:?NPA_PROJECT_ID must be set}"
: "${NPA_S3_BUCKET:?NPA_S3_BUCKET must be set}"

# ── Node definitions ──────────────────────────────────────────────────────

B300_PROJECT="${NPA_B300_PROJECT_ID:-$NPA_PROJECT_ID}"
B300_TENANT="${NPA_B300_TENANT_ID:-${NPA_TENANT_ID:-${NEBIUS_ACCOUNT_ID:-}}}"
B300_REGION="uk-south1"
B300_GPU_TYPE="gpu-b300-sxm"
B300_GPU_PRESET="1gpu-24vcpu-346gb"
B300_PROJ_ALIAS="uk-south1"
B300_WB_NAME="b300"

H200_PROJECT="${NPA_H200_PROJECT_ID:-$NPA_PROJECT_ID}"
H200_TENANT="${NPA_H200_TENANT_ID:-${NPA_TENANT_ID:-${NEBIUS_ACCOUNT_ID:-}}}"
H200_REGION="eu-north1"
H200_GPU_TYPE="gpu-h200-sxm"
H200_GPU_PRESET="1gpu-16vcpu-200gb"
H200_PROJ_ALIAS="eu-north1"
H200_WB_NAME="h200"

# ── Benchmark parameters ─────────────────────────────────────────────────

BATCH_SIZE=8
BENCHMARK_ARGS=(
  --run "act:lerobot/pusht:100"
  --run "diffusion:lerobot/pusht:100"
  --run "smolvla:lerobot/aloha_sim_insertion_human:100"
  --num-workers 8
  --num-workers 0
  --batch-size "$BATCH_SIZE"
)

# ── Helper ────────────────────────────────────────────────────────────────

banner() {
  echo ""
  echo "========================================================================"
  echo "  $*"
  echo "========================================================================"
  echo ""
}

# ── Destroy ───────────────────────────────────────────────────────────────

if [ "$MODE" = "destroy" ]; then
  banner "Destroying B300 (${B300_PROJ_ALIAS}/${B300_WB_NAME})"
  npa workbench lerobot -p "$B300_PROJ_ALIAS" -n "$B300_WB_NAME" deploy \
    --project-id "$B300_PROJECT" --tenant-id "$B300_TENANT" --region "$B300_REGION" \
    --gpu-type "$B300_GPU_TYPE" --gpu-preset "$B300_GPU_PRESET" \
    --destroy || echo "B300 destroy returned non-zero (may already be destroyed)"

  banner "Destroying H200 (${H200_PROJ_ALIAS}/${H200_WB_NAME})"
  npa workbench lerobot -p "$H200_PROJ_ALIAS" -n "$H200_WB_NAME" deploy \
    --project-id "$H200_PROJECT" --tenant-id "$H200_TENANT" --region "$H200_REGION" \
    --gpu-type "$H200_GPU_TYPE" --gpu-preset "$H200_GPU_PRESET" \
    --destroy || echo "H200 destroy returned non-zero (may already be destroyed)"

  echo "Done."
  exit 0
fi

# ── Deploy ────────────────────────────────────────────────────────────────

if [ "$MODE" = "full" ]; then
  banner "Deploying B300 — ${B300_GPU_TYPE} in ${B300_REGION} (non-preemptible)"
  npa workbench lerobot -p "$B300_PROJ_ALIAS" -n "$B300_WB_NAME" deploy \
    --project-id "$B300_PROJECT" \
    --tenant-id "$B300_TENANT" \
    --region "$B300_REGION" \
    --gpu-type "$B300_GPU_TYPE" \
    --gpu-preset "$B300_GPU_PRESET" \
    --no-preemptible \
    --default

  banner "Deploying H200 — ${H200_GPU_TYPE} in ${H200_REGION} (non-preemptible)"
  npa workbench lerobot -p "$H200_PROJ_ALIAS" -n "$H200_WB_NAME" deploy \
    --project-id "$H200_PROJECT" \
    --tenant-id "$H200_TENANT" \
    --region "$H200_REGION" \
    --gpu-type "$H200_GPU_TYPE" \
    --gpu-preset "$H200_GPU_PRESET" \
    --no-preemptible
fi

# ── Benchmark ─────────────────────────────────────────────────────────────

banner "System info — B300"
npa workbench lerobot -p "$B300_PROJ_ALIAS" -n "$B300_WB_NAME" system-info

banner "System info — H200"
npa workbench lerobot -p "$H200_PROJ_ALIAS" -n "$H200_WB_NAME" system-info

FAILURES=0

banner "Benchmark — B300 (${B300_GPU_TYPE}, ${B300_GPU_PRESET})"
if ! npa workbench lerobot -p "$B300_PROJ_ALIAS" -n "$B300_WB_NAME" benchmark \
  "${BENCHMARK_ARGS[@]}"; then
  echo "WARNING: B300 benchmark exited with failures"
  FAILURES=$((FAILURES + 1))
fi

banner "Benchmark — H200 (${H200_GPU_TYPE}, ${H200_GPU_PRESET})"
if ! npa workbench lerobot -p "$H200_PROJ_ALIAS" -n "$H200_WB_NAME" benchmark \
  "${BENCHMARK_ARGS[@]}"; then
  echo "WARNING: H200 benchmark exited with failures"
  FAILURES=$((FAILURES + 1))
fi

banner "All benchmarks finished"
echo "Check S3 bucket ${NPA_S3_BUCKET} for results under benchmarks/b300/ and benchmarks/h200/"
echo ""
echo "To tear down both VMs:"
echo "  bash npa/benchmark_b300_h200.sh destroy"

if [ "$FAILURES" -gt 0 ]; then
  echo ""
  echo "ERROR: $FAILURES benchmark(s) had failures — check output above."
  exit 1
fi
