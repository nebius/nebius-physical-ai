#!/bin/bash
# End-to-end demo: train → eval → serve → infer
# Reads all config from ~/.npa/config.yaml. No inline IPs, paths, or credentials.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMINGS=()
DATASET="lerobot/aloha_sim_transfer_cube_human"
JOB_NAME="demo-act-$(date +%Y%m%d-%H%M%S)"
ENV_TYPE="aloha"
ENV_TASK="AlohaTransferCube-v0"
TRAIN_STEPS=5000

record_time() {
    local label="$1"
    local start="$2"
    local end
    end="$(date +%s)"
    local duration=$((end - start))
    TIMINGS+=("$label: ${duration}s")
    echo ""
    echo "  [$label] completed in ${duration}s"
    echo ""
}

echo "============================================================"
echo "  NPA LeRobot End-to-End Demo"
echo "============================================================"
echo ""
echo "  Dataset:    $DATASET"
echo "  Job name:   $JOB_NAME"
echo "  Env:        $ENV_TYPE / $ENV_TASK"
echo "  Steps:      $TRAIN_STEPS"
echo ""

# ── 1. Status check ──────────────────────────────────────────────────────
echo "── Step 1: Checking VM status ──"
START="$(date +%s)"
npa workbench lerobot status
record_time "status" "$START"

# ── 2. Train ──────────────────────────────────────────────────────────────
echo "── Step 2: Training ACT policy ──"
START="$(date +%s)"
npa workbench lerobot train \
    --policy-type act \
    --dataset "$DATASET" \
    --job-name "$JOB_NAME" \
    --steps "$TRAIN_STEPS" \
    --env-type "$ENV_TYPE" \
    --env-task "$ENV_TASK" \
    --batch-size 8
record_time "train" "$START"

# ── 3. List checkpoints ──────────────────────────────────────────────────
echo "── Step 3: Listing checkpoints ──"
START="$(date +%s)"
npa workbench lerobot list-checkpoints
record_time "list-checkpoints" "$START"

# ── 4. Eval ───────────────────────────────────────────────────────────────
echo "── Step 4: Evaluating trained checkpoint ──"
CHECKPOINT="/opt/lerobot/checkpoints/$JOB_NAME/checkpoints/last/pretrained_model"
START="$(date +%s)"
npa workbench lerobot eval \
    --checkpoint "$CHECKPOINT" \
    --env "$ENV_TYPE" \
    --env-task "$ENV_TASK" \
    --episodes 10 \
    --output json
record_time "eval" "$START"

# ── 5. Serve ──────────────────────────────────────────────────────────────
echo "── Step 5: Starting PolicyServer ──"
START="$(date +%s)"
npa workbench lerobot serve --checkpoint "$CHECKPOINT"
record_time "serve" "$START"

# ── 6. Generate observation ───────────────────────────────────────────────
echo "── Step 6: Generating sample observation ──"
START="$(date +%s)"
python "$SCRIPT_DIR/generate_observation.py" \
    --dataset "$DATASET" \
    --output /tmp/demo_obs.json
record_time "generate-observation" "$START"

# ── 7. Infer ──────────────────────────────────────────────────────────────
echo "── Step 7: Running inference ──"
START="$(date +%s)"
npa workbench lerobot infer \
    --observation /tmp/demo_obs.json \
    --output json
record_time "infer" "$START"

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Demo Summary"
echo "============================================================"
for t in "${TIMINGS[@]}"; do
    echo "  $t"
done
echo "============================================================"
