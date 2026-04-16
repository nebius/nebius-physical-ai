#!/bin/bash
# Validate that LeRobot can train and evaluate ACT, Diffusion Policy, and
# SmolVLA on LIBERO-10.  Run on the GPU VM after cloud-init finishes.
#
# Usage:  bash /opt/lerobot/validate_policies.sh
#         bash training/validate_policies.sh        # from repo root locally
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────
DEPLOY_ROOT="${DEPLOY_ROOT:-/opt/lerobot}"
VENV="$DEPLOY_ROOT/venv/bin/activate"
OUTPUT_BASE="$DEPLOY_ROOT/validation"
DATASET="HuggingFaceVLA/libero"
ENV_TYPE="libero"
ENV_TASK="libero_10"
DATASET_VIDEO_BACKEND="${DATASET_VIDEO_BACKEND:-pyav}"
STEPS=1000
SAVE_FREQ=1000
EVAL_EPISODES=10
ACCELERATE_STEPS=100   # short run just to prove the wrapper works
TRAIN_EVAL_BATCH_SIZE=1
TRAIN_EVAL_EPISODES=1

export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Activate venv if available
if [ -f "$VENV" ]; then source "$VENV"; fi
if [ -f "$DEPLOY_ROOT/.env" ]; then set -a; source "$DEPLOY_ROOT/.env"; set +a; fi

PASS=0
FAIL=0
RESULTS=""

log()  { echo ""; echo "==== $* ===="; }
pass() { PASS=$((PASS + 1)); RESULTS="$RESULTS\n  PASS  $*"; echo "  PASS: $*"; }
fail() { FAIL=$((FAIL + 1)); RESULTS="$RESULTS\n  FAIL  $*"; echo "  FAIL: $*"; }

# ── 0. Prerequisites ─────────────────────────────────────────────────────
log "Checking prerequisites"

python3 -c "import lerobot; print(f'LeRobot {lerobot.__version__}')"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Ensure libero extra is installed (no-op if already present)
pip install --quiet "lerobot[libero]" 2>/dev/null || true

python3 -c "import libero" && pass "libero package importable" || fail "libero package import"

# ── 1. Verify LIBERO dataset loads from Hub ───────────────────────────────
log "Downloading / verifying LIBERO-10 dataset"

python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('$DATASET')
print(f'Dataset: {ds.repo_id}  episodes: {ds.num_episodes}  frames: {ds.num_frames}')
" && pass "LIBERO dataset loads" || fail "LIBERO dataset load"

# ── Helper: train + eval one policy ──────────────────────────────────────
train_and_eval() {
    local name="$1"    # human-readable
    local policy="$2"  # --policy.type value
    shift 2
    local extra_args=("$@")

    local out_dir="$OUTPUT_BASE/$policy"
    local ckpt_dir="$out_dir/checkpoints/last/pretrained_model"

    # ── Train ─────────────────────────────────────────────────────────
    log "Training $name ($STEPS steps)"
    rm -rf "$out_dir"

    if lerobot-train \
        --policy.type="$policy" \
        --policy.push_to_hub=false \
        --dataset.repo_id="$DATASET" \
        --dataset.video_backend="$DATASET_VIDEO_BACKEND" \
        --env.type="$ENV_TYPE" \
        --env.task="$ENV_TASK" \
        --output_dir="$out_dir" \
        --steps="$STEPS" \
        --save_freq="$SAVE_FREQ" \
        --eval_freq=10000 \
        --eval.batch_size="$TRAIN_EVAL_BATCH_SIZE" \
        --eval.n_episodes="$TRAIN_EVAL_EPISODES" \
        --policy.device=cuda \
        "${extra_args[@]}"
    then
        pass "$name training"
    else
        fail "$name training"
        return 1
    fi

    # ── Eval ──────────────────────────────────────────────────────────
    log "Evaluating $name (n_episodes=$EVAL_EPISODES)"

    if lerobot-eval \
        --policy.path="$ckpt_dir" \
        --env.type="$ENV_TYPE" \
        --env.task="$ENV_TASK" \
        --eval.n_episodes="$EVAL_EPISODES" \
        --eval.batch_size=1 \
        --output_dir="$out_dir/eval"
    then
        pass "$name eval"
    else
        fail "$name eval"
    fi

    # ── accelerate launch ─────────────────────────────────────────────
    log "Testing accelerate launch for $name ($ACCELERATE_STEPS steps)"

    local accel_dir="$OUTPUT_BASE/${policy}_accelerate"
    rm -rf "$accel_dir"

    if accelerate launch --num_processes=1 "$(which lerobot-train)" \
        --policy.type="$policy" \
        --policy.push_to_hub=false \
        --dataset.repo_id="$DATASET" \
        --dataset.video_backend="$DATASET_VIDEO_BACKEND" \
        --env.type="$ENV_TYPE" \
        --env.task="$ENV_TASK" \
        --output_dir="$accel_dir" \
        --steps="$ACCELERATE_STEPS" \
        --save_freq="$ACCELERATE_STEPS" \
        --eval_freq=10000 \
        --eval.batch_size="$TRAIN_EVAL_BATCH_SIZE" \
        --eval.n_episodes="$TRAIN_EVAL_EPISODES" \
        --policy.device=cuda \
        "${extra_args[@]}"
    then
        pass "$name accelerate launch"
    else
        fail "$name accelerate launch"
    fi
}

run_policy_check() {
    if ! train_and_eval "$@"; then
        # train_and_eval already recorded the failure; keep going so the
        # remaining policies run and the final summary prints.
        :
    fi
}

# ── 2. ACT ────────────────────────────────────────────────────────────────
run_policy_check "ACT" "act" --batch_size=8

# ── 3. Diffusion Policy ──────────────────────────────────────────────────
run_policy_check "Diffusion Policy" "diffusion" --batch_size=8

# ── 4. SmolVLA ────────────────────────────────────────────────────────────
run_policy_check "SmolVLA" "smolvla" --batch_size=4

# ── Summary ───────────────────────────────────────────────────────────────
log "Validation Summary"
echo -e "$RESULTS"
echo ""
echo "Total: $PASS passed, $FAIL failed"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "VALIDATION FAILED"
    exit 1
else
    echo "ALL CHECKS PASSED"
fi
