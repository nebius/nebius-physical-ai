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
DATASET="${DATASET:-HuggingFaceVLA/libero}"
ENV_TYPE="${ENV_TYPE:-libero}"
ENV_TASK="${ENV_TASK:-libero_10}"
DATASET_VIDEO_BACKEND="${DATASET_VIDEO_BACKEND:-pyav}"
STEPS="${STEPS:-1000}"
SAVE_FREQ="${SAVE_FREQ:-$STEPS}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
ACCELERATE_STEPS="${ACCELERATE_STEPS:-100}"   # short run just to prove the wrapper works
TRAIN_EVAL_BATCH_SIZE="${TRAIN_EVAL_BATCH_SIZE:-1}"
TRAIN_EVAL_EPISODES="${TRAIN_EVAL_EPISODES:-1}"
MIN_EVAL_PC_SUCCESS="${MIN_EVAL_PC_SUCCESS:-}"
MIN_EVAL_AVG_SUM_REWARD="${MIN_EVAL_AVG_SUM_REWARD:-}"

# Activate venv if available
if [ -f "$VENV" ]; then source "$VENV"; fi
if [ -f "$DEPLOY_ROOT/.env" ]; then set -a; source "$DEPLOY_ROOT/.env"; set +a; fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

PASS=0
FAIL=0
RESULTS=""

log()  { echo ""; echo "==== $* ===="; }
pass() { PASS=$((PASS + 1)); RESULTS="$RESULTS\n  PASS  $*"; echo "  PASS: $*"; }
fail() { FAIL=$((FAIL + 1)); RESULTS="$RESULTS\n  FAIL  $*"; echo "  FAIL: $*"; }

check_eval_behavior() {
    local name="$1"
    local eval_info_path="$2"

    python3 - "$name" "$eval_info_path" "$MIN_EVAL_PC_SUCCESS" "$MIN_EVAL_AVG_SUM_REWARD" <<'PY'
import json
import math
import pathlib
import sys


def parse_optional_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


name = sys.argv[1]
path = pathlib.Path(sys.argv[2])
min_pc_success = parse_optional_float(sys.argv[3])
min_avg_sum_reward = parse_optional_float(sys.argv[4])

if not path.is_file():
    print(f"{name} eval metrics missing: {path}", file=sys.stderr)
    raise SystemExit(1)

payload = json.loads(path.read_text(encoding="utf-8"))
overall = payload.get("overall")
if not isinstance(overall, dict):
    print(f"{name} eval metrics malformed: missing overall block in {path}", file=sys.stderr)
    raise SystemExit(1)

pc_success = overall.get("pc_success")
avg_sum_reward = overall.get("avg_sum_reward")
avg_max_reward = overall.get("avg_max_reward")
n_episodes = overall.get("n_episodes")

print(
    f"  INFO: {name} eval metrics: "
    f"pc_success={pc_success} avg_sum_reward={avg_sum_reward} "
    f"avg_max_reward={avg_max_reward} n_episodes={n_episodes}"
)

pc_success_value = float(pc_success) if pc_success is not None else math.nan
avg_sum_reward_value = float(avg_sum_reward) if avg_sum_reward is not None else math.nan

if min_pc_success is not None and (math.isnan(pc_success_value) or pc_success_value < min_pc_success):
    print(
        f"{name} eval pc_success {pc_success} is below MIN_EVAL_PC_SUCCESS={min_pc_success}",
        file=sys.stderr,
    )
    raise SystemExit(1)

if min_avg_sum_reward is not None and (math.isnan(avg_sum_reward_value) or avg_sum_reward_value < min_avg_sum_reward):
    print(
        f"{name} eval avg_sum_reward {avg_sum_reward} is below MIN_EVAL_AVG_SUM_REWARD={min_avg_sum_reward}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

run_privileged() {
    if [ "${EUID:-$(id -u)}" -eq 0 ]; then
        "$@"
    else
        sudo -n "$@"
    fi
}

ensure_env_var() {
    local name="$1"
    local value="$2"
    local file="$3"

    [ -f "$file" ] || return 0

    run_privileged env name="$name" value="$value" file="$file" sh -c '
        if grep -q "^${name}=" "$file"; then
            sed -i "s|^${name}=.*|${name}=${value}|" "$file"
        else
            printf "%s=%s\n" "$name" "$value" >> "$file"
        fi
    '
}

nvidia_egl_ready() {
    ldconfig -p 2>/dev/null | grep -q 'libEGL_nvidia\.so\.0' && \
        grep -Rqs 'libEGL_nvidia.so.0' /usr/share/glvnd/egl_vendor.d 2>/dev/null
}

ensure_nvidia_egl() {
    local driver_info
    local driver_pkg
    local driver_branch
    local driver_version

    if nvidia_egl_ready; then
        return 0
    fi

    driver_info="$(
        dpkg-query -W -f='${Package} ${Version}\n' 'libnvidia-cfg1-*' 2>/dev/null \
            | sed -n 's/^\(libnvidia-cfg1-[0-9][0-9]*\) \(.*\)$/\1 \2/p' \
            | head -n 1
    )"

    if [ -z "$driver_info" ]; then
        echo "No libnvidia-cfg1-* package is installed; cannot derive EGL package branch" >&2
        return 1
    fi

    driver_pkg="${driver_info%% *}"
    driver_version="${driver_info#* }"
    driver_branch="${driver_pkg##libnvidia-cfg1-}"

    log "Installing missing NVIDIA EGL userspace"
    run_privileged apt-get update
    run_privileged env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        "libnvidia-common-${driver_branch}=${driver_version}" \
        "libnvidia-gl-${driver_branch}=${driver_version}" \
        libnvidia-egl-gbm1 \
        libnvidia-egl-wayland1 \
        libnvidia-egl-xcb1 \
        libnvidia-egl-xlib1
    run_privileged ldconfig

    if ! run_privileged test -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json && \
       ldconfig -p 2>/dev/null | grep -q 'libEGL_nvidia.so.0'
    then
        run_privileged install -d -m 0755 /usr/share/glvnd/egl_vendor.d
        printf '%s\n' \
            '{' \
            '    "file_format_version" : "1.0.0",' \
            '    "ICD" : {' \
            '        "library_path" : "libEGL_nvidia.so.0"' \
            '    }' \
            '}' \
            | run_privileged tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null
    fi

    nvidia_egl_ready || return 1

    ensure_env_var MUJOCO_GL egl "$DEPLOY_ROOT/.env"
    ensure_env_var PYOPENGL_PLATFORM egl "$DEPLOY_ROOT/.env"
}

# ── 0. Prerequisites ─────────────────────────────────────────────────────
log "Checking prerequisites"

python3 -c "import lerobot; print(f'LeRobot {lerobot.__version__}')"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Ensure policy extras needed by validation are installed (no-op if already present)
pip install --quiet "lerobot[libero]" num2words 2>/dev/null || true

if ensure_nvidia_egl; then
    pass "NVIDIA EGL userspace ready"
else
    fail "NVIDIA EGL userspace missing or could not be repaired"
fi

python3 -c "import num2words" && pass "num2words package importable" || fail "num2words package import"
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
        if check_eval_behavior "$name" "$out_dir/eval/eval_info.json"; then
            pass "$name eval"
        else
            fail "$name eval behavior"
        fi
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
