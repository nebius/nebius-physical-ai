#!/bin/bash
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-/opt/lerobot}"
VENV="$DEPLOY_ROOT/venv/bin/activate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-$DEPLOY_ROOT/benchmarks}"
METRICS_HELPER="${METRICS_HELPER:-$DEPLOY_ROOT/benchmark_metrics.py}"
if [ ! -f "$METRICS_HELPER" ]; then
    METRICS_HELPER="$SCRIPT_DIR/benchmark_metrics.py"
fi
DATASET="${DATASET:-HuggingFaceVLA/libero}"
ENV_TYPE="${ENV_TYPE:-libero}"
ENV_TASK="${ENV_TASK:-libero_10}"

PROFILE_POLICIES="${PROFILE_POLICIES:-act,diffusion,smolvla}"
SCALING_POLICIES="${SCALING_POLICIES:-act}"
MEMORY_POLICIES="${MEMORY_POLICIES:-smolvla}"

PROFILE_GPU_COUNT="${PROFILE_GPU_COUNT:-1}"
SCALING_GPU_COUNTS="${SCALING_GPU_COUNTS:-1,2,4,8}"
MEMORY_GPU_COUNT="${MEMORY_GPU_COUNT:-1}"
MEMORY_BATCH_SIZES="${MEMORY_BATCH_SIZES:-4,8,16,32,64}"

TRAIN_STEPS="${TRAIN_STEPS:-200}"
SCALING_STEPS="${SCALING_STEPS:-200}"
MEMORY_STEPS="${MEMORY_STEPS:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1.0}"
WARMUP_TRIM_SECONDS="${WARMUP_TRIM_SECONDS:-20}"
STOP_MEMORY_ON_OOM="${STOP_MEMORY_ON_OOM:-1}"

RUN_PROFILE="${RUN_PROFILE:-1}"
RUN_SCALING="${RUN_SCALING:-1}"
RUN_MEMORY="${RUN_MEMORY:-1}"
SCALING_IMPORT_RUN_SUMMARIES="${SCALING_IMPORT_RUN_SUMMARIES:-}"
MEMORY_IMPORT_RUN_SUMMARIES="${MEMORY_IMPORT_RUN_SUMMARIES:-}"

ACT_BATCH_SIZE="${ACT_BATCH_SIZE:-8}"
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-8}"
SMOLVLA_BATCH_SIZE="${SMOLVLA_BATCH_SIZE:-4}"

ACT_EXTRA_ARGS="${ACT_EXTRA_ARGS:-}"
DIFFUSION_EXTRA_ARGS="${DIFFUSION_EXTRA_ARGS:-}"
SMOLVLA_EXTRA_ARGS="${SMOLVLA_EXTRA_ARGS:-}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

CURRENT_SAMPLER_PID=""
LAST_SUMMARY_PATH=""
LAST_STATUS=""
LAST_FAILURE_KIND=""
LAST_EXIT_CODE=0

cleanup() {
    if [ -n "${CURRENT_SAMPLER_PID:-}" ]; then
        kill "$CURRENT_SAMPLER_PID" 2>/dev/null || true
        wait "$CURRENT_SAMPLER_PID" 2>/dev/null || true
        CURRENT_SAMPLER_PID=""
    fi
}

trap cleanup EXIT INT TERM

if [ -f "$VENV" ]; then
    source "$VENV"
fi
if [ -f "$DEPLOY_ROOT/.env" ]; then
    set -a
    source "$DEPLOY_ROOT/.env"
    set +a
fi

log() {
    echo ""
    echo "==== $* ===="
}

now_epoch() {
    python3 -c 'import time; print(time.time())'
}

elapsed_seconds() {
    python3 -c 'import sys; print(round(max(0.0, float(sys.argv[2]) - float(sys.argv[1])), 6))' "$1" "$2"
}

count_gpus() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo 0
        return
    fi
    nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' '
}

csv_to_array() {
    local input="$1"
    local -n output_ref="$2"
    IFS=',' read -r -a output_ref <<< "$input"
}

append_csv_to_array() {
    local input="$1"
    local -n output_ref="$2"
    local raw_values=()
    local value

    if [ -z "$input" ]; then
        return
    fi

    IFS=',' read -r -a raw_values <<< "$input"
    for value in "${raw_values[@]}"; do
        value="${value//[[:space:]]/}"
        [ -n "$value" ] || continue
        output_ref+=("$value")
    done
}

policy_display_name() {
    case "$1" in
        act) echo "ACT" ;;
        diffusion) echo "Diffusion Policy" ;;
        smolvla) echo "SmolVLA" ;;
        *) echo "$1" ;;
    esac
}

policy_batch_size() {
    case "$1" in
        act) echo "$ACT_BATCH_SIZE" ;;
        diffusion) echo "$DIFFUSION_BATCH_SIZE" ;;
        smolvla) echo "$SMOLVLA_BATCH_SIZE" ;;
        *) echo 8 ;;
    esac
}

policy_extra_args() {
    local policy="$1"
    local -n output_ref="$2"
    local raw=""
    case "$policy" in
        act) raw="$ACT_EXTRA_ARGS" ;;
        diffusion) raw="$DIFFUSION_EXTRA_ARGS" ;;
        smolvla) raw="$SMOLVLA_EXTRA_ARGS" ;;
    esac
    if [ -n "$raw" ]; then
        read -r -a output_ref <<< "$raw"
    else
        output_ref=()
    fi
}

classify_failure() {
    local exit_code="$1"
    local log_file="$2"
    if grep -Eiq 'CUDA out of memory|out of memory|CUBLAS_STATUS_ALLOC_FAILED|OOM|killed process' "$log_file"; then
        echo "oom"
    elif [ "$exit_code" -eq 137 ] || [ "$exit_code" -eq 9 ]; then
        echo "oom"
    else
        echo "runtime_error"
    fi
}

run_phase() {
    local phase="$1"
    local policy="$2"
    local batch_size="$3"
    local gpu_count="$4"
    local work_units="$5"
    local work_count="$6"
    local run_root="$7"
    shift 7

    mkdir -p "$run_root"

    local log_file="$run_root/${phase}.log"
    local sample_file="$run_root/${phase}_samples.csv"
    local summary_file="$run_root/${phase}_summary.json"
    local torch_memory_dir="$run_root/${phase}_torch_memory"
    local status="success"
    local failure_kind="none"

    rm -rf "$torch_memory_dir"
    mkdir -p "$torch_memory_dir"

    python3 "$METRICS_HELPER" sample --output "$sample_file" --interval "$SAMPLE_INTERVAL" &
    CURRENT_SAMPLER_PID=$!

    local start_ts
    local end_ts
    local duration

    start_ts="$(now_epoch)"
    set +e
    PYTHONPATH="$SCRIPT_DIR/benchmark_python${PYTHONPATH:+:$PYTHONPATH}" \
    LEROBOT_BENCHMARK_TORCH_MEMORY_DIR="$torch_memory_dir" \
    "$@" > >(tee "$log_file") 2>&1
    LAST_EXIT_CODE=$?
    set -e
    end_ts="$(now_epoch)"

    cleanup

    duration="$(elapsed_seconds "$start_ts" "$end_ts")"

    if [ "$LAST_EXIT_CODE" -ne 0 ]; then
        status="failure"
        failure_kind="$(classify_failure "$LAST_EXIT_CODE" "$log_file")"
    fi

    python3 "$METRICS_HELPER" summarize-run \
        --samples "$sample_file" \
        --phase "$phase" \
        --policy "$policy" \
        --gpu-count "$gpu_count" \
        --batch-size "$batch_size" \
        --status "$status" \
        --failure-kind "$failure_kind" \
        --run-seconds "$duration" \
        --work-units "$work_units" \
        --work-count "$work_count" \
        --warmup-trim-seconds "$WARMUP_TRIM_SECONDS" \
        --torch-memory-dir "$torch_memory_dir" \
        --output "$summary_file"

    LAST_SUMMARY_PATH="$summary_file"
    LAST_STATUS="$status"
    LAST_FAILURE_KIND="$failure_kind"

    return "$LAST_EXIT_CODE"
}

train_phase() {
    local policy="$1"
    local batch_size="$2"
    local gpu_count="$3"
    local steps="$4"
    local run_root="$5"
    shift 5
    local extra_args=("$@")
    local output_dir="$run_root/train_output"
    local cmd=()

    rm -rf "$output_dir"

    if [ "$gpu_count" -gt 1 ]; then
        cmd=(accelerate launch --multi_gpu --num_processes="$gpu_count" "$(which lerobot-train)")
    else
        cmd=(lerobot-train)
    fi

    cmd+=(
        --policy.type="$policy"
        --policy.push_to_hub=false
        --dataset.repo_id="$DATASET"
        --env.type="$ENV_TYPE"
        --env.task="$ENV_TASK"
        --output_dir="$output_dir"
        --steps="$steps"
        --save_freq="$steps"
        --eval_freq=1000000
        --policy.device=cuda
        --batch_size="$batch_size"
    )
    cmd+=("${extra_args[@]}")

    run_phase train "$policy" "$batch_size" "$gpu_count" steps "$steps" "$run_root" "${cmd[@]}"
}

eval_phase() {
    local policy="$1"
    local batch_size="$2"
    local checkpoint_dir="$3"
    local run_root="$4"
    local cmd=()

    rm -rf "$run_root/eval_output"

    cmd=(
        lerobot-eval
        --policy.path="$checkpoint_dir"
        --env.type="$ENV_TYPE"
        --env.task="$ENV_TASK"
        --eval.n_episodes="$EVAL_EPISODES"
        --eval.batch_size="$EVAL_BATCH_SIZE"
        --output_dir="$run_root/eval_output"
    )

    run_phase eval "$policy" "$batch_size" 1 episodes "$EVAL_EPISODES" "$run_root" "${cmd[@]}"
}

write_skip_summary() {
    local output_path="$1"
    local policy="$2"
    local reason="$3"
    python3 -c 'import json, pathlib, sys; pathlib.Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(sys.argv[1]).write_text(json.dumps({"policy": sys.argv[2], "status": "skipped", "reason": sys.argv[3]}, indent=2) + "\n", encoding="utf-8")' "$output_path" "$policy" "$reason"
}

benchmark_profile_policy() {
    local policy="$1"
    local batch_size
    local extra_args=()
    local run_root="$SUITE_DIR/profile/$policy"
    local checkpoint_dir
    local train_summary
    local eval_summary

    batch_size="$(policy_batch_size "$policy")"
    policy_extra_args "$policy" extra_args

    log "Profile benchmark: $(policy_display_name "$policy") on ${PROFILE_GPU_COUNT} GPU(s)"

    if [ "$PROFILE_GPU_COUNT" -gt "$AVAILABLE_GPUS" ]; then
        write_skip_summary "$run_root/profile_summary.json" "$policy" "Requested PROFILE_GPU_COUNT=$PROFILE_GPU_COUNT but only $AVAILABLE_GPUS GPU(s) are available."
        echo "Skipping profile benchmark for $policy"
        return
    fi

    rm -rf "$run_root"
    mkdir -p "$run_root"

    if ! train_phase "$policy" "$batch_size" "$PROFILE_GPU_COUNT" "$TRAIN_STEPS" "$run_root" "${extra_args[@]}"; then
        echo "Training failed for $policy; see $LAST_SUMMARY_PATH"
        return
    fi
    train_summary="$LAST_SUMMARY_PATH"
    checkpoint_dir="$run_root/train_output/checkpoints/last/pretrained_model"

    if [ ! -d "$checkpoint_dir" ]; then
        echo "Missing checkpoint for $policy at $checkpoint_dir"
        return
    fi

    if ! eval_phase "$policy" "$batch_size" "$checkpoint_dir" "$run_root"; then
        echo "Evaluation failed for $policy; see $LAST_SUMMARY_PATH"
        return
    fi
    eval_summary="$LAST_SUMMARY_PATH"

    python3 "$METRICS_HELPER" combine-profile \
        --train-summary "$train_summary" \
        --eval-summary "$eval_summary" \
        --output "$run_root/profile_summary.json"
}

benchmark_scaling_policy() {
    local policy="$1"
    local batch_size
    local extra_args=()
    local run_summaries=()
    local requested_gpu_counts=()
    local raw_gpu_count
    local gpu_count
    local run_root
    local summary_root="$SUITE_DIR/scaling/$policy"

    batch_size="$(policy_batch_size "$policy")"
    policy_extra_args "$policy" extra_args
    csv_to_array "$SCALING_GPU_COUNTS" requested_gpu_counts

    log "Scaling benchmark: $(policy_display_name "$policy")"

    if [ "$AVAILABLE_GPUS" -lt 2 ] && [ -z "$SCALING_IMPORT_RUN_SUMMARIES" ]; then
        write_skip_summary "$summary_root/scaling_summary.json" "$policy" "Scaling efficiency requires at least 2 GPUs; detected $AVAILABLE_GPUS."
        echo "Skipping scaling benchmark for $policy"
        return
    fi

    rm -rf "$summary_root"
    mkdir -p "$summary_root"

    for raw_gpu_count in "${requested_gpu_counts[@]}"; do
        gpu_count="${raw_gpu_count//[[:space:]]/}"
        if [ -z "$gpu_count" ] || [ "$gpu_count" -gt "$AVAILABLE_GPUS" ]; then
            continue
        fi

        run_root="$summary_root/gpu${gpu_count}"
        mkdir -p "$run_root"

        if train_phase "$policy" "$batch_size" "$gpu_count" "$SCALING_STEPS" "$run_root" "${extra_args[@]}"; then
            run_summaries+=("$LAST_SUMMARY_PATH")
        else
            run_summaries+=("$LAST_SUMMARY_PATH")
        fi
    done

    append_csv_to_array "$SCALING_IMPORT_RUN_SUMMARIES" run_summaries

    if [ "${#run_summaries[@]}" -eq 0 ]; then
        write_skip_summary "$summary_root/scaling_summary.json" "$policy" "No requested GPU counts were runnable on this host, and no imported scaling summaries were provided."
        return
    fi

    python3 "$METRICS_HELPER" summarize-scaling \
        --run-summaries "${run_summaries[@]}" \
        --output "$summary_root/scaling_summary.json" \
        --csv-output "$summary_root/scaling_summary.csv"
}

benchmark_memory_policy() {
    local policy="$1"
    local extra_args=()
    local requested_batch_sizes=()
    local raw_batch_size
    local batch_size
    local run_root
    local summary_root="$SUITE_DIR/memory/$policy"
    local run_summaries=()

    policy_extra_args "$policy" extra_args
    csv_to_array "$MEMORY_BATCH_SIZES" requested_batch_sizes

    log "Memory ceiling benchmark: $(policy_display_name "$policy") on ${MEMORY_GPU_COUNT} GPU(s)"

    if [ "$MEMORY_GPU_COUNT" -gt "$AVAILABLE_GPUS" ] && [ -z "$MEMORY_IMPORT_RUN_SUMMARIES" ]; then
        write_skip_summary "$summary_root/memory_summary.json" "$policy" "Requested MEMORY_GPU_COUNT=$MEMORY_GPU_COUNT but only $AVAILABLE_GPUS GPU(s) are available."
        echo "Skipping memory benchmark for $policy"
        return
    fi

    rm -rf "$summary_root"
    mkdir -p "$summary_root"

    for raw_batch_size in "${requested_batch_sizes[@]}"; do
        batch_size="${raw_batch_size//[[:space:]]/}"
        if [ -z "$batch_size" ]; then
            continue
        fi

        run_root="$summary_root/batch${batch_size}"
        mkdir -p "$run_root"

        if train_phase "$policy" "$batch_size" "$MEMORY_GPU_COUNT" "$MEMORY_STEPS" "$run_root" "${extra_args[@]}"; then
            run_summaries+=("$LAST_SUMMARY_PATH")
            continue
        fi

        run_summaries+=("$LAST_SUMMARY_PATH")
        if [ "$LAST_FAILURE_KIND" = "oom" ] && [ "$STOP_MEMORY_ON_OOM" = "1" ]; then
            break
        fi
    done

    append_csv_to_array "$MEMORY_IMPORT_RUN_SUMMARIES" run_summaries

    if [ "${#run_summaries[@]}" -eq 0 ]; then
        write_skip_summary "$summary_root/memory_summary.json" "$policy" "No memory runs were executed locally, and no imported memory summaries were provided."
        return
    fi

    python3 "$METRICS_HELPER" summarize-memory \
        --run-summaries "${run_summaries[@]}" \
        --output "$summary_root/memory_summary.json" \
        --csv-output "$summary_root/memory_summary.csv"
}

AVAILABLE_GPUS="$(count_gpus)"
SUITE_DIR="$OUTPUT_BASE/benchmark-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$SUITE_DIR"

log "Benchmark setup"
python3 -c "import lerobot, torch; print(f'LeRobot {lerobot.__version__}'); print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import libero; print('libero import OK')"
echo "Available GPUs: $AVAILABLE_GPUS"
echo "Benchmark output: $SUITE_DIR"

profile_policies=()
scaling_policies=()
memory_policies=()

csv_to_array "$PROFILE_POLICIES" profile_policies
csv_to_array "$SCALING_POLICIES" scaling_policies
csv_to_array "$MEMORY_POLICIES" memory_policies

if [ "$RUN_PROFILE" = "1" ]; then
    for policy in "${profile_policies[@]}"; do
        policy="${policy//[[:space:]]/}"
        [ -n "$policy" ] || continue
        benchmark_profile_policy "$policy"
    done
fi

if [ "$RUN_SCALING" = "1" ]; then
    for policy in "${scaling_policies[@]}"; do
        policy="${policy//[[:space:]]/}"
        [ -n "$policy" ] || continue
        benchmark_scaling_policy "$policy"
    done
fi

if [ "$RUN_MEMORY" = "1" ]; then
    for policy in "${memory_policies[@]}"; do
        policy="${policy//[[:space:]]/}"
        [ -n "$policy" ] || continue
        benchmark_memory_policy "$policy"
    done
fi

log "Benchmark complete"
echo "Artifacts written to: $SUITE_DIR"
echo "Profile summaries:"
find "$SUITE_DIR/profile" -name 'profile_summary.json' -print 2>/dev/null | sort || true
echo "Scaling summaries:"
find "$SUITE_DIR/scaling" -name 'scaling_summary.json' -print 2>/dev/null | sort || true
echo "Memory summaries:"
find "$SUITE_DIR/memory" -name 'memory_summary.json' -print 2>/dev/null | sort || true
