# Training Scripts

This directory contains the local source-of-truth for the LeRobot wrapper
scripts used by this repo.

Files:
- `train.sh` — wrapper around `lerobot-train`
- `eval.sh` — wrapper around `lerobot-eval`
- `validate_policies.sh` — end-to-end validation on LIBERO-10
- `benchmark_policies.sh` — performance benchmark suite
- `benchmark_metrics.py` — raw metric sampler and JSON summarizer
- `benchmark_python/sitecustomize.py` — records per-process torch CUDA peak memory during benchmarks
- `configs/` — example config overrides

## Local Vs VM Paths

On your Mac, the scripts live here:

```bash
/Users/timothyle/repos/lerobot-deploy/training
```

On the deployed VM, cloud-init writes runnable copies here:

```bash
/opt/lerobot/train.sh
/opt/lerobot/eval.sh
/opt/lerobot/validate_policies.sh
/opt/lerobot/benchmark_policies.sh
/opt/lerobot/benchmark_metrics.py
/opt/lerobot/benchmark_python/sitecustomize.py
```

Use the `/opt/lerobot/...` paths on the VM unless you explicitly copied this
repo onto the instance.

## SSH To The VM

Get the current IP from Terraform:

```bash
cd /Users/timothyle/repos/lerobot-deploy
terraform -chdir=terraform output -raw ssh_command
```

Or run the command directly if you already know the IP:

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@<VM_IP>
```

## Recommended VM Checks

Run these first after login:

```bash
python3 -c "import lerobot, torch; print('LeRobot', lerobot.__version__); print('CUDA', torch.cuda.is_available())"
python /opt/lerobot/s3_sync.py check
```

## Quick Smoke Test

This is the fastest useful training check:

```bash
bash /opt/lerobot/train.sh --policy.type=act --dataset.repo_id=lerobot/pusht --steps=10 --save_freq=10
```

The wrapper now adds `--policy.push_to_hub=false` by default so smoke tests do
not require a Hugging Face model repo.

Then evaluate the latest checkpoint:

```bash
RUN_DIR="$(ls -td /opt/lerobot/runs/run-* | head -1)"
bash /opt/lerobot/eval.sh --policy.path="$RUN_DIR/checkpoints/last/pretrained_model"
```

## Full Validation

This is the heavy validation pass:

```bash
bash /opt/lerobot/validate_policies.sh
```

What it checks:
- `lerobot` import
- `torch` import and CUDA visibility
- NVIDIA EGL userspace presence and automatic repair if the userspace is missing
- `libero` import
- LIBERO-10 dataset load
- ACT train/eval
- Diffusion Policy train/eval
- SmolVLA train/eval
- `eval_info.json` behavioral metrics from each eval run
- `accelerate launch` path

What it does not check:
- sustained GPU utilization
- multi-GPU scaling efficiency
- batch-size memory ceilings
- CPU / input-pipeline bottlenecks
- train-vs-eval wall-clock split

Validation logs `pc_success`, `avg_sum_reward`, `avg_max_reward`, and
`n_episodes` from `eval_info.json` after each eval run. To enforce a minimum
behavioral bar, set `MIN_EVAL_PC_SUCCESS` and/or `MIN_EVAL_AVG_SUM_REWARD`
before running `validate_policies.sh`.

## Performance Benchmarks

Use this when you want performance findings instead of pass/fail validation:

```bash
bash /opt/lerobot/benchmark_policies.sh
```

Default benchmark coverage:
- ACT, Diffusion Policy, and SmolVLA profile runs for GPU utilization, bottleneck heuristics, and train-vs-eval time split
- ACT scaling runs across `1,2,4,8` GPUs when that hardware is available
- SmolVLA memory-ceiling runs across batch sizes `4,8,16,32,64`

Artifacts are written under:

```bash
/opt/lerobot/benchmarks/benchmark-YYYYMMDD-HHMMSS/
```

Each benchmark run produces:
- raw per-sample metrics CSV files
- per-phase JSON summaries for train and eval
- profile summaries with train/eval split plus eval behavior from `eval_info.json`
- scaling and memory summary JSON/CSV tables

The suite is designed to answer:
- `Utilization`: whether a policy is leaving the GPU underused, and on which GPU model
- `Scaling efficiency`: whether more GPUs produce meaningful speedup, including transition points where scaling flattens
- `Memory ceiling`: the highest feasible batch size before OOM, with both `nvidia-smi` and torch peak-memory views
- `Bottleneck identification`: whether the run looks GPU-bound, CPU/decode-bound, I/O-bound, or memory-bound, and whether that bottleneck shifts as GPU count grows
- `Train vs eval split`: how much workflow time is spent in evaluation versus training

Notes:
- The default Terraform deployment in this repo is still a single-GPU VM. Scaling tests are skipped automatically unless the host has at least 2 GPUs.
- Bottleneck classification is heuristic. It uses sampled GPU utilization, GPU memory pressure, host CPU utilization, and host iowait.
- Per-run summaries trim an initial warmup window before classifying bottlenecks. Override that with `WARMUP_TRIM_SECONDS`.
- Profile summaries now record GPU model names, GPU utilization percentiles, and utilization variance.
- Scaling summaries emit `headline_findings` plus per-step `transitions`, so they can surface diminishing returns and bottleneck shifts instead of only comparing the smallest and largest GPU counts.
- Memory summaries record `peak_nvidia_smi_memory_mb`, `peak_torch_memory_allocated_mb`, and `peak_torch_memory_reserved_mb`. If you combine runs from multiple machines, the summary groups them by GPU model and GPU count.
- OOM classification treats both explicit PyTorch OOM strings and hard exits like `137` as memory-limit failures.
- Cross-machine comparisons can be folded into a local run with `SCALING_IMPORT_RUN_SUMMARIES` or `MEMORY_IMPORT_RUN_SUMMARIES`, each a comma-separated list of existing per-run summary JSON paths.

Useful overrides:

```bash
# Run only profile benchmarks
RUN_SCALING=0 RUN_MEMORY=0 bash /opt/lerobot/benchmark_policies.sh

# Restrict scaling to 1 and 2 GPUs
SCALING_GPU_COUNTS=1,2 bash /opt/lerobot/benchmark_policies.sh

# Sweep a different memory boundary
MEMORY_BATCH_SIZES=8,16,24,32 bash /opt/lerobot/benchmark_policies.sh

# Narrow the policy set
PROFILE_POLICIES=diffusion RUN_SCALING=0 RUN_MEMORY=0 bash /opt/lerobot/benchmark_policies.sh

# Trim less warmup from short runs
WARMUP_TRIM_SECONDS=10 bash /opt/lerobot/benchmark_policies.sh
```

Policy-specific overrides:
- `ACT_BATCH_SIZE`, `DIFFUSION_BATCH_SIZE`, `SMOLVLA_BATCH_SIZE`
- `ACT_EXTRA_ARGS`, `DIFFUSION_EXTRA_ARGS`, `SMOLVLA_EXTRA_ARGS`

## Train On The VM

Example training run:

```bash
bash /opt/lerobot/train.sh --policy.type=act --dataset.repo_id=lerobot/pusht
```

Another example using a local config pattern:

```bash
bash /opt/lerobot/train.sh \
  --policy.type=act \
  --dataset.repo_id=lerobot/pusht \
  --batch_size=8
```

Notes:
- The wrapper creates a unique run directory under `/opt/lerobot/runs/`.
- On exit, the wrapper tries to upload the latest checkpoint to S3.
- Do not pre-create the output directory yourself.

Resume after preemption or interruption:

```bash
RESUME=true bash /opt/lerobot/train.sh --policy.type=act --dataset.repo_id=lerobot/pusht
```

This finds the most recent run directory and passes `--resume=true` to
`lerobot-train`, which restores the optimizer state and training step from the
last saved checkpoint.  Without `RESUME=true`, a fresh run directory is always
created.  If the default Terraform deployment uses preemptible instances
(`enable_preemptible = true`), plan for resume when the VM restarts.

## Evaluate On The VM

Find the latest run:

```bash
ls -td /opt/lerobot/runs/run-* | head
```

Evaluate the latest checkpoint:

```bash
RUN_DIR="$(ls -td /opt/lerobot/runs/run-* | head -1)"
bash /opt/lerobot/eval.sh --policy.path="$RUN_DIR/checkpoints/last/pretrained_model"
```

## If You Copy This Folder To The VM

If you upload and unzip this `training/` directory onto the VM, you can also run:

```bash
cd ~/training
bash validate_policies.sh
bash benchmark_policies.sh
bash train.sh --policy.type=act --dataset.repo_id=lerobot/pusht
```

The scripts default `DEPLOY_ROOT` to `/opt/lerobot`, so they still use the VM's
installed environment unless you override that variable manually. NVIDIA EGL
repair now lives in cloud-init bootstrap plus the validation preflight rather
than a separate recovery script.
