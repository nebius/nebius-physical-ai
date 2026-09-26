# Cosmos 3 Nano WAM post-training on Nebius B200s

[Workbench docs](../README.md) · [Recipe and parameter reference](../../../npa/workflows/workbench/cosmos3-wam-slurm/README.md) · [Blog draft](../cosmos3-wam-scaling-blog.md)

This recipe makes NVIDIA's LIBERO-10 action-policy post-training experiment
repeatable on a reserved B200 Slurm cluster. The experiment predicts future
visual observations and action chunks from language and camera observations.
It uses the actual `action_policy_libero_nano` experiment and its `mode="wam"`
dataloader. Captioned-video generator SFT does not exercise this action pathway.

**Current evidence:** actual reserved B200 CUDA attention forward/backward and
native fused Adam checks, completed native checkpoint conversion, native trainer
dryruns for 8/16/32 GPUs, and decoded public data. A separate one-GPU WAM attempt
failed at its first optimizer update with FP32 parameters and EMA, even at one
sample per step. The eight-GPU Slurm run completed all 2,000 updates in
7 h 50 m 33 s, with an eight-rank NCCL check and complete checkpoint hashes.
The [full-run evidence](../evidence/cosmos3-wam-full-8/README.md) records its
conditions and timing scope. Scaling efficiency and full policy success remain
unmeasured. The [three measured timing repetitions](../evidence/cosmos3-wam-timing-8/README.md)
establish the eight-GPU baseline: mean step 13.3080 s, with 0.0136 s sample
standard deviation across the three run means. Their saved reports and all
447 timed iterations reproduce the aggregate.
The [GPU evidence record](../evidence/cosmos3-wam-b200-runtime.json)
includes runtime versions, hashes, and the memory failure.
Consult the [validation record](../../../npa/workflows/workbench/cosmos3-wam-slurm/validation.json).

For a visual look at the prepared setup, [play the actual B200 generation and
source-data preview](../evidence/cosmos3-wam-b200-visual/README.md). That record
contains the generated MP4, synchronized source cameras, reproduction settings
and GPU telemetry. It demonstrates base-model inference; it does not establish
post-training success.

The [trained-policy visual check](../evidence/cosmos3-wam-trained-visual/README.md)
uses the actual update-500 checkpoint and completes one trial on each of ten
tasks without infrastructure errors. It includes successful and unsuccessful
rollouts and prediction-versus-simulator videos. Its four observed successes
are illustrative evidence; full 500-trial quality measurements remain pending.

## What the experiment measures

Keep framework/model/data revisions, optimizer, precision, sample cap, action
semantics, seed and nominal global batch fixed. Change only topology and
gradient accumulation for the scaling comparison:

| Nodes | B200 GPUs | FSDP shard degree | HSDP replicas | Samples/rank cap | Accumulation | Nominal global batch |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 8 | 1 | 64 | 4 | 2,048 |
| 2 | 16 | 8 | 2 | 64 | 2 | 2,048 |
| 4 | 32 | 8 | 4 | 64 | 1 | 2,048 |

These are candidate configurations, not measured capacity requirements. Native
upstream uses 128 samples/rank on two nodes; 64 permits the same nominal global
batch across this entire comparison. The model retains its 74,000-token cap,
selective activation checkpointing, BF16 compute, learning rate 5e-5, 500-step
warmup and 16,000-step scheduler cycle. Native parameter selection trains the
generation and action pathways; it does not indiscriminately update every
parameter of the combined model.

Batch counts are nominal: packing, skipped samples and rank partitioning may
change actual work. Inspect native data/token counters and rank imbalance
alongside timings. If actual work differs materially, report it rather than
calling the result equivalent-work strong scaling. Comparing equal optimizer
steps alone cannot demonstrate equivalent policy quality.

## 1. Prepare a dedicated reserved-capacity cluster

The ongoing campaign uses [native Slurm on dedicated GPU VMs](../../../npa/workflows/workbench/cosmos3-wam-slurm/native-cluster.md),
with controller, accounting and NFS on the first worker. Follow that deployment
guide to reproduce the measured setup. The Soperator commands below are an
alternative deployment path and have not been validated by this campaign.

Use a separate project in the selected tenant and region. Complete credential
and exact model access checks before provisioning:

```bash
npa workbench health preflight --checks nebius,hf
npa workbench health access --capability cosmos3
```

Supply the private project, tenant and reservation IDs through the environment.
Choose an unused cluster name and a private spec location. From the checkout:

```bash
export WAM_RECIPE="$PWD/npa/workflows/workbench/cosmos3-wam-slurm"
export NPA_PYTHON="$PWD/npa/.venv/bin/python"
umask 077
"$NPA_PYTHON" "$WAM_RECIPE/cluster.py" \
  --name "$WAM_CLUSTER_NAME" --nodes 2 --output "$WAM_CLUSTER_SPEC"
npa soperator plan --spec "$WAM_CLUSTER_SPEC"
npa soperator deploy --spec "$WAM_CLUSTER_SPEC" \
  --root-login-ssh-public-key-file "$WAM_LOGIN_PUBLIC_KEY"
npa soperator status --name "$WAM_CLUSTER_NAME"
```

`WAM_CLUSTER_NAME`, `WAM_CLUSTER_SPEC`, and `WAM_LOGIN_PUBLIC_KEY` are required
operator-selected values. The public-key file grants root access to the login
node. This uses the pinned Soperator runtime and its mandatory direct CUDA
creation checks. The cluster requests eight-GPU B200 workers on one InfiniBand
fabric, STRICT reservation binding, accounting, and a 2 TiB shared jail.
Check CPU, SSD and filesystem quota as well as GPU reservation headroom.
Two free individual GPUs are not a two-node allocation: two workers need 16
free reserved B200 GPUs. For the four-node comparison, request four workers and
verify 32-GPU reservation capacity first.

Connect using the provider-verified login endpoint. Run Slurm inside its jail;
the following paths must be shared and identical on every worker, not `/tmp`.
Keep the training runtime, dataset, base DCP and run directories on that
filesystem for the baseline. A later node-local SSD experiment must explicitly
record that storage change and stage identical verified bytes on each host.

## 2. Install the pinned native training environment

On the Linux Slurm filesystem, place the Workbench checkout and initialize the
same `WAM_RECIPE` and `NPA_PYTHON` paths there. Choose shared roots with enough
space for downloads, converted base, optimizer checkpoints and traces:

```bash
export WAM_SHARED_ROOT=/shared/cosmos3-wam
export WAM_RUN_ROOT="$WAM_SHARED_ROOT/runs"
mkdir -p "$WAM_SHARED_ROOT" "$WAM_RUN_ROOT"
chmod 700 "$WAM_SHARED_ROOT" "$WAM_RUN_ROOT"
git init "$WAM_SHARED_ROOT/framework"
git -C "$WAM_SHARED_ROOT/framework" remote add origin https://github.com/NVIDIA/cosmos-framework.git
git -C "$WAM_SHARED_ROOT/framework" fetch --depth=1 origin cf5d68c00d97ccd2480a2320ed652b92dec63102
git -C "$WAM_SHARED_ROOT/framework" checkout --detach FETCH_HEAD
cd "$WAM_SHARED_ROOT/framework"
uv sync --frozen --all-extras --group cu130-train
```

Use NVIDIA's [pinned installation prerequisites](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/docs/setup.md),
including a compatible Linux CUDA toolchain/driver and the required system
libraries. The baked Workbench Cosmos inference environment is insufficient.
The campaign includes the frozen guardrail/inference dependencies required by
the native policy server, in addition to training dependencies.
All workers need the same Linux userland and access to the shared environment.
Preserve the upstream lockfile and capture `uv pip freeze` plus `nvidia-smi`
privately. No new public container or bundled vendor payload is published here.

## 3. Stage and inspect data, weights and tokenizer

```bash
srun --nodes=1 --ntasks=1 --gpus-per-task=8 --cpus-per-task=128 --exclusive \
  "$WAM_SHARED_ROOT/framework/.venv/bin/python" "$WAM_RECIPE/prepare.py" \
  --shared-root "$WAM_SHARED_ROOT"
```

Preparation uses the exact revisions in
[`sources.json`](../../../npa/workflows/workbench/cosmos3-wam-slurm/sources.json),
downloads the LIBERO-10 suite, checks real data, fetches Cosmos3-Nano and the Wan
VAE, and calls the native HF-to-DCP converter once. Keep `prepared.json`,
`conversion.log`, and the data hashes. Failed preparation never writes a
completion receipt. Use a fresh shared root for a new preparation; an existing
conversion log prevents silently overwriting a previous conversion.
Preparation runs on a worker because checkpoint conversion can exceed the
login node's memory. Charge this allocation to preparation, separately from
the training-process timings. The CPU-only `--data-only` check can run on the
login node or a development machine without reserving GPUs.

The converter's supported overrides select the staged processor and VAE;
the unmodified native defaults can otherwise fetch a moving processor revision.
Training's Qwen tokenizer is also downloaded at an immutable revision and passed
as a local path. Both launchers put the native virtualenv on `PATH`, including
the `uv` executable needed by upstream subprocesses.

The pinned data contains **379 episodes, 101,469 frames, and ten tasks at 20 Hz**.
Local inspection checked every action/state row and decoded a frame from each
of four camera video files; this is data evidence, not training evidence.

| Stored data | Native training transformation |
| --- | --- |
| LeRobot v3 `meta/info.json`, task and episode Parquet | Episode/task boundaries and timestamps |
| `data/chunk-*/file-*.parquet` | Raw 7D actions, 8D state, timestamp and indices |
| Third-person and wrist MP4s, each 256×256 | Concatenated 256×512, snapped to 192×320 model canvas |
| Stored per-frame action deltas | 16 actions per window; re-encode axis-angle to rot6d, yielding 10D |
| Bundled LIBERO quantile statistics | `quantile_rot` normalization, native coordinate frame |

The native loader uses a 1% episode validation split with seed 0, and seed 42
for episode shuffling. Actual loading retains 375 episodes and 94,250 valid
training windows. The training experiment disables automatic validation.
The [data-pipeline figure](../evidence/cosmos3-wam-data/README.md) includes a
native decoded sample and the commands to reproduce it. Keep those semantics
paired with the matching simulator/server. Never replace real action labels
with captions or generated videos. Source and payload terms are linked in the
[upstream recipe](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/docs/action_policy_libero_posttrain.md)
and the [existing Workbench policy guide](../cosmos3-policy-model-factory.md#runtime-and-access).
Dataset, model and source downloads remain operator-controlled artifacts.

## 4. Launch and compare

Use the same requested training duration on each topology. The native
LIBERO-10 schedule is 2,000 optimizer steps. Run a full schedule to measure
training time; a shorter explicitly chosen execution check proves only that
execution check. Run the baseline first, then the two-node candidate:

```bash
"$NPA_PYTHON" "$WAM_RECIPE/recipe.py" plan \
  --shared-root "$WAM_SHARED_ROOT" --run-dir "$WAM_RUN_ROOT/b200-8" \
  --name b200-8 --nodes 1 --steps 2000
sbatch --wait "$WAM_RUN_ROOT/b200-8/train.sbatch"
"$NPA_PYTHON" "$WAM_RECIPE/recipe.py" plan \
  --shared-root "$WAM_SHARED_ROOT" --run-dir "$WAM_RUN_ROOT/b200-16" \
  --name b200-16 --nodes 2 --steps 2000
sbatch --wait "$WAM_RUN_ROOT/b200-16/train.sbatch"
"$NPA_PYTHON" "$WAM_RECIPE/report.py" --run-dir "$WAM_RUN_ROOT/b200-8"
"$NPA_PYTHON" "$WAM_RECIPE/report.py" \
  --run-dir "$WAM_RUN_ROOT/b200-16" --baseline "$WAM_RUN_ROOT/b200-8"
```

For 32 GPUs, use a fresh `b200-32` run with `--nodes 4` on a cluster with four
available workers. Repeat comparable measurements with independent run names
to characterize variance; retain all runs, including failures. No seed,
checkpoint cadence, or batch-size change should be hidden inside a comparison.

Each worker verifies eight visible B200s and the pinned clean source. Rank zero
runs upstream `train --dryrun`; all ranks then verify host placement and a real
NCCL collective before training. This collective proves connectivity, not IB
bandwidth. Check NCCL transport diagnostics and the cluster's RDMA/NCCL bandwidth
qualification before interpreting multi-node performance. Do not force
`NCCL_IB_DISABLE=1` to get a passing performance run.

`measurement.json` contains mean/p50/p95 optimizer-step time, process duration,
training-process GPU-hours and checkpoint identity. Optimizer-step timing
excludes the callback warmup; checkpoint stalls remain in the distribution.
Process duration includes model loading and final checkpoint save, but excludes
queue wait, input staging, DCP conversion and the distributed preflight. Use
Slurm accounting for total allocation time and allocated GPU-hours:

```bash
sacct -j "$WAM_JOB_ID" --format=JobID,State,ExitCode,ElapsedRaw,AllocTRES,MaxRSS
```

`WAM_JOB_ID` is the exact job ID from `sbatch`. Preserve setup/queue/training/
checkpoint/evaluation durations separately. Report speedup as `mean_step_8 /
mean_step_N`, and scaling efficiency as `speedup / (N / 8)` only for matched
runs. An estimate `steps × steady_state_seconds` must be labeled a projection,
with startup and checkpoint overhead added explicitly; it is not an observed
time to a useful policy.

The campaign also runs three separate 200-update repetitions per topology,
starting from the same base checkpoint and excluding profiling, evaluation
and archival I/O from their timing windows. The
[repeat-analysis command](../../../npa/workflows/workbench/cosmos3-wam-slurm/README.md#measurements-and-cleanup)
checks the saved reports and numeric series, reports variation across run
means, and compares both step time and actual processed-token throughput.
Its steady-state comparison is separate from the observed 2,000-update
training-process duration.

## 5. Profile separately

Create a fresh run with `--profile` and the desired explicit step count. Native
PyTorch profiling warms up for three iterations and captures two iterations
per 100-step cycle; traces are exported for ranks 0, 8, 16 and 24 as applicable.
This affects runtime, so the reporting tool rejects profiler runs as speedup
inputs. Trace files live under the native job's `torch_trace/iteration_*/`.

Inspect the traces with Perfetto or TensorBoard. Compare VAE encoding, input
preparation, forward/backward kernels, collective communication, host gaps,
allocator peaks and checkpoint writes. Upstream IterSpeed logs already report
VAE/preparation time and rank-average versus rank-maximum timing. These timings
overlap; do not add overlapping fractions into a fictitious 100% breakdown.
Use Nsight Systems only as a separate diagnostic run with the pinned upstream
`trainer.profiling.enable_nsys`/CUDA profiler API mechanism and native rank
launch integration; this recipe's `--profile` selects PyTorch traces only.

The [completed eight-B200 profile](../evidence/cosmos3-wam-profile-8/README.md)
includes an actual trace-derived timeline and reproduction commands. Its
analyzer links CUDA kernels to CPU operators, distinguishes host step markers
from repeated GPU annotations, and measures overlapping kernel intervals.
Use `profile_report.py --run-dir RUN --output-path FRESH_REPORT` to reanalyze
without replacing an archived report. A single-rank kernel trace does not
measure all-GPU utilization or exposed cross-node communication overhead.

## 6. Measure policy quality and time to quality

A low training loss and a saved checkpoint do not establish a useful policy.
Evaluate saved checkpoints with the **same pinned framework's**
`action_policy_server_libero` and `simulation/libero/closed_loop_eval.py`,
following [its exact evaluation instructions](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/docs/action_policy_libero_posttrain.md#3-closed-loop-eval).
Use a separate simulator environment and CPU OSMesa if the worker lacks a
qualified rendering path; the existing [Workbench evaluation implementation](../../../npa/src/npa/workbench/cosmos/policy_runtime.py)
shows that environment separation. The current `policy-eval` CLI pins an older
framework, so do not silently mix it into this recipe's source contract.

Use all ten LIBERO-10 tasks and 50 initial states per task, report per-task and
aggregate success, and bind each summary to the evaluated checkpoint hash.
Keep 20 Hz, both cameras, image orientation, gripper mapping, 10D rot6d action
space and quantile normalization matched. Reject missing/duplicate trials and
simulator errors; they are not legitimate task failures. Compare checkpoints
at the same training steps, and report all evaluated checkpoints, rather than
choosing only the best one after seeing the result.

Choose the target success rate before training; 90% is the existing Workbench
absolute qualification convention, not a result promised by this recipe.
Time to quality is the observed cumulative training duration to the first
checkpoint meeting that predeclared threshold on the full protocol. Report
evaluation compute separately and retain uncertainty across tasks, trials and
training seeds. Partner data requires its own matched success criterion.

## 7. Preserve evidence and stop owned resources

Native logs and configs can contain host names, paths or environment values.
Keep raw evidence private. Scan any proposed public summary before publishing;
the blog needs sanitized measurements and hashes, not live infrastructure IDs.
Move datasets/checkpoints/reports between Workbench tools through private S3
artifacts and verify their hashes after transfer.

```bash
scancel "$WAM_JOB_ID"
squeue -j "$WAM_JOB_ID"
```

Once owned jobs are confirmed absent and checkpoints/evidence are retained,
destroy the exact dedicated cluster:

```bash
npa soperator destroy --name "$WAM_CLUSTER_NAME"
```

Do not cancel unrelated jobs, detach another project's reservation, or delete
shared resources to make room for the benchmark. Keep the dedicated project
and private deployment receipt unless its removal is separately intended.
