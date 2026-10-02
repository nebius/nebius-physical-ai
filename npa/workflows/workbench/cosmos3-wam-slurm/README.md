# Cosmos 3 Nano WAM on reserved B200s with Slurm

[Workbench reference assets](../README.md) · [Full runbook](../../../../docs/workbench/cookbooks/cosmos3-wam-slurm.md) · [Measured results](../../../../docs/workbench/evidence/cosmos3-wam-scaling/README.md)

This native Slurm recipe runs NVIDIA's **LIBERO-10 world action model (WAM)**
post-training experiment. It trains the action and generation pathways together.
It is a standalone Slurm application recipe, alongside the existing
[single-node policy workflow](../../../../docs/workbench/cosmos3-policy-model-factory.md).
It does not add a new `npa.workflow` toolRef or claim that the inference image is
a training image.

The native bootstraps install FFmpeg on every node. Each launcher checks real
TorchCodec decoding of both camera streams before starting distributed model
loading; the shared Python environment alone does not supply host libraries.

**Validation:** experimental; a reserved B200 passed real CUDA forward/backward
and native fused-optimizer checks. Native checkpoint conversion and trainer
configuration checks passed. A one-GPU WAM attempt exhausted memory at its first
optimizer update with FP32 parameters and EMA, even with one sample per step.
Both native Slurm runs completed 2,000 updates: **7 h 50 m 33 s on eight
B200s** and **4 h 23 m 33 s on sixteen**, using **62.74** and **70.28
training-process GPU-hours**. Three matched timing repeats per topology measured
**1.9334× speedup and 96.7% efficiency**; actual token work differed by 0.0108%.
See the [scaling evidence](../../../../docs/workbench/evidence/cosmos3-wam-scaling/README.md)
and [eight-GPU](../../../../docs/workbench/evidence/cosmos3-wam-full-8/README.md)
and [sixteen-GPU full reports](../../../../docs/workbench/evidence/cosmos3-wam-full-16/README.md).

Four 500-trial evaluations per topology completed without infrastructure errors.
The [eight-GPU quality curve](../../../../docs/workbench/evidence/cosmos3-wam-quality-8/README.md)
scored 45.4%, 85.6%, 92.8%, 95.0%; the
[sixteen-GPU curve](../../../../docs/workbench/evidence/cosmos3-wam-quality-16/README.md)
scored 49.0%, 81.4%, 94.2%, 95.8%. The first saved checkpoint above 90% was
update 1,500, ready after about 5 h 53 m and 3 h 19 m respectively. Evaluation
happened afterward; this does not establish exact threshold crossing or a
quality improvement from GPU count with one training seed.

The historical 8/16-GPU deployment used native Slurm 23.11.4 on dedicated
reserved GPU VMs, with controller, accounting and NFS on the first worker.
The four-node extension has
[successful NPA Soperator deployment and all-worker CUDA qualification](../../../../docs/workbench/evidence/cosmos3-wam-soperator-32/README.md).
[Actual 32-rank WAM execution](../../../../docs/workbench/evidence/cosmos3-wam-live-training-32/README.md)
passed NCCL with every GPU process attributed and all four workers sampled.
The [full 32-GPU schedule](../../../../docs/workbench/evidence/cosmos3-wam-full-32/README.md)
completed in **2 h 10 m 47 s**. Three [timing repetitions](../../../../docs/workbench/evidence/cosmos3-wam-timing-32/README.md)
measured **3.6046 ± 0.0026 seconds/update** (mean and sample standard deviation
of three run means). Four [500-trial evaluations](../../../../docs/workbench/evidence/cosmos3-wam-quality-32/README.md)
scored **47.4%, 81.8%, 94.0%, 95.6%**, with zero infrastructure errors.
Update 1,500 was the first saved checkpoint above 90%, ready after
**1 h 39 m 02 s**; the native log bounds this timestamp to a one-second interval.
[Four sampled CUDA ranks](../../../../docs/workbench/evidence/cosmos3-wam-profile-32/README.md),
[all-device full-run telemetry](../../../../docs/workbench/evidence/cosmos3-wam-full-gpu-activity-32/README.md)
and [final checkpoint recordings](../../../../docs/workbench/evidence/cosmos3-wam-final-visual-32/README.md)
provide profiling and visual evidence. This Soperator cohort changes the driver,
scheduler and storage; its timing ratios do not isolate GPU-count scaling.
An earlier [four-GPU execution check](../../../../docs/workbench/evidence/cosmos3-wam-live-training-4/README.md)
was cancelled at observed update 535 after correcting the requested topology to
**four nodes, eight B200s per node, 32 GPUs total**. Its checkpoint 500 is retained;
it is excluded from scaling and quality results. Two GPUs are untested. The
[32-GPU plan](../../../../docs/workbench/evidence/cosmos3-wam-plan-32/README.md)
retains the original capacity rejection. Capacity subsequently became available;
all 32 GPUs were deployed, qualified and used for the completed campaign.
Slurm allocates nodes and records accounting; torchrun launches the ranks.
Slurm is not required by the model, but every measured run used it. A standalone
torchrun-only deployment was not tested.

[Sixteen-rank InfiniBand proof](../../../../docs/workbench/evidence/cosmos3-wam-collective-16/README.md),
[process-attributed live training](../../../../docs/workbench/evidence/cosmos3-wam-live-training-16/README.md),
[252,656 full-run GPU samples](../../../../docs/workbench/evidence/cosmos3-wam-full-gpu-activity-16/README.md)
and [CUDA traces from both hosts](../../../../docs/workbench/evidence/cosmos3-wam-profile-16/README.md)
establish actual execution. Profiled runs are separate from timing repetitions.
The [final sixteen-GPU checkpoint videos](../../../../docs/workbench/evidence/cosmos3-wam-final-visual-16/README.md)
retain a successful and an unsuccessful native rollout, all ten illustrative
outcomes, checkpoint hashes and GPU telemetry. Their nine successes are a
visual execution check; the separate 500-trial evaluation establishes quality.
Historical raw training reports may say quality was unmeasured when they were
written. Their original bytes are retained; the subsequent linked evaluations
provide the final campaign status.

## Plan without cloud resources

From the Workbench checkout, choose absolute paths that will exist on the Slurm
shared filesystem. Planning creates a new directory and never submits a job:

```bash
export WAM_SHARED_ROOT=/shared/cosmos3-wam
export WAM_RUN_ROOT="$WAM_SHARED_ROOT/runs"
npa/.venv/bin/python npa/workflows/workbench/cosmos3-wam-slurm/recipe.py plan \
  --shared-root "$WAM_SHARED_ROOT" --run-dir "$WAM_RUN_ROOT/b200-8" \
  --name b200-8 --nodes 1 --steps 2000
```

The run directory contains `train.toml`, `train.sbatch`, `run.json`, and copies
of the launch/preflight scripts. Submit only after the runbook's reserved-capacity,
runtime, input preparation, and GPU checks:

```bash
sbatch "$WAM_RUN_ROOT/b200-8/train.sbatch"
```

There is one Slurm task per node; each task launches eight torchrun ranks. FSDP
shards within an eight-GPU node; HSDP replicates across nodes. The default
sample cap is 64 per rank, nominal global batch is 2,048, and gradient
accumulation is derived exactly. For 1/2/4 nodes it is 4/2/1. Incompatible batch
arithmetic fails before files are written. `--steps` is required: the launcher
adds no wall-time, spend, or automatic cancellation limit. The 2,000-step
example follows upstream LIBERO-10's training schedule.

## Four-GPU experiment

Use `--nodes 1 --gpus-per-node 4` to allocate four B200s on one host. The
launcher starts exactly four ranks, shards the model across those ranks and
checks that only four B200 devices are visible. With the default sample cap
of 64 and nominal batch of 2,048, gradient accumulation is eight. Eight GPUs
per node remains the default. Source, precision, optimizer, seed and dataset
settings are unchanged.

This option specifies active training devices, not the cloud VM shape. The
reserved B200 platform currently offers one- and eight-GPU VMs in the campaign
project. Running on four devices of an exclusive eight-GPU host still retains
the whole VM allocation. Report active training GPU-hours and provisioned
capacity separately. The [four-rank process proof and GPU telemetry](../../../../docs/workbench/evidence/cosmos3-wam-live-training-4/README.md)
establish initial native training with these settings. The attempt was cancelled
after the operator clarified the requested four-node topology. It did not finish
the schedule, timing repeats or policy evaluations. See the appended cancellation
receipt; the original capture is retained unchanged.

## Files and settings

| File | Purpose |
| --- | --- |
| `sources.json` | Immutable framework, model, tokenizer, dataset, VAE, and simulator revisions |
| `probe.py` | Actual BF16 CUDA attention forward/backward and native FP32 fused Adam update |
| `cluster.py` | Private reserved-capacity Soperator spec; 2 TiB shared jail, explicit node count |
| `prepare.py` | Pinned HF downloads, actual Parquet/camera checks, dataset hashes, DCP conversion |
| `export_data_sample.py` | Native decoded camera/action window for the reproducible data-pipeline figure |
| `recipe.py` / `train.toml.in` | Native TOML, batch arithmetic, Slurm launch, per-node completion evidence |
| `distributed_preflight.py` | Rank/host placement and a real NCCL all-reduce before training |
| `report.py` | Complete-checkpoint verification, timings, GPU-hours, and comparable-run speedup |
| `scaling_report.py` | Three-repeat scaling summary, run variability and actual token-work comparison |
| `profile_report.py` | CUDA trace hashes, kernel categories and overlap-aware observed busy time |
| `storage_telemetry.py` | Timestamped Linux disk counters to quantify checkpoint I/O |
| `benchmark-protocol.json` | Full schedules, repeated timing runs, separate profiles, and the 90% quality target |
| `bootstrap-controller.sh` / `slurm_controller.py` / `slurm.conf.in` | Fresh dedicated Ubuntu Slurm controller and first worker |
| `preserve-slurm-ipc.sh` | Prevent login-session cleanup from deleting batch-job shared memory |
| `ipc_probe.py` | Thirty-second background shared-memory survival check across SSH logout |
| `add_worker.py` / `bootstrap-worker.sh` / `slurm_worker.py` | Private second-worker bundle, shared storage, and native Slurm join |
| `prepare_simulation.py` / `simulation-requirements.txt` | Separate pinned Python 3.10.21 CPU LIBERO environment |
| `simulation_probe.py` | All 500 initial states and actual two-camera renders for all ten tasks |
| `simulation_client.py` | Native evaluator with restricted NumPy state deserialization |
| `evaluate.py` | Native policy servers and all ten tasks, with per-trial evidence and strict result checks |
| `quality_curve.py` | Checkpoint-linked success curve and observed training time to the first passing checkpoint |

`recipe.py plan` requires `--shared-root`, `--run-dir`, `--name`, `--nodes`, and
`--steps`. Optional `--samples-per-rank=64`, `--global-batch=2048`, `--seed=42`,
and `--profile` control the documented experiment. A run name accepts letters,
digits, underscores, and dashes. Reusing a run directory fails; silent restart
or warm-start is not implemented. The trainer keeps its native checkpoint
cadence of 500 and saves the final requested iteration even between cadence
boundaries. Training seeds are fixed without claiming bitwise identity across
different distributed topologies.

`prepare.py --shared-root PATH` must run in the pinned Linux training
environment. `--data-only` performs a CPU data check without downloading model
weights or converting a checkpoint. The data checker needs `huggingface_hub`,
NumPy, PyArrow and PyAV. It verifies raw state/action shapes and finite values,
all Parquet row/episode/task counts, both camera streams, and one decoded frame
per video file. It does **not** fully decode every video frame. Downloads use
the caller's HF configuration; `HF_TOKEN` is optional for these public payloads.
Conversion overrides the native moving processor reference with the staged
model's processor and uses the staged VAE. Training uses a separately pinned
Qwen tokenizer. Child processes receive the native virtualenv's executable path.

`export_data_sample.py --shared-root PATH --output-path FRESH_DIRECTORY` runs
the native LIBERO loader on CPU and exports its first training window. The
[data figure](../../../../docs/workbench/evidence/cosmos3-wam-data/README.md)
includes the recorded images, action arrays and regeneration instructions.

For a runtime check on an allocated B200, run in that Linux training environment:

```bash
COSMOS_TRAINING=1 LD_LIBRARY_PATH="" \
  "$WAM_SHARED_ROOT/framework/.venv/bin/python" "$WAM_RECIPE/probe.py" \
  --output "$WAM_SHARED_ROOT/runtime-probe.json"
```

This checks one visible device and refuses to overwrite an existing receipt.
It does not load WAM weights, test cross-node NCCL, or establish training capacity.

`cluster.py` requires `NPA_PROJECT_ID`, `NPA_TENANT_ID`, and
`NPA_CAPACITY_BLOCK_GROUP` from private operator configuration, plus explicit
`--name`, `--nodes`, and `--output`. It always selects non-preemptible reserved
B200 capacity in `us-central1`; NPA checks reservation ownership, available
quantity and `us-central1-b` fabric compatibility before deployment. It never
falls back to on-demand capacity. Exact IDs belong in the private generated
spec, never in Git. `WAM_SHARED_ROOT`, `WAM_RUN_ROOT`, `WAM_RECIPE`, and
`NPA_PYTHON` in the runbook are operator-selected shell paths.

At runtime, Slurm supplies `SLURM_NODEID`, `SLURM_NNODES`, and
`SLURM_JOB_NODELIST`. The batch script derives `MASTER_ADDR` from the allocation;
`MASTER_PORT` defaults to 29500 and may be overridden. It sets
`COSMOS_TRAINING=1`, `OMP_NUM_THREADS=4`, `TZ=UTC`, disables W&B, and supplies native
`LIBERO_ROOT`, `BASE_CHECKPOINT_PATH`, `WAN_VAE_PATH`, and
`IMAGINAIRE_OUTPUT_ROOT`. `NPA_WAM_RUN_DIR` is an internal preflight output path.
CUDA, NCCL, network-interface and InfiniBand settings otherwise come from the
qualified cluster environment; the recipe does not force TCP or disable IB.
If Soperator supplies a FIFO through `NCCL_DEBUG_FILE`, the launcher replaces
that logging destination with `nccl-node-<node-rank>-%p.log` in the run directory.
The FIFO reader exits after the preflight's writers close and would block the
subsequent trainer. Separate process logs preserve NCCL diagnostics without
changing the selected transport. Regular-file destinations remain unchanged.
The launcher clears the inherited `SLURM_TRES_PER_TASK` and explicitly supplies
128 CPUs and eight GPUs to `srun`. Slurm 23.11 otherwise rejects the inherited
CPU-only TRES together with `--gpus-per-task` before starting a worker.

## Native VM and evaluation additions

Follow [native-cluster.md](native-cluster.md) for the deployment used in the
campaign, including second-worker joining, accounting and cleanup.

On a **fresh dedicated Ubuntu 24.04 B200 VM**, with the qualified NVIDIA driver
and fabric manager already installed, `bootstrap-controller.sh` installs Slurm
23.11.4 and configures the first worker, controller and MariaDB accounting.
It refuses to replace an existing Slurm configuration. Set private
`NPA_SLURM_CLUSTER_NAME` and `NPA_SLURM_ACCOUNT` values before running it as the
operator user with sudo. Accounting credentials are generated on the VM and
written with mode 0600. The dedicated IPv4 firewall permits SSH, established
connections, loopback and the worker's own address. IPv6 permits SSH, loopback,
established connections and ICMPv6, denying new Slurm/NFS connections;
`persist-firewall.sh` saves both families for reboot. Required, ordered systemd
dependencies prevent Slurm, accounting, NFS and rpcbind from starting when either
restore fails. The [real systemd activation check](../../../../docs/workbench/evidence/cosmos3-wam-firewall/README.md)
injects both failures in a disposable container. This hardening postdates the
measured campaign; it is not a GPU-cluster reboot qualification. See
[native-cluster.md](native-cluster.md) for the original runtime scope.
`add_worker.py` requires unambiguous node, partition and accounting-host targets
before writing configuration, and refuses to overwrite an existing NFS export.
Multi-node joining additionally requires private peer
addresses, shared Munge credentials and a common filesystem. This research
deployment colocates the controller on a worker; it has no controller failover.

Both bootstrap scripts run `preserve-slurm-ipc.sh`, which sets
`RemoveIPC=no` in a systemd-logind drop-in and restarts that service. Without
this setting, closing the last SSH session can delete shared-memory objects
belonging to a running Slurm data loader. The campaign reproduced this failure
with a separate background process; a normal training exit status alone is
insufficient evidence. The reporter rejects Python tracebacks from worker
threads as well as failed optimizer updates.

`prepare_simulation.py --shared-root PATH` creates a fresh `PATH/simulation`
with Python 3.10.21, CPU PyTorch 2.14.0, the pinned LIBERO source and the exact
packages in `simulation-requirements.txt`. It needs `uv`, git, `libosmesa6`,
`cmake` and `build-essential`. It preserves the separate CUDA training venv.
The simulator's `LIBERO_CONFIG_PATH` and source `PYTHONPATH` are set explicitly.
The environment contains the dependencies used by evaluation; LIBERO's separate
policy-training dependencies are omitted. PyTorch retains weights-only loading.
The client permits only the NumPy constructors needed by the pinned initial-state
arrays, after checking both native source revisions. It does not enable general
pickle loading to accommodate legacy states.

Before allocating GPUs to evaluation, verify all initial states and simulator
assets with an actual CPU rendering pass:

```bash
"$WAM_SHARED_ROOT/simulation/libenv/bin/python" "$WAM_RECIPE/simulation_probe.py" \
  --shared-root "$WAM_SHARED_ROOT" \
  --output-path "$WAM_SHARED_ROOT/simulation/readiness"
```

The output directory must be fresh. Its manifest records all 500 state inputs,
twenty camera images, pixel hashes and runtime versions. This checks simulator
readiness; it does not execute or score a learned policy.

The [executed simulator check](../../../../docs/workbench/evidence/cosmos3-wam-simulation-runtime.json)
records a fresh bootstrap, all 500 finite states, twenty rendered camera frames,
and the exact dependency inventory hash. Updating the CPU environment preserved
all state values and rendered pixels. The recorded dependency scan is dated;
it is not a guarantee about future vulnerability reports.

The native policy server also initializes Cosmos Guardrail. Before evaluation,
run `npa workbench health access --capability cosmos3 --json` with the operator's
configured credential and require successful exact-payload checks. Pass the
same credential to each evaluation host through an owner-readable token file.
On the native-VM path, keep it outside the shared NFS tree. In the dedicated
Soperator jail, use a restricted operator-private directory that is explicitly
excluded from evidence exports and archives. Set `HF_TOKEN_PATH` to that file
in the actual Slurm batch script or service environment; an interactive SSH shell's credentials
do not automatically reach those processes. Keep the file's parent directory
mode 0700 and the file mode 0600. Never put the token value in job arguments,
scripts, logs or version control.

Prefetch the framework's pinned guardrail before measuring server startup:

```bash
export HF_TOKEN_PATH="$WAM_PRIVATE_HF_TOKEN_FILE"
export HF_HUB_DISABLE_XET=1
uv run --isolated --project "$WAM_SHARED_ROOT/framework/cosmos_framework/utils/hf_cli" \
  --locked --no-default-groups hf download --format=json \
  nvidia/Cosmos-Guardrail1 --repo-type model \
  --revision d6d4bfa899a71454a700907664f3e88f503950cf --include '*'
```

`WAM_PRIVATE_HF_TOKEN_FILE` is the operator-provided absolute file path on that
host. Keep both exports in the evaluation job environment. The pinned helper's
Xet client produced `Unable to parse string as hex hash value` in the live
deployment; `HF_HUB_DISABLE_XET=1` selects the documented alternative download
path without changing the pinned payload or disabling guardrails. See the
[Hugging Face environment reference](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).
Record download time separately from model loading and trial execution.

After training, use `evaluate.py --shared-root PATH --run-dir RUN --step 500
--output-dir OUTPUT` inside an exclusive eight-GPU Slurm allocation. Run only one
evaluation per host: fixed HTTP ports 8000–8007 and rendezvous ports 31000–31007
are part of this dedicated-host launcher; concurrent evaluations are unsupported.
Repeat for
steps 1000, 1500 and 2000. It defaults to eight policy servers, eight simulator
environments per server, all ten tasks, fifty trials per task and seed 42.
`--workers` selects one to eight visible GPUs; `--trials`, `--envs` and `--seed`
are explicit experimental controls. The 500-trial qualification flag is emitted
only for fifty trials per task. Server or simulator errors fail evaluation;
they do not become policy failures. The script checks revisions and the actual
checkpoint loaded by each loopback-bound server. Native evaluation loads EMA
weights, uses thirty UniPC denoising steps and guidance 1.0.
Simulator processes use one OpenMP, BLAS and software-renderer thread each,
so parallel environments do not create nested CPU thread pools. Tracked
modifications to either pinned source checkout fail the evaluation preflight.
Single-environment requests use the native batch client's 300-second HTTP
timeout: the observed first inference required 32.9 seconds and exceeded the
single-environment client's upstream 30-second default. This transport timeout
does not limit the number of trials or the total evaluation duration. Preserve
any failed attempt separately; a request timeout cannot count as a policy failure.

`--record-rollouts --envs 1` preserves native rollout and prediction-comparison
GIFs. Upstream vectorized evaluation does not save those videos, so the script
rejects video recording with multiple environments. Keep separately recorded
illustrative rollouts distinct from the full benchmark's trial results.

After the full schedule and all four 500-trial evaluations finish, run:

```bash
"$WAM_SHARED_ROOT/framework/.venv/bin/python" "$WAM_RECIPE/quality_curve.py" \
  --run-dir "$WAM_RUN_ROOT/b200-8" \
  --evaluations "$WAM_EVAL_500" "$WAM_EVAL_1000" "$WAM_EVAL_1500" "$WAM_EVAL_2000" \
  --output-path "$WAM_RUN_ROOT/b200-8/quality-curve.json"
```

Each `WAM_EVAL_*` value selects an `evaluate.py` output directory. The tool
requires all four scheduled checkpoints and rejects mismatched runs, seeds,
model hashes, trial counts or evaluation parallelism. Keep the number of policy
servers and simulator environments fixed across checkpoints and topologies;
single-trial visual passes are a separate protocol. It uses the native UTC
checkpoint-completion events and verified process duration to report training
time with one-second
log resolution. It identifies the first evaluated checkpoint reaching 90%
success without estimating an unobserved crossing. Evaluation duration and
process GPU-hours are separate from training time. Pooled Wilson intervals
describe the recorded trial counts; they do not measure variation across
training seeds or tasks. The [executed eight-GPU reduction](../../../../docs/workbench/evidence/cosmos3-wam-quality-8/README.md)
includes every original trial result and a reproducible quality figure.

The [completed checkpoint-500 visual check](../../../../docs/workbench/evidence/cosmos3-wam-trained-visual/README.md)
records ten real closed-loop trials, four successes and six failures, trained
model hashes, B200 telemetry and native prediction-versus-simulator videos.
Its one-trial-per-task protocol is separate from the full quality benchmark.

## Measurements and cleanup

```bash
npa/.venv/bin/python npa/workflows/workbench/cosmos3-wam-slurm/report.py \
  --run-dir "$WAM_RUN_ROOT/b200-8"
npa/.venv/bin/python npa/workflows/workbench/cosmos3-wam-slurm/report.py \
  --run-dir "$WAM_RUN_ROOT/b200-16" --baseline "$WAM_RUN_ROOT/b200-8"
```

`--warmup` defaults to 50 optimizer steps. The pinned upstream timing callback
starts emitting regular measurements at step 52; shorter execution checks
cannot produce a steady-state report. This timing exclusion is independent
of the optimizer schedule's 500-update learning-rate warmup. Missing node
receipts, failed jobs,
missing NCCL evidence, gaps/duplicates/non-finite timings and incomplete final
checkpoints fail reporting. Profiling runs cannot be compared as throughput
results. Reports also retain actual processed-token throughput and native VAE
and data-preparation timers. VAE time is included in data preparation; those
timers must not be added together. No automatic extrapolation or policy-quality
assertion is emitted by the training report.

The reporter also writes `iteration-series.csv` with the selected native step
times, loss, processed tokens, and overlapping VAE/data-preparation timers.
Its SHA-256 is recorded in `measurement.json`. This numeric export lets readers
reproduce timing distributions and learning curves without publishing raw
worker logs. It keeps checkpoint-bearing iterations in the series; distinguish
them explicitly when discussing steady compute time.

The pinned trainer saves periodic checkpoints before the timing callback
records the elapsed iteration time.
Consequently, the 2,000-update series includes the writes at updates 500, 1,000,
1,500 and 2,000. A 200-update repeat saves its final checkpoint after the timed
loop: its 149 reported iterations, 52 through 200, measure steady training,
while its process duration includes the final save. The CSV's loss is the
native rank-zero callback value for the last microbatch; it is not a measured
global loss average. Token counts cover every rank and accumulation microbatch.

Every node's log is checked for Python failures and skipped optimizer updates,
including nodes that emit no rank-zero timing lines. Successful process exit
codes alone cannot qualify a run with a failed background data-loader thread.

For the campaign's steady-state comparison, plan three separate 200-update
runs per topology using fresh run names and the same settings. Run `report.py`
on each completed job before archiving its checkpoint. Given repeat directories
named as below, aggregate their saved reports and numeric series:

```bash
npa/.venv/bin/python npa/workflows/workbench/cosmos3-wam-slurm/scaling_report.py \
  --run-dirs \
  "$WAM_RUN_ROOT/b200-8-repeat-1" "$WAM_RUN_ROOT/b200-8-repeat-2" \
  "$WAM_RUN_ROOT/b200-8-repeat-3" "$WAM_RUN_ROOT/b200-16-repeat-1" \
  "$WAM_RUN_ROOT/b200-16-repeat-2" "$WAM_RUN_ROOT/b200-16-repeat-3" \
  --output-path "$WAM_RUN_ROOT/repeated-scaling.json"
```

Both `--run-dirs` and `--output-path` are required. Supplying only the three
eight-GPU repetitions produces a baseline summary with no scaling claim. The
tool requires three reports per supplied topology, matching recorded source,
batch, seed and hardware contracts, and every timing from update 52 to 200.
It verifies CSV hashes and recomputes timing/token summaries before comparison;
duplicate directories or byte-identical copied reports cannot supply repeats.
It reads saved reports without rehashing archived checkpoint payloads or
overwriting the original measurements. Both producer and reducer use `math.fsum`
for elapsed-time totals so Python floating-point summation changes do not alter
receipts. Regression tests reproduce the nine committed producer work summaries
and both documented reducer outputs. Supplying three matching 32-GPU reports adds
`comparisons_vs_8_gpus`, using the fourfold GPU ratio for efficiency and preserving
the existing eight-to-sixteen comparison.

The measured Soperator extension changes the scheduler, driver and shared
storage relative to the original native-Slurm cohort. Report its three repeats
separately instead of assuming that the reducer's recorded fields establish
an otherwise identical environment:

```bash
npa/.venv/bin/python npa/workflows/workbench/cosmos3-wam-slurm/scaling_report.py \
  --single-topology \
  --run-dirs "$WAM_RUN_ROOT/repeat-32-a-r1" "$WAM_RUN_ROOT/repeat-32-b-r1" \
  "$WAM_RUN_ROOT/repeat-32-c-r1" \
  --output-path "$WAM_RUN_ROOT/repeated-timing-32.json"
```

This mode requires one GPU topology with three compatible completed reports.
It reports repeat variability and token throughput without a scaling baseline.

The result retains each run's mean, median and p95, then reports the mean and
sample standard deviation of the three run means. That deviation describes
run variability, not a confidence interval or independent step-to-step noise.
Speedup divides the eight-GPU average run mean by the sixteen-GPU average run
mean; efficiency divides speedup by two. Pooled token throughput and the ratio
of observed token work show whether the nominally matched batches processed
comparable work. The repetitions use the same seed and do not measure
variation in policy quality across training seeds.

For a completed 110-step `--profile` run, use `profile_report.py --run-dir RUN`.
It requires actual CUDA kernels in the two active profiler steps on ranks 0,
8 and so on, and writes `profile-summary.json`. Optional `--output-path PATH`
writes a separate analysis, so an archived original report can be preserved.
Host profiler annotations define distinct steps; duplicate GPU annotations do
not count as additional steps. Kernel categories prefer CPU operators linked
by trace `External id`, with names used for NCCL and unlinked kernels. Fused
Triton kernels remain other; generated names alone do not identify one operation.
Kernel-duration sums can exceed
wall time because streams overlap; the report computes interval unions for
observed kernel busy time and never labels collective duration as exposed
communication stalls. Memory and utilization come from the separate GPU
telemetry. Keep the compressed native traces for timeline inspection.

Retain private `node-*.log`, node receipts, Slurm accounting, native resolved
config, DCP state, profiler traces, `checkpoint-hashes.json`, and
`measurement.json`. Native setup logs can capture environment and infrastructure
details; inspect and redact before publication. Upload artifacts to a private
run-scoped S3 prefix for durable cross-tool handoff. Stop only this recipe's
active jobs with `scancel`, verify they are absent from `squeue`, preserve
artifacts, then remove the dedicated native VMs and GPU cluster as described in
[native-cluster.md](native-cluster.md). For the four-node Soperator cluster,
use `npa soperator destroy` after the same cancellation and archival checks.

## Tests

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/workflows/test_cosmos3_wam_slurm.py \
  npa/tests/workflows/test_cosmos3_wam_scaling_report.py \
  npa/tests/workflows/test_cosmos3_wam_worker.py \
  npa/tests/workflows/test_cosmos3_wam_add_worker.py -q
```

The opt-in live test runs real one- and two-node jobs from a Slurm login. It
requires `NPA_INTEGRATION_E2E=1`, `NPA_WAM_LIVE_ROOT` (fresh shared run parent),
`NPA_WAM_SHARED_ROOT` (prepared inputs/runtime), and `NPA_WAM_STEPS` (explicit
training steps, at least 53 for reporting). It has no automatic timeout:

```bash
npa/.venv/bin/python -m pytest npa/tests/e2e/test_cosmos3_wam_slurm_live.py -q
```
