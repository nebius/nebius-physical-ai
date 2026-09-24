# Cosmos 3 Nano WAM on reserved B200s with Slurm

[Workbench reference assets](../README.md) · [Full runbook](../../../../docs/workbench/cookbooks/cosmos3-wam-slurm.md) · [Blog draft](../../../../docs/workbench/cosmos3-wam-scaling-blog.md)

This native Slurm recipe runs NVIDIA's **LIBERO-10 world action model (WAM)**
post-training experiment. It trains the action and generation pathways together.
It is a standalone application recipe for `npa soperator`, alongside the existing
[single-node policy workflow](../../../../docs/workbench/cosmos3-policy-model-factory.md).
It does not add a new `npa.workflow` toolRef or claim that the inference image is
a training image.

**Validation:** experimental; B200 Slurm execution and scaling measurements are
pending. See [validation.json](validation.json). Offline launch tests, upstream
TOML schema validation, and real dataset inspection do not establish GPU runtime
support, training quality, or time to convergence.

## Plan without cloud resources

From the Workbench checkout, choose absolute paths that will exist on the Slurm
shared filesystem. Planning creates a new directory and never submits a job:

```bash
export WAM_SHARED_ROOT=/shared/cosmos3-wam
export WAM_RUN_ROOT=/shared/cosmos3-wam-runs
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

## Files and settings

| File | Purpose |
| --- | --- |
| `sources.json` | Immutable framework, model, dataset, VAE, and simulator revisions |
| `cluster.py` | Private reserved-capacity Soperator spec; 2 TiB shared jail, explicit node count |
| `prepare.py` | Pinned HF downloads, actual Parquet/camera checks, dataset hashes, DCP conversion |
| `recipe.py` / `train.toml.in` | Native TOML, batch arithmetic, Slurm launch, per-node completion evidence |
| `distributed_preflight.py` | Rank/host placement and a real NCCL all-reduce before training |
| `report.py` | Complete-checkpoint verification, timings, GPU-hours, and comparable-run speedup |

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
`COSMOS_TRAINING=1`, `OMP_NUM_THREADS=4`, disables W&B, and supplies native
`LIBERO_ROOT`, `BASE_CHECKPOINT_PATH`, `WAN_VAE_PATH`, and
`IMAGINAIRE_OUTPUT_ROOT`. `NPA_WAM_RUN_DIR` is an internal preflight output path.
CUDA, NCCL, network-interface and InfiniBand settings otherwise come from the
qualified cluster environment; the recipe does not force TCP or disable IB.

## Measurements and cleanup

```bash
npa/.venv/bin/python npa/workflows/workbench/cosmos3-wam-slurm/report.py \
  --run-dir "$WAM_RUN_ROOT/b200-8"
npa/.venv/bin/python npa/workflows/workbench/cosmos3-wam-slurm/report.py \
  --run-dir "$WAM_RUN_ROOT/b200-16" --baseline "$WAM_RUN_ROOT/b200-8"
```

`--warmup` defaults to 50 optimizer steps. The pinned upstream timing callback
starts emitting regular measurements at step 52; shorter execution checks
cannot produce a steady-state report. Missing node receipts, failed jobs,
missing NCCL evidence, gaps/duplicates/non-finite timings and incomplete final
checkpoints fail reporting. Profiling runs cannot be compared as throughput
results. No automatic extrapolation or policy-quality assertion is emitted.

Retain private `node-*.log`, node receipts, Slurm accounting, native resolved
config, DCP state, profiler traces, `checkpoint-hashes.json`, and
`measurement.json`. Native setup logs can capture environment and infrastructure
details; inspect and redact before publication. Upload artifacts to a private
run-scoped S3 prefix for durable cross-tool handoff. Stop only this recipe's
active jobs with `scancel`, verify they are absent from `squeue`, preserve
artifacts, then destroy only the dedicated cluster via `npa soperator destroy`.
The complete sequence is in the runbook.

## Tests

```bash
npa/.venv/bin/python -m pytest npa/tests/workflows/test_cosmos3_wam_slurm.py -q
```

The opt-in live test runs real one- and two-node jobs from a Slurm login. It
requires `NPA_INTEGRATION_E2E=1`, `NPA_WAM_LIVE_ROOT` (fresh shared run parent),
`NPA_WAM_SHARED_ROOT` (prepared inputs/runtime), and `NPA_WAM_STEPS` (explicit
training steps, at least 53 for reporting). It has no automatic timeout:

```bash
npa/.venv/bin/python -m pytest npa/tests/e2e/test_cosmos3_wam_slurm_live.py -q
```
