# Measured eight-to-sixteen B200 WAM scaling

Three matched 200-update repetitions per topology measured **1.9334× step
speedup and 96.6685% efficiency**. Eight-GPU mean ± sample SD across run means:
**13.3080 ± 0.0136 s**; sixteen-GPU: **6.8833 ± 0.0102 s**. Pooled throughput
rose from **68,652.78 to 132,745.58 tokens/s**. Actual token work ratio was
**1.0001082**, about 0.0108% more work on sixteen GPUs.

![Measured scaling and complete training duration](scaling-comparison.png)

The full 2,000-update schedules took **7 h 50 m 33 s** and **4 h 23 m 33 s**,
using **62.74** and **70.28 training-process GPU-hours**. That is **1.7855×
full-run speedup**, or 44.0% less time and 12.0% more training GPU-hours. These
exclude preparation, archive, evaluation and idle reservations, so are not bills
or total campaign resource usage. Full-run storage conditions differ as
recorded in the source evidence; separate repetitions are the primary scaling
measurement. No evaluation/archive I/O overlapped those repetitions.

![Matched checkpoint quality](quality-comparison.png)

The first saved checkpoint above the declared 90% target was update 1,500:
92.8% after about 5 h 53 m on eight GPUs, 94.2% after about 3 h 19 m on sixteen.
Final success was 95.0% and 95.8%, respectively. Each point represents 500
trials; quality was verified retrospectively. One training seed does not show
that GPU count improves quality, or establish unseen-task generalization.
Wilson intervals quantify trial sampling, not variation across training seeds.

## Conditions and source records

Both topologies use the same pinned framework, base weights, data, seed 42,
BF16 compute, FP32 parameters/EMA, per-rank sample cap 64 and global batch
2,048. Accumulation changes from four to two. FSDP shards within eight-GPU
nodes; HSDP synchronizes the two replicas. Each run has 149 timed iterations,
updates 52–200. This timing exclusion differs from the 500-update learning-rate
warmup. The final checkpoint save is outside those iterations.

[repeated-scaling.json](repeated-scaling.json) is the actually executed reducer
output. It validates source/setting contracts and every original CSV, recomputes
statistics and preserves report hashes. [evidence.json](evidence.json) links
the full schedules and quality curves without modifying their original bytes.
Historical earlier records retain pending wording appropriate to their capture;
this completed comparison and the linked quality records supply final status.

## Reproduce from committed data

The reducer uses `math.fsum` for elapsed-time aggregation so Python versions
with different built-in float summation produce the same committed JSON bytes.
The documented reducer command is exercised against all six committed runs in CI.

From the repository root, choose a fresh output path in `WAM_SCALING_REPORT`:

```bash
npa/.venv/bin/python npa/workflows/workbench/cosmos3-wam-slurm/scaling_report.py   --run-dirs   docs/workbench/evidence/cosmos3-wam-timing-8/repeat-1   docs/workbench/evidence/cosmos3-wam-timing-8/repeat-2   docs/workbench/evidence/cosmos3-wam-timing-8/repeat-3   docs/workbench/evidence/cosmos3-wam-timing-16/repeat-1   docs/workbench/evidence/cosmos3-wam-timing-16/repeat-2   docs/workbench/evidence/cosmos3-wam-timing-16/repeat-3   --output-path "$WAM_SCALING_REPORT"
npa/.venv/bin/python docs/workbench/evidence/cosmos3-wam-scaling/plot.py
npa/.venv/bin/python docs/workbench/evidence/cosmos3-wam-scaling/quality_plot.py
```

Matplotlib 3.11.2 and NumPy 2.5.3 produced the figures. Both were inspected and
rendered again with identical image hashes; the two render manifests retain
the results. For new GPU measurements follow the [full recipe](../../../../npa/workflows/workbench/cosmos3-wam-slurm/README.md)
and its fixed protocol, retaining all new outcomes rather than overwriting
these observations.

The [archive verification](archive-verification.json) records full-object download
and SHA-256 checks for every training archive and 515 additional evaluation,
visualization and telemetry files. Raw operational identifiers stay private;
the public records retain numeric evidence and source hashes. Both owned GPU
VMs were stopped after terminal jobs, empty queues and evidence preservation.
Checkpoint disks and private archives remain retained.
