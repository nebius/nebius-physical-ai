<!-- register: measured execution report | reader: training engineer and reviewer | consumed: assess and reproduce numeric evidence -->
# Cosmos3-DROID completes five updates on eight H200 GPUs

On October 14, 2026, one eight-H200 node completed five optimizer updates using
native video reads from Standard object storage in eu-north1. Each update used
8,192 samples: 32 per rank, eight ranks and accumulation 32. All 1,280 rank
microsteps had finite losses, covering 40,960 realized training samples.

| Update | Maximum observed rank duration, seconds |
| --- | ---: |
| 1 | 1,484.129464 |
| 2 | 1,599.441460 |
| 3 | 1,590.468639 |
| 4 | 1,552.465747 |
| 5 | 1,543.863512 |

The four post-first durations average **1,571.56 seconds, or 26.19 minutes**.
Their population coefficient of variation is **1.51%**. The predeclared stability
test required at least three post-first durations and variation at most 10%.
Peak allocated memory across ranks was **88.71 GiB**, including optimizer state.
Every rank recorded five actual optimizer operations. These observations verify
a practical batch; they do not establish the largest fitting batch.

The conditional projection is **3,492.36 GPU-hours per 1,000 optimizer updates**:
post-first mean duration multiplied by eight GPUs and 1,000, divided by 3,600.
It extrapolates this short instrumented eager window. Preparation and checkpoint
writes are excluded. It is not measured full-schedule time or a cloud bill.

## Measurement conditions

The source was NVIDIA cosmos-framework at
`cf5d68c00d97ccd2480a2320ed652b92dec63102`. See
[environment.json](environment.json) for the node-preparation runtime versions
and [source pins](../../../../npa/workflows/workbench/cosmos3-droid-h200/sources.json)
for the model, dataset, tokenizer, VAE and separately fetched window filter.

The configuration used bfloat16, full activation checkpointing, eager execution,
four loader workers per rank, inner batch 16 and prefetch factor two. Native
video readers used one 1 MiB buffer per handle, with the upstream 64-decoder LRU.
Metadata and actions stayed local. The original sampled windows, filter,
normalization and augmentation were preserved. Evaluation, telemetry and all
training checkpoint saves were disabled.

Callbacks measured complete optimizer-update intervals. Each published update
uses the largest duration among eight ranks. Model/checkpoint loading,
dataloader construction and prewarming precede the timer, so the first update
is not storage-cold. Synchronization instrumentation adds overhead. A microstep's
compute interval includes the optimizer on the update-ending microstep. The
separately recorded optimizer duration overlaps that interval. CUDA event
intervals include CPU dispatch gaps.

The earlier compiled attempts failed before a microstep on Triton shared-memory
and VAE AOT issues. Those failures did not establish an HBM capacity limit.
The original mounted cold-read gate also remained failed. The operator approved
native/eager continuation on Standard; this run does not change that gate or
establish a Standard service-bandwidth limit.

An earlier 2 GiB SSD cache comparison reduced requests about 70% on a fixed
first trace and 100% on its replay, while wall time improved about 1%. Native
streaming remained the training baseline. Exact replay does not establish
epoch-wide reuse; provider and OS caches were uncontrolled. The separate CPU
worker comparison used fixed map traces, while training used the native
prewarmed iterable stream. Its worker-count result does not isolate one cause
of faster training. No same-condition before/after training speedup is claimed.

## Numeric evidence and reproduction

[microsteps.csv](microsteps.csv) contains 1,280 rank microsteps.
[updates.csv](updates.csv) contains 40 rank-update and optimizer observations.
These are numeric extracts, not unmodified original reports.
[provenance.json](provenance.json) records extract checksums and hashes of the
privately retained original rank receipts. Infrastructure identifiers, private
paths, model weights, source dataset payloads and the window filter are excluded.

```bash
npa/.venv/bin/python npa/workflows/workbench/cosmos3-droid-h200/report.py \
  --evidence-dir docs/workbench/evidence/cosmos3-droid-h200
```

The [application recipe](../../../../npa/workflows/workbench/cosmos3-droid-h200/README.md)
provides a subsequent portable integration of the reader and launch settings.
The historical measurements above apply to the original instrumented code;
they are not timing measurements of that refactor. A separate
[October 15 integration check](integration.json) verifies the portable source
hashes, matched decoded samples, native plan/dryrun, one complete eight-rank
update, finite losses and clean end with no checkpoints. It supplies no
comparable timing result.

This is one node, one short window and one seed. Policy quality, long-run
stability, compiled performance, maximum batch capacity, multi-node scaling and
H200-versus-B200 performance are unmeasured. Nebius supplied the measured GPU
capacity and sells GPU infrastructure. No price comparison is made.
