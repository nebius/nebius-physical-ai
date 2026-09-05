# CLIP image retrieval with Ray Jobs

[Start with the complete GPU guide](../../../../docs/testing/fast-source-iteration.md).
It covers the tools, private backend setup, first image embeddings, a one-line
Python edit, result download and cleanup. The customer interface is ordinary
`ray job submit/status/logs/stop`; there is no NPA submission or finish wrapper.

The source is split by purpose:

- `embed.py`: the normal Ray Core application—CPU preprocessing, CUDA CLIP actor,
  Parquet vectors, Lance table, retrieval and RGB preview.
- `worker.py`: deterministic RGB inputs and the editable `CROP_POLICY`.
- `cluster.yaml` and `cluster/`: SkyPilot worker request, pinned environment
  preparation and an independent application Ray service.
- `application.py` and `validation.py`: optional distributed/checkpoint checks.
- `network-policy.yaml`: the network boundary described in the guide.

The guide's visible `cp` supplies the canonical Workbench
`bdd100k_udfs.py` as `npa_lancedb_bdd100k_udfs.py`. Ray working-directory upload
transfers those exact bytes. Keep credentials, models, output directories and
unreviewed files out of this source directory.

In `report.json`, `ray_nodes` counts distinct Ray node IDs used by the actors;
the basic report records those IDs in `actors[].node_id`. The advanced report
uses `model_initializations[].node_id` and `final_actors`. These count Ray nodes,
not physical hosts.
On Kubernetes, proving distinct physical workers additionally requires mapping
Ray node addresses to pod IPs and the pods' assigned Kubernetes nodes. The audit
uses that separate platform evidence for its one-worker and two-worker claims.

The pod preparation receipt separates dependency installation, model download
and weight hashing, and `cuda_environment_inspection_seconds`. That last phase
includes the CUDA/version probe and `pip freeze`; it is not model-download time.
Its recorded start, dependency-ready, model-ready and finish timestamps define
those boundaries. Actor `model_load_seconds` measures the later GPU model load.

## Optional: two GPU workers and actor recovery

Use the main guide's setup and connection sequence, adding `--num-nodes 2` to
its **new cluster** `sky launch` command (and its dry run). Each worker requests
one RTX PRO 6000 GPU. Do not resize a cluster that is running someone else's
work. The following commands run from this application directory with the
same Ray client and authenticated tunnel used by the main guide. If you already
ran its Finish step, repeat its setup and UDF copy: Finish removes the generated
UDF snapshot.

Set `CROP_POLICY = "left"` in `worker.py`. Submit the baseline:

```bash
ray job submit --address http://127.0.0.1:8265 \
  --submission-id complex-baseline --working-dir . -- \
  python application.py --actors 2 --records 16384 --recovery-check \
  --output-path /tmp/ray-clip-results/complex-baseline
```

This runs CPU preprocessing partitions, inference batches in two GPU actors,
and driver aggregation into Parquet and Lance. It records actual node/GPU
placement and overlapping inference intervals. `SPREAD` alone is not proof of
two physical nodes; inspect `report.json` together with platform placement.

After the first shard is committed, the application calls `ray.kill` on **one
owned actor**, creates its replacement and replays that shard. Its checkpoint
identity includes source, input, weights and dependency provenance. Replay
verifies the committed Parquet hash and skips a second inference/write. Other
shards continue normally. `recovery.json` reports a new actor instance, model
reload and zero inference calls for the reused checkpoint. This is explicit
application checkpointing; Ray does not promise exactly-once external writes.

Change the visible line to `CROP_POLICY = "right"` and rerun:

```bash
ray job submit --address http://127.0.0.1:8265 \
  --submission-id complex-changed --working-dir . -- \
  python application.py --actors 2 --records 16384 --recovery-check \
  --output-path /tmp/ray-clip-results/complex-changed \
  --compare-baseline-path /tmp/ray-clip-results/complex-baseline
```

Restore `CROP_POLICY = "left"` and submit again:

```bash
ray job submit --address http://127.0.0.1:8265 \
  --submission-id complex-restored --working-dir . -- \
  python application.py --actors 2 --records 16384 --recovery-check \
  --output-path /tmp/ray-clip-results/complex-restored \
  --compare-baseline-path /tmp/ray-clip-results/complex-baseline
```

The changed run requires at least 99% of vectors to differ by L2 distance
above 0.01; restoration requires absolute error at most `1e-5`. All rows and
retrieval checks must pass. Source redeployment creates new actors and reloads
weights; model residency is reused only between batches in the same actor.
Procedural images test execution/retrieval consistency, not semantic accuracy.

## Optional: stop a real running GPU job

This separate job commits a real shard, then continues CUDA inference until
you stop it. It has no implicit runtime limit:

```bash
ray job submit --address http://127.0.0.1:8265 --no-wait \
  --submission-id clip-cancel --working-dir . -- \
  python application.py --actors 2 --records 4096 --cancellation-probe \
  --output-path /tmp/ray-clip-results/cancellation
ray job logs --follow --address http://127.0.0.1:8265 clip-cancel
```

Wait for `RAY_CLIP_FIRST_CHECKPOINT` in the logs, then press Ctrl-C to stop
following logs. The GPU job continues. Cancel precisely that submission:

```bash
ray job stop --address http://127.0.0.1:8265 clip-cancel
ray job status --address http://127.0.0.1:8265 clip-cancel
```

Require `STOPPED`; a stop request alone is not terminal evidence. A cancelled
job has partial checkpoints, not a completed dataset or successful report.

## Preserve the outputs and finish

Use the `CLUSTER` and `RESULTS` values established by the main guide:

```bash
mkdir -p "$RESULTS"
rsync -az "$CLUSTER:/tmp/ray-clip-results/" "$RESULTS/"
for run in complex-baseline complex-changed complex-restored; do
  (cd "$RESULTS/$run" && sha256sum -c SHA256SUMS)
done
cat "$RESULTS/complex-changed/recovery.json"
cat "$RESULTS/complex-restored/retrieval.json"
```

Open `preview.png` and inspect each `report.json` for actual placement,
source hashes, model loads, timing boundaries and full-vector comparison.
The main guide's exact cleanup sequence closes the tunnel, cancels the one
SkyPilot service task and removes the owned development resources. Preserve
these local artifacts before deleting worker storage. No uploader Job,
finish marker, object-store credential or NPA workflow completion signal is
required.

The [audit](../../../../docs/architecture/ray-fast-development-audit.md) records
execution of these commands on two physical GPU workers, their measured timing
boundaries, source/recovery checks and limitations. Earlier session-workflow
results are retained separately as historical evidence.
