# View completed Ray CLIP results

[Example overview](README.md) · [First GPU run](../../../../docs/testing/fast-source-iteration.md)

Start with a completed, downloaded result and its checksum manifest. Set
`RESULTS` to the local result parent directory used by the main guide. This
conversion uses CPU only and does not start Ray or download a model.

## Review a downloaded result in Rerun

After the existing rsync and checksum step, convert either a basic `embed.py`
result or a completed advanced `application.py` result. From the repository root,
use the NPA environment (it already includes `rerun-sdk==0.31.4`, NumPy, PyArrow
and Pillow):

```bash
npa/.venv/bin/python npa/workflows/workbench/ray-clip-development/report.py \
  --input-path "$RESULTS/baseline" \
  --output-path "$RESULTS/baseline.rrd" \
  --run-id clip-baseline
npa/.venv/bin/rerun rrd verify "$RESULTS/baseline.rrd"
npa/.venv/bin/rerun rrd print -vv "$RESULTS/baseline.rrd"
npa/.venv/bin/rerun "$RESULTS/baseline.rrd"
```

Use your public-safe native Jobs submission label for `--run-id`; the producer
has no persisted submission ID, so this label is supplied by the reader and is
not independent Jobs status evidence. Keep recordings private if their images
are private. The converter requires a new `.rrd` outside the downloaded input
tree, preserves its checksum manifests, and emits one JSON receipt with source
and recording hashes plus independently decoded entity row counts. It never
loads CLIP or connects to Ray. Import `convert(input_path, output_path, run_id=...)`
from the same module for Python use.

The first view shows original RGB images beside the exact crops embedded by the
GPU. Move the `record_id` timeline through the first eight records to inspect
images; vectors and their norms cover every record. The Dataframe/Tensor views
can inspect `vectors/embedding`, and `retrieval/top_ids` contains the persisted
query and hit IDs. No spatial projection or semantic accuracy is implied.

Advanced results additionally expose `shard_index` checkpoint counts, verified
checkpoint hashes, per-shard preprocessing/inference durations, and a
`coordinator_elapsed` concurrency trace. This last timeline starts at the first
recorded coordinator observation and covers only its observed inference wave,
including RPC edges. Worker clocks are never combined. Model loads and aggregate
timings are static measurements; their intervals overlap and must not be added.
Recovery lineage is static because the source records no recovery timestamp.
Rerun's built-in `log_time` and `log_tick` describe converter logging; select the
custom timelines above to inspect persisted workload facts.
Application/converter source hashes and model byte identities are retained; host, node, process,
allocation and imported-file identifiers are omitted.

Missing final reports, partial/cancelled results, checksum gaps or corruption,
unsafe paths/symlinks, invalid vectors, mismatched preview bytes, inconsistent
retrievals and corrupt advanced checkpoints fail conversion before publication.
An advanced report also requires a successful actor cleanup receipt. A failed
writer or decoded-content check leaves no final RRD. The original Parquet,
Lance and model artifacts remain the authoritative result; the RRD is a derived
review artifact. Creating the file does not upload or share it.

To qualify a real downloaded result through the committed live test, run once
for each producer you used:

```bash
NPA_INTEGRATION_E2E=1 NPA_RAY_CLIP_RESULTS="$RESULTS/baseline" \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_ray_clip_report_live.py -q
```

The test requires actual CUDA actor and completed inference receipts, invokes
the converter CLI, runs Rerun's independent verifier and decodes every record
index. Unit fixtures use explicit synthetic vectors and are not GPU evidence.
