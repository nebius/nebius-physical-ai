# CLIP application source for Ray Jobs

This application exercises the canonical Workbench `udf_clip_embedding` on
rendered RGB images, then writes Parquet and a Lance table and validates
retrieval. Ray ships the application and explicitly selected UDF source with
`working_dir`; the image and its prepared GPU dependencies stay fixed. This directory contains the
application, not an infrastructure launcher or an alternate workflow catalog.

Use the [Ray + SkyPilot integration recipe](../../../../docs/testing/fast-source-iteration.md)
to start an isolated application Ray cluster, fetch the pinned public CLIP
snapshot, and persist output artifacts.
The application requires compatible Torch/CUDA, Pillow, PyArrow, LanceDB,
Transformers, and an application Ray runtime. The session uses an existing
official PyTorch image and prepares pinned Workbench dependencies once. Pass the
canonical [BDD100K UDF source](../../../src/npa/workbench/lancedb/bdd100k_udfs.py)
explicitly with `--udf-source`; the client copies its exact bytes into each
reviewed package as `npa_lancedb_bdd100k_udfs.py`. The application imports that
file alongside its own source. This directory does not maintain a duplicate UDF.
It refuses CPU execution. The model path must contain the exact runtime-fetched
snapshot; workers set the imported UDF's `CLIP_MODEL_NAME` to that local directory
and use offline model access during inference.

Prepare a client environment with the same pinned Ray version as the application
cluster. NumPy compares completed workload reports; boto3 verifies durable
artifacts and publishes the finish marker after the upload job succeeds.

```bash
uv pip install --python "$NPA_RAY_CLIENT_PYTHON" \
  'ray[default]==2.46.0' 'numpy==2.4.6' 'boto3==1.42.9'
```

Run the Jobs client:

```bash
set -euo pipefail
umask 077
# Resolve these values through your private development configuration.
"$NPA_RAY_CLIENT_PYTHON" npa/workflows/workbench/ray-clip-development/submit.py \
  --address "$NPA_RAY_JOBS_URL" \
  --app-address "$NPA_RAY_APP_ADDRESS" \
  --udf-source npa/src/npa/workbench/lancedb/bdd100k_udfs.py \
  --python "$NPA_RAY_WORKER_PYTHON" \
  --model-path "$NPA_CLIP_MODEL_PATH" \
  --model-revision "$NPA_CLIP_MODEL_REVISION" \
  --output-path "$NPA_RAY_OUTPUT_PATH" \
  --evidence-dir "$NPA_RAY_CLIENT_EVIDENCE" \
  --records 16384 --actors 2 --batch-size 64 --recovery-check --cancel-check
```

For the medium check use one actor and 4,096 records. The complex check uses
16,384 records and at least two actors, preferably across two GPU nodes when the allocation permits.
Only the driver writes local checkpoints and Lance data; actors return vectors
through Ray's object store and need no shared filesystem. Records are
rendered deterministically on CPU workers, cropped, and passed to GPU model
actors in concurrent batches. Each actor retains its model across batches.

The client creates source-only temporary directories containing three reviewed
application files plus the exact selected Workbench UDF. Baseline and restored source crop the left half of each image;
the changed source crops the right half. The rendered inputs stay identical.
Every CPU/GPU worker verifies the changed module hash. The client checks a
submitted hash against the application, worker, validation and UDF files on the
driver and imports in every actor, including replacements. Across source jobs it
requires identical actual model/configuration bytes, UDF, precision, GPU
capability, and runtime versions. It also checks a
meaningful change in the mean CLIP embedding and restoration within numerical
tolerance. The driver also compares every persisted vector: the changed crop
must change at least 99% by an L2 distance above 0.01, and restored vectors must
match within an absolute tolerance of 0.00001. It does not mutate this checkout. It uses upstream Ray source
packaging, rather than NPA's source overlay.

The recovery check kills one exact owned actor after its first atomic shard
commit, creates a replacement, and replays that shard through the driver's
checkpoint lookup. A matching checkpoint returns without another inference;
a changed source/input/model identity or
corrupted Parquet fails. The replacement reloads the model, which its receipt
records. This establishes recovery after a committed shard, not transparent
mid-kernel checkpointing. The separate cancellation job first performs real GPU
work, then continues inference until the client stops its exact submission ID
and verifies `STOPPED`.

Checkpoint identity includes the application, worker, and validation source
hashes, every local model artifact hash (excluding download bookkeeping), the Workbench UDF hash, precision,
GPU capability, and runtime package versions. All GPU actors must agree before
work starts; a replacement must match before replay. A mismatched or unmarked
nonempty output directory fails closed. Checkpoints are local to the driver;
persist them to S3 before the session ends. Actor recovery does not establish
recovery after loss of the driver node.

Receipts separate client packaging/upload/submission, server status transitions,
actor/model preparation, preprocessing/inference, and aggregation. GPU intervals
include CUDA synchronization. A coordinator barrier observes the concurrent
inference wave on one clock; its start/finish observations include RPC edges and
do not subtract clocks from different hosts. Source changes create new jobs and new actors;
they do not hot-reload existing models. Logs and receipts contain operational
paths and worker identities: keep them private, persist them before destroying
the run's worker, and publish only sanitized measurements.

Jobs accepts code execution. Keep the Dashboard bound to loopback, reach it
through authenticated SSH/Kubernetes forwarding, and use an isolated cluster.
The client rejects SkyPilot's management Dashboard port and overrides ambient
`RAY_ADDRESS` deliberately. Never run `ray stop` on a SkyPilot worker.
