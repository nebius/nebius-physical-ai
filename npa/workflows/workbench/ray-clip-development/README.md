# Build an image-retrieval dataset with Ray Jobs

[Ray guide](../../../../docs/workbench/ray.md) · [Reference examples](../README.md)

Embed generated RGB images with Workbench's real CLIP model on an RTX PRO 6000,
search the resulting vectors, then edit one line of Python and submit it again.
Ray Jobs transfers your source to the prepared workers; no image rebuild is
needed for a source edit.

## Start here

Follow the [complete GPU walkthrough](../../../../docs/testing/fast-source-iteration.md)
for setup through cleanup. It includes the source copy, launch, authenticated
connection, baseline, changed result, and download commands in runnable order.

| Prepare | Expected result |
| --- | --- |
| Linux development host, NPA, Ray 2.58 client, OpenSSH, and rsync | Native `ray job submit/status/logs/stop` access |
| Authorized Kubernetes context and [private SkyPilot API](platform/README.md) | One reusable RTX PRO 6000 worker for the basic run |
| Access to pinned public CLIP weights and Python packages | RGB preview, 512-dimensional vectors, and a searchable Lance table |
| A durable local output directory | Downloaded reports and hashes that survive compute teardown |

The basic example generates its own input images; it needs no customer dataset,
Hugging Face token, or S3 credential. Procedural-image results verify execution
and retrieval consistency, not model semantic accuracy.

## Inspect the result

| Artifact | Check |
| --- | --- |
| `preview.png` | Original image and the exact crop embedded by CLIP |
| Parquet vectors and Lance table | Complete row count and successful retrieval checks |
| `report.json` | Actual actor placement, source hashes, model identity, and timings |
| `SHA256SUMS` | Downloaded bytes match the completed output |

Ray node counts describe Ray runtimes. To claim separate physical hosts, also
map Ray node addresses to pods and the pods' assigned Kubernetes nodes.
Source resubmission creates new actors and reloads weights; a running actor
reuses its loaded model between batches.

<a id="optional-two-gpu-workers-and-actor-recovery"></a>
<a id="optional-stop-a-real-running-gpu-job"></a>
<a id="preserve-the-outputs-and-finish"></a>
<a id="review-a-downloaded-result-in-rerun"></a>

## Continue with a specific check

| Goal | Procedure |
| --- | --- |
| Review original images, crops, vectors, and timing in Rerun | [Convert a completed result](view-results.md) |
| Use two GPUs, replace an actor, or stop active inference | [Distributed and recovery checks](distributed-checks.md) |
| Resume an interrupted driver from verified shards | [Checkpoint resume](../../../../docs/testing/ray-clip-checkpoint-resume.md) |
| Preserve results in private S3 and restore them | [Archive and restore](../../../../docs/testing/ray-clip-archive.md) |
| Review measured validation and its limits | [Recorded audit](../../../../docs/architecture/ray-fast-development-audit.md) |

Save and verify outputs before cleanup. Follow the main guide's exact finish
sequence: stop owned Ray Jobs, cancel the SkyPilot service task, remove its
workers, and close the tunnel. The platform API and shared Kubernetes cluster
have a separate lifecycle.

## Find the source to edit

| File | Purpose |
| --- | --- |
| [worker.py](worker.py) | Generated RGB inputs and editable `CROP_POLICY` |
| [embed.py](embed.py) | Basic preprocessing, CUDA actor, vector export, and retrieval |
| [report.py](report.py) | CPU-only conversion of downloaded results to Rerun |
| [cluster.yaml](cluster.yaml) · [cluster/](cluster/) | Worker request and separate application Ray service |
| [application.py](application.py) · [validation.py](validation.py) | Distributed and checkpoint checks |
| [fast_sync.py](fast_sync.py) | CPU-only qualification of source transfer and exact job cleanup |
| [network-policy.yaml](network-policy.yaml) | Namespace ingress boundary used by the platform guide |

The walkthrough copies the canonical `bdd100k_udfs.py` into this directory as
`npa_lancedb_bdd100k_udfs.py`. Repeat that copy after editing the original UDF.
Keep credentials, model caches, results, and unrelated files outside the source
uploaded by `--working-dir`.

The preparation receipt separates dependency installation, model download and
hashing, and `cuda_environment_inspection_seconds` (the CUDA/version probe plus
`pip freeze`). Actor `model_load_seconds` measures the later GPU model load;
these timing boundaries describe different phases.
