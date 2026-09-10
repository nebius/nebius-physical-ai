# Use Workbench with Ray

Workbench uses Ray through a small set of explicit paths. There is no `npa ray`
command and no NPA wrapper around Ray Jobs. Use the native `ray` CLI for
application submission and control; use `npa` only for the infrastructure,
Workbench service, or workflow surface listed below.

| Need | Supported path | Start here |
| --- | --- | --- |
| Edit and run a distributed GPU application | SkyPilot hosts plus native Ray Jobs and Core | [GPU Ray Jobs guide](../testing/fast-source-iteration.md) |
| Exercise multi-node GPU training and recovery | Guarded Ray Train V2 synthetic reference | [Ray Train reference](../../npa/workflows/workbench/ray-train-synthetic/README.md) |
| Keep Cosmos3-Nano loaded for repeated batches | `npa workbench cosmos3 ray-serve` and `ray-batch` | [Cosmos3 native Ray Serve](cosmos3-ray-serve.md) |
| Provision a fixed CPU RayCluster | `npa fleet` KubeRay policy, then native Ray Jobs | [Fleet KubeRay guide](../fleet-kuberay.md) |
| Use Ray Data | No first-class Workbench path | Submit your own trusted Ray application through a Jobs path above. |
| Use Ray Tune | No first-class Workbench path | Submit your own trusted Ray application; Workbench ships no Tune recipe or artifact contract. |

The CLIP and Train examples are guarded application references outside the
`npa.workflow` catalog. The Cosmos batch client is the only Ray-backed path in a
checked-in `npa.workflow` spec (`workflows/testing/cosmos3-ray-batch.yaml`), and
its persistent service must already be ready. KubeRay is infrastructure policy,
not a workflow runtime or Jobs controller.

## Common native Jobs lifecycle

The detailed guides create the cluster and a loopback tunnel differently, but
application control is always native Ray Jobs. Use the client version pinned by
the selected guide, choose a unique submission ID, and keep ambient Ray address
variables from overriding the explicit endpoint.

```bash
unset RAY_ADDRESS RAY_API_SERVER_ADDRESS
export RAY_API=http://127.0.0.1:8265
ray job list --address "$RAY_API"
export JOB_ID=clip-baseline
ray job status --address "$RAY_API" "$JOB_ID"
ray job logs --address "$RAY_API" "$JOB_ID"
ray job stop --address "$RAY_API" "$JOB_ID"
```

`job stop` is a request, not completion evidence. Poll `job status` until the
exact submission is terminal. Ray Jobs accepts trusted code with the runtime's
permissions, so keep the dashboard and GCS private and use an authenticated SSH
or Kubernetes loopback forward rather than a public load balancer.

For the SkyPilot-hosted references, bootstrap the repository-pinned engine and
invoke its returned binary rather than `sky` from `PATH`:

```bash
npa skypilot bootstrap
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"
```

Follow each reference for its exact `launch`, tunnel, and application command.
Application Ray is separate from SkyPilot's management Ray; do not use ambient
discovery or a broad `ray stop`.

## Shortest useful paths

### Ray Jobs and Core

Use the [CLIP development guide](../testing/fast-source-iteration.md). It starts
a GPU service task with SkyPilot, submits `embed.py` with `ray job submit`, and
downloads checksummed Parquet, Lance, preview, retrieval, and report artifacts.
Its shortest application submission, from the example directory, is:

```bash
ray job submit --address "$RAY_API" --submission-id clip-baseline \
  --working-dir . -- \
  python embed.py --output-path /tmp/ray-clip-results/baseline
```

Python edits travel through `--working-dir` without an image rebuild. The
[advanced recipe](../../npa/workflows/workbench/ray-clip-development/README.md)
adds actor recovery, partial-checkpoint reuse, cancellation, and factual Rerun
conversion. Completed results can be moved to immutable S3 storage with the
[archive companion](../testing/ray-clip-archive.md).

### Ray Train

Use the [two-host synthetic reference](../../npa/workflows/workbench/ray-train-synthetic/README.md).
Its real submission is native Jobs:

```bash
cd npa/workflows/workbench/ray-train-synthetic
ray job submit --address "$RAY_API" --submission-id train-baseline \
  --working-dir . -- /opt/npa-ray-train/env/bin/python train.py \
  --storage-path "$TRAIN_STORAGE_URI" --run-name baseline \
  --output-dir /opt/npa-ray-train/exports/train-baseline
```

The reference uses Ray Train V2 and writes native checkpoints under the selected
S3 storage path. Its final export contains `state.pt`, `metrics.json`,
`metrics.rrd`, `result.json`, and `SHA256SUMS`; the checksum manifest is
published last and every object is read back. Retry a failed export from the
preserved host directory with `inspect_results.py --publish` instead of
retraining. Training recovery reuses the same `RunConfig(name, storage_path)`;
do not apply Ray Train V1 restore APIs.

### Ray Serve

Cosmos3-Nano has the only first-class Workbench Ray Serve path. Before starting
GPU work, verify credentials, model access, and the accepted image definition:

```bash
npa workbench health preflight --checks hf,ngc,s3 --json
npa workbench health access --capability cosmos3 --json
npa workbench golden-eval show cosmos3-ray-serve
```

Inside the digest-pinned service image, start the model and upstream batching
implementation with:

```bash
npa workbench cosmos3 ray-serve --world-size 1 --max-batch-size 4
```

Then submit a durable batch from a CPU client:

```bash
npa workbench cosmos3 ray-batch \
  --input-path s3://<bucket>/<prefix>/batch.json \
  --output-path s3://<bucket>/<prefix>/outputs/ \
  --endpoint http://<private-service>:8000
```

The bearer token comes from `NPA_COSMOS3_RAY_TOKEN`, not the command line. The
client verifies the response and publishes `request.json`, `response.json`,
generated media, and `provenance.json`. If client validation fails after model
execution, preserve the failed result and original native output directory
before retrying. See the [service guide](cosmos3-ray-serve.md) for readiness,
input schema, guardrails, GPU qualification, and recovery.

### KubeRay

Fleet can install one fixed, CPU-only RayCluster from a reviewed KubeRay recipe.
Start with `npa/examples/fleet/kuberay/cpu-raycluster.yaml`, copy it outside the
repository, and run:

```bash
npa workbench health preflight --checks nebius --json
npa fleet plan --spec /path/to/private-fleet.yaml
npa fleet deploy --spec /path/to/private-fleet.yaml --no-create-projects --yes
```

Forward `service/ray-cluster-head-svc` to loopback and use native `ray job`
commands as shown in the [Fleet KubeRay guide](../fleet-kuberay.md). This path
supports fixed CPU workers only: no GPU workers, RayService, autoscaling,
arbitrary image, Ray Train guarantee, or durable application artifact layer.
Save logs and outputs before teardown because the Ray object store and `/tmp`
are ephemeral.

## Artifacts, failure, and cleanup

| Path | Durable boundary | Recovery boundary |
| --- | --- | --- |
| CLIP Jobs/Core | Downloaded checksummed result; optional immutable S3 archive | Reuse only validated committed shards; an interrupted job is not a completed dataset. |
| Ray Train | Native S3 checkpoints plus checksum-bound S3 export | Train V2 restarts from its matching run; retry preserved export bytes separately. |
| Cosmos Ray Serve | Verified S3 request, response, media, and provenance | Preserve failed client/native output evidence; fix access or contract errors before resubmission. |
| Fleet KubeRay | Nothing by default | Save application outputs outside the cluster before owned teardown. |

Do not remove hosting resources while an application job is active. Stop every
exact native Jobs submission and confirm terminal state, then cancel the exact
SkyPilot service task or destroy the owned Fleet target. Preserve shared
Kubernetes clusters, SkyPilot APIs, controllers, projects, and storage unless
you separately own their lifecycle. The detailed guides contain the precise
cleanup commands for each ownership model.
