# PAIDF dataset batches, robot inputs, and private processing

[Workbench](../README.md) · [PAIDF setup](../../../workflows/guides/paidf-cosmos3.md)

Use one private batch manifest to submit multiple datasets to
`workflows/main/paidf-cosmos3.yaml`. Each episode/camera becomes an independent
workflow run with its own provenance, quality decisions, outputs, and resume
identity. The generic batch driver also accepts other `npa.workflow` specs.
It uses existing cluster capacity; it does not provision or resize node groups.

## Submit four datasets with 80 episodes each

Save this as a private `paidf-batch.yaml`. Replace every placeholder. Workflow
paths are absolute or relative to the manifest; camera keys come from each
dataset's `meta/info.json`.

```yaml
apiVersion: npa.workflow.batch/v0.0.1
batch_id: robot-augmentation-001
workflow: /path/to/workbench/workflows/main/paidf-cosmos3.yaml
project: <your-project-alias>
infra: k8s/<your-context>
vars:
  bucket: <your-bucket>
  variant_count: 20
  variant_parallelism: 1
entries:
  - id: dataset-a
    lerobot_uri: s3://<your-bucket>/datasets/a/
    lerobot_camera: observation.images.front
    episodes: {start: 0, stop: 80}
  - id: dataset-b
    lerobot_uri: s3://<your-bucket>/datasets/b/
    lerobot_camera: observation.images.front
    episodes: {start: 0, stop: 80}
  - id: dataset-c
    lerobot_uri: s3://<your-bucket>/datasets/c/
    lerobot_camera: observation.images.front
    episodes: {start: 0, stop: 80}
  - id: dataset-d
    lerobot_uri: s3://<your-bucket>/datasets/d/
    lerobot_camera: observation.images.front
    episodes: {start: 0, stop: 80}
```

The range includes `start` and excludes `stop`. Use `episodes: [0, 4, 12]` for
a subset. For MP4 objects, replace the LeRobot fields with
`input_uri: s3://<your-bucket>/clips/episode-000.mp4` and give each entry a
unique ID. Other workflow types use entries with `id` and optional `vars`.

```bash
npa workbench workflow batch plan ./paidf-batch.yaml
npa workbench workflow batch submit ./paidf-batch.yaml \
  --state-dir ./private-batch-state --max-concurrent-runs 10
```

This submits all 320 episodes with one command. With 20 variants per episode,
it requests 6,400 candidate videos per generation pass. For **200 episodes ×
20 variants**, use one entry with `episodes: {start: 0, stop: 200}`:
200 workflow runs request **4,000 candidates per pass**. Quality rejection,
curation, failures, and refinement retries mean this is neither a guarantee of
4,000 accepted outputs nor a bound on inference attempts. Count accepted
artifacts from committed manifests and final reports.

Planning is offline. It checks the workflow, explicit selectors, unique run
IDs, and isolation of declared outputs across reachable branches. It does not
read dataset objects, check credentials/capacity, or run inference. Use
`check-input` below to read and decode selected input before submission.

Run the foreground driver on a persistent operator host, for example in tmux.
Each slot remains occupied until the whole workflow client finishes. Available
slots refill independently, so episodes can prepare, generate, and curate at
the same time.

## Progress and recovery

```bash
npa workbench workflow batch status "robot-augmentation-001" \
  --state-dir ./private-batch-state
npa workbench workflow batch submit ./paidf-batch.yaml \
  --state-dir ./private-batch-state --max-concurrent-runs 10 --resume
```

Retain the same manifest and state directory. The private directory contains a
fingerprinted plan, workflow snapshot, atomic state, and per-attempt JSON/stderr
logs. Children inherit the exclusive local lock so a crashed driver cannot be
replaced while its original clients still hold it.

| Local status | Meaning |
| --- | --- |
| `pending` | No launch has been attempted by this driver. |
| `succeeded` | The standard runtime returned success for this exact run ID. |
| `reconcile-required` | A client has started or exited without confirmed success; the remote workload may still exist. |

An unconfirmed exit stops admission of pending entries; other active clients
finish and the batch exits nonzero. Resume uses standard
`workflow submit --resume-run` for started runs before admitting pending work.
Completed runs are also rechecked through standard resume before new work is admitted.
Changed workflow/source bytes, entries, project, explicit source selection or SkyPilot
configuration require a new batch ID. Keep the editable checkout unchanged while running. Concurrency may change, but
cannot be lowered below the number of unresolved runs.

If failure preceded standard submission-state creation, standard resume may
refuse it. Inspect the client logs and use existing workflow status/recovery
commands to establish what happened; do not erase the ledger to force a retry.
Interrupting the driver stops local clients **without cancelling remote jobs**.
Inspect original IDs with `workflow status`; use `workflow cancel` for exact
runs you intend to stop.

The standard runtime owns durable wave state, verified output reuse, and
infrastructure recovery. The batch adds no blind retries or per-wave execution
deadline (`--max-wait-seconds 0`); normal submit preflights remain enabled.
This is a local admission driver, not a distributed or highly available queue.
Use one operator host and retained state directory per batch. `batch status` is a
historical local observation; `batch submit --resume` rechecks durable run state.
Keep logs and state private. Batch planning requires an editable NPA checkout.
Its source fingerprint covers the exact source-staging files, and admission
checks that fingerprint between groups of launches. SkyPilot configuration is
snapshotted into the owner-only ledger; credentials can rotate independently.

## Check your own robot data

```bash
npa workbench workflow check-input \
  --project "$PROJECT_ALIAS" \
  --lerobot-uri "s3://$BUCKET/datasets/robot/" \
  --lerobot-camera observation.images.front --lerobot-episode 0

npa workbench workflow check-input --input-video ./robot-episode.mp4
npa workbench workflow check-input \
  --project "$PROJECT_ALIAS" --input-uri "s3://$BUCKET/clips/episode-000.mp4"
```

The command emits one JSON document and exits nonzero on failure. It uses
submission's exact LeRobot selector and MP4 codec checks, then fully decodes
the selected clip and checks its timeline. It downloads needed metadata and
media to temporary local storage without uploads, model calls, or GPU use.
Reports contain media properties and repair hints, excluding source URIs and
hashes. Install FFmpeg/ffprobe; LeRobot v3 metadata also requires PyArrow.
Storage credentials resolve through the selected project; a named project with
missing credentials does not fall back to another project or ambient boto configuration; `--s3-endpoint` explicitly overrides the endpoint.

| Input | Required contract |
| --- | --- |
| MP4 | One H.264 MP4 with positive dimensions, duration and frame rate; full decoding and increasing presentation timestamps must pass. The extension alone is insufficient. |
| LeRobot v2.x | Dataset root with `meta/info.json`, declared video features/episode count, and the selected video at the declared/default per-episode path. Preserve directory layout. |
| LeRobot v3.x | Root metadata plus episode Parquet shards under `meta/episodes/`; camera chunk/file indices and valid `from_timestamp`/`to_timestamp` intervals identify the selected part of a shared video. |
| Selection | Explicit episode index and exact full feature key with `dtype: video`, such as `observation.images.front`. |
| Other robot formats | Export the intended camera trajectory to H.264 MP4 or convert to the supported LeRobot layout. ROS bags, JPEG folders and action/state tables are not interchangeable inputs to this main workflow. |

| Failure | Repair |
| --- | --- |
| Missing `meta/info.json` | Select the dataset root, not `videos/` or a parent containing multiple datasets. |
| Camera not declared | Copy the exact video feature key from `meta/info.json`. |
| Episode outside range | Check the dataset metadata and zero-based, stop-exclusive range. |
| Missing v3 metadata/video | Preserve metadata shards and referenced shared files; video-file indices are not episode numbers. |
| Unsupported codec | Transcode a copy: `ffmpeg -i INPUT -c:v libx264 -pix_fmt yuv420p OUTPUT.mp4`; check that copy. |
| Decode/timeline failure | Check for truncated media or invalid timestamps; renaming the file cannot repair it. |
| Storage failure | Check project, endpoint, object-read permission and metadata-list permission. |
| Later quality rejection | Inspect the evaluator report and `reports/quality-evidence.rrd`; input acceptance does not prove generation quality. |

A check covers one selected episode/camera, not the whole dataset. PAIDF
prepares the complete source timeline for its model bucket (832×480 at the
configured rate, normally 24 fps); customers need not pre-resize the source.
Outputs are augmented video and review evidence, not regenerated synchronized
robot actions/states or a training-ready LeRobot dataset.

A repaired starter sample does not establish customer-data compatibility.
For a remaining failure, retain the check JSON, NPA Git revision, format/version,
selected episode/camera and failing-stage error in a private support handoff.

## Bring private Python code without a Workbench fork

For ordinary Python edits, start with the supported [native Ray development
walkthrough](../../testing/fast-source-iteration.md). It provisions a prepared
worker through SkyPilot, opens a private SSH tunnel to application Ray, and proves
that editing a complete module changes remote outputs. Use the pinned clients
and platform preflight from that guide; application Ray is separate from
SkyPilot's management environment.

Once that worker and tunnel are ready, use your own small application directory:

```bash
cd /path/to/private-app
ray job submit --address http://127.0.0.1:8265 \
  --submission-id private-baseline --working-dir . -- python main.py
# Edit your processing module locally, then submit a new job.
ray job submit --address http://127.0.0.1:8265 \
  --submission-id private-updated --working-dir . -- python main.py
ray job status --address http://127.0.0.1:8265 private-updated
ray job logs --address http://127.0.0.1:8265 private-updated
```

Clone a private source repository on your development host using its existing
Git authentication. Ray uploads the selected local directory; Git credentials
need not be supplied to workers for that path. Keep credentials, datasets and
outputs outside it; review `.rayignore`, symlinks and upstream package-size rules.
The worker must already have compatible dependencies. Python source edits need
no image rebuild; changes to system packages, CUDA or binary extensions may.

For a single sequential application, native SkyPilot can instead sync a workdir
and execute directly on a prepared cluster:

```bash
"$NPA_SKYPILOT_BIN" exec "$CLUSTER" --workdir . "python main.py"
```

Wait for that job to finish before syncing a changed revision into the same
cluster workdir. SkyPilot's cluster workdir is shared across queued jobs, whereas
Ray Jobs packages source per job. Neither command reruns cluster setup; follow
the development guide to change the image or dependencies.

SkyPilot owns prepared worker/service lifecycle. Native Ray Jobs owns application
submission, logs, status and stop; Ray Core schedules your tasks and actors.
Workbench keeps the production workflow, consent, run identity, quality gates
and durable artifact handoffs. A Ray worker pool is not automatic Kubernetes
node-group scaling. Use the standard workflow runtime for durable PAIDF stages;
no extra Ray Jobs controller or Workbench repository fork is needed for editing.

## Private images

Select your registry image by immutable digest in the private workflow resource
profile. Export the registry's exact host and its supplied pull credentials from
your secret store in the submitting process:

```bash
export SKYPILOT_DOCKER_SERVER="$PRIVATE_REGISTRY_HOST"
export SKYPILOT_DOCKER_USERNAME="$PRIVATE_REGISTRY_USERNAME"
export SKYPILOT_DOCKER_PASSWORD="$PRIVATE_REGISTRY_PULL_TOKEN"
npa workbench workflow preflight-images ./private-processing.yaml \
  --project "$PROJECT_ALIAS" --infra "k8s/$KUBE_CONTEXT"
```

The receiving variables above come from your secret store, not committed files.
NPA forwards credentials only to the matching registry host and does not mint
arbitrary provider tokens. On Kubernetes, verify the actual pull path and any
operator-managed image-pull secret before submission; successful registry
manifest access alone is insufficient. A successful local `docker login` alone
does not prove worker access. The image must support the SkyPilot bootstrap and
contain your executable/dependencies. An arbitrary existing image may need a
compatible private derivative. Qualification errors distinguish registry
transport problems from missing payloads.

## Private stages and external services

**Private workflows do not require a repository PR.** Keep a private YAML
copy, use `run.argv` or `run.shell` for your processing, and select a compatible
private image in its resource profile. Catalog `toolRef` registration is for
reusable first-party integrations.

This example defines an **adapter contract to implement in your image**.
It describes the contract for a private service adapter you supply:

```yaml
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: private-processing
  executionMode: runtime
config:
  bucket: <your-bucket>
  prefix: "private-processing/{{run.id}}"
  input_manifest_uri: s3://<your-bucket>/accepted-source/manifest.json
resources:
  private:
    cloud: kubernetes
    cpus: 4
    memory: 16Gi
    image: registry.example.com/team/private-adapter@sha256:<your-image-digest>
initial: process
states:
  process:
    resources: private
    run:
      argv:
        - /opt/company/bin/process
        - --input-path
        - "{{config.input_manifest_uri}}"
        - --output-path
        - "s3://{{config.bucket}}/{{config.prefix}}/result.json"
        - --idempotency-key
        - "{{run.id}}"
    outputs:
      - uri: "s3://{{config.bucket}}/{{config.prefix}}/result.json"
        schema: company.processing.result.v1
    terminal: true
```

The image must contain this executable and satisfy the SkyPilot bootstrap
contract. Validate and render the actual private spec before execution:

```bash
npa workbench workflow validate-spec ./private-processing.yaml
npa workbench workflow plan-spec ./private-processing.yaml --check-render
```

To append it to PAIDF, remove `terminal: true` from `finalize`, add
`next: process`, and add the private resource and processing state. Pass the
committed final report or accepted candidate manifest to the adapter. Preserve
PAIDF's workflow identity, input selection and quality route; rejected candidates
must not enter training/export stages.

For an external service, establish the actual API/SDK version, schemas, authentication,
network access, completion status and retry semantics. The private adapter can
submit accepted artifacts, persist an external job ID under the run prefix,
poll that exact job, and publish a result after verified completion. On resume,
reuse that identity or the provider's real idempotency mechanism. An HTTP
acceptance response alone is not completion.

Pass credential names through `--secret-env NAME` or batch
`secret_env: [NAME]`; keep values outside YAML and argv. The adapter reads
tokens from its environment, returns nonzero on failure, and declares outputs
it actually produces. No native service integration is claimed until a real
interface and credentials have been tested. For review-provider integration
over immutable PAIDF candidates, see [campaign reuse](paidf-campaign-reuse.md).

## Concurrency and existing capacity

| Control | Scope |
| --- | --- |
| `batch submit --max-concurrent-runs 10` | At most 10 active workflow clients in this batch. |
| `variant_count: 20` | Candidate variants per episode per generation pass. |
| `variant_parallelism` and GPU resources | Generation concurrency and GPU allocation within an episode; use the supported Cosmos3 profile mapping. |
| `parallel`, `maxConcurrency`, `submit --max-concurrency` | Parallel member stages inside one workflow, not dataset episodes. |
| Node-group size/autoscaling | Cluster capacity configured separately by the infrastructure owner. |

For the default one-GPU generation profile, 10 simultaneous generation stages
need 10 compatible schedulable GPUs; 20 need 20. Four GPUs per stage would
require 40 or 80 GPUs respectively. These are placement calculations, not
measured throughput. Also account for CPU/host-memory requests, node packing,
CPU stages/controllers, image/model startup, storage bandwidth and hosted-model
quotas. Twenty active workflows does not mean twenty GPUs are always busy.

Follow the PAIDF setup's [existing-cluster procedure](../../../workflows/guides/paidf-cosmos3.md#adopt-an-existing-cluster)
and select the registered project/context in the manifest. The batch passes
`--no-deploy-if-absent`. Verify node readiness, available resources, image
pulls, storage and model access first. Current submit-time capacity checks can
reject a wave before unschedulable pods exist: do not assume submission will
trigger scale-up from zero.

SkyPilot runs tasks on an existing Kubernetes cluster and Kubernetes schedules
their pods; see [SkyPilot Kubernetes](https://docs.skypilot.ai/en/latest/reference/kubernetes/index.html).
Nebius node-group autoscaling is configured separately with min/max node
counts and reacts to unschedulable pods; GPU groups also need CPU capacity
for cluster services. See [Nebius autoscaling](https://docs.nebius.com/kubernetes/node-groups/autoscaling).
Provider capacity and quotas still constrain provisioning; see
[Kubernetes autoscaling limitations](https://kubernetes.io/docs/concepts/cluster-administration/node-autoscaling/).

Qualify a customer episode end to end on the existing allocation, then measure
the requested 10- and 20-run settings. Record queue/startup/generation time, GPU
occupancy, accepted output count and failed-run recovery.

## Manifest and Python reference

Required fields: `apiVersion`, `batch_id`, `workflow`, `project`, and nonempty
`entries`. Optional: `infra`, `config_path` (SkyPilot global config, relative
to the manifest), `secret_env` (names only), and shared `vars`. Entry `vars`
override shared ones. Values are strings, integers or booleans; quote decimals.
Unknown fields, duplicate YAML keys/IDs/episodes, empty ranges, implicit PAIDF
starter selection, and overlapping declared outputs fail before submission.
`config.prefix` must contain `{{run.id}}`. Private code must also isolate
undeclared writes; the planner cannot inspect arbitrary executable commands.

`npa.sdk.workbench.workflow` exports `plan_batch`, `run_batch`, `batch_status`
and `check_paidf_input` over the same implementation:

```python
from pathlib import Path
from npa.sdk.workbench.workflow import plan_batch, run_batch

manifest = Path("paidf-batch.yaml")
plan = plan_batch(manifest)
result = run_batch(
    manifest, state_dir=Path("private-batch-state"), max_concurrent_runs=10,
)
```

For live acceptance, set `NPA_INTEGRATION_E2E=1`,
`NPA_E2E_BATCH_MANIFEST`, `NPA_E2E_BATCH_STATE_DIR`, and
`NPA_E2E_BATCH_CONCURRENCY`, then run
`npa/.venv/bin/python -m pytest npa/tests/e2e/test_workflow_batch_live.py -q`.
Use an operator-owned manifest with multiple real inputs and qualified existing
capacity. The test executes the batch and verifies that resuming the completed
local batch starts fresh standard resume clients with the original run IDs. The local tests separately exercise driver
concurrency, failed admission, retained child locks, and real media decoding;
they do not establish cloud throughput or external-service interoperability.
