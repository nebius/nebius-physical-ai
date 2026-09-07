# Workbench CLI / SDK / YAML Walkthrough

> Audience: anyone calling an existing Workbench tool.
> Prerequisites: complete [getting-started.md](getting-started.md) first.

Use `npa workbench <tool>` to run a tool, and `npa workbench workflow` to
compose tools into a workflow on Nebius. Behavior belongs in the tool's service
or shared implementation. The available access modes depend on the tool:
some call deployed HTTP services, some run inside a workload container, and
some expose local Python functions or wrappers around CLI callbacks.

This walkthrough uses `detection-training`, which trains Faster R-CNN detectors
from LanceDB materialized views. Its service, CLI, and typed SDK share request
schemas, but their options and completion behavior differ. Check a tool's
own documentation and `--help` before applying this example to another tool.

## Choose an access mode

| Access mode | Entry point | What to check |
| --- | --- | --- |
| CLI | `npa workbench <tool> <command>` | Actual flags, local versus service execution, output format, and whether the command waits for completion |
| Python | Tool-specific modules under `npa.sdk.workbench` or `npa.workbench` | Available functions, return types, and exceptions; coverage is not identical across tools |
| Workflow | `npa.workflow/v0.0.1` YAML with catalog `toolRef` stages | Input data, GPU resources, access requirements, runtime configuration, and artifact contracts |
| HTTP | A deployed tool service | Reachable endpoint, authentication, request schema, and completion/status contract |

The [toolRef catalog](npa-workflow-tool-catalog.md) identifies operations that
can be composed into workflows. A CLI command's existence does not imply a
matching SDK function, HTTP endpoint, or catalog entry.

## Detection-training prerequisites

Use an existing configured GPU cluster, a reachable authenticated
detection-training service, and a prepared LanceDB view. The
[BDD100K workflow](../../workflows/testing/bdd100k-pipeline.yaml)
shows the ingest, curation, train, and evaluation stages that produce and consume
those views. Merely naming `bdd100k_rider_train` does not create it.

Set the following to your prepared data and a new output prefix:

```bash
export NPA_LANCE_URI='s3://<your-bucket>/<prepared-lancedb-prefix>/'
export NPA_TRAIN_OUTPUT_URI='s3://<your-bucket>/<new-training-prefix>/'
export NPA_EVAL_OUTPUT_URI='s3://<your-bucket>/<new-evaluation-prefix>/'
export AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud
```

The endpoint must match the bucket's region. Supply S3 credentials through the
configured credential store or environment. Detection-training deployment uses
token authentication by default; supply `DETECTION_TRAINING_TOKEN` through your
private runtime environment before deploying and keep it available to clients.
See [service boundaries](../security/workbench-service-boundaries.md).

Run S3 credential preflight before submitting training:

```bash
npa workbench health preflight --checks s3 --json
```

The S3 health check proves listing access to the configured bucket. Confirm
that its configuration matches the input/output bucket and that the service
credentials also permit writing the output prefix.

`detection-training deploy` uses a minimal public `/readyz` probe. `/health`,
training, evaluation, run status, and artifacts require the service token. The
workload manifest contains a Kubernetes Secret reference; secret values are
provisioned separately through private files and stdin. The service becomes
Ready only after it opens its persistent run store. Deployment creates a retained
state PVC by default; use `--state-pvc` to supply an existing claim. Before
creating the Secret or applying the deployment, the CLI verifies the selected
project and exact saved cluster context, node selector/GPU product, output
prefix ownership, and write/read access using the same storage credential pair
that the service receives. `--project` selects project-scoped storage settings;
`--cluster-name` selects its exact saved context. Without that flag, the project
must identify one saved cluster. An explicit kubeconfig must match it.
GPU checks use currently free capacity. A `Recreate` replacement may also use
its existing allocation only when pod and ReplicaSet controller references
lead to the exact Deployment UID; matching labels alone grant no credit.
Unbound GPU requests or unreadable allocation ownership block submission.
`--dry-run` only renders the manifest and does not certify execution readiness.

The deployed service endpoint is cluster-internal. Clients inside that cluster
can use it directly. For a local notebook or shell, forward the service in
another terminal, adjusting the namespace and service name to match its deployment:

```bash
kubectl -n workbench port-forward service/npa-detection-training 8790:8790
```

Then configure the local client and verify the authenticated service:

```bash
export NPA_DETECTION_TRAINING_ENDPOINT=http://127.0.0.1:8790
npa workbench detection-training system-info --service
```

The service exposes `GET /health`, `GET /system-info`, `GET /runs`,
`POST /train`, `GET /status?run_id=...`, `GET /artifacts?run_id=...`,
`GET /artifacts/content?run_id=...&sha256=...`, and `POST /eval`. Its FastAPI schema is
available at `/openapi.json` on the same reachable endpoint. Consult the
[request schemas](../../npa/src/npa/workbench/detection_training/schemas.py)
for field definitions; these endpoints are specific to this service.

## 1. CLI: train, wait, and evaluate the produced checkpoint

For a view with numeric class labels:

```bash
npa workbench detection-training train \
  --service --wait \
  --view bdd100k_rider_train \
  --input-path "${NPA_LANCE_URI}" \
  --output-path "${NPA_TRAIN_OUTPUT_URI}" \
  --epochs 10 --batch-size 8 --learning-rate 0.005
```

For string labels, pass the dataset's category-to-index mapping with
`--label-map` on training. Checkpoints retain both the source category IDs and
the detector IDs: when a source map includes zero, all its IDs shift by one to
reserve detector zero for background. Numeric and string annotations use the
same mapping. Class count is inferred unless `--num-classes` is explicit; an
explicit count must include background and all mapped IDs. Evaluation reads the
checkpoint mapping by default and rejects a conflicting explicit map. Older
checkpoints keep the mapping with which their weights were trained.

Service training returns a `run_id` and an initial `queued` status unless
`--wait` is supplied. With `--wait`, the CLI polls until completion or failure
and verifies the reported epoch count and checkpoint pattern. A completed service
record requires verified checkpoint and metrics artifacts. Retain the JSON
response, including `run_id` and `artifacts`; each artifact records its role,
exact URI, media type, schema, size, and SHA-256.
If you submitted without waiting, monitor it with:

```bash
npa workbench detection-training status --service --run-id <run-id-from-train>
```

After completion, discover the actual checkpoint from the service's run
records and evaluate it:

```bash
npa workbench detection-training eval \
  --service --discover-checkpoint \
  --checkpoint-uri "${NPA_TRAIN_OUTPUT_URI}" \
  --eval-view bdd100k_rider_train \
  --input-path "${NPA_LANCE_URI}" \
  --output-path "${NPA_EVAL_OUTPUT_URI}" \
  --write-canonical-metrics
```

`--discover-checkpoint` searches the latest completed service run under the
training prefix and selects its verified final checkpoint artifact. Run status
and artifact records survive service restarts on the retained state PVC.
Authenticated clients can read artifact bytes through `/artifacts/content`; the
service checks their hash and rejects missing or modified objects. The configured
output prefix bounds both retrieval and service writes. Deploy one worker per
state volume: a process lock prevents concurrent service owners, and SQLite
transactions serialize run updates. After a crash, unfinished records become
`interrupted` and retain their progress and identity. They do not automatically
resume. `deploy --destroy` retains the PVC; remove it explicitly only when its
run history is no longer needed. Evaluation runs also persist their typed result
and metric artifacts under `eval_run_id`; `/status` and `/artifacts` accept either
training or evaluation IDs. An identical completed evaluation request returns
its stored result. Active, failed, or interrupted evaluations retain their
identity and require inspection before a new request. Evaluation writes the canonical `metrics.json` when
`--write-canonical-metrics` is set and validates numeric evaluation metrics.
Missing evaluation dependencies fail explicitly; undefined COCO metrics retain
their `-1` sentinel rather than becoming zero. A low measured score does not
mean the workload infrastructure failed. Evaluating the training view checks the journey; use a held-out view to measure
generalization.

For this tool, `--input-path` aliases `--lance-uri` and `--output-path` aliases
`--output-uri`. Train and eval emit JSON by default. Other tools can use
`--json`, `--output`, or `--output-format`; inspect the relevant command help.

Omit `--service` to train synchronously in the current GPU runtime. The local
CLI, SDK, and API preserve an absent or explicit label map through the same
request schema and training implementation. Direct execution returns the
verified artifact manifest; persistent `/status` records belong to service runs.

## 2. SDK: inspect the concrete Python contract

The detection-training SDK returns Pydantic response models. This is an
alternative submission of a new training run, using the same prepared numeric
label view and runtime environment as above:

```python
import os
from npa.sdk.workbench import detection_training

response = detection_training.train(
    view="bdd100k_rider_train",
    lance_uri=os.environ["NPA_LANCE_URI"],
    output_uri=os.environ["NPA_TRAIN_OUTPUT_URI"],
    mode="service",
    endpoint=os.environ["NPA_DETECTION_TRAINING_ENDPOINT"],
    epochs=10,
)
print(response.run_id, response.status)

status = detection_training.status(run_id=response.run_id, mode="service")
print(status.status, status.epochs_completed)
```

A successful service-mode `train()` response acknowledges submission. Poll
`status()` to a terminal state before consuming checkpoints. The SDK has no
`wait` argument. Both `train()` and `eval()` accept an optional `label_map`. Local SDK
execution calls the shared training function synchronously and requires its
engine dependencies and accessible data in the calling environment.

Import `DetectionTrainingServiceError` and
`DetectionTrainingValidationError` from
`npa.sdk.workbench.detection_training`. Pydantic request validation can also
raise `pydantic.ValidationError`. The separate [SDK error reference](../sdk/errors.md)
covers the serverless client rather than all Workbench exceptions.

These return and error contracts do not generalize to every module:

| Module | Implemented Python surface |
| --- | --- |
| `npa.sdk.workbench.detection_training` | Typed `train`, `eval`, and `status` functions |
| `npa.sdk.workbench.workflow` | Durable `status`, `logs`, `artifacts`, and `runs`; dictionaries, strings, and lists |
| `npa.orchestration.npa_workflow` | Specification loading, validation, and execution planning |
| `npa.workbench.lerobot`, `npa.workbench.genesis` | CLI callback wrappers; may print output and raise CLI exits |
| `npa.sdk.workbench.sonic` | Shared export helpers plus CLI wrappers for train/eval |

## 3. YAML: compose registered tool operations

Use the maintained
[BDD100K npa.workflow specification](../../workflows/testing/bdd100k-pipeline.yaml)
for the full ingest-to-evaluation journey. Its training stages reference
`workbench.detection_training.train_rider`, `train_nighttime`, and
`train_distant`; these invoke the real CLI service path and wait for training.

From the repository root, inspect the unchanged reference before preparing
private inputs and submission parameters:

```bash
npa workbench workflow validate-spec \
  workflows/testing/bdd100k-pipeline.yaml
npa workbench workflow plan-spec \
  workflows/testing/bdd100k-pipeline.yaml
npa workbench workflow submit --help
```

Validation and planning do not run the tools or prove that data, model access,
images, or GPU capacity are ready. Follow the
[workflow guide](npa-workflow-guide.md) for submission and the
[workflow catalog](../../workflows/README.md) for
workload-specific setup. NPA renders the workflow into SkyPilot tasks; the
current BDD100K reference is an `npa.workflow` state machine, not a raw
multi-document SkyPilot `curl` pipeline.

Keep the resolved run ID and durable state prefix returned by submission.
Inspect `npa workbench workflow status`, `logs`, and `artifacts` with the
matching project or durable S3 options. For a successful PAIDF run,
`load-artifact` retries the final load into the configured agent viewer; it is
not a generic viewer command for every workflow. S3 paths connect tool outputs
to downstream inputs; inspect the produced artifact as well as the run status.

## Next docs

- [Workbench](README.md): choose a tool or end-to-end journey.
- [Workflow guide](npa-workflow-guide.md): planning, runtime, and durable recovery.
- [ToolRef catalog](npa-workflow-tool-catalog.md): supported workflow operations.
- [Troubleshooting](troubleshooting/known-footguns.md): setup and runtime recovery.
- [Workbench command reference](../cli/workbench.md): tool groups; use each command's help for options.
