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

**Current deployment limitation:** `detection-training deploy` defaults to token
authentication, but its generated readiness probe requests `/health` without
authentication. The service rejects that probe with HTTP 401, preventing the
default deployment from becoming Ready. The probe needs an implementation fix;
do not disable service authentication to work around it. The examples below
assume an already reachable authenticated service and do not establish that a
fresh deployment works.

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
`POST /train`, `GET /status?run_id=...`, and `POST /eval`. Its FastAPI schema is
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
`--label-map` on **both** train and eval. Real BDD100K categories differ from
some synthetic fixtures; use the mapping matching your prepared view.

Service training returns a `run_id` and an initial `running` status unless
`--wait` is supplied. With `--wait`, the CLI polls until completion or failure
and verifies the reported epoch count and checkpoint pattern. Retain the JSON
response, including `run_id`, `checkpoint_uri_pattern`, and `metrics_uri`.
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
training prefix and resolves its final epoch checkpoint. It requires the
service's in-memory `/runs` record, which does not survive a service restart.
Retain the exact checkpoint URI for later evaluation. Checkpoints use
`<output-prefix>/<run-id>/checkpoints/epoch_<epoch>.pt`; do not assume a
`model_final.pt` filename. Evaluation writes the canonical `metrics.json` when
`--write-canonical-metrics` is set and validates numeric evaluation metrics.
Evaluating the training view checks the journey; use a held-out view to measure
generalization.

For this tool, `--input-path` aliases `--lance-uri` and `--output-path` aliases
`--output-uri`. Train and eval emit JSON by default. Other tools can use
`--json`, `--output`, or `--output-format`; inspect the relevant command help.

The current local CLI training path fails before training with
`TypeError: train() got an unexpected keyword argument 'label_map'`: it forwards
a schema field that the SDK does not accept. The service commands above avoid
that path. This is an implementation limitation, not a missing GPU dependency.

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
`wait` argument, and its current `train()` signature does not expose
`label_map`; use the CLI service path for explicit label mappings. Local SDK
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
