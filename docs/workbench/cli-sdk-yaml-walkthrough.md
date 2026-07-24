# Workbench CLI / SDK / YAML Walkthrough

> Audience: anyone calling an existing Workbench tool.
> Prerequisites: complete [getting-started.md](getting-started.md) first.

Every Workbench tool is a single containerized FastAPI service. That service is
the one source of truth for its behavior. The three things you actually use —
the CLI, the SDK, and SkyPilot YAML — are just three clients that reach the same
endpoints. Pick the client that fits where your code runs; the work performed is
identical.

This walkthrough uses the `detection-training` tool as the running example
because it exposes all three access modes cleanly and is the training stage of
the reference BDD100K pipeline. The same shape applies to `lerobot`, `lancedb`,
`sonic`, `cosmos`, and the other tools listed in
[../cli/workbench.md](../cli/workbench.md).

## The Mental Model

```text
                +-------------------------------+
   CLI  ─────►  |                               |
                |   Workbench FastAPI service   |
   SDK  ─────►  |   /health /train /eval        |  ──►  S3 artifacts
                |   /status /system-info /runs  |
   YAML ─────►  |                               |
                +-------------------------------+
```

| Access mode | Invocation | Where it runs | Best for |
| --- | --- | --- | --- |
| CLI | `npa workbench <tool> <command>` | Your shell / a SkyPilot task | Interactive runs, scripting, smoke tests |
| SDK | `npa.sdk.workbench.<tool>.<fn>(...)` | Your Python process | Notebooks, agents, custom orchestration |
| YAML | SkyPilot task that `curl`s the endpoint | The Nebius MK8s cluster | Multi-stage pipelines and sweeps |

All three either call a deployed service over HTTP or run the same shared
implementation in-process. They are not separate implementations. Behavior lives
in the service / shared layer; never expect one client to do something the
others cannot.

## Shared Endpoints

Each tool exposes a standard surface. For `detection-training`:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Readiness check; call before any state-changing request |
| `POST /train` | Start a Faster R-CNN training run |
| `GET /status` | Status for a run (`?run_id=...`) |
| `POST /eval` | Evaluate a checkpoint |
| `GET /system-info` | Runtime/image information, including the current image tag |
| `GET /runs` | List service-managed runs |

The CLI and SDK build the exact same JSON request bodies; the YAML builds them
with `jq`. Knowing the endpoint contract is enough to use any of the three.

## Prerequisites for This Walkthrough

```bash
export NPA_S3_BUCKET=<your-bucket>
export AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud
```

For the service and YAML paths, a deployed endpoint is also required:

```bash
npa workbench detection-training deploy \
  --output-path "s3://${NPA_S3_BUCKET}/detection-training/" \
  --namespace workbench \
  --gpu-type h100

export NPA_DETECTION_TRAINING_ENDPOINT=http://npa-detection-training.workbench.svc.cluster.local:8790
```

The deploy command prints the cluster-internal endpoint. From inside the cluster
(a SkyPilot task) use the `*.svc.cluster.local` form; the SDK and CLI use the
same value via `--endpoint` or `NPA_DETECTION_TRAINING_ENDPOINT`.

## 1. CLI

The CLI is the fastest way to drive a tool by hand or from a shell script. The
same command runs the work in-process by default, or against a deployed service
with `--service`.

Local in-process run (no deployed service required):

```bash
npa workbench detection-training train \
  --view bdd100k_rider_train \
  --lance-uri "s3://${NPA_S3_BUCKET}/bdd100k-pipeline/example-run/lancedb/" \
  --output-uri "s3://${NPA_S3_BUCKET}/detection-training/example-run/" \
  --epochs 10 \
  --batch-size 8 \
  --learning-rate 0.005
```

Service run (calls the deployed endpoint):

```bash
npa workbench detection-training train \
  --service \
  --endpoint "${NPA_DETECTION_TRAINING_ENDPOINT}" \
  --view bdd100k_rider_train \
  --lance-uri "s3://${NPA_S3_BUCKET}/bdd100k-pipeline/example-run/lancedb/" \
  --output-uri "s3://${NPA_S3_BUCKET}/detection-training/example-run/"
```

Both print a JSON object containing `run_id` and `status`. Follow up with:

```bash
npa workbench detection-training status \
  --service --endpoint "${NPA_DETECTION_TRAINING_ENDPOINT}" \
  --run-id <run-id-from-train>

npa workbench detection-training eval \
  --service --endpoint "${NPA_DETECTION_TRAINING_ENDPOINT}" \
  --checkpoint-uri "s3://${NPA_S3_BUCKET}/detection-training/example-run/model_final.pt" \
  --eval-view bdd100k_rider_train \
  --output-uri "s3://${NPA_S3_BUCKET}/detection-training/example-run/eval/"
```

Notes that generalize to other tools:

- `--input-path` / `--output-path` are accepted as aliases for the tool's
  input/output URIs so the command composes inside pipelines. Here they alias
  `--lance-uri` and `--output-uri`.
- Add `--output json` (the default for `train`/`eval`) for machine-readable
  output you can pipe into `jq`.
- Omit `--service` to run the shared implementation in your own process; pass
  `--service --endpoint ...` to hit a deployed service.

## 2. SDK

The SDK is the right client from Python — notebooks, agents, or a custom
orchestrator. Functions mirror the CLI commands and return typed Pydantic
response models.

Local (in-process) run:

```python
from npa.sdk.workbench import detection_training

resp = detection_training.train(
    view="bdd100k_rider_train",
    lance_uri="s3://my-bucket/bdd100k-pipeline/example-run/lancedb/",
    output_uri="s3://my-bucket/detection-training/example-run/",
    epochs=10,
    batch_size=8,
    learning_rate=0.005,
)
print(resp.run_id, resp.status)
```

Service-mode run against a deployed endpoint:

```python
from npa.sdk.workbench import detection_training

resp = detection_training.train(
    view="bdd100k_rider_train",
    lance_uri="s3://my-bucket/bdd100k-pipeline/example-run/lancedb/",
    output_uri="s3://my-bucket/detection-training/example-run/",
    mode="service",  # or service=True
    endpoint="http://npa-detection-training.workbench.svc.cluster.local:8790",
)

status = detection_training.status(run_id=resp.run_id, mode="service",
                                   endpoint="http://npa-detection-training.workbench.svc.cluster.local:8790")
print(status.status, status.epochs_completed)
```

Notes that generalize to other tools:

- `mode="local"` (the default) runs the shared implementation in-process;
  `mode="service"` (or `service=True`) makes an HTTP call.
- In service mode, `endpoint` defaults to the tool's endpoint environment
  variable (here `NPA_DETECTION_TRAINING_ENDPOINT`) when not passed explicitly.
- Errors raise typed exceptions (for this tool,
  `DetectionTrainingServiceError` for transport/HTTP failures and
  `DetectionTrainingValidationError` for bad local inputs). See
  [../sdk/errors.md](../sdk/errors.md).

## 3. YAML (SkyPilot)

YAML is the client for multi-stage pipelines and sweeps that run on the cluster.
A pipeline is a multi-document SkyPilot file; each task calls the tool's HTTP
endpoint with `curl`, building the request body with `jq` from `envs`. This is
exactly what the BDD100K reference pipeline does for the training stage.

Minimal single-task shape that mirrors the CLI/SDK `train` call above:

```yaml
name: detection-training-example
execution: serial
---
name: detection-train-rider
resources:
  cloud: kubernetes
  accelerators: H100:1
  cpus: 8
  memory: 32
setup: |
  set -euo pipefail
  command -v jq >/dev/null || (apt-get update && apt-get install -y jq)
envs:
  DETECTION_TRAINING_ENDPOINT: http://npa-detection-training.workbench.svc.cluster.local:8790
  VIEW_NAME: bdd100k_rider_train
  LANCE_URI: s3://<your-bucket>/bdd100k-pipeline/example-run/lancedb/
  TRAIN_OUTPUT_URI: s3://<your-bucket>/detection-training/example-run/
  TRAIN_EPOCHS: "10"
  TRAIN_BATCH_SIZE: "8"
  TRAIN_LEARNING_RATE: "0.005"
  BDD100K_LABEL_MAP: '{"person":0,"rider":1,"car":2,"truck":3,"bus":4,"train":5,"motor":6,"bike":7,"traffic light":8,"traffic sign":9}'
run: |
  set -euo pipefail
  curl -fsS "${DETECTION_TRAINING_ENDPOINT}/health"

  payload=$(jq -n \
    --arg view "${VIEW_NAME}" \
    --arg lance_uri "${LANCE_URI}" \
    --arg output_uri "${TRAIN_OUTPUT_URI}" \
    --argjson label_map "${BDD100K_LABEL_MAP}" \
    --argjson epochs "${TRAIN_EPOCHS}" \
    --argjson batch_size "${TRAIN_BATCH_SIZE}" \
    --argjson learning_rate "${TRAIN_LEARNING_RATE}" \
    '{view: $view, lance_uri: $lance_uri, output_uri: $output_uri, label_map: $label_map, epochs: $epochs, batch_size: $batch_size, learning_rate: $learning_rate}')

  curl -fsS -X POST "${DETECTION_TRAINING_ENDPOINT}/train" \
    -H 'Content-Type: application/json' \
    -d "${payload}"
```

Notes that generalize to other pipelines:

- Always `curl .../health` before any state-changing call.
- Use `--arg` for strings and `--argjson` for numbers/objects/booleans so JSON
  types survive into the request body.
- SkyPilot 0.12.2 does not expand same-block `envs` inside `image_id`, and does
  not self-reference `envs`. Keep committed YAMLs on explicit placeholders and
  render per-run values with a runner script.
- Dataset-specific config (like `label_map`) belongs in the YAML, not the tool.

For the full pipeline pattern, label-map injection, resources, and S3 path
conventions, see [../workbench-yaml-guide.md](../workbench-yaml-guide.md) and
the reference file
`npa/src/npa/workflows/skypilot/bdd100k-pipeline.yaml`.

## Same Work, Three Clients

The three calls below are equivalent — they produce the same `/train` request
against the same service:

| Client | Call |
| --- | --- |
| CLI | `npa workbench detection-training train --service --endpoint $E --view bdd100k_rider_train --output-uri s3://.../` |
| SDK | `detection_training.train(view="bdd100k_rider_train", output_uri="s3://.../", mode="service", endpoint=E)` |
| YAML | `curl -X POST "$E/train" -d "$payload"` |

## When to Use Which

| Situation | Use |
| --- | --- |
| Trying a tool by hand or in a shell script | CLI |
| Quick local run without deploying a service | CLI or SDK with `mode="local"` |
| Calling from a notebook, agent, or custom orchestrator | SDK |
| Composing multiple tools into a pipeline or a sweep | YAML |
| Passing data between stages | S3 URIs via `--input-path` / `--output-path` |

Tools never call each other directly for data transfer. Every stage reads its
input from an S3 URI and writes its output to an S3 URI, so any client can hand
work to the next stage.

## Next Docs

- [../cli/workbench.md](../cli/workbench.md): full list of workbench tools and commands.
- [../sdk/errors.md](../sdk/errors.md): typed SDK exceptions.
- [../workbench-yaml-guide.md](../workbench-yaml-guide.md): full pipeline YAML guide.
- [getting-started.md](getting-started.md): install, credentials, and first runs.
