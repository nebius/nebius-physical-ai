# Video variant sweep

Run source videos through visual description, prompt enhancement, parallel
Cosmos Transfer 2.5 generation, paired visual review, Postgres/MLflow tracking,
and accepted-dataset publication on Nebius through SkyPilot. The reference is
[video-variant-sweep.yaml](../testing/video-variant-sweep.yaml).

The two GPU workers process disjoint partitions of **every source × variant**
combination. Two workers describes the reference topology; it does not cap the
number of clips or variants. To change the worker count, change `workers` and
the worker states, their indices, and their declared output receipts together.
Every worker, including an empty partition, emits a receipt. The review stage
requires complete coverage before judging any candidate.

## Inputs and configuration

Use an operator-owned S3 source inventory with exact video object URIs:

```json
{
  "schema": "npa.video_sweep.sources.v1",
  "clips": ["s3://example-bucket/inputs/clip.mp4"]
}
```

The second input declares explicit sweep combinations. A user hint can come
from the existing agent chat or a manually authored manifest:

```json
{
  "schema": "npa.video_sweep.variants.v1",
  "variants": [
    {"hint": "Warm evening light", "seed": 7, "control": "edge", "control_weight": 1.0, "guidance": 3.0},
    {"hint": "Cool indoor light", "seed": 11, "control": "vis", "control_weight": 0.8, "guidance": 4.0}
  ]
}
```

Supported controls are `edge` and `vis`; this workflow does not fetch a depth
estimator or segmentation model. Seeds are nonnegative integers. Weights and
guidance must be finite and positive, and control weight must not exceed one.
Duplicate source URIs, duplicate video bytes, empty inputs, and duplicate
parameter combinations fail before generation.

`sources_uri` and `variants_uri` select these inventories. `bucket` and `prefix`
select the private run output. `root_uri` derives from them. `reasoner_model`
selects the exact hosted model used for source description and paired review;
`merge_model` selects the text model used for prompt enhancement. `samples`
defaults to eight evenly spaced decoded frames per clip, including its ends.
`threshold` defaults to 0.8. `source_overlay: true` installs the submitted NPA
source inside the pinned Transfer image.

The reference keeps `nvidia/Cosmos3-Super-Reasoner` explicit. The currently
verified account does not expose that model. Check your own model list and
select an available model explicitly; there is no automatic substitution.
The live hosted tests used `MiniMaxAI/MiniMax-M3`, which is a different model.
The merge default is `nvidia/Nemotron-3_5-Lightning`. Both IDs are checked before
GPU generation. Provider response model, request ID, token usage when returned,
and completion status are retained in the private artifacts.

## Run

Configure an existing Nebius Kubernetes cluster, project-owned S3 storage,
Hugging Face access to Transfer and its guardrails, and hosted inference access.
Run setup, submission, monitoring, and cleanup on the same Linux operator host;
macOS supports local authoring and planning but not the isolated SkyPilot API.
The reference requests one H200 per worker. Use the workflow GPU discovery and
image preflight commands to verify the selected cluster and image first. The
cluster must admit both workers concurrently: SkyPilot initializes networking
for the entire job group before starting its payloads.

Provide reachable Postgres and MLflow services before submission. Install
`npa[video-sweep]` when running the lineage stage locally; the workflow renderer
installs this extra for the lineage worker. Set these values privately:

| Environment variable | Meaning |
| --- | --- |
| `NPA_LINEAGE_POSTGRES_DSN` | Postgres connection string; role can create/write `npa_video_variants` |
| `MLFLOW_TRACKING_URI` | HTTPS tracking endpoint; loopback HTTP is supported for local tests |
| `MLFLOW_EXPERIMENT_ID` | Existing MLflow experiment ID |
| `MLFLOW_TRACKING_TOKEN` | Optional bearer token for the tracking service |
| `NEBIUS_TOKEN_FACTORY_KEY` | Hosted inference credential |
| `HF_TOKEN` | Credential with exact upstream payload access |

Keep credentials, prompts, clips, inventories, and run artifacts out of Git.
Source frames and hints are sent to the explicitly selected hosted service.
Only process inputs approved for that service.

```bash
npa workbench health preflight --project "$PROJECT" --checks s3,token_factory,hf
npa workbench health access --capability cosmos2 --json
npa workbench workflow validate-spec workflows/testing/video-variant-sweep.yaml --json
npa workbench workflow plan-spec workflows/testing/video-variant-sweep.yaml --run-id preview --waves --json
npa workbench workflow preflight-images workflows/testing/video-variant-sweep.yaml --json
npa workbench workflow submit workflows/testing/video-variant-sweep.yaml \
  --project "$PROJECT" --infra "k8s/$CONTEXT" --run-id "$RUN_ID" --runtime \
  --var "bucket=$BUCKET" --var "sources_uri=$SOURCES_URI" \
  --var "variants_uri=$VARIANTS_URI" --var "reasoner_model=$REASONER_MODEL" \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env NPA_LINEAGE_POSTGRES_DSN --secret-env MLFLOW_TRACKING_URI \
  --secret-env MLFLOW_EXPERIMENT_ID --secret-env MLFLOW_TRACKING_TOKEN
```

Omit the optional token flag for an MLflow service that does not use bearer
authentication. All services must be reachable from the worker pods; a local
SSH tunnel is useful for local integration tests but is not a pod endpoint.
The submit path uses the standard source staging and private secret channel.

## Artifacts and failure behavior

| Artifact beneath `root_uri` | Contents |
| --- | --- |
| `sources/` | Source snapshots named by their SHA-256 |
| `plan.json` | Exact source/variant inventory, descriptions, merged prompts, and inference provenance |
| `candidates/`, `workers/*.json` | Real generated videos and complete partition receipts |
| `review.json` | Scores, reasons, model provenance, and acceptance for every candidate |
| `lineage.json` | Receipt binding the exact review to committed Postgres and MLflow records |
| `dataset/manifest.json` | Accepted video objects, hashes, review digest, and lineage digest |
| `dataset/next-sources.json` | Source inventory for an explicitly launched subsequent run |

Transfer uses the actual upstream runtime with conditioning and content
guardrails enabled. It never falls back to reference augmentation. Inputs and
generated videos are decoded and fingerprinted; review and publication verify
the recorded bytes. Missing workers, incorrect partition coverage, changed
media, malformed judgments, unfinished model responses, and failed tracking
stop publication. A clip needs both `passed: true` and a score at least equal
to `threshold`. Rejected clips remain in private review/lineage evidence and
are excluded from the dataset. If no candidate passes, publication fails.

The visual gate assesses sampled frames; it cannot certify that every frame is
free of hallucinations. Source and generated samples use relative frame
positions; this is not dense optical-flow or physics validation.

The [MLflow REST API](https://mlflow.org/docs/latest/api_reference/rest-api.html)
records per-variant scores, acceptance, and identifiers. Postgres stores the
full lineage using
[parameterized Psycopg queries](https://www.psycopg.org/psycopg3/docs/basic/params.html).
Completed tracking replays reuse the stored MLflow run IDs. Postgres and MLflow
do not share a transaction: a failure between an MLflow write and the database
commit can leave an unreferenced tracking run. No dataset receipt is published
in that failure path.

Stage JSON artifacts reject conflicting rewrites. Use a fresh run ID for new
inputs or generation parameters. The feedback path deliberately produces a
new-run inventory instead of an unbounded workflow loop. Generic workflow
status, artifact discovery, and video viewing remain the existing agent UX;
this change does not add a new chat interface or an MLflow query tool.

For cleanup, cancel an unfinished run with `npa workbench workflow cancel`
before removing its task-owned resources. Keep accepted datasets and lineage
according to the operator's retention policy. Do not destroy shared services.

## Verification

The [readiness record](../testing/video-variant-sweep.readiness.json) separates
planning and component checks from a completed GPU workflow run.

```bash
npa/.venv/bin/python -m pytest npa/tests/workflows/test_video_sweep.py -q
NPA_INTEGRATION_E2E=1 NPA_VIDEO_SWEEP_REASONER_MODEL=MiniMaxAI/MiniMax-M3 \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_video_sweep_live.py -q
```

Live tests self-skip without their explicit prerequisites. The hosted test
uses a procedural identity pair and is not evidence of Transfer generation.
The Postgres/MLflow test writes real service records and verifies replay.
The S3 test uses `NPA_VIDEO_SWEEP_TEST_S3_URI` and deletes its exact test object.
The full GPU stage test requires `NPA_VIDEO_SWEEP_FULL_GPU=1`, the real Transfer
runtime, `NPA_VIDEO_SWEEP_SOURCES_URI`, `NPA_VIDEO_SWEEP_VARIANTS_URI`, and all
service credentials. The standard live-submit matrix registers the complete
runtime workflow; automated rotation waits for these operator-specific inputs
and services.
