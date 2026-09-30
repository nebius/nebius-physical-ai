# Video variant sweep

Run source videos through visual description, prompt enhancement, parallel
Cosmos Transfer 2.5 generation, paired visual review, Postgres/MLflow tracking,
and accepted-dataset publication on Nebius through SkyPilot. The reference is
[video-variant-sweep.yaml](../testing/video-variant-sweep.yaml).

This workflow ends at a versioned synthetic video dataset. Its unit of work is
a source video, an appearance prompt, and a generation parameter combination.
Downstream consumers choose how to use that dataset. Policy pretraining,
fine-tuning, train/test splits, simulator evaluation, and deployment are separate
workflows; they are not stages or acceptance criteria of this sweep.

The visual review gate checks augmentation fidelity: source objects, motion,
camera, and the requested appearance change. A passing sampled-frame judgment
does not establish robot task success or certify every frame as artifact-free.
The description model and video generator are separate components, and their
actual identities must be recorded separately. Selecting a Cosmos3 reasoner does
not change the Transfer 2.5 generation backend in this reference.

The two GPU workers process disjoint partitions of **every source × variant**
combination. Two workers describes the reference topology; it does not cap the
number of clips or variants. To change the worker count, change `workers` and
the worker states, their indices, and their declared output receipts together.
Every worker, including an empty partition, emits a receipt. The review stage
requires complete coverage before judging any candidate.

The present reference accepts explicit parameter combinations rather than
expanding parameter axes, and always describes sources and enhances hints.
It does not yet offer a direct, unmodified user-prompt mode. Worker receipts are
written after a whole partition completes; retrying a failed partition currently
regenerates that partition's clips. These limits matter for large sweeps.

## One-command operator kit and offline demo

After configuring the cluster and services described below, use the operator kit
from a checkout on a Linux host with `npa[video-sweep]` installed. It keeps the canonical YAML
as the stage graph and automates preflight, immutable source/variant inventories,
standard runtime submission, and verified HTML/MP4 export:

```bash
npa/.venv/bin/python -m npa.workflows.video_sweep.operator init \
  --config /private/video-sweep/sweep.json
# Fill in project, Kubernetes context, bucket, exact model IDs, and S3 source URIs.
# Supply the tracking and model credentials through the environment.
npa/.venv/bin/python -m npa.workflows.video_sweep.operator run \
  --config /private/video-sweep/sweep.json \
  --output-dir /private/video-sweep/demo
```

The generated configuration includes two editable variants, frame sampling,
threshold, GPU accelerator, and a fresh run ID. `check` runs the operator
preflight without submitting compute. `resume` uses the standard durable
`--resume-run` path with the same configuration. `export` rebuilds the demo from
an already reviewed run using only its configured project storage; it does not
submit compute or call a model. Select a new output directory for each export.
Use a new run ID for changed inputs or parameters.

If every candidate is rejected after tracking completes, `run` and `resume`
still return failure and publish no dataset. They export a review-only HTML/MP4
bundle so the result is inspectable. `export` also supports these tracked
rejections. It verifies that neither a dataset manifest nor a next-run inventory
exists; access errors do not count as absence. Incomplete tracking and failed
publication of accepted clips cannot use this path.

Exact-project S3 credentials come from NPA configuration. The required service
secrets are listed below; optional MLflow bearer token, private CA, and AWS session token are
forwarded only when present. Values never enter submission arguments.
The configuration and adjacent `.operator.log` are private files; keep both
outside the repository. Each run uses a separate adjacent `-runtime` directory
for SkyPilot state and stages the current source checkout. The helper does not provision the cluster, tracking
services, or gated model access. Both workers must fit concurrently, and tracking
endpoints must be reachable from their CPU pods. Operator preflight does not
prove that connectivity. The runtime waits without a per-wave deadline.

The `prefix` field defaults to `video-variant-sweep`; run objects live beneath
`s3://<bucket>/<prefix>/<run_id>/`.

The output directory contains:

- `index.html`: a standalone offline viewer with embedded clips, synchronized
  playback/scrubbing, candidate selection, stage evidence, and a threshold
  explorer that leaves recorded decisions unchanged.
- `demo.mp4`: a silent comparison film, with one full-length chapter per
  candidate, its source, measured score, and recorded acceptance.
- `summary.json`, silent preview MP4s, and JPEG posters: allowlisted metadata,
  original/preview media hashes, and the assets embedded in the viewer.

Export checks worker coverage, source/candidate hashes, accepted dataset bytes,
review/lineage bindings, and the next-run inventory before creating the final
output directory. It excludes prompts, raw review reasons, source paths, run IDs,
service addresses, and provider request identifiers. It removes container
metadata and audio. **Visible clip content remains**: an exported demo containing
private imagery still requires private handling. The procedural validation demo
contains no customer imagery. The HTML uses no external assets or requests.

The current export is a comparison of source and generated RGB clips. It does
not export conditioning maps or expose control weights and guidance alongside
each clip. Those inputs are needed for a complete visual investigation of a
parameter sweep; the current viewer should not be presented as that full surface.

The viewer reports artifact evidence separately from full workflow completion.
It marks all-rejected runs as held out, with no published dataset. The operator
wrapper does not turn component receipts into proof of a successful workflow.

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
The reference defaults to one H200 per worker. Set `accelerators` in the operator
configuration or pass `--var accelerators=RTXPRO6000:1` to select a verified
RTX PRO 6000 target for both generation workers. Use the workflow GPU discovery and
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
| `MLFLOW_TRACKING_CA_PEM` | Optional PEM certificate authority for private HTTPS; hostname and certificate verification stay enabled |
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

A single canonical submission completed preparation, two parallel RTX PRO 6000
Transfer workers, paired review, and tracking from CPU pods into real Postgres
and authenticated HTTPS MLflow services. Both workers produced distinct
121-frame videos with conditioning and content guardrails enabled. Both fresh
variants scored 0.30 against threshold 0.80: the judge found that the requested
lighting change was not preserved faithfully. Publication correctly failed
without creating a dataset or next-run inventory. The tracked rejection
export produced an offline HTML viewer and a 554-frame MP4; playback, mobile
layout, no-network behavior, and full video decoding passed.

The tracking stores were independently checked: two Postgres rows, two finished
MLflow runs, and two score/acceptance metric pairs. Earlier component execution
accepted one clip at 0.85 and rejected one at 0.30; its accepted-only S3
publication and next-run inventory passed. A successful single-submit run that
publishes accepted clips remains unverified. These results explicitly use
MiniMax-M3; the reference Cosmos3 model remains unavailable to the tested account.

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

The read-only live export test uses `NPA_VIDEO_SWEEP_DEMO_ROOT_URI` and
`NPA_VIDEO_SWEEP_DEMO_RUN_ID` to select a completed private run, verifies its
receipts and media, and decodes the exported MP4. It submits no compute:

```bash
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_video_sweep_live.py::test_export_published_run -q
```
