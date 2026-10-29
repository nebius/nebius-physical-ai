# Source selection, paired iteration and attribution

This workflow uses only the supplied HTTPS inference API after source media
has been staged in a registered S3 input connection. Python 3.10+ and this
portable package are enough for discovery, admission, polling, artifact
retrieval and usage export. Neither a private repository, Nebius CLI, SSH,
model weights nor direct GPU access is needed. An independently reachable
operator-supplied API origin and appropriate API grants are prerequisites;
a packaged client or an internal test does not establish public availability.

## Prepare independent customer access

Each customer uses their own API key, private working directory, connection
IDs and source/output objects. The operator binds that key to the actual
customer and traffic class; customers cannot set identity or classification
through requests. Do not count synthetic customer demos as production adoption. Keys for two genuinely separate customers must have separate
owners; two keys under one owner share that owner's resources.

From the `skills/tools/nebius-stream-beta` directory (or its portable copy):

```bash
umask 077
export WMH_WORK_DIR="$HOME/.local/share/cosmos-client/customer-one"
mkdir -p "$WMH_WORK_DIR"
chmod 700 "$WMH_WORK_DIR"
export WMH_BASE_URL='https://operator-supplied-origin.example'
export WMH_API_KEY_FILE="$WMH_WORK_DIR/api-key.txt"
export WMH_S3_CREDENTIAL_FILE="$WMH_WORK_DIR/output-s3.json"
python3 scripts/cosmos_client.py init-secrets
python3 scripts/cosmos_client.py discover --output "$WMH_WORK_DIR/contract.json"
```

Substitute the supplied HTTPS origin locally. `init-secrets` reads hidden
terminal input; never pass credential values through arguments, environment
variables, chat, or JSON examples. If the operator supplied an existing private
key file, use its path and `init-storage-secret` for a new S3 credential file.
For a separate input provider, change `WMH_S3_CREDENTIAL_FILE` to a distinct
unused file and run `init-storage-secret` again, then register its input
connection. The inference key file remains the same. The helper never
silently replaces existing secret files. Use a trusted operator CA only when
that is the supplied access contract; never disable certificate verification.

Register output and input using [the storage request guide](requests.md).
Input may be another approved public S3-compatible provider. Output must pass
server verification for Nebius's canonical regional endpoint and the actual
bucket region. Use the appropriate S3 credential file for each registration.
Keep source and output prefixes separate. To inspect a connection, including
a registration whose response was lost:

```bash
python3 scripts/cosmos_client.py connection --id customer-output --output "$WMH_WORK_DIR/output-connection.json"
```

## Choose what the source actually represents

For a normal MP4, retain its original byte hash and verify that the running
model's full-source contract accepts its decoded dimensions, frame rate and
duration. Use the source dataset and episode chosen for the evaluation.
Do not silently truncate a long source to fit a prefix-continuation profile.
`/v1/videos` prefix continuation conditions only its documented first source
frames; `/v1/augmentations` is the separate full-source structural-control
path. Discover each model's exposed, implemented, qualified controls first.

**Native LeRobot dataset import is not available in this API.** A LeRobot
dataset contains episode metadata, observations, timestamps and often actions
and state. Its camera video, or an MP4 rendered from dataset frames, is only
a media derivative. Inventorying that video does not load the dataset tables,
retain action/state alignment, preserve multi-camera synchronization, produce
a valid LeRobot dataset, or establish downstream training suitability.

If a dataset-derived experiment is acceptable, materialize the selected
complete episode/camera before API use with the dataset's own reader. Retain
original dataset revision, episode/camera, metadata/table hashes, frame and
timestamp mapping, transformation settings, original and derived byte hashes,
and any resizing or frame-rate conversion. Mark this preprocessing separately
from API inference. A whole episode rendered at a different FPS is still a
derivative. Do not label it native dataset ingestion. If the task requires
native LeRobot semantics, report that capability gap instead of substituting
an extract. Uploading local media to input storage is also a prerequisite;
this client and the inference API do not provide a local-file upload route.
The customer can stage it using their provider's console or existing tooling.

## Baseline and tuned full-source requests

Discover the live model matrix and schema before using this example. Make
`baseline-request.json` private (mode 0600), substituting the exact owner-bound
input/output and explicitly chosen model. The model must expose full-source
`edge` augmentation; this example does not assert availability on every host.

```json
{
  "model": "nvidia/Cosmos3-Super",
  "input": {"connection_id": "customer-input", "uri": "s3://CUSTOMER_INPUT/episodes/source.mp4"},
  "output": {"connection_id": "customer-output", "prefix": "s3://CUSTOMER_OUTPUT/generated/baseline"},
  "augmentation": {"prompt": "Preserve the visible robot motion while changing the lighting.", "structural_control": "edge", "variants": 1},
  "generation": {"seed": 101, "num_inference_steps": 20},
  "evaluation": {"mode": "technical_report", "rubric_id": "paired-source-iteration", "criteria": ["complete_decode", "timeline_coverage"]},
  "quality_profile": "robot-motion-preservation"
}
```

```bash
python3 scripts/cosmos_client.py submit --route /v1/augmentations --request "$WMH_WORK_DIR/baseline-request.json" --state "$WMH_WORK_DIR/baseline-state.json"
python3 scripts/cosmos_client.py resume --state "$WMH_WORK_DIR/baseline-state.json" --output-dir "$WMH_WORK_DIR/baseline-results"
```

For the tuned trial, copy the request locally and change only the intended
supported generation setting (for example `num_inference_steps` from 20 to
35) and its distinct output prefix. Retain the same model, source bytes, seed,
prompt, control and evaluation rubric for a paired comparison. Save it as
`tuned-request.json`, with a new `tuned-state.json`, and run the same two
commands. A new logical request needs its own state; retries of the same
request must keep its original state. More steps are an experimental variable,
not a promised quality improvement. Fixed I2V/video profiles have different
schemas and cannot inherit arbitrary augmentation settings.

`resume` writes private `job.json` and retrieves every published artifact with
SHA-256 verification. Inspect `manifest.json`, source/generated media,
comparison artifacts and `quality.json` when present. A requested
`technical_report` reports complete decode and full timeline coverage; read
each criterion's boolean. A report's existence or `technical_report_available`
status is not a semantic pass. These checks do not score motion identity,
contact, physical correctness, action labels, perceptual fidelity or training
benefit; they are report-only, not a delivery gate. A false criterion or missing
report must remain visible. The service has no automatic semantic acceptance,
hyperparameter optimizer, retraining, or native dataset export. Choose an
iteration only after separately reviewing source/baseline/tuned media and
recording your actual rubric observations. Keep failures and unchanged results.

## Source inventories and batches

Inventories discover already-staged eligible media under one owner-bound
input connection. They preserve source object identity for later admission;
they are not a dataset importer or an inference invocation. Example
`inventory-request.json`:

```json
{"connection_id": "customer-input", "prefix": "", "page_size": 100}
```

An empty prefix selects the connection's registered read scope. Narrow it
when appropriate. Alternatively set `manifest_uri` to an in-scope S3 object
containing one plain S3 URI or `{"uri":"s3://BUCKET/KEY"}` per line; do not
combine a nonempty prefix and manifest selector.

```bash
python3 scripts/cosmos_client.py submit --route /v1/source-inventories --request "$WMH_WORK_DIR/inventory-request.json" --state "$WMH_WORK_DIR/inventory-state.json"
python3 scripts/cosmos_client.py inventory-resume --state "$WMH_WORK_DIR/inventory-state.json" --output "$WMH_WORK_DIR/inventory-status.json"
python3 scripts/cosmos_client.py inventory-objects --state "$WMH_WORK_DIR/inventory-state.json" --output-dir "$WMH_WORK_DIR/inventory-objects"
```

Resume until `state=complete`, then inspect eligible/rejected/failed counts and
all saved object pages. Discovery does not guarantee that every selected
video passes its later model decode/shape contract. Do not discard exclusions.

For an augmentation batch, copy the augmentation template above, replace its
`input` with `inventory_id` from the retained inventory state, and choose a
new output prefix. The actual batch adapter currently accepts Super
full-source augmentation; a qualified single-job Nano path does not imply a
Nano inventory-batch adapter. An input `control` object is not part of this
batch schema. Check live schemas rather than extrapolating support.

```bash
python3 scripts/cosmos_client.py batch-preflight --request "$WMH_WORK_DIR/batch-request.json" --output "$WMH_WORK_DIR/batch-preflight.json"
python3 scripts/cosmos_client.py submit --route /v1/augmentation-batches --request "$WMH_WORK_DIR/batch-request.json" --state "$WMH_WORK_DIR/batch-state.json"
python3 scripts/cosmos_client.py resume --state "$WMH_WORK_DIR/batch-state.json" --output-dir "$WMH_WORK_DIR/batch-results"
```

Preflight admits no GPU work. Submission creates one durable child job per
eligible source, with no implicit additional variants. `resume` retains
`batch.json` and checksum-verified artifacts in child job directories; a
terminal batch containing failed children exits unsuccessfully and preserves
successful children. The API batch groups independently scheduled jobs, not
native tensor/model batching. A separate `/v1/batches` accepts explicit Nano
T2V items of shape `{"custom_id":"trial-a","request":VIDEO_REQUEST}` inside
`{"items":[...]}`; it is not the source-inventory batch endpoint. Both batch
forms require `batches:write` plus their generation scope and model grants.

## Prove your own attribution

Record UTC campaign start and end instants. Every customer can inspect their
own usage with `jobs:read`; cross-owner reporting requires an operator grant.
The helper always sends the authenticated owner filter, even for an operator
key. For a demo campaign, explicitly request synthetic traffic:

```bash
python3 scripts/cosmos_client.py usage-summary --include-synthetic --since 2026-10-27T00:00:00Z --until 2026-10-28T00:00:00Z --output "$WMH_WORK_DIR/usage-summary.json"
python3 scripts/cosmos_client.py usage-export --include-synthetic --since 2026-10-27T00:00:00Z --until 2026-10-28T00:00:00Z --state "$WMH_WORK_DIR/usage-state.json" --output-dir "$WMH_WORK_DIR/usage-pages"
```

Replace those illustrative timestamps with the actual interval. `since` is
inclusive and `until` exclusive. Timestamps must include a timezone. Without
`--include-synthetic`, only explicitly attributed production traffic appears;
demo, benchmark, internal-test and legacy unattributed traffic are excluded.
No production rows means no observed production adoption in that interval,
not proof that historical customers never existed.

Each private event page retains schema version, event identity, cursor and
one fixed high watermark. `usage-export` saves a page before advancing its
private recovery state. Rerun with the same state, directory and filters after
a disconnect/crash. A completed export is a stable snapshot; it does not grow
with later polling or its own export requests. For later events use a new
state and empty directory with `--after` set to the previous completed
`cursor` and the same intended filters. Keep the earlier pages. Page sizes
are 1–1000; pagination does not cap total records. The summary is convenience
aggregation (bounded groups); the paginated event export is the complete
source for analysis over a larger interval.

Match the retained job IDs to `job_admitted` and `job_delivered` events and
check the snapshotted customer/key IDs, name/type and traffic class. Count
unique admissions and delivered jobs separately from `request_started`,
HTTP completions, `admission=idempotent_replay`, polling and artifact reads.
A delivered event means successful server publication, not semantic approval
or a unique user download. Preserve null measured bytes/times as unknown;
never turn them into measured zeros or infer GPU time from HTTP latency.
Do not send owner IDs, private object locations or raw export pages to a
public report. Give only authorized, sanitized evidence to other people.
