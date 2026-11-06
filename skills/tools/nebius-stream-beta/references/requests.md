# Storage and generation requests

The HTTP API accepts JSON, not YAML. Use the privately supplied HTTPS origin
as `WMH_BASE_URL`; this guide deliberately contains no deployment address.
Run `discover` first to obtain the authenticated model capabilities and OpenAPI
schema. These examples describe request shapes, not current endpoint health.

Create these JSON files privately. Substitute actual customer values locally;
no secrets belong in these configuration files.

## Register storage once

Save this client configuration as `output-storage.json`. It uses one Nebius
bucket for both input and output, with separate prefixes:

```json
{
  "connection_id": "customer-output",
  "endpoint_url": "https://storage.REGION.nebius.cloud",
  "region": "REGION",
  "bucket": "CUSTOMER_BUCKET",
  "addressing_style": "path",
  "read_prefix": "inputs",
  "write_prefix": "generated",
  "purpose": "input_output"
}
```

Use the bucket's actual documented Nebius region. The server verifies its
location; do not invent a region or bypass a failed check. `input_output`
allows this connection to serve both roles with separate prefixes. For a
foreign S3 source, register a separate connection with `purpose: "input"`,
its real endpoint/region/read prefix and the corresponding credential file.
Deployment input allowlists still apply. Output remains Nebius only.

Register it with the bundled client:

```bash
python3 scripts/cosmos_client.py register --config "$WMH_WORK_DIR/output-storage.json"
```

The client reads `WMH_S3_CREDENTIAL_FILE` and adds its contents under
`credentials` when sending `POST /v1/storage/credentials`. That private file
contains this JSON shape; replace placeholders through local hidden input,
not command arguments or a committed request file:

```json
{
  "aws_access_key_id": "<S3_ACCESS_KEY_ID>",
  "aws_secret_access_key": "<S3_SECRET_ACCESS_KEY>"
}
```

Include `aws_session_token` only when the provider issues a session token.
For a custom HTTP client, the registration body is the full storage config
above plus `"credentials": { ... }` containing these credential fields.
Inject that object in memory from the protected credential file. Registration
requires `Authorization: Bearer <INFERENCE_API_KEY>` and
`Content-Type: application/json` over verified HTTPS. Credentials are sent
only during registration or rotation; generation requests reference the
registered connection ID. The server verifies the Nebius endpoint and bucket
region before admitting output.

## Submission, polling and completion

Generation uses durable asynchronous jobs. Send a validated JSON request to
`POST /v1/videos`, `/v1/images`, `/v1/augmentations` or `/v1/transfers` with the bearer and
`Idempotency-Key` headers shown below. Successful admission returns HTTP
**202 Accepted** with a job record. This is a simplified first-admission
response, not a completed result:

```json
{
  "job_id": "job_example",
  "accepted": true,
  "status": "queued",
  "status_url": "/v1/jobs/job_example"
}
```

Retain the job ID immediately. `accepted` indicates whether this submission
created a new job: an identical idempotent replay can return `accepted: false`
with the original job and its current status. It does not indicate rejection.
Reusing the key with a different request returns HTTP 409.

| Operation | Response and next step |
| --- | --- |
| `POST /v1/videos`, `/v1/images`, `/v1/augmentations` or `/v1/transfers` | HTTP 202 acknowledges admission; retain `job_id` and `status_url`. |
| `GET /v1/jobs/{job_id}` | Read the current job record; repeat while its status is active. |
| `GET /v1/jobs/{job_id}/artifacts` | After `succeeded`, retrieve the published artifact manifest. |
| `GET /v1/jobs/{job_id}/artifacts/{artifact_name}` | Retrieve a manifest-listed file and verify its SHA-256. |

Use the same supplied HTTPS origin and bearer header for every read. A status
GET returning HTTP 200 means the read succeeded; inspect the JSON `status` to
learn whether generation succeeded. A typical progression is `queued` →
`running` → `publishing` → `succeeded`; other staging, dispatch and recovery
states are possible. Preserve failed, cancelled, `publication_retryable` and
`needs_reconciliation` records instead of treating every non-success as active.
Advisory `quality_status` is separate from job delivery status.

### Wait with the portable client

The bundled `submit` command saves the run state and prints a small
acknowledgement with `status: "accepted"`, then exits. That CLI summary is
distinct from the server's job status. To read status once and save the full
record privately:

```bash
python3 scripts/cosmos_client.py status --state "$WMH_WORK_DIR/run.json" --output "$WMH_WORK_DIR/job-status.json"
```

To wait for completion and download the published artifacts:

```bash
python3 scripts/cosmos_client.py resume --state "$WMH_WORK_DIR/run.json" --output-dir "$WMH_WORK_DIR/results"
```

`resume` performs repeated status GETs, saves the latest private `job.json`,
and sleeps three seconds between active-state reads. On success it downloads
and checksum-verifies every published artifact before returning. On a failed,
cancelled, recovery-required or unknown state it exits unsuccessfully and
retains the status file. Transport or integrity errors also stop the command;
successful admission does not guarantee delivery.

There is no overall waiting deadline or `--timeout` option on `resume`.
Individual HTTPS requests use a separate 60-second timeout. Interrupting the
command or losing connectivity does not cancel the server job. Rerun `resume`
with the same state after an interruption; it does not resubmit generation.
If submission ended before retaining an ID, rerun `submit` with the identical
request and state to reconcile its original idempotency identity first.

Both batch forms also use asynchronous admission and polling: `/v1/batches`
uses `GET /v1/batches/{batch_id}`, and `/v1/augmentation-batches` uses
`GET /v1/augmentation-batches/{batch_id}`. The same `resume` command follows
the retained route and saves `batch.json`; inspect child outcomes, since a
`completed` batch can contain failures. See the
[batch guide](iteration.md#source-inventories-and-batches) for recovery.

### Optional separately supplied Python SDK

The portable client requires no SDK. If the operator separately supplies the
private `cosmos_sdk` distribution, it also offers completion waiting after
`.create(...)` has returned a job:

```python
job_id = job["job_id"]  # Persist the ID before waiting.
completed = client.jobs.wait(job_id, poll_interval=2, timeout=None)
```

`wait()` polls internally and returns a `succeeded` job. Failure, cancellation
or a recovery-required state raises `JobError` carrying the job ID and record;
HTTP/transport errors can also end the wait. `timeout=None` is the default and
sets no overall deadline. Supplying a positive timeout in seconds stops local
polling with `TimeoutError` without cancelling the job. The client's per-HTTP
request timeout is separate. `AsyncCosmos` supports
`await client.jobs.wait(job_id)` so other application work can continue.
Obtain this SDK only through the operator's authorized distribution, not an
unverified public package with a similar name.

Both the portable client's `resume` and the SDK's `wait()` use repeated HTTP
requests. The current HTTP contract does not expose a synchronous
`wait=true` request field, a held-open submission response or a long-poll
endpoint. Do not add undocumented fields; discover the running contract
before relying on any future extension.

## Full-source video-to-video

Save this complete Super request as `request.json`. Its source must already
exist in the registered read prefix. Both connection IDs refer to the single
`input_output` connection above; for a separate source provider, change only
`input.connection_id` and `input.uri` to that registered input scope.

```json
{
  "model": "nvidia/Cosmos3-Super",
  "input": {
    "connection_id": "customer-output",
    "uri": "s3://CUSTOMER_BUCKET/inputs/source.mp4"
  },
  "output": {
    "connection_id": "customer-output",
    "prefix": "s3://CUSTOMER_BUCKET/generated/run-001"
  },
  "augmentation": {
    "prompt": "Preserve the robot, objects, camera viewpoint and motion. Change the scene lighting to warm late-afternoon sunlight.",
    "structural_control": "edge",
    "variants": 1
  },
  "generation": {
    "seed": 101,
    "num_inference_steps": 35,
    "guidance_scale": 3.0,
    "control_guidance": 1.5,
    "flow_shift": 10.0
  },
  "evaluation": {
    "mode": "technical_report",
    "rubric_id": "lighting-test",
    "criteria": ["complete_decode", "timeline_coverage"]
  }
}
```

Submit and recover it through the client, which supplies authentication and
persists the idempotency key without exposing the secret:

```bash
python3 scripts/cosmos_client.py submit --route /v1/augmentations --request "$WMH_WORK_DIR/request.json" --state "$WMH_WORK_DIR/run.json"
python3 scripts/cosmos_client.py resume --state "$WMH_WORK_DIR/run.json" --output-dir "$WMH_WORK_DIR/results"
```

For a custom client, send that JSON to the supplied origin with this HTTP
contract. Header values below are placeholders, not shell commands:

```text
POST /v1/augmentations
Authorization: Bearer <INFERENCE_API_KEY>
Content-Type: application/json
Idempotency-Key: <UNIQUE_LOGICAL_REQUEST_ID>
```

Reuse the same request and idempotency key after an ambiguous
submission. A changed configuration is a new logical request and needs a new
key, client state and output prefix. Customer name, type and owner are derived
from the API key; do not add them to the request body. The same route selects
Super or Nano explicitly, without a separate model-specific origin.

For **Nano**, set `model` to `nvidia/Cosmos3-Nano` and retain the same request
structure and example sampling settings. Its qualified full-source edge path
requires a fully decodable **832 × 480, 24 FPS** source with **at least six
frames**. Check the current capability matrix before submission. This is
distinct from Nano's fixed text-to-video and prefix-continuation profiles;
do not add a `generation.profile` field to this augmentation body.

### Sampling settings and boundaries

| Field | Example | Meaning and accepted schema |
| --- | --- | --- |
| `seed` | `101` | Integer random seed; retain it for a paired settings comparison. |
| `num_inference_steps` | `35` | Denoising iterations, integer 1–200. More steps increase computation without guaranteeing better quality. |
| `guidance_scale` | `3.0` | Text-prompt guidance strength, number 0–20. |
| `control_guidance` | `1.5` | Structural-control guidance strength, number at least zero; stronger guidance does not guarantee source preservation. |
| `flow_shift` | `10.0` | Sampling schedule parameter; retain this example value initially. |

These are explicit example settings, not a claim that every model uses the
same omitted-field defaults or that every schema-valid combination is
quality-qualified. Keep settings finite and consult the running contract.
For a paired 20-versus-35-step trial, change only `num_inference_steps` plus
the request identity/output prefix; see the [iteration guide](iteration.md).

Only source-derived `edge` control is qualified in this full-source example;
do not supply a separate `control` object. `variants` currently accepts only
`1`. Resolution, frame rate, chunking, GPU allocation and scheduling are
service-managed on this route; clients cannot override them through this
JSON. Guardrails are enforced by the service, not a customer toggle.
Conditioning covers the source timeline but does not guarantee robot identity,
geometry, contact or motion preservation. The optional `evaluation` object
requests report-only decode and timeline checks, not semantic acceptance.

## Full-length structural transfers

Use `POST /v1/transfers` when you need uploaded controls, multiple control
types, individual weights or a negative prompt on either Nano or Super. Its
positive `prompt`, `negative_prompt` and `controls` are top-level fields; do
not put the legacy `augmentation` object in this request.

First inspect the authenticated `structural_transfer` capability row. It must
expose an implemented, qualified profile with enabled guardrails, each
requested control and its exact sorted combination, such as `depth+edge` or
`blur+depth+edge+seg`. Unequal weights need `control_weights_supported: true`;
a nonempty negative prompt needs `negative_prompt_supported: true`. The helper
checks those capabilities and verified output storage before submitting.

The [complete JSON example](../examples/structural-transfer.json) uses the
earlier registered `customer-output` connection with purpose `input_output`,
four uploaded controls and weights 1:3:2:0.5. Copy it privately, replace
`CUSTOMER_BUCKET` and the object keys with actual staged media, and retain an
explicit model. Change to `nvidia/Cosmos3-Nano` for Nano; do not silently
substitute models after a failure. With a separate input provider, replace
every source/control `input.connection_id` and URI with that owner's registered
input scope. Leave `output` on a verified Nebius connection.

```bash
cp examples/structural-transfer.json "$WMH_WORK_DIR/transfer-request.json"
chmod 600 "$WMH_WORK_DIR/transfer-request.json"
# Replace placeholders in the private request before submitting.
python3 scripts/cosmos_client.py submit --route /v1/transfers --request "$WMH_WORK_DIR/transfer-request.json" --state "$WMH_WORK_DIR/transfer-state.json"
python3 scripts/cosmos_client.py resume --state "$WMH_WORK_DIR/transfer-state.json" --output-dir "$WMH_WORK_DIR/transfer-results"
```

Submit the entire source and entire aligned control videos. The full-source
contract accepts any video frame count of at least two, including 227 and
302, with no application maximum. The retained
`structural-transfer-832x480-121-v1` identifier names the internal 121-frame
window; it does not require a 121-frame source or a `4n+1` total. Native
inference handles overlap and final-window padding, then returns the complete
source frame count. Do not add a `num_frames` field, crop a long source or
submit client-side chunks. Read the live constraints rather than inferring
source support from the profile identifier alone.

| Current transfer requirement | Contract |
| --- | --- |
| Media | Fully decodable 832x480, 24-FPS video; every control has the same decoded frame count as the source. |
| Controls | One to four entries keyed by `edge`, `depth`, `seg`, `blur`; only advertised combinations. |
| Preparation | Edge can be uploaded or derived by omitting its `input`; depth, segmentation and blur require prepared owner-bound videos. |
| Weights | Finite and nonnegative with a finite positive total. They are relative, without a 1.0 ceiling; use `generation.control_guidance` for overall strength. |
| Sampling | Integer steps 1–200, CFG 0–20, nonnegative finite control guidance and finite flow shift; seed is an integer. |
| Text | Positive prompt required; positive and negative prompts each allow up to 16,384 characters. |
| Precision | This transfer profile is BF16; it does not accept `precision_profile`. A legacy augmentation FP8 option does not enable FP8 transfers. |
| Delivery | Guardrails stay enabled; complete decode/timeline checks and blank/frozen-output heuristics can reject delivery. |

Prepared foreground-only controls are accepted as media. Black pixels remain
control data; there is no separate spatial mask or per-pixel weight field.
The structural controls condition the full timeline; source RGB frames do not
condition generation on this route.
Full-source technical delivery does not establish semantic fidelity, device
identity, contact preservation or training-label validity. Evaluation remains
optional and report-only; it does not disable the service's delivery checks.

Mutable request JSON is limited to 1 MiB; media belongs in registered S3
objects. The separate media-ingest upload helper has a 5-GiB per-file limit;
that is not a frame-count or size cap on already-staged transfer input objects.
This portable helper does not upload local media. Keep the exact request and
state for retries; a changed source, control, weight, prompt or output is a
different request and needs a distinct identity.

## Text-to-video and fixed profiles

For T2V, a request file uses the explicit selected model/profile:

```json
{
  "model": "nvidia/Cosmos3-Nano",
  "prompt": "A fixed camera observes a robot arm moving a block on a table.",
  "output": {"connection_id": "customer-output", "prefix": "s3://CUSTOMER_BUCKET/generated/run-001"},
  "generation": {"profile": "nano-480p-20", "seed": 101}
}
```

The helper submits `/v1/videos` by default. Current schema examples below must
be checked against the running capability matrix before use:

| Task | Explicit model | Profile | Additional input |
| --- | --- | --- | --- |
| T2V | `nvidia/Cosmos3-Nano` | `nano-480p-20` | None |
| T2V | `nvidia/Cosmos3-Super` | `super-t2v-960-square-93` | None |
| Generated-audio T2V | `nvidia/Cosmos3-Super` | `super-t2v-audio-960-square-93` | None; sound is generated, no input audio |
| I2V | `nvidia/Cosmos3-Nano` | `nano-native-i2v-832x480-121` | One owner-bound image |
| I2V | `nvidia/Cosmos3-Super` | `super-i2v-512-square-to-960-square-93` | One 512×512 image |
| Prefix V2V | `nvidia/Cosmos3-Nano` | `nano-v2v-prefix-first5-832x480-121` | One 832×480, 24-FPS source video |
| Prefix V2V | `nvidia/Cosmos3-Super` | `super-v2v-prefix-first5-960-square-93` | One source video matching the running source contract |

For I2V or prefix V2V, add this top-level object and choose the corresponding
model/profile above, changing the URI to the actual object:

```json
{"input": {"connection_id": "customer-output", "uri": "s3://CUSTOMER_BUCKET/inputs/first-frame.png"}}
```

Use a video object such as `source.mp4` for V2V. Prefix continuation conditions
only the first five decoded source frames; it does not preserve the entire
source timeline. Read the running constraints before choosing dimensions,
frame rate or duration. The listed fixed profiles generally admit only seed;
do not transplant Nano T2V overrides into fixed I2V/Super profiles. Keep
guardrails enabled. No listed audio profile performs STT or TTS.

The helper also accepts `--route /v1/images`, `--route /v1/augmentations` or
`--route /v1/transfers` for
request files you have validated against the running schema. Image generation
and full-source structural augmentation have separate typed contracts; they
are not implicit task flags on the video request. Persist every exact logical
request with its own run-state file. Keep an existing state for retries of the
same request; changing it to bypass a conflict can create duplicate GPU work.
