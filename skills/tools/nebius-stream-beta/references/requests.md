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

Successful admission returns HTTP 202 with the job response; it is not a
completed video. Reuse the same request and idempotency key after an ambiguous
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

The helper also accepts `--route /v1/images` or `--route /v1/augmentations` for
request files you have validated against the running schema. Image generation
and full-source structural augmentation have separate typed contracts; they
are not implicit task flags on the video request. Persist every exact logical
request with its own run-state file. Keep an existing state for retries of the
same request; changing it to bypass a conflict can create duplicate GPU work.
