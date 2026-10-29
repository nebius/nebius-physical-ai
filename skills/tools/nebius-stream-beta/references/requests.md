# Storage and generation requests

Create these JSON files privately. Substitute actual customer values locally;
no secrets belong in these configuration files. A storage registration config:

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
