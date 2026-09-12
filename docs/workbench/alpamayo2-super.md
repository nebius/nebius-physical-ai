# Alpamayo 2 Super

[Workbench docs](README.md)

NPA runs NVIDIA Alpamayo 2 Super through the upstream VLM-plus-diffusion-expert
inference entrypoint. The redistributable `npa-alpamayo2-super` image contains the pinned
Apache-2.0 source and CUDA runtime, but no model weights, dataset bytes,
Hugging Face token, or populated model cache.

## Terms and access

The Hugging Face repositories have separate terms:

- `nvidia/Alpamayo2-Super` is public under OpenMDW 1.1. NPA pins revision
  `00554695e729a6ff0b6281fd2c81b18d06e33dbe` and fetches it at runtime.
- `nvidia/PhysicalAI-Autonomous-Vehicles` is gated by NVIDIA's AV Dataset
  License. It is limited to accepted internal use, is not redistributable, and
  requires interactive acceptance on the operator's Hugging Face account. NPA
  pins revision `b719eea7f0a63619ef51ec7f54178af0937ef050` and fetches it only
  after acceptance. NPA cannot accept this click-through gate for the operator.

Review the current upstream terms before use. The runtime identity must provide
`HF_TOKEN`; never put it in workflow YAML, an image layer, or an artifact. The
model and dataset cache is node-local and operator-owned.

```bash
npa workbench alpamayo2-super terms
npa workbench health access --capability alpamayo2-super --json
```

Both repositories must be available before GPU provisioning. HTTP 401 means
the credential is absent or invalid; dataset HTTP 403 means the interactive
gate has not been accepted for that credential.

## Run the workflow

Copy `workflows/testing/alpamayo2-super-inference.yaml`, set
the operator-owned bucket/prefix and approved image reference required by the
deployment, then validate and plan before submission:

```bash
npa workbench workflow validate-spec alpamayo2-super-inference.yaml --json
npa workbench workflow plan-spec alpamayo2-super-inference.yaml --json
npa workbench workflow submit alpamayo2-super-inference.yaml \
  --var bucket=OPERATOR_BUCKET \
  --secret-env HF_TOKEN --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

`us-central1` maps to the configured Nebius target alias for that region;
workflow specs intentionally exclude tenant, project, cluster, registry,
bucket, and credential identifiers. The default is one B200 because NVIDIA
measured a 72,115 MiB peak for its published H100 configuration and B200 leaves
headroom. RTX PRO 6000 is an independent `sm_120` validation path, not something
inferred from B200 success.

Successful inference publishes `result.json` (pins and provenance),
`trajectory.json` (the predicted ego trajectory), and `trajectory.png` (the
calibrated-camera visualization).

## Serve inference

The image's `serve` entrypoint requires `NPA_ALPAMAYO2_SUPER_TOKEN` and
`NPA_ALPAMAYO2_SUPER_OUTPUT_ROOT` at startup. Supply the token through the
deployment's secret configuration. Set the output root to an operator-owned
local directory with mode `0700`, or an authorized S3 bucket/prefix. Use a
private network or an HTTPS proxy when exposing the service beyond localhost.
Every operational endpoint, including `/health`, requires
`Authorization: Bearer <service-token>`.

HTTP requests use the repository-pinned model and dataset revisions. The
server snapshots `NPA_ALPAMAYO2_SUPER_MANIFEST` at startup, defaulting to the
image's validation manifest. A request cannot replace that manifest or select
model code. Set `sample_index` and the usual inference controls in `/run`.
The optional `output_path` is a short result label containing only letters,
digits, underscores and hyphens; each request receives a new unique prefix
beneath the configured output root. Absolute paths and S3 destinations belong
in deployment configuration. The trusted CLI and SDK retain their operator
model and destination options.

`dry_run: true` validates this HTTP preparation path and returns the pinned
upstream invocation without downloading data or running inference. It does
not establish GPU or model correctness.

## Build and release gate

Build from the repository root. A local build automatically runs the payload
scan; any model weight, dataset media, populated Hugging Face cache, or
credential is a release blocker.

```bash
bash npa/docker/workbench/alpamayo2-super/build.sh
npa/.venv/bin/python npa/scripts/scan_image_alpamayo2_payload.py npa-alpamayo2-super:0.1.0-cu128
```

Build and scan prove redistribution hygiene only. A release also requires real
upstream inference on B200 and a separate result on RTX PRO 6000, with non-empty
JSON and PNG artifacts. Do not describe dry-run, image import, or CUDA import
checks as model validation.

## Accepted release evidence

Release `0.1.0-cu128` (OCI index digest
`sha256:2164450f8baf57d8798f64063ea27bf11611f5b695c467de0c2e319e3134ebd5`)
was validated on 2026-08-18 in operator-owned `us-central1` resources:

- The scanner inspected all 26 image layers and found no checkpoint, dataset,
  populated Hugging Face cache, credential, or token payload.
- One B200 (`sm_100`, 183,359 MiB) completed real upstream inference and wrote
  valid result JSON, trajectory JSON, and calibrated-camera PNG artifacts.
- One RTX PRO 6000 (`sm_120`, 97,887 MiB) independently completed the same
  workflow. Its observed peak was 71,447 MiB at 100% GPU utilization.
- Both runs used the exact pinned source, model, and dataset revisions above,
  produced projected trajectories of shape `[1, 1, 1, 64, 3]`, and required no
  recovery wave. Small floating-point metric differences across architectures
  are expected; cross-GPU bitwise identity is not a release criterion.

The first run downloads approximately 67 GB of operator-entitled model assets.
Neither the model cache nor the non-transferable dataset is part of the image or
published artifacts.

The accepted runtime-fetch image is available from the anonymous GHCR mirror
used by default. Neither that public image nor any private copy contains or
authorizes copying the gated dataset, model weights, or runtime cache.
