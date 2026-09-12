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
`trajectory.json` (upstream trajectory metadata and displacement-error metrics),
and `trajectory.png` (the calibrated-camera visualization). The upstream JSON
does not contain the full predicted coordinate array. NPA snapshots the selected
manifest before downloading weights, verifies the output sample identity, seed,
and required camera projection, and decodes the PNG before publication. Missing
or invalid manifests fail before model fetch; they cannot fall back to upstream's
default clip. `result.json` includes the manifest hash and validated metrics.

## Ray experiments

The [sweep template](../../workflows/testing/alpamayo2-ray-sweep.yaml) extends the
single-inference template into a scenario × seed × diffusion-step experiment.
Its default grid contains eight cases: samples `0,1`, seeds `42,43`, and step
counts `10,20`. Change these comma-separated config values to define an experiment;
there is no implicit truncation. `workers` defaults to one GPU actor. Increase
both the resource allocation and actor count together for multiple GPUs on one
node. The catalog rejects multi-node replication of this single-node command.

Ray GPU actors call the shared upstream inference implementation. Each actor
keeps one case in flight and reuses the downloaded pinned model snapshot. Model
weights load into a fresh upstream subprocess for each case. Actors emit
structured progress events in the driver logs: `alpamayo.case_started`
and `alpamayo.case_completed`. Completion events include measured ADE and elapsed
time after artifact publication; CLI progress stays on stderr. Ray CPU tasks
calculate per-scenario/per-setting mean ADE, mean FDE, and population standard
deviation across seeds. `elapsed_seconds` measures the whole case, including
model fetch/load, inference, rendering, and artifact upload. It does not isolate
inference latency. The runtime uses an independent local Ray instance with
a 256 MiB object store for small measurement records; it never discovers
SkyPilot's management Ray through an ambient address.

Use the current source with the accepted image: the release image predates the
new sweep command. Source staging is required until an image containing this
implementation is released. The renderer installs Ray 2.58.0 into the NPA stage
interpreter, independently of the upstream model interpreter.

```bash
npa workbench health preflight --checks nebius,hf,s3 --json
npa workbench health access --capability alpamayo2-super --json
npa workbench workflow validate-spec workflows/testing/alpamayo2-ray-sweep.yaml --json
npa workbench workflow submit workflows/testing/alpamayo2-ray-sweep.yaml \
  --project YOUR_PROJECT --infra k8s/YOUR_CONTEXT --stage-src \
  --var bucket=YOUR_BUCKET \
  --secret-env HF_TOKEN --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

On a verified RTX PRO 6000 target, set
`NPA_WORKFLOW_GPU_ACCELERATOR=RTXPRO6000:1` in the submitting process. The
checked-in B200 default remains the same as the inference base template.

The [hard-case template](../../workflows/testing/alpamayo2-ray-hardcases.yaml)
first measures a baseline, then consumes its S3 `report.json` to select every
scenario whose mean ADE exceeds `minimum_ade` (default `2.0` meters). It reruns
those scenarios at `refinement_steps` (default `20,30`) using the baseline seeds.
An empty selection completes with zero refinement cases. Comparisons require
the same model revision, dataset revision, manifest hash, scenario, and seed.
A negative `ade_change_m` means lower measured displacement error; extra steps
are not assumed to improve a prediction. These are offline error measurements,
not driving-safety or closed-loop policy evaluations.

Both templates publish a `report.json` and `SHA256SUMS`. Each measured case links
its independently verified `result.json`, `trajectory.json`, and `trajectory.png`
under a fresh execution subdirectory. A missing case, duplicate, invalid metric,
or failed actor prevents a complete report. Use a new run ID for each experiment;
this path does not implement checkpoint recovery or automatic inference retries.

For an already allocated GPU environment, `npa workbench alpamayo2-super sweep`
exposes `--output-path`, `--run-id`, `--sample-indices`, `--seeds`,
`--diffusion-steps`, and `--workers`. Supply `--input-path` with a baseline report
and `--minimum-ade` to refine it. Public CLI handoffs use S3. The Python SDK's
`sweep(AlpamayoSweepRequest(...))` uses the same implementation and can use local
paths for development. A dedicated Ray Jobs driver can invoke
`python -m npa.workbench.alpamayo2_super.ray_sweep` with an explicit
`--ray-address HOST:PORT`; deliver the NPA source and dependencies to every worker.
The module defaults to `local` and rejects `auto` to avoid management-cluster
discovery. `ray[default]==2.58.0` is required in the application environment.

Independently verify a completed live report with the read-only artifact test.
Set `NPA_ALPAMAYO_RAY_REPORT_URI` to the exact authorized S3 `report.json` and
provide that bucket's storage credentials in the test environment:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_ALPAMAYO_RAY_REPORT_URI=s3://YOUR_BUCKET/runs/YOUR_RUN/alpamayo2-ray-sweep/sweep/report.json \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_alpamayo_ray_artifacts_live.py -q
```

This downloads and decodes every case's artifacts and checks report hashes,
case coverage, pinned revisions, sample identity, projected trajectory shape,
and consistency of the measured errors. It requires a prior real GPU run;
it does not submit another workload.

The [RTX validation report](alpamayo2-ray-validation.md) records the measured
20-case experiment, test results, cache intervention, and remaining platform
coverage. The templates' B200 default and standard SkyPilot submission still
need their own live qualification.

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
