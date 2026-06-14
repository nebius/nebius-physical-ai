# Workbench Tools - Serverless Coverage

## Overview

The Workbench supports `--runtime serverless` on these non-LeRobot tools, each backed by Nebius Serverless Jobs:

| Tool | Command | Serverless Status | Smoke GPU | Smoke Result |
|------|---------|-------------------|-----------|--------------|
| Cosmos | `train` | available | H100 (W13), L40S historical | PASS (`w13-cosmos-e2e-20260521T233523Z`) |
| Isaac Lab | `train` | available | L40S | PASS |
| FiftyOne | `load-dataset` | available | L40S | PASS |
| Genesis | `train-teacher` | available | L40S | PASS |
| GR00T | `infer` | available | H200, L40S retries | SMOKE_FAILED; Nebius handoff |
| LeRobot | `train`, `profile-train` | available; see [LeRobot cookbook](lerobot-gpu-benchmarks.md) | varies | varies |
| SONIC | `train` | available; image build blocked | L40S planned | FAIL_PLATFORM; Phase B blocked |

GR00T code and unit coverage are in place, but smoke validation is still open.
W7p-groot-debug classified the W7-parallel-tools failures as an image tag
mismatch (the old unpushed GR00T tag was not pushed), fixed the default serverless image
tag to `npa-groot:0.1.0`, and retried once. The retry reached `STARTING` with a
running compute instance but produced no logs before cleanup. Nebius handoff:
`/tmp/w7pgd-20260514T001207Z/NEBIUS-SUPPORT-HANDOFF.md`.

## Common Interface

The serverless Job commands use the same option shape:

- `--project-id <NEBIUS_PROJECT_ID>`: Nebius project that owns the Job.
- `--image <CONTAINER_REF>`: override the default tool container.
- `--gpu-type <TYPE>`: GPU type such as `h200`, `h100`, `b300`, `l40s`, or `gpu-rtx-pro-6000`.
- `--gpu-count <N>`: number of GPUs.
- `--gpu-preset <PRESET>`: optional Nebius preset override.
- `--subnet-id <ID>`: optional VPC subnet override.
- `--job-name <NAME>`: explicit Job name.
- `--output-path <S3_URI>`: required output prefix for uploaded artifacts.
- `--submit-only`: return after Job creation.
- `--poll-interval <SECONDS>`: polling cadence.
- `--timeout <SECONDS>`: max wait for terminal state.

## Per-Tool Recipes

### Cosmos

```bash
npa workbench cosmos -p eu-north1 -n w13-cosmos train \
  --runtime serverless \
  --project-id <YOUR_PROJECT_ID> \
  --image cr.eu-north1.nebius.cloud/${NPA_REGISTRY_ID}/npa-cosmos:1.0.9 \
  --gpu-type h100 \
  --gpu-count 1 \
  --gpu-preset 1gpu-16vcpu-200gb \
  --output-path s3://${NPA_S3_BUCKET}/w13-cosmos-e2e/<run-id>/ \
  --job-name <run-id> \
  --smoke \
  --smoke-seconds 5 \
  --timeout 3600 \
  --poll-interval 15
```

### Isaac Lab

```bash
npa workbench isaac-lab -p uk-south1 -n w7p-isaac train \
  --runtime serverless \
  --project-id <YOUR_PROJECT_ID> \
  --task Isaac-Reach-Franka-v0 \
  --num-envs 1 \
  --steps 1 \
  --gpu-type l40s \
  --gpu-count 1 \
  --output-path s3://${NPA_S3_BUCKET}/<run-prefix>/isaac-lab-smoke/ \
  --job-name isaac-lab-smoke3-<run-id> \
  --timeout 3600 \
  --poll-interval 15
```

### FiftyOne

```bash
npa workbench fiftyone -p uk-south1 -n w7p-fiftyone load-dataset \
  --runtime serverless \
  --project-id <YOUR_PROJECT_ID> \
  --name w7p-curated \
  --input-path Voxel51/VisDrone2019-DET \
  --gpu-type l40s \
  --gpu-count 1 \
  --output-path s3://${NPA_S3_BUCKET}/<run-prefix>/fiftyone-smoke/ \
  --job-name fiftyone-smoke-<run-id> \
  --timeout 3600 \
  --poll-interval 15
```

### Genesis

```bash
npa workbench genesis -p uk-south1 -n w7p-genesis train-teacher \
  --runtime serverless \
  --project-id <YOUR_PROJECT_ID> \
  --n-envs 1 \
  --max-iterations 1 \
  --gpu-type l40s \
  --gpu-count 1 \
  --output-path s3://${NPA_S3_BUCKET}/<run-prefix>/genesis-smoke/ \
  --job-name genesis-smoke-<run-id> \
  --timeout 3600 \
  --poll-interval 15
```

### GR00T

```bash
npa workbench groot -p uk-south1 -n w7p-groot infer \
  --runtime serverless \
  --project-id <YOUR_PROJECT_ID> \
  --input-path s3://${NPA_S3_BUCKET}/<run-prefix>/groot-input/checkpoint/ \
  --dataset-path s3://${NPA_S3_BUCKET}/<run-prefix>/groot-input/dataset/ \
  --output-path s3://${NPA_S3_BUCKET}/<run-prefix>/groot-smoke/ \
  --gpu-type h200 \
  --gpu-count 1 \
  --model-variant nvidia/GR00T-N1.7-3B \
  --steps 1 \
  --action-horizon 1 \
  --job-name groot-smoke-<run-id> \
  --timeout 3600 \
  --poll-interval 15
```

Current status: `SMOKE_FAILED`. W7p-groot-debug fixed the missing-image-tag
issue and proved the corrected submission uses
`cr.eu-north1.nebius.cloud/<your-registry-id>/npa-groot:0.1.0`; the post-fix
retry stalled in `STARTING` with no logs and was deleted.

## Shared Infrastructure

All new non-LeRobot serverless paths use `npa.serverless_common` for:

- Environment variable construction and secret splitting.
- GPU platform and preset resolution.
- Output upload command generation.
- S3 output path validation.

Subnet resolution remains per-tool because project/workbench config conventions differ. Polling calls `ServerlessClient.poll_job()` directly. `npa/src/npa/clients/serverless.py` was not changed by this rollout.

## LanceDB Coverage Note

| Tool | Command | Serverless Status | Smoke GPU | Smoke Result |
|------|---------|-------------------|-----------|--------------|
| LanceDB | `deploy` | n/a; persistent CPU service, use `container`, `vm`, `byovm`, or `cloud` | n/a (CPU) | PASS (local container smoke via LanceDB subapp; parent Workbench registration follow-up required) |
