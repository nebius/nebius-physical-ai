# Workbench Tools - Serverless Coverage

## Overview

The Workbench supports `--runtime serverless` on these non-LeRobot tools, each backed by Nebius Serverless Jobs:

| Tool | Command | Serverless Status | Smoke GPU | Smoke Result |
|------|---------|-------------------|-----------|--------------|
| Cosmos | `train` | available | L40S | PASS |
| Isaac Lab | `train` | available | L40S | PASS |
| FiftyOne | `load-dataset` | available | L40S | PASS |
| Genesis | `train-teacher` | available | L40S | PASS |
| GR00T | `infer` | available | H200, L40S retries | SMOKE_FAILED; Nebius handoff |
| LeRobot | `train`, `profile-train` | available; see [LeRobot cookbook](lerobot-gpu-benchmarks.md) | varies | varies |

GR00T code and unit coverage are in place, but smoke validation is still open.
W7p-groot-debug classified the W7-parallel-tools failures as an image tag
mismatch (`npa-groot:n1.7` was not pushed), fixed the default serverless image
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
npa workbench cosmos -p uk-south1 -n w7p-cosmos train \
  --runtime serverless \
  --project-id YOUR_PROJECT_ID \
  --gpu-type l40s \
  --gpu-count 1 \
  --output-path s3://YOUR_S3_BUCKET_2/w7p-fresh/20260513T225839Z/cosmos-smoke/ \
  --job-name cosmos-smoke2-20260513T225839Z \
  --smoke \
  --smoke-seconds 5 \
  --timeout 3600 \
  --poll-interval 15
```

### Isaac Lab

```bash
npa workbench isaac-lab -p uk-south1 -n w7p-isaac train \
  --runtime serverless \
  --project-id YOUR_PROJECT_ID \
  --task Isaac-Reach-Franka-v0 \
  --num-envs 1 \
  --steps 1 \
  --gpu-type l40s \
  --gpu-count 1 \
  --output-path s3://YOUR_S3_BUCKET_2/w7p-fresh/20260513T225839Z/isaac-lab-smoke/ \
  --job-name isaac-lab-smoke3-20260513T225839Z \
  --timeout 3600 \
  --poll-interval 15
```

### FiftyOne

```bash
npa workbench fiftyone -p uk-south1 -n w7p-fiftyone load-dataset \
  --runtime serverless \
  --project-id YOUR_PROJECT_ID \
  --name w7p-curated \
  --input-path Voxel51/VisDrone2019-DET \
  --gpu-type l40s \
  --gpu-count 1 \
  --output-path s3://YOUR_S3_BUCKET_2/w7p-fresh/20260513T225839Z/fiftyone-smoke/ \
  --job-name fiftyone-smoke-20260513T225839Z \
  --timeout 3600 \
  --poll-interval 15
```

### Genesis

```bash
npa workbench genesis -p uk-south1 -n w7p-genesis train-teacher \
  --runtime serverless \
  --project-id YOUR_PROJECT_ID \
  --n-envs 1 \
  --max-iterations 1 \
  --gpu-type l40s \
  --gpu-count 1 \
  --output-path s3://YOUR_S3_BUCKET_2/w7p-fresh/20260513T225839Z/genesis-smoke/ \
  --job-name genesis-smoke-20260513T225839Z \
  --timeout 3600 \
  --poll-interval 15
```

### GR00T

```bash
npa workbench groot -p uk-south1 -n w7p-groot infer \
  --runtime serverless \
  --project-id YOUR_PROJECT_ID \
  --input-path s3://YOUR_S3_BUCKET_2/w7p-fresh/20260513T225839Z/groot-input/checkpoint/ \
  --dataset-path s3://YOUR_S3_BUCKET_2/w7p-fresh/20260513T225839Z/groot-input/dataset/ \
  --output-path s3://YOUR_S3_BUCKET_2/w7p-fresh/20260513T225839Z/groot-smoke/ \
  --gpu-type h200 \
  --gpu-count 1 \
  --model-variant nvidia/GR00T-N1.7-3B \
  --steps 1 \
  --action-horizon 1 \
  --job-name groot-smoke-20260513T225839Z \
  --timeout 3600 \
  --poll-interval 15
```

Current status: `SMOKE_FAILED`. W7p-groot-debug fixed the missing-image-tag
issue and proved the corrected submission uses
`cr.eu-north1.nebius.cloud/YOUR_REGISTRY_ID/npa-groot:0.1.0`; the post-fix
retry stalled in `STARTING` with no logs and was deleted.

## Shared Infrastructure

All new non-LeRobot serverless paths use `npa.serverless_common` for:

- Environment variable construction and secret splitting.
- GPU platform and preset resolution.
- Output upload command generation.
- S3 output path validation.

Subnet resolution remains per-tool because project/workbench config conventions differ. Polling calls `ServerlessClient.poll_job()` directly. `npa/src/npa/clients/serverless.py` was not changed by this rollout.
