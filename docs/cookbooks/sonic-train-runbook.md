# SONIC Train Runbook

This runbook covers the `npa workbench sonic train --runtime serverless` path
for Isaac Lab-based SONIC training and smoke validation.

## Purpose

The serverless train command submits a Nebius Serverless Job using the
self-contained SONIC image. The job validates that Isaac Lab and SONIC are
available in the same container, writes `sonic_smoke_result.json`, and uploads
artifacts to the requested S3 prefix.

Full SONIC training on BONES-SEED is a large multi-GPU workload. The Workbench
smoke target is intentionally minimal: it validates environment integration and
the training entry path, not convergence.

## Minimal Smoke

```bash
SMOKE_TS=$(date -u +%Y%m%dT%H%M%SZ)
npa workbench sonic -p uk-south1 -n w7sonic train \
  --runtime serverless \
  --project-id YOUR_PROJECT_ID \
  --gpu-type l40s \
  --gpu-count 1 \
  --embodiment unitree-g1 \
  --steps 10 \
  --output-path s3://YOUR_S3_BUCKET/w7sonic-smoke/$SMOKE_TS/ \
  --job-name sonic-smoke-$SMOKE_TS \
  --timeout 3600 \
  --poll-interval 15
```

Expected output artifacts:

- `sonic_smoke_result.json`
- `sonic_train_summary.json`
- `checkpoint_smoke.json`

## GPU Selection

Prefer RT-core GPUs because Isaac Lab simulation workloads use rendering and
physics paths that are best validated on RT-capable hardware:

- `l40s`
- `gpu-rtx-pro-6000`

H100/H200/B200/B300 may be useful for model training throughput, but they are not
the preferred smoke target for Isaac Lab simulation validation.

## Parameters

- `--embodiment`: defaults to `unitree-g1`, mapped to `UNITREE_G1_SONIC`.
- `--checkpoint`: defaults to `nvidia/GEAR-SONIC:sonic_release/last.pt`.
- `--data-path`: optional path or URI for training data.
- `--sample-data`: explicitly uses the SONIC sample data path.
- `--steps` / `--max-iterations`: minimal smoke iteration count.
- `--output-path`: required S3 prefix for serverless artifacts.
- `--submit-only`: submit and return without polling.

If `--data-path` is omitted, the command treats the run as a sample-data smoke.

## Cost Guard

The W7-sonic smoke budget is capped at $60. A single L40S smoke should stay well
below that if it reaches terminal state promptly. Stop retrying and classify the
run as platform or training failure if scheduling or startup consumes the budget
without SONIC logs.

## Failure Classification

- `PASS`: serverless Job succeeds and S3 artifacts are present.
- `FAIL_TRAINING`: the job starts but SONIC or Isaac Lab fails.
- `FAIL_PLATFORM`: image pull, subnet, auth, or scheduler failure before SONIC
  code runs.
- `FAIL_NER`: capacity or quota blocks scheduling.

If the serverless smoke fails after the retry budget, run a degraded container
smoke:

```bash
docker run --rm npa-sonic:0.1.0 train
```

That validates the container entrypoint and imports without consuming Nebius GPU
capacity.

## Known Limitations

- GR00T+SONIC orchestration is not part of this runbook.
- Additional embodiments beyond Unitree G1 are exposed as tags but not
  qualified by Workbench smoke.
- NIM distribution was not confirmed in discovery; the supported path is
  Hugging Face plus the upstream SONIC repository.
