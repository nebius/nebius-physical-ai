---
name: vlm-eval
description: Use when working on the VLM-eval workbench tool — scoring rollouts with a VLM, the offline stub backend, self-hosted vLLM or API backends, threshold/rubric/model benchmark sweeps, or the sim-to-real VLM-eval loop. This is the zero-credential first run and the eval backend used by sim-to-real.
---

# VLM-Eval

`vlm-eval` is a first-class Eval capability. It scores rollout artifacts against a
task with a self-hosted (vLLM), API, or offline `stub` backend and writes a
task-success report. It is the zero-credential first run and the `vlm-frames`
eval backend used by the sim-to-real loop. Cookbook:
`docs/workbench/cookbooks/vlm-eval-loop-runbook.md`.

## Commands

- `npa workbench vlm-eval run`: score one rollout/input to an output report.
- `npa workbench vlm-eval benchmark`: sweep thresholds, rubrics, and models over a
  labeled rollout set and write a ranked accuracy report with the best config.
- `npa workbench vlm-eval workflow`: render/print the checked-in SkyPilot template.
- `npa workbench vlm-eval status` / `list`: tool observability.

Like every workbench tool, all paths use `--input-path` / `--output-path` S3 URIs
and the behavior lives in one shared implementation; do not duplicate scoring
logic across CLI, SDK, and API.

## Zero-Credential First Run

No cloud, GPU, or credentials. Offline `stub` backend over shipped fixtures:

```bash
npa workbench vlm-eval benchmark \
  --dataset npa/src/npa/workbench/vlm_eval/fixtures/sample_benchmark/benchmark.json \
  --output /tmp/vlm-eval-benchmark.json \
  --backend stub \
  --thresholds 0.5,0.8,0.9 \
  --rubrics default,strict \
  --models Qwen/Qwen2-VL-7B-Instruct \
  --format json
```

Expected: a ranked report with `accuracy: 1.0` over four labeled rollouts. Swap
`--backend stub` for `self-hosted` (with `--endpoint-url`) or `api` once
credentials and a serving endpoint exist.

## Backends

- `stub`: offline, deterministic. Use for smoke, fixtures, and CI.
- `self-hosted`: OpenAI-compatible vLLM endpoint (`--endpoint-url`). The
  self-hosted workflow starts vLLM, then calls `run` or `benchmark`.
- `api`: external OpenAI-compatible API.

## Loop Inputs And Outputs

`ROLLOUTS` is a local path or `s3://` prefix; each direct child directory is one
rollout (image files, RGB `.npy`/`.npz`, or a supported video). Task text comes
from the call or from `manifest.json` / `info.json`. `OUTPUT_DIR` receives one
`vlm_eval_stub.json` per rollout plus `task_success_report.json` (with
`total_rollouts`, `passed_rollouts`, `success_rate`, `mean_score`,
`task_success`, and per-rollout `{success, score, rationale}`).

Use `task_success` as the coarse gate, then inspect low-score rationales before
iterating on policy, simulation, or rubric. Defaults: model
`Qwen/Qwen2-VL-7B-Instruct`, frame selection `keyframes`, threshold `0.8`.

## Tune

Calibrate against labeled rollouts before trusting the gate:

```bash
npa workbench vlm-eval benchmark \
  --dataset s3://$NPA_S3_BUCKET/vlm-eval/benchmark/benchmark.json \
  --output s3://$NPA_S3_BUCKET/vlm-eval/benchmark/results/ \
  --backend self-hosted --endpoint-url http://127.0.0.1:8000/v1 \
  --models Qwen/Qwen2-VL-7B-Instruct \
  --rubrics default,strict --thresholds 0.5,0.8,0.9 --format json
```

Apply the best threshold/rubric to `SUCCESS_THRESHOLD` and `RUBRIC` in the loop
workflow.

## Common Failures

- `sky check` not showing Nebius enabled -> fix SkyPilot creds before launch.
- vLLM never healthy -> check `vlm-server.log`, confirm the image has CUDA + vLLM,
  use the YAML's documented GPU failover.
- No rollouts evaluated -> `ROLLOUTS` must point at a prefix of rollout dirs.
- S3 writes fail -> verify `AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud`
  and that keys can read `ROLLOUTS` and write `OUTPUT_DIR`.

## Tests

Mock serving and S3 at the call site; never hit a live VLM or bucket in unit
tests. Use the `stub` backend and shipped fixtures for deterministic assertions.
