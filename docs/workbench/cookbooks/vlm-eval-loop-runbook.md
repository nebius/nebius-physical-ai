# VLM-Eval Loop Runbook

This runbook runs the sim-to-real VLM-eval loop on the self-hosted serving path:
serve a VLM with vLLM, score rollout directories with `vlm-eval`, and write a
task-success report.

## Prerequisites

- `npa` is installed from this repository.
- SkyPilot is configured and `sky check` shows Nebius enabled.
- The workflow image contains `npa`, CUDA/PyTorch, vLLM, `transformers`,
  `qwen-vl-utils`, `curl`, and `jq`.
- Object storage credentials are available through `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, and `AWS_ENDPOINT_URL`.
- Rollouts are available under one prefix, with one child directory per rollout.

## One Command

Set the input and output paths, render the checked-in blueprint to a temporary
workflow, then submit it:

```bash
export RUN_ID="vlm-eval-loop-smoke"
export NPA_S3_BUCKET="<your-bucket-name>"
export ROLLOUTS="s3://${NPA_S3_BUCKET}/sim-to-real/${RUN_ID}/rollouts/"
export OUTPUT_DIR="s3://${NPA_S3_BUCKET}/sim-to-real/${RUN_ID}/vlm-eval-loop/"
export NPA_VLM_IMAGE="cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-cosmos:1.0.9"
# For production BYO, set NPA_VLM_IMAGE to a prebuilt VLM/vLLM serving image.

npa/.venv/bin/python - <<'PY'
import os
from pathlib import Path

import yaml

source = Path("npa/workflows/workbench/skypilot/sim-to-real-loop.yaml")
target = Path("/tmp/vlm-eval-loop.yaml")
docs = list(yaml.safe_load_all(source.read_text(encoding="utf-8")))
task = docs[1]
task["envs"]["NPA_VLM_IMAGE"] = os.environ["NPA_VLM_IMAGE"]
task["resources"]["image_id"] = f"docker:{os.environ['NPA_VLM_IMAGE']}"
task["envs"]["ROLLOUTS"] = os.environ["ROLLOUTS"]
task["envs"]["OUTPUT_DIR"] = os.environ["OUTPUT_DIR"]
target.write_text(yaml.safe_dump_all(docs, sort_keys=False), encoding="utf-8")
print(target)
PY

npa workbench workflow submit /tmp/vlm-eval-loop.yaml --run-id "${RUN_ID}"
```

The default model is `Qwen/Qwen2-VL-7B-Instruct`, the default frame selection is
`keyframes`, and the default success threshold is `0.8`.

## Inputs

`ROLLOUTS` points to a local path or `s3://` prefix. The loop treats each direct
child directory as one rollout:

```text
rollouts/
  rollout-000/
    frame-000.png
    frame-001.png
    manifest.json
  rollout-001/
    frame-000.png
    frame-001.png
```

Each rollout can contain image files, RGB `.npy` or `.npz` arrays, or a video
file supported by the `vlm-eval` frame loader. If the task text is not supplied,
`vlm-eval` looks for it in `manifest.json`, `info.json`, or task metadata.

## Outputs

`OUTPUT_DIR` receives:

- `rollouts/<rollout-id>/vlm_eval_stub.json`: one structured result per rollout.
- `task_success_report.json`: aggregate report with `total_rollouts`,
  `passed_rollouts`, `success_rate`, `mean_score`, `task_success`, and the
  per-rollout `{success, score, rationale}` records.

Read the report:

```bash
aws s3 cp "${OUTPUT_DIR%/}/task_success_report.json" -
```

Use `task_success` as the coarse gate, then inspect low-score rollouts and their
rationales before iterating on policy, simulation, or rubric.

## Plug In Real Labeled Rollouts

For unlabeled gating, set `ROLLOUTS` to the rollout prefix and keep the loop
workflow unchanged. For labeled calibration, create a benchmark manifest that
points at the same rollout directories and includes `expected_label` for each
item, then run the sweep below.

## Tune

Sweep thresholds, rubrics, and models against labeled rollouts:

```bash
npa workbench vlm-eval benchmark \
  --dataset s3://${NPA_S3_BUCKET}/vlm-eval/benchmark/benchmark.json \
  --output s3://${NPA_S3_BUCKET}/vlm-eval/benchmark/results/ \
  --backend self-hosted \
  --endpoint-url http://127.0.0.1:8000/v1 \
  --models Qwen/Qwen2-VL-7B-Instruct \
  --rubrics default,strict \
  --thresholds 0.5,0.8,0.9 \
  --format json
```

Use the best threshold and rubric from the benchmark report to update
`SUCCESS_THRESHOLD` and `RUBRIC` in the loop workflow.

## Troubleshooting

- `sky check` does not show Nebius enabled: fix SkyPilot credentials before
  launching the workflow.
- vLLM never becomes healthy: check `vlm-server.log`, verify the image has CUDA
  and vLLM, and try the documented GPU failover in the YAML comments.
- No rollouts are evaluated: confirm `ROLLOUTS` points to a prefix with child
  rollout directories or directly to one rollout directory.
- Scores are all low or noisy: tighten `RUBRIC`, switch `FRAME_SELECTION`, or run
  `vlm-eval benchmark` on labeled rollouts before using the gate.
- S3 writes fail: verify `AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud`
  and that the storage keys can read `ROLLOUTS` and write `OUTPUT_DIR`.
