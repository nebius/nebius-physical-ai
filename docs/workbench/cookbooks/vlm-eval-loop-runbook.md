# VLM-Eval Loop Runbook

[Cookbooks](README.md)

This runbook runs the sim-to-real VLM-eval loop on the self-hosted serving path:
serve a VLM with vLLM, score rollout directories with `vlm-eval`, and write a
task-success report.

Each evaluation records the requested `model` and the endpoint's returned
`served_model`. Self-hosted servers that omit the model identity leave
`served_model` null; NPA does not infer it from the requested alias. A supplied
identity must be a nonempty string. Retain the serving deployment's checkpoint
revision separately: a model name alone does not identify its weight bytes.

Every scored response must report the exact value
`choices[0].finish_reason="stop"`. NPA rejects truncated, filtered, tool-call,
aborted, empty, malformed, or missing completion metadata before parsing a
score, even if the response contains a complete-looking JSON verdict. If an
existing self-hosted adapter omits `finish_reason`, update it to emit the
standard OpenAI-compatible field; parseable JSON alone does not prove that
generation completed.

To verify this against your existing GPU endpoint, set
`NPA_INTEGRATION_E2E=1` and point `NPA_VLM_PROVENANCE_LIVE_CONFIG` at a private
JSON file containing `input_path`, `output_path` (a local JSON filename),
`endpoint_url`, `model`, `expected_served_model`, and `task`. Supply credentials
through the environment variable named by `api_key_env` (default
`VLM_EVAL_API_KEY`). Run
`npa/.venv/bin/python -m pytest npa/tests/e2e/test_vlm_served_model_live.py -q`.
The test calls the real endpoint and retains its verdict; it provisions and
destroys no resources.

## Prerequisites

- `npa` is installed from this repository.
- SkyPilot is configured and `sky check` shows Nebius enabled.
- A GPU with enough memory for the chosen VLM. No prebuilt serving image is
  required: the renderer installs vLLM (and `ninja`, which its JIT sampler needs)
  into whatever image the stage runs in.
- Object storage credentials are available through `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, and `AWS_ENDPOINT_URL`.
- Rollouts are available under one prefix, with one child directory per rollout.

## One Command

The loop is an `npa.workflow` spec, so a submit plus config overrides is the whole
invocation — no YAML rendering step:

```bash
export RUN_ID="vlm-eval-loop-smoke"
export NPA_S3_BUCKET="<your-bucket-name>"

npa workbench workflow submit \
  workflows/testing/vlm-eval-loop.yaml \
  --run-id "${RUN_ID}" \
  --var "bucket=${NPA_S3_BUCKET}" \
  --var "prefix=sim-to-real/${RUN_ID}" \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

That reads rollouts from `s3://${NPA_S3_BUCKET}/sim-to-real/${RUN_ID}/rollouts/` and
writes to `.../scores/`; override `rollouts_uri` / `scores_uri` with `--var` to point
elsewhere. The stage runs on SkyPilot's default GPU image: the renderer installs vLLM,
starts it, health-checks `/health`, and tears it down when the stage exits, so no
prebuilt serving image is required. Set `--var vlm_model=<repo-id>` for a different VLM
and `--var vlm_serve_ready_seconds=<n>` if a cold checkpoint download needs longer than
the 900 s default.

The default model is `Qwen/Qwen2-VL-7B-Instruct`, the default frame selection is
`keyframes`, and the default success threshold is `0.8`.

To score a *single* rollout instead of a set, use
`workflows/testing/vlm-eval-single.yaml`, or call
`npa workbench vlm-eval run` directly.

## Inputs

`rollouts_uri` points to a local path or `s3://` prefix. The loop treats each direct
child directory as one rollout, and falls back to treating the prefix itself as a
single rollout when it has no child directories:

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

`scores_uri` receives:

- `rollouts/<rollout-id>/vlm_eval_stub.json`: one structured result per rollout.
- `task_success_report.json`: aggregate report with `total_rollouts`,
  `passed_rollouts`, `success_rate`, `mean_score`, `task_success`, and the
  per-rollout `{success, score, rationale}` records.

Read the report:

```bash
aws s3 cp "s3://${NPA_S3_BUCKET}/sim-to-real/${RUN_ID}/scores/task_success_report.json" -
```

Use `task_success` as the coarse gate, then inspect low-score rollouts and their
rationales before iterating on policy, simulation, or rubric.

## Plug In Real Labeled Rollouts

For unlabeled gating, point `rollouts_uri` at the rollout prefix and keep the loop
spec unchanged. For labeled calibration, create a benchmark manifest that
points at the same rollout directories and includes `expected_label` for each
item, then run the sweep below. A benchmark must include at least one pass and
one fail label, and every resolved item ID must be unique. Invalid class balance
or duplicate IDs fail before rollout-frame materialization, VLM provider
credentials, or evaluator/provider activity. An S3-hosted manifest still needs
storage credentials before its labels and IDs can be parsed.

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
`vlm_success_threshold` in the loop spec (or pass `--var` at submit time).
`workflows/testing/vlm-eval-benchmark.yaml` runs the same sweep as
a workflow stage. Compare balanced accuracy first; the report also retains
specificity, accuracy, precision, recall, F1, and all four confusion counts.
This prevents an all-positive judge from being selected because the calibration
set contains more positive than negative cases.

## Separate outcome from agency

The packaged `isaac-agency` dataset is a hermetic example of a paired control:
the same exact six stylized frames support the state claim that the red cube
becomes elevated, but reject the agency claim that the separated, retracting arm
grasped and lifted it.

```bash
npa workbench vlm-eval benchmark \
  --dataset isaac-agency \
  --output /tmp/isaac-agency-benchmark.json \
  --backend stub \
  --frame-selection sequence \
  --max-frames 6 \
  --thresholds 0.5 \
  --format json
```

The manifest's optional `structural_check` is deliberately narrow. Before any
backend work, it validates the configured frame selection, labels, normalized
PNG hashes, color-mask completeness, motion direction, and expected structural
verdict for every configured item. The benchmark report records those
measurements under `sweep.structural_checks`, and real scoring reuses the exact
selected frame objects rather than reading or encoding the rollout again.
When an item relies on rollout metadata for its task, preflight freezes that
resolved task during the same materialization and reuses it for real-backend
scoring; stub and fixture-score behavior stays unchanged. Datasets without this
field retain no structural-check key.

The command above uses the stub only to exercise report plumbing; it does not
qualify a model. The fixture contains synthetic stylized stand-ins, not Isaac
Sim renders. Its geometry cannot prove contact, grasp, force, dynamics,
causality, photoreal generalization, policy success, physical correctness, or
robot safety.
The frozen measurements, hardware table, and reproduction commands are in the
[Isaac agency calibration evidence record](../evidence/vlm-isaac-agency-calibration.md).

## Troubleshooting

- `sky check` does not show Nebius enabled: fix SkyPilot credentials before
  launching the workflow.
- vLLM never becomes healthy: the stage fails fast and prints the last 200 lines of
  `/tmp/npa-vlm-server.log`, which is where the cause almost always is (out of GPU
  memory, an unsupported model, or a missing CUDA toolchain).
- No rollouts are evaluated: confirm `rollouts_uri` points to a prefix with child
  rollout directories or directly to one rollout directory.
- Scores are all low or noisy: tighten `RUBRIC`, switch `FRAME_SELECTION`, or run
  `vlm-eval benchmark` on labeled rollouts before using the gate.
- Simulator axes or debug gizmos appear in submitted frames: disable or crop
  them. A judge can misclassify overlays as task objects.
- A qualitative judgment appears to contradict a sealed numeric gate: keep the
  synchronized simulator or telemetry measurement authoritative. Text such as
  "a visible gap" does not establish that a predeclared height was crossed.
- S3 writes fail: verify `AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud`
  and that the storage keys can read `rollouts_uri` and write `scores_uri`.
