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

Successful real-backend results also contain an `evidence` record. It binds the
requested and returned model to:

- SHA-256 hashes, dimensions, media types, byte counts, and source-relative
  labels for the exact normalized image bytes submitted to the model;
- source kind, zero-based source index, source frame count, and source video
  timestamp when extraction can establish them;
- hashes of the prompt, rubric, and secret-free request manifest;
- request time, endpoint role, HTTP status when available, latency, finish
  reason, provider request ID and usage when returned;
- the exact provider response body, its SHA-256 hash, and parser version.

The request manifest intentionally excludes authorization, endpoint addresses,
input/output locations, prompts, and base64 image data. It contains enough
information to recompute what was submitted without copying pixels or secrets.
Its `sampling` block records the strategy, requested frame limit, selected
indices and timestamps, source count, and whether index and timestamp coverage
are complete. Unknown video metadata stays null; a generated extraction ordinal
is never presented as a source frame index. `coverage_complete` means every
submitted frame has auditable source-index metadata, not that every available
source frame was submitted.
The full result still belongs in private run storage because the provider
response and existing task fields can describe operator data.

`stub` results and `--score` overrides do not contain provider evidence. They are
wiring checks, never visual proof. Real `benchmark` reports retain the evidence
for every case so calibration failures and model disagreement remain inspectable.
The hosted API parser removes a single Markdown JSON fence around one complete
object and appends `+markdown-fence-v1` to the retained parser version.
Prefixes, suffixes, duplicate keys, non-finite numbers, invalid types, and
partial output fail on that strict path. The self-hosted parser deliberately
keeps its older compatibility behavior: it can extract an object from
surrounding text, uses the last duplicate key, and coerces compatible score and
`success` values. Retained parser versions distinguish these paths. None of
these fields turns a visual judgment into objective task, geometry, collision,
or safety evidence.

The serialized result retains the effective `rubric`, so the exact prompt can
be reconstructed from `task`, `rubric`, `frame_selection`, and `frame_count`.
`passed` is always `score >= success_threshold`. When a real backend actually
returns a `success` boolean, the result records it as `provider_success` and
reports whether it agrees in `provider_success_matches_score_gate`.
Self-hosted responses that omit the boolean leave both fields null rather than
presenting a score-derived fallback as provider output. Legacy non-boolean
values such as `"true"` are likewise not promoted to provider booleans. A real
disagreement is calibration evidence, not permission to replace the
score-derived label. Before reviewing thin geometry or skeletons, compare
retained submitted-frame dimensions with the source because normalization can
remove the defect.

For a consequential or disputed review, `compare-judges` preserves two hosted
outcomes without averaging:

```bash
npa workbench vlm-eval compare-judges \
  --input-path <one-rollout> \
  --output-path <private-evidence-prefix> \
  --primary-model <hosted-vision-model-a> \
  --secondary-model <hosted-vision-model-b> \
  --task "Describe the exact visible completion evidence."
```

The direct SDK surface takes a typed
`npa.sdk.workbench.vlm_eval.VlmJudgeComparisonRequest` and passes it to
`npa.sdk.workbench.vlm_eval.compare_judges`.

The command materializes and selects frames once, builds one prompt, and proves
the transported request objects differ only in `model`. It writes
`vlm_judge_disagreement.json`, retains each complete result or typed error, and
requires escalation on disagreement or judge error. The report is always
`audit_only`; agreement does not qualify either model, estimate an operational
disagreement rate, establish physical correctness, or certify robot safety.
Markdown-fenced JSON is a typed judge error on this strict path, not repaired
into a verdict. The command also does not defend against instructions embedded
in the submitted pixels or prove that a critical visible defect is absent. Full
rationales and raw provider responses are written only to the private artifact;
console output is a bounded summary.

For a matched baseline/candidate image comparison, use neutral labels and both
orders:

```bash
npa workbench vlm-eval compare-preference \
  --baseline-path <matched-image-1> \
  --candidate-path <matched-image-2> \
  --output-path <private-evidence-prefix> \
  --task "Compare two matched scene views." \
  --rubric "Prefer visible measured detail and penalize unsupported surfaces."
```

The typed SDK request is
`npa.sdk.workbench.vlm_eval.VlmPreferenceComparisonRequest`; call
`npa.sdk.workbench.vlm_eval.compare_preference`. The command is hosted API-only
and needs no local GPU. It writes `vlm_preference_comparison.json` exactly once,
retains both full provider outcomes privately, and escalates errors, unresolved
or low-confidence output, and
`order_disagreement_or_nondeterminism`. Even an order-consistent candidate
preference is an audit observation, not proof of geometry accuracy, physical
validity, or robot safety. The prompt's instruction to ignore image text is not
a defense against in-image instructions. For S3 output, private access remains
an operator/storage-policy requirement; the client uses an atomic create-only
write but does not infer bucket policy or ACL state.

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

- `rollouts/<rollout-id>/vlm_eval.json`: one structured result per rollout.
- `task_success_report.json`: aggregate report with `total_rollouts`,
  `passed_rollouts`, `success_rate`, `mean_score`, `task_success`, and the
  per-rollout `{success, score, rationale}` records.

`vlm_eval.json` is backend-neutral; inspect the payload's `backend` and
`evidence.provider` fields to distinguish real inference from a fixture.
Readers retain `vlm_eval_stub.json` only for historical bundles. Do not declare
that legacy name in new workflows.

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
item, then run the sweep below.

## Tune

Use neutral identify-then-judge task text. Ask what the frames show before
asking whether they satisfy the target; do not ask the model to confirm the
desired answer. A blank and an unrelated rollout must score low under the exact
same task-plus-rubric prompt before the positive score is usable evidence.

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
a workflow stage.

Benchmark report schema `npa_vlm_eval_benchmark_report_v2` makes calibration
errors explicit for every model/rubric/threshold configuration. Inspect its 2x2
`confusion_matrix`, `false_positive_rate`, `false_negative_rate`, and ordered
failure item IDs, then resolve each ID in that configuration's complete
`results` list. A null rate means the labeled set lacked the denominator class;
it does not mean zero errors. Dataset item IDs must be unique. Reports without
`schema_version` are legacy v1 records: their complete item results can be
recomputed, but consumers must not invent v2 fields.

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
- S3 writes fail: verify `AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud`
  and that the storage keys can read `rollouts_uri` and write `scores_uri`.
