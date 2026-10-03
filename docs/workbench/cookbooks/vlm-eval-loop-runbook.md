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
- hashes of the prompt, rubric, and secret-free request manifest;
- request time, endpoint role, HTTP status when available, latency, finish
  reason, provider request ID and usage when returned;
- the exact provider response body, its SHA-256 hash, and parser version.

The request manifest intentionally excludes authorization, endpoint addresses,
input/output locations, prompts, and base64 image data. It contains enough
information to recompute what was submitted without copying pixels or secrets.
The full result still belongs in private run storage because the provider
response and existing task fields can describe operator data.

`stub` results and `--score` overrides do not contain provider evidence. They are
wiring checks, never visual proof. Real `benchmark` reports retain the evidence
for every case so calibration failures and model disagreement remain inspectable.
If a model wraps one complete JSON object in a single Markdown JSON fence, the
parser removes only that transport wrapper and appends `+markdown-fence-v1` to
the retained parser version. Hosted `api` evaluation rejects surrounding prose,
duplicate keys, non-finite numbers, invalid types, incomplete output and model
substitution. The `self-hosted` backend retains its legacy compatibility parser:
it can extract embedded JSON, accept duplicate keys and coerced types, clamp
scores, and return a verdict with missing or non-`stop` completion metadata.
Retained evidence does not make such a verdict eligible for promotion. The live
provenance lane below separately requires complete output and checks its framing.
None of these fields turns a visual judgment into objective task, geometry,
collision, or safety evidence.

The serialized result retains the effective `rubric`, so the exact prompt can
be reconstructed from `task`, `rubric`, `frame_selection`, and `frame_count`.
`passed` is always `score >= success_threshold`, using the serialized score
rounded to four decimal places for real, stub, and override evaluations. When
a real backend actually returns a `success` boolean, the result records it as
`provider_success` and reports whether it agrees in
`provider_success_matches_score_gate`.
Self-hosted responses that omit the boolean leave both fields null rather than
presenting a score-derived fallback as provider output. Legacy non-boolean
values such as `"true"` are likewise not promoted to provider booleans. A real
disagreement is calibration evidence, not permission to replace the
score-derived label. Before reviewing thin geometry or skeletons, compare
retained submitted-frame dimensions with the source because normalization can
remove the defect.

## Live provenance verification

The operator lane runs `test_vlm_served_model_live.py` against an existing GPU
endpoint. It fails if configuration, authentication, expected model readiness,
or actual inference is missing. A skipped or empty test run cannot pass. This
is separate from the nightly hosted Token Factory suites, which need no GPU.

Before provisioning a dedicated endpoint, prove credentials with
`npa workbench health preflight --checks nebius --json` and verify exact model
payload access before any download. Use the
[access preflight](../../../skills/atomic/access-approval/SKILL.md) for gated
weights and record the checkpoint revision, serving image digest, selected GPU
family/count and resource ownership privately. Use a compatible serving runtime;
the model name returned by inference does not identify its checkpoint bytes.
Provision and clean up only resources owned by this validation run.

Stage a local rollout fixture with decodable images or video outside the
checkout. Its actual normalized frames will be reloaded and hashed to verify
the request evidence. Create an owner-only output directory and a fresh local
JSON result filename; S3 fixtures/results are not supported by this verification
lane. Keep the task, fixture, provider response, and result private.

Set `NPA_VLM_PROVENANCE_LIVE_CONFIG` to the absolute path of an owner-only
(`0600`) JSON file outside the checkout. Required keys are `input_path`
(absolute local fixture path), `output_path` (absolute fresh `.json` filename
in an existing `0700` directory), `endpoint_url` (OpenAI-compatible `/v1` base
or `/v1/chat/completions` URL), `model` (requested ID), `expected_served_model`
(actual server ID), and `task`. Optional `api_key_env` defaults to
`VLM_EVAL_API_KEY`; supply the endpoint credential in that environment variable,
never in the file or URL. The endpoint must expose authenticated `/v1/models`
and `/v1/chat/completions` routes.

From the repository root, with the configuration and credential in the process
environment:

```bash
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python \
  npa/scripts/vlm_provenance_live_recheck.py \
  --evidence-dir "$NPA_PRIVATE_EVIDENCE_DIR"
```

Use a fresh absolute evidence directory outside the checkout for each run.
`receipt.json` contains test counts, source hashes, and sanitized status. The
verdict stays at the configured `output_path`. Success requires an executed
provider call, HTTP 200, `finish_reason=stop`, the expected returned model,
recomputable frame/manifest/response hashes, and saved-result readback. A model
listing is readiness evidence only. This lane proves traceability of inference;
it does not prove a policy succeeded or a scene is physically safe.

The runner provisions and deletes nothing. After collecting the result and
receipt, cancel run-owned jobs, stop the endpoint and destroy run-owned compute
using the [run lifecycle](../../run-lifecycle.md). Preserve evidence and shared
resources. Cleanup remains required when validation fails.

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

## Blinded preference audits

For a matched baseline/candidate image comparison, use neutral labels and both
orders:

```bash
npa workbench vlm-eval compare-preference \
  --baseline-path "<matched-image-1>" \
  --candidate-path "<matched-image-2>" \
  --output-path "<private-evidence-prefix>" \
  --task "Compare two matched scene views." \
  --rubric "Prefer visible measured detail and penalize unsupported surfaces."
```

The typed SDK request is
`npa.sdk.workbench.vlm_eval.VlmPreferenceComparisonRequest`; call
`npa.sdk.workbench.vlm_eval.compare_preference`. The command is hosted API-only
and uses model-specific request settings identically for both image orders. Kimi-K3
uses low reasoning effort and JSON output without a temperature field;
MiniMax retains its existing request settings. Neither path adds an output-token cap
and needs no local GPU. It writes `vlm_preference_comparison.json` exactly once,
retains both full provider outcomes privately, and escalates errors, unresolved
or low-confidence output, and
`order_disagreement_or_nondeterminism`. Even an order-consistent candidate
preference is an audit observation, not proof of geometry accuracy, physical
validity, or robot safety. The prompt's instruction to ignore image text is not
a defense against in-image instructions. For S3 output, private access remains
an operator/storage-policy requirement; the client uses an atomic create-only
write but does not infer bucket policy or ACL state.

Both image orders receive the same neutral typed JSON Schema. Preference
image normalization copies only RGB pixels, discarding embedded profiles and
metadata that could reveal a source role. Visible text remains part of the
image and is not removed or trusted as an instruction. The strict
response contract requires a nonempty `critical_defects` list for each image:
when no critical defect is visible, use a truthful absence statement without
inventing defects. Positive observations belong in `observable_support`.
The exact `uncertainty` field must contain text describing what the views cannot
establish. Schema validity alone does not establish that observations are grounded
in the pixels; retain both outcomes for review.

An omitted or empty rubric uses the shared task-completion rubric; supply
`rubric` or `rubric_path` when comparing other visible qualities. The effective
rubric is retained in the report. The SDK returns that report and retains a
private journal; unlike the CLI, it leaves the canonical report write to the
caller:

```python
from dataclasses import asdict
from npa.sdk.workbench.vlm_eval import compare_preference
from npa.workbench.vlm_eval import write_preference_report

report = compare_preference(request)
write_preference_report(asdict(report), result_uri=report.result_uri)
```

If the caller stops after inference, the journal's `report-ready.json` retains
the complete report. Pass that JSON payload to `write_preference_report` with
the original result URI to finish the write without another model call. The
writer refuses an existing canonical report. Preserve the journal; rerunning
the comparison against that output is deliberately refused.

The scheduled hosted lane runs real visual controls; see the
[live audit contract](../../testing/vlm-audit-live-contracts.md) for configuration,
strict execution counts, and evidence privacy.
