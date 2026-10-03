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
metadata to compare against separately retained source media. A syntactically
valid SHA-256 digest establishes only format. Recomputing the manifest hash
checks the recorded metadata and claimed digests; it does not independently
prove that those digests describe the submitted pixels. For payload binding,
normalize the retained source frames to RGB PNGs (at most 768 pixels per side)
and recompute their hashes, dimensions, and byte counts. Hashing the original
JPEG or video file is not equivalent to hashing the normalized submitted PNG.
Its `sampling` block records the strategy, requested frame limit, selected
indices and timestamps, source count, and whether index and timestamp coverage
are complete. Unknown video metadata stays null; a generated extraction ordinal
is never presented as a source frame index. `coverage_complete` means every
submitted frame has auditable source-index metadata, not that every available
source frame was submitted.
This producer emits `npa_vlm_eval_evidence_v2`; legacy v1 records have no
source-sampling contract and must not be interpreted as complete sampling.
The v2 aggregate `source_kind` is null for absent, unrecognized, or mixed kinds,
and `coverage_complete` is false even if their counts and indices agree.
`source_count` is retained only when all frames agree on a non-null count.
`timestamps_complete` is boolean for a known uniform video source and null
otherwise. These fields do not upgrade or reinterpret existing v1 artifacts.
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
The promotion gate rejects compatibility-only results, requiring a completed,
non-refused, strictly typed retained verdict for either backend.
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

### Served-model sampling live lane

To verify this against your existing authenticated GPU endpoint, set
`NPA_INTEGRATION_E2E=1` and point `NPA_VLM_PROVENANCE_LIVE_CONFIG` at a private
owner-only JSON file containing `input_path` (absolute local media path),
`output_path` (a new absolute JSON filename in an existing owner-only directory),
`endpoint_url`, `model`, `expected_served_model`, and `task`. Supply credentials
through the environment variable named by `api_key_env` (default
`VLM_EVAL_API_KEY`). Both gating variables are unset by default. Install the
`dev` extra, `ffmpeg`, and `ffprobe`, then run from the repository root:

Use a bare HTTP(S) endpoint, `/v1` base, or full `/v1/chat/completions` URL
without embedded credentials, query, or fragment. The endpoint must expose
authenticated `/v1/models` and `/v1/chat/completions` routes. The serving model
identifier is checked, but does not by itself identify checkpoint bytes.

```bash
export NPA_INTEGRATION_E2E=1
export NPA_VLM_PROVENANCE_LIVE_CONFIG="<private-config.json>"
npa/.venv/bin/python npa/scripts/vlm_provenance_live_recheck.py \
  --evidence-dir "<new-private-evidence-directory>"
```

The runner requires all ten real inference cases to pass: one operator input
plus `final`, `keyframes`, and `sequence` sampling over six-frame image, NumPy,
and lossless-video inputs. Known source pixels and timestamps provide an
independent oracle for normalized hashes and selected indices. Fixture creation
alone is not inference evidence. Missing credentials, missing video tools,
skipped cases, partial collection, or failed teardown hooks fail the lane.
The shared provenance runner verifies authenticated expected-model readiness
before inference. Readiness alone cannot pass the lane. Configuration and
outputs must remain outside the checkout; the custom result must not collide
with the runner's `receipt.json` or `pytest/` paths. Private output includes the
configured custom result, a sanitized `receipt.json`, and per-case inputs and
provider results under `pytest/`. Preserve both the custom result and the
complete evidence directory. The receipt binds the sampling helper source.
Every case also checks exact bare/fenced parser tagging, provider-success versus
score-derived gate agreement, and hashes reconstructed from the effective rubric.

The endpoint lifecycle belongs to the operator job. Before creating compute,
use a dedicated project and task-scoped NPA configuration, prove its selected
profile with `npa workbench health preflight --checks nebius`, and verify exact
model payload access before fetching gated weights. Record the model revision,
serving image digest, GPU product/count, and private deployment identity. Verify
authenticated endpoint readiness, run the lane, then collect results and tear
down only job-owned serving compute in the job's cleanup path, including when
inference fails. Retain the cleanup receipt separately: this runner consumes an
already provisioned endpoint and cannot certify cloud teardown. Reused shared
endpoints remain their owner's responsibility and must not be destroyed.

The hosted Token Factory nightly runner has a separate credential and workload
contract. It does not execute this GPU lane. Scheduling requires an operator
job with the private configuration, access checks, endpoint lifecycle, and
cleanup above; adding this suite to hosted `SUITES` without them is insufficient.
These checks establish request traceability, not color recognition accuracy or
physical task completion.

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
  per-rollout score records, including served model identity and provider
  evidence. The report also emits
  `independent_human_label_calibration_established: false` and ordered
  `limitations` so JSON-only consumers can see that the gate uses the mean
  score, samples rather than continuous behavior, and does not establish
  physical correctness or safety. Stub reports state that no VLM call occurred.

Each direct result also emits `provider_call_made`: real backend execution sets
it to true; stub and caller-supplied score overrides set it to false. Every path
retains `independent_human_label_calibration_established: false`.

`vlm_eval.json` is backend-neutral; inspect the payload's `backend` and
`evidence.provider` fields to distinguish real inference from a fixture.
Readers retain `vlm_eval_stub.json` only for historical bundles. Do not declare
that legacy name in new workflows.

For external scripts, dashboards, and workflow consumers migrating to this version:

1. Write and declare `vlm_eval.json` for new directory or object-prefix outputs,
   including each rollout subdirectory. No legacy alias or duplicate is emitted.
2. Read the canonical filename first. Fall back to `vlm_eval_stub.json` only when
   the canonical object is absent. If it exists but is malformed, empty, unreadable,
   or fails validation, report that failure; never substitute a stale legacy score.
3. Preserve explicitly supplied `.json` output paths exactly. Custom paths do not
   receive a renamed file, and `task_success_report.json` remains unchanged.
4. Determine fixture versus inference status from the payload and retained evidence,
   never from either filename. Historical bundles can keep their original names.

The data-factory `grade_gate` requires consistent retained inference evidence
before a VLM result can promote a checkpoint. Stub results, score overrides, and
historical reports without provider evidence produce `loop_back` with an explicit
reason. The gate checks submitted-frame metadata, request and response hashes,
and agreement between the retained response and serialized result. These checks
establish internal consistency, not provider authentication or visual correctness.
Frame digests are checked for valid SHA-256 format and binding to the request
manifest; this gate does not fetch image bytes to recompute their hashes. Consumers
that need payload verification must retain and independently hash the submitted
normalized images. Schema-v2 sampling counts, indices, timestamps, and coverage
flags must agree with the frame metadata. A known, uniform source kind, a source
count, and in-range selected indices are required for `coverage_complete: true`;
unknown or mixed source kinds normalize to null with incomplete coverage.
The producer emits v2. Historical v1 evidence remains valid without sampling
fields and is not upgraded to complete sampling. Other schema versions fail closed.

New results retain the effective `rubric` so custom-rubric prompt and rubric
hashes can be checked. Historical v1 results without this field are accepted only
when the default rubric reproduces both hashes. The gate requires exact requested
and returned model identity for hosted results, including unregistered models.
Invalid evidence retains `reason: vlm_provider_evidence_invalid` for existing
consumers and adds `evidence_reason` to distinguish schema, request, digest,
sampling, frame metadata, provider response/transport/metadata, and verdict failures.

Promotion eligibility is stricter than the legacy self-hosted reader. Both VLM
backends must retain an explicit `finish_reason: stop`, no provider refusal, and
one complete JSON object, optionally inside a complete Markdown JSON fence.
The verdict must contain a boolean `success`, a finite numeric `score` in `[0, 1]`,
and a nonempty string `rationale`. Duplicate keys, surrounding prose, partial
fences, clamped scores, and coerced field types cannot promote a checkpoint.
The self-hosted reader still parses its historical compatibility inputs and
records its original parser version; obtaining a score through that reader does
not make the result eligible for promotion. An absent completion status also
blocks promotion. Missing self-hosted request/model identity metadata remains
compatible when the completion and other evidence satisfy the gate.

These refusals retain the public `vlm_provider_evidence_invalid` reason.
`evidence_reason` distinguishes `provider_completion_incomplete`,
`provider_completion_filtered`, `provider_completion_refused`,
`provider_refusal_invalid`, and `provider_verdict_invalid`. Duplicate keys in the
retained response envelope produce `provider_response_invalid`.
Provider `success` remains in the retained response; optional serialized
`provider_success` fields are checked when present. The numeric score and
threshold still determine the score gate. The Cosmos Evaluator contract is unchanged.

Read the report:

```bash
aws s3 cp "s3://${NPA_S3_BUCKET}/sim-to-real/${RUN_ID}/scores/task_success_report.json" -
```

Use `task_success` as the coarse gate, then inspect low-score rollouts and their
rationales before iterating on policy, simulation, or rubric. It is a
mean-score gate, not the per-rollout `success_rate`.

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

Benchmark reports also set
`independent_human_label_calibration_established: false`: the manifest accepts
caller labels but does not establish their human authorship or independence.
Accuracy, agreement, precision, recall, F1, and confusion counts describe that
one labeled dataset; they are not operational error rates or evidence of
generalization, physical correctness, or safety. Report limitations identify
cases whose `score_source` is `fixture` or `stub` as dry-validation or wiring
inputs rather than VLM evidence.

Manifests may also declare `evidence_scope` as exactly `unspecified` or
`illustrative_only`, plus an ordered `limitations` array. Reports preserve them
as `dataset_evidence_scope` and `dataset_limitations`. The packaged default is
`illustrative_only`: its four inputs are synthetic 2x2 color swatches whose
caller labels are color-correlated, so its metrics prove benchmark wiring only.
Custom manifests that omit the metadata remain `unspecified`; omission does not
certify task validity.

These disclosure keys are additive. Consumers that reject unknown JSON keys
must update their schema; consumers that ignore unknown keys retain the previous
fields and values.

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
