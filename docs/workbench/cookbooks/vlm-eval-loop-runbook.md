# VLM-Eval Loop Runbook

[Cookbooks](README.md)

For a separate qualitative audit, use [rich visual review](../vlm-visual-review.md).
Its private evidence, counterbalanced comparisons, and usefulness hypotheses
never supply a completion score or pipeline gate.

For known-count inputs, `sequence` remains uniform across the full span and
`keyframes` allocates half its budget to a terminal window covering at least
the final 10%, widening when needed for unique frames. Earlier samples span
the remaining evidence; short inputs return all frames. This is deterministic
temporal sampling, not content-aware event detection. Both preserve first and
final frames when at least two are selected. Unknown-count video compatibility
is unchanged and never invents source indices or timestamps. The original
[hosted sampling failure](../evidence/vlm-frame-selection-semantics.md) remains
separate from deterministic sampler correctness.

The default deliberately favors terminal evidence when the budget is small:
three of 100 frames select `[0, 90, 99]`, and four of 1,000 select
`[0, 450, 900, 999]`. Only the first frame supplies early evidence in the
three-frame case; unsampled middle events can be missed. This is not complete
episode coverage or a demonstrated improvement in model judgment. Use the
retained selected indices and frame hashes to audit what was actually shown;
the original SO-100 failure and gray-control rationale errors remain failures.

This runbook runs the sim-to-real VLM-eval loop on the self-hosted serving path:
serve a VLM with vLLM, score rollout directories with `vlm-eval`, and write a
task-success report.

Each evaluation records the requested `model` and the endpoint's returned
`served_model`. Self-hosted servers that omit the model identity leave
`served_model` null; NPA does not infer it from the requested alias. A supplied
identity must be a nonempty string. Retain the serving deployment's checkpoint
revision separately: a model name alone does not identify its weight bytes.

Every scored response must report the exact value
`choices[0].finish_reason="stop"`. Both real backends reject incomplete or
missing completion metadata before parsing the verdict, including parseable
JSON from a truncated response. Update self-hosted adapters that omit the field.

Incomplete judge output aborts `run`, the entire `loop`, or the benchmark sweep;
it is not converted to a score-zero valid verdict. The CLI exits nonzero. In a
two-rollout loop, a completed first rollout's artifact may remain, but an
incomplete second rollout produces neither its result nor a new aggregate
`task_success_report.json`. Treat those earlier artifacts as partial evidence,
not a successful run. Do not continue promotion after a failed judge command.
Use a new run-specific output location: the evaluator does not erase artifacts
from previous runs, and an independently invoked historical-artifact reader
cannot infer which producer invocation failed. In particular, an absent
canonical report can permit a valid legacy report to be read; malformed or
ineligible **present** canonical reports never fall back to favorable legacy.

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
scores, but both real backends reject missing or non-`stop` completion metadata
before parsing. Retained evidence does not make compatible framing equivalent
to strict hosted JSON. The live
provenance lane below separately requires complete output and checks its framing.
The promotion gate rejects compatibility-only results, requiring a completed,
non-refused, strictly typed retained verdict for either backend.
None of these fields turns a visual judgment into objective task, geometry,
collision, or safety evidence.

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
requires escalation on disagreement or judge error. Both requested model IDs
must return their exact requested identity. A substituted or missing returned
model is a typed judge error, with its original response retained for review.
Models requiring incompatible generation settings are rejected before input
preparation or provider calls. Compatibility uses the shared hosted profile's
temperature support, not a separate model-name list. Kimi-K3 cannot participate in this
shared-temperature experiment. This command does not change single-judge model support.
The report is always
`audit_only`; agreement does not qualify either model, estimate an operational
disagreement rate, establish physical correctness, or certify robot safety.
The paired prompt explicitly requires one bare JSON object with no Markdown
fences or surrounding text. This output contract is identical for both judges
and included in their prompt and request hashes; it does not change the rubric
or scalar prompt. Markdown-fenced JSON remains a typed judge error on this
strict path, not repaired into a verdict. Extracted paired child results are
not scalar promotion evidence: their paired prompt does not match the scalar
grade validator's prompt. Run a scalar evaluation for that separate gate.
The command also does not defend against instructions embedded
in the submitted pixels or prove that a critical visible defect is absent. Full
rationales and raw provider responses are written only to the private artifact;
console output is a bounded summary.

Paired artifacts are created atomically without replacing an existing file or
S3 object. Local reports are `0600` regardless of umask; local symlink targets
are rejected. Use a new output prefix for each comparison. Invalid artifact
filenames fail before any provider call. To execute the configured live audit
lane, follow [the audit contract guide](../../testing/vlm-audit-live-contracts.md).

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
endpoints remain their owner's responsibility and must not be destroyed. Follow
the [run lifecycle](../../run-lifecycle.md) when canceling jobs and stopping
job-owned endpoints; this runner provisions and deletes nothing.

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
  per-rollout `{success, score, rationale}` records.

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
The producer emits v2. The reader supports v1 without sampling fields when its
prompt can still be reconstructed; it does not upgrade v1 to complete sampling.
Other schema versions fail closed.

New results retain the effective `rubric` so custom-rubric prompt and rubric
hashes can be checked. Historical v1 results without this field are accepted only
when the default rubric reproduces both hashes. The gate requires exact requested
and returned model identity for hosted results, including unregistered models.
The ordinal-grounding prompt changes that reconstruction for pre-ordinal v1 and
v2 reports, including reports with a retained custom rubric. Those reports fail
closed with `evidence_reason: digest_mismatch` and need fresh evaluation before
promotion. Preserve the old evidence unchanged; never rewrite its hashes or
claim its model saw the new instructions. Reading a legacy result filename does
not make an incompatible historical prompt eligible for promotion.
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

Add a source-matched missing-terminal
control that retains plausible progress but omits the requested outcome, an
ambiguous terminal control, and blank evidence. The default rubric instructs the
judge to assign `0.0` for missing or ambiguous terminal evidence with no
partial-progress credit. This is a prompt instruction, not an independent visual
validator. The gate still uses `score >= success_threshold`, so select a positive,
calibrated threshold. Custom rubrics replace the default instruction. Selected
stills cannot prove hidden state, continuous execution, or safety.

The production request labels each image `Frame 1`, `Frame 2`, and so on in
supplied order. These are ordinal anchors, not source indices or timestamps;
the original selected-frame labels and hashes remain in provider evidence.
The prompt asks for visible observations and rationale references to those
anchors, forbids invented times and hidden states, and distinguishes an outcome
not shown from an outcome that did not happen. Blank or unrelated images do not
establish the objects named in the task. These instructions request grounded
explanations but do not independently validate a model's explanation. Review
actual retained rationales and pixels alongside the score, including after
prompt changes.

The [narrow terminal-evidence qualification](../evidence/vlm-terminal-ordinal-qualification.md)
records four correct frozen decisions for the default hosted model while
retaining real rationale inaccuracies. It is not broad explanation-grounding
acceptance or qualification of other models.

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
For constructor compatibility, manually built reports still default to v1 even
when optional calibration fields are populated. Detect an available matrix with
`metrics.get("confusion_matrix") is not None`; the version alone does not prove
field absence.

F1 is computed as `2 * TP / (2 * TP + FP + FN)` before rounding to four decimal
places. A defined all-error result is zero; when that denominator is zero, F1
is null. Precision and recall remain separate class-dependent metrics.

The packaged benchmark includes `progress-without-terminal-fail`; its tiny
synthetic frames and prerecorded score test wiring only. Validate the exact
task, rubric, threshold, sampling and model on reviewed real inputs.

## Terminal-evidence live check

This hosted lane tests the default rubric on operator-reviewed real rollout
frames. It needs no local GPU or cloud resource creation. Stage four private
image directories before calling the model: a complete real sequence, a strict
prefix of that sequence ending during plausible progress, a real ambiguous
terminal sequence (for example, an occluded final placement), and repeated
blank frames. Use neutral filenames and identical task text. Review the actual
pixels and freeze labels before inference; the harness verifies byte bindings,
not the truth of the operator's visual labels.

The lane opts in through its private configuration; without it, collection skips
this lane even when global integration is enabled. Once configured, it is
fail-closed: an invalid configuration, missing credential, mismatched model or
frames, incomplete response, or incorrect control verdict fails. A passing result does not prove
physical completion or safety. Before spending inference tokens, verify the
hosted credential and list models available to that same key:

```bash
npa/.venv/bin/python -m npa workbench token-factory verify
npa/.venv/bin/python -m npa workbench token-factory models
```

Select the exact returned vision-capable model ID; do not substitute an alias
after this check. Credential verification and model listing are not inference
or semantic acceptance evidence.
These commands use Token Factory configuration. Ensure any VLM credential or
endpoint overrides select that same verified identity and service; checking one
credential does not validate another override used by the live lane.

Set `NPA_VLM_TERMINAL_LIVE_CONFIG` to an absolute path for an owner-only (`0600`)
JSON file outside the checkout, containing exactly these fields:

| Field | Required value |
| --- | --- |
| `model` | Exact hosted vision model ID |
| `task` | One explicit neutral identify-then-judge task for all controls, not the auto-discovery sentinel `sim-to-real` |
| `max_frames` | Integer at least as large as the largest staged frame set |
| `success_threshold` | Frozen positive threshold no greater than 1 |
| `output_dir` | New private local directory; existing paths are refused |
| `cases` | Objects named `complete`, `missing-terminal`, `ambiguous-terminal`, and `no-evidence` |

The following is a shape example, not an executable frozen control set. Replace
each hash list with all actual selected-frame digests and set `max_frames` to
retain every staged frame:

```json
{
  "model": "<exact-hosted-model-id>",
  "task": "<identify-then-judge task>",
  "max_frames": 4,
  "success_threshold": 0.8,
  "output_dir": "/absolute/private/new-output-directory",
  "cases": {
    "complete": {
      "input_path": "/absolute/private/complete-rollout",
      "frame_sha256": ["<selected-frame-sha256>"]
    },
    "missing-terminal": {
      "input_path": "/absolute/private/truncated-rollout",
      "frame_sha256": ["<selected-frame-sha256>"]
    },
    "ambiguous-terminal": {
      "input_path": "/absolute/private/ambiguous-rollout",
      "frame_sha256": ["<selected-frame-sha256>"]
    },
    "no-evidence": {
      "input_path": "/absolute/private/blank-rollout",
      "frame_sha256": ["<selected-frame-sha256>"]
    }
  }
}
```

Each case contains only `input_path` (its local image directory) and
`frame_sha256` (ordered SHA-256 digests of normalized submitted PNG bytes).
Compute the digests using `select_rollout_frames` with `frame_selection="sequence"`
and the frozen `max_frames`, then review and preserve that config. Every staged
frame must be selected. `max_frames` is an integer of at least 2, each case must
contain at least two frames, and `success_threshold` must be in `(0, 1]`.
The truncated control must be a shorter byte-identical prefix of the complete
one; the ambiguous control must differ from both the complete sequence and the
truncated prefix. The blank
control must repeat one frame hash absent from the complete sequence. Frame
counts and sampling differ between those controls; this is a semantic regression
check, not a one-factor experiment. Credentials resolve from `VLM_EVAL_API_KEY`,
`NEBIUS_TOKEN_FACTORY_KEY`, `OPENAI_API_KEY`, or the private NPA credential store,
never through the test JSON.

From the repository root, after selecting task-scoped private NPA configuration:

```bash
: "${NPA_VLM_TERMINAL_LIVE_CONFIG:?Set the private frozen control configuration}"
npa/.venv/bin/python -m npa workbench health preflight --checks token_factory --json
NPA_INTEGRATION_E2E=1 \
NPA_VLM_TERMINAL_LIVE_CONFIG="$NPA_VLM_TERMINAL_LIVE_CONFIG" \
npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_vlm_terminal_evidence_live.py -q -n 0
```

Run serially (`-n 0`), not with pytest-xdist workers: the private panel output is
shared by all four cases. Without `NPA_VLM_TERMINAL_LIVE_CONFIG`, this lane skips,
including under the broad `make test-e2e` target. A supplied missing, malformed
or invalid file fails closed; no default panel is substituted.

Configured acceptance requires all four cases to pass with zero skips or
deselections. Neither disabled integration nor absent lane configuration is live
evidence. The cases execute separately
so a failed verdict does not prevent the remaining frozen controls from running.
Keep every outcome and raw provider response privately; never rerun only failed
cases to assemble a passing panel.

Review all four private result files after a pass. Each must retain `backend`
as `api`, the configured `served_model`, non-null request and provider evidence,
HTTP 200, `finish_reason` `stop`, and hashes matching the frozen frames and
default rubric. Only `complete` may pass; every other case must return score
`0.0` and `success: false`. A failure means this model and these frozen controls
did not meet the gate; do not lower the threshold, relax the rubric, or replace
the evidence claim to make it pass.

Keep logs private. Each successfully parsed response is retained before that
case's visual expectations are checked. Model substitution, changed frame bytes,
a false pass, or any nonzero negative-control score fails the lane. Transport or
parser failures fail their case without a result artifact; other retained results
remain on disk. Ordinary pytest continues the remaining cases after a failure.
An interrupted or fail-fast run that omits a control cannot qualify the panel.
Transport and parser failures are not completed visual judgments. Do
not retry until a failure passes or edit the frozen threshold to fit results. Independent review
must still compare each rationale with the retained pixels: a correct label
with an invented visual explanation is not accepted evidence. This test is
separate from the nightly hosted smoke suite because it needs reviewed private
rollout inputs. A skipped or unconfigured run does not validate the rubric.
Retain the config, responses, and reviewed inputs; this lane creates no compute
resources to tear down. Four controls cannot qualify a model or estimate error
rates. Historical measurements in the [earlier control
review](../evidence/vlm-missing-terminal-control-review.md) do not validate a
changed rubric or current commit.

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

Endpoint and credentials are a matched choice: the default Nebius route supports
the usual Nebius environment/file credential, but never falls back to
`OPENAI_API_KEY`. Ambient API base URLs require explicit `--endpoint-url`.
For another provider, specify that endpoint and the populated `--api-key-env`
variable; no other credential is substituted if that variable is absent. See
[the preference credential contract](../../../npa/README.md#blinded-vlm-preference-audits)
for exact defaults and environment names.

Both image orders receive the same neutral typed JSON Schema. Preference
request manifests use `npa_vlm_preference_request_evidence_v1` and declare an
`independent-image-pair` input contract. Their ordered A/B frame digests identify
the exact submitted pixels, including two entries for identical images. These
are two independent images, not samples from one rollout: preference evidence
does not claim a shared source count, timeline, or sampling coverage. The private
report retains the baseline/candidate mapping separately from the neutral request.
Scalar evaluation evidence uses a separate schema.

Preference
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
