---
name: vlm-eval
description: Use to score rollouts with a vision-language model and turn the score into a pipeline gate — single rollout, prefix-wide loop, rubric/threshold benchmark sweeps, backend selection (self-hosted, api, stub), and judging against a plan an earlier stage wrote.
---

# VLM eval (scoring rollouts and gating pipelines)

`vlm-eval` answers "did this rollout complete the task?" as a number, then turns
that number into a gate. It is the judging half of the loop whose generating half
is Cosmos/Genesis/Isaac rollouts and whose reasoning half is
`skills/tools/token-factory/SKILL.md`.

## Separate rich visual audit

Use `npa workbench vlm-eval review-visual` for a private qualitative record,
with required `--input-path`, `--output-path`, `--model`, and neutral `--task`.
Its SDK is `npa.sdk.workbench.vlm_eval.review_visual`. Optional `--baseline-path`
sends both independently sampled sources in both neutral A/B orders. Keep
task evidence, content fidelity, reviewability, subjective impressiveness,
and usefulness hypotheses separate; dimension-signature disagreement requires
escalation. Matching signatures do not establish semantic agreement of prose;
inspect both retained outcomes, including contradictory text.
This record never affects the normalized score or gate.

Use a fresh private output identity and preserve consumed journals after
failure; the mechanism will not replay an output after transport starts.
Inspect actual provider image capacity before paired inference. References and
matched-view metadata remain unverified and private. See
`docs/workbench/vlm-visual-review.md` for options, artifacts, and limitations.

## Pick the right command

```bash
npa workbench vlm-eval run   --input-path <one-rollout>  --output-path <eval.json>
npa workbench vlm-eval loop  --input-path <prefix>       --output-path <prefix>
npa workbench vlm-eval benchmark --dataset <manifest> --output <report.json>
npa workbench vlm-eval compare-judges --input-path <one-rollout> \
  --output-path <prefix> --primary-model <model-a> --secondary-model <model-b>
npa workbench vlm-eval status
npa workbench vlm-eval list
npa workbench vlm-eval workflow
```

**`run` scores exactly one rollout.** It discovers frames recursively, so if you
point it at a prefix holding many rollouts they blend into a single meaningless
score. That is the mistake `loop` exists to prevent: `loop` treats each directory
under the prefix as its own rollout, scores each, and writes per-rollout results
plus an aggregate task-success report.

**`benchmark` sweeps configuration, not data.** Use it to choose a threshold,
rubric, or model against a labeled set before you trust any of them in a gate.

## Backends

`--backend` is `self-hosted` (default), `api`, or `stub`.

- `self-hosted` — an OpenAI-compatible server you run, addressed with
  `--endpoint-url`. This is the GPU-bearing path.
- `api` — a hosted OpenAI-compatible endpoint; the key comes from the environment
  variable named by `--api-key-env` (default `VLM_EVAL_API_KEY`). Point this at
  Token Factory for a zero-GPU judge. The standard Token Factory key in
  `~/.npa/credentials.yaml` is also resolved when no key environment variable is
  set.
- `stub` — deterministic, no model call. For wiring tests and CI only; a stub
  score is never evidence about a policy.

`--endpoint-url` accepts either a base URL or a full `/chat/completions` URL.
Default model is `Qwen/Qwen2-VL-7B-Instruct`; `--timeout-s` defaults to 120.

## Evidence retained by real backends

A successful `api` or `self-hosted` call writes an `evidence` object alongside
the scalar result. It records hashes and dimensions for the exact normalized
frames sent; source kind, index, count, and video timestamp when known; prompt
and rubric hashes; a secret-free request-manifest hash; requested/returned model
identity; request time; finish reason; latency; usage and provider request ID
when returned; plus the exact provider response and its hash. The manifest's
`sampling` block records the requested strategy and frame limit, selected
indices/timestamps, and whether coverage is complete. Unknown source metadata
stays null rather than turning extraction ordinals into source indices.
`coverage_complete` means selected-frame provenance is complete, not that all
available source frames were sent. Real benchmark reports retain this record per
case.

Directory and object-prefix outputs use the backend-neutral
`vlm_eval.json`; inspect the payload's `backend` and provider evidence to
distinguish real inference from fixtures. An explicitly supplied `.json` output
path remains unchanged. Readers accept the old `vlm_eval_stub.json` name only
for historical bundles; new workflows must not declare it.

Recompute these hashes before accepting a result. The request manifest must not
contain authorization, endpoints, local/S3 paths, prompts, base64 bytes, or data
URIs. Keep the whole result private because the existing task and provider
rationale can still describe operator data.

`evidence: null` means no provider call occurred, as with `stub` or `--score`.
It cannot support a visual claim. Hosted `api` evaluation rejects provider
refusal, truncation, filtering, malformed JSON, and any served-model mismatch.
Single results, loop rows, and benchmark cases disclose requested/served identity
and `served_model_match_enforced`. All current hosted profiles require exact
identity; self-hosted responses and non-provider scores report false. Never
infer enforcement from a model name, a score, or an unverified returned alias.
Aggregate and case model names use the same effective default/environment
resolution. On stub/override paths, `requested_model` names configuration only;
it does not mean a request occurred. Promotion validates a present enforcement
boolean against retained backend/identity evidence, without inventing the field
for historical reports that omit it.
One complete JSON object wrapped only in a Markdown JSON fence is transport
de-framed; its retained parser version ends in `+markdown-fence-v1`. Do not
accept surrounding prose, trailing output, duplicate keys, invalid types, or a
partial fence on that hosted path.

Both real backends require the exact response value
`choices[0].finish_reason="stop"` before parsing a verdict. Truncated, filtered,
tool-call, aborted, empty, malformed, or missing completion metadata is rejected.
Self-hosted adapters that omit this metadata must emit the standard field.

An incomplete response aborts the whole `loop` or benchmark sweep and makes the
CLI fail nonzero; never substitute a fabricated score-zero valid verdict.
Previously completed per-rollout artifacts may remain, but no new aggregate
report is written after the rejection. Retain these as partial evidence and do
not run promotion after the failed producer. Use new run-scoped output paths;
old artifacts are not deleted. A standalone historical report reader can accept
a valid legacy file when the canonical file is absent, but cannot establish
that a new producer invocation succeeded. Present malformed/ineligible canonical
reports remain authoritative and fail closed, without favorable legacy fallback.

The `self-hosted` backend preserves legacy JSON framing, including embedded
objects and duplicate keys, while requiring literal boolean `success`, a finite
numeric `score` in [0, 1], and a nonempty string `rationale`. Invalid verdict
fields fail without coercion or clamping. Its parser version is
`npa_vlm_eval_compatible_json_v2`, with the framing suffix when applicable.
Retained self-hosted reports tagged `npa_vlm_eval_compatible_json_v1` (including
its framing suffix) no longer satisfy the current promotion parser contract,
even if their content has valid literal fields. The gate returns `loop_back`
with `provider_metadata_mismatch`; this can mean a superseded parser, not
tampering. Preserve the old artifact and obtain a new evaluated report when
promotion is needed; never edit its parser tag to simulate new execution.
Malformed score/type/success verdicts also abort the whole loop/sweep; no score
is manufactured. Correct the serving output contract before a new evaluation.
Completion evidence does not certify promotion eligibility. The operator
provenance lane separately requires HTTP
200, `finish_reason=stop`, expected served identity and verifiable framing.
The promotion gate separately requires a completed, non-refused, strictly typed
verdict for either backend. Compatibility-only parsing, surrounding prose,
trailing output, duplicate keys, invalid types and partial fences cannot promote.
This evidence proves judge traceability, not physical correctness or safety.

`compare-judges` is an API-only audit path for consequential or disputed
reviews. It selects and normalizes frames once, sends an otherwise identical
request to two explicitly distinct hosted models, and writes
`vlm_judge_disagreement.json`. It retains both complete outcomes, never emits a
mean score, and sets `passed=false` plus `escalation_required=true` when the
score-derived verdicts disagree or either judge errors. The artifact is always
`deployment_status: audit_only`: agreement does not qualify either judge, and a
weak judge can make disagreement common without making the scene intrinsically
ambiguous. Unlike ordinary single-judge compatibility parsing, this path
rejects Markdown-fenced JSON as a typed judge error instead of transforming the
output. It also does not defend against in-image instructions or prove a
critical visible defect absent. Full provider responses stay in the private
artifact; CLI output is a bounded summary.

`passed` and `status` come only from `score >= success_threshold`, using the
serialized score rounded to four decimal places for every backend and override.
The model's own `success` boolean is retained as `provider_success` when the
response includes it, and `provider_success_matches_score_gate` exposes
disagreement. Missing and non-boolean success values are rejected on both
real backends. Stub and override provenance fields remain null.
Never substitute the provider boolean for the score-derived gate.

## Read calibration and limitation evidence correctly

Every full result also emits
`independent_human_label_calibration_established: false` and ordered
`limitations`. The name is deliberate: `false` says the artifact does not
establish independent-human-label calibration; it does not claim a caller could
never have supplied human labels. Results produced by `stub` or `--score`
identify those values as wiring or dry-validation inputs for which no VLM call
occurred.

Promotion rejects an explicit `provider_call_made` value unless it is literal
`true`, even when retained evidence is internally consistent. Historical reports
without this newer field remain subject to the full inference-evidence checks;
the field and hashes do not authenticate a provider or establish model quality.

Loop and benchmark reports carry report-level limitations. They are additive
JSON keys, so strict consumers that reject unknown keys need a schema update.

## Scoring controls that actually change the verdict

```bash
npa workbench vlm-eval run \
  --input-path s3://<bucket>/runs/<id>/rollout/ \
  --output-path s3://<bucket>/runs/<id>/eval.json \
  --task "pick and place the cube" \
  --backend api --model <model> --api-key-env NEBIUS_TOKEN_FACTORY_KEY \
  --frame-selection keyframes --max-frames 4 \
  --rubric-path ./rubric.txt \
  --success-threshold 0.8
```

- `--frame-selection` is `final`, `keyframes` (default), or `sequence`. `final`
  cannot distinguish "reached the goal" from "was already there"; `sequence`
  costs the most tokens. `keyframes` is the default for a reason.
- `--max-frames` (default 4) bounds both cost and how much of the episode the
  judge can actually see. A four-frame view of a long episode judges a summary.
- `--rubric` / `--rubric-path` carry the scoring instructions. The default rubric
  reserves 1.0 for clear completion and 0.0 for clear failure, with intermediate
  values for partial progress, and penalizes unsafe or ambiguous outcomes. It
  instructs the judge to assign `0.0` and `success: false` when the requested
  terminal state is missing or ambiguous in the supplied frames. That instruction
  overrides partial-progress credit: approach, contact, grasp, lift, transfer,
  or disappearance alone cannot prove placement, release, stability, or completion.
  This is a prompt instruction, not an independent visual validator; the gate
  still uses only the returned score and threshold. Custom rubrics replace it.
- Write `--task` as identify-then-judge: ask what the frames show before asking
  whether they meet the target. A leading confirmation question such as "does
  this show X rather than a blank?" can make an unrelated negative control pass.
  Run blank and unrelated controls through the exact same task-plus-rubric prompt.
  Production requests interleave supplied-order `Frame N` labels with images and
  ask rationales to cite those ordinals, not invented timestamps or source indices.
  Original frame labels and hashes remain unchanged in evidence. Check that the
  rationale distinguishes missing evidence from an event that did not happen;
  a correct numeric label alone is not grounded acceptance.
- `--success-threshold` (default 0.8) is the gate. In `loop` it applies to the
  **mean** score across rollouts, which is a coarser claim than per-rollout
  success — do not report it as a per-rollout success rate. The loop report
  repeats that caveat in machine-readable `limitations`.
- `--score <float>` overrides the score and skips the VLM call entirely. It exists
  for tests and dry validation. Never use it to produce a result you then report.

## Judging against a plan instead of a fixed task

`--task-from <reasoning-artifact>` reads the task from the artifact's `analysis`
field rather than from `--task`. This is what makes the scene-to-judge pattern
work: a Cosmos reasoner writes a plan for the scene, and the judge scores the
rollout against *that* plan instead of a hardcoded string. The workflow toolRef
is `workbench.vlm_eval.judge_against_plan`.

## Choosing a threshold honestly

```bash
npa workbench vlm-eval benchmark \
  --dataset <manifest.json> --output <report.json> \
  --thresholds 0.5,0.8,0.9 \
  --rubrics default,@./strict-rubric.txt \
  --models <model-a>,<model-b> \
  --backend api
```

`--rubrics` accepts names from the dataset, inline text, or `@file` paths.
`--dataset` defaults to a packaged sample fixture, which is useful for proving
the sweep runs but tells you nothing about your task. `--use-fixture-scores`
honors recorded `fixture_score` values for non-stub backends; stub always uses
them when present.

Benchmark `expected_label` values are caller-supplied; the manifest does not
establish independent human authorship or independence. Reports therefore keep
`independent_human_label_calibration_established` false and qualify accuracy,
agreement, precision, recall, F1, and TP/TN/FP/FN as measurements of that one
dataset, not operational error rates or evidence of generalization, physical
correctness, or safety. Limitations name `fixture` and deterministic `stub`
score sources when they occur so mixed reports do not imply those cases made a
model call.

Every benchmark must contain at least one pass label and one fail label, and
resolved item IDs must be unique. Both conditions are checked before frame
selection or evaluator/backend activity. Reports include specificity and
balanced accuracy in addition to the existing confusion counts and metrics;
configuration ranking uses balanced accuracy first so an all-positive judge
does not win on an imbalanced set.

The packaged `isaac-agency` alias is a CPU-only calibration control:

```bash
npa workbench vlm-eval benchmark \
  --dataset isaac-agency \
  --output /tmp/isaac-agency-benchmark.json \
  --backend stub \
  --frame-selection sequence \
  --max-frames 6 \
  --thresholds 0.5
```

It pairs the same six exact stylized frames with a true state claim (the red
cube becomes elevated) and a false agency claim (the robot grasps and lifts the
cube). Before scoring, its opt-in structural check verifies frame order and
hashes, complete color masks, signed vertical motion, and the absence of actor
proximity. The exact preselected frames and any task resolved from rollout
metadata are then reused for real-backend scoring. A structural pass or stub
score is not model evidence: this fixture does not establish contact, causality,
photoreal performance, physical correctness, policy success, or robot safety.

The packaged sample's `progress-without-terminal-fail` case exercises an
omitted-outcome negative, but its tiny synthetic frames and prerecorded score
remain wiring-only. For a real rollout gate, retain a source-matched truncated
case with plausible progress and no terminal outcome, plus a complete case,
ambiguous terminal evidence, and blank or unrelated evidence under the same
task and rubric. Run the [terminal-evidence live check](../../../docs/workbench/cookbooks/vlm-eval-loop-runbook.md#terminal-evidence-live-check)
and independently review retained rationales against pixels. Report sampling
differences; these controls do not estimate error rates or qualify a model.

Each `npa_vlm_eval_benchmark_report_v2` configuration includes the full 2x2
confusion matrix, false-positive and false-negative rates, and ordered
`false_positive_item_ids` / `false_negative_item_ids`. Resolve those IDs in the
same configuration's complete `results` list before choosing a threshold; an
aggregate accuracy can hide the exact false pass that matters. Historical or
manually constructed metrics can have null rates when a required class is absent;
new sweeps reject such single-class datasets before evaluation. Item IDs must
be unique. Historical reports without `schema_version` are v1; their counts can
be recomputed from retained per-item labels and predictions, but absent v2
fields must not be presented as if the producer emitted them.
Manually constructed reports retain the v1 constructor default even if optional
calibration fields are supplied. Feature-detect a non-null `confusion_matrix`
rather than inferring field absence from the version alone.

## In workflows

toolRefs: `workbench.vlm_eval.run`, `.loop`, `.judge_against_plan`, `.benchmark`.
The reusable audit-only paired primitive is
`workbench.vlm_eval.compare_judges`.
Specs under `workflows/testing/`: `vlm-eval-single.yaml`,
`vlm-eval-loop.yaml`, `vlm-eval-benchmark.yaml`, `vlm-eval-token-factory.yaml`
(the zero-GPU judge), plus the rollout-judge combinations listed in
`skills/tools/token-factory/SKILL.md`.

Self-hosted VLM steps need a GPU image; set it with `--image` on
`vlm-eval workflow` or the `NPA_VLM_IMAGE` environment variable. The `api` and
`stub` backends need neither.

## Gotchas

- **Never move the threshold to make a run pass.** The threshold is the claim. If
  a gate fails, the policy failed; report the measured score.
- **`run` on a multi-rollout prefix silently produces one blended score.** Use
  `loop`. This does not error.
- **A stub score is not evidence.** Neither is `--score`. Both are wiring tests.
- **The judge sees only the frames you send it.** A low score with
  `--frame-selection final --max-frames 1` may be a sampling artifact rather than
  a policy failure; re-score with keyframes before believing it.
- **Inspect retained frame dimensions before judging thin defects.** Image
  normalization can downscale source pixels enough to erase faceting, skeletons,
  unsupported surfaces, or other narrow structures.
- **Benchmark the rubric before trusting it.** Rubric wording moves scores more
  than most people expect, which is precisely what `benchmark` is for. The task
  text and rubric form one prompt; changing either invalidates prior calibration.
- **Do not calibrate only on positive examples.** Such a set cannot measure
  false-positive behavior and is rejected before evaluation.
- **Outcome is not agency.** A moved object does not prove that the visible
  actor grasped or caused its motion. Use paired state and agency controls.
- **Do not send simulator gizmos as task evidence.** Disable coordinate axes
  and debug overlays, or crop them before a VLM audit; judges can inventory
  those markers as physical task objects.
- **Keep sealed numeric gates external.** Qualitative text such as "a visible
  gap" cannot replace synchronized simulator height or another predeclared
  numeric reference.
- **A green gate does not mean a good policy.** It means the judge, at this
  rubric and threshold, on these frames, said yes.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

## Blinded preference audits

`compare-preference` is an API-only, audit-only matched-image primitive. It
normalizes each input exactly once from RGB pixels without embedded metadata,
hides source semantics behind neutral A/B
labels, and sends the pair in both orders with identical prompting and
generation settings. The private `vlm_preference_comparison.json` retains both
exact requests and complete provider outcomes. Errors, unresolved output,
confidence below `high`, or different mapped preferences produce escalation;
the latter is named `order_disagreement_or_nondeterminism` because one request
per order cannot isolate an order effect from provider nondeterminism. The
command never retries or averages preferences, and refuses an existing output.
Task and rubric text containing `baseline` or `candidate` is rejected before
transport. Telling the model not to follow image text is not a defense against
in-image instructions. Its CLI summary omits paths, prompts, visible support,
uncertainty, request IDs, and raw responses.

Use toolRef `workbench.vlm_eval.compare_preference`. See
[the live audit contract](../../../docs/testing/vlm-audit-live-contracts.md)
for the scheduled real hosted lane and private operator configuration.
