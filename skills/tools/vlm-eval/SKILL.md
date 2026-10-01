---
name: vlm-eval
description: Use to score rollouts with a vision-language model and turn the score into a pipeline gate — single rollout, prefix-wide loop, rubric/threshold benchmark sweeps, backend selection (self-hosted, api, stub), and judging against a plan an earlier stage wrote.
---

# VLM eval (scoring rollouts and gating pipelines)

`vlm-eval` answers "did this rollout complete the task?" as a number, then turns
that number into a gate. It is the judging half of the loop whose generating half
is Cosmos/Genesis/Isaac rollouts and whose reasoning half is
`skills/tools/token-factory/SKILL.md`.

## Pick the right command

```bash
npa workbench vlm-eval run   --input-path <one-rollout>  --output-path <eval.json>
npa workbench vlm-eval loop  --input-path <prefix>       --output-path <prefix>
npa workbench vlm-eval benchmark --dataset <manifest> --output <report.json>
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
  Token Factory for a zero-GPU judge.
- `stub` — deterministic, no model call. For wiring tests and CI only; a stub
  score is never evidence about a policy.

`--endpoint-url` accepts either a base URL or a full `/chat/completions` URL.
Default model is `Qwen/Qwen2-VL-7B-Instruct`; `--timeout-s` defaults to 120.

Both real backends require the exact response value
`choices[0].finish_reason="stop"` before parsing a score. Truncated, filtered,
tool-call, aborted, malformed, empty, or missing completion metadata is an error
even when the response already contains valid-looking verdict JSON. A custom
self-hosted endpoint that omits `finish_reason` must be adapted to emit the
standard OpenAI-compatible field; NPA does not infer completion from parseable
JSON.

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
  values for partial progress, and penalizes unsafe or ambiguous outcomes.
- `--success-threshold` (default 0.8) is the gate. In `loop` it applies to the
  **mean** score across rollouts, which is a coarser claim than per-rollout
  success — do not report it as a per-rollout success rate.
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

## In workflows

toolRefs: `workbench.vlm_eval.run`, `.loop`, `.judge_against_plan`, `.benchmark`.
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
- **Benchmark the rubric before trusting it.** Rubric wording moves scores more
  than most people expect, which is precisely what `benchmark` is for.
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
