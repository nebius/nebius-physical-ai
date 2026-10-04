# VLM Isaac outcome-versus-agency calibration

This evidence record covers a CPU-only structural calibration packaged with
`vlm-eval`. It does not report a hosted or self-hosted model result.

## What is measured

The fixture copies the six bytes-exact diagrams from
[`docs/assets/hackathon/isaac-franka-lift-cube`](../../assets/hackathon/isaac-franka-lift-cube/)
into the installed `npa` package. The
[`frame-manifest.json`](../../../npa/src/npa/workbench/vlm_eval/fixtures/isaac_agency_calibration_v1/frame-manifest.json)
pins source-file, normalized-PNG, and decoded-RGB SHA-256 values.

The red-object and light-gray-actor masks produce:

| Frame | actor right edge | object left edge | horizontal gap | object bottom |
| --- | ---: | ---: | ---: | ---: |
| `frame_00.png` | 150 | 200 | 50 | 195 |
| `frame_01.png` | 134 | 200 | 66 | 195 |
| `frame_02.png` | 118 | 200 | 82 | 190 |
| `frame_03.png` | 114 | 200 | 86 | 175 |
| `frame_04.png` | 110 | 200 | 90 | 155 |
| `frame_05.png` | 106 | 200 | 94 | 135 |

Both masks are present in all six frames. The cube has a 60-pixel signed
first-to-last rise while the minimum arm-to-cube gap is 50 pixels and grows as
the arm retracts. The same selected bytes therefore support these paired
labels:

| Item | Claim | Structural verdict |
| --- | --- | --- |
| `cube-elevated-positive` | The red cube becomes elevated above the ground surface. | pass |
| `robot-grasp-lift-negative` | The robot arm grasps the red cube and lifts it off the ground. | fail |

The benchmark preflights every configured item before backend or evaluator
activity and records its measurements under `sweep.structural_checks`. Real
backend scoring receives the exact same immutable selected-frame objects used
by preflight.

## Reproduction

```bash
npa workbench vlm-eval benchmark \
  --dataset isaac-agency \
  --output /tmp/isaac-agency-benchmark.json \
  --backend stub \
  --frame-selection sequence \
  --max-frames 6 \
  --thresholds 0.5 \
  --format json

npa/.venv/bin/python -m pytest \
  npa/tests/workbench/test_vlm_eval_agency.py \
  npa/tests/workbench/test_vlm_eval_backend.py \
  npa/tests/cli/test_workbench_vlm_eval_cli.py \
  npa/tests/scripts/test_capture_isaac_lab_scene_frames.py -q
```

### Historical execution claims (not current-head gates)

The following figures were recorded in the original contribution without
execution-head, environment or raw-receipt bindings. They remain historical
claims, not independently bound evidence for a later integration candidate.
Current validation uses the separately bound
[original full/local gate record](../../testing/evidence/cursor-agency-calibration-20261003/local-gates-31b64fd6.json)
and [hosted observations](https://github.com/nebius/nebius-physical-ai/blob/6efa5e3b08b0653082173f4a547d98f1de56e624/docs/testing/evidence/cursor-agency-calibration-20261003/readme.md).
Those records retain their own execution SHAs; neither is a later-head full or
inference claim.

The historical focused source run was reported as 94 passed, one expected live-GPU skip,
and zero failures. The inherited completion-boundary suite separately completed
with 62 passed and zero failures. No provider, credential, GPU, or network call
was made.

An offline wheel build succeeded, and archive inspection found the benchmark,
frame manifest, and all six PNGs (eight fixture files total) in the installed
package.

An isolated source-overlay mutation run killed 25 of 25 mutants. The mutations
removed the alias, changed required hashes and labels, reversed or unsigned the
rise calculation, weakened mask completeness to 80%, promoted inconclusive
proximity, bypassed dataset-wide preflight and selected-frame reuse, accepted an
unknown kind or claim, accepted duplicate labels or malformed hashes, added a
legacy null field, corrupted both new metric formulas and ranking, removed class
and ID guards, applied string truthiness to `fail`, discarded a metadata-derived
task, accepted booleans or an unsupported frame selection, exported a
non-callable SDK constant, and bypassed the inherited `finish_reason="stop"`
guard.

## Hardware and applicability

| Surface | Hardware used | What the result supports |
| --- | --- | --- |
| Fixture hashing and pixel masks | CPU | Exact six packaged 320×240 stylized frames |
| Structural preflight and benchmark plumbing | CPU | Paired state-versus-agency labels, zero-call ordering, metrics, and frame reuse |
| Hosted or self-hosted model qualification | Not run | No model, rubric, or threshold is qualified by this record |
| Robot or simulator behavior | Not run | No policy, contact event, or physical trajectory is qualified |

These frames are synthetic stylized stand-ins, not Isaac Sim renders. Color
masks are valid only for this explicitly configured fixture. Horizontal
mask boxes fuse every matching pixel; they do not separate connected components.
Additional same-color objects or overlays below the excluded title rows can
change the box and invalidate its interpretation for other images. Horizontal
separation can reject the frozen grasp claim. `actor_causes_motion` is therefore
a one-sided refuter: `fail` disproves the configured claim; `inconclusive`
blocks benchmark preflight and must not be treated as success. Proximity is
inconclusive because it cannot establish grasp, force, dynamics, or causality.
The claim has no `pass` outcome, so benchmark parsing rejects
`actor_causes_motion` items whose `expected_label` is positive. The result does
not establish photoreal generalization, policy success, physical correctness,
or robot safety.
