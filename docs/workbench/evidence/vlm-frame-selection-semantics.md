# VLM frame-selection semantics

This evidence note separates the deterministic sampler contract from hosted
visual-model observations. The source change makes `keyframes` and `sequence`
distinct only when the input frame count is known.

## Deterministic contract

| Strategy | Known-count behavior | Unknown-count video compatibility |
| --- | --- | --- |
| `final` | Last source frame only | One trailing frame via the existing `-sseof` path |
| `sequence` | Uniform full-span indices | Bounded one-frame-per-second sample from the beginning |
| `keyframes` | When the source exceeds the limit, half of the budget is in a terminal window covering at least the final 10%; the window widens when needed for unique tail frames, and the rest spans earlier evidence. Shorter sources return every frame | Same fallback as `sequence` |

Known-count `keyframes` includes the first and last source frames whenever at
least two frames are selected. It is terminal-biased temporal stratification,
not content-aware event detection. Unknown-count fallback frames retain null
source indices, counts, and timestamps with incomplete sampling coverage.

The exact legacy `sequence` formula is checked over frame counts 0 through
1,000 crossed with frame limits 1 through 128. Candidate `keyframes` checks use
the same 128,128-pair domain for cardinality, uniqueness, order, bounds, short
inputs, and endpoint retention. Public image-sequence, NumPy, and known-count
video paths are checked separately.

## Visual evidence status

No hosted call has been made for this change yet. The frozen operational
regression contains one complete real SO-100 trajectory, a bit-exact
221-frame prefix ending before visible placement, and a neutral-gray control.
The exact model, task, rubric, threshold, request order, and six one-shot calls
are frozen privately. Transport remains prohibited until an independent
reviewer approves the exact candidate source, production-selected bytes,
requests, and crash-safe evidence harness.

Any later six-call result is a narrow operational regression for those exact
controls. It cannot qualify a model, estimate error rates, establish
repeatability, prove that pixels alone caused a difference between strategies,
generalize to unknown-count videos, or establish task performance, physical
correctness, policy quality, or robot safety.
