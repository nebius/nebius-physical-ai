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

Small budgets deliberately trade middle-of-episode coverage for terminal
evidence: three frames from a 100-frame source select `[0, 90, 99]`; four from
a 1,000-frame source select `[0, 450, 900, 999]`. The three-frame case has only
one early sample, so events between the initial and terminal samples can be
missed. This changes the default `keyframes` behavior, not `sequence`, and
does not guarantee continuous coverage, event detection, or better model
judgments. Exact selected indices, source counts, timestamps when known, and
normalized frame hashes are already bound into the request-manifest hash;
no additional sampling-schema field is needed to distinguish these requests.

The exact legacy `sequence` formula is checked over frame counts 0 through
1,000 crossed with frame limits 1 through 128. Candidate `keyframes` checks use
the same 128,128-pair domain for cardinality, uniqueness, order, bounds, short
inputs, and endpoint retention. Public image-sequence, NumPy, and known-count
video paths are checked separately.

## Visual evidence status

The frozen operational regression used one complete real SO-100 trajectory, a
bit-exact 221-frame prefix ending before visible placement, and a uniform
neutral-gray control. An independent pre-call review approved the exact
candidate source, selected PNG bytes, request order, model, task, rubric,
threshold, and crash-safe evidence harness.

Exactly six one-shot Token Factory calls then ran in the frozen order: one call
per strategy and control. All six returned HTTP 200 from the exact requested
and served `openbmb/MiniCPM-V-4_5` model with `finish_reason=stop`, unique
provider request IDs, and zero retries. The observed labels and scores were:

| Disclosed control | Expected | `sequence` | `keyframes` |
| --- | --- | --- | --- |
| C01: complete SO-100 placement | `true` | `false`, 0.0 | `false`, 0.0 |
| C02: prefix before visible placement | `false` | `false`, 0.0 | `false`, 0.0 |
| C03: uniform neutral gray | `false` | `false`, 0.0 | `false`, 0.0 |

Both strategies were correct on 2 of 3 disclosed calibration controls: 0 true
positives, 2 true negatives, 0 false positives, and 1 false negative. This is
calibration-only accounting, not an accuracy estimate.

Independent pixel-first review found the C01 expected label visibly supported
by the exact submitted frames. The `sequence` request included source indices
302 and 453 with the red cube inside the white box; `keyframes` included
indices 408 and 453 with the same terminal state. Both model rationales denied
that visible placement. The C02 labels were correct, but both rationales
overstated the cube's location through submitted frames where it was occluded
or not visibly locatable. The `sequence` C03 rationale stayed grounded in the
absence of evidence. The `keyframes` C03 rationale instead claimed that its
final uniform-gray frame showed a cube on a table, hallucinating objects absent
from the submitted pixels.

The precommitted gate required `keyframes` to label C01 true, C02 and C03 false,
avoid regressions where `sequence` was correct, and keep its rationales
grounded. It therefore **failed** on C01 and rationale grounding. The result
does not show a hosted label improvement over `sequence`, and this change is
not merge-ready under that evaluation plan. The deterministic sampler contract
above remains separately reproducible; a draft may carry this negative hosted
result for review without presenting it as visual acceptance.

## Evidence identity and applicability

| Item | Identity or applicability |
| --- | --- |
| Candidate source | commit `f77226cfb18126478769c97e2cb3a5063011299e`; tree `2b480d4107483db7e0cc5aede3be4cd57817b3f9` |
| Evaluation plan SHA-256 | `fcbbcacc71583186ebbb3c4c9055169566a7a37f0b97b30e84f0792ed958d70e` |
| Independent source/call approval SHA-256 | `101b875924924a7d9cf8c605e9fc40b2114cf8f41b96494ab0238a9c9d6b7985` |
| Frozen request manifest SHA-256 | `9a75fdb683bf9d94da8aa32187d8fc845f1bdc0a8ea4dcbba90de0b5d6ad5061` |
| Hosted result manifest SHA-256 | `2cd91b69a9dc7a8ec7b601e6e891dddc675f02d2e16a29dfe63a064f7d1be768` |
| Independent review-input manifest SHA-256 | `334cef017176fb170ea1d0273e126933370ae8a2ed89e84d3877a7c3b7ec8aa9` |
| Independent hosted review SHA-256 | `77f54079cd97d78724a315e0c4a80e0bcba8b3850eae34eeaee2b8b987a14372` |
| Deterministic validation | CPU-only source, property, public-path, and mutation checks; no GPU required |
| Visual inference | Provider-managed hosted execution; no local GPU was used and provider accelerator details were unavailable |
| Physical system | Previously recorded SO-100 camera trajectory; this evaluation did not actuate a robot |

Exact requests, responses, selected PNGs, contact sheets, provider metadata,
and recomputed hashes remain in private evidence. No credentials, endpoints,
object URIs, provider request IDs, or raw responses are published here. The
unchanged six calls will not be replayed for a preferred result.

This regression cannot qualify a model, estimate operational error rates,
establish repeatability, prove that pixels alone caused a difference between
strategies, generalize to unknown-count videos, or establish task performance,
physical correctness, policy quality, or robot safety.
