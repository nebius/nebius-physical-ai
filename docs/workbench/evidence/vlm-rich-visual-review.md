# Rich visual review hosted contract smoke

This report records one audit-only hosted check of `vlm-eval review-visual`.
The feature keeps rich visual judgments separate from the normalized
task-completion score and gate. The hosted check is a strict negative result:
none of the six attempts produced a contract-valid visual review.

## Hardware and applicability evidence

| Item | Evidence |
| --- | --- |
| Execution source commit | `e7db276c6a129a96551b845c3d247ba464d1d7fa` |
| Container image | Not applicable; the installed NPA client made hosted API requests |
| Local CPU | Real image decode, RGB and file hashing, request construction, strict parsing, private journaling, and evidence validation |
| Local GPU | Not used |
| Hosted API | Six approved attempts: three each for `MiniMaxAI/MiniMax-M3` and `openbmb/MiniCPM-V-4_5` |
| Provider hardware | Unavailable and unattested by the hosted API |
| Full workload | Three seven-frame controls; 70 submitted image instances across four paired and two single requests |
| Focused execution-head tests | 313 passed and 1 skipped |
| Independent review | Plan, source/harness, exact-call permission, submitted pixels, raw outcomes, and all retained hash chains were reviewed separately |
| Applicability | Known contract smoke only; not calibration, qualification, a performance benchmark, or a safety result |

## Product boundary

`review-visual` writes a separate `vlm_visual_review.json`. A single review
uses one neutral image set. A paired review sends the same two sets once in
each A/B order. The strict schema separates:

- visible task evidence;
- visible artifact fidelity;
- reviewability and presentation limits;
- subjective impressiveness;
- a bounded, unverified Physical AI usefulness hypothesis.

The report cannot supply a normalized score or gate decision. Objective
references remain private, code-owned, and unverified by the vision judge.
Provider errors, malformed structures, missing frame citations, cross-order
disagreement, and unsupported usefulness fields fail closed.

## Frozen visual controls

The neutral task was to pick up a cube and place it in a box.

- `C01` contains seven real frames and ends with the cube visibly inside the
  box. Pixels do not prove release or stability.
- `C02` contains seven source-matched real frames and stops during held or
  occluded transfer before placement. Its last frame matches a nonterminal
  `C01` frame.
- `C03` contains seven byte-identical uniform RGB `(128, 128, 128)` frames.

Independent review decoded all 21 source frames and all 70 submitted image
instances at 640×480 RGB. Every submitted RGB hash matched its frozen source.
Private roles, labels, paths, objective values, and credentials were absent
from provider-facing requests.

## Hosted outcome

Exactly six approved attempts started once, sequentially, with no retries,
replays, or substitutions. Six responses and outcomes were retained. Four
attempts returned HTTP 200 with `finish_reason=stop`; two paired MiniCPM
attempts returned HTTP 400 because that model interface accepts at most ten
images while the frozen paired request contained fourteen.

All four job reports ended `judge_error`.

| Model | Control and neutral order | Transport | Strict result | Independent pixel-first assessment |
| --- | --- | --- | --- | --- |
| MiniMax-M3 | `C01=A`, `C02=B` | HTTP 200, `stop` | Contract error: malformed trailing JSON, wrong field types, missing citations, invalid comparison enum | Preferred `C01`, but called its visible terminal placement only partial; medium confidence and one unsupported `C02` assertion |
| MiniMax-M3 | `C02=A`, `C01=B` | HTTP 200, `stop` | Contract error: wrong field types, missing citations, invalid comparison enum | Correctly called `C02` partial and `C01` complete and preferred `C01`; overclaimed downstream suitability and mixed reviewability limits into fidelity |
| MiniMax-M3 | `C03` single | HTTP 200, `stop` | Contract error: uncited strings and wrong uncertainty type | Correctly found no task evidence and no useful scene, but returned inconsistent impressiveness fields |
| MiniCPM-V-4_5 | `C01=A`, `C02=B` | HTTP 400 | Provider image-count error | No model judgment; complete-versus-truncated discrimination was not tested |
| MiniCPM-V-4_5 | `C02=A`, `C01=B` | HTTP 400 | Provider image-count error | No model judgment; order consistency was not assessable |
| MiniCPM-V-4_5 | `C03` single | HTTP 200, `stop` | Contract error: scalar fields, invalid booleans and nulls, missing citations | Correctly found no cube or box, but contradicted its own unreviewable status and returned invalid fidelity, impressiveness, and usefulness fields |

No HTTP-200 output was refused, Markdown-fenced, or truncated. Semantically
plausible prose was not repaired with a permissive post-hoc parser: a strict
contract failure remained a product failure.

## What the failure exposed

1. Both hosted models need stronger structured-output conformance before they
   can produce accepted rich review records.
2. MiniMax showed an order effect: `C01` changed from partial to complete and
   its fidelity classification changed across neutral reversal. It preferred
   `C01` both times, but neither comparison used the required enum.
3. The seven-frames-per-arm paired shape exceeds MiniCPM's ten-image provider
   limit. Model-specific request compatibility must be checked before treating
   paired behavior as exercised.
4. Both models recognized the uniform-gray absence control in substance, but
   neither returned a complete contract-valid record.

These are failure cases, not evidence for accepting either judge.

## Public evidence bindings

- execution source:
  `e7db276c6a129a96551b845c3d247ba464d1d7fa`;
- request freeze:
  `19b2899129f7ccc83036fae853e66557f0e893b391a4bc656355ddcf716dae23`;
- source approval:
  `a0bd44761d393672aa8b7d3a3e15836f6666d651cdac01b03563b8c41e0a0c92`;
- exact-call approval:
  `3826f73b5db11d976805fac133ca944cff4565dc08ac3aff7a8fc917a1f5c1cf`;
- immutable hosted summary:
  `2d8c2a00b1b4552d7bf233583faf0cf77ebad6160cc2fb5c6a002da50525d153`;
- independent retained-evidence review:
  `19f01742421c7160dcd856e60a74c4904be393b23b9486fdac79276014727ad4`.

Raw responses, provider request identities, source pixels, private mappings,
paths, endpoint details, and objective references remain outside the public
repository. The independent reviewer recomputed all six
wire→request→transport→response→outcome→report chains and their summary
bindings without mismatch.

## Bounded conclusion

The separate rich-record mechanism failed closed and left the normalized score
and gate unchanged. The hosted evidence does not qualify either model, prove
task completion, establish usefulness, estimate an error rate, or make a
deployment recommendation.

Selected images cannot establish continuous execution, hidden simulator state,
release mechanics, stability, policy success, physical correctness, or safety.
A vision-language review cannot certify robot safety.
