# Token Factory nested no-image normalization evidence

This addendum covers source commit
`3fdced71f6d42f3e1fa5e9375c305ae363101348` (tree
`64a26f0d62ecb8e94511be44299efc41417d3091`). It follows the
[one-layer normalization record](../token-factory-image-unavailable-normalization/)
without rewriting that historical evidence.

## Problem and result

The prior classifier removed at most one presentation wrapper. A model answer
that was itself a complete no-image admission could therefore be stored as a
successful caption when formatting used two layers:

| Exact visible answer | Prior result | Candidate result |
| --- | --- | --- |
| `**"NO IMAGE RECEIVED."**` | `completed` | `image_unavailable` |
| `"**NO IMAGE RECEIVED.**"` | `completed` | `image_unavailable` |
| `***NO IMAGE RECEIVED.***` | `completed` | `image_unavailable` |

The candidate repeatedly removes only exact matching outer pairs from a closed
set: Markdown asterisk or underscore emphasis, ASCII quotes, and smart quotes.
Every removal shortens the answer, so processing terminates after at most half
the input length. Matching remains case-insensitive and whole-answer-only. The
stored caption remains the exact visible answer after the pre-existing outer
whitespace strip.

## Deterministic objective controls

All controls are CPU-only and use local synthetic completions:

| Control class | Count | Result |
| --- | ---: | --- |
| Unwrapped and single-wrapper punctuation, whitespace, and case controls | 36 | `image_unavailable` |
| Every ordered pair of 8 wrapper forms over punctuated and periodless cores | 128 | `image_unavailable` |
| Exact trigger and triple-emphasis controls | 4 | `image_unavailable` |
| Generated nested compositions, depths 1 through 16 | 16 | `image_unavailable` |
| Longer, mismatched, code-wrapped, other-punctuation, and paraphrase answers | 18 | `completed` |
| Empty and wrapper-only classifier controls | 5 | non-match |

All 202 end-to-end rows made exactly one synthetic completion call, produced
the expected item and aggregate status, and retained the exact stripped visible
answer. A four-layer mixed quote/emphasis answer passed, while a corresponding
wrapper-only answer remained a non-match.

The unchanged failure path was also exercised through a real
`TokenFactoryClient` over `httpx.MockTransport`: one local HTTP 200 response
with empty visible content raised `TokenFactoryToolError`, emitted no caption
item, and wrote no artifact. A captured client request contained exactly one
image data URL whose strict base64 payload decoded as a nonempty 32x24 RGB PNG.
A three-image continuation control still attempted the image after an
unavailable item and returned `completed`, `image_unavailable`, `completed`.

## Mutation and validation

Seventeen meaningful mutants were killed, including one-layer-only processing,
a fixed two-layer cap, complete emphasis-family removal, individual
single-marker removal, every quote family, substring matching, mismatched quote
matching, unbounded period removal, changed punctuation order, removed
inner-boundary stripping, and retained-caption rewriting.

Removing only the explicit `**` tuple or only the explicit `__` tuple is
behaviorally equivalent once recursive `*` or `_` removal remains. Those two
mutants each passed all 222 focused tests and are explicitly excluded from the
killed-mutant denominator rather than being reported as kills.

Validation at the exact source commit:

- 222 focused caption-tool tests passed after mutation restoration.
- 315 Token Factory workbench, batch, client, CLI, and SDK tests passed.
- 163 three-tier and skill-contract tests passed.
- 4,987 guardrails, 118 smoke tests, and 266 managed-Git prechecks passed.
- Ruff lint and formatting, generated-doc drift, confidentiality, and Gitleaks
  checks passed.
- Independent review recomputed the baseline and candidate matrices and reran
  all 19 mutations from an isolated source archive: 17 killed, 2 equivalent,
  zero findings.

## Evidence and hardware applicability

| Stage | Customer hardware | Hosted hardware | Applicability and result |
| --- | --- | --- | --- |
| Classifier, request capture, mutations, tests, and review | CPU; no GPU or cluster | Not applicable | Exact source and deterministic controls passed |
| Smoke tests | CPU | Not applicable | 118 passed; reported separately from the objective matrix |
| Inherited vision completions | No customer cluster or GPU | Provider-managed; accelerator not exposed | No new hosted execution; prior run observed only the exact punctuated sentinel and one positive pixel-grounded caption |
| Visual or GPU evaluation | Not applicable | Not applicable | No pixels, frame selection, rendering, visual scoring, or GPU behavior changed |

The nested, italic, and triple-emphasis answers were **not** observed in the
inherited hosted run. They are deterministic local controls. This follow-up
made zero provider, health, or model-list calls and accessed no credentials.

## Limits

Whole-answer matching limits false positives but does not eliminate them. A
legitimate image whose complete salient text normalizes to the sentinel can
false-fail, and operator or in-image text can induce or suppress the cooperative
answer. This change does not semantically classify arbitrary refusals, prove
that provider pixels were consumed, estimate a provider failure rate, qualify a
visual judge, or establish physical correctness, usefulness, or robot safety.

The machine-readable summary is in [evidence.json](evidence.json).
