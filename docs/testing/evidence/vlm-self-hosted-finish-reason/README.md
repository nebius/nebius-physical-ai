# Self-hosted VLM completion-state evidence

VLM-eval uses model scores as rollout gates. Before this change, a self-hosted
response with valid verdict JSON could score `0.93` even when
`finish_reason="length"` proved that generation was truncated. The hosted
backend rejected the same response.

Candidate `b6fcc370049ab73c7c6536a3b58554d6be4d4592` requires the exact,
case-sensitive and whitespace-sensitive value `finish_reason="stop"` before
either real backend parses a verdict. Missing metadata is intentionally rejected.

## Objective result

The frozen control uses the same valid synthetic verdict for both backends and
varies only completion metadata.

| Exact-source matrix | Hosted scores | Hosted errors | Self-hosted scores | Self-hosted errors |
| --- | ---: | ---: | ---: | ---: |
| Base `59514b990` | 1 | 11 | 12 | 0 |
| Candidate `b6fcc3700` | 1 | 11 | 1 | 11 |

The 12 cases per backend are `stop`, `length`, `content_filter`, `tool_calls`,
`abort`, null, empty, missing, `STOP`, leading-space `stop`, trailing-space
`stop`, and integer zero. The candidate accepts only exact `stop`. All 22
candidate negative rows raise `VlmEvalError` naming `finish_reason=stop` before
either verdict parser runs.

A separate ordering control combines `finish_reason="length"` with malformed
JSON. The base self-hosted path reaches the parser; the candidate rejects both
backends at completion metadata first. A completed Markdown-fenced legacy
control still returns `success=true`, `score=1.0`, rationale `legacy`, and no
invented served-model identity.

The machine-readable rows, exact source identities, hashes, validation counts,
hardware applicability, and limitations are in [evidence.json](evidence.json).

## Sensitivity and validation

- 6/6 non-equivalent mutations were killed: hosted-only guarding, allowing
  missing metadata, allowing `length`, normalizing near-miss `stop` values,
  parsing before checking completion, and rejecting valid `stop`.
- Exact archived source: 80 focused tests passed; one opt-in live-GPU test was
  inapplicable and skipped.
- Related local VLM surfaces: 109 passed, 2 skipped.
- Independent skill and documentation checks: 845 passed.
- Ruff, formatting, diff confidentiality, and Gitleaks passed.
- Independent exact-source review returned `APPROVED_SOURCE` with no findings.

Every matrix request reached a call-site transport substitute under outbound
socket denial. Observed provider, health, model-list, and outbound socket calls
were all zero. These are deterministic parser controls, not fixture scores
presented as a model evaluation.

## Hardware applicability

| Stage | Customer hardware | Hosted hardware | Applicability |
| --- | --- | --- | --- |
| Response matrix, parser ordering, mutations, tests, and review | CPU | Not applicable | Reproducible without a GPU, cluster, or provider |
| Visual evaluation | Not applicable | Not applicable | The change emits no visual artifact and makes no visual claim |

## Reproduce

From exact candidate `b6fcc370049ab73c7c6536a3b58554d6be4d4592`:

```bash
PYTHONPATH="$PWD/npa/src" npa/.venv/bin/python -m pytest \
  npa/tests/workbench/test_vlm_eval_token_factory.py \
  npa/tests/workbench/test_vlm_eval_backend.py -q

npa/.venv/bin/python -m pytest \
  npa/tests/guardrails/test_skills_index.py \
  npa/tests/guardrails/test_develop_skills.py -q
```

The first file contains the cross-backend completion matrix and parser-ordering
spies. The second retains the retry and served-model controls with an explicit
completed-response fixture.

## Integration order

This focused current-main correction should integrate before the overlapping
VLM provenance and Kimi branches are refreshed. PRs 596 and 678, plus dependent
PRs 597, 612, 617, 622, 624, 627, 634, and 647, must preserve the exact-stop
guard and both migration documents on their resolved combined trees. Each
refreshed head needs this matrix, applicable skill/documentation checks, and
fresh exact-head review.

## Limits

- A server-reported `stop` value does not prove that the server is truthful,
  that image pixels were consumed, or that the score is visually or physically
  correct.
- Self-hosted adapters that omit `finish_reason` must be updated before NPA can
  accept their scores.
- The exact base has no explicit `message.refusal` guard. Refusal and shared
  response-helper hardening are outside this change.
- No model, threshold, rubric, retry, JSON-repair, physical-correctness, or
  robot-safety claim changes.
