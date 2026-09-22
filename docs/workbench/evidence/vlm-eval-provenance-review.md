# VLM evaluation provenance: reviewer-safe evidence

This report summarizes the real workload evidence for
[PR 596](https://github.com/nebius/nebius-physical-ai/pull/596) and
[PR 597](https://github.com/nebius/nebius-physical-ai/pull/597). Raw provider
responses, request identifiers, source paths, and operational logs remain in
access-controlled evidence. No image or customer artifact is published here.

## Hardware and acceptance matrix

| PR | Implementation tested | Execution kind | Real workload | Objective acceptance | Independent review | Remaining gate |
| --- | --- | --- | --- | --- | --- | --- |
| 596 | `e22071af12648d8a253d471bd01007b61cf80c73` | Local CPU client; hosted-provider hardware unknown; no local-GPU claim | Nine real image requests across three key-scoped vision models | All nine submitted-frame, prompt, rubric, request-manifest, and raw-response hashes recomputed with zero mismatches | Read-only exact-candidate review is complete | Proof-only head still needs exact-head CI and review |
| 597 | `56c0315b83a264597243392632468c7929dc36bd` | Local CPU client plus hosted vision API; hosted-provider hardware unknown; no local-GPU claim | Three fresh identify-then-judge controls, one consistent standalone blank call, and one replayed historical contradiction | TP 1, TN 2, FP 0, FN 0 on the fresh controls; provider/score contradictions remain visible | Read-only exact-candidate review is complete | Exact-head CI and review of this public proof |

CPU smoke and static checks are not counted as workload proof. They passed
separately: PR 596 had 110 focused tests with 2 skipped, 671 documentation and
skill checks, and 3,862 guardrails; PR 597 had 125 focused tests with 2 skipped,
114 onboarding smoke tests, 781 documentation and skill checks, and 3,862
guardrails.

## PR 596: exact request and response provenance

The real workload used three frozen image-identity controls at one rubric and
threshold against:

- `MiniMaxAI/MiniMax-M3`: 3/3 labels correct;
- `google/gemma-3-27b-it`: 3/3 labels correct;
- `openbmb/MiniCPM-V-4_5`: 2/3 labels correct, with one false positive on an
  unrelated viewer image.

The two private reports have these reviewer-verifiable SHA-256 identities:

- three-case report:
  `51f66030caacf9a5f2e135d69254b089fa45818ffe21554d9742fd70455f6c51`;
- six-case model sweep:
  `1f7d37bc71a0a11a29fcc3fac48cd3c6048e0e1906427d0cae76103b3f9e688c`.

Producer-side recomputation found zero mismatches across 3/3 and 6/6 case
bundles. An independent reviewer separately recomputed the three-case bundle
and reproduced its false positive. Every provider call returned HTTP 200, the
requested model identity, and `finish_reason: stop`. The required GitHub CI run
for the tested implementation passed all five Python 3.12 shards, coverage,
lint, guardrails, security scanning, and the aggregate security gate. These
were read-only program reviews; no GitHub review has been submitted.

These are traceability and image-dependence controls, not threshold
calibration. Every observed score was 0.0 or 1.0, so the controls cannot select
between thresholds inside `(0, 1)`. Submitted frames were normalized from
2400x1200 to 768x384, which is inadequate evidence for thin geometry or
skeleton defects. MiniCPM-V's false positive blocks that exact model, task, and
rubric prompt from acceptance-gate use.

## PR 597: provider and score-gate consistency

The neutral three-control report has SHA-256
`20ec0c5b607c02b93b3af46e9552643c7e851b351cf8f6ca9f47475bdafcb175`.
Independent recomputation verified all three normalized frame, prompt, rubric,
request-manifest, and raw-response identities with zero mismatches.

Score-derived labels were correct for one positive, one blank negative, and one
unrelated-image negative. The provider nevertheless returned `success: true`
with `score: 0.0` on both negatives. PR 597 preserves those contradictions as
`provider_success_matches_score_gate: false`; `passed` still comes only from
`score >= success_threshold`. Omitted or non-boolean self-hosted values remain
null in provider-provenance fields.

One reviewer who recomputed the controls was also involved in the neutral task
design, so that review is not independent evidence that the three-item design
is sufficient. The set is a narrow regression for one leading-prompt failure,
not threshold calibration, geometry-quality evidence, or a production judge.

## Reproduction boundary

A credential holder can exercise the same public command surface without
embedding credentials or infrastructure identifiers. This is an exerciser, not
a bit-for-bit reproducer: exact inputs and raw responses remain private, and
hosted-service outputs are not deterministic.

```bash
export NEBIUS_TOKEN_FACTORY_KEY="<token-factory-key>"
npa workbench vlm-eval run \
  --input-path <one-rollout-or-image-directory> \
  --output-path <private-result.json> \
  --backend api \
  --model MiniMaxAI/MiniMax-M3 \
  --frame-selection keyframes \
  --max-frames 4 \
  --task "<task>" \
  --rubric "<frozen-rubric>" \
  --success-threshold 0.8 \
  --output json
```

Freeze labels, rubric, threshold, frame selection, and failure controls before
inference. Keep raw responses private, recompute every retained hash, and do not
interpret a vision verdict as task performance, geometry accuracy, collision
suitability, physical correctness, or robot safety.
