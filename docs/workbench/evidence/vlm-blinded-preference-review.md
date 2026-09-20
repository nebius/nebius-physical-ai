# Blinded VLM preference review

This report records an audit-only `vlm-eval compare-preference` run. It does
not qualify a model, establish geometry accuracy or physical correctness,
estimate an operational preference rate, or certify robot safety.

## Hardware and workload applicability

| Item | Evidence |
| --- | --- |
| Source commit | `6c3c7823faeed9e0b348e78753603965edb961d8` |
| Container image | Not applicable; the installed NPA client made hosted API requests |
| CPU | Real image normalization, hashing, request construction, strict parsing, private journaling, and report validation |
| Local GPU | Not applicable; no model was served locally |
| Hosted API | Six real image requests to `MiniMaxAI/MiniMax-M3`; provider hardware identity was unavailable |
| Full workload | Three frozen matched pairs, each submitted once in both A/B orders; six calls completed and zero calls were repeated |
| Objective controls | Neutral A/B transport, exact reversed image order, identical non-image request fields, strict returned-model and completion checks, private no-clobber evidence |
| Focused exact-head tests | 388 passed and 2 skipped |
| Repository guardrails | 4,019 passed |
| Onboarding smoke | 114 passed |
| Independent review | Source and harness approval preceded all calls; a separate reviewer reconciled controls, requests, boundaries, responses, parsed outcomes, reports, hashes, and visible pixels |

## Contract

The command normalizes each input once, hides its source role behind neutral
`A` and `B` labels, and sends exactly two requests with reversed image order.
It never retries or averages preferences. A provider or contract error,
unresolved output, confidence below `high`, or different mapped preferences
requires escalation. The private report remains
`deployment_status=audit_only` even when the two mapped preferences agree.

## Hosted outcome

All six attempts returned HTTP 200 with `finish_reason=stop`, the exact
requested model, usage, and a unique provider request identity. No provider or
parser error occurred. Provider request identities and raw text remain private.

| Order | Pair | Submitted order | Mapped preference | Confidence | Prompt / completion / total tokens | Label attribution |
| --- | --- | --- | --- | --- | ---: | --- |
| 01 | pair-001 | baseline as A | candidate | medium | 1563 / 279 / 1842 | ineligible |
| 02 | pair-002 | candidate as A | candidate | high | 1563 / 164 / 1727 | match |
| 03 | pair-003 | baseline as A | candidate | medium | 1563 / 197 / 1760 | ineligible |
| 04 | pair-001 | candidate as A | candidate | high | 1563 / 192 / 1755 | match |
| 05 | pair-002 | baseline as A | candidate | high | 1563 / 177 / 1740 | match |
| 06 | pair-003 | candidate as A | candidate | high | 1563 / 189 / 1752 | match |

The `baseline` and `candidate` names above describe private post-run
attribution, not text sent to the judge. Four high-confidence outcomes
were eligible for comparison with the independently frozen labels, and all four
matched. Medium-confidence outcomes were excluded rather than counted as
matches.

| Pair | Mapped preferences | Confidence pattern | Report status | Escalation |
| --- | --- | --- | --- | --- |
| pair-001 | candidate / candidate | medium / high | `low_confidence` | required |
| pair-002 | candidate / candidate | high / high | `consistent_candidate_preference` | no |
| pair-003 | candidate / candidate | medium / high | `low_confidence` | required |

There was no mapped-preference disagreement. The confidence change across
reversed orders for pair-001 and pair-003 is still an
order-or-nondeterminism signal: one call per order cannot distinguish those
causes. The mechanism therefore escalated two of the three pairs.

## Independent pixel review

The independent evidence reviewer found the core comparisons grounded in the
submitted pixels: broad gray surfaces were contrasted with exposed point and
detail structure. Some auxiliary rationale was not reliable. Two responses
misread a visible `1.0 m` scale as `10 m`, and several semantic, clipping, or
measurement-causality details exceeded what the pixels supported. Raw
rationales therefore remain private and are not presented as factual evidence.

No in-image instruction was present or followed in these controls. That
observation is not a defense against in-image instructions; input-integrity risk
remains unresolved.

## Public evidence bindings

- source commit:
  `6c3c7823faeed9e0b348e78753603965edb961d8`
- hosted harness:
  `49c7b16b07df19f2617c0040509fb97b5e022b365fb034493b2e7241c34c2d63`
- sanitized hosted summary:
  `fb62feecd8b745442b10c3968cb12b3c2a178e1340e3291b13dc13bf2fee1b3a`
- pair-001 report:
  `f0131cc690e599dd7c991cb22cadf101dfaf07cb303f50d95d59b9518eae41a4`
- pair-002 report:
  `7e8d80da6453fea0b0e4be426f0a93d72b13bb869ae8aa433e4d9f9d18c2646c`
- pair-003 report:
  `95cfe250ad342a7d6890e3da6d3556626424f7b2ab5bf538e642ff808a0047b9`

Normalized control pixels, source-role mappings, raw responses, request
identities, private paths, endpoint details, and infrastructure identifiers are
retained outside the public repository. The independent retained-evidence
review approved the counts and hashes above without modifying evidence.

## Limitations

- Three views of one scene do not estimate an operational preference rate.
- One request per order cannot separate order effects from provider
  nondeterminism.
- Four eligible label matches do not qualify the judge or establish correctness.
- Visible preference does not establish geometry accuracy, task completion,
  physical validity, or safety.
- A vision judgment cannot certify robot safety.
