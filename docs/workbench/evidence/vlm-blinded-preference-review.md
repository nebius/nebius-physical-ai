# Blinded VLM preference review

This report records the final audit-only `vlm-eval compare-preference`
workload at source `2ffc616fdebfdfadc24fb6f94b78e8c431e0a66c`. Preference
requests impose no output-token cap. This run does not qualify a model,
establish geometry accuracy or physical correctness, estimate an operational
preference rate, or certify robot safety.

## Hardware and workload applicability

| Item | Evidence |
| --- | --- |
| Executed source | `2ffc616fdebfdfadc24fb6f94b78e8c431e0a66c` |
| Image digest | Not applicable; no container image changed |
| Container image | Not applicable; the installed NPA client made hosted API requests |
| CPU | Real image normalization, hashing, request construction, strict parsing, private journaling, and report validation |
| Local GPU | Not applicable; no model was served locally |
| Hosted API | Six real image requests to `MiniMaxAI/MiniMax-M3`; provider hardware identity was unavailable |
| Full workload | Three frozen matched pairs, each submitted once in both A/B orders; six uncapped calls completed and zero calls were retried or replayed |
| Objective controls | Exact four-field requests with no `max_tokens` or `response_format`, neutral A/B transport, reversed image bytes, strict returned-model/completion checks, and private no-clobber evidence |
| Focused source validation | 207 tests passed; restoring `max_tokens: 1000` failed the regression |
| Negative controls | Provider, parser, retention, low-confidence, and order-disagreement paths remain fail-closed; two real pairs escalated for low confidence |
| Independent review | Plan, exact source, harness, pixels, requests, responses, reports, and all hash chains were reviewed separately |
| Remaining merge condition | This PR is stacked on #624 and must follow the ordered rebase/retarget/revalidation sequence; required current-head CI is shown on the PR |

## Contract

The command normalizes each input once, hides its source role behind neutral
`A` and `B` labels, and sends exactly two requests with reversed image order.
The exact request fields are `model`, `temperature`, `messages`, and
`chat_template_kwargs`; there is no product-imposed output-token limit. It
never retries or averages preferences. A provider or contract error, unresolved
output, confidence below `high`, or different mapped preferences requires
escalation. The private report remains
`deployment_status=audit_only` even when the two mapped preferences agree.

## Hosted outcome

All six attempts returned HTTP 200 with `finish_reason=stop`, the exact
requested model, usage, and a unique provider request identity. No provider or
parser error occurred. Provider request identities and raw text remain private.

| Order | Pair | Submitted order | Mapped preference | Confidence | Prompt / completion / total tokens | Label attribution |
| --- | --- | --- | --- | --- | ---: | --- |
| 01 | pair-001 | baseline as A | candidate | medium | 1563 / 279 / 1842 | ineligible |
| 02 | pair-002 | candidate as A | candidate | high | 1563 / 209 / 1772 | match |
| 03 | pair-003 | baseline as A | candidate | medium | 1563 / 182 / 1745 | ineligible |
| 04 | pair-001 | candidate as A | candidate | high | 1563 / 162 / 1725 | match |
| 05 | pair-002 | baseline as A | candidate | high | 1563 / 227 / 1790 | match |
| 06 | pair-003 | candidate as A | candidate | high | 1563 / 174 / 1737 | match |

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
detail structure. Some auxiliary rationale was not reliable. Three responses
misread a visible `1.0 m` scale as `10 m`, and several semantic, clipping, or
measurement-causality details exceeded what the pixels supported. Raw
rationales therefore remain private and are not presented as factual evidence.

No in-image instruction was present or followed in these controls. That
observation is not a defense against in-image instructions; input-integrity risk
remains unresolved.

## Public evidence bindings

- source commit:
  `2ffc616fdebfdfadc24fb6f94b78e8c431e0a66c`
- hosted harness:
  `be37cd8f19558f13854d2507fedf7c904f155551d8d13c9fde90b39a591d7e54`
- pre-call approval:
  `e50000df2e7e377daae059c98f71e350f458ef60f590a30aeff3da2a6d74a109`
- sanitized hosted summary:
  `f95f61621f5b4425ee4f73e29c9c25f177aa85f59c7fcfd945f9549d239efd97`
- pair-001 report:
  `841d7b8495eaf450021a3e2620b2b3557317a9c66c5eddfb5f7ef1f648be5290`
- pair-002 report:
  `6311745f4dc781911fe9bb4f079e04727d992e00bbd3ad56d69b90cfad48cd36`
- pair-003 report:
  `ec8e751938b0e8e1a55d5f0b3fb27d32e24e8d383f45c5adcbb591886813800d`
- independent retained-evidence review:
  `6dc289527497d2bff4599b6a19d0e9b5a534796b12433570b013eb9223e69051`

Normalized control pixels, source-role mappings, raw responses, request
identities, private paths, endpoint details, and infrastructure identifiers are
retained outside the public repository. The independent retained-evidence
review approved the counts and hashes above without modifying evidence.

## Limitations

- Three views of one scene do not estimate an operational preference rate.
- One request per order cannot separate order effects from provider
  nondeterminism.
- The observed completions used 162 to 279 tokens, so this run proves the
  submitted requests are uncapped but does not exercise a response longer than
  the removed 1,000-token limit.
- Four eligible label matches do not qualify the judge or establish correctness.
- Visible preference does not establish geometry accuracy, task completion,
  physical validity, or safety.
- A vision judgment cannot certify robot safety.
