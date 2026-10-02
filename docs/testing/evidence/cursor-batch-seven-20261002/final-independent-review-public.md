# Independent architecture and live-evidence review

Exact first-nine shipping source: `afa83be5230667c2377f774da6dfb37007e59487`, tree `02dafd818ce8ae8f661f70e68eb4e376746c2c0b`. The later integration including held #647 was tested at `61476e8425039df2058d4488b66adf748f194bb0` and reviewed at `c0757b55ffc00432f49d236746b35a47e6fb8f9a`; those two share tree `bedaf63d527f65f5ee9e88a8ac9afadbe1953c41`.

No remaining code-correctness blocker was found in the reviewed scope. **Recommend the first nine PRs for the merge queue once current-head CI is clear; keep #647 draft.** Its positive paired-rich live acceptance remains unsatisfied. The [testing-conventions skill](https://github.com/nebius/nebius-physical-ai/blob/afa83be5230667c2377f774da6dfb37007e59487/skills/atomic/testing-conventions/SKILL.md) requires: “the `token_factory_e2e` live tests must pass with a real `NEBIUS_TOKEN_FACTORY_KEY`.” The applicable positive paired-rich test has not passed. All original failed and inconclusive outcomes remain retained. This is an automated independent review, not a human GitHub approval.

## Exact evidence scope

The reviewer inspected all 21 submitted control images and froze visible-only findings before opening fresh provider outputs. Earlier code-review context included historical control semantics, so this was not ignorance of all prior context. The reviewer independently checked frozen request bytes, decoded image byte/RGB hashes, source-file hashes, returned model identities, raw response bytes, report bindings, and every retained file-manifest entry: 71 files for experiment 1, 71 for experiment 2, 46 for the profile experiment, and 28 for the exact shipping-source experiment.

There were **29 actual hosted provider attempts**, with no retries or replacement of failed outcomes. All 29 returned HTTP 200, the exact requested model, and `finish_reason=stop`; **26 passed their strict response contract and 3 did not**. HTTP success does not mean a valid or useful review. These attempts reuse three known controls and are not 29 independent calibration samples. Hosted hardware was not reported; no local GPU run is claimed.

| Frozen experiment | Source | Attempts | Strict response contract | Committed audit test cases |
| --- | --- | ---: | --- | --- |
| Initial matrix | `f006cde3b990ece18a13296507ac3334f2b16155` | 10 | 8 valid, 2 invalid | 2 passed, 2 failed |
| Prompt-contract correction | `61476e8425039df2058d4488b66adf748f194bb0` | 10 | 10 valid | 2 passed, 2 failed |
| Additional model profiles | same final source | 5 | 4 valid, 1 invalid | 2 passed, 1 failed |
| Exact first-nine shipping source | `afa83be5230667c2377f774da6dfb37007e59487` | 4 | 4 valid | 2 passed, 0 failed |
| Total | three source snapshots | **29** | **26 valid, 3 invalid** | **8 passed, 5 failed** |

The six ordinary score-control calls are separate from the 13 audit test-case executions. On the three known controls, Kimi scores were 0.95 / 0.60 / 0.00 initially and 1.00 / 0.60 / 0.00 after the prompt update; the fixed threshold stayed 0.80. Offline benchmark aggregation of each fresh response set reported one true positive and two true negatives. That replay is not additional inference or a general accuracy estimate.

## Grounded outcomes and retained failures

- Paired score judges: MiniMax and MiniCPM agreed on the visible positive in all three paired runs, including the actual first-nine shipping source. The request body was identical except for model selection. The shipping-source MiniMax rationale invents a reference to final frames 7–11 although exactly seven images were submitted. The terminal pixels support its numeric positive, but that frame-range claim is unsupported and remains retained. Kimi profiles incompatible with the fixed paired-judge experiment shape are rejected before side effects.
- Blinded preference: Kimi on both the later integration and the actual first-nine shipping source selected the visible object-in-box still in both counterbalanced orders with high confidence. The two images support that preference, while release mechanics, stability and hidden geometry remain unproven. The final-source MiniMax run returned unresolved in both orders and still fails its frozen positive expectation. The initial MiniMax response with the misspelled uncertainty field remains a strict failure.
- Rich paired review: **zero of three positive cases passed the frozen acceptance expectation**. Initial Kimi hit the private-role lexical guard. Final-source Kimi parsed both orders but disagreed on material fidelity/reviewability dimensions; the code correctly escalated. Final-source MiniMax parsed the first order and rejected the reverse order: its content ended with an extra closing brace. The actual parser raised `VlmVisualReviewError`, caused by `JSONDecodeError: Extra data: line 1 column 4598 (char 4597)`. No output was repaired or accepted after stripping the brace.
- MiniMax's strict-valid first rich response also had grounding weaknesses: it called the visible terminal sequence partial, described the final gripper as above the box when it was to the right, and inferred artifact defects from missing/occluded action observations. Valid citations and schema do not establish factual correctness.
- Gray negative control: final-source Kimi and MiniMax faithfully described uniform gray with no task evidence, no impressiveness, and unsupported usefulness. The initial Kimi gray response invented tiny speckles despite every pixel being exactly RGB(128,128,128); that prose defect remains preserved even though the negative-control enums passed.

No physical correctness, safety, measured downstream utility, operational error rate, or reliable rich-review agreement is established by this small matrix.

## Per-PR assessment

Hosted CI and merge-queue readiness must be read from the final current-head coordinator snapshot. This lane reviewed code and retained live evidence; it does not replace that fresh check. The earlier CI snapshot has been superseded by test-only backport heads. CI/queue state is independent of the live outcomes below.

| PR | Reviewed change | Code and evidence assessment | Remaining qualification |
| --- | --- | --- | --- |
| #596 | Verifiable VLM provenance | Clear; all 29 request/response chains independently bound | Hash integrity is not provider authentication or semantic truth |
| #678 | Kimi hosted completion contract | Clear; Kimi score and rich branches exercised on later integration; preference also passed on exact shipping source | Valid responses do not qualify all model judgments |
| #597 | Provider/score-gate contradictions | Clear; numeric gate remains distinct from provider judgment | Live control set is small; deterministic disagreement controls remain test coverage |
| #612 | Source-frame sampling provenance | Clear after extracted-count reconciliation and real ffmpeg regression evidence | Exact-head CI must finish; image-sequence live proof is not every video-codec proof |
| #617 | Backend-neutral artifacts and grade gate | Clear after rejecting evidence-free/stub/score-override promotion; real retained reports exercised | Evidence hashes verify integrity; they do not authenticate an untrusted producer |
| #622 | Paired judge disagreement preservation | Clear; real MiniMax/MiniCPM pair passed on exact shipping source; strict common-request invariant verified | No operational agreement-rate claim; incompatible Kimi shape fails before transport |
| #624 | Benchmark confusion failures | Clear; aggregation replayed exact fresh score responses | Offline replay is not new provider inference or model calibration |
| #627 | Blinded order-balanced preference | Clear; exact shipping-source Kimi positive is visibly grounded and counterbalanced | MiniMax unresolved outcome remains failed; audit-only and model-specific evidence |
| #634 | Visible terminal-evidence gate | Clear; complete, incomplete-observation and gray controls separate at fixed threshold | Missing visible terminal evidence does not prove an underlying action physically failed |
| #647 | Audit-only rich visual reviews | Code clear; strict failures, dimension disagreement and gray negative behavior verified | **Positive paired-rich live acceptance remains unsatisfied. Do not describe as fully live-qualified or proven impressive/useful.** |

The earlier grade-gate and video-count correctness findings were reproduced and fixed. Shared profile handling now preserves experiment invariants. Default preference rubric and SDK finalization contracts are documented; completed evidence can be finalized without new inference. The final prompt changes preserve strict parsers, thresholds and frozen expectations. Separate Claude Opus architectural review sessions and exact outputs are retained; their review tokens are not counted as workload inference.

## Exact shipping-source qualification

The initial 25 attempts included #647, which changes the shared HTTP transport by adding an optional exact request body and consistent retained-response latency handling. They were not sufficient by themselves to label the first-nine shipping source qualified. A separate four-call freeze therefore bound the actual #634 source, its 20 source files and both harnesses. Its paired-judge and Kimi preference request bytes were identical to previously frozen requests; source, prompts, models, controls, settings and expectations were checked before inference. Both committed live cases passed. This is source-specific qualification, not selection of a successful rich review; no additional rich request was made. All prior 25 outcomes and the new unsupported MiniMax frame-range claim remain retained.

## Committed coverage amendment

Holding #647 must not withhold the live tests for earlier features. The judge-only subset now lands in #622 (`00fff528c3a3c8f9e09a541dcf3976e999195ed3`) and propagates through #624 (`d65de5fe9a1d4cae34026d7ab94b4fb55642e7a7`). The judge-and-preference subset lands in #627 (`5cdda09616f2f08645ddd3f1679938b31161e12f`) and propagates through #634 (`afa83be5230667c2377f774da6dfb37007e59487`). Independent AST comparison verified every retained assertion helper, active dispatch path, final report persistence, private-mode check, frozen expectation, and test marker against the actual live-tested file. These heads change only that test file. The complete #647 file and entire final integration tree remain byte-identical to the frozen live source. No new inference was performed, and no expectation was relaxed.

## Receipt bindings

- Pixel-first findings SHA-256: `f15464d6cf89c8bcb2fbfbac75e4e0e8fa5abb52df514f68d5f66e426d074c65`.
- Initial experiment freeze SHA-256: `13df1270c43e233750857ab2425a636119f7a22741d04eb3a837909196f66e36`.
- Final-source matrix freeze SHA-256: `f98804e6a27b8a390caf4574efa105e60c7d84805d91201e2567cd6994b7f483`.
- Additional profiles freeze SHA-256: `483eae070a4a62721f8970975f4c5d5aeead9f5f831f03b2199aae4b988e58c7`.
- Exact shipping-source freeze SHA-256: `c55a2795b35ce48f06977433386986e1e1b6c52fdf814d086a9452a6ab57d7ab`.

Private raw evidence remains access-controlled. This summary intentionally contains only public code/model identifiers, measurements, limitations and integrity hashes.
