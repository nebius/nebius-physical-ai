# Independent review of hosted evidence

All original requests, responses and failed expectations remain immutable. The reviewer inspected every control frame before fresh model interpretations, checked file bytes and decoded pixels, and then independently verified all request/response/report chains. Historical control semantics were already present in the review context; this is not a blinded human study.

## Experiment 1

Ten real requests returned the requested model with completed responses. All 71 retained files, 21 control frames and report bindings were verified. Normal scores separated complete, truncated and gray controls. Paired judges agreed on the positive sequence. Preference failed exact schema parsing in one order and was medium-confidence in the other; rich paired review failed a literal vocabulary rule. These remain failed acceptance checks.

Gray review's no-evidence/unreviewable/unsupported categories were appropriate, but its prose invented speckles in uniformly gray images. That unsupported observation is retained as a model-grounding failure.

## Experiment 2

All ten real responses parsed under the strict contracts. Normal scoring and the positive paired-judge boundary remained supported. Preference was unresolved in both orders: the judge could see only the two terminal stills and questioned cube identity, while the reviewer had also seen full sequences. Its caution is defensible, and the report correctly requires escalation.

Both rich review orders identified the complete versus truncated sequence and preferred the complete sequence at high confidence. They disagreed on reviewability for the complete sequence and artifact fidelity for the truncated sequence. The implementation correctly refused eligible agreement. Gray review was grounded in uniform gray pixels, with no repeated speckle claim.

The frozen live tests remain **two passed and two failed** in each experiment. No threshold, parser, response, confidence requirement or expected label was relaxed to turn those into passing results.

## Experiment 3: model-profile coverage

Five real requests exercise Kimi preference and MiniMax rich review at the unchanged final source and with the same inputs, prompts and expectations as experiment 2. The committed tests returned **two passed and one failed**. Both Kimi preference orders identify the visible completed placement at high confidence, and MiniMax correctly describes the gray control as uniform and without task evidence.

The MiniMax rich pair does not pass. Its first order parses but incorrectly marks the visibly completed sequence partial; its reversed response adds one extra trailing closing brace, so exact JSON parsing rejects it. Neither response is presented as qualified positive paired-rich evidence. Provider `finish_reason=stop` does not make invalid JSON valid.

Across the three frozen protocols, 25 requests and 188 original evidence files are retained and hash-verified. There are zero provider retries or repaired responses. Six committed audit-test executions passed and five failed. Successful single-profile results do not erase the earlier failures or constitute an operational model ranking.

## Experiment 4: exact shipping-nine tree

#647's shared transport changes are absent from the nine PRs being recommended for merge. A separate four-request protocol therefore qualifies `afa83be5230667c2377f774da6dfb37007e59487` directly. The judge and Kimi preference requests match the earlier wire bytes exactly, with unchanged inputs and expectations. Both backported committed tests passed: MiniMax/MiniCPM score the visible positive 1.0/1.0, and Kimi maps both orders to the completed placement at high confidence. MiniMax nevertheless cited “frames 7–11” in a seven-image input. That invented frame range is retained as a rationale-grounding flaw, even though the visible final placement supports its positive score. No new rich review, retry or response repair was performed.

Across all four protocols there are 29 real requests and 13 committed audit-test executions: eight passes and five failures. Earlier failures remain failures; the fourth run validates a distinct shipping source tree.

## What this establishes

These results exercise successful transport, provenance, strict parsing, actual negative controls and conservative escalation on the executed source. Inconclusive provider judgments are limitations of the observed model/control combination; no remaining code-correctness blocker was identified in the audit-only APIs. #627 has a successful Kimi preference case on the exact final source. #647 remains a draft because every positive paired-rich case failed its frozen acceptance expectation. Correct refusal/escalation and green CI do not replace the required positive live proof.

This evidence establishes neither operational model accuracy nor physical correctness, release/stability mechanics, measured downstream usefulness, or robot safety. Exact hashes and per-model outcomes accompany this review.
