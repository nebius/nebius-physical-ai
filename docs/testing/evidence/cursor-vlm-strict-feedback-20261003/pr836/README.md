# PR836 fixed VLM controls

[Sanitized numeric/source-hash evidence](evidence.json) preserves the original
execution and a separate fresh execution after landed sampling-provenance v2.
Four actual hosted responses per execution used a frozen white/red-circle
positive and uniform blank negative, unchanged rubric and threshold 0.8, and
one inference per model/case. No answer, label, threshold or model tuning.

| Observed path | Red-circle score | Blank score |
| --- | --- | --- |
| MiniMax hosted API | 1.0 | 0.0 |
| Gemma hosted-compatible self-hosted client | 1.0 | 0.0 |

[Positive input](positive.png) and [negative input](negative.png) are the exact
task-generated source control PNGs; source and submitted-frame hashes are
separately recorded. Fresh v2 evidence retains actual source index 0/count 1,
keyframes/max_frames 8 sampling controls, and complete request/frame hashes.
Later source bridges compare immutable evaluator/client bytes; no old execution
is relabeled as a newer head.

Whole-CLI synthetic replay controls rejected 20 malformed numeric/type/nonfinite/out-of-range scores
before publishing an artifact. Those mutations derive from retained real
completions but made zero provider calls; they are not naturally malformed
provider answers. Exact source/error/count proof is in the JSON.

These two geometry labels do not establish calibrated model quality, policy
performance, safety or deployment acceptance. Compatible self-hosted client
calls, where present, reached an authenticated hosted service, not an
operator-owned GPU server. Raw collector-associated responses and original
failures remain private. Distinct AI review accepted the original scoped
source/observed paths; current integration/publication review is requested.
Native Claude was unavailable; no human approval or final readiness is claimed.

[Separate strict/disclosure composition](strict-disclosure-composition.json)
records immutable inputs, 318 passing unit/client/sampling tests (one opt-in
operator GPU skip), and four actual hosted v2 controls with explicit call=true,
identity enforcement=true, calibration=false and retained limitations. The
synthetic no-clamp regression explicitly passes call=false while retaining its
failure spy, exact rounding and failed threshold gate. This is composed-path
evidence, not calibrated model quality or final PR readiness.
