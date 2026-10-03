# Current hosted default vision path validation

Five fixed live tests passed on execution commit
`50111ebcb9a8604c63ad6cd9d702c3c703044e7d`: original key-scoped MiniMax catalog
membership, three saved image captions plus scene reasoning, positive and
negative visual judgments, and two attribute checks. The visual judges scored
1.0 and 0.0 at the frozen threshold 0.8; their provider booleans agreed with the
score gate. Attribute checks passed 2/2. Successful completions retained exact
requested/served identity and `finish_reason=stop`.

[catalog-membership-bridge.json](catalog-membership-bridge.json) adds the measured
MiniMax membership fact from the original 2,345-byte catalog response and binds
the unchanged catalog-test assertion across source revisions. This is a recheck
of retained bytes, not a new catalog/provider call or present-day availability
claim. The original response hash and execution identity remain unchanged.

Four additional fixed controls used the exact
[two-colour PNG](../../../workbench/evidence/vlm-default-vision-model-capability-regression-v1/probe.png).
MiniMax and Gemma returned the correct `left=red, right=blue`; both text-only
Qwen controls returned HTTP 400 specifically rejecting image input. There were
15 captured physical requests, including one catalog request, with no semantic
retries. The separate successful credential preflight has an unexposed internal
retry count and is accounted independently.

[evidence.json](evidence.json) records every captured request, response and
submitted image hash, decoded image dimensions/pixels, numeric usage, saved
artifact results, frozen protocol and source hashes. It omits private operational
paths, endpoints, provider request identifiers, credentials and reasoning text.
The supplied media and original historical four-call capability proof remain
available with their original provenance.

Main advanced through `f1ecd4255131374c417cdbb1f6c3e0d84b2a3514` after execution.
The client and caption core are identical by SHA-256; this bridge preserves their
original execution SHA/tree. Evaluator sampling evidence changed, and its judge
controls required a separate fresh provenance run. The separately recorded
`0b265c0c` execution independently asserted schema v2, actual source indices/counts
and exact sampling controls.
Fresh combined CLI/SDK registration, dependency/local and current-head CI gates
are required separately. The retained original run does not execute Kimi cases;
their explicit parameterized coverage remains in source, not a current live claim.

## Fresh affected provenance path

The landed evaluator interface was exercised separately on execution commit
`0b265c0cf422dbc24678f57ce7d391aadf21009d`, using only the same fixed positive
and negative judge controls. Both passed in two actual MiniMax requests, scoring
1.0 and 0.0 at the unchanged 0.8 threshold. Schema v2, sequence/max_frames=4,
source indices 0/1/2, source_count=3, null image-sequence timestamps, and exact
request-manifest/response/frame hashes were asserted, not inferred.
[provenance-evidence.json](provenance-evidence.json) retains the numeric receipt
with its separate execution head/tree and source bindings. The original 15
requests above remain original observations; the two new calls are not relabelled
as that earlier execution. No unaffected provider controls were repeated.

No hosted execution occurred at the current incoming-617 revisions of the test
or evaluator. The source bridge separately records their reviewed `2cc3cefd`
hashes; the catalog assertion's AST remains identical to the original run.
Private integration `380f1e7a` exercised both original judge responses offline
through the actual canonical writer and grade consumer with identical serialized
requests, raw response bytes, frames and source indices, retaining scores
1.0/0.0 and threshold0.8. Its 612 affected tests, one inherited opt-in GPU skip,
and seven invalid-score negatives were independently reviewed. Those two source
files are byte-identical at `380f1e7a` and reviewed `2cc3cefd`; this is a measured
source/offline interface bridge, not new inference or final current-head CI.

## Landed paired-audit integration

[incoming622-source-bridge.json](incoming622-source-bridge.json) binds separate
integration `8931a4ec` on externally landed main `cbd10457`. The four caption
files, default-contract and live-test bytes remain unchanged. Three existing
evaluator functions changed: task metadata resolution, response capture, and
artifact writing. Request, prompt, parser and score functions retain their prior
ASTs; blanket evaluator equality is not claimed.

This integration passed 1,350 affected CPU tests and replayed the same two
original `0b265c0c` responses through the actual scalar transport, canonical
writer and grade gate, with identical serialized requests, raw responses,
frames and indices, scores 1.0/0.0 and frozen threshold 0.8. Seven invalid-score
negatives still refused promotion. The new paired audit artifact remains
distinct from a scalar promotion report. This is offline integration with zero
new hosted calls, not inference or a replacement for the original full suite.

Exact reviewable media: [outside frame](outside-frame.png),
[inside final frame](inside-frame.png), and the original
[attribute frame](attribute-frame.png). Positive sequence = outside, outside,
inside; negative sequence = outside, outside, outside. Byte and decoded-pixel
hashes bind them to the retained requests. Complete sampling coverage here means
selected-frame provenance is known, not that all frames of arbitrary episodes
would be sent.

## Landed benchmark/confusion integration

[incoming624-source-bridge.json](incoming624-source-bridge.json) binds the
separate `5097b90f` execution on externally landed main `e1f627df`. It passed
180 affected caption, SDK, CLI, benchmark and scalar-response controls; one
inherited opt-in local-model/GPU control skipped. This is not a new full gate.
Nine client/caption/SDK/grade/dependency paths are byte-identical to the prior
published source. Five existing evaluator AST changes are confined to benchmark
reports, loading and metrics; scalar request, prompt, transport, parser and
result-writer ASTs are unchanged.

The incoming live benchmark tests referenced a deliberately removed dynamic
default constant. Two test references now use the explicit MiniMax model pin,
preserving independent drift checks and all response/score assertions. The
original lint failure is retained privately. The same two original responses
passed offline through the actual canonical path, with identical serialized
requests and raw bytes, scores 1.0/0.0 and frozen threshold 0.8; seven malformed
score negatives refused promotion. No hosted calls, full suite or additional
Claude pass were selected for this test-only composition repair. Original
hosted/full execution identities and immutable artifacts remain unchanged.

These fixed synthetic diagrams demonstrate the observed paths, not calibrated
model quality, temporal competence, physical correctness, global availability or
robot safety. Provider hardware was unobserved, and the input-response association
is collector-retained rather than server-attested. Labels, prompts, models and
thresholds were frozen; no unfavorable model answer was retried or relabelled.
