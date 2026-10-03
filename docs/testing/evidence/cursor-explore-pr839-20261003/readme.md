# Current hosted default vision path validation

Five fixed live tests passed on execution commit
`50111ebcb9a8604c63ad6cd9d702c3c703044e7d`: current key-scoped MiniMax catalog
membership, three saved image captions plus scene reasoning, positive and
negative visual judgments, and two attribute checks. The visual judges scored
1.0 and 0.0 at the frozen threshold 0.8; their provider booleans agreed with the
score gate. Attribute checks passed 2/2. Successful completions retained exact
requested/served identity and `finish_reason=stop`.

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
controls require a separate fresh provenance run. The live tests now independently
assert schema v2, actual source indices/counts and exact sampling controls.
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

Exact reviewable media: [outside frame](outside-frame.png),
[inside final frame](inside-frame.png), and the original
[attribute frame](attribute-frame.png). Positive sequence = outside, outside,
inside; negative sequence = outside, outside, outside. Byte and decoded-pixel
hashes bind them to the retained requests. Complete sampling coverage here means
selected-frame provenance is known, not that all frames of arbitrary episodes
would be sent.

These fixed synthetic diagrams demonstrate the observed paths, not calibrated
model quality, temporal competence, physical correctness, global availability or
robot safety. Provider hardware was unobserved, and the input-response association
is collector-retained rather than server-attested. Labels, prompts, models and
thresholds were frozen; no unfavorable model answer was retried or relabelled.
