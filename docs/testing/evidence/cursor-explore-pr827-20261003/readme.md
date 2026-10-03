# Hosted caption sentinel-normalization validation

Two frozen hosted controls passed on execution commit
`d533859918d2fc03c898ca1a39aee94fc0943958`: the positive image caption contained
all six required colour/shape terms, and the same instruction without an image
returned the exact `NO IMAGE RECEIVED.` sentinel. Both completions returned the
requested MiniMax model with HTTP 200 and `finish_reason=stop`. One offline
replay through the actual client and caption classifier produced one failed
item, `failed_count=1`, and aggregate failure.

The exact submitted [three-shape image](../token-factory-image-availability-sentinel/three-shapes-submitted.png)
and its byte/decoded-pixel hashes are linked in [evidence.json](evidence.json).
The report retains exact request-body and raw-response hashes, numeric usage,
the frozen protocol and source hashes. The credential preflight authenticated;
its unexposed transport retry count is distinct from the two captured inference
requests. No semantic provider retry or answer-driven tuning occurred.

Nested quote/emphasis normalization and positive/sentinel mixed batches are
covered by hermetic whole-path tests; this direct hosted run covers the original
sentinel only. Synthetic wrapped answers are not represented as fresh provider
observations. Composition with explicit thinking controls is a separate matrix
bound to its own immutable source and protocol.

Main advanced through `f1ecd4255131374c417cdbb1f6c3e0d84b2a3514` after execution.
The client, caption core and live-test bytes match the original run by SHA-256.
Evaluator sampling evidence changed, but this caption protocol does not invoke
the evaluator. Original execution and integrated candidate SHAs remain distinct;
fresh combined registration/dependency/local and CI gates are separate.

This is synthetic path validation, not calibrated vision quality, a delivery
failure-rate measurement, or robot safety evidence. The cooperative sentinel is
prompt-sensitive, can false-fail a legitimate matching caption, and does not
detect arbitrary unavailable-image wording. Earlier normalization proof remains
bound to its historical source and is not relabelled as current execution.
