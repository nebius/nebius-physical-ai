# Hosted caption image-availability validation

Two frozen hosted controls passed on execution commit
`d5d146b2e2c7a8d9bc662d09ac44a007075ecb2a`: the positive image caption contained
all six required colour/shape terms, and the same instruction without an image
returned the exact `NO IMAGE RECEIVED.` sentinel. Both completions returned the
requested MiniMax model with HTTP 200 and `finish_reason=stop`. One offline
replay of that retained no-image response through the actual client and caption
classifier produced one failed item, `failed_count=1`, and aggregate failure.

The exact submitted [three-shape image](../token-factory-image-availability-sentinel/three-shapes-submitted.png)
is retained with its byte and decoded-pixel hashes in [evidence.json](evidence.json).
That report includes request-body hashes of the exact HTTP client bytes, raw
response hashes, numeric usage and the frozen protocol/source hashes. No headers,
credentials, provider request identities, private paths or hidden reasoning are
published. The separate credential preflight authenticated successfully; its
internal physical retry count was not exposed and is not included in the two
recorded inference requests.

Main advanced through `f1ecd4255131374c417cdbb1f6c3e0d84b2a3514` after execution.
The client, caption core and exercised live-test bytes are identical by SHA-256.
Evaluator sampling evidence changed, but this caption protocol does not invoke
the evaluator. The report preserves the original execution commit/tree and
records each source binding separately. Combined CLI registration, dependency
and CI gates require fresh verification; this source bridge does not certify them.

These synthetic controls do not calibrate general vision quality, operational
image-delivery failure rates, or robot safety. The sentinel is cooperative and
prompt-sensitive. A legitimate complete caption exactly equal to the sentinel
false-fails, and other unavailable-image wording may escape this narrow rule.
The offline replay is a classifier test, not an observed real delivery incident.
The original historical proof remains separately bound to its original SHA.
