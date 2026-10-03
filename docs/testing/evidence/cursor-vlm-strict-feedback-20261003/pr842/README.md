# PR842 post-merge strict feedback proof

PR842 was externally merged at `443d346fb7ec56ba44a4065873c530ae41f62f49`.
This continuation did not alter its closed branch or perform a merge.

At that original execution SHA, an owned real HTTP service rejected all20
malformed whole batches with HTTP400 and `success must be a boolean` before
creating any output. Both single invalid items and a valid-prefix/invalid-item
batch were exercised. Literal false/true controls each performed one CPU
optimizer update; their actual checkpoints were independently decoded.

| Literal success | Decoded target | Loss before | Loss after | Weight after |
| --- | --- | --- | --- | --- |
| false | 0.5 | 0 | 0 | 0 |
| true | 0.9 | 0.15999998 | 0.15602508 | 0.02 |

[Sanitized numeric and source-hash proof](evidence.json) records the exact
execution head/tree, source module hash, checkpoint hashes and frozen protocol.
The policy-container module is byte-identical in later main `f1ecd425`; this
is a source bridge, not a claim that the original run executed on later main.
Focused feedback/security tests passed34. The owned service was terminated.

Two initial harness failures expected reward fields that the HTTP response does
not expose; the raw failures remain private. Final verification used checkpoint
decoding instead. This is CPU scalar adapter behavior, not robot/GPU policy
performance, model quality, or VLM acceptance. Exact paths/raw data remain
private. Distinct AI review is requested; native Claude was unavailable and no
human approval or merge-readiness claim is made by this postmerge proof.
