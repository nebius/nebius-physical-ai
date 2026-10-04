# Sampler source and strict-parser CPU bridge

This is additive proof, not a replacement for the
[six original hosted responses](actual-main88-hosted-readme.md). Inference
remains at `a52fb1b4e55c0114e8eb37204dfcbc42604e3be9`, tree
`cf8c3ef9b80635221dbda03ea9e83c50869f8f03`, with the request protocol integrated
from `88fd1b1356c942745244e49d2316800634c530ae`.

CPU replay ran at `c699564d9dd7b9dc7da9674bfbbb9675bb7d0cf1`, tree
`e7c9c19f8384bda677dee94f624185cdd9b831da`, integrated main
`09548f6939e0216d15b76819c6cc9d7537bf8e40`. It replayed six original exact
request objects and retained response bytes through the current public
evaluator, canonical writer and complete grade consumer, with zero network
calls. Every original result field matched except newly generated time,
requested time and measured latency. The added model-match-enforcement field
was separately checked as literal true. Six legacy reports intentionally
omitting that field remained eligible; 30 inconsistent or nonliteral variants
were refused by the complete grade consumer.

The unchanged threshold is 0.8. Scores remained `1/1/0/0/0/0`, producing two
promotions and four loop-backs. Request, prompt, frame, sampling, byte capture
and writer function ASTs remained equal to the original execution. Strict
parser definitions matched the landed integration; this is not a claim that
the entire evaluator file stayed unchanged.

The original six API reports use the unchanged hosted parser contract. Old
self-hosted compatible-parser-v1 reports are archival and ineligible under
compatible-v2, even if their literal fields happen to be valid. Their original
tags were not edited. Affected controls covered literal boolean success,
finite numeric scores in [0,1], nonempty rationale, no coercion or clamping,
completion metadata, invalid CLI artifacts and old/current parser eligibility.

The affected c699 run completed 1,466 passed, one inherited opt-in GPU skip,
zero failures and eight warnings. Head, tree, dirty state, imported modules
and installed packages were captured and stable before/after. Precheck passed
270; documentation generation finished before filesystem-sensitive tests;
merge-precheck and aggregate confidentiality/gitleaks passed. These are c699
executions, not final published-head CI or full-suite proof.

Signed candidate `6b7d0ed16495787735d0bcccb2495ffbb83cae0b`, tree
`451dab0d1df166a14d93c352c17917e23448d8d7`, differs from c699 only in accepted
catalog test fixtures. Production, documentation, skills and workflow bytes
are equal. Its own precheck passed 270 and dependency/environment checks and
aggregate scans passed. Distinct AI review accepted its exact consuming source
and native/network runner, not an unexecuted full gate or human approval.
The [source measurements](strict-source-cpu-bridge-measurements.json) bind
these identities separately.

The mandatory passing full and current published-head CI, main integration,
signatures and lifecycle remain separate requirements. A genuine full started
on exact6b on October 4 at 02:55:46 UTC with unchanged selection and coverage
floor. Two further inherited fixtures were subsequently observed waiting on
native processes. This proof does not call that running/incomplete execution
a pass, infer absent JUnit counters or claim recovery of shared filesystem
health. A future start requires the separately owned immutable fixture repairs;
accepted earlier catalog and native-component scopes are not withdrawn.

Both gray-control rationales incorrectly say white despite uniform
RGB(128,128,128). Missing completion evidence does not establish real task
failure. Three stylized states do not qualify a robot, continuous episode,
physical safety, causal motion or calibrated sampler/model quality. The
small-budget tradeoff remains three of 100 `[0,90,99]` and four of 1,000
`[0,450,900,999]`: early events can receive only the first frame.

Original six writer failures, failed SO100 improvement acceptance, reversed
paired failure, original39 full 27 failed/40,219 passed/213 skipped/one
unexpected pass/77.58% coverage, and aborted a52/2ba attempts with unknown
terminal populations remain unchanged. No label, threshold, prompt, model
retry or new hosted call was selected for this source/CPU bridge. Original
artifacts remain immutable. This proof does not mark PR830 ready.
