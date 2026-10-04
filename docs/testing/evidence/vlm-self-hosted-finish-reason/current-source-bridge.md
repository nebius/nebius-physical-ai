# Current-source completion bridge

This addendum does not replace or relabel the archived source-pair report.
The current base is `432ea8b0478872f59ce364fa1435f9f81a6a9e56`.
Frozen integration source `dc7c19d085ed400cff999511e129c619d043bedf` has tree
`9fce3373a58562294022a25f3e2c27533e21dfe3`; its current-main successor
`6d4e7e1202bd2579942de547a30e71b62ceebf08` has tree
`dc526da5c41a33a3c2e820dcb3a8f8bcc7259d3a`. Later additions of this note and
abort/legacy regression assertions do not change production evaluator bytes.

## Original execution and measured source bridge

Four actual hosted geometry responses were executed at
`2bcc2badfb102a17341df354f95782316874e7e3`, not at the integration heads above.
Their immutable [sanitized proof and source images](https://github.com/nebius/nebius-physical-ai/blob/585d7703c4c392022f0680d760433b839c909da5/docs/testing/evidence/cursor-vlm-strict-feedback-20261003/pr828/README.md)
retain original request/frame/prompt/rubric/response hashes and execution SHAs.
MiniMax and hosted-compatible Gemma each scored the frozen positive/negative
controls 1.0/0.0. This is not operator-owned GPU serving, calibrated quality,
physical task completion, or robot safety evidence.

Exact evaluator file SHA-256 at the original execution was
`011df1d16046c5d745c0e368f5dc0ab2dae45bef0cc42e66f09ba8db64d1c6d8`;
the integration file SHA-256 is
`103c454c74ff3b8687fe03d78f95a5dd4896f5c2b77dd002716a8e4fda1a444f`.
They are **not byte-equal**: canonical/legacy filename constants changed.
All 118 top-level function/class ASTs compare exactly equal. Production Token
Factory client and VLM CLI bytes are unchanged; their SHA-256 values are
`4c2c60f217c61ddf4c4441b706a7b8c3c73273b8d241c9ee70e5e09e0a391f6d` and
`2b3f02b361920466e5804e03b9546fa77b20535c75fa08c8f7e34a01dd2f83b7`.

Fresh offline CPU replay at `dc7c19d0` exercised all four original responses
through the actual CLI canonical writer and new promotion validator. Exact
request objects and canonical serialized request hashes, scores, labels,
rationales and response hashes were unchanged. Five canonical/legacy precedence
controls and 15 tampered completion/score/model/sampling controls passed. Zero
new provider calls occurred; this is not a network capture or fresh inference.
Current-main VLM/grade/caption paths are byte-equal to that integration source.

## Local gates and retained failures

The frozen `dc7c19d0` affected 12-file union passed 740 tests with one inherited
opt-in GPU skip; seven related contract files passed 1,466 tests. At `6d4e7e12`,
current Cosmos/SDK/CLI registration checks passed 1,402 tests with one inherited
opt-in GPU skip, precheck passed 270 and docs checks passed. Initial consumer
fixture results of 24 failures/716 passes and an intermediate required-keyword
fixture failure remain retained. Repairs mutate historical evidence only after
constructing a valid strict producer; no completion or promotion assertion was
weakened. Original full-suite execution remains at `2bcc2bad`: 39,824 passed,
213 skipped, one non-strict XPASS, coverage 77.47 percent against a 60-percent
floor. It is not presented as a full execution at either integration head.

## Accepted failure semantics and historical fallback boundary

An incomplete real judge response aborts the whole loop/sweep, before producing
a verdict for that response. No fabricated score is assigned. A completed first
rollout can leave its artifact behind, but an incomplete second rollout causes
nonzero CLI exit and no new aggregate report. Promotion must not continue after
that failed producer. Use new run-specific output locations; prior artifacts
are preserved, not deleted or represented as successful current execution.

The independent `grade_gate` reader intentionally accepts a valid, complete,
strictly bound historical real-backend report under the legacy filename only
when canonical output is absent. The exact regression is
`test_grade_gate_reads_legacy_vlm_result_when_canonical_is_absent`. A present
malformed/ineligible canonical artifact blocks favorable fallback. Filename and
hash checks authenticate neither a provider nor a current workflow attempt;
ignoring producer failure and manually reusing an old prefix is not supported
current-run promotion evidence. This remains an explicit lifecycle limitation.
