# Rich review: main935 integration source gate

This additive record binds the audited rich-review implementation to integration
commit `97a59e88c4f370b82d589fb7aeb60fb4b359bcde`, tree
`0b4385690ba9888d7cfd042cb4f6dbc518fd8019`, and main parent
`935bc9de16aebd9c08600fdba195781615c15eca`. It does not replace the
[main88 source bridge](https://github.com/nebius/nebius-physical-ai/blob/1e287ff11076256ce2f12e4b21ce111f11e9a005/docs/testing/evidence/cursor-explore-vlm-rich-sampling-20261003/pr647/main88-source-bridge-readme.md),
the [once-only changed-prompt proof](https://github.com/nebius/nebius-physical-ai/blob/1e287ff11076256ce2f12e4b21ce111f11e9a005/docs/testing/evidence/cursor-explore-vlm-rich-sampling-20261003/pr647/repair-a860-readme.md),
or retained adverse results in the [original record](https://github.com/nebius/nebius-physical-ai/blob/1e287ff11076256ce2f12e4b21ce111f11e9a005/docs/testing/evidence/cursor-explore-vlm-rich-sampling-20261003/pr647/readme.md).

[Measurements](main935-integration-measurements.json) separately bind source
hashes, current local results, and the scope boundary. The current integration
adds main's fail-closed self-hosted completion handling. The custom rich
request implementation, schema, CLI, SDK, Token Factory client, and grade
consumer remain byte-identical to `ca4748e1ee9226570a4a44c1d87798a7a5177976`.
Its original three a860 hosted responses therefore remain source-bound original
evidence; the current CPU/source gate made no new provider calls.

The fresh combined affected suite passed 1,049 tests with one inherited skip
and eight warnings. Fresh precheck passed 270 tests, and committed
merge-precheck accepted the exact integration tree. Built-in aggregate
confidentiality scanning and gitleaks found no findings. The historical full
run at `8565a9f9edb3d917b9cc5750c3b59881bc779f8d` remains 39,995 passed,
213 skipped, one unexpected pass, and 77.55% coverage at its original source;
it is not represented as a main935 or final-head full execution.

No claim is made about calibrated model quality, physical safety, a robot or
GPU workload, storage concurrency, or a passing paired result. The original
reversed paired strict-JSON failure and the six historical rich qualification
failures remain failures. The prior first-party Claude and independent review
receipts remain source-scoped rather than human approval or current-head
acceptance. An existing independent-review lane must close this current source;
publication, current-head CI, clean merge state, and the coordinator-owned
queue/lifecycle decision remain open.
