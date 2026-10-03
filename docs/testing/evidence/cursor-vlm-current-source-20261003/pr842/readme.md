# PR 842 postmerge evidence clarification

PR 842 was externally merged at `443d346fb7ec56ba44a4065873c530ae41f62f49`. Its closed branch remains unchanged; this additive note does not replay, reopen, approve or re-qualify it. The [original 20 HTTP rejection and two CPU checkpoint controls](https://github.com/nebius/nebius-physical-ai/blob/64b904cc1cc58edd3b9277cdaff8fa3b940553dd/docs/testing/evidence/cursor-vlm-strict-feedback-20261003/pr842/README.md) remain at their original execution identity.

The original false control has target 0.5, initial weight zero, zero loss and zero gradient. It demonstrates reward/checkpoint semantics, **not a parameter change**. Only the true control demonstrates weight change from zero to approximately 0.02. Original numbers, checkpoint hashes and artifacts have not been rewritten; this note supersedes the broader wording that both controls demonstrated an optimizer update.

The implementation does not establish authenticated application access. Earlier private review context saying “authenticated CPU service” was unsupported; the original service was task-owned loopback validation, not proof of product authentication. No authentication or credential configuration was changed.

Real first-party Claude Opus reviewed the exact original merge source. Its verdict was changes required in follow-up/context scope, with core strictness, reward math and whole-batch validation found sound—not human or retrospective merge approval. The actual unchanged `literal_values.require_boolean` requires exact bool type and raises ValueError on wrong types; eight later source-bound controls close the missing-helper context. A hypothetical TypeError from a substituted helper is not its actual contract. The endpoint still requires the success key; the helper's optional default is not endpoint acceptance of a missing field.

Four additional synthetic requests through the actual FastAPI ASGI application confirmed a **pre-existing adjacent defect**: malformed score string/null/list/object returns HTTP 500 without output or a training call. This is not introduced by the strict-success fix, and is not hidden as a pass. Schema-before-output-jail precedence returned HTTP 400 without output in a separate control. These later diagnostics ran at documentation-only source `64b904cc` with policy/helper bytes equal the original merge; no new GPU, VLM or training workload ran.

Pre-existing multi-worker checkpoint-publication concurrency and malformed-score handling require separately scoped follow-up routing; this continuation has not silently redesigned the externally merged feature. See [evidence.json](evidence.json) for exact source hashes and diagnostic scope.

The original native-Claude-unavailable observation remains historical; operator Mac first-party access became available later. CPU scalar-adapter proof does not establish robot/GPU policy performance, calibrated quality or safety.
