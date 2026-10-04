# Illustrative benchmark scope and observed limits

[Sanitized numeric/source evidence](summary.json) binds four actual hosted responses at `a01dda9c915a731e13bec6d72a5affcf9d49745d`, after landed schema-v2 provenance `f1ecd4255131374c417cdbb1f6c3e0d84b2a3514`. Original schema-v1 four-call execution at `60982ba6d948b1462547599236672571fc3dc3bf` remains separate, not relabeled. Reviewable original sample inputs: [place-block](place-block-clear-pass.ppm), [align-tool](align-tool-pass.ppm), [missed-target](missed-target-fail.ppm), [unstable-state](unstable-end-state-fail.ppm). They are2×2 color swatches and do not depict those tasks.

The unchanged sample tasks/rubric, MiniMax model, threshold0.8 and caller labels were frozen before inference. All four actual scores were0.0: TP0/TN2/FP0/FN2, accuracy0.5. The separately measured deterministic fixture report made zero provider calls and had accuracy1.0. Neither establishes physical task validity, generalization, independent-human calibration or an operational error rate. The disagreement is retained, not tuned away.

Reports retain illustrative_only, the three ordered dataset caveats and calibration=false. Actual API cases retain exact served identity, completed response hashes, normalized pixels and schema-v2 source index/count/sampling metadata. Invalid scope calibrated rejects before a fifth call. Legacy unspecified manifests remain compatible; invalid numeric/type/whitespace/enum metadata controls reject rather than normalize into stronger claims.

## Current-main execution

[Current-main sanitized evidence](current-main935.json) records one fixed
four-request hosted execution at `065f34b7e0de54d7903772eee5e8d3e01ca15d68`.
It used the byte-frozen four-item protocol (`25dee38f69e05f6b1b236b8a2a9c598aa219fdc54a1525eb65a8abeac5f31595`),
the unchanged MiniMax model and threshold 0.8. All four canonical request hashes
matched the reviewed proposal in order. Each response was HTTP 200, returned the
requested served identity, and reported `finish_reason=stop`.

The observed scores remained 0.0 for every item: TP0/TN2/FP0/FN2, accuracy 0.5,
and F1 0.0. Those two false negatives are preserved. This is a fixed illustrative
control result, not a calibration, physical-task, safety, generalization, or
operational-error-rate claim. A first harness setup error used an already-existing
evidence directory and therefore made zero benchmark transport calls; it is retained
in private evidence separately and was not a model-answer retry.

Opt-in command: `NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest npa/tests/e2e/test_vlm_benchmark_scope_live_e2e.py -q -s`. Use a fresh, unique evidence directory: the harness intentionally refuses an existing directory to preserve earlier receipts. Frozen protocol/dataset hash, each request/frame/prompt/rubric/response hash and numeric result are in the summary; exact raw operational results remain private. Source/evidence review by distinct Codex accepted the original observed scope only, not later source or final readiness. Native Claude was unavailable when historically measured; actual first-party review became available later and remains a distinct AI lane, not human approval. Full gates, publication scans, anonymous downloads and final published-head CI are separate conditions.

## Current-main strict-verdict source bridge

[Sanitized current-main bridge](current-main09548-source-bridge.json) records the
signed integration of landed main `09548f6939e0216d15b76819c6cc9d7537bf8e40`
with the PR838 disclosure contract. A bounded CPU bridge first exposed one direct
test helper that needed the existing explicit no-call argument; that first result
(464 pass, one failure, one skip) is retained privately. The narrow test-only
repair then passed 466 controls with one inherited skip, and current precheck
passed 270 controls. The bridge used synthetic or mocked completions only: it
made no hosted-provider call, native/GPU workload, full run, or new AI review.

The earlier four-hosted-response result in `current-main935.json` remains at its
own execution SHA and byte-frozen protocol; it was not rerun merely because main
advanced. This source bridge does not claim physical validity, calibration,
safety, generalization, an operational error rate, or final readiness. Current
published-head CI, all accepted catalogue-fixture components, the immutable native
binding, a final owner runner, full validation, and independent final-output review
remain separate requirements.

## Main44 audit-only capture bridge

[Main44 source bridge](main44-source-bridge.json) binds the signed integration of
landed audit-only rich-review capture, response journaling, CLI, and SDK interfaces
to the PR838 disclosure source. The frozen request evidence, public writers, grade
consumer, four public frame bytes, and scope protocol remain bound. The affected
offline CPU selection passed 740 tests with three explicit live-provider skips;
it made zero provider calls. The prior fixed four-hosted-response result remains
all zero with two false negatives, accuracy 0.5, and F1 0.0. It was not retried,
retuned, or treated as model-quality acceptance. This bridge is source evidence
only: held full-fixture prerequisites, independent output review, anonymous
publication verification, current-head CI, and coordinator lifecycle work remain
separate.
