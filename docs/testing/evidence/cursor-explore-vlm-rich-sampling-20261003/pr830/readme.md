# PR830: deterministic terminal-stratified sampling validation

Known-count `keyframes` reserves part of its sample for the terminal region while preserving first/final endpoints. `sequence` keeps its existing uniformly spaced indices; unknown-count fallback retains honest null source indices/timestamps. Landed #612 source-kind/count/index/timestamp and evidence-v2 bindings are preserved.

For six source frames and a three-frame budget, the frozen indices are sequence `[0,2,5]` and keyframes `[0,4,5]`. Tests cover 128,128 count/budget combinations, source-image/NumPy byte identity, real decoded-video timestamps and unknown-count negative cases.

Six actual hosted requests ran once at `f4787965d19ae1269808802d5c44aabe17300eb2`, crossing complete/incomplete/gray diagrams with both strategies, at the frozen 0.8 threshold. All six returned HTTP200/stop and passed strict response parsing. The live-test harness then failed all six tests in its artifact writer because it omitted the required result destination. Those failing logs, exact requests and retained decoded response text are preserved. They are not six live-test passes.

Offline adjudication of those original responses at `289530b47e2b462841c4105e08eeca7bc5c8673b` matched all six fixed labels: complete scored 1.0 for both strategies; incomplete and gray scored 0.0 for both. No provider request was repeated. The repaired signed successor `8a531e01fb51864c593b6f73d585abac37e6f864` reproduced all six original requests byte-identically offline, retained the evaluator/client source bytes, and passed schema-v2 checks plus actual corrected-writer JSON round-trips. This is offline successor verification, not hosted execution at that SHA.

See sanitized [measurements](./measurements.json) and the self-contained [original source-media download](./media.html). The latter embeds original PNG bytes and decoded-RGB hashes; download/open locally, since GitHub's source view does not render HTML. Raw operational evidence remains private.

Retention limitation: original observations capture decoded provider response text and its UTF-8 re-encoding hash, not separately retained wire bytes. Independent #647 finding 647-S1 reproduced invalid/non-UTF8 failure-byte loss in the shared retention path. Its repair must compose into the implementation; the original six observations are not retroactively labeled wire-byte exact.

Limits: the earlier six-call SO100 visual-improvement protocol failed and remains failed in the existing sampling evidence document. These tiny synthetic controls do not establish improvement over sequence, calibrated model quality, robot-policy quality or safety. Strategy prompt wording differs, so there is no pixel-only causal comparison. No labels/thresholds were tuned and no favorable provider retries occurred. The unresolved historical acceptance condition and failed live harness receipt remain explicit; the PR stays draft pending independently reviewed scope and all readiness conditions.
