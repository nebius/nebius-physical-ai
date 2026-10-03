# PR828: changed-request evidence addendum

Actual hosted execution used signed source `c5085c8757f50889cc296bce69b25cbb4ed77e40` (tree `49e32764ea17a58bad8b6ed3f911ed82ccedf9aa`), based on landed main `88fd1b1356c942745244e49d2316800634c530ae`. Main634 changed the default rubric, grounded prompt, image-interleaved Frame labels and grade reconstruction. These four responses are new-payload evidence; earlier responses remain historical.

The frozen positive red-circle and negative uniform-gray controls used MiniMaxAI/MiniMax-M3 through the API backend and google/gemma-3-27b-it through the compatible self-hosted client. All four actual responses returned HTTP200, exact requested/served identities and completion `stop`; scores were **1.0 / 0.0 / 1.0 / 0.0** at unchanged threshold0.8. This is single-frame synthetic geometry, not calibrated model quality, robot terminal-sequence performance, safety or operator GPU deployment.

The original harness attempt **failed all four post-retention assertions**: it passed an in-memory tuple representation into a consumer requiring serialized JSON lists. All requests, response bytes, persisted reports and failures were retained unchanged. A separate network-blocked CPU check validated those same serialized reports successfully with **zero additional provider calls**. The in-memory boundary refusal remains in [numeric/source-hash evidence](evidence.json), alongside exact request, response, frame, prompt, rubric and source hashes. The original failed attempt is not relabeled a passing run.

MiniMax's blank-image rationale said there were no visible colors despite the gray background. That wording limitation is retained. No labels, threshold or answers were tuned, and no favorable-answer retries occurred.

## Current local integration and historical gates

At the exact execution source: affected tests **1,600 passed / two inherited opt-in GPU skips**; standard smoke **118 passed**; guardrails **5,899 passed / one skip**; protected-caller controls **39 passed / zero skips**, with no provider calls; precheck **270 passed**. Docs, aggregate confidentiality/gitleaks and merge-precheck passed. Before/after source, imported-module and distribution observations matched.

The original full gate **39,824 passed / 213 skipped / one non-strict XPASS / 77.47% package coverage** belongs to `2bcc2bad`, above the60% floor. It is not a full execution at this later head. [Original durable proof](https://github.com/nebius/nebius-physical-ai/blob/585d7703c4c392022f0680d760433b839c909da5/docs/testing/evidence/cursor-vlm-strict-feedback-20261003/pr828/README.md) and [earlier additive source bridge](https://github.com/nebius/nebius-physical-ai/blob/7e8fd7435c3c96502985cddd56d894fe80a16e66/docs/testing/evidence/cursor-vlm-current-source-20261003/pr828/readme.md) retain their original execution scopes and adverse attempts.

The unchanged public control images are [positive](https://github.com/nebius/nebius-physical-ai/blob/585d7703c4c392022f0680d760433b839c909da5/docs/testing/evidence/cursor-vlm-strict-feedback-20261003/pr828/positive.png) and [negative](https://github.com/nebius/nebius-physical-ai/blob/585d7703c4c392022f0680d760433b839c909da5/docs/testing/evidence/cursor-vlm-strict-feedback-20261003/pr828/negative.png). No private images or operational identifiers are published here.

## Design and review limits

Incomplete output aborts the whole run: an earlier valid rollout may remain, but no incomplete second report or new aggregate is fabricated. Shipped workflows stop before grade/publication/decision reads. Manually ignoring a failed run with an absent canonical report and favorable valid legacy report remains a disclosed compatibility limitation; present malformed/incomplete canonical reports never fall back.

First-party Claude Opus AI review is bound to its exact earlier reviewed source and output hashes, not human approval or future-head approval. Later incoming source/live independent review, current-head CI, actual-main integration and coordinator queue admission remain distinct. This evidence addendum alone does not declare merge readiness.
