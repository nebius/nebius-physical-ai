# VLM experiment 1: retained results

Source `f006cde3b990ece18a13296507ac3334f2b16155`; freeze SHA-256 `13df1270c43e233750857ab2425a636119f7a22741d04eb3a837909196f66e36`. Ten hosted API attempts, no retries, no replacements, all HTTP 200. The client ran on CPU; provider hardware is not disclosed. Every received body and request start was retained before product parsing. The 21 submitted control PNGs preserve historical bytes and RGB values.

| Public path/control | Real calls | Result |
| --- | ---: | --- |
| Evaluate complete | 1 | Score 0.95, provider success true, score gate passed |
| Evaluate truncated | 1 | Score 0.60, provider success false, score gate failed |
| Evaluate gray | 1 | Score 0.00, provider success false, score gate failed |
| Paired judges | 2 | Both scores 1.00; distinct served models; identical requests except model; committed live test passed |
| Blinded preference | 2 | First order rejected misspelled `uncertainity`; reversed order preferred candidate at medium confidence; agreement ineligible; committed live test failed |
| Rich complete/truncated pair | 2 | First order rejected generic adjective `candidate` under neutral vocabulary rule; reversed order parsed; committed live test failed |
| Rich gray | 1 | Parsed no-evidence/unreviewable/unsupported categories; committed live test passed, but independent pixel review found invented speckles on uniformly gray RGB128 images |

Committed live pytest: 2 passed, 2 failed, 0 skipped. The rich and preference failures are retained failures, not successes. The rejected rich phrase described a “candidate demonstration”; it did not disclose actual private source roles, but violated the parser’s literal forbidden-word rule. The frozen prompt requested neutral source roles without explicitly listing those forbidden words. No parser or response was changed to pass it.

The public benchmark consumed the three exact fresh responses in offline replay: TP=1, TN=2, FP=0, FN=0. Those are three known controls, not an operational accuracy estimate. No fresh response contradicted its provider-success boolean. The public grade gate consumed unchanged canonical reports and returned promote_checkpoint / loop_back / loop_back. The frozen harness mistakenly expected the positive enum `promote`; the original failed comparison remains in its summary. A separate offline amendment verifies the correct public enum without new inference or rewritten evidence.

All 71 original run-file hashes were independently verified, along with source/commit/tree/harness, ten exact wire requests, ten starts, ten raw responses, report chains, and submitted image byte/RGB mappings. Independent pixel observations were frozen before provider interpretations were read. Real provider results do not establish physical correctness, release stability, measured downstream usefulness, robot safety, or model operational qualification.

The original summary remains unchanged in `run/summary.public-candidate.json`. `publication-candidates/summary.json` is a derived presentation that converts the source-digest map into a list to avoid a secret-scanner false positive on the token_factory filename plus its verified SHA-256. It and the grade-gate amendment pass Gitleaks; original candidates passed built-in infrastructure confidentiality. Raw responses, rationales, private model request IDs, controls, and source-role mappings stay private.

A separate second experiment has been prepared for general prompt/schema fixes, preserving the same full matrix and all first-experiment failures. No selective rerun or replacement of first-experiment evidence is permitted.
