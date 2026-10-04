# PR833 completion and identity integration bridge

This addendum preserves the original execution identities and observations in
[the signed earlier proof](https://github.com/nebius/nebius-physical-ai/blob/64e4d2873b78933fa619aed2e32dcf53ed8a98b7/docs/testing/evidence/cursor-vlm-main634-20261003/pr833/readme.md).
It records integration with landed completion checks, not a new provider run,
calibration result, or independent approval of a future source revision.

## Frozen source and comparison

Signed integration source `1ce45e2b7a9fb44bcd1fcc7404f313ce6c89f617` has tree
`b8c8921428609fa0cd91f56e2227e63b05cd59d5`. Its parents are published
`bc333102bcd9346fa78ac215aba64ca494076bd1` and actual landed main
`935bc9de16aebd9c08600fdba195781615c15eca`.

The three conflicts were the changelog, runbook, and shared client response AST
pin. Both completion/whole-run-abort and model-disclosure documentation remain.
The combined hosted-response AST pin is
`356c3bacc9440618b641d716940bc4569b77e6ebdfa42126531470bb83e325da`;
the assertion and mutation controls remain active.

Compared with original hosted execution
`760c77b6665d7422d43b64a93241d954921edafc`, exactly three evaluator function
ASTs changed: `_strict_comparison_verdict`, `_hosted_structured_response`, and
`_verdict_with_evidence`. They exactly match the previously accepted combined
completion/identity implementation at
`4d7f25457f730eae4178ccab205327f08e3c9c2c`. Every other evaluator function/class
AST is equal to original760; request, prompt, frame, manifest, rubric, and
parser behavior outside those three functions is preserved.

| Source file | Current SHA-256 | Comparison with original760 |
| --- | --- | --- |
| Evaluator | `d373a19e6a746c86aa8b93c1e498f9ff93069933d46aa5522ff8e70ac71a1e33` | Three composed functions differ |
| Token Factory client | `4c2c60f217c61ddf4c4441b706a7b8c3c73273b8d241c9ee70e5e09e0a391f6d` | Byte equal |
| VLM CLI | `65e7fff58b20cb70e0023d4d8c8cc4c3d6479867221decca2b7db8d709e19c4d` | Byte equal |
| Promotion evidence validator | `0bcb218a48299e475d1ff538fea17f8767bffe8db8e975c7110498d65bccd581` | Byte equal |

The package declaration, CI requirements, CI constraints, and lock file are also
byte equal to original760. The fresh owned constrained environment contained
139 distributions. Complete module/package observations surround only this new
CPU execution; they do not backfill earlier compact three-source-hash records or
the distinct later 146-package contexts.

## Fresh CPU checks at the frozen integration source

- Affected completion, whole-loop-abort, client/profile, model-enforcement,
  optional-claim, grade, caller, CLI and paired-audit union: 1,360 passed,
  two inherited opt-in GPU tests skipped, eight warnings, 26.07 seconds.
- Guardrails plus onboarding smoke: 5,927 passed, one skipped, one warning.
- Precheck: 270 passed; dependency fingerprint, Ruff and formatting passed.
- Generated CLI documentation was current. Observed source, imports and package
  inventory were unchanged before/after each execution.

Four original own760 responses, all with exact completion status `stop`, were
replayed through the current canonical CLI writer and promotion validator in a
network namespace with socket and HTTP-send denial. Exact request objects,
manifest hashes, response bytes, normalized frame hashes, decoded RGB, scores,
rationales and enforcement disclosures were preserved. MiniMax API and the
hosted-compatible Gemma self-hosted-client panel each retained positive/negative
scores 1.0/0.0 and promote/loop-back decisions. Enforcement remained true for API
and false for self-hosted. New provider calls and network attempts were zero.

Canonical, legacy-only, malformed-canonical with favorable legacy, and
missing-both controls passed. Present malformed canonical output remains
authoritative; a valid historical legacy report remains readable only when the
canonical file is absent. An incomplete current producer aborts the command and
must stop downstream promotion; reading historical artifacts alone cannot prove
that invocation succeeded. Optional enforcement claims still require literal
booleans consistent with retained backend/identity evidence; missing historical
claims remain compatible. Stub and score overrides remain no-call paths.

The first new replay harness attempt failed because it addressed a single-result
field as `requested_model` instead of `model`. The failed attempt was retained;
the corrected harness passed without modifying the frozen product source.

## Historical scope and limits

The original full-suite result remains its original `c391` execution: 39,812
passed, 213 skipped, one non-strict XPASS and 77.47% package coverage against the
60% floor. The accepted original 898-test security run and earlier source/review
bridges remain historical evidence. None is relabeled as execution at main935,
the frozen integration source, or a later prose-only publication commit.

The original independent oracle's 27 passes and 22 failures remain retained.
These fixed geometry cases do not estimate model error rates, qualify robot
terminal outcomes, prove safety, or validate an operator-owned GPU server.
Current-head CI, signatures and current-main integration state require separate
actual publication observations. A prose-only successor must compare the entire
`npa`, `skills` and `.github` Git trees with the frozen tested source.
