# Hosted caption sentinel-normalization validation

Two frozen hosted controls passed on execution commit
`d533859918d2fc03c898ca1a39aee94fc0943958`: the positive image caption contained
all six required colour/shape terms, and the same instruction without an image
returned the exact `NO IMAGE RECEIVED.` sentinel. Both completions returned the
requested MiniMax model with HTTP 200 and `finish_reason=stop`. One offline
replay through the actual client and caption classifier produced one failed
item, `failed_count=1`, and aggregate failure.

The exact submitted [three-shape image](../token-factory-image-availability-sentinel/three-shapes-submitted.png)
and its byte/decoded-pixel hashes are linked in [evidence.json](evidence.json).
The report retains exact request-body and raw-response hashes, numeric usage,
the frozen protocol and source hashes. The credential preflight authenticated;
its unexposed transport retry count is distinct from the two captured inference
requests. No semantic provider retry or answer-driven tuning occurred.

Nested quote/emphasis normalization and positive/sentinel mixed batches are
covered by hermetic whole-path tests; this direct hosted run covers the original
sentinel only. Synthetic wrapped answers are not represented as fresh provider
observations. Composition with explicit thinking controls is a separate matrix
bound to its own immutable source and protocol.

Main advanced through `f1ecd4255131374c417cdbb1f6c3e0d84b2a3514` after execution.
The client, caption core and then-integrated live-test bytes matched the original
run by SHA-256. That historical binding remains unchanged; inherited successor
tests now hoist the credential check before fixture access without changing the
request or inference assertions.
Evaluator sampling evidence changed, but this caption protocol does not invoke
the evaluator. Original execution and integrated candidate SHAs remain distinct;
fresh combined registration/dependency/local and CI gates are separate.

This is synthetic path validation, not calibrated vision quality, a delivery
failure-rate measurement, or robot safety evidence. The cooperative sentinel is
prompt-sensitive, can false-fail a legitimate matching caption, and does not
detect arbitrary unavailable-image wording. Earlier normalization proof remains
bound to its historical source and is not relabelled as current execution.

## Current normalization scope and historical supersession

[normalization-scope-addendum.json](normalization-scope-addendum.json) records
the current recursive matching contract and measured source bindings separately
from the original one-layer and nested evidence packs. Those original files and
hashes are preserved; the one-layer exclusions describe that older execution,
not the current recursive classifier.

The current pytest population is **183 accepted forms plus 18 completed-caption
negative controls = 201 whole-path cases**, with five separate classifier-only
non-matches. The parameter values and IDs are identical at historical
`3fdced71` and reviewed `f1b1f7e6`; no pytest row was removed between them.
The historical standalone **202-case** claim names a different population that
has not been reverified here. It remains historical and is not silently
rewritten, identified with these 201 cases, or claimed as a new execution.

Recognized cooperative sentinel responses produce failed item/aggregate status,
retain raw visible answers, continue sibling captions and persist a partial
manifest. Provider transport/empty-content aborts and storage-write failures
retain their pre-existing behavior; this is not a promise to publish a manifest
after every possible failure.

## Landed parent and current-main integration

[landed-parent-source-bridge.json](landed-parent-source-bridge.json) separately
records the actual landed #826 parent `47f33358` and the integration of main
`7c09e0df`. Frozen execution `80004e84` passed 616 affected controls; subsequent
execution `c3fc6779` passed 633 incoming metadata/audit/registration controls,
both with zero skips and no new provider calls. Caption/client/SDK and the
201-case normalization test file retain the exact prior-published source hashes.
The inherited evaluator resolver and comparison-capture changes are explicitly
distinguished from the unchanged scalar request/parser/writer paths.

These are affected CPU integration executions, not new full-suite or hosted
executions. The original `f1b1f7e6` full result, `d5338599` hosted controls,
standalone 202-case limitation and all original evidence files remain unchanged.
Main-target publication and current-head CI are separate integration gates;
these private execution identities do not themselves claim queue admission.

## Subsequent terminal-evidence integration

[incoming634-source-bridge.json](incoming634-source-bridge.json) binds frozen
`dc098692` on actual main `88fd1b13`: 1,482 affected CPU tests passed, with one
skip and eight warnings. The quiet skip node identity was not captured.
The selected terminal/grounded/grade, paired, CLI/SDK, catalog and registration
controls passed; source and this checkout's packages were stable before/after.

This incoming main changes evaluator rubric, prompt, frame-label content and
grade reconstruction. It is not a same-request evaluator bridge. The caption
protocol does not invoke that evaluator, and its client/core/CLI/SDK plus the
201-case normalization test file retain the prior exact bytes. Consequently no
direct caption inference was repeated. The original full and hosted executions
above, their skips/XPASS and the unverified historical 202-case population remain
unchanged. This additive integration report is not a new full, hosted answer,
calibrated quality claim or current-head CI result.
