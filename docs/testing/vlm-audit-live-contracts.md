# Real hosted preference audit contracts

`compare-preference` is an API-only, audit-only capability. It compares two
images in both orders; it does not qualify a deployment or estimate a success
rate. Core behavior lives in `npa.workbench.vlm_eval`, shared by CLI and SDK.

## Scheduled execution

The protected `token-factory-live` nightly workflow keeps its existing hosted
suites and separately invokes `npa/scripts/vlm_audit_live_recheck.py` with
`--generated-controls`. The same protected Token Factory key is required. No
GPU, project, served-model endpoint, vendor weights, or new secret is needed.
The runner verifies credential presence and key-scoped model availability before
inference. Hosted use remains subject to the operator's provider/model terms;
model listing alone is not inference evidence or a license entitlement.

Three frozen visual controls exercise both preference directions and identical
images. Each control makes two real hosted requests with swapped image order,
requiring exact served-model identity, complete JSON, nonempty observations,
provider usage, request IDs, and exact retained request/response hashes. Pixel
hashes and expectations are fixed before inference. Confidence below high,
unresolved output, order disagreement, or errors require escalation and cannot
satisfy these frozen high-confidence controls. A failed control retains its
actual private outcomes rather than retrying until it agrees.

The lane rejects zero tests, skipped tests, xfail/XPASS, partial selection,
deselection, or a mismatch between configured, collected, executed, and passed
counts. Ambient pytest flags and CI shard selectors are removed. Synthetic
images are inputs to real inference, not substitutes for provider responses.

Raw requests, images, outputs, and pytest logs stay in an owner-only directory
outside Git. Scheduled raw evidence stays on the ephemeral runner; only the
allowlisted, confidentiality-scanned `receipt.json` reaches Actions artifacts
or summaries. Local operator runs preserve complete raw evidence for review.
There is no job-owned cloud compute to clean up.

## Operator execution

Use the checkout's isolated Python environment with `npa[dev,adapter]`. Provide
`NEBIUS_TOKEN_FACTORY_KEY` through an owner-only credential store/environment,
never through command arguments, workflow configuration, or committed files.
Verify service credentials with `npa workbench health preflight` first.

```bash
umask 077
npa/.venv/bin/python npa/scripts/vlm_audit_live_recheck.py \
  --generated-controls --evidence-dir "<new-private-directory-outside-checkout>"
```

Alternatively set `NPA_VLM_AUDIT_LIVE_CONFIG` to an owner-only JSON file. Its
`cases.blinded-preference` contains a `request` matching
`VlmPreferenceComparisonRequest` and `expectations`, or a nonempty `controls`
mapping whose values each contain those objects. Each expectation must freeze
two `mapped_preferences` values (`baseline`, `candidate`, `tie`, or `unresolved`)
and a boolean `escalation_required`. Other dotted report fields may also be
asserted. The configured `api_key_env` must name an explicitly populated
environment variable. The runner replaces output paths with unique private
locations without altering the source config. Run without `--generated-controls`.

The executable test is `npa/tests/e2e/test_vlm_audits_live.py`; direct invocation
requires both `NPA_INTEGRATION_E2E=1` and `NPA_VLM_AUDIT_LIVE_CONFIG`. Its ordinary
unconfigured skip is not live proof. Use the runner for fail-closed designated
validation. The separate served-model provenance config is not consumed here.

Scheduled wiring is a runnable contract, not a claim that this unmerged branch
has already run on the nightly schedule. Retain the exact-SHA operator receipt
and inspect future scheduled receipts independently.
