---
name: access-approval
description: Prepare, verify, and resume operator-owned Hugging Face or NVIDIA NGC access for exact NPA dependencies, including human-bound upstream approvals, while reusing one bounded run-scoped use declaration without duplicate NPA acceptance prompts.
---

# Access Approval

Use NPA to discover gated HF/NGC dependencies, verify exact access, and preserve
a resumable handoff. Keep the human as the only party that can accept terms.
For the maintained readiness commands and credential resolution rules, read
`skills/atomic/health-preflight/SKILL.md`. For agreement scope, the Isaac-only
`ACCEPT_EULA` policy, and redistribution boundaries, read
`skills/atomic/third-party-eula-preflight/SKILL.md`.

## Choose the narrow check

- During interactive `npa configure`, offer the optional full-catalog audit.
  The explicit equivalent is `npa configure --prepare-catalog-access`; opening
  pages remains a separate affirmative choice. Declining or postponing this
  audit must not block public assets or unrelated capabilities.
- Immediately before workflow execution, resolve only the selected toolRefs'
  exact HF/NGC dependency closure and require `Ready` before provisioning or
  runtime startup. Apply the same gate to local, hosted, serverless, VM, and
  Kubernetes execution; backend choice must not bypass artifact access.
- Re-run the exact check after the human completes upstream steps, then resume
  with the persisted redacted command. Do not treat a prior full-catalog audit
  as permanent authorization.

Interpret evidence, not credentials:

- `Ready`: an exact-revision HF payload-byte authorization probe or exact NGC
  registry artifact probe succeeded, or a cached success still matches the
  credential fingerprint, artifact revision, payload probe, and terms revision.
  This is operationally sufficient for NPA to proceed with that exact fetch
  without another NPA attestation. It means technical fetch entitlement at that
  moment, never legal acceptance, proof of compliance, or redistribution rights.
- `Pending`: credentials or a human-bound upstream approval are still needed.
- `Denied`: the provider rejected the credential or exact entitlement.
- `Unavailable`: the exact probe could not produce reliable evidence; do not
  convert this into success.

Token presence alone proves identity only. Tokens inherit account access; they
do not accept terms or own licences. A generic login, HF repository/revision
metadata response, or registry token exchange is not entitlement evidence.
Probe a representative payload path for every exact model or dataset revision
and the exact NGC container needed by the selected operation. Never select
README, model-card, licence, tokenizer, or config-only files because those can
remain public before approval.

The operator who supplies the credential and invokes the fetch is responsible
for using both consistently with the upstream terms. Record the operator's
exact field-of-use statement once under one bounded manager task/run ID. Child
solutions may reference the same scope record when their exact terms are
compatible; do not ask again per image. The record expires with that task/run
and is never a global or permanent declaration. Re-probe access when the
provider, account credential fingerprint, artifact, revision, probe payload, or
terms revision changes. Reopen the use question only when the operator changes
scope or authoritative terms require a concrete additional fact not present in
the record.

## Preserve human consent

Agents may discover requirements, show sanitized evidence, ask whether to open
the exact official pages, open them only after an affirmative answer, re-check,
persist a resumable handoff, and resume. Never click an acceptance control,
submit legal assent, automate browser acceptance, or claim that NPA accepted
terms. Hugging Face access requests are browser-only and user-bound.

Support an individual NGC account with its personal API key; do not require an
enterprise organization, administrator, or service key when the artifact's
upstream policy permits individual access.

In non-interactive or JSON mode, never prompt or open a browser. Return the
structured blocked plan and non-zero status, with official URLs and a redacted
resume command. Persist only owner-readable, non-secret evidence; never emit
tokens, keys, authorization headers, signed URLs, credential values, or private
infrastructure identifiers. Re-probe non-ready evidence rather than caching it
as approval.

## Register catalog changes once

When a solution adds or changes an exact gated dependency:

1. Add or update its `GatedAsset` in
   `npa/src/npa/workbench/model_access.py`, including provider, capability,
   artifact type, exact revision, representative gated `probe_path`, official
   page, and terms revision. The path must identify payload bytes, not metadata;
   catalog guardrails fail when a gated HF entry lacks a usable probe.
2. Add that capability to `ToolDefinition.access_capabilities` for every
   consuming toolRef in
   `npa/src/npa/orchestration/npa_workflow/catalog.py`.
3. Test that planning reports the exact artifact and that execution blocks
   before runtime until the exact probe is `Ready`.

This metadata is the inheritance point for configure audits, just-in-time
workflow gates, and agent prompts. A later catalog addition changes the catalog
digest and appears in the next full audit; workflows see it as soon as a
selected toolRef declares the capability. Do not maintain parallel allowlists.

Keep this flow separate from Token Factory availability, telemetry or privacy
consent, artifact redistribution rights, and the narrowly scoped Isaac
`ACCEPT_EULA` default. Public and unrelated capabilities remain usable when an
HF/NGC approval plan is pending.
