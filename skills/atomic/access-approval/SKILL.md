---
name: access-approval
description: Prepare, verify, and resume human-bound Hugging Face or NVIDIA NGC access approvals for exact NPA catalog and workflow dependencies, including agent prompts that may open official pages only after affirmative consent.
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
  It means technical fetch entitlement at that moment, never legal acceptance.
- `Pending`: credentials or a human-bound upstream approval are still needed.
- `Denied`: the provider rejected the credential or exact entitlement.
- `Unavailable`: the exact probe could not produce reliable evidence; do not
  convert this into success.

An HF token or NGC key proves identity only. Tokens inherit account access; they
do not own licences. A generic login, HF repository/revision metadata response,
or registry token exchange is not entitlement evidence. Probe a representative
payload path for every exact model or dataset revision and the exact NGC
container needed by the selected operation. Never select README, model-card,
licence, tokenizer, or config-only files because those can remain public before
approval.

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
