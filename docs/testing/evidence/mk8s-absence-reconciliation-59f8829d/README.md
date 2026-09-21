# Supported recovery after verified MK8s resource absence

`npa cluster reconcile-absent --evidence-file <private-manifest.json>` closes an owned operation and matching project lease after exact provider absence is established.

The actual run retained eight provider responses covering four original resources and complete, scoped inventories. Independent review rehashed 25 receipts and replayed those exact responses against their requested resource identities and selectors. The original journal and matching lease reached durable terminal state. A fresh CLI invocation returned already reconciled without further provider reads. The command made no provider mutations.

The live producer was `742cfda0d72bf272806054d32469f531a508fb16`. Reviewed source `59f8829d23e16a553da310f26d7cc85f398a775f` has identical production bytes; its only successor change fixes a test fixture's explicitly unsafe directory mode under restrictive umasks. The earlier failed full run remains recorded.

The current source passed one uninterrupted full Linux suite: **25,561 passed, 155 skipped, one non-strict xpass, zero failures or errors**. Smoke, guardrail, security, documentation, lint and confidentiality gates passed. Native scanners found 607 baseline and 607 candidate findings, with zero new regressions or blocking findings. This is a baseline comparison, not a claim of zero findings.

Adversarial checks include a real process exit between terminal journal persistence and lease release, followed by a successful fresh-process retry. Unknown/live resources, unsafe audit paths, authority changes and ownership races remain refusing conditions.

[Machine-readable review](review.json) · [SHA-256 manifest](SHA256SUMS)

This is provider-state and local recovery proof. It does not establish a GPU workload, GPU capacity, robot capability or merge readiness. Raw credentials, provider identities and operational logs are not included.
