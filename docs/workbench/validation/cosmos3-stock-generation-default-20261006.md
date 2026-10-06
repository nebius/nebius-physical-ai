# Stock Cosmos3 generation default, 2026-10-06

The stock `workflows/testing/cosmos3-generate.yaml` uses
`workbench.cosmos3.generate`. The repaired image selected for PAIDF variant
generation was absent from this action's governed validation scope, so image
preflight and submit still failed with the public-release quarantine error.
`validate-spec` and `plan-spec` check the workflow graph and do not prove that a
runtime image can be selected.

This draft adds only `workbench.cosmos3.generate` to the existing governed
Cosmos3 candidate. Its source remains
`6462f27f98e1ded31d17b00944f943db3381ac35`, which is in main's history, and its
exact digest remains
`sha256:1aa6c473d95863709766f02d3e199cf8cf860adbe585b409a23a82bae4a2c37e`.
The old public release stays quarantined and the candidate remains outside the
accepted-release inventory. Operator overrides and non-official registry
selection keep their existing precedence.

## Acceptance status

Native stock generation qualification is pending. Keep this PR in draft until
the exact candidate completes the stock Cosmos3-Nano text-to-image workload on
Nebius RTX PRO 6000, including its enabled guardrails and published artifact.
Preserve the stock prompt, seed, guidance, steps, CPU and memory requests;
the operator selects the target GPU and storage locations. An explicit immutable
override is required for qualification against main before this change lands.

The tests exercise actual image planning, submit's plan-only route, and
`preflight-images` with mocked external probes. They prove selection and probe
arguments; they do not prove image pulling, GPU inference, other generation
modes, checkpoint evaluation or policy training. Those additional Cosmos3
actions keep their existing quarantine.

After native qualification, record the exact source/digest, success status and
artifact validation here before marking the PR ready. Concrete infrastructure
identifiers and raw operational logs remain in private operator evidence.
