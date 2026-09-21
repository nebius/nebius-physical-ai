# Kubernetes image-pull authority proof

PR [#683](https://github.com/nebius/nebius-physical-ai/pull/683) verifies what the target Kubernetes Pod actually admits and pulls. A successful host registry check cannot certify a missing target Secret, a changed ServiceAccount, changed placement, or an admitted cache-only pull policy.

Tested implementation `33f670a5a6eda11e66c1737f3e41201f67702e49` and published signed head `43e60003e633cd92b64e37c5577a976679eea964` share the exact source tree `74979d8a2461ad5e2bfd2c4c2bb312e4322664fa`. Signature reconstruction preserves all commit trees, parent-relative changes, original authors and messages.

| Validation | Result |
|---|---:|
| Uninterrupted full local CPU suite | 25,399 passed; 155 skipped; 1 non-strict xpass; 0 failures/errors |
| Complete registry-preflight module | 125 passed |
| Independent SkyPilot merge / realm / admitted-authority / defaulting-cleanup / original-boundary controls | 14 / 7 / 6 / 16 / 11 passed |
| Actual scanner regressions and base-to-candidate security comparison | Passed; 0 blocking findings |
| Real local Kubernetes cases | 6 passed; 34 actual kubectl calls |
| Owned cleanup | All 4 probe Pods and the local test cluster removed |

The live checks exercised anonymous immutable-image pulls, exact requested node placement, ServiceAccount-inherited pull-secret defaulting, a missing ServiceAccount, a nonexistent image digest, and a missing declared Secret while the host returned HTTP 200. Each created Pod was deleted using its observed UID; retained inventories show no remaining probe Pods.

This is Kubernetes 1.35.8 with containerd 2.3.4 on one local kind node. The pull-secret fixture contains empty public auth data. It does not establish authenticated private-registry success, a Nebius target, multi-node behavior or a cold-cache pull. GPU/VLM evaluation does not apply to this non-visual preflight code. RuntimeClass-added placement remains explicitly unverified.

Earlier candidates that incorrectly accepted admitted authority changes or cache-only pull policies remain recorded as failures. The accepted repair does not overwrite those results. The full local suite excludes live/GPU/end-to-end markers; the real Kubernetes checks above are separate executions. Hosted CI must still pass on the published head before merge readiness is established.

[Machine-readable outcomes and verification hashes](validation.json) · [SHA-256 manifest](SHA256SUMS)

Only allowlisted outcome fields and hashes are published. Exact commands, operational identities, inventories and logs remain private.
