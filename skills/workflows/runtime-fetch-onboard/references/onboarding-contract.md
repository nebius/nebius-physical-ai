# Runtime-fetch onboarding contract

Copy this worksheet into the solution's design evidence and replace every
placeholder. Keep it free of credentials and live infrastructure identifiers.

## Decision

| Field | Required value |
| --- | --- |
| Solution and capability | `<upstream name and exact capability>` |
| Packaging shape | `weights/data runtime fetch`, `whole-source/SDK runtime fetch`, or `build-your-own` |
| Image redistribution | `public`, `restricted`, or `unvalidated` plus the reason |
| Intended use | `<training, inference, evaluation, simulation, or other>` |
| Service use allowed | `yes`, `no`, or `human decision required` with official source |
| Field/output restrictions | `<obligation or none found>` |

## Six boundaries

| Boundary | Exact artifact and immutable identity | Official license/terms | Baked, runtime-fetched, or excluded | Evidence |
| --- | --- | --- | --- | --- |
| Source | `<repo@full-sha>` | `<official URL>` | `<choice>` | `<file/hash>` |
| Baked runtime | `<base@sha256 and packages>` | `<official URLs>` | `baked` | `<SBOM/scan>` |
| Weights | `<repo@revision and files>` | `<official URL>` | `<choice>` | `<access probe/hash>` |
| Dataset/assets | `<dataset@revision and files>` | `<official URL>` | `<choice>` | `<manifest/hash>` |
| Runtime cache | `<provider/artifact/revision/format key>` | `<inherits artifact terms>` | `runtime only` | `<empty-image/reuse proof>` |
| Outputs | `<artifact types>` | `<applicable terms>` | `external output` | `<fail-closed downstream gate, or note only when no automated consumer exists>` |

## Runtime delivery

| Field | Required value |
| --- | --- |
| Upstream endpoint/provider | `<provider, no signed URL>` |
| Immutable revision/digest | `<full value>` |
| Expected files/checksums | `<manifest path>` |
| Authorization source | `<runtime secret name or anonymous>` |
| Acceptance mechanism | `<vendor entitlement, exact product mechanism, or none>` |
| Cache tier | `node-local ephemeral` or `shared durable PVC/object storage` |
| Cache owner/access policy | `<operator/tenant isolation and authorized consumers; default ephemeral if unknown>` |
| Cache reuse permission | `<official source establishing permitted persistence/reuse>` |
| Temporary-download path | `<unique non-image path>` |
| Atomic ready marker | `<path and contents>` |
| Offline/restart behavior | `<safe reuse or explicit refusal>` |
| Cleanup | `<exact task-owned cache/workload cleanup>` |

## Validation ledger

| Gate | Command/evidence | Result |
| --- | --- | --- |
| Missing-access refusal before network | `<command and named failure>` | `<pass/fail>` |
| Empty image/cache and byte-level payload scan | `<scanner and image digest>` | `<pass/fail>` |
| SBOM, vulnerability, license, and secret scans | `<artifacts>` | `<pass/fail>` |
| Private staging digest before public publication | `<registry digest>` | `<pass/fail>` |
| Exact runtime fetch and checksum verification | `<non-secret manifest>` | `<pass/fail>` |
| Real capability on target hardware | `<run evidence>` | `<pass/fail>` |
| Restart/cache reuse and concurrent-population safety | `<test evidence>` | `<pass/fail>` |
| Workflow validate/plan and declared artifacts | `<commands>` | `<pass/fail>` |

## Claim disposition

- Accepted: `<only byte-scanned and live-proven capabilities>`
- Deferred: `<capability and precise access/legal/validation dependency>`
- Rejected: `<capability and why runtime fetch cannot make it permissible>`
- Human/vendor decision required: `<exact question and official terms link>`
