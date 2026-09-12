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
| Runtime cache | `<provider/artifact/revision/format key, or not applicable for build-your-own without runtime fetch>` | `<inherits artifact terms, or not applicable>` | `runtime only` or `not applicable` | `<empty-image/reuse proof, or packaging-shape rationale>` |
| Outputs | `<artifact types>` | `<applicable terms>` | `external output` | `<fail-closed downstream gate, or note only when no automated consumer exists>` |

## Shared delivery identity

| Field | Required value |
| --- | --- |
| Upstream endpoint/provider | `<provider, no signed URL>` |
| Immutable revision/digest | `<full value>` |
| Expected files/checksums | `<manifest path>` |
| Authorization source | `<runtime secret name, operator-owned build-time credential source, or anonymous>` |
| Credential phase | `runtime`, `build-only`, or `none` |
| Acceptance mechanism | `<vendor entitlement, exact product mechanism, or none>` |
| Cleanup | `<exact task-owned image/cache/workload cleanup>` |

Complete exactly one of the following packaging-shape sections. Do not invent
runtime-cache evidence for a build-your-own image, and do not use private
containment as a substitute for proving restricted-byte absence from a public
runtime-fetch bootstrap.

## Runtime-fetch delivery only

Use for `weights/data runtime fetch` and `whole-source/SDK runtime fetch`. For
`build-your-own`, mark this entire section `not applicable — build-your-own`.

| Field | Required value |
| --- | --- |
| Cache tier | `node-local ephemeral` or `shared durable PVC/object storage` |
| Cache owner/access policy | `<operator/tenant isolation and authorized consumers; default ephemeral if unknown>` |
| Cache reuse permission | `<official source establishing permitted persistence/reuse>` |
| Temporary-download path | `<unique non-image path>` |
| Atomic ready marker | `<path and contents>` |
| Offline/restart behavior | `<safe reuse or explicit refusal>` |

## Build-your-own delivery only

Use for `build-your-own`. For either runtime-fetch shape, mark this entire
section `not applicable — runtime-fetch shape`.

| Field | Required value |
| --- | --- |
| Restricted build inputs | `<artifact, exact digest/checksum, official terms, and operator authorization evidence>` |
| Trusted build receipt | `<builder plus non-secret input/result provenance>` |
| Resulting image inventory | `<image digest plus files/layers/history/OCI configuration/SBOM evidence>` |
| Build-secret absence | `<files/layers/history/OCI configuration/log/SBOM scan evidence>` |
| Private-registry containment | `<operator-controlled registry pull receipt and proof that no public tag/copy was created>` |
| Public-promotion disposition | `not permitted`, `deferred`, or `<separate verified redistribution/publication evidence>` |

## Shared validation ledger

| Gate | Command/evidence | Result |
| --- | --- | --- |
| SBOM, vulnerability, license, and secret scans | `<artifacts>` | `<pass/fail>` |
| Real capability on target hardware | `<run evidence>` | `<pass/fail>` |
| Workflow validate/plan and declared artifacts | `<commands>` | `<pass/fail>` |

### Runtime-fetch validation only

Use for either runtime-fetch shape. For `build-your-own`, mark this entire
validation block `not applicable — build-your-own` rather than fabricating
pass/fail download, restricted-byte-absence, or cache evidence.

| Gate | Command/evidence | Result |
| --- | --- | --- |
| Applicable refusal before network | `<missing gated entitlement, explicit documented opt-out, or missing exact product opt-in; command and named failure>` | `<pass/fail; not applicable only for anonymous access with no documented acceptance gate>` |
| Empty image/cache and byte-level restricted-payload absence | `<scanner and image digest>` | `<pass/fail>` |
| Immutable delivery digest | `<for authorized official publication: full-SHA public development digest and anonymous pull before GPU validation; otherwise: private validation digest plus explicit no-publication disposition>` | `<pass/fail>` |
| Exact delivery and checksum verification | `<non-secret runtime-fetch manifest>` | `<pass/fail>` |
| Restart/cache reuse and concurrent-population safety | `<test evidence>` | `<pass/fail>` |

### Build-your-own validation only

Use for `build-your-own`. For either runtime-fetch shape, mark this entire
validation block `not applicable — runtime-fetch shape` rather than fabricating
private-build input or containment evidence.

| Gate | Command/evidence | Result |
| --- | --- | --- |
| Exact restricted-input provenance | `<digest/checksum, official terms, and trusted-build receipt>` | `<pass/fail>` |
| Resulting-byte and SBOM inventory | `<private image digest and inspection evidence>` | `<pass/fail>` |
| Build-secret absence | `<image and build-log scan evidence>` | `<pass/fail>` |
| Private-registry containment and pullability | `<private digest pull receipt and no-public-copy evidence>` | `<pass/fail>` |

## Claim disposition

- Accepted: `<only byte-scanned and live-proven capabilities>`
- Deferred: `<capability and precise access/legal/validation dependency>`
- Rejected: `<capability and why runtime fetch cannot make it permissible>`
- Human/vendor decision required: `<exact question and official terms link>`
