# Actual recovery after a launch failed before accepting a job ID

The supported `npa workbench workflow reconcile-absent` command recovered a real failed operation after fresh reads proved its controller resources absent. It preserved the original error and every earlier event, then added one absence-reconciliation event through a generation-checked update of the original local operation and project lease.

| Evidence | Actual result |
| --- | --- |
| Execution source | `b4c6568101e835e60e494206c6ec7ef71d908b61` |
| Real apply | Exit 0 on September 22, 2026, at 00:48 UTC |
| Accepted native job IDs | Zero; none invented |
| Fresh authority | One provider identity read, three exact Kubernetes 404 responses, and two complete metadata lists containing 41 Pods and 21 Services |
| Independent verification | All 21 retained evidence files rehashed; original errors, prior events and identity preserved |
| Local state update | Original operation and lease finalized as absent, with one appended event |
| Cloud mutation | None |
| GPU / visual judgment | Not applicable to this control-plane recovery operation |

The original submission failed storage-capability preflight. Its first absence decision came from a controller check; its second came from source-bound queue reconciliation. Historical queue stdout was not retained. These fresh reads establish current absence. They do not establish historical workload success, cancellation, an original Pod UID, or a cloud deletion. The historical outcome remains **unknown**. Local locks serialize cooperating NPA writers; separately timed cloud reads are not an atomic cloud snapshot.

The preceding preview was read-only and independently reviewed before apply. Identity ambiguity, changed original evidence, accepted native state, incomplete metadata or remaining resources are refusal cases. Earlier failed parser and source-validation attempts remain in the private record.

The published implementation is `04f64f4a966ad6f9b5e9b66703704ff714150570`. The full suite passed 25,690 tests, with 155 skips and one non-strict xpass; ten local gates and the current-main native security comparison passed. A final source-pin representation change preserves the exact four mapping values and all other tracked files; 157 focused consumers passed on the final tree. [Validation and exact source relationships](validation.json) retain the distinct live, full-test and publication identities. Hosted CI was pending when this proof was prepared.

[Execution measurements and source bindings](execution.json) · [Sanitized fresh-read observations](read-observations.json) · [Checksums](SHA256SUMS)

Raw provider records, configuration, resource names, credentials and original logs remain access-controlled. This report publishes only allowlisted roles, counts, times and hashes. Implementation and validation were performed through Codex while Cursor requests were blocked by the team's monthly per-user usage limit.
