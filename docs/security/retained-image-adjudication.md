# Retained private image adjudication

An otherwise complete scan can flag public cryptographic test vectors, parser
delimiters, or package metadata. Public origin alone cannot resolve a finding.
Each exact occurrence needs retrievable provenance, typed producer/consumer
evidence, and an actual independent review. Operational credentials and private
operator identifiers remain disqualifying. This procedure preserves the raw
failure and emits a separate acceptance receipt; it does not alter scanner rules,
publish an image, or establish functional GPU qualification.

The private qualification workflow retains the original authorization bytes,
completed raw report and ledger, successful native scanner controls, non-secret
helper inputs, source hashes, and original input snapshots before its temporary
workspace closes. The private receipt contains a retention manifest plus
content-addressed files. It excludes confidentiality policy values and literal
inventories. This hosted route supports CI regex policy only. A historical scan
without this retention cannot acquire a reconstructed authorization afterward.

## Trust and transport

The operator first verifies the complete canonical receipt's hash and original
run identity. The retention manifest hash is a separate authorization input.
The new hosted invocation checks out exactly the scanner revision that performed
the original scan. Every committed source and helper binding must match.

Original path, device, inode, timestamps and ownership receipts remain unchanged
in retained evidence. Transported files receive independent current snapshots.
The verifier authenticates each original path/hash to a specific current path,
then rechecks those new identities after adjudication. A copied file never
claims the original inode. The current hosted secrets generate a fresh private
policy file whose exact hash and compiled policy receipt must match the original
scan. Neither this file nor its values leave the runner.

Generic OCI input must preserve its original index, all runtime and attestation
descriptors, config and ancestor layers. The config revision comes from the
digest-bound runtime config selected by the original graph verifier. Docker's
convenience manifest is not required. Changed original identities, incomplete
graphs, unsupported schemas, changed policies and unaccounted occurrences refuse
acceptance.

## Prepare the independently reviewed request

The trusted CI host has a separate private namespace:

    ~/.local/share/npa/private-image-adjudication/requests/<request-sha256>/
      request.json
      review.tar

Create these files only in the existing operator-owned private namespace, with
the same owner and no writable public permissions. Preserve the original
canonical export and scan receipt in their existing namespaces. The request has
the exact fields below; all SHA-256 values are lowercase 64-digit hashes.

| Field | Required binding |
| --- | --- |
| schema_version | npa.private-image-adjudication.v1 |
| image_manifest_sha256 | Original immutable canonical export selector |
| scan_run_id | Original hosted run and attempt, digits-digits |
| scan_receipt_sha256, scan_receipt_bytes | Complete retained original result.tar |
| retention_sha256 | Original retained-input manifest |
| review_tar_sha256, review_tar_bytes | Exact independently reviewed evidence bundle |
| disposition_manifest_sha256 | Exact disposition manifest authorized by review |
| independent_review_sha256 | Exact independent review receipt |
| workspace_bytes | Assessed additional local working space |

The SHA-256 of the exact serialized request is the dispatch input. It is obtained
from the reviewed request; the caller never infers permission by hashing an
unreviewed upload. The independently reviewed disposition context also binds
retention_sha256 alongside the original authorization, image, source, policy,
report and ledger identities.

The uncompressed review tar contains regular files only: manifest.json,
review.json, and content-addressed evidence files named by SHA-256. Proof and
evidence bindings use relative names review/<sha256>. Links, devices, directory
traversal, duplicate members, extended metadata and unexpected paths fail.
Every finding occurrence must have exactly one accepted proof and corresponding
independent review decision. Logical path findings retain their original record
kind and full occurrence identity.

## Hosted execution and receipts

Dispatch private-image-adjudication.yml on the repository's current default
branch with request_sha256. The workflow grants contents: read only, checks out
the reviewed immutable scanner, obtains the existing DEV_VM_SSH_* credentials and
CUSTOMER_DENYLIST/optional INFRA_DENYLIST directly from GitHub secrets, and uses
the established host-key-verified SSH transport. It cannot run from a feature
branch. It never uploads private evidence as public Actions artifacts.

Before transfer, the runner requires space for the exact image archive, both
retained evidence bundles and their unpacked copies, plus the assessed workspace.
It rechecks remaining image/workspace capacity after unpacking. Source-host
retention capacity is a separate concern.

The invocation retrieves only the hash-selected request, review tar, original
canonical receipt and unchanged image export. A successful result is retained
privately under the original image selector and the new hosted run/attempt.
Raw failures remain in the original receipt. Missing retention, failed upload,
signal interruption or any unresolved occurrence produces a failed result.
Review acceptance does not authorize main merges, image publication, GPU jobs
or registry promotion; those retain their separate operational gates.
