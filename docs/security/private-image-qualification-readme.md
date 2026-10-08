# Private image qualification

The manual **Private image qualification** workflow scans one exact saved OCI
image using the existing repository customer policy. It has no publication,
registry login, image execution, IAM, or public artifact operation. Dispatch it
from the default branch with only the SHA-256 of its manifest. The workflow
definition must first receive review and merge; this document does not authorize
dispatch or claim that any image has passed.

The existing `DEV_VM_SSH_HOST`, `DEV_VM_SSH_USER`, optional `DEV_VM_SSH_PORT`,
`DEV_VM_SSH_PRIVATE_KEY`, and independently verified `DEV_VM_SSH_KNOWN_HOSTS`
secrets supply the transport. No host, account, private URI, or path belongs in
dispatch inputs. Their presence does not establish which host they reach or
whether that account can read a particular task archive. The manifest handshake
proves those exact bytes are accessible before an archive is transferred.

## Prepare a task export

On the host reached by that existing SSH configuration, the source owner places
two regular files beneath the account's home directory:

```text
.local/share/npa/private-image-qualification/exports/<manifest-sha256>/manifest.json
.local/share/npa/private-image-qualification/exports/<manifest-sha256>/image.tar
```

The export directory must be owner-only (0700), and both files owner-only
(0600 or 0400) and owned by that account. The manifest must be singly linked.
The archive may instead be a same-filesystem hardlink to an already frozen,
owner-only read-only (0400) task archive, avoiding a second full copy. The helper
does not change the original file or its permissions. Link count, inode, size,
mode, and timestamps must remain stable across the read, in addition to the
independent exact-content hash checks. The held archive descriptor is hashed
again after streaming, so same-size alias changes cannot hide behind coarse
filesystem timestamps; this adds a sequential read without copying the archive.
A writable multiply-linked archive is
refused. No path component may be a symlink.
Do not stage credentials, policies, Docker state, or unrelated
archives. The archive must preserve the original supported Linux/amd64 OCI graph,
including ancestor layers. Do not rebuild an image to substitute for exact bytes.

The UTF-8 JSON manifest has exactly these fields, with no duplicate keys:

| Field | Value |
| --- | --- |
| `schema_version` | `npa.private-image-qualification.v1` |
| `archive_sha256` | SHA-256 of every byte of `image.tar` |
| `archive_bytes` | Exact positive integer archive size |
| `expected_image_id` | Independently obtained original OCI index `sha256:` digest |
| `workspace_bytes` | Positive integer free-space allowance assessed by the owner for scanner preparation, records, results, and the retained-result tar |

Hash the exact manifest bytes, not a reserialized document. Its directory name
and the public dispatch selector are that hash. The small metadata protocol
accepts at most 16 KiB; image archives have no fixed size limit. The allowance
must be based on the actual image and observed scanner output, including ancestor
records, rather than a generic image-size assumption.

The source host needs only manifest and directory space when an already frozen
archive can be hardlinked on the same filesystem; otherwise it needs the full
copy or an independently verified supported reflink. Private result storage also
needs the actual returned bundle size. The receiver checks available bytes before
accepting that bundle; it never deletes or modifies the original archive.

The job records actual available bytes and requires archive size plus the
declared allowance before preparation and again immediately before transfer.
It does not delete runner content to make room. A standard hosted runner may
be too small; then select an already-approved larger runner through a reviewed
workflow change. There is no caller-selected runner or hidden archive limit.
The allowance is a capacity assessment, not a proof that output cannot grow;
disk exhaustion or an interrupted scan cannot qualify an image.

## Source, policy, and evidence boundaries

The interface runs only from the reviewed default-branch workflow. The scanner
is separately pinned to `551b5da4d9c62297236c103a30040c78e3b50dd6`, which includes
the reviewed generic OCI verifier. The helper verifies that checkout is exact
and clean, including ignored files such as pre-existing Python bytecode, before
any scanner child starts. A scanner-pin update requires review; callers cannot select code.
Its existing pinned Go and native matcher preparation plus real native
integration checks run before the image scan. The original graph is verified
again on the runner, then the existing authorization and complete-byte scanner
run against that exact local archive. No received Python or archive member is
executed or extracted to the host filesystem.

`CUSTOMER_DENYLIST` is required inside GitHub. `INFRA_DENYLIST` is optional and
the public summary and native report explicitly record whether it is configured.
The summary validates the policy digest, requires customer policy to be configured,
and accepts only fixed configured/not-configured infrastructure statuses.
Missing optional policy must not be described as full infrastructure-policy coverage. Neither secret
is passed to SSH, copied to the VM, or logged. Scanner authorization and policy
files remain within the ephemeral private runner directory and are removed
when the helper exits normally. The job never uploads Actions artifacts.
SSH keys and host-trust files have a separate owner-only temporary directory
outside the image analysis root supplied to scanner children.

Only fixed receipt files are bundled: the manifest, capacity and summary,
generic graph, raw scan report and records, and native integration receipt.
The pinned scanner's report and records store policy hashes and rule locations,
not policy values or matching text. Authorization, policy, tool logs, image
bytes, and unknown files are excluded. Exact receipt bytes are streamed back
through the same trusted SSH channel without remote extraction to:

```text
.local/share/npa/private-image-qualification/receipts/<manifest-sha256>/<run-id>-<attempt>/result.tar
```

The receiver creates an exclusive owner-only run directory, hashes the complete
bundle, and returns only its size and digest. Receipt retention is required for
success. Public logs contain fixed status and failure codes, bounded exception
classes, numeric counts, and cryptographic digests only. Private summaries also
identify the failed stage. Host diagnostics, exception text, and raw scanner
output stay out of public logs.
A failed manifest sender is classified before its bytes are parsed or hashed.
Authentication, host trust, connectivity, and missing or unreadable remote exports
produce fixed transport codes; raw SSH diagnostics remain private. A successful
sender must still satisfy the exact manifest digest and schema.
If the connection or host storage fails during retention, the job fails and
cannot claim durable evidence; do not treat an earlier scanner exit as success.

This qualifies only complete-byte confidentiality for the configured policy.
It does not replace payload/licensing, vulnerabilities, attestations, image
bootstrap checks, finding adjudication, or real CPU/GPU capability proof. These
synthetic transport tests cannot prove live SSH access, available runner capacity,
or native Linux scanner execution; the first reviewed dispatch must establish
those facts and retain its exact image receipts.
