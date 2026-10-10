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

### Original registry manifest profile

An image registry may serve a single Docker schema-2 or OCI image manifest,
rather than an OCI index. Such an image uses the separate
`npa.private-registry-manifest-qualification.v1` export schema. It has the same
five fields above, but `expected_image_id` is the **independently obtained
original registry manifest digest**. Config IDs, a Docker-save normalized
manifest, and a locally generated transport index cannot substitute for that
digest. The existing `npa.private-image-qualification.v1` index profile and
its attestation-graph requirements remain unchanged; there is no fallback
from a failed index verification to this profile.

The manifest profile preserves the original manifest bytes, config and
compressed layer descriptors, verifies every digest and size, and checks the
decoded layer diff IDs in order. It supports one Linux/amd64 runtime image,
OCI tar/gzip layers and Docker schema-2 gzip layers. Foreign URLs, encrypted
layers, unsupported media types, additional graph nodes, missing or unreferenced
blobs, duplicate tar entries, links and conflicting descriptors fail closed.
Repeated references to one valid layer blob retain their original order and
are decoded and counted for every occurrence.

Its report explicitly records zero attestations **in the bound manifest graph**
and `external_referrers: not-inspected`. This is not evidence that the registry
has no external attestations. It provides no provenance/attestation acceptance,
signature approval, vulnerability waiver, redistribution grant or native
capability proof. Those gates remain separate. Every original manifest/config
byte, compressed layer, decoded ancestor file including later deletions,
history field, archive header and padding still enters the existing complete-byte
scanner. Incomplete accounting, unresolved findings and unfinished scanner
children cannot qualify an image.

Prepare original bytes using an existing authenticated registry tool, with the
source selected by its independently recorded digest. A supported Skopeo
directory export preserves the original representation:

```bash
skopeo copy --all --digestfile "$qualification_root/copied-digest.txt" \
  "docker://$source_image_ref" "dir:$qualification_root/original-source"
```

Use a fresh destination under qualified private storage; never overwrite a
retained export. `source_image_ref` must contain the digest-only reference,
without a redundant tag. Keep credentials in their existing credential store;
do not put credentials, private references or paths in dispatch inputs. Do not
request manifest conversion, compression/decompression, platform filtering or
signature removal. Newer Skopeo releases also provide `--preserve-digests`;
when available it is useful additional protection, but the checks below remain
mandatory. Skopeo 1.4.1 does not provide that flag. See the
[Skopeo copy contract](https://github.com/podman-container-tools/skopeo/blob/main/docs/skopeo-copy.1.md)
and [directory transport](https://github.com/containers/image/blob/main/directory/directory_dest.go).

Require both the copy's digest receipt and SHA-256 of the raw `manifest.json`
to equal the independent original digest. A copy that converted bytes is a
failed export, not a reason to select a new expected digest. Keep the dedicated
source directory and all its newly created regular files owner-only. The
exporter accepts only `manifest.json`, the exact directory-transport version
marker and the uniquely named referenced blobs; detached signature files or
other entries require a separately reviewed representation and are not discarded.

```bash
npa/.venv/bin/python npa/scripts/image_byte_scan/registry_manifest_export.py \
  --analysis-root "$qualification_root" --trusted-root "$PWD" \
  --source-dir "$qualification_root/original-source" \
  --expected-image-id "$original_manifest_digest" \
  --output-dir "$qualification_root/original-export"
```

This writes a new `image.tar` and structural `export.json` receipt. The tar
contains the unchanged original manifest and blobs beneath content-addressed
paths, plus an explicit local transport index referring to that manifest.
The transport index is never described or used as the original registry
identity. The exporter verifies original compressed bytes and decoded counts
without running the image, extracting archive members or rebuilding it.
Its structural receipt does not qualify confidentiality. Stage the resulting
archive through the same private export mechanism using the new schema, its
exact archive hash/size and assessed workspace allowance. The trusted workflow
independently verifies and scans it again under the configured policy.

## Source, policy, and evidence boundaries

The interface runs only from the reviewed default-branch workflow. The scanner
is separately pinned to `ef7b307212c1335de2e1eb9bef4ba8d6a7c6d41c`, which includes
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
generic graph, raw scan report and records, native integration receipt, phase
journal, and the five named non-policy child diagnostic logs.
The pinned scanner's report and records store policy hashes and rule locations,
not policy values or matching text. Policy-check and authorization command
output is suppressed even from private diagnostics, because invalid rules may
appear in that output. Authorization, policy, other tool logs, image bytes,
and unknown files are excluded. Exact receipt bytes are streamed back
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

Each phase emits a fixed progress event, and each scanner child records its
exit code. The private phase journal is flushed to disk at each event. SIGTERM
and SIGINT mark the run cancelled, notify its owned child, and wait for the
child's existing cleanup protocol before attempting final private retention.
A signal during child creation is forwarded once its handle is available;
unrelated processes are not signalled. No workload deadline is added.
If the child does not cooperate, a subsequent SIGTERM or SIGINT forces that
exact child to stop before it is joined. This escalation requires another
signal; there is no cleanup timer. Repeated signals do not interrupt the final
private receipt transfer. A kernel task stuck in uninterruptible I/O can still
delay joining even after a forced stop request.
Cancellation always fails qualification, including a signal received during
receipt transfer. Acceptance requires the successful job and final public
summary as well as the bound private scanner report; an earlier private summary
alone cannot establish acceptance after a late cancellation. Forced process
termination or loss of the hosted runner can still prevent final retention.
A failed manifest sender is classified before its bytes are parsed or hashed.
Authentication, host trust, connectivity, and missing or unreadable remote exports
produce fixed transport codes; raw SSH diagnostics remain private. A successful
sender must still satisfy the exact manifest digest and schema.
If the connection or host storage fails during retention, the job fails and
cannot claim durable evidence; do not treat an earlier scanner exit as success.
The public summary preserves any original failure and separately identifies the
receipt-retention failure, without claiming that a private receipt was stored.

This qualifies only complete-byte confidentiality for the configured policy.
It does not replace payload/licensing, vulnerabilities, attestations, image
bootstrap checks, finding adjudication, or real CPU/GPU capability proof. These
synthetic transport tests cannot prove live SSH access, available runner capacity,
or native Linux scanner execution; the first reviewed dispatch must establish
those facts and retain its exact image receipts.
## Retained finding review

Complete hosted scans retain exact authorization and non-secret input evidence
for a later [private retained-image adjudication](retained-image-adjudication.md).
That separate default-branch workflow requires a hash-authorized independent
review of every occurrence, unchanged original image/source/policy bindings,
and explicit transport identity checks. It preserves the raw scanner verdict
and never exports policy values or establishes native GPU qualification.
