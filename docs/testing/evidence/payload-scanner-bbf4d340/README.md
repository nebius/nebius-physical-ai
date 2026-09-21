# Credential-scanner controls after independent review

PR [#688](https://github.com/nebius/nebius-physical-ai/pull/688) source `bbf4d340d7db9463bb55ad9988bde5a370e96877` passes the retained adversarial scanner controls. Real disposable keys are detected in application and other paths; ordinary OpenSSH parser binaries pass, while the same binaries with a key appended fail.

[Detailed outcomes and hashes](report.json) · [File verification manifest](SHA256SUMS)

These are **36 real CLI invocations over 12 minimal one-member Docker-save archives**, not a full-container acceptance claim. The ordinary Linux binaries were extracted unchanged from a retained image and are identified by their content hashes in the report. Keys were generated solely for these private controls, parsed, and confirmed to share the expected public key; no account credential was used or published.

| Input | Alpamayo2 | Cosmos3 serving | Cosmos3 Ray Serve |
|---|---|---|---|
| Parse-valid EC key: application path | Reject | Reject | Reject |
| Identical EC key: other path | Reject | Reject | Reject |
| Parse-valid encrypted PKCS8: application path | Reject | Reject | Reject |
| Identical encrypted PKCS8: other path | Reject | Reject | Reject |
| Credential assignment: application path | Reject | Reject | Reject |
| Identical assignment: other path | Reject | Reject | Reject |
| Ordinary text | Accept | Accept | Accept |
| Unchanged actual libcrypto binary | Accept | Accept | Accept |
| Unchanged actual ssh-keygen binary | Accept | Accept | Accept |
| Unchanged actual sshd-auth binary | Accept | Accept | Accept |
| Same ssh-keygen with disposable EC key appended | Reject | Reject | Reject |
| Same sshd-auth with disposable EC key appended | Reject | Reject | Reject |

Independent review also replayed 30 distinct case/caller outcomes covering content beyond the old read boundary, long whitespace, a 2 MiB quoted assignment, root/home credential paths and Docker-config aliases. Restoring the old application-skip branch or marker-only detector in isolated review copies was detected by nine native invocations. That establishes two independently checked mutants; the author's separate sixteen-mutant campaign is not claimed here.

All **193 focused tests** and the scoped source/static gates passed. The uninterrupted exact-source full suite passed **25,433 tests**, with 155 skips, 1 non-strict xpass and zero failures/errors. Process exit: 0. The source remained clean and unchanged. Original log SHA-256: `628ffb9827d617a4b45ce9a176f6a1aad520b3056e49cf3269ac78d7dc8a1ba9`. The repository `make test` target excludes live/GPU markers and end-to-end tests; this is the full local CPU unit gate. Hosted coverage CI, signature policy and required merge-queue integration remain separate requirements.

The original review rejected intermediate candidates for realistic OpenSSH false positives and an inherited application-path omission. Those findings are closed at the stated commit without adding directory, filename or whole-binary exceptions. The detector remains rule based: a reported pattern is not cryptographic proof of a live credential, and passing these fixtures does not certify arbitrary images as credential-free. GPU and VLM tests do not apply to this host-side scanning change.

This pack includes selected measured outcomes and original receipt hashes. Raw image members, generated private-key bytes, private paths and operational metadata remain private. It does not requalify the older image that supplied the binary fixtures or claim any public-image promotion.
