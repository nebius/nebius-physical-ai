# RoboCasa storage repair and exact-image evidence

PR #595 remains draft. This update records a substantive storage/service repair
and its validation, not completed GPU, policy-quality, or image-release acceptance.
The [earlier immutable evidence](https://github.com/nebius/nebius-physical-ai/blob/d34c525cb7d14fbaf7ae17325eb105c71b6aab94/docs/testing/evidence/cursor-explore-20261003/robocasa-readiness.md)
retains the original asset, synthetic CPU, scanner and image provenance. Its
statements about unavailable Claude and the then-current image are historical.

## Source and execution identities

| Exact source | Evidence scope |
| --- | --- |
| `d34c525cb7d14fbaf7ae17325eb105c71b6aab94` | Published predecessor; actual storage failure controls; first Claude review |
| `57c13f0a2c5c2b9035c580b2a6dcba01ced88bda` | Material storage, concurrency and Python 3.10 repair; full suite and host-provider controls |
| `8eae78ccf47778508d9cda64ac7d89478a634841` | CI-selection and image-source-label guards; sole focused Claude follow-up |
| `8c6341a0d20e4282e3b5d3624f45b905e7d6c98f` | Integration of published main `cbd10457`; affected tests, actual Python 3.10 CI selection, new image and image-provider controls |
| `df4f2a85eecf18146c95020b74ba29bcfe93ab66` | Integration of published main `7c09e0df`; 1,960 affected checks passed with two skips; unchanged RoboCasa, recipe and scanner bytes |

The RoboCasa runtime is byte-identical from `57c13f0a` through `df4f2a85`.
The capabilities SHA-256 is
`b2a50a1c1b9b5f0bd0a2e2e8217334b281bdac7fbe54357fa06896fa417a3971`;
the service SHA-256 is
`0a6a0ad2088025087bf17d9c1d91e03c1d7eee7a32a0781020b4918e922aaa7b`.
Integration and later proof-only changes do not alter earlier execution SHAs.

## Failures preserved and repairs

Five frozen real-provider controls on the predecessor produced three passes
and two safety failures. A destination-conditional COPY overwrote existing bytes;
the production publication helper also returned successfully after overwriting
a conflicting target. Conditional PUT and source-ETag COPY negatives returned
412 and preserved the expected destination state. These are observed
client/provider-path outcomes, not proof from captured HTTP wire traffic. The
original failing report remains unchanged, SHA-256
`933c79fd5843c9fa924998722679be0ee678f01a7231ac0a7f688256d99cf546`.

The repaired producer verifies each local file into a private, anonymous,
seekable snapshot while holding and rechecking no-follow source and parent
identities. It writes with `PutObject` and `IfNoneMatch="*"`, then checks metadata,
size and complete downloaded bytes. A fresh claim is checked in final population
verification, not immediately after its successful PUT. Safety still depends on
the provider enforcing the conditional write. Actual 412 permits only an exact-byte resume;
other failures remain failures. Claims, completion records, foreign-object
rejection and complete destination-population checks remain enforced. There is
no producer COPY, remote staging namespace, or remote deletion.

The service uses a dedicated asynchronous capacity-one limiter, leaving the
default shared worker-thread pool available. Accepted work keeps its gate until actual worker
cleanup, including HTTP waiter cancellation. The reproduced issue was starvation
of the shared 40-token pool, not a proven permanent deadlock. Python 3.10 asset
replacement now uses the held parent's anchored path instead of unsupported
`rmtree(dir_fd=...)`. Actual release-builder argument and asset-boundary CI
selection guards were added. PR #595 does not introduce a competing shared
storage helper; shared-helper ownership remains with PR #846.

## Local gates and review

The required full suite at immutable `57c13f0a` passed: 40,545 passed, 181 skipped,
one xpassed, 340 warnings, zero failures, in 2168.87 seconds. Log SHA-256:
`4d94a83144f137afe450e88bacc655e30c2eadec5c63dd1f511a7901f7366fee`.
The narrow `8eae78cc` child passed 143 affected checks. Integrated `8c6341a0`
passed 1,383 affected tests with one skip and all 174 tests in the actual
Python 3.10 CI selection, including all nine asset-boundary cases. The initial
integration run's six subprocess import failures remain retained; a fresh,
checkout-bound private environment corrected the editable-source binding without
changing source, dependency versions, or assertions. Current source-security
comparison reports 678 baseline and 677 candidate findings, zero regressions and
zero blocking findings—not zero findings at all severities.

The real first-party Claude primary review of `d34c525c` returned changes required.
The sole focused follow-up reviewed `8eae78cc` and returned clear at source level,
with no blocking repair defect. It executed no tests. Its low-severity advisories
remain explicit: one snapshot needs additional temporary-disk capacity up to the
largest output file, and an unusual cleanup exception could supersede a waiter's
cancellation exception. No arbitrary job, time or cost cap was added. The actual
follow-up result SHA-256 is
`3aea4baf391a988efaadef29e7ad41504a5e18c579e7c8b832de253109297339`.
Claude source review is distinct from independent whole-population evidence
review and is not human approval. Independent Codex accepted the material repairs,
both complete host/image storage populations, and the unchanged `df4f2a85` source
bridge. Its retained receipts have SHA-256
`1e76c332788568319a68a72353338937485ff27b558ffaf7a46c6fa133a47684`
and `54d0436f28d4f921a6ffb47fdae6217a6c280a688277b89c5f6b3d7cdf01a697`.
This is scoped source/component acceptance, not merge readiness, human approval,
or an independent review of inherited VLM/caption changes.

## Actual repaired-image storage behavior

The canonical local-only image built from `8c6341a0` passed its import, conversion,
untrained ACT/checkpoint and non-root registration gates. Its runnable manifest is
`sha256:3061a684e767c90bf8d288186a48439f4e9ff951f7dccc8a2d57e4e03a2ce382`;
its OCI config is
`sha256:507d27adc6e5c8d7d2c56597b9c7d783c6ad925b3bfb11bddd6f2d3e2c40aba5`.
The 9,850,599,424-byte archive has SHA-256
`b9950db3a477123b3653c913c6611844b692b0d8bc66ed47170ad707a2ffcb53`.
The original storage receipt mislabeled Docker's manifest-backed image ID as a
config digest. A separate hash-bound correction records this naming error; the
original artifacts and correctly selected runtime image remain untouched.

All 13 frozen storage controls passed using actual image dependencies:
boto3 1.42.9, botocore 1.42.97 and s3transfer 0.16.1. This is separate from the
earlier host 1.43.101 result. All ten private synthetic objects (2,323 bytes) were
read back and all 13 snapshot handles closed. The entire image-control execution
recorded 14 PUT, 47 GET, 25 HEAD and 11 LIST calls, with zero COPY, remote staging or DELETE.
A separate authorized-scope preflight cleaned only its own tiny probe.

Conflicting writes returned 412 and preserved the existing bytes. Two concurrent
different writes produced one 200 and one 412, with exactly one winner. An actual
completed write followed by explicitly injected response loss propagated failure;
a separately named resume accepted only 412 plus complete byte verification.
An injected prewrite SDK exception produced zero provider writes and no success.
Source mutation and foreign-prefix controls failed closed, identical retry made
zero mutations, and all six predecessor positive-control objects stayed unchanged.

Frozen protocol SHA-256:
`afb500a205cba66831cf70f232bda894b8ca0e31a5fe6ae2c94fbb8d25a9573e`.
Whole report:
`8e69e21801339937b8b06c26b54a01c7381956be5d1d89ba72f768cce41f846d`.
Operation ledger:
`8b4c2775516311da65eb82b489c09dc80f9fc3c389e3adcf6b766c8e4ba79794`.
Original invocation evidence:
`8192e69363e2a60189cd52af7ddbc659cb72cb17a77ac9518dc0116f0c58b50c`.
It records the immutable image, non-root execution, read-only root, dropped
capabilities and no GPU request or host credential-file mount. The task-owned
container exited zero and was removed. Credentials were not serialized in
evidence or command arguments. Tiny private objects remain for read-only review.

## Exact-image service concurrency control

Five frozen cases ran once against the production image-baked ASGI service,
using an explicitly synthetic capability executor. The image's FastAPI 0.136.1
and Starlette 1.6.0 differ from the host environment; AnyIO 4.15.1 and
httpx 0.28.1 are recorded separately. This used httpx ASGITransport, not Uvicorn
network transport, authentication, a GPU worker, simulator, or model inference.

A wrong source identity returned 409 without queueing or execution. With 41
accepted requests and one synthetic executor held, all 40 default shared-pool
tokens remained free. Actual system-info and health returned 200 before release.
Repeated cancellation of active and queued HTTP waiters did not release accepted
work. All 41 runs completed with at most one executor active; all 41 temporary
output directories were removed and the gate became available. There were 39
successful HTTP waiters and two canceled waiters. Independent review checked the
entire 88-event population, frozen runner, protocol and original terminal records.

Protocol SHA-256:
`8439157a887d5269e99c9b78a1404635aacad3906179110911ec2d126b80f73d`.
Report:
`73ef9a97365d69cc202aaf0182c935486744f66b93665468a85d8177dd585d6f`.
Event ledger:
`904fea3230ac027b8ffcdf70ca49ea2107a7da842d00c357491cab0f6bb03584`.
Original invocation:
`346665bc95542d2caec825d011f4f5a383cc1cbd6d8f492cac7de3f5952ac5d1`.
Separate terminal completion:
`652cff4e026268b89efb2fd6e9012f41ec3ecab0a3ad507cbb4177778b89dcfc`.
The immutable image ran non-root, network-disabled and read-only, without
credentials or a GPU request. Docker exited zero and the task container was
removed. The service report alone preceded terminal cleanup and is not substituted
for the completion record. These controls remain separate from the real S3 tests.

## Image gates and remaining limits

Graph verification passed over all 33 layers, 83,626 entries, 72,788 regular files
and 18,012,211,737 content bytes. The restricted-payload scan completed over
82,437 entries with no hits. Workflow-pinned Trivy 0.70.0 found zero findings under
the configured critical/fixed policy and zero secrets in its separate all-severity
scan. This is not an all-severity vulnerability claim. The SPDX 2.3 SBOM contains
576 packages and 1,555 relationships, SHA-256
`b876683aca042b571291be63122a25a736cbd0362a328024ec4c1e989aefc3ee`.
An initial incorrectly versioned/configured Trivy attempt is retained and excluded
from acceptance; it did not replace the correctly pinned gates.

The new whole-byte scan remains active with the unchanged scanner implementation
and the frozen 129-literal `exact-substring-v1` policy. Original incomplete scans
and predecessor findings remain retained. No findings exception, automatic
acceptance, complete customer-policy equivalence or image-publication authority
is inferred from other gates.

No repaired-candidate GPU kitchen rollout, learned-policy quality result or
hosted VLM evaluation has run. The original synthetic CPU clips remain only
conversion controls. Exact-image nonpublishing delivery, compatible GPU/service
and trainer capacity, accepted trainer-image resolution, frozen matched
policy/random rollouts, calibrated blinded VLM evaluation, final independent
evidence review and final published-head CI remain required. The committed
595/846 composition additionally needs dependency-fingerprint reconciliation.
The later `df4f2a85` integration preserves the exact image and RoboCasa source
bytes but includes real inherited VLM/caption/preference changes; those are not
claimed equivalent to their predecessors. Earlier full-suite and image results
keep their original execution identities, and new public-head CI is still required.
No image was published, and there is no merge, enqueue, human-approval, useful-policy,
generalization, sim-to-real transfer or robot-safety claim.
