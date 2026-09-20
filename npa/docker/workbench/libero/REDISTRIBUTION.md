# LIBERO neutral-bootstrap redistribution boundary

This image is a public-eligible **neutral bootstrap**, not a LIBERO runtime.
Its layers contain only the digest-pinned official Python slim base, the exact
Debian bootstrap tools recorded in `debian-packages.lock`, NPA-owned scripts,
and immutable manifests. They contain no LIBERO or robomimic source, MuJoCo,
PyTorch, CUDA, cuDNN, NCCL, NVIDIA wheels, BERT weights, task data, render
assets, demonstrations, populated cache, output, checkpoint, credential, or
private-infrastructure value.

The base is `python:3.10-slim-bookworm` linux/amd64 manifest
`sha256:999137905e8718de681744822ccd965e1950e1baba089035060418e05e1d7496`.
Docker's published SLSA provenance binds its immediate Debian rootfs material
to `sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867`
and its source to docker-library/python commit
`688a0b86bb44289df16a363e9f41d90514c1a5f9`. The corresponding official SPDX
SBOM and the signed 2026-08-24 Debian indexes were independently recorded
before this packaging decision. The exact published in-toto provenance blob is
SHA-256 `0d66ce85e6ecad0d044a1d4bae712afe24ff2eb0a5d89eb224944df3895f226b`; the
published SPDX blob is SHA-256
`b290dbd3087fc5d2cf4af106f1d253c417a26080b70bdcf440ff96314a12c2bb`.

`debian-packages.lock` maps every inherited or selected Debian binary to the
signed snapshot index and lists the immutable `.dsc`, upstream source, and
Debian patch archives for every source package. Those snapshot URLs provide
equivalent corresponding-source access for copyleft packages. Python 3.10.21
is distributed under the Python Software Foundation License; its upstream
source hash is `a0da1e72132e950154eca0f6f47d5db828454700de20e5113667940d81e0db04`.

At run time the customer must personally acknowledge the seven exact terms and
sign a short-lived authorization with a customer-controlled key. NPA may
authenticate the caller, transport the evidence, and validate it, but neither
the manager nor the control plane accepts or signs the terms assertion. The
authorization binds the customer signer and identity, run, workflow profile,
exact terms, source revision, runtime manifest, and immutable qualified image.
Missing, denied,
expired, invalid, or mismatched authorization refuses before cache creation or
network access. HF/NGC credentials prove upstream access only. Runtime fetch
changes delivery only; it is not consent and
does not grant use, redistribution, commercial, service, or output rights.
The fetched cache stays non-root, atomic, and separate from
`NPA_SMOKE_OUTPUT_DIR`; it is never uploaded as a workflow artifact. A bootstrap
owner seals it group-readable/non-writable, while fetched code executes as a
different unprivileged UID under a shared lock and stable descriptor. Warm reuse
revalidates the canonical governing-terms identity from the pinned manifest
plus a complete source/runtime/data/model file inventory without network
access, and refuses manifest drift, writable trees, or changed bytes. Cold
population resolves and verifies the official terms sources before its first
cache mutation. The qualification gate itself requires a cold-fetch receipt
from the current run.

The manifest separately pins seven official governing-terms documents by URL,
size, and SHA-256. They are resolved into ephemeral storage only after the
customer authorization passes and before any cache mutation; drift or unavailability
refuses. The image-shipped runtime requirements lock must exactly match all 135
artifact identities, and installation uses `--require-hashes --no-deps` from a
read-only wheelhouse. Materialization additionally requires a positive reviewed
size and license expression for all 135 artifacts plus a bounded total download;
the manifest closes those metadata checks at 3,277,640,175 total bytes. Runtime
materialization still refuses without the customer/run authorization. These
controls prove identity and refusal; they do not accept terms for a customer.

The candidate remains unvalidated and unreleased. The trusted public workflow
has a payload-free development-staging path, but it is conditional: a current
strict checked-in qualification record and complete local byte gates must first
bind the exact development source revision, OCI graph, and scanner identities.
An unqualified or missing record stops before any registry write; dispatch
inputs cannot self-attest qualification. Once that prerequisite is met,
staging may copy only the exact locally gated OCI graph into the official
private package, without a public tag or visibility transition. The inventory
binds every byte in each ordered uncompressed layer tar and the canonical
flattened-rootfs records. The dedicated complete-byte/layer/exported-rootfs
scanner requires equality before push. Both builds derive `SOURCE_DATE_EPOCH` from the exact source
commit; the package layer removes APT/dpkg/account logs and normalizes the
non-root account's shadow day to that epoch. It must also pass the
SBOM/provenance/security gates, anonymous pull proof, and an exact-digest B200
hard gate before any supported release or public catalog claim.
LIBERO public disclosure, public tagging, and registry deletion remain fail-closed
until the exact-digest live-B200 and registry-atomicity gates pass. No exact-digest
live-B200 authorization path has been accepted, and
the workflow has no registry-enforced exclusive-writer or atomic compare-and-set
primitive for visibility changes or identity-and-tag-bound deletion. Repository
workflow concurrency and repeated graph reads do not provide those guarantees.
The ordinary publication preflight also refuses absent and private destinations
before any registry write, rather than creating a private artifact in a path
that must subsequently refuse publication.

Any qualified private state is retained. Requested and failed-build cleanup may
validate identities and graph bytes, but explicitly refuse deletion and leave
versions and package configuration intact. They do not claim public downloads
can be revoked. Publication, public tagging, or deletion claims require a later
exact-digest live-B200 authorization and implemented, independently evidenced
registry-enforced atomic visibility/cleanup mechanisms. Source-only test results
prove refusal behavior, not live publication capability or runtime validation.
