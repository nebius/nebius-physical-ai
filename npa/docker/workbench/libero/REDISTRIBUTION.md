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
may build and publish the neutral bootstrap at the exact reviewed development
SHA without customer credentials or runtime qualification. Its local scan
establishes the ordered-layer and flattened-rootfs inventory from the built
artifact, checks the independent rootfs export, pinned base provenance, exact
Buildx config, SBOM, and all payload/security gates. The pushed digest must match
that local inventory and config, pass the same payload scans, and support
anonymous pull. These are image-byte gates; they do not authorize runtime fetch.

The control plane mounts its storage verification key at
`/opt/npa/libero/output-storage-authorization-public-key.b64` as a root-owned
mode-0444 file. The image contains no customer or storage trust root. Runtime
verification rejects missing, writable, invalid, or customer-reused storage
keys, and still requires the customer-signed authorization before materializing
any runtime payload.

The common trusted workflow retains exact run-owned development cleanup and
refuses deletion of a digest that also carries another tag. Supported release
promotion and public catalog validation remain quarantined until real
exact-digest B200 capability evidence is accepted. Local source tests do not
prove publication, anonymous pull, or GPU execution.
