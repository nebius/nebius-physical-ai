# robomimic neutral candidate redistribution boundary

This candidate is deliberately incomplete. It contains the immutable MIT-licensed
robomimic source, an exact 78-package Debian dependency closure, and a hash-locked
set of non-CUDA Python dependencies on a pinned neutral Python base. Its build
path stages the exact Git tree and canonical archive plus every exact Debian
package outside the image build, verifies them before use, and exposes the
payloads to the Dockerfile only through a read-only build mount. The Dockerfile
does not contact APT or Git. It contains no PyTorch, torchvision, Triton, NVIDIA
CUDA, cuDNN, NCCL, model weights, dataset bytes, populated runtime cache,
credentials, or run output.

The source and Debian lock metadata are reproducibility inputs, not built-byte
or license acceptance. Debian package copyright notices must remain installed,
and all applicable source-conveyance duties must close before public
publication. The image is quarantined and must not be published merely because
its source is open source. Before any publication, an authorized transaction
must re-resolve the base manifest, build from the reviewed commit, inspect every
resulting layer, complete license/security/SBOM/provenance checks, validate the
private digest, and prove anonymous pull of that identical digest. None of
those claims is established by the checked-in candidate.

The CUDA-capable training runtime is a separate boundary. An operator-supplied,
pre-populated, read-only runtime volume may be considered only after the applicable
CUDA/cuDNN distribution, use, and service-rights decision is recorded by an
authorized Nebius/operator representative, with NVIDIA guidance when needed. A
runtime fetch, credential, private registry, image selector, or environment flag
does not grant that permission.

The runtime volume's own inventory is not self-attestation. The manager must
select its exact inventory SHA-256 independently, and the bootstrap must match
that hash while observing the mount read-only. Execution copies only declared
objects into a private staging tree, verifies the copy again, removes write
bits, and atomically publishes that run-local snapshot before invoking its
interpreter. Because the runtime UID owns this copy, absent write bits are
hygiene rather than an enforced read-only boundary. The independently selected
inventory hash and observed read-only source mount supply byte identity; the
private snapshot supplies point-in-time race resistance only. Neither grants a
license, entitlement, redistribution, or service right.
