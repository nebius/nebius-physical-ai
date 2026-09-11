# robomimic neutral candidate redistribution boundary

This candidate is deliberately incomplete. It contains the immutable MIT-licensed
robomimic source and a hash-locked set of non-CUDA Python dependencies on a pinned
neutral Python base. It contains no PyTorch, torchvision, Triton, NVIDIA CUDA,
cuDNN, NCCL, model weights, dataset bytes, populated runtime cache, credentials,
or run output.

The image is quarantined and must not be published merely because its source is
open source. Before any publication, an authorized transaction must re-resolve
the base manifest, build from the reviewed commit, inspect every resulting layer,
complete license/security/SBOM/provenance checks, validate the private digest, and
prove anonymous pull of that identical digest. None of those claims is established
by the checked-in candidate.

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
interpreter. This is byte identity and race resistance only; it grants no
license, entitlement, redistribution, or service right.
