# OpenPI policy-server redistribution record

Review date: 2026-08-19. This is an engineering classification, not legal advice.

The image is eligible for public redistribution only when the pushed registry
artifact passes the repository's layer, credential, SBOM, history, vulnerability,
and proprietary-payload scans. It ships the immutable Apache-2.0 OpenPI source,
its lock-resolved redistributable dependency closure, the NPA cache bootstrap,
and a digest-pinned CUDA runtime. It ships no pi0.5/Gemma checkpoint, tokenizer or
other model payload, credential, populated cache, operator data, Antioch runtime,
Isaac, Isaac Lab, Omniverse Kit, or host NVIDIA driver.

At runtime the operator explicitly accepts the two documented Gemma policies.
Only the cache warmer receives that run-scoped value. It resolves the exact GCS
object generations for both the checkpoint tree and PaliGemma tokenizer,
verifies every size and MD5, and atomically publishes separately keyed immutable
ready directories. The tokenizer's upstream-compatible path is a verified
symlink to its immutable identity and is never overwritten. The policy server
receives neither acceptance nor a model credential and mounts only the verified
cache read-only. The public
GCS source requires no provider credential; acceptance is not persisted in cache
metadata. A durable Kubernetes PVC can retain those separately governed bytes,
while the default `emptyDir` is node-local ephemeral state.

Any built artifact containing checkpoint/cache bytes or restricted NVIDIA/Antioch
payload is a different artifact and must remain private regardless of its tag.
