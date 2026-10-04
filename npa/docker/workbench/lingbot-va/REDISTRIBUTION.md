# Redistribution: npa-lingbot-va

This is a source-only, runtime-fetch candidate for LingBot-VA LIBERO-Long
workflows. It packages the upstream [`Robbyant/lingbot-va`](https://github.com/Robbyant/lingbot-va)
source at `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb` under Apache-2.0, preserving
its `LICENSE.txt` in `/opt/lingbot-va`. NPA adapter code is Apache-2.0.

The image contains no model weights, LoRA/adapters, data, latent caches,
credentials, CUDA Python packages, or user acceptance records. The public
Apache-2.0 checkpoint identities are runtime-only:

- `robbyant/lingbot-va-base@68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c`
- `robbyant/lingbot-va-posttrain-libero-long@0e89d1e753019988aba484e8da2dc0810e264d9f`

The workflow accepts only an operator-staged copy of
`HuggingFaceVLA/libero@affa19c0de0f6bce2a7edd26dddef8a532e7e6f6`. Its immutable
card declares CC-BY-4.0 and credits the original LIBERO dataset. The raw data,
its source image bytes, and fresh derived MP4/latent/output artifacts are
neither fetched during image construction nor packaged into an image. The
runtime manifest retains the source revision, task/episode mapping, license,
and LIBERO citation so downstream run-scoped artifacts preserve attribution.
The separate `robbyant/libero-long-lerobot` CC-BY-NC-SA latent dataset is not a
workflow input. Source access is not treated as a license grant for weights,
data, outputs, or caches.

CUDA and Python dependencies are built into an operator-owned writable runtime
cache by `lingbot-va-runtime ensure`. Runtime installation uses the exact
upstream PyTorch 2.9/CUDA 12.6 major/minor contract and does not add an EULA,
acceptance variable, telemetry consent, or vendor-specific attestation.
Public release is not authorized by this source-only record. The only permitted
near-term use is operator-private validation after exact image scans, a native
GPU execution, and independently inspected candidate-digest evidence. See
`THIRD_PARTY_NOTICES.md` for attribution and dependency lineage.
