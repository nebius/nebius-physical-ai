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

The official `robbyant/libero-long-lerobot@8c0313b1c7cd9fa3798798479cbf59b11af8979d`
dataset declares CC BY-NC-SA 4.0. It is neither fetched nor packaged here. The
workflow accepts only an operator-staged source URI and records that NPA has not
made a use, redistribution, commercial, or noncommercial assertion. Source
access is not treated as a license grant for weights, data, outputs, or caches.

CUDA and Python dependencies are built into an operator-owned writable runtime
cache by `lingbot-va-runtime ensure`. Runtime installation uses the exact
upstream PyTorch 2.9/CUDA 12.6 major/minor contract and does not add an EULA,
acceptance variable, telemetry consent, or vendor-specific attestation.
Publication is not authorized by this source-only record: it requires exact
image scans, a native GPU execution, and separate evidence for the candidate
digest. See `THIRD_PARTY_NOTICES.md` for attribution and dependency lineage.
