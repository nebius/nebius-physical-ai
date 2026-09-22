# Real 3B/B200 encoder mechanism verification

**Four actual GPU arms passed independent byte-level verification. Restoration quality is still unproven.**

The shipped 3B entrypoint ran the two original main/light inputs with `sample` and `posterior-mode` conditioning in fresh processes. It executed the real VAE and both downstream noise draws, then stopped before diffusion or decoding. The posterior-mode arm selects the posterior mean after the upstream sample call, preserving random-number consumption.

| Check | Actual result |
| --- | --- |
| Source | `f1de67279ef6435884bc2670781eba98000e321e` |
| Image digest | `sha256:72a7774a8f7b5262bba5cfdb01155edee1acbc9bfc9e878fd777ce158e9b25ef` |
| GPU | B200, capability 10.0, 183,359 MiB, MIG disabled |
| Runtime | Torch 2.13.0+cu130, CUDA 13.0, driver 580.173.02 |
| Actual workload | One successful Job, four fresh-process arms, exit zero, zero restarts |
| Complete output readback | 84 arm files, 1,464,559,248 bytes |
| Selected conditioning | Independently reconstructed from the actual posterior; exact raw bytes match in every arm |
| Paired controls | Input, posterior mean/log variance/sample, recorded CPU/CUDA RNG states, diffusion noise and augmentation noise all match exactly |
| Intervention observed | Selected conditioning bytes differ between sample and posterior-mode for both inputs |
| Independent replay | Root separately reran the hash-pinned verifier over every real tensor/RNG file and obtained the identical result |

[Results and evidence hashes](results.json) and [every observed tensor/RNG digest](tensor-digests.json) are directly readable. Raw operational records and the complete tensor archive remain access-controlled; no credentials or private infrastructure are included here.

These runs did not generate restored pixels, score visual quality, or consume the held-out episode. They do not supersede the earlier failed H100/3B or B200/7B quality experiments. Next is a separately frozen actual 3B restoration comparison with the original LPIPS, temporal, spatial, detail, codec, unchanged-light and critical task-state gates. Public image release remains blocked on complete-byte acceptance. PR #593 remains draft.
