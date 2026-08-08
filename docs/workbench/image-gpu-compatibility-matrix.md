# Image ↔ Nebius GPU compatibility matrix

Every Workbench container image against every Nebius GPU platform, and — separately — which of those cells has actually been run on real hardware.

**Last measured:** 2026-08-08

Two things are deliberately kept apart here, because conflating them is how "Blackwell ready" claims go wrong:

- **Can it execute there?** Decided by the architectures baked into the image's torch wheel plus any source-compiled CUDA extensions. Measurable without a GPU.
- **Has it been proven there?** Only a real capability run on that GPU answers this. An import check is not a proof — see [the flash-attn finding](#the-import-check-that-lied).

Machine-readable source of record: [`npa/docker/workbench/blackwell-dc-images.json`](../../npa/docker/workbench/blackwell-dc-images.json). Companion runbook: [Blackwell datacenter image compatibility](blackwell-datacenter-image-compatibility.md).

## Nebius GPU platforms

| GPU | Platform id | Family | Compute capability | SM |
| --- | --- | --- | --- | --- |
| L40S | `gpu-l40s-a` / `gpu-l40s-d` | Ada | 8.9 | `sm_89` |
| H100 | `gpu-h100-sxm` | Hopper | 9.0 | `sm_90` |
| H200 | `gpu-h200-sxm` | Hopper | 9.0 | `sm_90` |
| RTX PRO 6000 Blackwell | `gpu-rtx6000` | Blackwell workstation (GB20x) | 12.0 | `sm_120` |
| B200 | `gpu-b200-sxm` | Blackwell datacenter (GB100) | 10.0 | `sm_100` |
| B300 (Blackwell Ultra) | `gpu-b300-sxm` | Blackwell datacenter (GB300) | 10.3 | `sm_103` |

H100 and H200 are both `sm_90`, so they share a column below.

Also offered: `gpu-gb300` (Grace-Blackwell Ultra). Its GPU is the same `sm_103`, but the host is aarch64, and the x86_64 workbench images do not run there.

Two compatibility rules govern every cell:

- **SASS does not cross a CUDA major.** `sm_120` (major 12) and `sm_100`/`sm_103` (major 10) binaries are mutually incompatible, so a green run on RTX PRO 6000 says nothing about B200/B300.
- **Within a major, forward compatibility holds.** `sm_86` SASS runs on an `sm_89` device (L40S), and `sm_100` SASS runs on an `sm_103` device (B300). Not the reverse.

## Measured torch stack per image

`arch_list` is `torch._C._cuda_getArchFlags()` read out of the published image. It is fixed when the wheel is built — `TORCH_CUDA_ARCH_LIST` cannot change it — so it decides which GPUs the image can execute on. Reproduce any row with `npa/scripts/validate_blackwell_image.sh <image> --target b200`.

| Image | Tag measured | Torch / CUDA | Measured SASS set | Covers `sm_100`? |
| --- | --- | --- | --- | --- |
| `npa-base` | `cuda13-b300-sm80-sm90-sm100-sm103-sm120-20260803T032705Z` | 2.9.0+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` + `compute_120` PTX | yes |
| `npa-lerobot` | `…-0.5.1-…-20260803T034152Z` | 2.9.0+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` + `compute_120` PTX | yes |
| `npa-lerobot-policy` | `0.1.1` | 2.12.1+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` | yes |
| `npa-lancedb` | `…-0.30.3-…-20260803T031514Z` | 2.12.1+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` | yes |
| `npa-detection-training` | `bdd100k-golden-eval-smoke-20260614T210000Z` | 2.12.1+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` | yes |
| `npa-cosmos3` | `1.2.2-cu130-r2` (index `sha256:c65712832f6a…`) | 2.10.0+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` + `compute_120` PTX | yes |
| `npa-cosmos3-reason` | `…-3.0.1-…-20260803T034152Z` | 2.9.0+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` + `compute_120` PTX | yes |
| `npa-genesis` | `…-0.4.6-…-20260803T034152Z` | 2.9.0+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` + `compute_120` PTX | yes |
| `npa-envgen` / `npa-reference-policy` / `npa-lerobot-vlm-rl` / `npa-loop-eval` | `…-20260803T034152Z` | inherited 2.9.0+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` + `compute_120` PTX | yes |
| `npa-sonic` | `…-0.1.2-k8s-runtime-…-20260803T034152Z` | 2.9.0+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` + `compute_120` PTX | yes |
| `npa-cosmos` | `cu128-torch27-sm100-1.0.9-20260803T002017Z` | 2.7.0+cu128 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` + `compute_120` PTX | yes |

The old `npa-cosmos:1.0.9` cu126 image stopped at Hopper. Its additive cu128/torch-2.7 replacement now carries `sm_100`, and the custom kernels passed on B200. Predict2 v1.0.9 still has a separate software allowlist that rejects L40S, RTX PRO 6000, and B300 before dispatch, so wheel coverage alone does not make those cells supported. The exact final Genesis and Sim2Real tags compiled their runtime kernels and passed their real smokes on B200 and B300; the inherited Taichi blocker did not reproduce. SONIC remains separately blocked on the NVIDIA Isaac vendor stack. Not measured yet: `npa-workbench-cuda-base` (covered through its children), `npa-isaac-lab`, and `npa-groot`.

## Compatibility matrix

| Image | L40S `sm_89` | H100 / H200 `sm_90` | RTX PRO 6000 `sm_120` | B200 `sm_100` | B300 `sm_103` |
| --- | --- | --- | --- | --- | --- |
| `npa-base` | supported | **verified** [22] | **verified** [23] | **verified** [20] | **verified** [21] |
| `npa-workbench-cuda-base` | supported | supported | supported | supported | supported |
| `npa-lerobot` | supported | **verified** [41] | **verified** [42] | **verified** [39] | **verified** [40] |
| `npa-lerobot-policy` | supported | supported | supported | supported | supported |
| `npa-lancedb` | supported | **verified** [26] | **verified** [27] | **verified** [24] | **verified** [25] |
| `npa-detection-training` | supported | **verified** [29] | **verified** [30] | **verified** [28] | **verified** [31] |
| `npa-cosmos3` | supported | supported | **verified** [59] | supported | supported |
| `npa-cosmos3-reason` | supported | **verified** [38] | **verified** [43] | **verified** [36] | **verified** [37] |
| `npa-cosmos2-transfer` | supported | supported | supported | **verified** [9] | blocked (cu128 NVRTC cannot JIT `sm_103`) |
| `npa-cosmos` | blocked (Predict2 allowlist) | **verified** [33] | blocked (Predict2 allowlist) | **verified** [32] | blocked (Predict2 allowlist) |
| `npa-genesis` | supported | **verified** [46] | **verified** [14] | **verified** [44] | **verified** [45] |
| `npa-envgen` | supported | **verified** [49] | **verified** [15] | **verified** [47] | **verified** [48] |
| `npa-reference-policy` | supported | **verified** [52] | **verified** [16] | **verified** [50] | **verified** [51] |
| `npa-loop-eval` | supported | **verified** [58] | **verified** [18] | **verified** [56] | **verified** [57] |
| `npa-lerobot-vlm-rl` | supported | **verified** [55] | **verified** [17] | **verified** [53] | **verified** [54] |
| `npa-isaac-lab` | supported | supported (headless) | supported | blocked | blocked |
| `npa-sonic` | supported | supported (headless) | supported | blocked | blocked |
| `npa-sonic-mujoco` | supported | supported (headless) | supported | blocked | blocked |
| `npa-groot` | supported | supported | supported | blocked | blocked |
| `npa-cosmos-curate` | CPU | CPU | CPU | CPU | CPU |
| `npa-cosmos-evaluator` | CPU | CPU | CPU | CPU | CPU |
| `npa-fiftyone` | CPU | CPU | CPU | CPU | CPU |
| `npa-retargeting` | CPU | CPU | CPU | CPU | CPU |
| `npa-rerun-viewer` | CPU | CPU | CPU | CPU | CPU |
| `npa-lichtblick` | CPU | CPU | CPU | CPU | CPU |
| `npa-foxglove-embed` | CPU | CPU | CPU | CPU | CPU |
| `npa-sonic-export` | CPU | CPU | CPU | CPU | CPU |

**verified** — run on that GPU with a real capability smoke; see [Verified runs](#verified-runs).
**supported** — the toolchain can execute there, but no capability run on that GPU has been recorded.
**no SASS** — measured wheel does not carry the architecture; the image cannot run there until it is ported.
**blocked** — an upstream dependency does not support the architecture. Reason and tracking link are in the manifest's per-image fields or `known_gaps`.
**CPU** — CPU-only image. It runs on a host with any of these GPUs; only node-pool scheduling matters.

### Rendering is not portable across these columns

Isaac Lab and SONIC rasterized rendering needs RT cores. L40S and RTX PRO 6000 have them; H100, H200, B200, and B300 do not. The "supported (headless)" cells above mean state-based training only. `npa.workbench.sonic.routing` enforces this and rejects a render workload routed to a datacenter part.

### Blackwell datacenter hardware status

Managed-Kubernetes nodes were placed successfully for both B200 in us-central1 and B300 in uk-south1 on 2026-08-03. The temporary nodes enabled the first current-hardware validation runs below. Cells remain merely **supported** unless that exact image completed its real capability smoke; hardware availability alone does not flip a cell.

## Verified runs

| # | Date | Image | GPU | What ran | Result |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026-08-02 | `npa-base` `…-20260802T181419Z` | H100 80GB HBM3 (`sm_90`) | positive arch check, negative cross-major check, capability smoke (bf16 matmul, torch SDPA, flash-attn-4 CuTe forward vs SDPA) | `ALL_GPU_VALIDATION_PASSED`; flash-attn max abs error 0.00186 |
| 2 | 2026-08-02 | `npa-base` `…-20260802T181419Z` | RTX PRO 6000 Blackwell Server Edition (`sm_120`) | same three checks | `ALL_GPU_VALIDATION_PASSED`; flash-attn recorded as the known TMA gap |
| 3 | 2026-05-14 | `npa-base` cuda13-b300 | 8× B300 (`sm_103`), driver 580.126.09 | torch import, device capability `(10, 3)`, flash-attn-4 forward pass, NCCL init | PASS — [B300 validation matrix](../b300-validation-matrix.md) |
| 4 | 2026-05-14 | `npa-lerobot` cuda13-b300 | B300 (`sm_103`) | ACT on `lerobot/pusht_image`, batch 8, 100 steps | PASS, 71 s wall — [B300 validation matrix](../b300-validation-matrix.md) |
| 5 | 2026-08-03 | `npa-base` `…-20260802T181419Z` | NVIDIA B200 (`sm_100`), driver 580.159.04 | positive native-SASS check, negative `sm_120` cross-major check, bf16 matmul, torch SDPA, flash-attn-4 CuTe forward vs SDPA | capability `(10, 0)`, `sass_covered=True`, cross-major check failed as required, flash-attn max abs error 0.00206, `ALL_GPU_VALIDATION_PASSED` |
| 6 | 2026-08-03 | `npa-base` `…-20260802T181419Z` | NVIDIA B300 SXM6 AC (`sm_103`) | positive same-major SASS check, negative `sm_120` cross-major check, bf16 matmul, torch SDPA, flash-attn-4 CuTe forward vs SDPA | capability `(10, 3)`, `sm_100` SASS covered `sm_103`, cross-major check failed as required, flash-attn max abs error 0.00206, `ALL_GPU_VALIDATION_PASSED` |
| 7 | 2026-08-03 | `npa-detection-training:bdd100k-golden-eval-smoke-20260614T210000Z` | NVIDIA B200 (`sm_100`) | real Faster R-CNN forward, backward, and optimizer step on synthetic detector data | `DETECTOR_TRAIN_STEP_PASSED`; classifier, box-regression, objectness, and RPN losses produced |
| 8 | 2026-08-03 | `npa-base` `…-20260802T234708Z` | NVIDIA B300 SXM6 AC (`sm_103`) | repeated positive same-major SASS check, negative `sm_120` cross-major check, bf16 matmul, torch SDPA, and flash-attn-4 CuTe forward on the rebuilt/published image | capability `(10, 3)`, `sm_100` SASS covered `sm_103`, flash-attn max abs error 0.00206, `ALL_GPU_VALIDATION_PASSED` |
| 9 | 2026-08-03 | `npa-cosmos2-transfer:2.5.1-skypilot-ready-20260801T053000Z` | NVIDIA B200 (`sm_100`) | real depth-conditioned video-to-video transfer, including two 35-step generation passes and prompt/video guardrails | PASS; generated `robot_depth.mp4` (3,891,548 bytes) |
| 10 | 2026-08-03 | rebuilt `npa-lerobot` `…-20260803T000551Z` | NVIDIA B200 (`sm_100`) | base and child native-SASS checks, datacenter flash-attn-4 CuTe kernel, then official ACT PushT: 50 training steps, checkpoint, and one evaluation episode | PASS; 5/5 functional checks and flash-attn max abs error 0.00206 |
| 11 | 2026-08-03 | same rebuilt `npa-lerobot` | NVIDIA H100 (`sm_90`) | same ACT train→checkpoint→evaluation smoke plus native H100 SASS and flash-attn | PASS; 5/5 functional checks and flash-attn max abs error 0.00186 |
| 12 | 2026-08-03 | rebuilt `npa-cosmos3-reason` `…-20260803T000551Z` | NVIDIA B200 (`sm_100`) | base validators plus a real gated `nvidia/Cosmos-Reason2-8B` VLM reason pass over two frames | PASS; datacenter flash-attn kernel passed and the VLM emitted a completed judgment |
| 13 | 2026-08-03 | same rebuilt `npa-cosmos3-reason` | RTX PRO 6000 (`sm_120`) | native SASS/base controls plus the same real VLM reason pass | PASS; expected non-TMA flash-attn gap recorded, real VLM inference completed |
| 14 | 2026-08-03 | final rebased `npa-genesis` `…-20260803T034152Z` | RTX PRO 6000 (`sm_120`) | native SASS/base controls, raw environment generation, Genesis CUDA scene construction, runtime kernel compilation, and a physics step | PASS; `gs.cuda` on the physical GPU |
| 15 | 2026-08-03 | final rebased `npa-envgen` `…-20260803T034152Z` | RTX PRO 6000 (`sm_120`) | validators, real environment generation, and Genesis CUDA step | PASS |
| 16 | 2026-08-03 | final rebased `npa-reference-policy` `…-20260803T034152Z` | RTX PRO 6000 (`sm_120`) | validators, reference-policy variant assertion, real environment generation, and a Genesis CUDA scene/physics step | PASS; no policy rollout was claimed |
| 17 | 2026-08-03 | final rebased `npa-lerobot-vlm-rl` `…-20260803T034152Z` | RTX PRO 6000 (`sm_120`) | validators and a real VLM-signal adapter parameter update with checkpoint output | PASS; parameter delta 0.000250 |
| 18 | 2026-08-03 | final rebased `npa-loop-eval` `…-20260803T034152Z` | RTX PRO 6000 (`sm_120`) | validators and a scored two-environment Franka pick-place rollout | PASS; two CUDA environments |
| 19 | 2026-08-03 | `npa-cosmos:cu128-torch27-sm100-1.0.9-20260803T002017Z` | NVIDIA B200 (`sm_100`) | measured wheel arches, flash-attn 2.7.3 forward vs torch SDPA, and the exact Predict2 v1.0.9 `NeighborhoodAttention` module using the first shipped 2B Video2World NATTEN configuration | `ALL_COSMOS_CU128_KERNEL_VALIDATION_PASSED`; flash-attn max abs error 0.0009765625; NATTEN output `(1, 256, 4, 128)` |
| 20 | 2026-08-03 | corrected `npa-base` `…-20260803T032705Z` | NVIDIA B200 (`sm_100`) | committed positive/negative arch checks, bf16 matmul, SDPA, and flash-attn-4 CuTe vs SDPA | `ALL_GPU_VALIDATION_PASSED`; native `sm_100`; flash-attn max error 0.00206 |
| 21 | 2026-08-03 | same corrected `npa-base` | NVIDIA B300 SXM6 AC (`sm_103`) | same controls, using baked same-major `sm_100` → `sm_103` coverage logic | `ALL_GPU_VALIDATION_PASSED`; `sass_covered=True`; flash-attn max error 0.00206 |
| 22 | 2026-08-03 | same corrected `npa-base` | NVIDIA H100 80GB HBM3 (`sm_90`) | same controls with native `sm_90` SASS | `ALL_GPU_VALIDATION_PASSED`; flash-attn max error 0.00186 |
| 23 | 2026-08-03 | same corrected `npa-base` | RTX PRO 6000 (`sm_120`) | same controls with native `sm_120`; non-TMA exception allowed only on this GPU | `ALL_GPU_VALIDATION_PASSED`; bf16 and SDPA passed; known flash-attn TMA gap recorded |
| 24 | 2026-08-03 | corrected `npa-lancedb` `…-20260803T031514Z` | NVIDIA B200 (`sm_100`) | CLIP embedded three images, checked normalized/distinct 512-D vectors, inserted a Lance table, and required top-1 self-search | `LANCEDB_CLIP_EXTENSIVE_VALIDATION_PASSED`; 3 rows |
| 25 | 2026-08-03 | same corrected `npa-lancedb` | NVIDIA B300 SXM6 AC (`sm_103`) | same real CLIP → Lance → search path | `LANCEDB_CLIP_EXTENSIVE_VALIDATION_PASSED`; 3 rows |
| 26 | 2026-08-03 | same corrected `npa-lancedb` | NVIDIA H100 80GB HBM3 (`sm_90`) | same real CLIP → Lance → search path | `LANCEDB_CLIP_EXTENSIVE_VALIDATION_PASSED`; 3 rows |
| 27 | 2026-08-03 | same corrected `npa-lancedb` | RTX PRO 6000 (`sm_120`) | same real CLIP → Lance → search path | `LANCEDB_CLIP_EXTENSIVE_VALIDATION_PASSED`; 3 rows |
| 28 | 2026-08-03 | `npa-detection-training:bdd100k-golden-eval-smoke-20260614T210000Z` | NVIDIA B200 (`sm_100`) | three Faster R-CNN forward/backward/SGD updates followed by inference | `DETECTOR_EXTENSIVE_VALIDATION_PASSED`; three finite losses; 89 detections |
| 29 | 2026-08-03 | same detector image | NVIDIA H100 80GB HBM3 (`sm_90`) | same three-step train + inference path | `DETECTOR_EXTENSIVE_VALIDATION_PASSED`; 89 detections |
| 30 | 2026-08-03 | same detector image | RTX PRO 6000 (`sm_120`) | same three-step train + inference path | `DETECTOR_EXTENSIVE_VALIDATION_PASSED`; 89 detections |
| 31 | 2026-08-03 | same detector image | NVIDIA B300 SXM6 AC (`sm_103`) | same three-step train + inference path | `DETECTOR_EXTENSIVE_VALIDATION_PASSED`; 89 detections |
| 32 | 2026-08-03 | `npa-cosmos:cu128-torch27-sm100-1.0.9-20260803T002017Z` | NVIDIA B200 (`sm_100`) | flash-attn matrix over bf16/fp16, causal/noncausal, three sequence/head shapes; real NATTEN forward for five shipped configurations | `COSMOS_KERNEL_EXTENSIVE_VALIDATION_PASSED` |
| 33 | 2026-08-03 | same Predict2 image | NVIDIA H100 80GB HBM3 (`sm_90`) | same flash-attn matrix and five real NATTEN configurations | `COSMOS_KERNEL_EXTENSIVE_VALIDATION_PASSED` |
| 34 | 2026-08-03 | rebuilt `npa-cosmos3-reason` `…-20260803T000551Z` | NVIDIA H100 80GB HBM3 (`sm_90`) | downloaded and loaded the gated `nvidia/Cosmos-Reason2-8B`, then ran a reason pass over two frames | functional execution PASS; completed judgment emitted (score 0.0, success false) |
| 35 | 2026-08-03 | rebuilt `npa-lerobot` `…-20260803T000551Z` | RTX PRO 6000 (`sm_120`) | official ACT PushT: 50 train steps, checkpoint, and one evaluation episode | `LEROBOT_VALIDATION_PASSED`; 5/5 checks |
| 36 | 2026-08-03 | corrected `npa-cosmos3-reason` `…-20260803T034152Z` | NVIDIA B200 (`sm_100`) | final-image validators, flash-attn-4 vs SDPA, four gated Reason2-8B checkpoint shards, and a two-frame VLM judgment | functional execution PASS; score 0.0, success true |
| 37 | 2026-08-03 | same corrected `npa-cosmos3-reason` | NVIDIA B300 SXM6 AC (`sm_103`) | same final-image controls and real gated Reason2-8B inference | functional execution PASS; score 0.0, success false |
| 38 | 2026-08-03 | same corrected `npa-cosmos3-reason` | NVIDIA H100 80GB HBM3 (`sm_90`) | same final-image controls and real gated Reason2-8B inference | functional execution PASS; score 0.0, success false |
| 39 | 2026-08-03 | final rebased `npa-lerobot` `…-20260803T034152Z` | NVIDIA B200 (`sm_100`) | both wheel-arch validators, flash-attn-4 vs SDPA, then official ACT PushT: 50 train steps, checkpoint, and one evaluation episode | `LEROBOT_VALIDATION_PASSED`; 5/5 checks |
| 40 | 2026-08-03 | same final `npa-lerobot` | NVIDIA B300 SXM6 AC (`sm_103`) | same final-image ACT train → checkpoint → evaluation path and same-major SASS control | `LEROBOT_VALIDATION_PASSED`; 5/5 checks |
| 41 | 2026-08-03 | same final `npa-lerobot` | NVIDIA H100 80GB HBM3 (`sm_90`) | same final-image ACT train → checkpoint → evaluation path and native H100 SASS | `LEROBOT_VALIDATION_PASSED`; 5/5 checks |
| 42 | 2026-08-03 | same final `npa-lerobot` | RTX PRO 6000 (`sm_120`) | same final-image ACT train → checkpoint → evaluation path | `LEROBOT_VALIDATION_PASSED`; 5/5 checks; known non-TMA exception only |
| 43 | 2026-08-03 | corrected `npa-cosmos3-reason` `…-20260803T034152Z` | RTX PRO 6000 (`sm_120`) | final-image validators, four gated Reason2-8B checkpoint shards, and a real two-frame VLM judgment | functional execution PASS; score 0.0, success false; known non-TMA exception only |
| 44 | 2026-08-03 | final rebased `npa-genesis` `…-20260803T034152Z` | NVIDIA B200 (`sm_100`) | validators, raw environment generation, `gs.cuda`, plane/Franka/box scene construction, runtime kernel compilation, and a physics step | PASS; inherited Taichi blocker did not reproduce |
| 45 | 2026-08-03 | same final `npa-genesis` | NVIDIA B300 SXM6 AC (`sm_103`) | same real scene, runtime kernel compilation, and physics path | `DATACENTER_CHILD_VALIDATION_PASSED`; same-major `sm_100` SASS covered `sm_103` |
| 46 | 2026-08-03 | same final `npa-genesis` | NVIDIA H100 80GB HBM3 (`sm_90`) | same real scene, runtime kernel compilation, and physics path | `DATACENTER_CHILD_VALIDATION_PASSED` |
| 47 | 2026-08-03 | final rebased `npa-envgen` `…-20260803T034152Z` | NVIDIA B200 (`sm_100`) | validators, real environment generation, runtime kernel compilation, and Genesis CUDA step | `DATACENTER_CHILD_VALIDATION_PASSED` |
| 48 | 2026-08-03 | same final `npa-envgen` | NVIDIA B300 SXM6 AC (`sm_103`) | same real environment-generation and Genesis CUDA path | `DATACENTER_CHILD_VALIDATION_PASSED` |
| 49 | 2026-08-03 | same final `npa-envgen` | NVIDIA H100 80GB HBM3 (`sm_90`) | same real environment-generation and Genesis CUDA path | `DATACENTER_CHILD_VALIDATION_PASSED` |
| 50 | 2026-08-03 | final rebased `npa-reference-policy` `…-20260803T034152Z` | NVIDIA B200 (`sm_100`) | validators, variant assertion, environment generation, and Genesis CUDA scene/physics step | `DATACENTER_CHILD_VALIDATION_PASSED`; no policy rollout claimed |
| 51 | 2026-08-03 | same final `npa-reference-policy` | NVIDIA B300 SXM6 AC (`sm_103`) | same variant/environment/physics path | `DATACENTER_CHILD_VALIDATION_PASSED`; no policy rollout claimed |
| 52 | 2026-08-03 | same final `npa-reference-policy` | NVIDIA H100 80GB HBM3 (`sm_90`) | same variant/environment/physics path | `DATACENTER_CHILD_VALIDATION_PASSED`; no policy rollout claimed |
| 53 | 2026-08-03 | final rebased `npa-lerobot-vlm-rl` `…-20260803T034152Z` | NVIDIA B200 (`sm_100`) | validators and real VLM-signal adapter parameter update with checkpoint output | PASS; parameter delta L2 0.000250; no Genesis rollout claimed |
| 54 | 2026-08-03 | same final `npa-lerobot-vlm-rl` | NVIDIA B300 SXM6 AC (`sm_103`) | same real parameter-update/checkpoint path | PASS; parameter delta L2 0.000250 |
| 55 | 2026-08-03 | same final `npa-lerobot-vlm-rl` | NVIDIA H100 80GB HBM3 (`sm_90`) | same real parameter-update/checkpoint path | PASS; parameter delta L2 0.000250 |
| 56 | 2026-08-03 | final rebased `npa-loop-eval` `…-20260803T034152Z` | NVIDIA B200 (`sm_100`) | validators and scored two-environment Franka pick-place CUDA rollout | `DATACENTER_CHILD_VALIDATION_PASSED` |
| 57 | 2026-08-03 | same final `npa-loop-eval` | NVIDIA B300 SXM6 AC (`sm_103`) | same scored two-environment CUDA rollout | `DATACENTER_CHILD_VALIDATION_PASSED` |
| 58 | 2026-08-03 | same final `npa-loop-eval` | NVIDIA H100 80GB HBM3 (`sm_90`) | same scored two-environment CUDA rollout | `DATACENTER_CHILD_VALIDATION_PASSED` |
| 59 | 2026-08-08 | `npa-cosmos3:1.2.2-cu130-r2` (index `sha256:c65712832f6a…`, amd64 `sha256:19dc6be7d2f9…`) | NVIDIA RTX PRO 6000 Blackwell Server Edition (`sm_120`) | exact release bytes, non-root UID 1000, native `sm_120` SASS, gated Cosmos3-Nano text-to-image generation with Xet enabled (`huggingface_hub 0.36.2`, `hf-xet 1.3.2`, `HF_HUB_DISABLE_XET` unset) | PASS; 960×960 JPEG, 260,808 bytes, SHA-256 `e4f017a75266b937b7479a0a5090bf644ef442da6972cae59988d1d5b5daa861`; guardrail discovery still failed open as tracked in [#270](https://github.com/nebius/nebius-physical-ai/issues/270) |

## Measured failures and negative controls

| Date | Image | GPU | What ran | Outcome |
| --- | --- | --- | --- | --- |
| 2026-08-03 | `npa-cosmos2-transfer:2.5.1-skypilot-ready-20260801T053000Z` | NVIDIA B300 SXM6 AC (`sm_103`) | real depth-conditioned Video2Video entrypoint through gated downloads and `Control2WorldInference` model construction | FAIL before inference: CUDA 12.8 NVRTC rejected torch's runtime-generated `erfinv` kernel for capability 10.3; B300 cell downgraded to blocked |
| 2026-08-03 | `npa-cosmos:cu128-torch27-sm100-1.0.9-20260803T002017Z` | NVIDIA B300 SXM6 AC (`sm_103`) | six-case flash-attn matrix, then real Predict2 NATTEN forward | flash-attn passed; NATTEN rejected capability 103 through upstream's `[90, 100]` allowlist as expected |
| 2026-08-03 | same Predict2 image | RTX PRO 6000 (`sm_120`) | six-case flash-attn matrix, then real Predict2 NATTEN forward | flash-attn passed; NATTEN rejected capability 120 through the same upstream allowlist as expected |

Runs 1, 2, 5, 6, and 8 used [`npa/scripts/blackwell-gpu-validation-job.yaml`](../../npa/scripts/blackwell-gpu-validation-job.yaml). Each does three checks, so a pass means something: the target architecture must pass *with native SASS* (including same-major `sm_100` coverage for `sm_103`), a different CUDA major must **fail** (proving the checker cannot hand out a false "Blackwell ready" on the wrong GPU family), and then the capability smoke runs real kernels.

The job runs non-root with dropped capabilities and a read-only root filesystem. flash-attn-4's CuTe kernels JIT-compile at runtime, so `HOME` and every torch/CUDA cache point at a scratch `emptyDir`; the H100 run above confirms the kernel still compiles and executes under those constraints.

The original READY-set images were also tested before rebuild. The historical `npa-lancedb:0.30.3` CLIP path fails on a current transformers return type and is superseded by the corrected image in runs 24–27. The superseded `npa-lerobot:0.5.1` failed a torchcodec/FFmpeg mismatch, and the superseded Cosmos3 Reason image lacked the functional smoke module. The rebuilt SONIC image passes its native-SASS controls and reaches real environment construction after fixing two undeclared upstream dependencies, but both cold and warm fine-tune attempts fail inside Isaac's runtime-fetched URDF extension while opening a temporary pelvis USD layer, before a checkpoint. Those failures are recorded in `validation_evidence`; they were not converted into verified cells. The Cosmos kernel runs are not generated-video claims: checkpoint-backed Predict2 Video2World was attempted again during the extensive run and remains unverified because the available Hugging Face identity received HTTP 403 for NVIDIA's gated checkpoint.

## The import check that lied

`npa-base`'s golden eval used to be `python -c "import torch; assert torch.cuda.is_available(); import flash_attn"`. It passed on every Blackwell part for months. The first time anyone executed the kernel — run 2 above — it failed on `sm_120`.

flash-attn-4's CuTe forward kernel partitions its epilogue with a TMA (Tensor Memory Accelerator) copy atom. TMA is a datacenter feature: `sm_90`, `sm_100`, and `sm_103` have it; RTX PRO 6000 does not. On `sm_120` the atom is `None` and the kernel raises `AttributeError: 'NoneType' object has no attribute '_trait'`.

What makes that conclusion safe rather than a guess:

- All four configurations tried (bf16/fp16 × head_dim 64/128 × seqlen 64/256) fail at the identical line — architecture-wide, not a config quirk.
- torch SDPA and bf16 matmul both pass on the same `sm_120` device, ruling out the GPU and the wheel.
- The same image passes the kernel on H100 (run 1) and the previously published image passed it on B300 (run 3) — both TMA-capable.
- The previously published `npa-base:cuda13-b300-sm80-sm90-sm120-latest` fails identically on `sm_120`, so this is pre-existing rather than a regression.

Callers on `sm_120` should use torch SDPA. The eval now runs [`gpu_capability_smoke.py`](../../npa/docker/workbench/base/cuda13-b300/scripts/gpu_capability_smoke.py), which executes the kernel; `--allow-no-tma` records the `sm_120` gap without excusing a TMA failure on a datacenter part, where it would be real.

## Reproducing a cell

```bash
# Wheel arch set only - no GPU needed, catches a Hopper-capped wheel for free
npa/scripts/validate_blackwell_image.sh "$NPA_REGISTRY/npa-lerobot:0.5.1" --target b200

# Full check on a real node
npa/scripts/validate_blackwell_image.sh "$NPA_REGISTRY/npa-base:<tag>" --target b300 --gpu

# On an already-deployed Kubernetes GPU pool
# (substitute NPA_IMAGE / NPA_GPU_INSTANCE / NPA_TARGET_* and apply)
kubectl apply -f npa/scripts/blackwell-gpu-validation-job.yaml
```
