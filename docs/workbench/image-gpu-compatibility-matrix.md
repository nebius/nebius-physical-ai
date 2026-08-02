# Image ↔ Nebius GPU compatibility matrix

Every Workbench container image against every Nebius GPU platform, and — separately — which of those cells has actually been run on real hardware.

**Last measured:** 2026-08-02

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
| `npa-base` | `cuda13-b300-sm80-sm90-sm100-sm103-sm120-20260802T181419Z` | 2.9.0+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` + `compute_120` PTX | yes |
| `npa-lerobot` | `0.5.1` | 2.12.1+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` | yes |
| `npa-lerobot-policy` | `0.1.1` | 2.12.1+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` | yes |
| `npa-lancedb` | `0.30.3` | 2.12.1+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` | yes |
| `npa-detection-training` | `bdd100k-golden-eval-smoke-20260614T210000Z` | 2.12.1+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` | yes |
| `npa-cosmos3-reason` | `3.0.1-genuine-sm120` | 2.9.0+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` + `compute_120` PTX | yes |
| `npa-loop-eval` | `0.1.3-genuine-sm120` | 2.9.0+cu130 | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` + `compute_120` PTX | yes |
| `npa-genesis` | `0.4.6` | 2.6.0+cu124 | `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90` | **no** |
| `npa-cosmos` | `1.0.9` | 2.6.0+cu126 | `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90` | **no** |

The two `no` rows are the whole reason `npa-cosmos` is a port and the default `npa-genesis` needs moving off cu124: cu124 and cu126 stop at Hopper. Not measured yet: `npa-workbench-cuda-base` (covered through its children), `npa-cosmos2-transfer`, `npa-isaac-lab`, `npa-sonic`, `npa-groot`, and the remaining sim2real chain.

## Compatibility matrix

| Image | L40S `sm_89` | H100 / H200 `sm_90` | RTX PRO 6000 `sm_120` | B200 `sm_100` | B300 `sm_103` |
| --- | --- | --- | --- | --- | --- |
| `npa-base` | supported | **verified** [1] | **verified** [2] | supported | **verified** [3] |
| `npa-workbench-cuda-base` | supported | supported | supported | supported | supported |
| `npa-lerobot` | supported | supported | supported | supported | **verified** [4] |
| `npa-lerobot-policy` | supported | supported | supported | supported | supported |
| `npa-lancedb` | supported | supported | supported | supported | supported |
| `npa-detection-training` | supported | supported | supported | supported | supported |
| `npa-cosmos3` | supported | supported | supported | supported | supported |
| `npa-cosmos3-reason` | supported | supported | supported | supported | supported |
| `npa-cosmos2-transfer` | supported | supported | supported | supported | supported |
| `npa-cosmos` | supported | supported | **no SASS** | **no SASS** | **no SASS** |
| `npa-genesis` | supported | supported | **no SASS** | **no SASS** | **no SASS** |
| `npa-envgen` | supported | supported | blocked | blocked | blocked |
| `npa-reference-policy` | supported | supported | blocked | blocked | blocked |
| `npa-loop-eval` | supported | supported | blocked | blocked | blocked |
| `npa-lerobot-vlm-rl` | supported | supported | blocked | blocked | blocked |
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
**blocked** — an upstream dependency does not support the architecture. Reason and tracking link per image in the manifest's `blocked_reason` / `upstream_tracking`.
**CPU** — CPU-only image. It runs on a host with any of these GPUs; only node-pool scheduling matters.

### Rendering is not portable across these columns

Isaac Lab and SONIC rasterized rendering needs RT cores. L40S and RTX PRO 6000 have them; H100, H200, B200, and B300 do not. The "supported (headless)" cells above mean state-based training only. `npa.workbench.sonic.routing` enforces this and rejects a render workload routed to a datacenter part.

### Why the Blackwell datacenter column is mostly not "verified"

B200 and B300 instances could not be placed in this tenant during validation: serverless jobs submit but their instance cycles `STARTING → STOPPED`, in both us-central1 (B200) and uk-south1 (B300), and a VM deploy failed earlier on the tenant's public IPv4 quota. That is capacity, not the images. The manifest records the blocker and the command to finish each cell.

## Verified runs

| # | Date | Image | GPU | What ran | Result |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026-08-02 | `npa-base` `…-20260802T181419Z` | H100 80GB HBM3 (`sm_90`) | positive arch check, negative cross-major check, capability smoke (bf16 matmul, torch SDPA, flash-attn-4 CuTe forward vs SDPA) | `ALL_GPU_VALIDATION_PASSED`; flash-attn max abs error 0.00186 |
| 2 | 2026-08-02 | `npa-base` `…-20260802T181419Z` | RTX PRO 6000 Blackwell Server Edition (`sm_120`) | same three checks | `ALL_GPU_VALIDATION_PASSED`; flash-attn recorded as the known TMA gap |
| 3 | 2026-05-14 | `npa-base` cuda13-b300 | 8× B300 (`sm_103`), driver 580.126.09 | torch import, device capability `(10, 3)`, flash-attn-4 forward pass, NCCL init | PASS — [B300 validation matrix](../b300-validation-matrix.md) |
| 4 | 2026-05-14 | `npa-lerobot` cuda13-b300 | B300 (`sm_103`) | ACT on `lerobot/pusht_image`, batch 8, 100 steps | PASS, 71 s wall — [B300 validation matrix](../b300-validation-matrix.md) |

Runs 1 and 2 used [`npa/scripts/blackwell-gpu-validation-job.yaml`](../../npa/scripts/blackwell-gpu-validation-job.yaml) against already-deployed Kubernetes GPU pools. Each does three checks, so a pass means something: the target architecture must pass *with native SASS*, a different CUDA major must **fail** (proving the checker cannot hand out a false "Blackwell ready" on the wrong GPU family), and then the capability smoke runs real kernels.

The job runs non-root with dropped capabilities and a read-only root filesystem. flash-attn-4's CuTe kernels JIT-compile at runtime, so `HOME` and every torch/CUDA cache point at a scratch `emptyDir`; the H100 run above confirms the kernel still compiles and executes under those constraints.

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
