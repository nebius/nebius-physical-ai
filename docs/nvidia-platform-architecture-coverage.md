# NVIDIA Physical AI Platform Architecture Coverage

**Last verified:** 2026-08-02

NVIDIA publishes different CUDA / PyTorch versions per host CPU architecture for its Physical AI stack. This shapes which Workbench tools can be validated on which Nebius hardware today.

There is a second, independent axis: the GPU architecture the image was compiled for. Per-image verdicts and the build/validate runbook live in [Blackwell datacenter image compatibility](workbench/blackwell-datacenter-image-compatibility.md); the machine-readable plan of record is `npa/docker/workbench/blackwell-dc-images.json`.

## "Blackwell" is a family, not a CUDA target

| GPU | Family | Compute capability | SM target | Nebius platform |
|---|---|---|---|---|
| RTX PRO 6000 Blackwell | Blackwell workstation (GB20x) | 12.0 | `sm_120` | `gpu-rtx6000` |
| B200 | Blackwell datacenter (GB100) | 10.0 | `sm_100` | `gpu-b200-sxm` (us-central1) |
| B300 (Blackwell Ultra) | Blackwell datacenter (GB300) | 10.3 | `sm_103` | `gpu-b300-sxm` (uk-south1) |
| H100 / H200 | Hopper | 9.0 | `sm_90` | `gpu-h100-sxm` / `gpu-h200-sxm` |
| L40S | Ada | 8.9 | `sm_89` | `gpu-l40s-a` / `gpu-l40s-d` |

SASS is per-arch and does not cross a major version, so `sm_120` (major 12) and `sm_100`/`sm_103` (major 10) binaries are mutually incompatible: **a green smoke on RTX PRO 6000 does not prove B200 or B300.** Within a major, minor-version forward compatibility does hold, so `sm_100` SASS runs on a `sm_103` device. Build for `sm_100` first and confirm on B300.

Two independent knobs decide the architecture an image can execute on:

- **Prebuilt torch wheels** ship a fixed fat-binary arch set. `TORCH_CUDA_ARCH_LIST` cannot change it; only the wheel index does. cu128 and cu130 include `sm_100` and `sm_120`; cu124 and cu126 stop at `sm_90` and are therefore not Blackwell-capable. Check with `torch.cuda.get_arch_list()`.
- **Source-compiled extensions** (flash-attn from source, Taichi/Genesis, natten, custom ops) obey `TORCH_CUDA_ARCH_LIST` at build time. Omit an architecture and you get `no kernel image is available for execution on the device` at runtime, not a build error.

Two more constraints follow from the hardware rather than the toolchain:

- **No RT cores on datacenter Blackwell**, same as H100/H200. Isaac Lab and SONIC rasterized rendering must stay on L40S or RTX PRO 6000; only headless / state-based training routes to B200/B300. `npa.workbench.sonic.routing` enforces this.
- **Driver and toolkit floor**: R570+ driver, CUDA 12.8 or 13.0.

## The split

| Component | x86_64 | aarch64 (Jetson Thor, DGX Spark) |
|---|---|---|
| Isaac Lab 2.3+ | torch 2.7.0 + CUDA 12.8 | torch 2.9.0 + CUDA 13.0 |
| GR00T N1.7 | CUDA 12.8 + Python 3.10 (dGPU) | CUDA 13.0 + Python 3.12 |
| Cosmos Predict2.5 / Transfer2.5 | CUDA 12.8.1 + Python 3.10 | CUDA 13.0 |

## Per-tool architecture dependence

| Workbench tool | Dependence | Datacenter Blackwell readiness (B200 `sm_100` / B300 `sm_103`) |
|---|---|---|
| LeRobot | Independent of NVIDIA vendor stack | Exact final image validated with ACT train/checkpoint/eval on B200 and B300 |
| SONIC | Isaac Lab -> Isaac Sim -> x86_64 CUDA 12.8 | Vendor-paced; render can never target B200/B300 (no RT cores) |
| GR00T | NVIDIA GR00T, x86_64 = CUDA 12.8 | Vendor-paced; a headless compute-only finetune variant is plausible since cu128 carries `sm_100` |
| Isaac Lab | NVIDIA Isaac Lab, x86_64 = CUDA 12.8 | Vendor-paced; render can never target B200/B300 |
| Cosmos Predict2 | NVIDIA Cosmos, v1.2.0 cu128/torch-2.7 wheels carry `sm_100` | B200 custom kernels verified; Predict2 v1.0.9's capability allowlist still blocks B300 `sm_103` |
| Cosmos Transfer2.5 | Already on the cu128 track | Real Video2Video verified on B200; B300 blocked because CUDA 12.8 NVRTC rejects runtime JIT for `sm_103` |
| FiftyOne | Not GPU-perf critical | Not architecture-gated |
| LanceDB | cu130 base, no custom CUDA build | CLIP → Lance table → self-search verified on B200 and B300 |
| Genesis | Taichi runtime kernel compilation | Real scene construction, kernel compilation, and physics step verified on B200 and B300 |

## Tracking signals for vendor movement

- Isaac Lab GitHub releases: a torch 2.9.0 + cu130 install path for `Linux (x86_64)`.
- GR00T README: "CUDA / Python per platform" line listing dGPU on CUDA 13.
- Cosmos prerequisites page: x86-64 moving to CUDA 13, or the Blackwell nightly promoted to documented default.

## Customer messaging

- B200 and B300 are ready for LeRobot training and the validated headless Genesis/Sim2Real paths on Nebius today.
- NVIDIA Isaac workloads and specific Cosmos paths remain gated on their own vendor/runtime constraints, not on Nebius placement.
- aarch64 (Jetson Thor, DGX Spark) already has CUDA 13 from NVIDIA. Nebius does not currently offer aarch64 GPU compute.
- Do not generalize the Genesis result to Isaac or Cosmos custom kernels; every image keeps its own measured verdict.
- Rendering workloads belong on L40S or RTX PRO 6000. Datacenter Blackwell has no RT cores, so this is a hardware fact rather than a software gap that will be fixed.

## Sources

- Isaac Lab: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
- GR00T N1.7: https://github.com/NVIDIA/Isaac-GR00T
- Cosmos prerequisites: https://docs.nvidia.com/cosmos/latest/prerequisites.html
- Cosmos Predict2.5 setup: https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/setup.md
- Cosmos Transfer2.5 setup: https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/setup.md
