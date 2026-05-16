# NVIDIA Physical AI Platform Architecture Coverage

**Last verified:** 2026-05-14

NVIDIA publishes different CUDA / PyTorch versions per host CPU architecture for its Physical AI stack. This shapes which Workbench tools can be validated on which Nebius hardware today.

## The split

| Component | x86_64 | aarch64 (Jetson Thor, DGX Spark) |
|---|---|---|
| Isaac Lab 2.3+ | torch 2.7.0 + CUDA 12.8 | torch 2.9.0 + CUDA 13.0 |
| GR00T N1.7 | CUDA 12.8 + Python 3.10 (dGPU) | CUDA 13.0 + Python 3.12 |
| Cosmos Predict2.5 / Transfer2.5 | CUDA 12.8.1 + Python 3.10 | CUDA 13.0 |

## Per-tool architecture dependence

| Workbench tool | Dependence | B300 readiness today |
|---|---|---|
| LeRobot | Independent of NVIDIA vendor stack | Tier 1, validated |
| SONIC | Isaac Lab -> Isaac Sim -> x86_64 CUDA 12.8 | Vendor-paced |
| GR00T | NVIDIA GR00T, x86_64 = CUDA 12.8 | Vendor-paced |
| Isaac Lab | NVIDIA Isaac Lab, x86_64 = CUDA 12.8 | Vendor-paced |
| Cosmos | NVIDIA Cosmos, x86_64 = CUDA 12.8.1 | Vendor-paced (Blackwell nightly exists) |
| FiftyOne | Not GPU-perf critical | Not architecture-gated |
| LanceDB | CPU-bound | Not architecture-gated |
| Genesis | Taichi sm_103 (separate axis) | Upstream-blocked |

## Tracking signals for vendor movement

- Isaac Lab GitHub releases: a torch 2.9.0 + cu130 install path for `Linux (x86_64)`.
- GR00T README: "CUDA / Python per platform" line listing dGPU on CUDA 13.
- Cosmos prerequisites page: x86-64 moving to CUDA 13, or the Blackwell nightly promoted to documented default.

## Customer messaging

- B300 is ready for LeRobot training on Nebius today.
- Other Physical AI workloads are gated on NVIDIA's x86_64 CUDA 13 alignment, not on Nebius infrastructure.
- aarch64 (Jetson Thor, DGX Spark) already has CUDA 13 from NVIDIA. Nebius does not currently offer aarch64 GPU compute.
- Genesis is independently blocked on upstream Taichi sm_103, unrelated to this split.

## Sources

- Isaac Lab: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
- GR00T N1.7: https://github.com/NVIDIA/Isaac-GR00T
- Cosmos prerequisites: https://docs.nvidia.com/cosmos/latest/prerequisites.html
- Cosmos Predict2.5 setup: https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/setup.md
- Cosmos Transfer2.5 setup: https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/setup.md
