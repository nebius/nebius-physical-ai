# B300 Validation Matrix

B300 is validated for the `cuda13-b300` base image and the LeRobot ACT smoke-training workload listed below. SONIC is not yet validated on B300 because its current Isaac Sim / Isaac Lab dependency path does not preserve the CUDA 13 / PyTorch 2.9 x86_64 base contract. Cosmos, GR00T, and Isaac Lab remain vendor-paced, while Genesis remains upstream-blocked on Taichi Blackwell support.

## NVIDIA Physical AI x86_64 vs aarch64 architecture split

As of May 2026, NVIDIA publishes different CUDA / PyTorch versions per host architecture. B300 in the dGPU form factor is x86_64, which is on the older CUDA 12.8 track for the major vendor frameworks.

| Component | x86_64 | aarch64 (Jetson Thor, DGX Spark) |
|---|---|---|
| Isaac Lab 2.3+ | torch 2.7.0 + CUDA 12.8 | torch 2.9.0 + CUDA 13.0 |
| GR00T N1.7 | CUDA 12.8 + Python 3.10 (dGPU) | CUDA 13.0 + Python 3.12 |
| Cosmos Predict2.5 / Transfer2.5 | CUDA 12.8.1 + Python 3.10 | CUDA 13.0 |

Sources verified 2026-05-14: see `docs/nvidia-platform-architecture-coverage.md`.

Implications:

- Tools depending on these vendor stacks inherit x86_64 CUDA 12.8 on B300.
- SONIC depends on Isaac Lab 2.3.2 and is transitively vendor-paced.
- LeRobot is independent of these vendor stacks; it is the Tier 1 deliverable for B300 today.
- A Cosmos Blackwell nightly Dockerfile exists for x86_64 but is opt-in, not the documented default.

Unblocking signal: NVIDIA publishing CUDA 13 install paths for `Linux (x86_64)` in any of the three vendor sources.

## Validated Workloads

| Tool | Image | Validation | B300 result | Baseline comparison | Date |
|---|---|---|---|---|---|
| Base CUDA 13 B300 | `cr.eu-north1.nebius.cloud/YOUR_REGISTRY_ID/npa-base:cuda13-b300-20260514T214550Z` | PyTorch 2.9.0+cu130 import, device capability `(10, 3)`, flash-attn-4 forward pass, NCCL init | PASS on 8x B300, driver 580.126.09 | No H200/H100 comparison; functional base smoke | 2026-05-14 |
| LeRobot | `cr.eu-north1.nebius.cloud/YOUR_REGISTRY_ID/npa-lerobot:cuda13-b300-20260514T214550Z` | ACT, `lerobot/pusht_image`, batch size 8, 100 steps | PASS; 71 s wall time, 39 s training loop, 2.51 step/s end-to-end progress line, 27.19 step/s warm step-100 sample | No exact H200 wall-clock baseline found. Reference H200 profiler run used LeRobot v0.5.1 on `lerobot/pusht` at 34.56 step/s, so it is not a like-for-like speedup claim. | 2026-05-14 |

## Tier 1 Not Yet Validated

| Tool | Intended image | Status | Blocking evidence | Next action |
|---|---|---|---|---|
| SONIC | `workbench-sonic:cuda13-b300-20260514T214550Z` | Vendor-paced (Isaac Lab 2.3.2 dependency); no image pushed | Isaac Lab 2.3.2 imports `isaacsim`. Isaac Sim 5.0 conflicts with Isaac Lab 2.3.2 on Pillow. Isaac Sim 5.1 resolves farther but attempts to install PyTorch 2.7 / CUDA 12 packages on x86_64, which violates the B300 base contract. | ETA tied to NVIDIA Isaac Sim x86_64 CUDA 13 alignment; rerun SONIC validation after that package matrix exists. |

## Vendor-Paced Workloads

| Tool | Current status | Vendor timeline / source signal | Dependency chain | Confidence |
|---|---|---|---|---|
| Cosmos | Promising Blackwell support, not validated here on Nebius B300 | CUDA 13 supports B300/GB300. TensorRT-LLM 1.1 added B300/GB300 support and uses PyTorch 2.9. Cosmos Reason2 lists GB200/DGX Spark/Thor CUDA 13 paths; Predict2.5 and Transfer2.5 note Blackwell + ARM inference support. | Cosmos WFMs -> PyTorch/vLLM/TRT-LLM -> CUDA 13/Blackwell kernels | Medium-high |
| GR00T | Vendor-paced for B300 x86_64 | GR00T N1.7 is Early Access; repo guidance lists CUDA 12.8 for dGPU x86_64 and CUDA 13 for Thor/Spark. The repo warns sm_103 users that `torch.compile` can fail with the pinned Triton/PyTorch stack and recommends eager mode or TensorRT inference. | GR00T -> PyTorch/Triton/flash-attn/TensorRT -> platform-specific CUDA stack | High |
| Isaac Lab | Vendor-paced for B300 x86_64 | Isaac Sim 5.0 / Isaac Lab 2.2 reached GA in 2025. Current Isaac Lab pip docs use Isaac Sim 5.1 with PyTorch 2.7 + CUDA 12.8 on x86_64, and PyTorch 2.9 + CUDA 13 on aarch64. | Isaac Lab -> Isaac Sim/Omniverse/PhysX -> PyTorch/CUDA/vendor packages | High |

## Upstream-Blocked Workloads

| Tool | Blocking dependency | Upstream tracking | Engagement status | ETA estimate |
|---|---|---|---|---|
| Genesis | Taichi CUDA codegen for Blackwell / sm_103 | Taichi PR [#8735](https://github.com/taichi-dev/taichi/pull/8735), issue [#8730](https://github.com/taichi-dev/taichi/issues/8730) | Draft comment prepared in run artifacts; not filed | No defensible ETA from public sources |
| Genesis Warp bypass | No public `gs.warp` / NVIDIA Warp backend found | Genesis repo [README](https://github.com/Genesis-Embodied-AI/genesis-world) credits Taichi as the compute backend; repo search found no Warp backend path | Draft maintainer question prepared in run artifacts; not filed | No defensible ETA from public sources |

## Out Of Scope

| Tool | Rationale |
|---|---|
| FiftyOne | CPU indexing and light embedding are not B300 performance-critical for this validation pass. |
| LanceDB | CPU database workload; not part of CUDA 13 / sm_103 hardening. |

## Reproduction

Pull and smoke the base image on a B300 host with NVIDIA driver 580 or newer:

```bash
docker pull cr.eu-north1.nebius.cloud/YOUR_REGISTRY_ID/npa-base:cuda13-b300-20260514T214550Z
docker run --gpus all --rm \
  cr.eu-north1.nebius.cloud/YOUR_REGISTRY_ID/npa-base:cuda13-b300-20260514T214550Z \
  python -c "import torch; print(torch.cuda.get_device_capability(0)); import flash_attn; print(flash_attn.__version__)"
```

Run the validated LeRobot workload:

```bash
docker pull cr.eu-north1.nebius.cloud/YOUR_REGISTRY_ID/npa-lerobot:cuda13-b300-20260514T214550Z
docker run --gpus all --rm \
  -e WANDB_MODE=disabled \
  -v /tmp/lerobot-b300:/output \
  cr.eu-north1.nebius.cloud/YOUR_REGISTRY_ID/npa-lerobot:cuda13-b300-20260514T214550Z \
  lerobot-train \
    --policy.type=act \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/pusht_image \
    --batch_size=8 \
    --steps=100 \
    --output_dir=/output/act-pusht-100 \
    --save_freq=100 \
    --eval_freq=0 \
    --log_freq=10 \
    --num_workers=4 \
    --wandb.enable=false
```

## Sources

- CUDA 13 Blackwell support: <https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/>
- TensorRT-LLM release notes: <https://nvidia.github.io/TensorRT-LLM/release-notes.html>
- Cosmos prerequisites: <https://docs.nvidia.com/cosmos/latest/latest/prerequisites.html>
- Cosmos Reason2: <https://github.com/nvidia-cosmos/cosmos-reason2>
- Cosmos Predict2.5: <https://github.com/nvidia-cosmos/cosmos-predict2.5>
- Cosmos Transfer2.5: <https://github.com/nvidia-cosmos/cosmos-transfer2.5>
- Isaac GR00T: <https://github.com/NVIDIA/Isaac-GR00T>
- Isaac Lab pip installation: <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>
- Isaac Sim / Isaac Lab GA announcement: <https://developer.nvidia.com/blog/isaac-sim-and-isaac-lab-are-now-available-for-early-developer-preview/>
- Genesis: <https://github.com/Genesis-Embodied-AI/genesis-world>
