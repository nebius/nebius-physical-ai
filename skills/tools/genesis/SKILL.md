---
name: genesis
description: Use when working on Genesis simulation, RL teacher training, visual demo generation, or related serverless/EGL behavior.
---

# Genesis

Genesis is the GPU-accelerated physics simulation tool. It uses vendored `genesis.ext.pyrender` plus PyOpenGL EGL.

## Current Capability

- RL teacher training works headless on Nebius AI Jobs with Mesa llvmpipe fallback.
- Serverless RL teacher training works.
- 480x640 targeted renders pass via Mesa; scale performance is untested.

## Rendering Gap

Visual demo generation is blocked on EGL/DRI device access in containers. `libEGL_nvidia.so.0` is absent despite graphics capability environment variables, so workloads fall back to Mesa, which cannot scale.

Serverless visual demo generation is blocked by the same EGL/DRI issue.

BatchRenderer/Madrona is deferred; do not prioritize it. There is no platform escalation pending and no `BUNDLE_EGL_NVIDIA` action item.

## Validation

E2E: 8/8 passing except visual demo generation.

## Teacher checkpoint compatibility

The RSL-RL 5 migration uses separate MLP actor and critic configurations and
TensorDict observation groups. Preserve deterministic legacy ActorCritic loading
and the current MLPModel checkpoint path. Load external checkpoints only with
`weights_only=True`; never route them through the upstream unrestricted runner
loader. Validate action and observation widths, normalization, and real ONNX
outputs. CPU PPO round trips cover the framework contract; promotion still
requires the Genesis GPU workload and camera-demo checks.
