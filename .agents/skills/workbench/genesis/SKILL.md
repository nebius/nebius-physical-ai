---
name: genesis
description: Use when working on Genesis simulation, RL teacher training, visual demo generation, or related serverless/EGL behavior.
last_verified: 2026-05-26
owner: workbench
version: 1.0.0
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

## Changelog

- 2026-05-26: Added frontmatter metadata (last_verified, owner, version) and Changelog section per skill-authoring.
