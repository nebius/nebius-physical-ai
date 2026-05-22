---
name: architecture
description: Use for Claude Code architectural review of the Nebius Physical AI platform, workbench layer, orchestrator choices, and partner model.
---

# Architecture Context

Detailed rationale and the preserved May 2026 architecture snapshot live in
`docs/architecture/contributor-context.md`.

The platform progression is:

1. Tools: Workbench.
2. Composition: Platform.
3. Intelligence: agentic layer.

The current phase is Workbench, the Tools layer: marketplace model, pay per use, `npa` CLI/SDK/agents.

There are 8 tools: LeRobot, FiftyOne, Genesis, Isaac Lab, Cosmos, LanceDB, GR00T, and SONIC.

Tool validation state: 7/8 validated end-to-end. Isaac Lab is pending a training command fix. Cosmos is EGL/DRI-blocked for rendering.

SkyPilot is the sole orchestrator. Argo is deprecated.

Partner model: partners listed in the ecosystem must run workloads on Nebius infrastructure when accessed through the platform.

OSMO is available for existing announced relationships but is not the foundation for Nebius's own implementation.

LeRobot connects data generation to robot policy training and is the default training framework for robot policy workflows.

The sim-to-real gap is the most important and least-tooled layer. Existing approaches are either deeply custom or unproductized.
