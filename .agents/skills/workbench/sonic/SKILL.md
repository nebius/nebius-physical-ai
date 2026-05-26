---
name: sonic
description: Use when working on SONIC robot policy training, GPU routing, validation, or CUDA 13 alignment.
last_verified: 2026-05-26
owner: workbench
version: 1.0.0
---

# SONIC

SONIC is a workbench tool for robot policy training.

## Interfaces

CLI:

```bash
npa workbench sonic deploy
npa workbench sonic train
npa workbench sonic eval
npa workbench sonic status
npa workbench sonic system-info
npa workbench sonic list
```

## Routing And Validation

Route SONIC to H100. Do not use L40S; the `1gpu-40vcpu-160gb` preset has effectively zero on-demand availability.

SONIC is validated end-to-end on Nebius when routed to H100.

Known issue: job ID reuse anomaly. Investigation is medium priority and deferred.

CUDA 13 alignment is vendor-paced on NVIDIA x86_64 CUDA 13 and is not Nebius-blocked.

## Changelog

- 2026-05-26: Added frontmatter metadata (last_verified, owner, version) and Changelog section per skill-authoring.
