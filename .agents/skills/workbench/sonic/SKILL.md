---
name: sonic
description: Use when working on SONIC robot policy training, GPU routing, validation, or CUDA 13 alignment.
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

Route first-party SONIC images through `npa/src/npa/deploy/sonic_image_manifest.json`.
Use the baked `npa-sonic:0.1.2` image for L40S VM targets and the host-mounted
`npa-sonic:0.1.2-k8s-runtime` image for RTX PRO 6000 Blackwell Kubernetes targets with
the NVIDIA GPU Operator.

SONIC render validation requires RT-capable GPUs. H100 can still be useful for
non-render training throughput, but it is not the default render validation
target.

GPU-routing rules are encoded in `npa/src/npa/workbench/sonic/routing.py` so a
misroute fails fast: retarget is CPU-only; fine-tune / train / MuJoCo eval run
on datacenter-headless (H100/H200/A100/B200) or RT-core GPUs; Isaac-Lab render
is RT-core only (L40S, RTX PRO 6000, Blackwell `sm_120`) and never H100/H200/
A100. The eval container backend (the Isaac render path) validates its
`--container-gpu-target` against `validate_render_gpu_target` before launch.

Known issue: job ID reuse anomaly. Investigation is medium priority and deferred.

CUDA 13 alignment is vendor-paced on NVIDIA x86_64 CUDA 13 and is not Nebius-blocked.
