---
name: gpu-selection
description: Use when choosing or reviewing GPU targets for NPA workbench tools, training, rendering, inference, or workflow YAML resources.
---

# GPU Selection

## When To Use

Use this skill when a task asks which GPU family to use, changes workflow
resources, updates image routing, or reviews render/training placement.

## Procedure

1. Identify whether the workload needs RT cores, tensor throughput, multi-GPU
   scaling, or only CPU resources.
2. Check the tool-specific skill for hard constraints.
3. Encode the choice in CLI flags, SDK config, or workflow YAML env/resources.
4. Keep image variants aligned with GPU selection.

## Three-Tier Contract

- CLI: commands expose GPU choices through flags such as `--gpu-type`,
  `--gpu-preset`, `--runtime`, or tool-specific image variant options.
- SDK: runtime config and request builders should carry GPU type/count rather
  than deriving it from private environment names.
- YAML: workflow resources and env vars must express the GPU target explicitly
  enough for reviewers to validate routing.

## Current Defaults

- H100: general training, CLIP embedding, detection, MJLab, Cosmos inference,
  LeRobot training smoke, and non-render throughput.
- L40S: Isaac Lab and SONIC render validation on VM hosts.
- RTX PRO 6000 Blackwell: Isaac Lab and SONIC render validation on Kubernetes
  with mounted NVIDIA GPU Operator drivers.
- CPU: Retargeting and many dataset curation/import steps.

## Gotchas

- H100 and H200 lack RT cores; do not route Isaac Lab or SONIC render validation
  there.
- L40S capacity can be constrained; if the task only needs non-render training,
  H100 may be the pragmatic target.
- B300/Blackwell enablement depends on upstream library support. Treat it as
  vendor-paced unless current tests prove the path.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

The smoke test invokes help for GPU-sensitive training commands and parses the
workflow YAML resources referenced by the manifest.
