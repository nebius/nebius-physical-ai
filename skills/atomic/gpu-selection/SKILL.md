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
- B200 / B300: headless, state-based training and inference only.
- CPU: Retargeting and many dataset curation/import steps.

## Blackwell Is Two Different Targets

"Blackwell" spans two CUDA majors, and their binaries are mutually incompatible:

| GPU | Compute capability | SM | Nebius platform |
|---|---|---|---|
| RTX PRO 6000 Blackwell | 12.0 | `sm_120` | `gpu-rtx6000` |
| B200 | 10.0 | `sm_100` | `gpu-b200-sxm` (us-central1) |
| B300 (Blackwell Ultra) | 10.3 | `sm_103` | `gpu-b300-sxm` |

A green smoke on RTX PRO 6000 does not prove B200/B300. Within major 10,
forward compatibility holds, so `sm_100` SASS runs on `sm_103`: target B200
first, then confirm on B300. See
`docs/workbench/blackwell-datacenter-image-compatibility.md` and the per-image
verdicts in `npa/docker/workbench/blackwell-dc-images.json`.

## Gotchas

- H100, H200, and datacenter Blackwell (B200/B300) lack RT cores; do not route
  Isaac Lab or SONIC render validation there. `npa.workbench.sonic.routing`
  classifies these as `datacenter-headless` and rejects render workloads.
- L40S capacity can be constrained; if the task only needs non-render training,
  H100 may be the pragmatic target.
- Preemptible GPU placement does not change any boot-disk allocation. Preserve
  the identical `compute.disk.count` and `compute.disk.size.network-ssd` byte
  requirements in quota plans.
- B200/B300 enablement depends on upstream library support per tool. Treat it as
  vendor-paced unless current tests prove the path. The 2026-08-03 final
  Genesis/Sim2Real tags passed real kernel compilation and physics smokes on
  both B200 and B300; the NVIDIA Isaac vendor stacks and the per-image Cosmos
  blockers in `blackwell-dc-images.json` remain separate constraints.
- Terraform's canonical compute outputs are `platform` and `preset`, with
  `cpu_platform`/`cpu_preset` for CPU-only instances. Deprecated
  `gpu_platform`/`gpu_preset` aliases are GPU-only and return null for CPU
  instances; do not interpret a historical CPU value under those aliases as GPU
  placement.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

The smoke test invokes help for GPU-sensitive training commands and parses the
workflow YAML resources referenced by the manifest.
