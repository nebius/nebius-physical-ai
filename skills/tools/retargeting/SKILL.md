---
name: retargeting
description: Use when working on Workbench motion retargeting, SONIC retargeted motion artifacts, SkyPilot retargeting YAMLs, or retargeting CLI behavior.
---

# Retargeting

Retargeting converts already-retargeted SOMA/G1/Bones motion artifacts into the
real motion-lib PKL schema consumed by SONIC locomotion training.

## Interfaces

CLI:

```bash
npa workbench sonic retargeting run
npa workbench sonic retargeting workflow
npa workbench sonic retargeting status
npa workbench sonic retargeting list
```

SkyPilot YAML:

- `npa/src/npa/workflows/skypilot/retargeting.yaml`
- `npa/src/npa/workflows/skypilot/sonic-locomotion-finetuning.yaml`

## Routing And Data Flow

Retargeting is CPU-only by default.

Inputs and outputs use S3 paths:

- `--input-path`: source motion prefix or object.
- `--output-path`: retargeted motion output prefix.
- `--retarget-map`: optional map artifact.

The result artifacts are real `.pkl` motion-lib files plus
`retargeting_result.json` metadata. Do not replace this with a manifest-only
shim.

Raw BVH inputs can be converted to upstream SONIC SOMA skeleton PKLs with
`extract_soma_joints_from_bvh.py`, but upstream SONIC does not bundle a raw
BVH-to-G1 robot retargeter. Use external SOMA Retargeter/GMR before the final
SONIC motion-lib conversion when starting from raw BVH.

## Workflow Constraint

Keep orchestration logic in SkyPilot YAML. Do not add a Python runner script for
the SONIC locomotion fine-tuning path.
