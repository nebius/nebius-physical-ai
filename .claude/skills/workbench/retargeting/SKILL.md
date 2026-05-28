---
name: retargeting
description: Use when working on Workbench motion retargeting, SONIC retargeted motion artifacts, SkyPilot retargeting YAMLs, or retargeting CLI behavior.
---

# Retargeting

Retargeting converts source motion artifacts into the embodiment schema consumed
by SONIC locomotion training.

## Interfaces

CLI:

```bash
npa workbench retargeting run
npa workbench retargeting workflow
npa workbench retargeting status
npa workbench retargeting list
```

SkyPilot YAML:

- `npa/workflows/workbench/skypilot/retargeting.yaml`
- `npa/workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml`

## Routing And Data Flow

Retargeting is CPU-only by default.

Inputs and outputs use S3 paths:

- `--input-path`: source motion prefix or object.
- `--output-path`: retargeted motion output prefix.
- `--retarget-map`: optional map artifact.

The result artifact is `retargeting_manifest.json`.

## Workflow Constraint

Keep orchestration logic in SkyPilot YAML. Do not add a Python runner script for
the SONIC locomotion fine-tuning path.
