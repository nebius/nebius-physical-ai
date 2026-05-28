---
name: mjlab
description: Use when working on MJLab locomotion evaluation, SONIC checkpoint scoring, SkyPilot MJLab YAMLs, or Workbench MJLab CLI behavior.
---

# MJLab

MJLab is the locomotion evaluation stage for SONIC Workbench workflows.

## Interfaces

CLI:

```bash
npa workbench mjlab eval
npa workbench mjlab workflow
npa workbench mjlab status
npa workbench mjlab list
```

SkyPilot YAML:

- `npa/workflows/workbench/skypilot/mjlab-eval.yaml`
- `npa/workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml`

## Routing And Data Flow

Route MJLab evaluation to H100 for the checked-in workflow templates.

Inputs and outputs use S3 paths:

- `--input-path`: retargeted motion or rollout artifacts.
- `--checkpoint`: SONIC checkpoint artifact.
- `--output-path`: MJLab evaluation output prefix.

The result artifact is `mjlab_eval.json`.

## Workflow Constraint

Keep orchestration logic in SkyPilot YAML. Do not add a Python runner script for
the SONIC locomotion fine-tuning path.
