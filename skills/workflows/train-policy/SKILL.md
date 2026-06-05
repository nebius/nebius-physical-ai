---
name: train-policy
description: Use when planning, reviewing, or operating robot policy training across LeRobot, Isaac Lab, SONIC, and workflow YAMLs.
---

# Train Policy

## When To Use

Use this skill when a task asks how to train, fine-tune, evaluate, or export a
robot policy through NPA workbench tools. It is the workflow-level entry point
before choosing LeRobot, Isaac Lab, SONIC, or GR00T-specific skills.

## Procedure

1. Identify the policy family and data contract:
   LeRobotDataset for LeRobot, Isaac Lab task config for RL, retargeted motion
   artifacts for SONIC, or model-specific inputs for GR00T.
2. Select the GPU target with `skills/atomic/gpu-selection/SKILL.md`.
3. Configure input and output S3 prefixes. Checkpoints and evaluation artifacts
   must be run-scoped.
4. Choose the executable path:
   direct CLI for a single tool, SDK for application code, or SkyPilot YAML for
   composed training workflows.
5. Verify command help and YAML parsing locally before live GPU submission.

## Three-Tier Contract

- CLI: `npa workbench lerobot train`, `npa workbench isaac-lab train`,
  `npa workbench sonic train`, and related `eval`, `export`, `serve`, or
  `infer` commands.
- SDK: use the workbench SDK modules for application code and shared helper
  functions for request construction.
- YAML: `isaac-lab-rl-train.yaml`, `isaac-lab-rl-sweep.yaml`,
  `sonic-train-standalone.yaml`, and sim-to-real workflow YAMLs are executable
  references.

## Gotchas

- Do not route RT-core-dependent training or render validation to H100/H200.
- Do not substitute repository-local output directories for S3 artifact paths in
  public examples.
- Treat tiny smoke trainers as verification substitutes only when the prompt
  explicitly allows minimal production-input substitution.
- Keep W&B, Hugging Face, NGC, and S3 credentials redacted.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

The smoke test invokes training command help and parses the referenced training
YAMLs.
