---
name: sim-to-real
description: Use when designing, reviewing, or operating sim-to-real workflows that move data through simulation, policy training, synthetic generation, evaluation, and loop control.
---

# Sim-To-Real

## When To Use

Use this skill for robotics teams that need configurable data and artifact flow
through simulation, policy training, synthetic data generation, evaluation, and
iteration without customer-specific names or infrastructure baked into source.

## Procedure

1. Define a run ID and run-scoped S3 prefixes before launching.
2. Import source robot, scene, or task data into the run prefix.
3. Generate or augment simulation data with configured workbench tools.
4. Train or fine-tune the policy against the run-scoped dataset and write
   checkpoints to S3.
5. Evaluate the policy with deterministic metrics or a configured VLM backend.
6. Decide whether to stop, continue, or route artifacts for review based on
   configurable thresholds.

## Three-Tier Contract

- CLI: use `npa workbench workflow`, `npa workbench trigger`, and tool commands
  such as Genesis, LeRobot, SONIC, MJLab, Retargeting, LanceDB, Cosmos, and
  VLM-eval.
- SDK: keep workflow submission and config materialization in shared helpers so
  notebooks and services use the same artifact paths.
- YAML: `sim-to-real-loop.yaml`, `sim-to-real-pipeline.yaml`, and
  `sim-to-real-trigger.yaml` are the executable workflow references.

## Gotchas

- Do not hardcode customer names, event names, personal names, tenant IDs,
  registry IDs, bucket names, VM IPs, or private endpoints.
- Prefer dry-run plans for data movement, autoscaling, and external service
  calls when validating workflow shape.
- Keep artifacts partitioned by run ID so repeated experiments do not overwrite
  each other.
- Use existing workbench tools for S3 sync, model inference, training, and VLM
  evaluation instead of one-off scripts.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

The smoke test parses the sim-to-real YAMLs and confirms the workflow CLI help
loads through the installed package.
