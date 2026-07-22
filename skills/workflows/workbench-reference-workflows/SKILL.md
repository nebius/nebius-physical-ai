---
name: workbench-reference-workflows
description: Use when working on NPA reference SkyPilot YAMLs, runner scripts, cookbooks, or customer-adaptable pipeline implementations.
---

# Workbench Reference Workflows

> The supported, customer-facing catalog is the `npa.workflow` spec set under
> `npa/workflows/workbench/npa-workflows/`. The raw SkyPilot task YAMLs below are
> internal runtime templates relocated to `npa/src/npa/workflows/skypilot/`; they
> back the `run_*.py` wrappers and SkyPilot-only capabilities, and must not be
> re-added to the shown `npa/workflows/workbench/` catalog (guardrail-enforced).

## When To Use

Use this skill for repository workflow YAMLs, runner scripts, cookbooks,
artifact contracts, and customer-adaptable pipeline implementations.

## Procedure

1. Start from the checked-in SkyPilot YAML under
   `npa/src/npa/workflows/skypilot/`.
2. Keep the runner thin. Python runners should materialize config, call the
   workflow submission helper, and report artifacts; they should not duplicate
   YAML orchestration logic.
3. Keep all input and output paths configurable and run-scoped through S3.
4. Validate YAML parsing and command help locally before live submission.

## Current Reference YAMLs

- `bdd100k-pipeline.yaml`: BDD100K ingest, backfill, CLIP embedding,
  materialized views, training, and evaluation.
- `cosmos3-ea-fetch.yaml`: Cosmos3 source/checkpoint fetch.
- `cosmos3-text-to-image-inference.yaml`: H100 text-to-image smoke inference.
- `isaac-lab-rl-train.yaml`: single Isaac Lab RL job.
- `isaac-lab-rl-sweep.yaml`: Isaac Lab parameter sweep.
- `mjlab-eval.yaml`: MJLab evaluation.
- `retargeting.yaml`: motion retargeting.
- `sim-to-real-loop.yaml`: iterative sim-to-real loop.
- `sim-to-real-pipeline.yaml`: full sim-to-real pipeline.
- `sim-to-real-trigger.yaml`: trigger wrapper for sim-to-real work.
- `sonic-train-standalone.yaml`: standalone SONIC training.
- `sonic-export.yaml`: SONIC export.
- `sonic-eval.yaml`: SONIC evaluation.
- `sonic-export-eval.yaml`: export plus evaluation.
- `sonic-locomotion-finetuning.yaml`: retargeting, SONIC, and MJLab flow.
- `vlm-eval.yaml` and `vlm-eval-benchmark.yaml`: VLM evaluation loops.

## Three-Tier Contract

- CLI: use `npa workbench workflow ...` and tool-specific workflow commands
  such as `npa workbench mjlab workflow` or `npa workbench retargeting workflow`.
- SDK: route through shared workflow submission helpers rather than shelling out
  from business logic.
- YAML: SkyPilot YAML is the executable source of truth for workflow order,
  resources, environment, and artifact paths.

## Gotchas

- SkyPilot `envs` does not support self-referencing interpolation. Use explicit
  values and comments for alternatives.
- `sky jobs launch` has no dry-run flag in the pinned path. Use local YAML
  parsing, command help, and mock-endpoint tests before live submission.
- Keep orchestration in YAML for SONIC locomotion; do not add a Python runner
  that re-implements the DAG.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

The smoke test parses the listed workflow YAMLs and invokes workflow CLI help.
