---
name: workflows
description: Use when working on NPA reference workflow YAMLs, runner scripts, cookbooks, or customer-adaptable pipeline implementations.
---

# Workflows

Reference workflows follow one pattern: SkyPilot YAML plus a thin Python runner plus S3 artifact output.

## Reference Implementations

- BDD100K pipeline: `npa/workflows/skypilot/bdd100k-pipeline.yaml`. It has 10 tasks across 6 logical stages: ingest, CPU backfill, CLIP embedding, materialized views, training x3, eval x3.
- Isaac Lab RL training: `npa/workflows/skypilot/isaac-lab-rl-train.yaml`. It is a single RL job and requires L40S.
- Isaac Lab RL sweep: `npa/workflows/skypilot/isaac-lab-rl-sweep.yaml`. It runs N parallel jobs with different hyperparameters.

## Runner Scripts

- `npa/scripts/run_bdd100k_pipeline.py`
- `npa/scripts/run_isaac_lab_rl.py`

## Docs

- Cookbook: `docs/demos/bdd100k-lancedb-demo.md`
- Cookbook: `docs/cookbooks/bdd100k-pipeline.md`
- YAML guide: `docs/workbench-yaml-guide.md`

`docs/workbench-yaml-guide.md` covers label map injection, env var patterns, service endpoints, and S3 paths.
