---
name: workflows
description: Use when working on NPA reference workflow YAMLs, runner scripts, cookbooks, or customer-adaptable pipeline implementations.
---

# Workflows

Reference workflows follow one pattern: SkyPilot YAML plus a thin Python runner plus S3 artifact output.

## Reference Implementations

All checked-in workflow YAMLs live in `npa/workflows/workbench/skypilot/`. Read
that directory for the current set rather than assuming a fixed list; it covers
BDD100K, sim-to-real (pipeline/loop/trigger), Isaac Lab RL (train/sweep), SONIC
(train/export/eval/locomotion), retargeting, MJLab eval, VLM-eval, and Cosmos
paths. Representative shapes:

- BDD100K pipeline: `bdd100k-pipeline.yaml`. Serial pipeline, ingest -> CPU
  backfill -> CLIP embedding -> materialized views -> train x3 -> eval x3.
- Isaac Lab RL training: `isaac-lab-rl-train.yaml`. Single RL job; requires L40S.
- Isaac Lab RL sweep: `isaac-lab-rl-sweep.yaml`. N parallel jobs, varied
  hyperparameters.

## Runner Scripts

Thin wrappers around `npa.orchestration.skypilot.submit_workflow`:

- `npa/scripts/run_bdd100k_pipeline.py`
- `npa/scripts/run_isaac_lab_rl.py`
- `npa/scripts/run_sim_to_real_pipeline.py`
- `npa/scripts/run_sim_to_real_quickstart.py`

Not every YAML has a runner; SONIC/MJLab/retargeting paths submit via
`npa workbench workflow submit` or the SDK.

## Docs

- Cookbook index: `docs/workbench/cookbooks/README.md`. For mapping a user goal
  to a validated entrypoint, load the `cookbooks` skill.
- Demo: `docs/demos/bdd100k-lancedb-demo.md`
- YAML guide: `docs/workbench-yaml-guide.md` covers label map injection, env var
  patterns, service endpoints, and S3 paths.
