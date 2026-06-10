---
name: cookbooks
description: Use when a user wants to run an end-to-end NPA workflow and asks how to run BDD100K, sim-to-real, the VLM-eval loop, LeRobot benchmarks, Isaac Lab, or SONIC. Maps each working cookbook in docs/workbench/cookbooks/ to its validated entrypoint.
---

# Cookbooks

Cookbooks in `docs/workbench/cookbooks/` are the validated end-to-end recipes.
Route a user goal to the right cookbook and its exact entrypoint; link the
cookbook rather than re-deriving its steps.

## Working Paths (Validated Now)

Each has a checked-in entrypoint and a dry/offline validation step, so a user can
prove the path before spending GPU.

### BDD100K SkyPilot Pipeline

Perception pipeline: ingest -> CPU backfill -> CLIP embedding -> materialized
views -> train x3 -> eval x3 -> FiftyOne app. The reference end-to-end workflow;
canonical YAML `npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml`. Cookbook:
`docs/workbench/cookbooks/bdd100k-pipeline.md`.

First step (no infrastructure):

```bash
npa/.venv/bin/python npa/scripts/run_bdd100k_pipeline.py \
  --yaml npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml \
  --synthetic 5000 \
  --mock-endpoints \
  --run-id demo-validate \
  --output-json /tmp/bdd100k-validation.json
```

The live run drops `--mock-endpoints`, adds `--lancedb-endpoint`, and provisions
the cluster/services first. One YAML describes the full pipeline; results
complete in about 30 minutes on a single H100.

### Sim-To-Real Pipeline

One-command H100 proof run. Cookbook:
`docs/workbench/cookbooks/sim-to-real-pipeline.md`.

```bash
npa/.venv/bin/python npa/scripts/run_sim_to_real_quickstart.py
```

It renders and submits `sim-to-real-pipeline.yaml` on `H100:1`, prints the
task-success score and S3 URIs, and tears down the run-scoped cluster. SDK local
smoke (`sim_to_real.local_smoke(...)`) runs the same spine without a cluster.

### VLM-Eval Loop

Serve a VLM with vLLM, score rollout directories with `vlm-eval`, write a
task-success report. Cookbook: `docs/workbench/cookbooks/vlm-eval-loop-runbook.md`.
The fully offline first taste is the `vlm-eval benchmark --backend stub` run in
the `quickstart` skill.

### LeRobot GPU Benchmarks

Reproduce LeRobot GPU benchmarks across L40S, H200, B300, RTX PRO 6000 on
serverless Jobs. Cookbooks: `lerobot-gpu-benchmarks.md`,
`lerobot-gpu-benchmarks-runbook.md`. LeRobot is Tier 1 validated on B300.

### Isaac Lab BYOF

Layer a custom Isaac Lab image over the digest-pinned base and run it through the
SkyPilot image override surface. Cookbook: `byof-isaac-lab/README.md`. Isaac Lab
requires RT cores (L40S or RTX PRO 6000); H100/H200 have none.

## Reference Paths (Vendor-Paced)

SONIC, GR00T, Isaac Lab, and Cosmos are paced on NVIDIA CUDA 13 alignment. Route
SONIC to H100 (L40S on-demand capacity is effectively zero). Treat these as
authoring/reference recipes, not first-run proofs; load `mjlab` or `retargeting`
for SONIC locomotion specifics.

- `sonic-locomotion-finetuning.md`: retarget motion, fine-tune, MJLab eval.
- `sonic-mvp-g1-mujoco.md`: G1 warm-start fine-tune + headless MuJoCo eval.
- `sonic-eval-runbook.md` / `sonic-train-runbook.md`: ONNX export and eval.
- `sonic-whole-body-control.md`: whole-body control reference.
- `lancedb-vector-search.md` / `lancedb-deploy-runbook.md`: LanceDB details used
  by the BDD100K pipeline.
- `serverless-tools-coverage.md`: serverless coverage per tool.

## Rules

- Treat buckets, endpoints, registry IDs, run IDs, and thresholds as config.
- Prefer the dry/mock/local step before any live GPU submission.
- Always tear down run-scoped SkyPilot clusters after live runs.
