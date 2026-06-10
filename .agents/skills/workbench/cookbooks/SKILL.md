---
name: cookbooks
description: Use when a user wants to run an end-to-end NPA workflow and asks "how do I run X" for BDD100K, sim-to-real, VLM-eval loop, LeRobot benchmarks, Isaac Lab, or SONIC. Maps each working cookbook in docs/workbench/cookbooks/ to its exact, validated entrypoint command.
---

# Cookbooks

Cookbooks in `docs/workbench/cookbooks/` are the validated end-to-end recipes.
This skill routes a user goal to the right cookbook and its exact entrypoint.
Always link the cookbook; do not re-derive its steps from scratch.

## Working Paths (Validated Now)

These have a checked-in entrypoint and a dry/offline validation step, so a user
can prove the path before spending GPU.

### BDD100K SkyPilot Pipeline

Full perception pipeline: ingest -> CPU backfill -> CLIP embedding -> materialized
views -> train x3 -> eval x3 -> FiftyOne app. Cookbook:
`docs/workbench/cookbooks/bdd100k-pipeline.md`.

First step (no infrastructure, mock endpoints):

```bash
npa/.venv/bin/python npa/scripts/run_bdd100k_pipeline.py \
  --yaml npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml \
  --synthetic 5000 \
  --mock-endpoints \
  --run-id demo-validate \
  --output-json /tmp/bdd100k-validation.json
```

Expected: exit `0`, all tasks return `0`, request order
`import-bdd100k -> 6x backfill -> 3x create-mv` and `3x train -> 3x eval`. The
live run drops `--mock-endpoints`, adds `--lancedb-endpoint`, and provisions the
cluster/services first per the cookbook.

### Sim-To-Real Pipeline

One-command H100 proof run (zero-flag credential resolution). Cookbook:
`docs/workbench/cookbooks/sim-to-real-pipeline.md`.

```bash
npa/.venv/bin/python npa/scripts/run_sim_to_real_quickstart.py
```

It renders `sim-to-real-pipeline.yaml`, submits on `H100:1`, prints the
task-success score plus checkpoint/report/Rerun S3 URIs, and tears down the
run-scoped cluster. Local smoke without a cluster: `sim_to_real.local_smoke(...)`
(SDK path in the cookbook). The CLI wrapper `run_sim_to_real_pipeline.py` exposes
every knob (eval backend, feedback source/type, GPU + failover, BYO image).

### VLM-Eval Loop

Serve a VLM with vLLM, score rollout directories with `vlm-eval`, write a
task-success report. Cookbook: `docs/workbench/cookbooks/vlm-eval-loop-runbook.md`.
The cookbook's "One Command" block renders `sim-to-real-loop.yaml` to a temp
workflow and submits it. The fully offline first taste is the zero-credential
`vlm-eval benchmark --backend stub` run from the `quickstart` skill. Load the
`vlm-eval` skill for backends, benchmark sweeps, and loop inputs/outputs.

### LeRobot GPU Benchmarks

Reproduce the LeRobot GPU benchmark across L40S, H200, B300, RTX PRO 6000 on
serverless Jobs, with seconds/step measurements, artifact checks, and cleanup.
Cookbooks: `lerobot-gpu-benchmarks.md` and `lerobot-gpu-benchmarks-runbook.md`.

### Isaac Lab BYOF

Layer a custom Isaac Lab image over the digest-pinned Workbench base and run it
through the SkyPilot image override surface. Cookbook:
`docs/workbench/cookbooks/byof-isaac-lab/README.md`. Isaac Lab requires RT cores
(L40S or RTX PRO 6000); H100/H200 have none.

## Reference Paths (Vendor-Paced)

SONIC locomotion is real but gated on NVIDIA CUDA 13 alignment and H100 routing
(L40S on-demand capacity is effectively zero). Treat these as authoring/reference
recipes, not first-run proofs. Load the `sonic`, `mjlab`, or `retargeting` skill
before changing behavior.

- `sonic-locomotion-finetuning.md`: retarget motion, fine-tune, MJLab eval (YAML
  only, no Python runner).
- `sonic-mvp-g1-mujoco.md`: G1 warm-start fine-tune + headless MuJoCo eval.
- `sonic-eval-runbook.md` / `sonic-train-runbook.md`: ONNX export and eval.
- `sonic-whole-body-control.md`: whole-body control reference.
- `lancedb-vector-search.md` / `lancedb-deploy-runbook.md`: LanceDB table, query,
  import, and deploy details used by the BDD100K pipeline.
- `serverless-tools-coverage.md`: which tools have serverless coverage.

## Rules

- Treat buckets, endpoints, registry IDs, run IDs, and thresholds as config;
  never hardcode them.
- Prefer the dry/mock/local step in each cookbook before any live GPU submission.
- Always tear down run-scoped SkyPilot clusters after live runs.
