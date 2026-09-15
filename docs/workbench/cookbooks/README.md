# Workbench cookbooks

[Workbench docs](../README.md) · [Workload guides](../guides/README.md) · [Workflow catalog](../../../workflows/README.md)

Recipes for specific workloads, benchmarks, and integrations. Start with
[Workbench setup](../getting-started.md); each recipe states its own inputs and
validation scope. Dated measurements apply to the versions and hardware recorded.

## Policy training and simulation

| Recipe | Purpose |
| --- | --- |
| [GR00T N1.7](groot-1-7-training.md) | Fine-tune, recover the run ID, and inspect checkpoint provenance |
| [Isaac Lab BYOF](byof-isaac-lab/README.md) | Run a custom Isaac Lab fork in a container |
| [SONIC training](sonic-train-runbook.md) | Prepare a single training stage and its runtime |
| [SONIC locomotion](sonic-locomotion-finetuning.md) | Prepare retarget → train → MJLab resource and artifact contracts |
| [SONIC export and eval](sonic-eval-runbook.md) | Export ONNX and inspect evaluation results |
| [G1 MuJoCo image status](sonic-mvp-g1-mujoco.md) | Active evaluation capability and retired combined-image limitations |
| [SONIC whole-body control](sonic-whole-body-control.md) | Training, export, serving, and data contracts |
| [LeRobot benchmarks](lerobot-gpu-benchmarks.md) · [runbook](lerobot-gpu-benchmarks-runbook.md) | Reproduce recorded training measurements |
| [SONIC B300 evidence](sonic-b300-routing-evidence.md) | Scope of the recorded datacenter-GPU checks |

## Data and evaluation

| Recipe | Purpose |
| --- | --- |
| [BDD100K pipeline](bdd100k-pipeline.md) | Ingest, curate, train, and evaluate detectors |
| [LanceDB deployment](lancedb-deploy-runbook.md) | Deploy the service and load data |
| [LanceDB vector search](lancedb-vector-search.md) | Build and query embeddings |
| [VLM evaluation loop](vlm-eval-loop-runbook.md) | Score rollouts and inspect task-success reports |
| [Token Factory with compute](tokenfactory-compute-combos.md) | Pair GPU jobs with hosted inference |
| [Physical reasoning challenge](physical-reasoning-challenge.md) | Recorded reasoning recipe; use current Token Factory model guidance |
| [Serverless coverage](serverless-tools-coverage.md) | Tool-specific modes and recorded checks |
| [CLI / SDK / YAML parity](workbench-parity-tools.md) | Compare supported access paths |
