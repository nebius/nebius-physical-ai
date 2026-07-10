# NPA workflow specs (`apiVersion: npa.workflow/v0.0.1`)

Customer-facing authoring DSL for chaining Workbench tools. Author and submit
these specs; do not hand-edit scheduler YAML.

## Commands

```bash
npa workbench workflow validate-spec <spec.yaml>
npa workbench workflow plan-spec <spec.yaml> --run-id demo
npa workbench workflow submit <spec.yaml> --run-id demo
npa workbench workflow submit <spec.yaml> --plan-only     # plan + render only
```

`npa workbench workflow submit` plans the state graph and launches the run.
Use `--plan-only` to inspect the planned steps without launching.

## Live GPU / CPU submit E2E

Skip-by-default. On an operator VM with Nebius creds:

```bash
# Cheap first: Token Factory CPU twins only
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=cpu \
  ./scripts/npa-workflow-submit-live-e2e.sh

# Full matrix (cpu + gpu + multi)
./scripts/npa-workflow-submit-live-e2e.sh

# Plan-only preflight for every twin (no job launch)
NPA_E2E_NPA_WORKFLOW_SUBMIT_PLAN_ONLY=1 \
  ./scripts/npa-workflow-submit-live-e2e.sh
```

Requires `NPA_REGISTRY` (or `NPA_E2E_REGISTRY`), and for cpu-tier twins
`NEBIUS_TOKEN_FACTORY_KEY`. Matrix source of truth:
`npa/src/npa/orchestration/npa_workflow/submit_matrix.py`.

## Spec catalog

| Spec | Notes |
| --- | --- |
| `vlm-eval-single.yaml` | Self-hosted VLM eval |
| `vlm-eval-benchmark.yaml` | VLM benchmark |
| `token-factory-caption.yaml` | Zero-GPU; needs `NPA_SRC_S3_URI` (or `--image`) + `--secret-env NEBIUS_TOKEN_FACTORY_KEY` |
| `token-factory-generate.yaml` | Zero-GPU; needs `NPA_SRC_S3_URI` (or `--image`) |
| `token-factory-cosmos-reason.yaml` | Zero-GPU; needs `NPA_SRC_S3_URI` (or `--image`) |
| `tokenfactory-rollout-judge.yaml` | Reason → VLM chain |
| `tokenfactory-cosmos-gate.yaml` | Gate loop |
| `bdd100k-pipeline.yaml` | 11-stage AV pipeline |
| `mjlab-eval.yaml` | MJLab locomotion eval |
| `retargeting.yaml` | Motion retargeting |
| `sonic-train.yaml` | SONIC train |
| `sonic-export.yaml` | SONIC export |
| `sonic-eval.yaml` | SONIC eval |
| `sonic-export-eval.yaml` | Export → eval |
| `sonic-locomotion-finetuning.yaml` | Retarget → train → mjlab |
| `cosmos3-reason.yaml` | Cosmos3 reason |
| `byof.yaml` | BYOF via `run_byof_repo.py` |
| `rl-policy-training-sim-success.yaml` | Isaac Lab RL train (partial) |
| `sim2real-vlm-rl.yaml` | Demo loops; stub toolRefs (not the 14-stage engine) |

The Sim2Real **14-stage engine** is a separate path under
`npa/workflows/workbench/sim2real/` (`npa workbench workflow submit` detects the
runbook and routes to direct K8s).

## Guide

See `docs/workbench/npa-workflow-guide.md` and
`docs/workbench/npa-workflow-tool-catalog.md`.
