# NPA workflow specs (`apiVersion: npa.workflow/v0.0.1`)

Customer-facing authoring DSL for chaining Workbench tools. Prefer these over
raw SkyPilot YAMLs when starting a new pipeline.

## Commands

```bash
npa workbench workflow validate-spec <spec.yaml>
npa workbench workflow plan-spec <spec.yaml> --run-id demo
npa workbench workflow submit <spec.yaml> --run-id demo   # renders → SkyPilot
npa workbench workflow submit <spec.yaml> --plan-only     # render only
```

`npa workbench workflow submit` accepts **both** `npa.workflow/v0.0.1` specs and
legacy SkyPilot YAMLs. For npa.workflow specs it plans the state graph, renders
a serial SkyPilot multi-doc YAML, and submits that. SkyPilot originals under
`../skypilot/` are kept as the production runtime reference until every twin
has a live E2E.

## Live GPU / CPU submit E2E

Skip-by-default. On an operator VM with Nebius creds + SkyPilot:

```bash
# Cheap first: Token Factory CPU twins only
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=cpu \
  ./scripts/npa-workflow-submit-live-e2e.sh

# Full matrix (cpu + gpu + multi)
./scripts/npa-workflow-submit-live-e2e.sh

# Plan-only preflight for every twin (no sky jobs launch)
NPA_E2E_NPA_WORKFLOW_SUBMIT_PLAN_ONLY=1 \
  ./scripts/npa-workflow-submit-live-e2e.sh
```

Requires `NPA_REGISTRY` (or `NPA_E2E_REGISTRY`), and for cpu-tier twins
`NEBIUS_TOKEN_FACTORY_KEY`. Matrix source of truth:
`npa/src/npa/orchestration/npa_workflow/submit_matrix.py`.

## Twins of SkyPilot YAMLs

| npa.workflow spec | SkyPilot twin | Notes |
| --- | --- | --- |
| `vlm-eval-single.yaml` | `../skypilot/vlm-eval.yaml` | Self-hosted VLM; renderer adds vLLM setup |
| `vlm-eval-benchmark.yaml` | `../skypilot/vlm-eval-benchmark.yaml` | |
| `token-factory-caption.yaml` | `../skypilot/token-factory-caption.yaml` | Zero-GPU; needs `NPA_SRC_S3_URI` (or `--image`) + `--secret-env NEBIUS_TOKEN_FACTORY_KEY` |
| `token-factory-generate.yaml` | `../skypilot/token-factory-generate.yaml` | Zero-GPU; needs `NPA_SRC_S3_URI` (or `--image`) |
| `token-factory-cosmos-reason.yaml` | `../skypilot/token-factory-cosmos-reason.yaml` | Zero-GPU; needs `NPA_SRC_S3_URI` (or `--image`) |
| `tokenfactory-rollout-judge.yaml` | `../skypilot/tokenfactory-rollout-judge.yaml` | Spec is reason→VLM; SkyPilot twin is LeRobot→VLM |
| `tokenfactory-cosmos-gate.yaml` | (creative) | Gate loop |
| `bdd100k-pipeline.yaml` | `../skypilot/bdd100k-pipeline.yaml` | 11-stage AV pipeline |
| `mjlab-eval.yaml` | `../skypilot/mjlab-eval.yaml` | |
| `retargeting.yaml` | `../skypilot/retargeting.yaml` | |
| `sonic-train.yaml` | `../skypilot/sonic-train-standalone.yaml` | |
| `sonic-export.yaml` | `../skypilot/sonic-export.yaml` | |
| `sonic-eval.yaml` | `../skypilot/sonic-eval.yaml` | |
| `sonic-export-eval.yaml` | `../skypilot/sonic-export-eval.yaml` | |
| `sonic-locomotion-finetuning.yaml` | `../skypilot/sonic-locomotion-finetuning.yaml` | retarget → train → mjlab |
| `cosmos3-reason.yaml` | `../skypilot/cosmos3-reason.yaml` | |
| `byof.yaml` | `../skypilot/byof-*-rtxpro*.yaml` | Delegates to `run_byof_repo.py` |
| `rl-policy-training-sim-success.yaml` | `../skypilot/isaac-lab-rl-train.yaml` | Partial |
| `sim2real-vlm-rl.yaml` | (demo) | Stub toolRefs; not the 14-stage engine |

## Still SkyPilot-only (documented exceptions)

These stay under `../skypilot/` for now — do not invent incomplete twins:

| SkyPilot YAML | Why it stays |
| --- | --- |
| `isaac-lab-rl-sweep.yaml` | `execution: parallel` — out of v0.0.1 scope |
| `skypilot-kubernetes-rtxpro.yaml` | Global SkyPilot config, not a workflow |
| `isaac-lab-cosmos-sdg-burst-smoke.yaml` | `npa burst submit-yaml` path |
| `sim-to-real-pipeline.yaml` | Monolithic driver, not a toolRef graph |
| `sim-to-real-trigger.yaml` | Poll/submit orchestrator |
| `sim-to-real-loop.yaml` | Custom eval loop script |
| `sim2real-envgen-split.yaml` / `sim2real-actions.yaml` | Staged engine siblings |
| `cosmos3-ea-fetch.yaml` / `cosmos3-text-to-image-inference.yaml` | Catalog toolRefs pending |
| `isaac-franka-capture-reason.yaml` | Capture toolRef pending |
| `tokenfactory-train-triage.yaml` / `tokenfactory-scene-to-rollout-judge.yaml` | Extra stages pending |
| `byof-container-smoke-rtxpro.yaml` / `byof-datagen-rtxpro-smoke.yaml` | Covered via `byof.yaml` + runner |
| `isaac-lab-rl-train-rtxpro*.yaml` | Resource-profile variants of Isaac train |

The Sim2Real **14-stage engine** (`../sim2real/runbook.yaml`) is a separate
path (`npa workbench workflow submit` detects it and routes to direct K8s).

## Guide

See `docs/workbench/npa-workflow-guide.md` and
`docs/workbench/npa-workflow-tool-catalog.md`.
