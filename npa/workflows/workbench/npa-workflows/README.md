# NPA workflow specs (`apiVersion: npa.workflow/v0.0.1`)

Customer-facing authoring DSL for chaining Workbench tools. Author and submit
these specs; do not hand-edit scheduler YAML.

Agent skills: `skills/workflows/author-npa-workflow/SKILL.md` (edit) and
`skills/workflows/generate-npa-workflow/SKILL.md` (design new pipelines).

## Commands

```bash
npa workbench workflow validate-spec <spec.yaml>
npa workbench workflow plan-spec <spec.yaml> --run-id demo
npa workbench workflow submit <spec.yaml> --run-id demo
npa workbench workflow submit <spec.yaml> --plan-only     # plan + render only
```

`npa workbench workflow submit` plans the state graph and launches the run.
Use `--plan-only` to inspect the planned steps without launching. Successful
submit output always includes `run_id`; JSON callers receive it as the top-level
`run_id` field. Runs whose specs configure `bucket` can also be rediscovered:

```bash
npa workbench workflow list \
  --s3-bucket <bucket> --workflow-s3-prefix <parent-prefix> --json
```

Specs with a `parallel:` fan-out group or a loop that must **early-exit on the
real decision artifact** are submitted with the runtime orchestrator:

```bash
npa workbench workflow submit <spec.yaml> --run-id demo --runtime
npa workbench workflow plan-spec <spec.yaml> --waves    # offline wave preview
```

See `docs/workbench/npa-workflow-guide.md` (Runtime orchestrator) and the
repo-root `DESIGN.md`.

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

## Flagship blueprint

The **NVIDIA Physical AI Data Factory** blueprint (annotate → Cosmos augment →
evaluate/validate gate → re-label → FiftyOne curate → Rerun visualize; no OSMO)
is promoted to the top of the workflow tree for prominence:
[`physical-ai-data-factory.yaml`](physical-ai-data-factory.yaml).
It is still an `npa.workflow/v0.0.1` spec and is discovered alongside the specs
below (see `npa/src/npa/orchestration/npa_workflow/blueprints.py`). Deploy guide:
`docs/workbench/guides/physical-ai-data-factory-deploy.md`.

## Spec catalog

| Spec | Notes |
| --- | --- |
| `alpamayo2-super-inference.yaml` | Real Alpamayo 2 Super 34B trajectory inference on `B200:1`; runtime-only OpenMDW weights and separately gated PhysicalAI-AV sample data ([guide](../../../../docs/workbench/alpamayo2-super.md)) |
| `vlm-eval-single.yaml` | Self-hosted VLM eval |
| `vlm-eval-benchmark.yaml` | VLM benchmark |
| `token-factory-caption.yaml` | Zero-GPU; needs `NPA_SRC_S3_URI` (or `--image`) + `--secret-env NEBIUS_TOKEN_FACTORY_KEY` |
| `token-factory-generate.yaml` | Zero-GPU; needs `NPA_SRC_S3_URI` (or `--image`) |
| `token-factory-cosmos-reason.yaml` | Zero-GPU; needs `NPA_SRC_S3_URI` (or `--image`) |
| `tokenfactory-rollout-judge.yaml` | Reason → VLM chain |
| `tokenfactory-cosmos-gate.yaml` | Gate loop |
| `token-factory-parallel-fanout.yaml` | Zero-GPU **parallel** fan-out (JobGroup) + join barrier; submit with `--runtime` |
| `token-factory-gate-loop.yaml` | Zero-GPU **runtime** gate loop: real early-exit + `goto` branch; submit with `--runtime` |
| `isaac-lab-rl-sweep.yaml` | **Parallel** GPU sweep (port of the `execution: parallel` SkyPilot template) + ranking barrier; submit with `--runtime` |
| `bdd100k-pipeline.yaml` | 11-stage AV pipeline |
| `av-night-scene-hardening.yaml` | AV night-scene hardening from diagram |
| `cosmos-synth-fanout-curation.yaml` | Cosmos synth fan-out + curation |
| `mjlab-eval.yaml` | MJLab locomotion eval |
| `retargeting.yaml` | Motion retargeting |
| `sonic-train.yaml` | SONIC train |
| `sonic-export.yaml` | SONIC export |
| `sonic-eval.yaml` | SONIC eval |
| `sonic-export-eval.yaml` | Export → eval |
| `sonic-locomotion-finetuning.yaml` | Retarget → train → mjlab |
| `sonic-b300-routing-evidence.yaml` | CPU-only, fail-closed explicit B300 routing evidence with a time-structured RRD ([cookbook](../../../../docs/workbench/cookbooks/sonic-b300-routing-evidence.md)) |
| `groot-1-7-finetune.yaml` | Real GR00T data → parameterized 1-to-many-GPU optimizer smoke → immutable checkpoint → aligned offline evaluation → outcome classification → RRD/MCAP → inspected S3 publication → NPA agent viewer handoff; no rollout or statistical-learning claim |
| `cosmos3-reason.yaml` | Cosmos3 reason |
| `cosmos3-checkpoint-eval.yaml` | B200-only guarded Cosmos3 still-image checkpoint evaluation |
| `paidf-cosmos3.yaml` | Independent dynamic PAIDF: generic LeRobot/video input → real Cosmos 3 video2video variants → evaluator gate/refinement → real Curator + FiftyOne Brain + Rerun |
| `content-agents-rigid-object.yaml` | Restricted operator-built NVIDIA Content Agents: source USD → real Material/Physics Agents + OVRTX → upstream validation → rigid Isaac object USDZ/adapter ([guide](../../../../docs/workbench/content-agents.md)) |
| `byof.yaml` | BYOF via `run_byof_repo.py` |
| `byof-maniskill.yaml` | OSS registry: ManiSkill pinned image + PickCube smoke |
| `byof-mujoco-playground.yaml` | OSS registry: MuJoCo Playground pinned image + Cartpole smoke |
| `byof-robocasa.yaml` | OSS registry: RoboCasa pinned image + headless kitchen-task smoke |
| `byof-openpi.yaml` | OSS registry: OpenPI pi0.5 Polaris direct + WebSocket-served Franka joint-position inference on `B200:1`; runtime-only checkpoint and scoped Gemma gate ([guide](../../../../docs/workbench/openpi-pi05-polaris.md)) |
| `openpi-pi05-four-mode.yaml` | Connected OpenPI runtime graph: live negative gate, direct inference, private cross-pod ClusterIP serving, real pi0.5 LoRA optimizer/checkpoint smoke, and disjoint held-out evaluation; consumes the immutable digest built by `byof-openpi.yaml` ([guide](../../../../docs/workbench/openpi-pi05-polaris.md)) |
| `byof-droid-policy-learning.yaml` | OSS registry: DROID policy learning pinned image + RLDS config smoke |
| `rl-policy-training-sim-success.yaml` | Isaac Lab RL train (partial) |
| `sim2real-two-step.yaml` / `sim2real-two-step-agent.yaml` | **DEMO ONLY** two-state DSL fixtures |
| `sim2real-envgen-shards.yaml` | **DEMO ONLY** isolated envgen fan-out fixture |

The single canonical Sim2Real YAML is
`npa/workflows/workbench/npa-workflows/sim2real.yaml`. It is planned, rendered,
submitted, and resumed by the same standard `npa.workflow` + SkyPilot runtime as
every other spec; there is no filename detector or direct-Kubernetes bypass.

## Guide

See `docs/workbench/npa-workflow-guide.md` and
`docs/workbench/npa-workflow-tool-catalog.md`. For the GR00T training workflow,
see `docs/workbench/cookbooks/groot-1-7-training.md`.
