# NPA workflow catalog

[Docs](../docs/README.md) · [Authoring guide](../docs/workbench/npa-workflow-guide.md)

These `npa.workflow/v0.0.1` YAML files compose Workbench operations into a state
graph. NPA validates the graph, renders SkyPilot tasks, and manages run-scoped
artifacts. Start with a runbook that matches the result you want.

## Choose a starting point

| Goal | Spec and runbook |
| --- | --- |
| Augment a video or LeRobot episode | [PAIDF + Cosmos 3](guides/paidf-cosmos3.md) — public starter, local MP4, and episode/camera inputs |
| Generate an image or video | [Cosmos 3](../docs/workbench/cosmos3-generate.md) |
| Reconstruct a captured scene | [NuRec](../docs/workbench/guides/neural-reconstruction.md) |
| Compose the 14-stage robot loop | [Sim2Real](../docs/workbench/guides/sim2real-workflow.md) |
| Train a GR00T policy | [GR00T N1.7](../docs/workbench/cookbooks/groot-1-7-training.md) |
| Package your own repository | [BYOF](../docs/workbench/cookbooks/byof-isaac-lab/README.md) |

A catalog entry describes a contract, not a guarantee that every configuration
has run successfully. The `testing/` directory includes executable examples,
integration tests, and explicitly labeled fixtures. Read each guide's required
input, image/GPU compatibility, and validation scope before allocating compute.

## Commands

From the repository root with [NPA installed](../docs/install.md), inspect a
checked-in example locally:

```bash
npa workbench workflow validate-spec workflows/testing/cosmos3-generate.yaml
npa workbench workflow plan-spec workflows/testing/cosmos3-generate.yaml --run-id demo
```

Expect a valid spec and one `generate` stage. Its bucket is a placeholder;
validation and planning do not verify model access, stage data, or reserve GPUs.

For execution, follow the selected runbook in order:

1. Configure the project and prepare its input, storage, credentials, and compute.
2. Validate and plan with the configuration overrides you will submit.
3. Use `prepare-run` to persist a run ID, then check image pullability.
4. Submit through `submit --runtime` using the same project, context, and values.
5. Inspect the exact run's `status`, `logs`, and `artifacts`; verify the outputs.

Successful submission reports `run_id`, including at the top level of JSON
output. Keep it with your private run notes. Runs with durable bucket-backed
state can also be rediscovered:

```bash
npa workbench workflow list \
  --s3-bucket '<your-bucket>' --workflow-s3-prefix '<parent-prefix>' --json
```

See [run lifecycle](../docs/run-lifecycle.md) for launch, monitoring, interrupted
submissions, and safe resume, and [teardown](../docs/teardown.md) for cleanup.

### Runtime choices

Use the runtime orchestrator for parallel groups and decisions evaluated during
the run. `metadata.executionMode: runtime` selects it automatically and rejects
`--no-runtime`. Runtime-required workflows reject `--assume-decision` during
execution; use assumed decisions only to inspect alternate paths offline.
`plan-spec --waves` previews scheduling waves. `submit --plan-only` previews the
static render and launches no workload.

`config.source_overlay: true` uses the automatically staged checkout's NPA code
with the selected images' installed dependencies. Other workflows default to
false; `NPA_SRC_OVERLAY=1` is an operator override. Keep the checkout available
during submission. The [authoring guide](../docs/workbench/npa-workflow-guide.md)
explains the full source and image contract.

## Layout

| Directory | Contents |
| --- | --- |
| `main/` | The three principal pipelines listed below |
| `testing/` | All other catalog specs, including component tests and fixtures |
| [`guides/`](guides/README.md) | Setup and operation runbooks |

The CLI, agent, and live-submit matrix discover both YAML directories. Raw
SkyPilot tasks are separate tool-specific examples; see the
[reference-assets index](../npa/workflows/workbench/README.md).

The Cosmos Transfer VDA and direct NVIDIA DIG, IAA, and EVG translations are
all testing-tier PAIDF specs. Together with the Cosmos 3 alternative, all five
record `reports/upstream.json` and execute through SkyPilot rather than OSMO or
Airflow.

## Spec catalog

### Main workflows

| Spec | Notes |
| --- | --- |
| [`nurec-reconstruct.yaml`](main/nurec-reconstruct.yaml) | Real NCore V4 capture → 3DGUT training on an RT-core GPU → USDZ → rig-offset novel views → Rerun; [guide and measured evidence](../docs/workbench/guides/neural-reconstruction.md#promotion-evidence), [readiness record](main/nurec-reconstruct.readiness.json) |
| [`paidf-cosmos3.yaml`](main/paidf-cosmos3.yaml) | Generic LeRobot/video input → prepared timeline → guarded Cosmos 3 full-video edge transfer → aligned evaluator gate/refinement → captions for every accepted variant → real Curator + FiftyOne Brain + Rerun ([guide](guides/paidf-cosmos3.md)) |
| [`sim2real.yaml`](main/sim2real.yaml) | Canonical 14-stage Sim2Real workflow through the standard SkyPilot runtime ([guide](../docs/workbench/guides/sim2real-workflow.md)) |

### Testing and reference workflows

Jump to: [Generation and reconstruction](#generation-and-reconstruction) · [Robot learning and simulation](#robot-learning-and-simulation) · [Data, perception, and scenario analysis](#data-perception-and-scenario-analysis) · [Bring your own framework](#bring-your-own-framework) · [Hosted inference and VLM evaluation](#hosted-inference-and-vlm-evaluation) · [Infrastructure and runtime examples](#infrastructure-and-runtime-examples)

#### Generation and reconstruction

| Spec | Notes |
| --- | --- |
| [`content-agents-rigid-object.yaml`](testing/content-agents-rigid-object.yaml) | NVIDIA Content Agents with a public image and runtime-fetched OVRTX: source USD → real Material/Physics Agents + OVRTX → upstream validation → rigid Isaac object USDZ/adapter ([guide](../docs/workbench/content-agents.md)) |
| [`cosmos-fetch.yaml`](testing/cosmos-fetch.yaml) | Check Cosmos source/checkpoint access and materialize a local cache |
| [`cosmos-synth-fanout-curation.yaml`](testing/cosmos-synth-fanout-curation.yaml) | Cosmos synth fan-out + curation |
| [`cosmos2-transfer.yaml`](testing/cosmos2-transfer.yaml) | Standalone Cosmos Transfer 2.5 GPU augmentation → video and frames |
| [`cosmos3-checkpoint-eval.yaml`](testing/cosmos3-checkpoint-eval.yaml) | B200-only guarded Cosmos3 still-image checkpoint evaluation |
| [`cosmos3-generate.yaml`](testing/cosmos3-generate.yaml) | Cosmos3-Nano image generation with default guardrails; gated guardrail assets require HF access |
| [`cosmos3-ray-batch.yaml`](testing/cosmos3-ray-batch.yaml) | Prepared SDG batch through an existing Cosmos3-Nano Ray Serve deployment → media and provenance |
| [`cosmos3-reason.yaml`](testing/cosmos3-reason.yaml) | Cosmos3 reason |
| [`cosmos3-super-b200-benchmark.yaml`](testing/cosmos3-super-b200-benchmark.yaml) | Cosmos3-Super serving benchmark on one eight-GPU B200 node |
| [`cosmos3-super-h200-benchmark.yaml`](testing/cosmos3-super-h200-benchmark.yaml) | Cosmos3-Super serving benchmark on one eight-GPU H200 node |
| [`cosmos3-super-h200-single-gpu.yaml`](testing/cosmos3-super-h200-single-gpu.yaml) | Isolated Cosmos3-Super TP-1 validation on one H200; distinct from node-throughput benchmarks |
| [`cosmos3-text-to-image.yaml`](testing/cosmos3-text-to-image.yaml) | Public Cosmos3-Nano image generation with guardrails disabled → verified image and manifest |
| [`nurec-colmap-reconstruct.yaml`](testing/nurec-colmap-reconstruct.yaml) | Full COLMAP source -> Apache-2.0 NCore CPU conversion -> separately licensed NRE full-default reconstruction/render on RTX PRO 6000 -> Rerun -> final report; not yet live validated ([guide](../docs/workbench/guides/nurec-colmap-reconstruct.md)) |
| [`paidf-defect-image-generation.yaml`](testing/paidf-defect-image-generation.yaml) | Direct DIG Day-1 manual-ROI translation → runtime base-checkpoint setup → real AnomalyGen fine-tune → inference and native labels; B200; operator-authorized data/weights only |
| [`paidf-event-video-generation.yaml`](testing/paidf-event-video-generation.yaml) | Direct EVG DAG translation → Cosmos3 Super image2video → real detection/captioning/two Visual-QA passes/PAS → anomaly dataset |
| [`paidf-image-attribute-augmentation.yaml`](testing/paidf-image-attribute-augmentation.yaml) | Direct IAA DAG translation → Qwen Image Edit service → real paidf-augmentation verification → real Person Attribute Search → dataset |
| [`physical-ai-data-factory.yaml`](testing/physical-ai-data-factory.yaml) | Cosmos Transfer 2.5 PAIDF blueprint ([deploy guide](../docs/workbench/guides/physical-ai-data-factory-deploy.md)) |

#### Robot learning and simulation

| Spec | Notes |
| --- | --- |
| [`curobo-benchmark.yaml`](testing/curobo-benchmark.yaml) | Complete pinned MotionBenchMaker and MPiNets benchmark in cuRobo V2 kinematic and payload-dynamics modes; image remains publication-quarantined pending image checks and real GPU validation ([guide](../docs/workbench/curobo.md)) |
| [`groot-1-7-finetune.yaml`](testing/groot-1-7-finetune.yaml) | Real GR00T data → parameterized 1-to-many-GPU optimizer smoke → immutable checkpoint → aligned offline evaluation → outcome classification → RRD/MCAP → inspected S3 publication → NPA agent viewer handoff; no rollout or statistical-learning claim |
| [`isaac-franka-capture-reason.yaml`](testing/isaac-franka-capture-reason.yaml) | Headless Isaac Lab Franka RGB capture on GPU → hosted manipulation reasoning |
| [`isaac-lab-rl-sweep.yaml`](testing/isaac-lab-rl-sweep.yaml) | **Parallel** GPU sweep (port of the `execution: parallel` SkyPilot template) + ranking barrier; submit with `--runtime` |
| [`mjlab-eval.yaml`](testing/mjlab-eval.yaml) | MJLab locomotion eval |
| [`openpi-pi05-four-mode.yaml`](testing/openpi-pi05-four-mode.yaml) | Connected OpenPI runtime graph: live negative gate, direct inference, private cross-pod ClusterIP serving, real pi0.5 LoRA optimizer/checkpoint smoke, and disjoint held-out evaluation; consumes the immutable digest built by `byof-openpi.yaml` ([guide](../docs/workbench/openpi-pi05-polaris.md)) |
| [`openpi-pi05-full-droid-finetune.yaml`](testing/openpi-pi05-full-droid-finetune.yaml) | Complete upstream pi0.5 full-DROID recipe: checksum-synced RLDS 1.0.1 and preparation RRD, ten-million-frame normalization, fixed 100-update distributed qualification RRD, global batch 256, 100,000 updates on eight one-RTX-PRO-6000 nodes, durable resume, immutable checkpoint lineage, and verified progress RRD snapshots at 1k/10k/25k/50k/75k/100k ([guide](../docs/workbench/openpi-pi05-polaris.md)) |
| [`retargeting.yaml`](testing/retargeting.yaml) | Motion retargeting |
| [`rl-policy-training-sim-success.yaml`](testing/rl-policy-training-sim-success.yaml) | Isaac Lab RL train (partial) |
| [`robocasa-data-policy.yaml`](testing/robocasa-data-policy.yaml) | Native multi-task PandaOmron trajectories → LeRobotDataset v3 → real ACT training → exact-checkpoint evaluation on disjoint RoboCasa tasks → insights |
| [`robocasa-smoke.yaml`](testing/robocasa-smoke.yaml) | Native RoboCasa workbench: task registration, asset availability, headless EGL reset, and a real random rollout with video through the npa-robocasa service |
| [`sonic-eval.yaml`](testing/sonic-eval.yaml) | SONIC eval |
| [`sonic-export-eval.yaml`](testing/sonic-export-eval.yaml) | Export → eval |
| [`sonic-export.yaml`](testing/sonic-export.yaml) | SONIC export |
| [`sonic-locomotion-finetuning.yaml`](testing/sonic-locomotion-finetuning.yaml) | Retarget → train → mjlab |
| [`sonic-train.yaml`](testing/sonic-train.yaml) | SONIC train |

#### Data, perception, and scenario analysis

| Spec | Notes |
| --- | --- |
| [`alpamayo2-ray-sweep.yaml`](testing/alpamayo2-ray-sweep.yaml) | Ray GPU actors sweep scenario, seed, and diffusion settings; Ray CPU tasks reduce measured ADE/FDE and seed variability ([guide](../docs/workbench/alpamayo2-super.md#ray-experiments)) |
| [`alpamayo2-ray-hardcases.yaml`](testing/alpamayo2-ray-hardcases.yaml) | Ray baseline → mean-error selection → refinement with matched scenarios and seeds; reports measured error changes ([guide](../docs/workbench/alpamayo2-super.md#ray-experiments)) |
| [`alpamayo2-super-inference.yaml`](testing/alpamayo2-super-inference.yaml) | Real Alpamayo 2 Super 34B trajectory inference on `B200:1`; runtime-only OpenMDW weights and separately gated PhysicalAI-AV sample data ([guide](../docs/workbench/alpamayo2-super.md)) |
| [`adversarial-scenario-hardening.yaml`](testing/adversarial-scenario-hardening.yaml) | Adversarial scenario generation and ranking → policy hardening loop → promotion gate |
| [`av-night-scene-hardening.yaml`](testing/av-night-scene-hardening.yaml) | AV night-scene hardening from diagram |
| [`bdd100k-pipeline.yaml`](testing/bdd100k-pipeline.yaml) | 11-stage AV pipeline |
| [`dataset-ingest-curate.yaml`](testing/dataset-ingest-curate.yaml) | Sensor-data ingest → validation gate → slice curation → queryable version registration |
| [`dataset-of-record-smoke.yaml`](testing/dataset-of-record-smoke.yaml) | CPU dataset-of-record smoke using the manifest-backed query fallback |
| [`hardening-with-insights.yaml`](testing/hardening-with-insights.yaml) | Adversarial hardening loop → policy publication → insights metrics, lineage, and dashboard |
| [`insights-aggregate.yaml`](testing/insights-aggregate.yaml) | CPU aggregation of an existing run prefix → dashboard and static HTML |
| [`insights-smoke.yaml`](testing/insights-smoke.yaml) | CPU fixture-run ingestion → comparison and dashboard artifacts |
| [`scenario-gen-smoke.yaml`](testing/scenario-gen-smoke.yaml) | CPU adversarial scenario generation and ranking smoke |

#### Bring your own framework

| Spec | Notes |
| --- | --- |
| [`byof-droid-policy-learning.yaml`](testing/byof-droid-policy-learning.yaml) | OSS registry: DROID policy learning pinned image + RLDS config smoke |
| [`byof-ltx2.yaml`](testing/byof-ltx2.yaml) | LTX-2.5 video generation and FiftyOne curation; source and gated weights fetched at runtime |
| [`byof-maniskill.yaml`](testing/byof-maniskill.yaml) | OSS registry: ManiSkill pinned image + PickCube smoke |
| [`byof-mujoco-playground.yaml`](testing/byof-mujoco-playground.yaml) | OSS registry: MuJoCo Playground pinned image + Cartpole smoke |
| [`byof-open-dreamer.yaml`](testing/byof-open-dreamer.yaml) | Open Dreamer multi-GPU tokenizer and dynamics training on Minecraft/VPT data → action-conditioned dream rollout and Rerun evidence |
| [`byof-openpi.yaml`](testing/byof-openpi.yaml) | OSS registry: OpenPI pi0.5 Polaris direct + WebSocket-served Franka joint-position inference on `B200:1`; runtime-only checkpoint and scoped Gemma gate ([guide](../docs/workbench/openpi-pi05-polaris.md)) |
| [`byof-robocasa.yaml`](testing/byof-robocasa.yaml) | OSS registry: RoboCasa pinned image + headless kitchen-task smoke |
| [`byof-wan2.2-multigpu.yaml`](testing/byof-wan2.2-multigpu.yaml) | Wan 2.2 generation across four participating GPU ranks; MP4, topology, and Rerun evidence |
| [`byof-wan2.2.yaml`](testing/byof-wan2.2.yaml) | Wan 2.2 TI2V-5B on one RTX PRO 6000; decoded MP4 and verified Rerun evidence |
| [`byof.yaml`](testing/byof.yaml) | BYOF via `run_byof_repo.py` |

#### Hosted inference and VLM evaluation

| Spec | Notes |
| --- | --- |
| [`token-factory-batch-generate.yaml`](testing/token-factory-batch-generate.yaml) | Hosted asynchronous batch text generation → generations JSONL |
| [`token-factory-caption.yaml`](testing/token-factory-caption.yaml) | Hosted vision captioning; pass `--secret-env NEBIUS_TOKEN_FACTORY_KEY` |
| [`token-factory-cosmos-reason.yaml`](testing/token-factory-cosmos-reason.yaml) | Hosted inference; pass `--secret-env NEBIUS_TOKEN_FACTORY_KEY` |
| [`token-factory-gate-loop.yaml`](testing/token-factory-gate-loop.yaml) | Zero-GPU **runtime** gate loop: real early-exit + `goto` branch; submit with `--runtime` |
| [`token-factory-generate.yaml`](testing/token-factory-generate.yaml) | Hosted inference; pass `--secret-env NEBIUS_TOKEN_FACTORY_KEY` |
| [`token-factory-parallel-fanout.yaml`](testing/token-factory-parallel-fanout.yaml) | Zero-GPU **parallel** fan-out (JobGroup) + join barrier; submit with `--runtime` |
| [`token-factory-trigger-watch.yaml`](testing/token-factory-trigger-watch.yaml) | Wait for frames in an inbox prefix → hosted vision captioning |
| [`tokenfactory-cosmos-gate.yaml`](testing/tokenfactory-cosmos-gate.yaml) | Gate loop |
| [`tokenfactory-rollout-judge-combo.yaml`](testing/tokenfactory-rollout-judge-combo.yaml) | LeRobot GPU rollout → hosted VLM evaluation |
| [`tokenfactory-rollout-judge.yaml`](testing/tokenfactory-rollout-judge.yaml) | Reason → VLM chain |
| [`tokenfactory-scene-to-rollout-judge.yaml`](testing/tokenfactory-scene-to-rollout-judge.yaml) | Hosted scene reasoning → GPU policy rollout → hosted VLM evaluation against the plan |
| [`tokenfactory-train-triage.yaml`](testing/tokenfactory-train-triage.yaml) | LeRobot GPU policy training → hosted text triage of run artifacts |
| [`vlm-eval-benchmark.yaml`](testing/vlm-eval-benchmark.yaml) | VLM benchmark |
| [`vlm-eval-loop.yaml`](testing/vlm-eval-loop.yaml) | Self-hosted VLM rollout-set evaluation → aggregate task-success report |
| [`vlm-eval-single.yaml`](testing/vlm-eval-single.yaml) | Self-hosted VLM eval |
| [`vlm-eval-token-factory.yaml`](testing/vlm-eval-token-factory.yaml) | Hosted Token Factory VLM evaluation over a rollout prefix |

#### Infrastructure and runtime examples

| Spec | Notes |
| --- | --- |
| [`multi-node-probe.yaml`](testing/multi-node-probe.yaml) | Gang-scheduled multi-node stage with evidence from every rank |
| [`sim2real-envgen-shards.yaml`](testing/sim2real-envgen-shards.yaml) | **DEMO ONLY** isolated envgen fan-out fixture |
| [`sim2real-two-step-agent.yaml`](testing/sim2real-two-step-agent.yaml) | **DEMO ONLY** agent-generated two-state DSL fixture |
| [`sim2real-two-step.yaml`](testing/sim2real-two-step.yaml) | **DEMO ONLY** two-state DSL fixture |
| [`sonic-b300-routing-evidence.yaml`](testing/sonic-b300-routing-evidence.yaml) | CPU-only, fail-closed explicit B300 routing evidence with a time-structured RRD ([cookbook](../docs/workbench/cookbooks/sonic-b300-routing-evidence.md)) |

The canonical Sim2Real spec is `workflows/main/sim2real.yaml`; it uses the same
standard runtime as the other workflows. Its small `sim2real-*` fixtures do not
substitute for a complete robot-learning run.

## Live GPU / CPU submit E2E

Skip-by-default. Use a configured operator VM with Nebius credentials,
SkyPilot bootstrapped, and access to the workflow's models and services:

```bash
# CPU-tier workflows
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=cpu \
  ./scripts/npa-workflow-submit-live-e2e.sh

# Full matrix (cpu + gpu + multi)
./scripts/npa-workflow-submit-live-e2e.sh

# Plan-only preflight for the matrix (no job launch)
NPA_E2E_NPA_WORKFLOW_SUBMIT_PLAN_ONLY=1 \
  ./scripts/npa-workflow-submit-live-e2e.sh
```

Supported images use public GHCR by default; `NPA_E2E_REGISTRY` is an optional
explicit override for custom images. Hosted inference stages need
`NEBIUS_TOKEN_FACTORY_KEY`; model and data access depend on the selected spec.
Use `NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=paidf-cosmos3.yaml` to select the Cosmos 3
workflow. Matrix source of truth:
[`submit_matrix.py`](../npa/src/npa/orchestration/npa_workflow/submit_matrix.py).
Set `NPA_E2E_NPA_WORKFLOW_RUNTIME=1` to run its full dynamic pipeline; the
one-shot test verifies that assumed promotion is refused. Runtime validation
downloads and fully decodes the source and generated videos, verifies their
timelines and control hashes, and checks every downstream component report.

## Further reading

- [Workflow runbooks](guides/README.md) and [robot guides](../docs/workbench/guides/README.md).
- [Tool catalog](../docs/workbench/npa-workflow-tool-catalog.md) and [authoring reference](../docs/workbench/npa-workflow-guide.md).
- Agent skills: [author a workflow](../skills/workflows/author-npa-workflow/SKILL.md) or [design a pipeline](../skills/workflows/generate-npa-workflow/SKILL.md).
