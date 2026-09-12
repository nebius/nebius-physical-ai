# NPA workflow specs (`apiVersion: npa.workflow/v0.0.1`)

YAML specifications for composing Workbench tools. Each `toolRef` names a
catalog operation. Author, validate, and submit these specs; NPA renders the
scheduler YAML.

Start with the [workflow guides](guides/README.md) for setup and execution.
For PAIDF with Cosmos 3 on Nebius, follow the
[PAIDF Cosmos 3 setup and run guide](guides/paidf-cosmos3.md). It includes
credentials, current CLI installation, runtime submission, monitoring recovery,
and output inspection, including full-video structural conditioning, aligned
evaluation, configurable quality acceptance and per-variant caption coverage.
Its [LeRobot instructions](guides/paidf-cosmos3.md#r3a-augment-one-lerobot-episode-and-camera)
include a pinned public v3 example, S3 staging, explicit episode/camera selection,
and the full submission command. One run augments one selected video; it does
not produce a reconstructed LeRobot action/state dataset.

Agent skills: [author a workflow](../skills/workflows/author-npa-workflow/SKILL.md)
and [design a new pipeline](../skills/workflows/generate-npa-workflow/SKILL.md).

## Commands

Run commands from the repository root after completing the selected guide's
setup. Replace `<spec.yaml>` with a catalog path such as
`workflows/main/paidf-cosmos3.yaml`; live submission also needs the guide's
project, storage, secret, and runtime options.

```bash
npa workbench workflow validate-spec '<spec.yaml>'
npa workbench workflow plan-spec '<spec.yaml>' --run-id demo
npa workbench workflow submit '<spec.yaml>' --run-id demo --runtime
npa workbench workflow submit '<spec.yaml>' --plan-only     # plan + render only
```

`npa workbench workflow submit` plans the state graph and launches the run.
Use `--plan-only` to inspect the planned steps without launching. Successful
submit output always includes `run_id`; JSON callers receive it as the top-level
`run_id` field. Runs whose specs configure `bucket` can also be rediscovered:

```bash
npa workbench workflow list \
  --s3-bucket '<bucket>' --workflow-s3-prefix '<parent-prefix>' --json
```

Specs with a `parallel:` fan-out group or a loop that must **early-exit on the
real decision artifact** are submitted with the runtime orchestrator:

```bash
npa workbench workflow submit '<spec.yaml>' --run-id demo --runtime
npa workbench workflow plan-spec '<spec.yaml>' --waves    # offline wave preview
```

See the [workflow guide](../docs/workbench/npa-workflow-guide.md)
(Runtime orchestrator) and [design reference](../DESIGN.md).

`metadata.executionMode: runtime` automatically selects the runtime orchestrator
and rejects `--no-runtime`. Runtime-required workflows also reject
`--assume-decision` for execution before staging or provisioning. Offline plans
may use assumed decisions to inspect each route. PAIDF Cosmos 3 declares this
mode so actual evaluator reports control refinement and downstream work.
Runtime submission checks free GPU capacity against each stage as it launches.
CPU stages can continue after generation finishes even when other workloads
occupy the GPUs. GPU stages retain their gang-size and placement checks.

`config.source_overlay: true` selects the automatically staged checkout's NPA
code in pinned workbench images while keeping their installed dependencies.
The default is false for other workflows; `NPA_SRC_OVERLAY=1` remains available
as an operator override. Keep the source checkout available during submission.
PAIDF's additional CLI flags and YAML defaults are described in its
[generation and evaluation settings](guides/paidf-cosmos3.md#r5-find-and-change-generation-and-evaluation-settings).

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

## Layout

- `main/` contains [sim2real.yaml](main/sim2real.yaml),
  [paidf-cosmos3.yaml](main/paidf-cosmos3.yaml), and
  [nurec-reconstruct.yaml](main/nurec-reconstruct.yaml).
- `testing/` contains all other supported declarative workflow YAMLs, including
  examples, integration workflows, and component validation pipelines.
- [`guides/`](guides/README.md) contains setup and operating guides alongside
  the catalog. It contains documentation, not executable workflow specs.
- This README documents the catalog. The non-YAML robot specification example
  lives with its [guide](../docs/workbench/guides/sim2real-robot-spec.md).

Both workflow directories are discovered by the CLI, agent, and live-submit
matrix. Raw SkyPilot examples and resource profiles remain in their guarded
BYOF, burst, and NuRec homes; they are not part of this catalog.

The Cosmos Transfer 2.5 Physical AI Data Factory blueprint remains available as
[physical-ai-data-factory.yaml](testing/physical-ai-data-factory.yaml). Its
[deploy guide](../docs/workbench/guides/physical-ai-data-factory-deploy.md)
describes the real component and artifact contracts.
The direct NVIDIA DIG, IAA, and EVG translations are also testing-tier specs;
all five PAIDF YAMLs emit `reports/upstream.json` and execute through SkyPilot,
not OSMO or Airflow.

## Spec catalog

The tables list every checked-in spec in `main/` and `testing/`. A catalog
entry describes the workflow contract; it does not establish live validation
of every stage. Consult each guide and spec for prerequisites and evidence.

### Main workflows

| Spec | Notes |
| --- | --- |
| [`nurec-reconstruct.yaml`](main/nurec-reconstruct.yaml) | Real NCore V4 capture → 3DGUT training on an RT-core GPU → USDZ → rig-offset novel views → Rerun; [guide and measured evidence](../docs/workbench/guides/neural-reconstruction.md#promotion-evidence), [readiness record](main/nurec-reconstruct.readiness.json) |
| [`paidf-cosmos3.yaml`](main/paidf-cosmos3.yaml) | Generic LeRobot/video input → prepared timeline → guarded Cosmos 3 full-video edge transfer → aligned evaluator gate/refinement → captions for every accepted variant → real Curator + FiftyOne Brain + Rerun ([guide](guides/paidf-cosmos3.md)) |
| [`sim2real.yaml`](main/sim2real.yaml) | Canonical 14-stage Sim2Real workflow through the standard SkyPilot runtime ([guide](../docs/workbench/guides/sim2real-workflow.md)) |

### Testing and reference workflows

| Spec | Notes |
| --- | --- |
| [`adversarial-scenario-hardening.yaml`](testing/adversarial-scenario-hardening.yaml) | Adversarial scenario generation and ranking → policy hardening loop → promotion gate |
| [`alpamayo2-super-inference.yaml`](testing/alpamayo2-super-inference.yaml) | Real Alpamayo 2 Super 34B trajectory inference on `B200:1`; runtime-only OpenMDW weights and separately gated PhysicalAI-AV sample data ([guide](../docs/workbench/alpamayo2-super.md)) |
| [`av-night-scene-hardening.yaml`](testing/av-night-scene-hardening.yaml) | AV night-scene hardening from diagram |
| [`bdd100k-pipeline.yaml`](testing/bdd100k-pipeline.yaml) | 11-stage AV pipeline |
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
| [`curobo-benchmark.yaml`](testing/curobo-benchmark.yaml) | Complete pinned MotionBenchMaker and MPiNets benchmark in cuRobo V2 kinematic and payload-dynamics modes; image remains publication-quarantined pending image checks and real GPU validation ([guide](../docs/workbench/curobo.md)) |
| [`dataset-ingest-curate.yaml`](testing/dataset-ingest-curate.yaml) | Sensor-data ingest → validation gate → slice curation → queryable version registration |
| [`dataset-of-record-smoke.yaml`](testing/dataset-of-record-smoke.yaml) | CPU dataset-of-record smoke using the manifest-backed query fallback |
| [`groot-1-7-finetune.yaml`](testing/groot-1-7-finetune.yaml) | Real GR00T data → parameterized 1-to-many-GPU optimizer smoke → immutable checkpoint → aligned offline evaluation → outcome classification → RRD/MCAP → inspected S3 publication → NPA agent viewer handoff; no rollout or statistical-learning claim |
| [`hardening-with-insights.yaml`](testing/hardening-with-insights.yaml) | Adversarial hardening loop → policy publication → insights metrics, lineage, and dashboard |
| [`insights-aggregate.yaml`](testing/insights-aggregate.yaml) | CPU aggregation of an existing run prefix → dashboard and static HTML |
| [`insights-smoke.yaml`](testing/insights-smoke.yaml) | CPU fixture-run ingestion → comparison and dashboard artifacts |
| [`isaac-franka-capture-reason.yaml`](testing/isaac-franka-capture-reason.yaml) | Headless Isaac Lab Franka RGB capture on GPU → hosted manipulation reasoning |
| [`isaac-lab-rl-sweep.yaml`](testing/isaac-lab-rl-sweep.yaml) | **Parallel** GPU sweep (port of the `execution: parallel` SkyPilot template) + ranking barrier; submit with `--runtime` |
| [`mjlab-eval.yaml`](testing/mjlab-eval.yaml) | MJLab locomotion eval |
| [`multi-node-probe.yaml`](testing/multi-node-probe.yaml) | Gang-scheduled multi-node stage with evidence from every rank |
| [`nurec-colmap-reconstruct.yaml`](testing/nurec-colmap-reconstruct.yaml) | Full COLMAP source -> Apache-2.0 NCore CPU conversion -> separately licensed NRE full-default reconstruction/render on RTX PRO 6000 -> Rerun -> final report; not yet live validated ([guide](../docs/workbench/guides/nurec-colmap-reconstruct.md)) |
| [`openpi-pi05-four-mode.yaml`](testing/openpi-pi05-four-mode.yaml) | Connected OpenPI runtime graph: live negative gate, direct inference, private cross-pod ClusterIP serving, real pi0.5 LoRA optimizer/checkpoint smoke, and disjoint held-out evaluation; consumes the immutable digest built by `byof-openpi.yaml` ([guide](../docs/workbench/openpi-pi05-polaris.md)) |
| [`openpi-pi05-full-droid-finetune.yaml`](testing/openpi-pi05-full-droid-finetune.yaml) | Complete upstream pi0.5 full-DROID recipe: checksum-synced RLDS 1.0.1 and preparation RRD, ten-million-frame normalization, fixed 100-update distributed qualification RRD, global batch 256, 100,000 updates on eight one-RTX-PRO-6000 nodes, durable resume, immutable checkpoint lineage, and verified progress RRD snapshots at 1k/10k/25k/50k/75k/100k ([guide](../docs/workbench/openpi-pi05-polaris.md)) |
| [`paidf-defect-image-generation.yaml`](testing/paidf-defect-image-generation.yaml) | Direct DIG Day-1 manual-ROI translation → runtime base-checkpoint setup → real AnomalyGen fine-tune → inference and native labels; B200; operator-authorized data/weights only |
| [`paidf-event-video-generation.yaml`](testing/paidf-event-video-generation.yaml) | Direct EVG DAG translation → Cosmos3 Super image2video → real detection/captioning/two Visual-QA passes/PAS → anomaly dataset |
| [`paidf-image-attribute-augmentation.yaml`](testing/paidf-image-attribute-augmentation.yaml) | Direct IAA DAG translation → Qwen Image Edit service → real paidf-augmentation verification → real Person Attribute Search → dataset |
| [`physical-ai-data-factory.yaml`](testing/physical-ai-data-factory.yaml) | Cosmos Transfer 2.5 PAIDF blueprint ([deploy guide](../docs/workbench/guides/physical-ai-data-factory-deploy.md)) |
| [`retargeting.yaml`](testing/retargeting.yaml) | Motion retargeting |
| [`rl-policy-training-sim-success.yaml`](testing/rl-policy-training-sim-success.yaml) | Isaac Lab RL train (partial) |
| [`robocasa-data-policy.yaml`](testing/robocasa-data-policy.yaml) | Native multi-task PandaOmron trajectories → LeRobotDataset v3 → real ACT training → exact-checkpoint evaluation on disjoint RoboCasa tasks → insights |
| [`robocasa-smoke.yaml`](testing/robocasa-smoke.yaml) | Native RoboCasa workbench: task registration, asset availability, headless EGL reset, and a real random rollout with video through the npa-robocasa service |
| [`scenario-gen-smoke.yaml`](testing/scenario-gen-smoke.yaml) | CPU adversarial scenario generation and ranking smoke |
| [`sim2real-envgen-shards.yaml`](testing/sim2real-envgen-shards.yaml) | **DEMO ONLY** isolated envgen fan-out fixture |
| [`sim2real-two-step-agent.yaml`](testing/sim2real-two-step-agent.yaml) | **DEMO ONLY** agent-generated two-state DSL fixture |
| [`sim2real-two-step.yaml`](testing/sim2real-two-step.yaml) | **DEMO ONLY** two-state DSL fixture |
| [`sonic-b300-routing-evidence.yaml`](testing/sonic-b300-routing-evidence.yaml) | CPU-only, fail-closed explicit B300 routing evidence with a time-structured RRD ([cookbook](../docs/workbench/cookbooks/sonic-b300-routing-evidence.md)) |
| [`sonic-eval.yaml`](testing/sonic-eval.yaml) | SONIC eval |
| [`sonic-export-eval.yaml`](testing/sonic-export-eval.yaml) | Export → eval |
| [`sonic-export.yaml`](testing/sonic-export.yaml) | SONIC export |
| [`sonic-locomotion-finetuning.yaml`](testing/sonic-locomotion-finetuning.yaml) | Retarget → train → mjlab |
| [`sonic-train.yaml`](testing/sonic-train.yaml) | SONIC train |
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

The single canonical Sim2Real YAML is
`workflows/main/sim2real.yaml`. It is planned, rendered,
submitted, and resumed by the same standard `npa.workflow` + SkyPilot runtime as
every other spec; there is no filename detector or direct-Kubernetes bypass.

## Further reading

- [Workflow guides](guides/README.md): setup and operating instructions.
- [Workflow authoring and runtime](../docs/workbench/npa-workflow-guide.md).
- [Tool catalog](../docs/workbench/npa-workflow-tool-catalog.md).
- [GR00T training cookbook](../docs/workbench/cookbooks/groot-1-7-training.md).
