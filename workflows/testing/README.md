# Testing and reference workflows

[Workflow catalog](../README.md) · [Main workflows](../main/README.md)

Choose the YAML and its guide from the same row. These specs include GPU
workflows, component checks, and explicitly labeled fixtures or stubs. A catalog
entry does not establish live qualification; read the guide and spec for required
inputs, image compatibility, and validation scope before submitting.

A CLI reference or the general workflow guide is labeled as such when no
workflow-specific runbook is linked. Detailed guides stay in their existing
locations so related workflows can share setup instructions.

Jump to: [Generation and reconstruction](#generation-and-reconstruction) · [Robot learning and simulation](#robot-learning-and-simulation) · [Data, perception, and scenario analysis](#data-perception-and-scenario-analysis) · [Bring your own framework](#bring-your-own-framework) · [Hosted inference and VLM evaluation](#hosted-inference-and-vlm-evaluation) · [Infrastructure and runtime examples](#infrastructure-and-runtime-examples)

## Generation and reconstruction

| Workflow | Guide | Purpose |
| --- | --- | --- |
| [`content-agents-rigid-object.yaml`](content-agents-rigid-object.yaml) | [Guide](../../docs/workbench/content-agents.md) | NVIDIA Content Agents with a public image and runtime-fetched OVRTX: source USD → real Material/Physics Agents + OVRTX → upstream validation → rigid Isaac object USDZ/adapter |
| [`cosmos-fetch.yaml`](cosmos-fetch.yaml) | [Access setup](../../docs/workbench/cosmos3-access-preflight.md) | Check Cosmos source/checkpoint access and materialize a local cache |
| [`cosmos-synth-fanout-curation.yaml`](cosmos-synth-fanout-curation.yaml) | [General workflow guide](../../docs/workbench/npa-workflow-guide.md) | Cosmos synth fan-out + curation |
| [`cosmos2-transfer.yaml`](cosmos2-transfer.yaml) | [Transfer CLI](../../docs/cli/cosmos.md) | Standalone Cosmos Transfer 2.5 GPU augmentation → video and frames |
| [`cosmos3-checkpoint-eval.yaml`](cosmos3-checkpoint-eval.yaml) | [Checkpoint evaluation](../../docs/workbench/cosmos3-b200-checkpoint-evaluation-20260814.md) | B200-only guarded Cosmos3 still-image checkpoint evaluation |
| [`cosmos3-generate.yaml`](cosmos3-generate.yaml) | [Generation guide](../../docs/workbench/cosmos3-generate.md) | Cosmos3-Nano image generation with default guardrails; gated guardrail assets require HF access |
| [`cosmos3-policy-model-factory.yaml`](cosmos3-policy-model-factory.yaml) | [Scope and prerequisites](../../docs/workbench/cosmos3-policy-model-factory.md) | Experimental native LIBERO policy SFT → simulator evaluation → failure feedback → guarded video candidates |
| [`cosmos3-ray-batch.yaml`](cosmos3-ray-batch.yaml) | [Ray Serve guide](../../docs/workbench/cosmos3-ray-serve.md) | Prepared SDG batch through an existing Cosmos3-Nano Ray Serve deployment → media and provenance |
| [`cosmos3-reason.yaml`](cosmos3-reason.yaml) | [General workflow guide](../../docs/workbench/npa-workflow-guide.md) | Cosmos3 reason |
| [`cosmos3-super-b200-benchmark.yaml`](cosmos3-super-b200-benchmark.yaml) | [Serving guide](../../docs/workbench/cosmos3-super-serving.md) | Cosmos3-Super serving benchmark on one eight-GPU B200 node |
| [`cosmos3-super-b200-single-gpu.yaml`](cosmos3-super-b200-single-gpu.yaml) | [Serving guide](../../docs/workbench/cosmos3-super-serving.md) | Isolated Cosmos3-Super TP-1 validation on one B200; distinct from node-throughput benchmarks |
| [`cosmos3-super-h200-benchmark.yaml`](cosmos3-super-h200-benchmark.yaml) | [Serving guide](../../docs/workbench/cosmos3-super-serving.md) | Cosmos3-Super serving benchmark on one eight-GPU H200 node |
| [`cosmos3-super-h200-single-gpu.yaml`](cosmos3-super-h200-single-gpu.yaml) | [Serving guide](../../docs/workbench/cosmos3-super-serving.md) | Isolated Cosmos3-Super TP-1 validation on one H200; distinct from node-throughput benchmarks |
| [`cosmos3-text-to-image.yaml`](cosmos3-text-to-image.yaml) | [Generation guide](../../docs/workbench/cosmos3-generate.md) | Public Cosmos3-Nano image generation with guardrails disabled → verified image and manifest |
| [`lyra-reconstruction.yaml`](lyra-reconstruction.yaml) | [Review and scene integration](../../docs/workbench/guides/lyra-physical-augmentation.md) | Standalone Lyra 2 captured-video reconstruction → native Gaussians, predicted depth/cameras, rendered video and offline interactive HTML. Private internal R&D scope. |
| [`living-lab-nurec-fanout.yaml`](living-lab-nurec-fanout.yaml) | [NuRec fan-out guide](../../docs/workbench/guides/living-lab-nurec-fanout.md) | Sixteen zone/view jobs from eight NCore capture pairs → native RTX reconstruction → reports and contact sheet. |
| [`nurec-colmap-reconstruct.yaml`](nurec-colmap-reconstruct.yaml) | [Guide](../../docs/workbench/guides/nurec-colmap-reconstruct.md) | Full COLMAP source -> Apache-2.0 NCore CPU conversion -> separately licensed NRE full-default reconstruction/render on RTX PRO 6000 -> Rerun -> final report; not yet live validated |
| [`nvidia-paidf-vda-cosmos-transfer25.yaml`](nvidia-paidf-vda-cosmos-transfer25.yaml) | [Deploy guide](../../docs/workbench/guides/physical-ai-data-factory-deploy.md) | Separately named NVIDIA-derived VDA semantic translation → pinned upstream contract → real Cosmos Transfer 2.5/Evaluator/Curator/FiftyOne → Rerun |
| [`open3d-registration.yaml`](open3d-registration.yaml) | [Open3D CLI](../../docs/cli/open3d.md) | CPU point-cloud registration → optimized pose graph → Poisson surface → verified Rerun recording; public sample or supplied scans. |
| [`paidf-defect-image-generation.yaml`](paidf-defect-image-generation.yaml) | [PAIDF deploy guide](../../docs/workbench/guides/physical-ai-data-factory-deploy.md) | Direct DIG Day-1 manual-ROI translation → runtime base-checkpoint setup → real AnomalyGen fine-tune → inference and native labels; B200; operator-authorized data/weights only |
| [`paidf-event-video-generation.yaml`](paidf-event-video-generation.yaml) | [PAIDF deploy guide](../../docs/workbench/guides/physical-ai-data-factory-deploy.md) | Direct EVG DAG translation → Cosmos3 Super image2video → real detection/captioning/two Visual-QA passes/PAS → anomaly dataset |
| [`paidf-image-attribute-augmentation.yaml`](paidf-image-attribute-augmentation.yaml) | [PAIDF deploy guide](../../docs/workbench/guides/physical-ai-data-factory-deploy.md) | Direct IAA DAG translation → Qwen Image Edit service → real paidf-augmentation verification → real Person Attribute Search → dataset |
| [`physical-ai-data-factory.yaml`](physical-ai-data-factory.yaml) | [Deploy guide](../../docs/workbench/guides/physical-ai-data-factory-deploy.md) | Cosmos Transfer 2.5 PAIDF blueprint |
| [`rgbd-scan-to-isaac.yaml`](rgbd-scan-to-isaac.yaml) | [Guide](../../docs/workbench/guides/rgbd-scan-to-isaac.md) | Metric RGB-D with poses → real Open3D TSDF and held-out depth gate → derived colored USDZ and exact triangle colliders → native Isaac PhysX; complete managed public capture qualified, with a verified navigation-input handoff |
| [`rgbd-scan-to-policy-demo.yaml`](rgbd-scan-to-policy-demo.yaml) | [Guide](../../docs/workbench/guides/public-workflow-demos.md) | Automatic public RGB-D sample → measured collision scene → native navigation training → held-out goals and offline HTML |
| [`scan-to-isaac-navigation.yaml`](scan-to-isaac-navigation.yaml) | [Guide](../../docs/workbench/guides/scan-to-isaac-navigation.md) | Existing NuRec visual scene + operator collision USD mesh and measured transforms → portable USDZ/provenance → actual Isaac PhysX ray probes; live qualification pending, no navigation-policy claim ([readiness](scan-to-isaac-navigation.readiness.json)) |
| [`seedvr2-video-restoration.yaml`](seedvr2-video-restoration.yaml) | [SeedVR2 guide](../../docs/workbench/seedvr2.md) | Pinned SeedVR2-3B restoration with H100/sample defaults and explicit B200/posterior-mode controls → independent S3 readback verification → non-blended bicubic/candidate review package; image remains publication-quarantined pending live objective and VLM evidence. |
| [`video-variant-sweep-cosmos3.yaml`](video-variant-sweep-cosmos3.yaml) | [Variant sweep guide](../guides/video-variant-sweep.md) | Explicit or Cartesian variants → parallel Cosmos3-Nano full-source edge transfer → paired review, lineage and accepted clips. |
| [`video-variant-sweep.yaml`](video-variant-sweep.yaml) | [Variant sweep guide](../guides/video-variant-sweep.md) | Source captions and hints → parallel Cosmos Transfer 2.5 variants → paired review → Postgres/MLflow lineage and accepted clips. |

## Robot learning and simulation

| Workflow | Guide | Purpose |
| --- | --- | --- |
| [`behavior-challenge-eval.yaml`](behavior-challenge-eval.yaml) | [Challenge onboarding](../../docs/workbench/challenge-onboarding.md) · [Scope and validation](../../docs/workbench/behavior-campaign.md#scope-and-validation-status) | BEHAVIOR 2026 evaluation with the prescribed cases; operator simulator runtime, assets and task-configured policy required. |
| [`behavior-comet-native-full-training.yaml`](behavior-comet-native-full-training.yaml) | [Guide](../../docs/workbench/comet-native-full-training.md) | Portable real OpenPI native training reference: same-entrypoint CPU input preflight → direct native GPU updates → complete FP32 TrainState/optimizer milestones with provider readback and durable resume; exact private inputs remain operator supplied |
| [`curobo-benchmark.yaml`](curobo-benchmark.yaml) | [Guide](../../docs/workbench/curobo.md) | Complete pinned MotionBenchMaker and MPiNets benchmark in cuRobo V2 kinematic and payload-dynamics modes; image remains publication-quarantined pending image checks and real GPU validation |
| [`field-failure-policy-improvement.yaml`](field-failure-policy-improvement.yaml) | [Native and operator adapter runbook](../../docs/workbench/cookbooks/field-failure-policy-improvement.md) | Navigation failures → reconstruction and policy improvement; sealed data/runtime required and GPU acceptance pending. |
| [`field-failure-reference-demo.yaml`](field-failure-reference-demo.yaml) | [Guide](../../docs/workbench/guides/public-workflow-demos.md) | Public inputs → baseline training → observed simulation failures → capture admission and reconstruction → mixed-scene replay → development gate → independently held-out comparison and offline HTML |
| [`flex-pi-b200-inference.yaml`](flex-pi-b200-inference.yaml) | [Flex-pi guide](../../docs/workbench/flex-pi.md) | Single-B200 action-only inference on a pinned public RoboTwin observation; runtime-fetched checkpoint and inputs. |
| [`flex-pi-b200-public-training.yaml`](flex-pi-b200-public-training.yaml) | [Flex-pi guide](../../docs/workbench/flex-pi.md) | Four-B200 training on pinned public YAM data, with held-out evaluation and fresh-process resume; no private-baseline comparison. |
| [`flex-pi-b300-inference.yaml`](flex-pi-b300-inference.yaml) | [Flex-pi guide](../../docs/workbench/flex-pi.md) | Compiled B300 action-only inference with pinned runtime dependencies and source overlay. |
| [`flex-pi-b300-multinode-public-training.yaml`](flex-pi-b300-multinode-public-training.yaml) | [Flex-pi guide](../../docs/workbench/flex-pi.md) | Public YAM training on four single-B300 hosts, with held-out evaluation and resume verification. |
| [`flex-pi-rtxpro-inference.yaml`](flex-pi-rtxpro-inference.yaml) | [Flex-pi guide](../../docs/workbench/flex-pi.md) | Single-RTX-PRO-6000 action-only inference on a public RoboTwin observation; no simulator or task-success claim. |
| [`franka-rl-transfer.yaml`](franka-rl-transfer.yaml) | [Runbook](../../docs/workbench/guides/franka-rl-transfer.md) | PPO with selectable Franka, UR10e/Robotiq, or Kinova JACO2 embodiment, bounded learned targets and exploration, stable-hold rewards, and an outcome-driven curriculum (`learning_recipe=adaptive-bounded-exploration`). Measured-state checks reject invalid simulation before learning; sealed USD parts, paired physics tests, Token Factory evaluation, and actual LeRobot/Rerun rollouts retain the evidence. Latest native UR diagnostic **failed**; physics remains unqualified despite passing CPU checks. One RTX GPU plus a hosted VLM credential. |
| [`groot-1-7-finetune.yaml`](groot-1-7-finetune.yaml) | [Training guide](../../docs/workbench/cookbooks/groot-1-7-training.md) | Real GR00T data → parameterized 1-to-many-GPU optimizer smoke → immutable checkpoint → aligned offline evaluation → outcome classification → RRD/MCAP → inspected S3 publication → NPA agent viewer handoff; no rollout or statistical-learning claim |
| [`isaac-arena-evaluation-b200.yaml`](isaac-arena-evaluation-b200.yaml) | [Guide](../../docs/workbench/isaac-arena.md) | Four-seed Arena zero-action state regression on B200; completed scored episodes and hash-bound reports, with no visual claim |
| [`isaac-arena-evaluation-rtxpro.yaml`](isaac-arena-evaluation-rtxpro.yaml) | [Arena guide](../../docs/workbench/isaac-arena.md) | Arena replay on RTX PRO 6000; exact-digest qualification completed with upstream task success and simulator-ground-truth-bound viewport motion ([readiness](isaac-arena-evaluation-rtxpro.readiness.json)) |
| [`isaac-franka-capture-reason.yaml`](isaac-franka-capture-reason.yaml) | [Hosted/GPU composition](../../docs/workbench/composing-cloud-and-token-factory.md) | Headless Isaac Lab Franka RGB capture on GPU → hosted manipulation reasoning |
| [`isaac-lab-rl-sweep.yaml`](isaac-lab-rl-sweep.yaml) | [Isaac Lab CLI](../../docs/cli/isaac-lab.md) | **Parallel** GPU sweep (port of the `execution: parallel` SkyPilot template) + ranking barrier; submit with `--runtime` |
| [`lerobot-subtask-proof.yaml`](lerobot-subtask-proof.yaml) | [Guide](../../docs/workbench/guides/lerobot-subtask-labeling.md) | CPU post-review gate: complete LeRobot v3 `subtask_index` coverage → catalog resolution → row-level proof bound to the source Parquet digest |
| [`lerobot-transfer.yaml`](lerobot-transfer.yaml) | [Runbook](../../docs/workbench/guides/lerobot-transfer.md) | Four-phase LeRobot demonstration experiment: paired ACT training → native PushT transfer stress tests → measured comparison and next-expert-demo queue. |
| [`lyra-scene-actions.yaml`](lyra-scene-actions.yaml) | [Scene integration](../../docs/workbench/guides/lyra-physical-augmentation.md) | Execute physical variations in a calibrated Lyra collision scene and record workspace/wrist cameras. Prepared robot mounting and RTX required; imported-scene qualification remains separate. |
| [`mjlab-eval.yaml`](mjlab-eval.yaml) | [MJLab guide](../../docs/workbench/mjlab.md) | Measured native MJLab checkpoint evaluation |
| [`mjlab-render.yaml`](mjlab-render.yaml) | [MJLab guide](../../docs/workbench/mjlab.md) | Trained MJLab checkpoint to measured evaluation, rendered MP4 and self-contained HTML on RTX PRO 6000 |
| [`mjlab-train-eval.yaml`](mjlab-train-eval.yaml) | [MJLab guide](../../docs/workbench/mjlab.md) | Native MJLab training, measured evaluation, ONNX export and independent-seed evaluation |
| [`molmoact-finetune.yaml`](molmoact-finetune.yaml) | [MolmoAct CLI](../../docs/cli/molmoact.md) | **Stub stage:** fine-tuning toolRef and three-tier contract; not real training. |
| [`multicamera-rgbd-capture.yaml`](multicamera-rgbd-capture.yaml) | [Guide](../../docs/workbench/multicamera-rgbd-capture.md) | Calibrated USD sensor rig → synchronized RGB/depth/poses and optional colored world points → decoded S3 validation; GPU acceptance pending |
| [`multicamera-rgbd-warehouse.yaml`](multicamera-rgbd-warehouse.yaml) | [Guide](../../docs/workbench/multicamera-rgbd-capture.md) | Runtime-collect NVIDIA's full warehouse → four 1280×720 RGB-D streams at 265 poses → per-camera and fused world clouds with decoded S3 validation; complete native RTX qualification and artifact readback recorded for the public demo ([source-bound evidence](../../docs/workbench/evidence/public-demos/README.md)) |
| [`newton-train-teacher.yaml`](newton-train-teacher.yaml) | [Newton CLI](../../docs/cli/newton.md) | **Stub stage:** teacher-training toolRef contract; not real training. |
| [`openarm-simulators.yaml`](openarm-simulators.yaml) | [OpenArm guide](../../docs/workbench/openarm.md) | Enactic bimanual MuJoCo, Isaac Lab reach and RSL-RL training, with artifact validation. |
| [`openpi-pi05-four-mode.yaml`](openpi-pi05-four-mode.yaml) | [Guide](../../docs/workbench/openpi-pi05-polaris.md) | Connected OpenPI runtime graph: live negative gate, direct inference, private cross-pod ClusterIP serving, real pi0.5 LoRA optimizer/checkpoint smoke, and disjoint held-out evaluation; consumes the immutable digest built by `byof-openpi.yaml` |
| [`openpi-pi05-full-droid-finetune.yaml`](openpi-pi05-full-droid-finetune.yaml) | [Guide](../../docs/workbench/openpi-pi05-polaris.md) | Complete upstream pi0.5 full-DROID recipe: checksum-synced RLDS 1.0.1 and preparation RRD, ten-million-frame normalization, fixed 100-update distributed qualification RRD, global batch 256, 100,000 updates on eight one-RTX-PRO-6000 nodes, durable resume, immutable checkpoint lineage, and verified progress RRD snapshots at 1k/10k/25k/50k/75k/100k |
| [`openvla-train.yaml`](openvla-train.yaml) | [OpenVLA CLI](../../docs/cli/openvla.md) | **Stub stage:** OpenVLA-OFT LoRA toolRef contract; not real training. |
| [`physical-augmentation.yaml`](physical-augmentation.yaml) | [One-command demo](../../docs/workbench/guides/physical-augmentation.md) | Fresh Franka actions across position, mass and friction changes → interactive HD replay, comparison film and accepted LeRobot/Rerun demonstrations. Requires RTX PRO 6000. |
| [`retargeting.yaml`](retargeting.yaml) | [Retargeting CLI](../../docs/cli/retargeting.md) | Motion retargeting |
| [`rl-policy-training-sim-success.yaml`](rl-policy-training-sim-success.yaml) | [Isaac Lab CLI](../../docs/cli/isaac-lab.md) | Isaac Lab RL train (partial) |
| [`robocasa-data-policy.yaml`](robocasa-data-policy.yaml) | [RoboCasa CLI](../../docs/cli/robocasa.md) | Native multi-task PandaOmron trajectories → LeRobotDataset v3 → real ACT training → exact-checkpoint evaluation on disjoint RoboCasa tasks → insights |
| [`robocasa-smoke.yaml`](robocasa-smoke.yaml) | [RoboCasa CLI](../../docs/cli/robocasa.md) | Native RoboCasa workbench: task registration, asset availability, headless EGL reset, and a real random rollout with video through the npa-robocasa service |
| [`shared-scene-navigation.yaml`](shared-scene-navigation.yaml) | [Contract and runbook](../../docs/workbench/guides/shared-scene-navigation.md) | Native Isaac public quadruped reference or BYOF navigation with shared-scene physics/perception probes, checkpoint resume and held-out evaluation. Public bundle builder supplies a cluttered warehouse; reconstructed scenes require measured resets. GPU acceptance unverified. |
| [`sonic-eval.yaml`](sonic-eval.yaml) | [Evaluation runbook](../../docs/workbench/cookbooks/sonic-eval-runbook.md) | SONIC eval |
| [`sonic-export-eval.yaml`](sonic-export-eval.yaml) | [Evaluation runbook](../../docs/workbench/cookbooks/sonic-eval-runbook.md) | Export → eval |
| [`sonic-export.yaml`](sonic-export.yaml) | [Evaluation runbook](../../docs/workbench/cookbooks/sonic-eval-runbook.md) | SONIC export |
| [`sonic-locomotion-finetuning.yaml`](sonic-locomotion-finetuning.yaml) | [Fine-tuning guide](../../docs/workbench/cookbooks/sonic-locomotion-finetuning.md) | Retarget → train → export → native SONIC eval |
| [`sonic-train.yaml`](sonic-train.yaml) | [Training runbook](../../docs/workbench/cookbooks/sonic-train-runbook.md) | SONIC train |

## Data, perception, and scenario analysis

| Workflow | Guide | Purpose |
| --- | --- | --- |
| [`adversarial-scenario-hardening.yaml`](adversarial-scenario-hardening.yaml) | [General workflow guide](../../docs/workbench/npa-workflow-guide.md) | Adversarial scenario generation and ranking → policy hardening loop → promotion gate |
| [`alpamayo2-ray-hardcases.yaml`](alpamayo2-ray-hardcases.yaml) | [Guide](../../docs/workbench/alpamayo2-super.md#ray-experiments) | Ray baseline → mean-error selection → refinement with matched scenarios and seeds; reports measured error changes |
| [`alpamayo2-ray-sweep.yaml`](alpamayo2-ray-sweep.yaml) | [Guide](../../docs/workbench/alpamayo2-super.md#ray-experiments) | Ray GPU actors sweep scenario, seed, and diffusion settings; Ray CPU tasks reduce measured ADE/FDE and seed variability |
| [`alpamayo2-super-inference.yaml`](alpamayo2-super-inference.yaml) | [Guide](../../docs/workbench/alpamayo2-super.md) | Real Alpamayo 2 Super 34B trajectory inference on `B200:1`; runtime-only OpenMDW weights and separately gated PhysicalAI-AV sample data |
| [`av-night-scene-hardening.yaml`](av-night-scene-hardening.yaml) | [General workflow guide](../../docs/workbench/npa-workflow-guide.md) | 8-stage AV night-scene pipeline ending when both detector metrics artifacts are written; human/FiftyOne inspection is post-run |
| [`bdd100k-pipeline.yaml`](bdd100k-pipeline.yaml) | [Cookbook](../../docs/workbench/cookbooks/bdd100k-pipeline.md) | 10-stage AV pipeline ending when all three detector metrics artifacts are written; human/FiftyOne inspection is post-run |
| [`dataset-ingest-curate.yaml`](dataset-ingest-curate.yaml) | [Dataset CLI](../../docs/cli/dataset.md) | Sensor-data ingest → validation gate → slice curation → queryable version registration |
| [`dataset-of-record-smoke.yaml`](dataset-of-record-smoke.yaml) | [Dataset CLI](../../docs/cli/dataset.md) | CPU dataset-of-record smoke using the manifest-backed query fallback |
| [`hardening-with-insights.yaml`](hardening-with-insights.yaml) | [General workflow guide](../../docs/workbench/npa-workflow-guide.md) | Adversarial hardening loop → policy publication → insights metrics, lineage, and dashboard |
| [`insights-aggregate.yaml`](insights-aggregate.yaml) | [Insights CLI](../../docs/cli/insights.md) | CPU aggregation of an existing run prefix → dashboard and static HTML |
| [`insights-smoke.yaml`](insights-smoke.yaml) | [Insights CLI](../../docs/cli/insights.md) | CPU fixture-run ingestion → comparison and dashboard artifacts |
| [`scenario-gen-smoke.yaml`](scenario-gen-smoke.yaml) | [General workflow guide](../../docs/workbench/npa-workflow-guide.md) | CPU adversarial scenario generation and ranking smoke |

## Bring your own framework

| Workflow | Guide | Purpose |
| --- | --- | --- |
| [`byof-apriltag.yaml`](byof-apriltag.yaml) | [BYOF CLI](../../docs/cli/byof.md) | Plan-only catalog definition; run pinned CPU fiducial detection and controls through the direct BYOF runner |
| [`byof-cogvideox-2b.yaml`](byof-cogvideox-2b.yaml) | [Video/perception guide](../../docs/workbench/video-generation-byof.md) | Native B200 text-to-video with pinned runtime and runtime-fetched weights. |
| [`byof-depth-anything-v2.yaml`](byof-depth-anything-v2.yaml) | [Video/perception guide](../../docs/workbench/video-generation-byof.md) | GPU relative-depth prediction on a hash-verified video → raw predictions and visualization; no ground-truth or robot-success claim. |
| [`byof-droid-policy-learning.yaml`](byof-droid-policy-learning.yaml) | [BYOF CLI](../../docs/cli/byof.md) | OSS registry: DROID policy learning pinned image + RLDS config smoke |
| [`byof-evo.yaml`](byof-evo.yaml) | [BYOF CLI](../../docs/cli/byof.md) | Plan-only catalog definition; run pinned evo APE/RPE controls and matched KITTI plots through the direct BYOF runner |
| [`byof-gymnasium-robotics.yaml`](byof-gymnasium-robotics.yaml) | [Qualification guide](../../docs/workbench/byof-gymnasium-robotics.md) | Neutral bootstrap development candidate with a nonresolving placeholder image; requires a reviewed build digest and runtime Shadow Hand cache. |
| [`byof-libero.yaml`](byof-libero.yaml) | [Qualification guide](../../docs/workbench/byof-libero.md) | Quarantined managed qualification path; requires an explicitly qualified candidate and operator-authorized use. |
| [`byof-lingbot-world.yaml`](byof-lingbot-world.yaml) | [Video/perception guide](../../docs/workbench/video-generation-byof.md) | Native camera-conditioned video generation on four B200 GPUs, with distributed execution evidence and fully decoded outputs. |
| [`byof-ltx2.yaml`](byof-ltx2.yaml) | [LTX guide](../../docs/workbench/ltx2.md) | LTX-2.5 video generation and FiftyOne curation; source and gated weights fetched at runtime |
| [`byof-maniskill.yaml`](byof-maniskill.yaml) | [BYOF CLI](../../docs/cli/byof.md) | OSS registry: ManiSkill pinned image + PickCube smoke |
| [`byof-mochi-1.yaml`](byof-mochi-1.yaml) | [Video/perception guide](../../docs/workbench/video-generation-byof.md) | Native B200 text-to-video with a pinned public runtime and runtime-fetched weights. |
| [`byof-mujoco-playground.yaml`](byof-mujoco-playground.yaml) | [BYOF CLI](../../docs/cli/byof.md) | OSS registry: MuJoCo Playground pinned image + Cartpole smoke |
| [`byof-open-dreamer.yaml`](byof-open-dreamer.yaml) | [Open Dreamer operations](../../skills/tools/open-dreamer/SKILL.md) | Open Dreamer multi-GPU tokenizer and dynamics training on Minecraft/VPT data → action-conditioned dream rollout and Rerun evidence |
| [`byof-openpi.yaml`](byof-openpi.yaml) | [Guide](../../docs/workbench/openpi-pi05-polaris.md) | OSS registry: OpenPI pi0.5 Polaris direct + WebSocket-served Franka joint-position inference on `B200:1`; runtime-only checkpoint and scoped Gemma gate |
| [`byof-robocasa.yaml`](byof-robocasa.yaml) | [BYOF CLI](../../docs/cli/byof.md) | OSS registry: RoboCasa pinned image + headless kitchen-task smoke |
| [`byof-robomimic.yaml`](byof-robomimic.yaml) | [Guide](../../docs/workbench/byof-robomimic.md) | Quarantined neutral candidate: plans real robomimic BC optimizer steps, disjoint held-out validation, exact-checkpoint reload, and held-out Lift PH low-dimensional action inference on `B200:1`; CUDA runtime use and live/public acceptance remain deferred |
| [`byof-robotwin.yaml`](byof-robotwin.yaml) | [Qualification guide](../../docs/workbench/byof-robotwin.md) | Neutral bootstrap candidate; CPU launcher submits an authorized RTX job only after the required payload lock is complete. |
| [`byof-sam2.1.yaml`](byof-sam2.1.yaml) | [Video/perception guide](../../docs/workbench/video-generation-byof.md) | GPU mask propagation on a hash-verified video → raw predictions and visualization; no ground-truth or robot-success claim. |
| [`byof-wan2.1-14b.yaml`](byof-wan2.1-14b.yaml) | [Video/perception guide](../../docs/workbench/video-generation-byof.md) | Native B200 text-to-video with pinned runtime and runtime-fetched weights. |
| [`byof-wan2.2-multigpu.yaml`](byof-wan2.2-multigpu.yaml) | [Wan guide](../../docs/workbench/wan2.2.md) | Wan 2.2 generation across four participating GPU ranks; MP4, topology, and Rerun evidence |
| [`byof-wan2.2.yaml`](byof-wan2.2.yaml) | [Frames, sampling steps and seed](../../docs/workbench/wan2.2.md#generate-a-longer-clip) | Wan 2.2 TI2V-5B on one RTX PRO 6000; decoded MP4 and verified Rerun evidence on both Wan routes |
| [`byof.yaml`](byof.yaml) | [General workflow guide](../../docs/workbench/npa-workflow-guide.md) | BYOF via `run_byof_repo.py` |
| [`habitat-sim-smoke.yaml`](habitat-sim-smoke.yaml) | [Qualification guide](../../docs/workbench/byof-habitat-sim.md) | Quarantined dedicated Habitat image: exact runtime-fetched Skokloster RGB/depth traversal, Bullet, and NVIDIA EGL on one STRICT-bound RTX PRO 6000 (never B200); image and live proof remain pending |

## Hosted inference and VLM evaluation

| Workflow | Guide | Purpose |
| --- | --- | --- |
| [`token-factory-batch-generate.yaml`](token-factory-batch-generate.yaml) | [Token Factory guide](../../docs/workbench/token-factory.md) | Hosted asynchronous batch text generation → generations JSONL |
| [`token-factory-caption.yaml`](token-factory-caption.yaml) | [Token Factory guide](../../docs/workbench/token-factory.md) | Hosted vision captioning; pass `--secret-env NEBIUS_TOKEN_FACTORY_KEY` |
| [`token-factory-cosmos-reason.yaml`](token-factory-cosmos-reason.yaml) | [Token Factory guide](../../docs/workbench/token-factory.md) | Hosted inference; pass `--secret-env NEBIUS_TOKEN_FACTORY_KEY` |
| [`token-factory-gate-loop.yaml`](token-factory-gate-loop.yaml) | [Token Factory guide](../../docs/workbench/token-factory.md) | Zero-GPU **runtime** gate loop: real early-exit + `goto` branch; submit with `--runtime` |
| [`token-factory-generate.yaml`](token-factory-generate.yaml) | [Token Factory guide](../../docs/workbench/token-factory.md) | Hosted inference; pass `--secret-env NEBIUS_TOKEN_FACTORY_KEY` |
| [`token-factory-parallel-fanout.yaml`](token-factory-parallel-fanout.yaml) | [Token Factory guide](../../docs/workbench/token-factory.md) | Zero-GPU **parallel** fan-out (JobGroup) + join barrier; submit with `--runtime` |
| [`token-factory-robot-sdg.yaml`](token-factory-robot-sdg.yaml) | [Robot SDG guide](../../docs/workbench/token-factory-robot-sdg.md) | Hosted scene plans → Fetch pick-and-place simulation → physics-checked RGB/action LeRobot dataset and video. |
| [`token-factory-sdg.yaml`](token-factory-sdg.yaml) | [Text SDG guide](../../docs/workbench/token-factory-sdg.md) | Hosted seeds → instruction/answer pairs → review and deduplication → JSONL dataset. |
| [`token-factory-trigger-watch.yaml`](token-factory-trigger-watch.yaml) | [Token Factory guide](../../docs/workbench/token-factory.md) | Wait for frames in an inbox prefix → hosted vision captioning |
| [`tokenfactory-cosmos-gate.yaml`](tokenfactory-cosmos-gate.yaml) | [Hosted/GPU cookbook](../../docs/workbench/cookbooks/tokenfactory-compute-combos.md) | Gate loop |
| [`tokenfactory-rollout-judge-combo.yaml`](tokenfactory-rollout-judge-combo.yaml) | [Hosted/GPU cookbook](../../docs/workbench/cookbooks/tokenfactory-compute-combos.md) | LeRobot GPU rollout → hosted VLM evaluation |
| [`tokenfactory-rollout-judge.yaml`](tokenfactory-rollout-judge.yaml) | [Hosted/GPU cookbook](../../docs/workbench/cookbooks/tokenfactory-compute-combos.md) | Reason → VLM chain |
| [`tokenfactory-scene-to-rollout-judge.yaml`](tokenfactory-scene-to-rollout-judge.yaml) | [Hosted/GPU cookbook](../../docs/workbench/cookbooks/tokenfactory-compute-combos.md) | Hosted scene reasoning → GPU policy rollout → hosted VLM evaluation against the plan |
| [`tokenfactory-train-triage.yaml`](tokenfactory-train-triage.yaml) | [Hosted/GPU cookbook](../../docs/workbench/cookbooks/tokenfactory-compute-combos.md) | LeRobot GPU policy training → hosted text triage of run artifacts |
| [`vlm-eval-benchmark.yaml`](vlm-eval-benchmark.yaml) | [VLM evaluation CLI](../../docs/cli/vlm-eval.md) | VLM benchmark |
| [`vlm-eval-loop.yaml`](vlm-eval-loop.yaml) | [Evaluation loop runbook](../../docs/workbench/cookbooks/vlm-eval-loop-runbook.md) | Self-hosted VLM rollout-set evaluation → aggregate task-success report |
| [`vlm-eval-single.yaml`](vlm-eval-single.yaml) | [VLM evaluation CLI](../../docs/cli/vlm-eval.md) | Self-hosted VLM eval |
| [`vlm-eval-token-factory.yaml`](vlm-eval-token-factory.yaml) | [VLM evaluation CLI](../../docs/cli/vlm-eval.md) | Hosted Token Factory VLM evaluation over a rollout prefix |

## Infrastructure and runtime examples

| Workflow | Guide | Purpose |
| --- | --- | --- |
| [`multi-node-probe.yaml`](multi-node-probe.yaml) | [Multi-node execution](../../docs/workbench/npa-workflow-guide.md) | Gang-scheduled multi-node stage with evidence from every rank |
| [`sim2real-envgen-shards.yaml`](sim2real-envgen-shards.yaml) | [General workflow guide](../../docs/workbench/npa-workflow-guide.md) | **DEMO ONLY** isolated envgen fan-out fixture |
| [`sim2real-two-step-agent.yaml`](sim2real-two-step-agent.yaml) | [General workflow guide](../../docs/workbench/npa-workflow-guide.md) | **DEMO ONLY** agent-generated two-state DSL fixture |
| [`sim2real-two-step.yaml`](sim2real-two-step.yaml) | [General workflow guide](../../docs/workbench/npa-workflow-guide.md) | **DEMO ONLY** two-state DSL fixture |
| [`sonic-b300-routing-evidence.yaml`](sonic-b300-routing-evidence.yaml) | [Cookbook](../../docs/workbench/cookbooks/sonic-b300-routing-evidence.md) | CPU-only, fail-closed explicit B300 routing evidence with a time-structured RRD |

The canonical Sim2Real pipeline is in [main/sim2real.yaml](../main/sim2real.yaml).
The small `sim2real-*` fixtures here do not substitute for a complete
robot-learning run. Use the [catalog commands](../README.md#commands) to
validate, plan, submit and inspect a selected workflow from the repository root.
