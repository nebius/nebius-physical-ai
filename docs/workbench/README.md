# Workbench documentation

[All docs](../README.md) · [Quickstart](../quickstart.md) · [CLI reference](../cli/workbench.md)

Use `npa workbench <tool> <command>` for a capability and
`npa workbench workflow` for a pipeline. Tools exchange artifacts through S3;
Python and HTTP access follow each tool's documented contract.

## Start and operate a run

| Task | Guide |
| --- | --- |
| Choose a workload | [Robot and workflow guides](guides/README.md) · [workflow catalog](../../workflows/README.md) |
| Use your coding agent | [First-run prompts](agent-first-run.md) · [workflow operations](agent-workflow-operations.md) |
| Prepare the runtime | [Workbench setup](getting-started.md) · [Kubernetes](kubernetes.md) · [direct runtime modes](runtime-modes.md) |
| Author and submit | [Workflow guide](npa-workflow-guide.md) · [toolRef catalog](npa-workflow-tool-catalog.md) |
| Integrate from Python or HTTP | [CLI / SDK walkthrough](cli-sdk-yaml-walkthrough.md) · [SDK errors](../sdk/errors.md) |
| Inspect or recover | [Run lifecycle](../run-lifecycle.md) · [troubleshooting](troubleshooting/known-footguns.md) · [CLI errors](../cli-errors.md) |
| Finish | [Teardown](../teardown.md) |

## Generation and scenes

| Capability | Guide |
| --- | --- |
| Cosmos 3 batch generation | [Generate](cosmos3-generate.md) · [access preflight](cosmos3-access-preflight.md) |
| Cosmos 3 persistent serving | [Nano with Ray Serve](cosmos3-ray-serve.md) · [Super serving](cosmos3-super-serving.md) |
| Video augmentation and dataset production | [PAIDF + Cosmos 3](guides/paidf-cosmos3.md) · [Cosmos Transfer data factory](guides/physical-ai-data-factory-deploy.md) · [concepts](guides/physical-ai-data-factory.md) · [campaign reuse](guides/paidf-campaign-reuse.md) |
| Scene reconstruction | [NuRec](guides/neural-reconstruction.md) · [living-lab fan-out](guides/living-lab-nurec-fanout.md) |
| USD object preparation | [Content Agents](content-agents.md) |
| Other video models | [Wan 2.2](wan2.2.md) · [LTX-2](ltx2.md) |

## Robotics and simulation

| Capability | Guide |
| --- | --- |
| Robot policy walkthroughs | [Franka / Genesis](guides/franka-pick-and-place-genesis.md) · [PushT SDK smoke](guides/pusht-sim-to-real.md) · [Reachy 2 / LeRobot](guides/reachy2-lerobot-policy.md) |
| Locomotion | [G1 / SONIC](guides/g1-humanoid-walk-sonic.md) · [quadruped / Isaac Lab](guides/quadruped-isaac-lab.md) |
| GR00T fine-tuning | [GR00T N1.7](cookbooks/groot-1-7-training.md) |
| OpenPI policy training | [Pi0.5 / Polaris](openpi-pi05-polaris.md) |
| Simulation-to-policy pipeline | [Sim2Real runbook](guides/sim2real-workflow.md) · [data contracts](guides/sim2real-data-contracts.md) · [customer assets](guides/sim2real-customer-assets.md) · [robot spec](guides/sim2real-robot-spec.md) |
| Browser teleoperation | [LeIsaac](leisaac-teleoperation.md) · [latency measurement](guides/leisaac-transport-latency.md) |
| Motion planning | [cuRobo](curobo.md) |
| Isaac Lab versions | [Isaac Lab 3](isaac-lab-3.md) |
| Policy evaluation in Isaac Lab | [Isaac Arena](isaac-arena.md) |

## Data and evaluation

| Capability | Guide |
| --- | --- |
| Curation, vector search, and detection training | [BDD100K pipeline](cookbooks/bdd100k-pipeline.md) · [LanceDB search](cookbooks/lancedb-vector-search.md) · [LanceDB deployment](cookbooks/lancedb-deploy-runbook.md) |
| Hosted captioning, generation, and reasoning | [Token Factory](token-factory.md) · [cloud composition](composing-cloud-and-token-factory.md) |
| Evaluate rollouts with a VLM | [VLM evaluation loop](cookbooks/vlm-eval-loop-runbook.md) |
| View and share artifacts | [Rerun](rerun-sharing.md) · [Foxglove / MCAP](foxglove-export.md) · [browser workbench](../agent.md) |
| Autonomous-driving inference | [Alpamayo 2 Super](alpamayo2-super.md) |
| Native Ray jobs, training, and serving | [Ray](ray.md) |

## Access, images, and operations

| Task | Guide |
| --- | --- |
| Configure credentials | [Project configuration](../configuration.md) · [Hugging Face](huggingface-token.md) · [NGC](ngc-api-key.md) · [Token Factory key](token-factory-key.md) |
| Select images | [Public catalog](container-image-catalog.md) · [GPU compatibility](image-gpu-compatibility-matrix.md) · [SONIC variants](sonic-image-catalog.md) |
| Use Blackwell | [B200 / B300](blackwell-datacenter-image-compatibility.md) · [RTX PRO 6000](sm120-image-catalog.md) |
| Configure nodes and caches | [GPU driver strategy](mk8s-gpu-driver-strategy.md) · [model-weight cache](model-weight-cache.md) · [preemptible VMs](preemptible-vms.md) |
| Reproduce benchmarks and demos | [Cookbooks](cookbooks/README.md) · [validation scope](solutions-validation.md) |
| Add or package a solution | [Contributing](../../CONTRIBUTING.md) · [containerized solutions](contributing-a-containerized-solution.md) · [OSS catalog](oss-solution-catalog.md) · [packaging contract](container-packaging.md) |

Inspect the selected guide's actual output artifacts after the run reaches a
terminal state. A plan, successful status response, or historical benchmark
alone does not establish a new run's result.
