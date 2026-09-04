# Workbench Documentation

This directory contains documentation for the `npa workbench` CLI, SDK, tools,
workflows, and operational runbooks.

## Index

| Path | Purpose |
| --- | --- |
| [getting-started.md](getting-started.md) | Fresh-clone onboarding path for install, credentials, and first Workbench runs |
| [container-image-catalog.md](container-image-catalog.md) | Verified public GHCR image names, exact published tags, build dates, and capabilities |
| [container-packaging.md](container-packaging.md) | Container packaging tiers, security baseline, and feature exposure contract |
| [isaac-lab-3.md](isaac-lab-3.md) | Isaac Lab 3 beta pin, payload-clean runtime bootstrap, hardened RL sweep, and generation 2 comparison method |
| [model-weight-cache.md](model-weight-cache.md) | Durable cache for model weights and reviewed SDKs the public images do not bake, so a second run is a cache hit |
| [rerun-sharing.md](rerun-sharing.md) | Time-boxed Rerun browser shares, one-time least-privilege bucket CORS setup, and native local fallback |
| [cosmos3-b200-checkpoint-evaluation-20260814.md](cosmos3-b200-checkpoint-evaluation-20260814.md) | Reserved-B200, 72-image Cosmos3 checkpoint evaluation and three-seed investment decision |
| [cosmos3-ray-serve.md](cosmos3-ray-serve.md) | Persistent Cosmos3-Nano serving through native Ray Serve batching with durable S3 outputs |
| [leisaac-teleoperation.md](leisaac-teleoperation.md) | Capability-gated LeIsaac agent tab, RT-core launch, keyboard teleoperation, and cleanup |
| [../architecture/oss-onboarding-ladder.md](../architecture/oss-onboarding-ladder.md) | OSS → BYOF → workflow → first-class tool promotion ladder |
| [npa-workflow-tool-catalog.md](npa-workflow-tool-catalog.md) | `toolRef` catalog for declarative `npa.workflow` specs |
| [agent-workflow-operations.md](agent-workflow-operations.md) | Provider-neutral, bounded NPA operations for agents authoring and running Workbench workflows |
| [kubernetes.md](kubernetes.md) | User setup and operational checklist for running Workbench services and SkyPilot workflows on Kubernetes |
| [mk8s-gpu-driver-strategy.md](mk8s-gpu-driver-strategy.md) | Managed GPU driver modes, fail-closed post-deploy health validation, recipe compatibility, and existing-pool migration |
| [../../npa/workflows/workbench/npa-workflows/README.md](../../npa/workflows/workbench/npa-workflows/README.md) | **Workflow catalog** — find the right `npa.workflow` spec by what you want to do |
| [oss-solution-catalog.md](oss-solution-catalog.md) | OSS Physical AI registry candidates with pinned refs, cloud-fit notes, and E2E gates |
| [cli-sdk-yaml-walkthrough.md](cli-sdk-yaml-walkthrough.md) | How to call any Workbench tool through the CLI, SDK, and SkyPilot YAML against the same service |
| [guides/physical-ai-data-factory-deploy.md](guides/physical-ai-data-factory-deploy.md) | **Physical AI Data Factory** — copy-paste runbook (stage input frames, submit) for the annotate → Cosmos augment → curate → visualize blueprint |
| [../quickstart.md](../quickstart.md) | Full `npa` CLI quickstart |
| [../cli/README.md](../cli/README.md) | CLI command reference index |
| [../cli-errors.md](../cli-errors.md) | End-user CLI error formatting, exit codes, and JSON error output |
| [../sdk/errors.md](../sdk/errors.md) | Typed exceptions for programmatic SDK consumers and agents |
| [cookbooks/README.md](cookbooks/README.md) | Reproducibility cookbooks for specific workloads |
| [cookbooks/vlm-eval-loop-runbook.md](cookbooks/vlm-eval-loop-runbook.md) | Sim-to-real VLM-eval loop: self-hosted VLM serving, rollout scoring, and task-success reporting |
| [cookbooks/lerobot-gpu-benchmarks.md](cookbooks/lerobot-gpu-benchmarks.md) | Reproducing the May 2026 LeRobot GPU benchmark research |
| [troubleshooting/known-footguns.md](troubleshooting/known-footguns.md) | Known Workbench operational footguns and mitigations |
| [../testing/e2e-serverless.md](../testing/e2e-serverless.md) | E2E test conventions for serverless workloads |
| [../testing/e2e.md](../testing/e2e.md) | General E2E test conventions |

## Audience

| Reader | Start with |
| --- | --- |
| Salesperson or evaluator | [Workflow catalog](../../npa/workflows/workbench/npa-workflows/README.md) to see what the platform runs |
| Customer running their first Workbench workload | [getting-started.md](getting-started.md) |
| Customer or operator using managed Kubernetes | [kubernetes.md](kubernetes.md) |
| Anyone choosing between CLI, SDK, and YAML | [cli-sdk-yaml-walkthrough.md](cli-sdk-yaml-walkthrough.md) |
| Partner onboarding an OSS repo | [../architecture/oss-onboarding-ladder.md](../architecture/oss-onboarding-ladder.md) |
| Engineer packaging or hardening a container | [container-packaging.md](container-packaging.md) |
| Operator watching the same weights download every run | [model-weight-cache.md](model-weight-cache.md) |
| Customer running the first H100 sim-to-real proof | [guides/sim2real-workflow.md](guides/sim2real-workflow.md) |
| Operator reproducing a workload | [cookbooks/README.md](cookbooks/README.md) |
| SDK integrator or agent author | [../sdk/errors.md](../sdk/errors.md) |
| Automation agent operating a workflow | [agent-workflow-operations.md](agent-workflow-operations.md) |
| Internal engineer triaging a failure | [../cli-errors.md](../cli-errors.md) |
| Operator running e2e tests | [../testing/e2e-serverless.md](../testing/e2e-serverless.md) |
