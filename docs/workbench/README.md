# Workbench Documentation

Workbench is the primary solution in Nebius Physical AI. Use
`npa workbench <tool> <command>` to run individual capabilities, and
`npa workbench workflow` to compose them into pipelines. Tools exchange data
through object storage. Python wrappers and service APIs are available according
to each tool's documented contract.

## From a first run to an inspected result

1. [Install and configure npa](../quickstart.md), including credentials and
   model-access checks. Use [platform installation details](../install.md) when
   needed.
2. Choose a GPU workload from the [guides](guides/README.md) or the
   [workflow catalog](../../workflows/README.md).
   Follow that workload's input, model, and GPU requirements.
3. Prepare its runtime using [Workbench setup](getting-started.md) and, for
   cluster-backed runs, [managed Kubernetes](kubernetes.md).
4. [Validate, plan, and submit the workflow](npa-workflow-guide.md). The
   [run lifecycle](../run-lifecycle.md) explains preflight gates, run identity,
   status, and restart behavior.
5. Inspect the outputs described by the selected guide.
   `npa workbench workflow artifacts` lists durable run outputs. For a successful PAIDF run,
   `load-artifact` retries the final viewer load without rerunning stages. The
   [agent viewer](../agent.md) and [Rerun sharing](rerun-sharing.md) provide
   viewing paths for supported artifacts.
6. Use [troubleshooting](troubleshooting/known-footguns.md) to resolve failures,
   then follow [safe teardown](../teardown.md) for resources you own.

For Python or HTTP integration, start with the
[CLI / SDK walkthrough](cli-sdk-yaml-walkthrough.md). To extend Workbench, read
[Contributing](../../CONTRIBUTING.md) and the
[OSS onboarding ladder](../architecture/oss-onboarding-ladder.md).

## Index

| Path | Purpose |
| --- | --- |
| [getting-started.md](getting-started.md) | Runtime and storage setup after the platform quickstart |
| [guides/README.md](guides/README.md) | Choose a robot, generation, reconstruction, or data workflow |
| [npa-workflow-guide.md](npa-workflow-guide.md) | Author, validate, plan, submit, and resume declarative workflows |
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
| [../../workflows/README.md](../../workflows/README.md) | **Workflow catalog** — find the right `npa.workflow` spec by what you want to do |
| [oss-solution-catalog.md](oss-solution-catalog.md) | OSS Physical AI registry candidates with pinned refs, cloud-fit notes, and E2E gates |
| [cli-sdk-yaml-walkthrough.md](cli-sdk-yaml-walkthrough.md) | Detection-training CLI, Python, and HTTP examples; check each tool's supported surfaces |
| [guides/physical-ai-data-factory-deploy.md](guides/physical-ai-data-factory-deploy.md) | **Physical AI Data Factory** — copy-paste runbook (stage input frames, submit) for the annotate → Cosmos augment → curate → visualize blueprint |
| [../quickstart.md](../quickstart.md) | Full `npa` CLI quickstart |
| [../cli/README.md](../cli/README.md) | CLI command reference index |
| [../cli-errors.md](../cli-errors.md) | End-user CLI error formatting, exit codes, and JSON error output |
| [../sdk/errors.md](../sdk/errors.md) | Typed exceptions for programmatic SDK consumers and agents |
| [../run-lifecycle.md](../run-lifecycle.md) | Run identity, preflight checks, status, and restart safety |
| [../teardown.md](../teardown.md) | Cancel and clean up owned resources in the required order |
| [../../CONTRIBUTING.md](../../CONTRIBUTING.md) | Implementation patterns, registration, documentation, and validation for contributors |
| [cookbooks/README.md](cookbooks/README.md) | Reproducibility cookbooks for specific workloads |
| [cookbooks/vlm-eval-loop-runbook.md](cookbooks/vlm-eval-loop-runbook.md) | Sim-to-real VLM-eval loop: self-hosted VLM serving, rollout scoring, and task-success reporting |
| [cookbooks/lerobot-gpu-benchmarks.md](cookbooks/lerobot-gpu-benchmarks.md) | Reproducing the May 2026 LeRobot GPU benchmark research |
| [troubleshooting/known-footguns.md](troubleshooting/known-footguns.md) | Known Workbench operational footguns and mitigations |
| [../testing/e2e-serverless.md](../testing/e2e-serverless.md) | E2E test conventions for serverless workloads |
| [../testing/e2e.md](../testing/e2e.md) | General E2E test conventions |

## Audience

| Reader | Start with |
| --- | --- |
| Salesperson or evaluator | [Workflow catalog](../../workflows/README.md) to see what the platform runs |
| Customer running their first Workbench workload | [Quickstart](../quickstart.md), then [guides](guides/README.md) |
| Customer or operator using managed Kubernetes | [kubernetes.md](kubernetes.md) |
| Anyone choosing between CLI, SDK, and YAML | [cli-sdk-yaml-walkthrough.md](cli-sdk-yaml-walkthrough.md) |
| Partner onboarding an OSS repo | [../architecture/oss-onboarding-ladder.md](../architecture/oss-onboarding-ladder.md) |
| Engineer packaging or hardening a container | [container-packaging.md](container-packaging.md) |
| Operator watching the same weights download every run | [model-weight-cache.md](model-weight-cache.md) |
| Customer running the compositional Sim2Real workflow | [guides/sim2real-workflow.md](guides/sim2real-workflow.md) |
| Operator reproducing a workload | [cookbooks/README.md](cookbooks/README.md) |
| SDK integrator or agent author | [cli-sdk-yaml-walkthrough.md](cli-sdk-yaml-walkthrough.md), then [../sdk/errors.md](../sdk/errors.md) |
| Automation agent operating a workflow | [agent-workflow-operations.md](agent-workflow-operations.md) |
| Internal engineer triaging a failure | [../cli-errors.md](../cli-errors.md) |
| Operator running e2e tests | [../testing/e2e-serverless.md](../testing/e2e-serverless.md) |
