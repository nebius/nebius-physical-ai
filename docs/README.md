# Nebius Physical AI Documentation

`npa` is the CLI and Python package for Nebius Physical AI. **Workbench** is its
primary solution: tools and declarative workflows for data curation, simulation,
synthetic data, policy training, and evaluation on Nebius.


[Configuration](configuration.md) covers credentials, project storage, and SSO.
Use the [coding-agent first run](workbench/agent-first-run.md) for guided setup
and PAIDF + Cosmos 3 execution.

## Start with your task

| I want to… | Start here | Continue with |
| --- | --- | --- |
| Run my first GPU workload | [Quickstart](quickstart.md) for installation, project setup, and credential checks | [Workbench guides](workbench/guides/README.md) to choose a workload and inspect its result |
| Find an existing pipeline | [Workflow catalog](../npa/workflows/workbench/npa-workflows/README.md) | [Workflow guide](workbench/npa-workflow-guide.md) for validation, planning, and submission |
| Prepare compute for my workload | [Workbench setup](workbench/getting-started.md) after the quickstart | [Managed Kubernetes](workbench/kubernetes.md) for cluster-backed runs |
| Call a tool from a shell or Python | [CLI reference](cli/README.md) and [CLI / SDK walkthrough](workbench/cli-sdk-yaml-walkthrough.md) | [SDK errors](sdk/errors.md); check the selected tool's supported modes |
| Inspect or recover a run | [Run lifecycle](run-lifecycle.md) | [Known failure modes](workbench/troubleshooting/known-footguns.md) and [safe teardown](teardown.md) |
| Use the browser workbench and viewers | [NPA agent](agent.md) | [Rerun sharing](workbench/rerun-sharing.md) |
| Add a tool, workflow, or integration | [Contributing](../CONTRIBUTING.md) | [OSS onboarding ladder](architecture/oss-onboarding-ladder.md) and [container contract](workbench/container-packaging.md) |

The [Workbench documentation index](workbench/README.md) collects tool guides,
workflow references, and operator runbooks. For platform-specific installation
details, use [Install npa](install.md).

## Index

| Path | Purpose |
| --- | --- |
| [quickstart.md](quickstart.md) | Start here: install `npa`, configure Nebius, check access, and run a GPU workload |
| [install.md](install.md) | Platform-specific installation and optional dependencies |
| [workbench/guides/physical-ai-data-factory-deploy.md](workbench/guides/physical-ai-data-factory-deploy.md) | **Physical AI Data Factory** — copy-paste quickstart to stage input frames and run the annotate → Cosmos augment → curate → visualize blueprint |
| [workbench/](workbench/) | Workbench solution docs, including getting started, cookbooks, and troubleshooting |
| [workbench/kubernetes.md](workbench/kubernetes.md) | User setup and operational guide for running Workbench on managed Kubernetes |
| [workbench/cosmos3-generate.md](workbench/cosmos3-generate.md) | Cosmos 3 generation (`npa-cosmos3`) — build, run via CLI/SDK/workflow, and the runtime-credential posture that keeps weights out of the image |
| [workbench/cosmos3-b200-checkpoint-evaluation-20260814.md](workbench/cosmos3-b200-checkpoint-evaluation-20260814.md) | Reserved-B200 Cosmos3 still-image checkpoint benchmark, blind three-seed review, and recommendation |
| [workbench/cosmos3-super-serving.md](workbench/cosmos3-super-serving.md) | Cosmos3-Super serving (`npa-cosmos3-serving`), an 8-GPU single-node endpoint: build, run, readiness window, and guardrail posture |
| [../npa/workflows/workbench/npa-workflows/README.md](../npa/workflows/workbench/npa-workflows/README.md) | **Workflow catalog** — find the right `npa.workflow` spec by what you want to do |
| [architecture/solutions-model.md](architecture/solutions-model.md) | Platform model for adding and maintaining solutions |
| [architecture/cli-namespaces.md](architecture/cli-namespaces.md) | CLI namespace conventions |
| [cluster-backends.md](cluster-backends.md) | Shared Managed Kubernetes and soperator backend architecture, fleet specs, state ownership, and safe teardown |
| [run-lifecycle.md](run-lifecycle.md) | What `workflow submit` checks before it spends a GPU-hour: gates, run identity, restart safety, and status semantics |
| [agent.md](agent.md) | The self-hosted `npa agent` browser workbench VM — deploy, what it can see, and its artifact paging contract |
| [teardown.md](teardown.md) | Stop spend safely: the ordered teardown sequence, what each phase guards, and receipt-based recovery |
| [cli/README.md](cli/README.md) | CLI command reference index |
| [cli-errors.md](cli-errors.md) | End-user CLI error formatting, exit codes, and JSON error output |
| [sdk/errors.md](sdk/errors.md) | Typed exceptions for programmatic SDK consumers and agents |
| [testing/e2e-serverless.md](testing/e2e-serverless.md) | E2E test conventions for serverless workloads |
| [testing/e2e.md](testing/e2e.md) | General E2E test conventions |
| [testing/dev-vm-daily.md](testing/dev-vm-daily.md) | Daily tests run on the dev VM over SSH from GitHub Actions |
| [hackathon-isaac-token-factory.md](hackathon-isaac-token-factory.md) | Isaac Lab Franka simulation frames + Token Factory reasoner (workflow + SDK example) |
| [hackathon-cosmos3-reasoner.md](hackathon-cosmos3-reasoner.md) | Hosted Cosmos3 reasoning through Token Factory |

## Audience

| Reader | Start with |
| --- | --- |
| Salesperson or evaluator | [Workflow catalog](../npa/workflows/workbench/npa-workflows/README.md) to see what the platform runs |
| Customer running a first Workbench workload | [quickstart.md](quickstart.md), then [Workbench guides](workbench/guides/README.md) |
| Operator connecting Workbench to Kubernetes | [workbench/kubernetes.md](workbench/kubernetes.md) |
| Developer adding a solution | [Contributing](../CONTRIBUTING.md), then [architecture/solutions-model.md](architecture/solutions-model.md) |
| SDK integrator or agent author | [CLI / SDK walkthrough](workbench/cli-sdk-yaml-walkthrough.md), then [sdk/errors.md](sdk/errors.md) |
| Internal engineer triaging a failure | [cli-errors.md](cli-errors.md) |
| Operator running e2e tests | [testing/e2e-serverless.md](testing/e2e-serverless.md) |
