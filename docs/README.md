# Nebius Physical AI Documentation

This directory contains platform documentation and solution-specific docs for
Nebius Physical AI.

## Index

| Path | Purpose |
| --- | --- |
| [hackathon-cosmos3-reasoner.md](hackathon-cosmos3-reasoner.md) | **Hackathon quickstart** — copy-paste path to the serverless Cosmos3 reasoner (Token Factory), no GPU/VM |
| [hackathon-isaac-token-factory.md](hackathon-isaac-token-factory.md) | **Hackathon combo** — Isaac Lab Franka sim frames + Token Factory reasoner (workflow + SDK example) |
| [workbench/](workbench/) | Workbench solution docs, including getting started, cookbooks, and troubleshooting |
| [workbench/kubernetes.md](workbench/kubernetes.md) | User setup and operational guide for running Workbench on managed Kubernetes |
| [../npa/workflows/workbench/npa-workflows/README.md](../npa/workflows/workbench/npa-workflows/README.md) | **Workflow catalog** — find the right `npa.workflow` spec by what you want to do |
| [architecture/solutions-model.md](architecture/solutions-model.md) | Platform model for adding and maintaining solutions |
| [architecture/cli-namespaces.md](architecture/cli-namespaces.md) | CLI namespace conventions |
| [quickstart.md](quickstart.md) | Full `npa` CLI quickstart (macOS, Linux, WSL2 install blocks) |
| [cli/README.md](cli/README.md) | CLI command reference index |
| [cli-errors.md](cli-errors.md) | End-user CLI error formatting, exit codes, and JSON error output |
| [sdk/errors.md](sdk/errors.md) | Typed exceptions for programmatic SDK consumers and agents |
| [testing/e2e-serverless.md](testing/e2e-serverless.md) | E2E test conventions for serverless workloads |
| [testing/e2e.md](testing/e2e.md) | General E2E test conventions |

## Audience

| Reader | Start with |
| --- | --- |
| Salesperson or evaluator | [Workflow catalog](../npa/workflows/workbench/npa-workflows/README.md) to see what the platform runs |
| Customer running a first Workbench workload | [workbench/getting-started.md](workbench/getting-started.md) |
| Operator connecting Workbench to Kubernetes | [workbench/kubernetes.md](workbench/kubernetes.md) |
| Developer adding a solution | [architecture/solutions-model.md](architecture/solutions-model.md) |
| SDK integrator or agent author | [sdk/errors.md](sdk/errors.md) |
| Internal engineer triaging a failure | [cli-errors.md](cli-errors.md) |
| Operator running e2e tests | [testing/e2e-serverless.md](testing/e2e-serverless.md) |
