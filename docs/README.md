# Nebius Physical AI Documentation

This directory contains platform documentation and solution-specific docs for
Nebius Physical AI.

## Index

| Path | Purpose |
| --- | --- |
| [workbench/](workbench/) | Workbench solution docs, including getting started, cookbooks, and troubleshooting |
| [workbench/kubernetes.md](workbench/kubernetes.md) | User setup and operational guide for running Workbench on managed Kubernetes |
| [architecture/solutions-model.md](architecture/solutions-model.md) | Platform model for adding and maintaining solutions |
| [architecture/cli-namespaces.md](architecture/cli-namespaces.md) | CLI namespace conventions |
| [quickstart.md](quickstart.md) | Full `npa` CLI quickstart |
| [cli/README.md](cli/README.md) | CLI command reference index |
| [cli-errors.md](cli-errors.md) | End-user CLI error formatting, exit codes, and JSON error output |
| [sdk/errors.md](sdk/errors.md) | Typed exceptions for programmatic SDK consumers and agents |
| [testing/e2e-serverless.md](testing/e2e-serverless.md) | E2E test conventions for serverless workloads |
| [testing/e2e.md](testing/e2e.md) | General E2E test conventions |

## Audience

| Reader | Start with |
| --- | --- |
| Customer running a first Workbench workload | [workbench/getting-started.md](workbench/getting-started.md) |
| Operator connecting Workbench to Kubernetes | [workbench/kubernetes.md](workbench/kubernetes.md) |
| Developer adding a solution | [architecture/solutions-model.md](architecture/solutions-model.md) |
| SDK integrator or agent author | [sdk/errors.md](sdk/errors.md) |
| Internal engineer triaging a failure | [cli-errors.md](cli-errors.md) |
| Operator running e2e tests | [testing/e2e-serverless.md](testing/e2e-serverless.md) |
