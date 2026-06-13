# Nebius Physical AI Documentation

This directory contains platform documentation and solution-specific docs for
Nebius Physical AI.

## Index

| Path | Purpose |
| --- | --- |
| [workbench/](workbench/) | Workbench solution docs, including getting started, cookbooks, and troubleshooting |
| [../npa/workflows/workbench/skypilot/README.md](../npa/workflows/workbench/skypilot/README.md) | **Workflow catalog** — find the right SkyPilot YAML by what you want to do |
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
| Salesperson or evaluator | [Workflow catalog](../npa/workflows/workbench/skypilot/README.md) to see what the platform runs |
| Customer running a first Workbench workload | [workbench/getting-started.md](workbench/getting-started.md) |
| Developer adding a solution | [architecture/solutions-model.md](architecture/solutions-model.md) |
| SDK integrator or agent author | [sdk/errors.md](sdk/errors.md) |
| Internal engineer triaging a failure | [cli-errors.md](cli-errors.md) |
| Operator running e2e tests | [testing/e2e-serverless.md](testing/e2e-serverless.md) |
