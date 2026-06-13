# Workbench Documentation

This directory contains documentation for the `npa workbench` CLI, SDK, tools,
workflows, and operational runbooks.

## Index

| Path | Purpose |
| --- | --- |
| [getting-started.md](getting-started.md) | Fresh-clone onboarding path for install, credentials, and first Workbench runs |
| [guides/sim2real-workflow.md](guides/sim2real-workflow.md) | **13-stage VLM→RL sim-to-real loop** — operator quickstart (start here for production) |
| [guides/sim2real-data-contracts.md](guides/sim2real-data-contracts.md) | Canonical data types, JSON schemas, and S3 layout for the 13-stage loop |
| [guides/sim2real-customer-assets.md](guides/sim2real-customer-assets.md) | Customer uploads, stock vs BYO, scorecard |
| [guides/sim2real-architecture.md](guides/sim2real-architecture.md) | Control flow, K8s sibling jobs, fallbacks |
| [../../npa/workflows/workbench/skypilot/README.md](../../npa/workflows/workbench/skypilot/README.md) | **Workflow catalog** — find the right SkyPilot YAML by what you want to do |
| [cli-sdk-yaml-walkthrough.md](cli-sdk-yaml-walkthrough.md) | How to call any Workbench tool through the CLI, SDK, and SkyPilot YAML against the same service |
| [sim-to-real-quickstart.md](sim-to-real-quickstart.md) | Legacy one-command H100 proof run (`sim_to_real` module) — see [guides/sim2real-workflow.md](guides/sim2real-workflow.md) for the 13-stage loop |
| [../quickstart.md](../quickstart.md) | Full `npa` CLI quickstart |
| [../cli/README.md](../cli/README.md) | CLI command reference index |
| [../cli-errors.md](../cli-errors.md) | End-user CLI error formatting, exit codes, and JSON error output |
| [../sdk/errors.md](../sdk/errors.md) | Typed exceptions for programmatic SDK consumers and agents |
| [cookbooks/README.md](cookbooks/README.md) | Reproducibility cookbooks for specific workloads |
| [cookbooks/sim-to-real-pipeline.md](cookbooks/sim-to-real-pipeline.md) | Legacy `sim_to_real` YAML/CLI — superseded for production by [guides/sim2real-workflow.md](guides/sim2real-workflow.md) |
| [cookbooks/vlm-eval-loop-runbook.md](cookbooks/vlm-eval-loop-runbook.md) | Sim-to-real VLM-eval loop: self-hosted VLM serving, rollout scoring, and task-success reporting |
| [cookbooks/lerobot-gpu-benchmarks.md](cookbooks/lerobot-gpu-benchmarks.md) | Reproducing the May 2026 LeRobot GPU benchmark research |
| [troubleshooting/known-footguns.md](troubleshooting/known-footguns.md) | Known Workbench operational footguns and mitigations |
| [../testing/e2e-serverless.md](../testing/e2e-serverless.md) | E2E test conventions for serverless workloads |
| [../testing/e2e.md](../testing/e2e.md) | General E2E test conventions |

## Audience

| Reader | Start with |
| --- | --- |
| Salesperson or evaluator | [Workflow catalog](../../npa/workflows/workbench/skypilot/README.md) to see what the platform runs |
| Customer running their first Workbench workload | [getting-started.md](getting-started.md) |
| Anyone choosing between CLI, SDK, and YAML | [cli-sdk-yaml-walkthrough.md](cli-sdk-yaml-walkthrough.md) |
| Customer running the 13-stage VLM→RL sim-to-real loop | [guides/sim2real-workflow.md](guides/sim2real-workflow.md) + [guides/sim2real-data-contracts.md](guides/sim2real-data-contracts.md) |
| Customer running the first H100 sim-to-real proof | [sim-to-real-quickstart.md](sim-to-real-quickstart.md) (legacy module) |
| Operator reproducing a workload | [cookbooks/README.md](cookbooks/README.md) |
| SDK integrator or agent author | [../sdk/errors.md](../sdk/errors.md) |
| Internal engineer triaging a failure | [../cli-errors.md](../cli-errors.md) |
| Operator running e2e tests | [../testing/e2e-serverless.md](../testing/e2e-serverless.md) |
