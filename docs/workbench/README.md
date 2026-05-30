# Workbench Documentation

This directory contains documentation for the `npa workbench` CLI, SDK, tools,
workflows, and operational runbooks.

## Index

| Path | Purpose |
| --- | --- |
| [getting-started.md](getting-started.md) | Fresh-clone onboarding path for install, credentials, first deploy, and BDD100K pipeline validation |
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
| Customer running their first Workbench workload | [getting-started.md](getting-started.md) |
| Operator reproducing a workload | [cookbooks/README.md](cookbooks/README.md) |
| SDK integrator or agent author | [../sdk/errors.md](../sdk/errors.md) |
| Internal engineer triaging a failure | [../cli-errors.md](../cli-errors.md) |
| Operator running e2e tests | [../testing/e2e-serverless.md](../testing/e2e-serverless.md) |
