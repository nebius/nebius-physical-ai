# Nebius Physical AI documentation

`npa` runs robotics and physical-AI tools on Nebius. Start with a workload,
prepare its project and compute, then inspect the result.

## Start here

| Task | Read |
| --- | --- |
| Run a first GPU workload | [Quickstart](quickstart.md) → [workload guides](workbench/guides/README.md) |
| Work with a coding agent | [First-run prompts](workbench/agent-first-run.md) |
| Install host tools | [Installation](install.md) |
| Connect a project and credentials | [Configuration](configuration.md) |
| Prepare Kubernetes and SkyPilot | [Workbench setup](workbench/getting-started.md) |
| Find a tool or pipeline | [Workbench docs](workbench/README.md) · [workflow catalog](../workflows/README.md) |

## Run, integrate, and inspect

| Task | Read |
| --- | --- |
| Author and submit YAML | [Workflow guide](workbench/npa-workflow-guide.md) · [toolRef catalog](workbench/npa-workflow-tool-catalog.md) |
| Use CLI, Python, or HTTP | [CLI reference](cli/README.md) · [SDK walkthrough](workbench/cli-sdk-yaml-walkthrough.md) · [SDK errors](sdk/errors.md) |
| Choose a direct deployment mode | [Runtime modes](workbench/runtime-modes.md) |
| Understand status and resume | [Run lifecycle](run-lifecycle.md) |
| View results in a browser | [Agent workbench](agent.md) · [Rerun shares](workbench/rerun-sharing.md) · [Foxglove export](workbench/foxglove-export.md) |
| Diagnose failures | [Troubleshooting](workbench/troubleshooting/known-footguns.md) · [CLI errors](cli-errors.md) |
| Remove owned resources | [Teardown](teardown.md) |

## Operate infrastructure

| Task | Read |
| --- | --- |
| Manage Kubernetes | [Kubernetes](workbench/kubernetes.md) · [GPU driver strategy](workbench/mk8s-gpu-driver-strategy.md) |
| Configure workflow scheduling | [SkyPilot setup](orchestration/skypilot-setup.md) |
| Manage fleets or Slurm | [Cluster backends](cluster-backends.md) · [Fleet storage verification](fleet-storage-verification.md) · [RTX MIG](fleet-rtx-pro-6000-mig.md) |
| Choose an image and GPU | [Public image catalog](workbench/container-image-catalog.md) · [compatibility matrix](workbench/image-gpu-compatibility-matrix.md) |
| Reuse model downloads | [Model-weight cache](workbench/model-weight-cache.md) |
| Use preemptible VMs | [Preemptible capacity](workbench/preemptible-vms.md) |
| Reproduce a workload | [Cookbooks](workbench/cookbooks/README.md) |

## Contribute and verify

| Task | Read |
| --- | --- |
| Add a tool or integration | [Contributing](../CONTRIBUTING.md) · [OSS onboarding ladder](architecture/oss-onboarding-ladder.md) |
| Understand the platform | [Contributor context](architecture/contributor-context.md) · [solutions model](architecture/solutions-model.md) · [CLI namespaces](architecture/cli-namespaces.md) |
| Package a container | [Container contract](workbench/container-packaging.md) · [image reproducibility](security/image-reproducibility.md) |
| Run local and live checks | [Package test commands](../npa/README.md#developing-and-testing-npa) · [E2E](testing/e2e.md) · [serverless E2E](testing/e2e-serverless.md) · [daily dev VM](testing/dev-vm-daily.md) |
| Verify release quality | [Golden evals](security/container-golden-evals.md) · [merge security gate](security/merge-security-gate.md) · [releasing](releasing.md) |

Dated benchmark and audit pages record the stated image, GPU, and test scope.
Use the current tool guide and compatibility matrix when preparing a new run.
