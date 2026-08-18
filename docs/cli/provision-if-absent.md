# `npa provision-if-absent`

## Command Tree

```text
Usage: npa provision-if-absent [OPTIONS] COMMAND [ARGS]...

Ensure configured Kubernetes and S3 runtime resources exist.

Options
--project  <str>  Project alias from ~/.npa/config.yaml.
--cluster-name  <str>  Cluster profile/context name. [default: npa-cluster]
--terraform-dir  <path>  Terraform cluster directory.
--kubeconfig  <path>  Dedicated kubeconfig path.
--context  <str>  Kubeconfig context name.
--skip-k8s  Do not ensure Kubernetes.
--skip-s3  Do not ensure S3.
--validate  --skip-validate  Run post-apply Kubernetes validation. [default: validate]
--sky-smoke  --skip-sky-smoke  Run a SkyPilot GPU smoke task. [default: skip-sky-smoke]
--gpu-nodes  <int>  Number of GPU nodes, matching `npa cluster up`. -1 keeps the configured value. [default: -1]
--cpu-nodes  <int>  Number of CPU nodes, matching `npa cluster up`. -1 keeps the configured value. [default: -1]
--cpu-platform  <str>  CPU node platform, matching `npa cluster up`.
--cpu-preset  <str>  CPU node preset, matching `npa cluster up`.
--gpu-platform  <str>  GPU node platform, matching `npa cluster up`.
--gpu-preset  <str>  GPU node preset, matching `npa cluster up`.
--gpu-driver-mode  <str>  GPU driver strategy (auto, managed-image, or operator), matching `npa cluster up`.
--managed-driver-preset  <str>  Nebius managed driver preset, matching `npa cluster up`.
--allow-unsafe-nvswitch-operator  --deny-unsafe-nvswitch-operator  Explicit diagnostic acknowledgement for operator mode on NVSwitch systems.
--gpu-health-stabilization-seconds  <int>  Required stable GPU-health interval, matching `npa cluster up`. [default: 120]
--gpu-health-timeout-minutes  <int>  GPU/MIG health deadline, matching `npa cluster up`. [default: 60]
--gpu-cuda-smoke  --skip-gpu-cuda-smoke  Run CUDA vectorAdd on every requested GPU node. [default: gpu-cuda-smoke]
--gpu-cuda-smoke-image  <str>  Container image for CUDA vectorAdd validation.
    [default: nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda12.5.0-ubuntu22.04]
--mig  --no-mig  Enable the same pinned RTX PRO 6000 MIG policy as fleet. [default: no-mig]
--mig-strategy  <str>  [default: mixed]
--mig-config  <str>  [default: all-balanced]
--capacity-block-group  <str>  Runtime-only strict GPU capacity block selector.
--preemptible  --on-demand  Run the GPU node group as preemptible, matching `npa cluster up`. This changes the capacity pool but not hard
    instance/disk/IP quotas; a reclaim stops the node mid-run.
--dry-run  Resolve settings and print intended actions only.
--timeout  <int>  Terraform apply timeout in minutes. [default: 120]
--accelerator  <str>  Requested SkyPilot accelerator (for example RTXPRO6000:1) to gate readiness.
--gpu-readiness-timeout  <float>  Seconds to wait for SkyPilot GPU discovery without deleting capacity. [default: 600.0]
--gpu-readiness-poll-interval  <float>  Seconds between SkyPilot GPU discovery checks. [default: 10.0]
--sky-bin  <str>  Pinned SkyPilot executable.
--output-format  <text|json>  Output format. [default: text]
--help  Show this message and exit.
```

## Options

| Option | Description |
| --- | --- |
| `--project` | <str>  Project alias from ~/.npa/config.yaml. |
| `--cluster-name` | <str>  Cluster profile/context name. [default: npa-cluster] |
| `--terraform-dir` | <path>  Terraform cluster directory. |
| `--kubeconfig` | <path>  Dedicated kubeconfig path. |
| `--context` | <str>  Kubeconfig context name. |
| `--skip-k8s` | Do not ensure Kubernetes. |
| `--skip-s3` | Do not ensure S3. |
| `--validate` | --skip-validate  Run post-apply Kubernetes validation. [default: validate] |
| `--sky-smoke` | --skip-sky-smoke  Run a SkyPilot GPU smoke task. [default: skip-sky-smoke] |
| `--gpu-nodes` | <int>  Number of GPU nodes, matching `npa cluster up`. -1 keeps the configured value. [default: -1] |
| `--cpu-nodes` | <int>  Number of CPU nodes, matching `npa cluster up`. -1 keeps the configured value. [default: -1] |
| `--cpu-platform` | <str>  CPU node platform, matching `npa cluster up`. |
| `--cpu-preset` | <str>  CPU node preset, matching `npa cluster up`. |
| `--gpu-platform` | <str>  GPU node platform, matching `npa cluster up`. |
| `--gpu-preset` | <str>  GPU node preset, matching `npa cluster up`. |
| `--gpu-driver-mode` | <str>  GPU driver strategy (auto, managed-image, or operator), matching `npa cluster up`. |
| `--managed-driver-preset` | <str>  Nebius managed driver preset, matching `npa cluster up`. |
| `--allow-unsafe-nvswitch-operator` | --deny-unsafe-nvswitch-operator  Explicit diagnostic acknowledgement for operator mode on NVSwitch systems. |
| `--gpu-health-stabilization-seconds` | <int>  Required stable GPU-health interval, matching `npa cluster up`. [default: 120] |
| `--gpu-health-timeout-minutes` | <int>  GPU/MIG health deadline, matching `npa cluster up`. [default: 60] |
| `--gpu-cuda-smoke` | --skip-gpu-cuda-smoke  Run CUDA vectorAdd on every requested GPU node. [default: gpu-cuda-smoke] |
| `--gpu-cuda-smoke-image` | <str>  Container image for CUDA vectorAdd validation. [default: nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda12.5.0-ubuntu22.04] |
| `--mig` | --no-mig  Enable the same pinned RTX PRO 6000 MIG policy as fleet. [default: no-mig] |
| `--mig-strategy` | <str>  [default: mixed] |
| `--mig-config` | <str>  [default: all-balanced] |
| `--capacity-block-group` | <str>  Runtime-only strict GPU capacity block selector. |
| `--preemptible` | --on-demand  Run the GPU node group as preemptible, matching `npa cluster up`. This changes the capacity pool but not hard instance/disk/IP quotas; a reclaim stops the node mid-run. |
| `--dry-run` | Resolve settings and print intended actions only. |
| `--timeout` | <int>  Terraform apply timeout in minutes. [default: 120] |
| `--accelerator` | <str>  Requested SkyPilot accelerator (for example RTXPRO6000:1) to gate readiness. |
| `--gpu-readiness-timeout` | <float>  Seconds to wait for SkyPilot GPU discovery without deleting capacity. [default: 600.0] |
| `--gpu-readiness-poll-interval` | <float>  Seconds between SkyPilot GPU discovery checks. [default: 10.0] |
| `--sky-bin` | <str>  Pinned SkyPilot executable. |
| `--output-format` | <text\|json>  Output format. [default: text] |
| `--help` | Show this message and exit. |

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa provision-if-absent --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `provision-if-absent`.
