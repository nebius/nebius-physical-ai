# NPA Kubernetes Cluster Terraform

This directory contains a thin Terraform wrapper around a vendored copy of the
Nebius `k8s-training` solution from `nebius/nebius-solutions-library`.

The wrapper provisions a Managed Kubernetes cluster. The **default shape is a
small FTUE / Physical AI Data Factory cluster**, not a training farm:

- **1× GPU node** (`gpu-rtx6000`, preset `1gpu-24vcpu-218gb`) — a single RTX PRO
  6000 GPU.
- **1× CPU node** (`cpu-d3`, preset `8vcpu-32gb`) for CPU stages — big enough to
  schedule the shipped workflows' CPU requests (the Physical AI Data Factory asks
  for 4 CPU / 16Gi, which a 4vcpu-16gb node cannot fit after kubelet reserve).
- Nebius managed GPU driver image (`gpu_driver_mode = "auto"`) plus the
  provider-managed NVIDIA device plugin. CPU-only pools do not receive GPU
  driver settings.
- Nebius Network Operator through the upstream solution.
- **Shared Filesystem OFF by default** (`enable_filestore = false`). npa.workflow
  stages — including the Physical AI Data Factory — hand off artifacts via S3
  URIs, so no cross-node `/mnt/data` (and no Shared Filesystem SSD quota) is
  required. The platform block-storage StorageClass stays the cluster default.
- Optional strict GPU capacity-block reservation selector.
- Grafana, Prometheus, Loki, KubeRay, and OPA Gatekeeper disabled by default.
- Optional node-group service-account creation disabled by default, so the
  wrapper does not mutate tenant IAM groups unless explicitly requested.

### Opt into a larger cluster or Shared Filesystem

These remain available as explicit opt-in via `terraform.tfvars`, `TF_VAR_*`, or
`-var` flags — they are just not the default:

- **More / bigger GPUs:** raise `gpu_nodes_count` and/or set a multi-GPU preset
  (e.g. `gpu_nodes_preset = "8gpu-192vcpu-1744gb"`), and set
  `enable_gpu_cluster = true` (with `infiniband_fabric`) for multi-node
  InfiniBand training.
- **Preemptible GPU nodes:** `gpu_nodes_preemptible = true` draws on the
  preemptible capacity pool instead of on-demand. It does not bypass the hard
  instance, boot-disk-count, `compute.disk.size.network-ssd` byte-capacity, or
  public-IP allowances. The default 128 GiB CPU disk plus 1,023 GiB GPU disk
  requires 1,151 GiB of incremental NETWORK_SSD capacity. `npa cluster up` checks all
  cumulative hard quotas before `terraform apply`; capacity advice remains a
  separate availability signal.
- **Shared Filesystem:** set `enable_filestore = true` (optionally
  `filestore_disk_size_gibibytes`) to create one, or `existing_filestore = <id>`
  to attach an existing one (that alone implies `enable_filestore`). Either
  promotes the filesystem CSI StorageClass to the cluster default.

The vendored solution is based on upstream tag `main-v2026-05-25` with local
patches for GPU node-group reservation policy, configurable managed-driver
preset, and zero-CPU node-group omission so raw Terraform usage stays
standalone.

### GPU driver strategy and health gate

`gpu_driver_mode` has three stable values:

- `auto` (default) selects a Nebius managed-driver node image for every GPU
  node group and leaves CPU-only clusters untouched.
- `managed-image` explicitly requires that same safe path.
- `operator` uses the recipe's in-cluster NVIDIA GPU Operator driver path. It
  is an escape hatch for diagnostics and recipes that genuinely require it.

The managed preset defaults to `cuda13.0` and is configurable through
`managed_driver_preset`. Operator mode on an NVSwitch topology (a multi-GPU
SXM/NVL preset or `enable_gpu_cluster = true`) is rejected unless
`allow_unsafe_nvswitch_operator = true` is also set. That acknowledgement is
deliberately noisy: the operator/Fabric Manager path can start before the
Network Operator/MOFED has exposed host `/dev/infiniband/umad*` and `issm*`
devices, leaving Fabric in progress and CUDA uninitialized.

`npa cluster up` does not record the cluster as `RUNNING` merely because
Terraform and kubeconfig creation succeeded. For GPU clusters it waits for the
requested node topology to remain stable, verifies Ready nodes, boot IDs,
`NebiusGPUError`, generalized `nvidia.com/gpu` allocatable capacity, exposed
NVSwitch Fabric state, and mode-appropriate NVIDIA components, then runs CUDA
vectorAdd on every requested GPU node. Use `--validation-timeout` and
`--gpu-health-stabilization-seconds` to tune the wait; live validation can only
be disabled explicitly with `--skip-validate` or the CUDA workload alone with
`--skip-gpu-cuda-smoke`.

Changing these settings does not repair nodes that have already booted with the
operator-managed driver. Existing affected GPU pools require a controlled
rolling node-group update or recreation under the managed-image setting so
each replacement node boots from the new image. Follow workload disruption and
capacity-reservation policy; a code or CLI upgrade by itself cannot retrofit
the image on an existing node.

## Usage

**Terraform >= 1.12 is required.** The vendored modules declare
`required_version >= 1.12.0` and use `ephemeral` blocks; Terraform loads every
referenced module during `init`, including the ones this wrapper disables, so an
older binary fails to initialise the directory. `npa cluster up` /
`npa cluster down` check the version first and point at the upgrade; set
`NPA_TERRAFORM_BIN=/path/to/terraform` to use a newer binary without changing
`PATH`.

Provider checksums are tracked for `linux_amd64`, `linux_arm64`,
`darwin_amd64`, and `darwin_arm64`. `npa cluster up/down` verifies the current
platform and the SHA-bound coverage metadata before authentication or provider
download, runs `terraform init -lockfile=readonly`, and keeps both `TF_DATA_DIR`
and the platform-scoped plugin cache outside this source directory. A mismatch
is a stop condition: regenerate intentionally with Terraform's `providers lock
-platform=...` workflow in a clean checkout and review the diff; never remove
the lock file or bypass checksum verification.

Copy `terraform.tfvars.example` to `terraform.tfvars` and replace placeholders
with local values. `terraform.tfvars` is ignored by git. The example ships the
small default shape and shows the larger-cluster / Shared Filesystem opt-ins in
comments. Leave `iam_token` commented out when driving Terraform through `npa`:
the CLI mints a fresh token per run, and Terraform would prefer the pinned one
in `terraform.tfvars` (so `npa cluster up` rejects that file outright).

The default cluster needs **no** Shared Filesystem SSD quota, so
`npa cluster up` / `npa provision-if-absent` succeed with zero SFS quota. Only
when you opt in with `enable_filestore = true` and `existing_filestore = ""` does
the CLI check Shared Filesystem SSD quota before `terraform apply`; if quota is
not available, provide an existing filesystem ID or raise quota before running
`up`.

Set `capacity_block_group` only in private runtime configuration, such as a
gitignored `terraform.tfvars`, `TF_VAR_capacity_block_group`, or a direct
Terraform var:

```bash
terraform apply -var capacity_block_group=<capacity-block-group-id>
```

Then run:

```bash
npa cluster up --terraform-dir deploy/cluster --capacity-block-group <capacity-block-group-id>
```

The command runs `terraform init`, `terraform apply -auto-approve`, writes a
kubeconfig under `~/.npa/clusters/<cluster-name>/kubeconfig`, validates stable
GPU health and CUDA execution with `kubectl`, and can run an additional
SkyPilot Kubernetes GPU smoke test. See
[`docs/workbench/mk8s-gpu-driver-strategy.md`](../../docs/workbench/mk8s-gpu-driver-strategy.md)
for direct and Fleet configuration, recipe compatibility, diagnostics, and
migration guidance.

To inspect Terraform outputs alongside the local cluster cache:

```bash
npa cluster status --terraform-dir deploy/cluster
```

To destroy a Terraform-managed cluster:

```bash
npa cluster down --terraform-dir deploy/cluster
```
