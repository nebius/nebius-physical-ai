# NPA Kubernetes Cluster Terraform

This directory contains a thin Terraform wrapper around a vendored copy of the
Nebius `k8s-training` solution from `nebius/nebius-solutions-library`.

The wrapper provisions a Managed Kubernetes cluster. The **default shape is a
small FTUE / Physical AI Data Factory cluster**, not a training farm:

- **1× GPU node** (`gpu-rtx6000`, preset `1gpu-24vcpu-218gb`) — a single RTX PRO
  6000 GPU.
- **1× small CPU node** (`cpu-d3`, preset `4vcpu-16gb`) for CPU stages.
- NVIDIA GPU Operator through the upstream solution.
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
- **Shared Filesystem:** set `enable_filestore = true` (optionally
  `filestore_disk_size_gibibytes`) to create one, or `existing_filestore = <id>`
  to attach an existing one. Either promotes the filesystem CSI StorageClass to
  the cluster default.

The vendored solution is based on upstream tag `main-v2026-05-25` with local
patches for GPU node-group reservation policy and zero-CPU node-group omission
so raw Terraform usage stays standalone.

## Usage

Copy `terraform.tfvars.example` to `terraform.tfvars` and replace placeholders
with local values. `terraform.tfvars` is ignored by git. The example ships the
small default shape and shows the larger-cluster / Shared Filesystem opt-ins in
comments.

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
kubeconfig under `~/.npa/clusters/<cluster-name>/kubeconfig`, validates the
cluster with `kubectl`, and can run a SkyPilot Kubernetes GPU smoke test.

To inspect Terraform outputs alongside the local cluster cache:

```bash
npa cluster status --terraform-dir deploy/cluster
```

To destroy a Terraform-managed cluster:

```bash
npa cluster down --terraform-dir deploy/cluster
```
