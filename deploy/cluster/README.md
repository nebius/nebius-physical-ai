# NPA Kubernetes Cluster Terraform

This directory contains a thin, pinned Terraform wrapper around the Nebius
`k8s-training` solution from `nebius/nebius-solutions-library`.

The wrapper provisions a Managed Kubernetes cluster for GPU training with:

- RTX PRO 6000 GPU nodes by default.
- NVIDIA GPU Operator through the upstream solution.
- Nebius Network Operator through the upstream solution.
- Shared Filesystem CSI installed and promoted to the default StorageClass.
- Grafana, Prometheus, Loki, KubeRay, and OPA Gatekeeper disabled by default.
- Optional node-group service-account creation disabled by default, so the
  wrapper does not mutate tenant IAM groups unless explicitly requested.

The upstream solution is pinned at `main-v2026-05-25`.

## Usage

Copy `terraform.tfvars.example` to `terraform.tfvars` and replace placeholders
with local values. `terraform.tfvars` is ignored by git.

When `enable_filestore = true` and `existing_filestore = ""`, the CLI checks
Shared Filesystem SSD quota before `terraform apply`. If quota is not available,
provide an existing filesystem ID or raise quota before running `up`.

Then run:

```bash
npa cluster up --terraform-dir deploy/cluster
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
