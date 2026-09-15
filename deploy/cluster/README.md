# Managed Kubernetes for NPA

[Infrastructure docs](../../docs/cluster-backends.md) · [Workbench setup](../../docs/workbench/getting-started.md)

This Terraform wrapper provisions Nebius Managed Kubernetes using the vendored
`k8s-training` solution. Start with a workload's GPU and memory requirements,
then use NPA to configure, provision, and validate its cluster.

## Prerequisites

- [NPA installed](../../docs/install.md), a configured project, and an
  authenticated Nebius CLI profile with permission to create its resources.
- Terraform **1.12+**, `kubectl`, and an SSH public key available locally.
- Quota and capacity for the requested CPU/GPU nodes, boot disks, and public IPs.
- Model access checked before provisioning for a model-dependent workload.

NPA installs neither Terraform nor `kubectl`. Use
`NPA_TERRAFORM_BIN=/path/to/terraform` to select a compatible Terraform binary.

## Provision through NPA

Run from the repository root. Replace the alias and cluster name, then inspect
the plan against your workload's resource profiles:

```bash
project_alias='<your-project-alias>'
cluster_name='<your-cluster-name>'

npa workbench health preflight --checks nebius --json
npa provision-if-absent --project "$project_alias" \
  --cluster-name "$cluster_name" --dry-run --output-format json
```

Require a ready preflight, not just exit code zero. A preview can return
`status: blocked`; its `preflight.reasons` identifies missing quota or access.
Reserved GPUs still require sufficient boot-disk quota.

When the project, region, and topology are correct, run the same command without
`--dry-run`. It may create storage, networking, and a cluster. An existing
external cluster needs explicit registration; see
[use an existing cluster](../../docs/workbench/getting-started.md#verify-kubernetes-access).

After successful provisioning:

```bash
npa cluster status --project "$project_alias" --name "$cluster_name"
npa workbench workflow gpus --project "$project_alias" --cluster "$cluster_name" --json
```

Expect Ready nodes and the requested GPU count. Use the kubeconfig reported by
NPA and complete [SkyPilot setup](../../docs/orchestration/skypilot-setup.md)
before submitting a workflow. A provider-created cluster is not yet proof that
a model can run on it.

## Default shape

| Component | Default | Why |
| --- | --- | --- |
| GPU pool | One `gpu-rtx6000` node, `1gpu-24vcpu-218gb` | One RTX PRO 6000 GPU |
| CPU pool | One `cpu-d3` node, `8vcpu-32gb` | Room for a 4 CPU / 16 GiB stage, the controller, and system pods |
| GPU drivers | `auto`, managed `cuda13.0` image | Driver and device-plugin setup validated by NPA |
| Shared filesystem | Disabled | Workflow stages normally exchange artifacts through S3 |
| Monitoring, KubeRay, OPA Gatekeeper | Disabled | Enable only what the workload needs |
| Node-group service account creation | Disabled | Avoid unnecessary tenant IAM changes |

The default 128 GiB CPU disk plus 1,023 GiB GPU disk requires **1,151 GiB of
NETWORK_SSD quota**, in addition to disk-count and instance quota. This is a
small starter topology; it does not fit every workflow in the catalog.

## Use this Terraform directory directly

For explicit wrapper configuration, copy the example locally:

```bash
cp deploy/cluster/terraform.tfvars.example deploy/cluster/terraform.tfvars
```

Replace its tenant, project, region, subnet, cluster name, and public-key path.
`terraform.tfvars` is gitignored; keep private infrastructure values there.
Leave `iam_token` commented out when using NPA: the CLI supplies authentication
and rejects a pinned token that Terraform would prefer over it.

```bash
npa cluster up --project "$project_alias" --terraform-dir deploy/cluster
npa cluster status --project "$project_alias" --terraform-dir deploy/cluster
```

`cluster up` applies Terraform and validates the resulting cluster. It creates
resources without a separate Terraform approval prompt. To use reserved GPU
capacity, supply the exact authorized reservation through private configuration:

```bash
npa cluster up --project "$project_alias" --terraform-dir deploy/cluster \
  --capacity-block-group '<capacity-block-group-id>'
```

The reservation uses strict placement. `TF_VAR_capacity_block_group` is the
equivalent environment setting; an explicit CLI flag takes precedence.

## Change the topology

| Need | Configuration |
| --- | --- |
| More GPUs | Increase `gpu_nodes_count` and select a matching `gpu_nodes_preset`; a multi-GPU task must fit on one node |
| Multi-node InfiniBand | Set `enable_gpu_cluster = true` and the matching `infiniband_fabric` for a supported topology |
| Preemptible nodes | Set `gpu_nodes_preemptible = true`; this changes the capacity pool, while disk/IP/instance quotas still apply |
| New shared filesystem | Set `enable_filestore = true` and its size; requires Shared Filesystem SSD quota and the approved CSI chart repository |
| Existing shared filesystem | Set `existing_filestore`; it implies filesystem enablement and does not create a second filesystem |

Shared filesystem enablement promotes its CSI StorageClass to the cluster
default. Otherwise, platform block storage remains the default. See
[filesystem verification](../../docs/fleet-storage-verification.md) before
using a shared volume for multi-node work.

## GPU health and driver changes

The default `auto` and explicit `managed-image` strategies use a managed driver
image. `operator` is a separate driver path; NVSwitch topologies reject it
without the explicit unsafe-topology acknowledgement. For workloads that need
operator-mounted graphics libraries, follow the
[RTX rendering profile](../../docs/workbench/mk8s-gpu-driver-strategy.md).

Before reporting GPU readiness, NPA checks stable Ready nodes, GPU allocatable
capacity, driver components, exposed fabric state, and CUDA vectorAdd on every
requested GPU node. Keep these checks enabled. Changing configuration alone
does not change the image of already booted nodes; driver migration requires a
controlled update or recreation. The driver guide covers that procedure and
common failures.

## Terraform reproducibility

The vendored recipe is based on `main-v2026-05-25` with local reservation,
managed-driver, and node-group patches. Provider checksums cover Linux and macOS
on AMD64 and ARM64. NPA checks the SHA-bound platform metadata, initializes with
`-lockfile=readonly`, and stores provider caches outside this source directory.
On a lock mismatch, intentionally regenerate and review the provider lock;
removing it would discard the verification contract.

## Cleanup

Stop active jobs and preserve needed outputs before removing this cluster:

```bash
npa cluster down --project "$project_alias" --terraform-dir deploy/cluster
```

Read the proposed scope before confirming. Cluster deletion does not mean all
project storage or services are gone; follow the [teardown guide](../../docs/teardown.md)
for those separately owned resources. Preserve Terraform state after an
incomplete deletion so the exact operation can be resumed.
