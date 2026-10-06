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

For RTX rendering, add `--gpu-workload-profile rtx-rendering`. Both the shared
backend and this direct wrapper forward the exact platform and preset to the
GPU Operator. Its RTX toolkit configuration reads the containerd file with
`RUNTIME_CONFIG_SOURCE=file`, preserving the on-disk config schema when
containerd's command output would migrate it to a newer version.

The default SkyPilot smoke requires a Linux operator host with `/proc` for
process ownership verification. NPA checks this before provisioning. On another
host, use `--skip-sky-smoke` for provisioning and run SkyPilot validation from
the Linux operator host before submitting workflows; the GPU, CUDA, and graphics
validation gates still run.
`provision-if-absent` makes the same early check when `--accelerator` or
`--sky-smoke` requests SkyPilot setup. Its `--dry-run` plans and `--skip-k8s`
storage operations remain usable on other hosts.

The direct RTX release has a read-only live acceptance test. Keep its JSON
configuration outside Git with `terraform_state_path`, `kubeconfig_path`,
`kube_context`, `platform`, `preset`, and `gpu_nodes` for the owned cluster:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_CLUSTER_RTX_TOOLKIT_LIVE_CONFIG="$private_verification_config" \
  npa/.venv/bin/python -m pytest \
    npa/tests/e2e/test_cluster_rtx_toolkit_live.py -q
```

It verifies the direct release's configuration revision, the exact RTX driver
selector, Ready GPU nodes, and Ready toolkit pods using the file schema source.

## Change the topology

> **Security note: the Kubernetes API endpoint is public by default.**
> `mk8s_cluster_public_endpoint` defaults to `true` so `kubectl` works out of the
> box (no VPN/bastion). For production clusters, set
> `mk8s_cluster_public_endpoint = false` in your `terraform.tfvars` (or
> `TF_VAR_mk8s_cluster_public_endpoint=false`) to keep the control plane off the
> public internet.

| Need | Configuration |
| --- | --- |
| Private Kubernetes API endpoint | Set `mk8s_cluster_public_endpoint = false`; the default is a public endpoint |
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

### Provider request deadlines

Deadline injection is advisory. Unsupported HCL/JSON, ambiguous providers, symlinks,
concurrent source changes, or unavailable writes leave recipe bytes unchanged and
record `status: advisory` with a stable `reason_code` in the deployment sidecar.
Terraform still validates and applies the original materialized recipe. Invalid
requested apply timeouts remain errors; unsafe paths are never traversed or rewritten.

The shared MK8s backend supplies omitted Nebius provider `timeout`,
`per_retry_timeout`, and `auth_timeout` attributes from the existing NPA apply
`--timeout` budget, expressed in minutes. This avoids the provider SDK's shorter
request default ending a create RPC before its resource identity is returned.
These are request deadlines; they are not a guarantee that asynchronous resource
creation succeeds or that GPU capacity is available.

In the pinned provider schema, `auth_timeout` covers the request including
authentication; it is not an authentication-only budget. Leaving that ceiling
or the per-attempt deadline at a shorter default would still interrupt a slow
create request. Giving an attempt the full apply budget deliberately allows one
slow attempt to consume that budget. Earlier transient failures can still retry
under the existing retry count; NPA does not promise time for every retry.

NPA edits only the owned materialized provider block. Explicit operator values,
expressions, nulls, retry counts, aliases, and override-file settings remain
unchanged. The source recipe remains unchanged. The deployment sidecar records
the inserted defaults and materialized source hashes before Terraform runs.

Destroy retains those materialized provider settings so the recorded KubeRay
input digest and recovery provenance remain valid. Its own NPA outer deadline
and cancellation still apply. Existing installations without these generated
defaults retain their original settings until an ordinary supported reapply.


The provider inspection uses `python-hcl2` 8.x and supports the pinned NPA
recipes, ordinary block/line comments, heredocs, aliases and Terraform JSON.
It refuses syntax the parser cannot represent before writing or running
Terraform. One known valid-Terraform limitation is a block comment between the
`provider` keyword and its label, such as `provider /* note */ "nebius" {}`.
Such a custom recipe must move that comment outside the block header; NPA does
not attempt a text/regex rewrite or silently run with uninspected defaults.

For live verification, set `NPA_MK8S_RPC_LIVE_CONFIG` to a private JSON evidence
configuration and run `npa/tests/e2e/test_mk8s_provider_rpc_live.py` with
`NPA_INTEGRATION_E2E=1`. Run its `live` phase after a supported owned provision,
and its `cleanup` phase after supported teardown. The verifier is read-only:
it binds the producing source/start/result, exact state and deployment
sidecar, materialized provider hash, and actual provider identities. Cleanup
requires typed NotFound for the exact cluster and each recorded node group.
Before those reads, the verifier requires the same plain service-account
profile, credential/config file byte hashes, endpoint, project and tenant that
the provision-start receipt recorded. It checks the live project/tenant identity
and removes ambient Nebius selectors from the subprocess environment. Missing
legacy authority bindings are refused; never backfill them after a run. This
read-only harness supports explicit key-backed profiles, not attached metadata
or arbitrary authentication plugins.
It does not adopt or destroy resources, and a failed provision cannot be
reported as a successful lifecycle. Keep all configuration and receipts
private because they contain operational identifiers.

## Cleanup

Stop active jobs and preserve needed outputs before removing this cluster:

```bash
npa cluster down --project "$project_alias" --terraform-dir deploy/cluster
```

Read the proposed scope before confirming. Cluster deletion does not mean all
project storage or services are gone; follow the [teardown guide](../../docs/teardown.md)
for those separately owned resources. Preserve Terraform state after an
incomplete deletion so the exact operation can be resumed.
### Terminal recovery after an already completed teardown

`npa cluster reconcile-absent --evidence-file <private-manifest.json>` closes one
failed standalone MK8s operation only after verifying its original producer
records and fresh provider absence. It changes the exact local journal to
`destroyed` and releases that operation's project lease. It never deletes,
adopts, restores Terraform state, or relaunches a cloud resource. The original
provisioning failure and original records remain retained.

The version-1 private manifest pins the original journal, provision start/result,
runner source and stdout/stderr, original key-backed profile binding, native
backend archive, and cleanup-intent receipt by absolute path and SHA-256. It also
selects the original producer repository and operation ID. The native archive
must contain exactly one regular `.npa-fleet-env.json` and
`k8s-training/terraform.tfstate`. The implementation binds the operation ID
printed by the original NPA process, operation-generation timestamps, exact
project/tenant/name, source revision and cleanup hash chain. It does not invent
an `operation_id` missing from old native metadata. A newly authored summary
cannot replace missing original records; incomplete legacy evidence is refused.

Recovery verifies the same plain service-account config and credential bytes,
the active project/tenant, typed absence of every recorded cluster, node group
and application release, and complete paginated inventories. Unsupported managed
resource instances, live resources, denied/unknown reads, ambiguous names,
changed original bytes, active execution, and journal/lease races fail closed.
Local leases serialize cooperating NPA writers, not external provider actors.
The verification is a fresh observation, not a permanent guarantee of absence.

Keep the manifest and raw records private. Successful recovery retains an
owner-only audit under the original operation directory. Repeating the same
terminal recovery verifies its recorded audit and exact original project lease.
If a process stopped after writing the terminal journal, the retry completes only
that same lease release under the original project and execution locks. Changed
or foreign leases are refused. It returns `already-reconciled` only after durable
release readback, without claiming fresh provider reads. A failed provision is never relabeled
as a successful deployment or workload result.

The opt-in `test_cluster_absence_recovery_live.py` accepts
`NPA_ABSENCE_RECOVERY_LIVE_CONFIG` with the exact reviewed source, operation,
manifest hash and `allow_terminal_reconciliation: true`. Run it only for the
operator-owned operation after scoped cleanup and source review. Offline tests
use synthetic original records and real separate processes to exercise locks,
pagination, identity mismatches and hostile provider responses.


Absence verification uses a 120-second deadline for each external read. Set
`--verification-timeout-seconds SECONDS` on `npa cluster reconcile-absent` to
adjust it; `0` disables that deadline while retaining cancellation cleanup.
This is a read deadline, not a provisioning or workflow execution budget.
Timeouts kill and join only the verifier's owned process group, preserve the
failed journal and lease, and return `verification-unavailable` with
`reconciled: false`. Retry after restoring the provider or local Git reader.
Malformed evidence also returns that sanitized envelope without private paths.

The verifier observes child exit without reaping, then stops the owned process
group while its leader PID remains reserved, and only then joins the leader.
Nested reader mode requires the parent-issued marker and an owned child session;
it must never be entered by the caller holding the recovery locks. SIGINT joins
owned children before propagating; SIGTERM temporarily becomes a sanitized
verification failure and the previous handler is restored afterward.

`NPA_ABSENCE_MAX_OUTPUT_BYTES` sets the accepted bytes per captured stream
(default 67108864, or 64 MiB; `0` explicitly disables this size limit). Capture
sizes are monitored while the read runs and checked before loading output into
memory. Excess output refuses verification and stops owned children; it never
establishes absence. This is an evidence-read bound, not a workflow/run budget.
