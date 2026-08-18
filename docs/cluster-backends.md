# Cluster backends: standalone and fleet

`nebius mk8s` is the Nebius cloud-service CLI. NPA does not expose a primary
`npa mk8s` command. The NPA standalone Managed Kubernetes surface is `npa
cluster up` (and the agent `POST /api/infra/mk8s/provision` endpoint); `npa
fleet` expands the same desired state across projects. Slurm-on-Kubernetes uses
`npa soperator` standalone and `backend: soperator` in a fleet.

All four paths select a cluster backend adapter. Fleet owns project expansion,
target selection, concurrency, aggregate results, and inventory. Each adapter
owns validation, provider-free planning, recipe materialization, provider
preflight, apply, status/verification, reconciliation, and destroy for one
cluster. This keeps mk8s MIG logic and Soperator recovery logic out of fleet.

Existing `npa.fleet/v0.0.1` files are unchanged: a cluster with no `backend`
continues to mean mk8s and retains its historical plan output. New files may use
strict backend envelopes:

Replace angle-bracket placeholders with identifiers from your configured
Nebius environment; fleet specs do not expand shell variables.

One-entry mk8s:

```yaml
apiVersion: npa.fleet/v0.0.1
name: one-kube
region: us-central1
projects:
  - project_id: <project-id>
    clusters:
      - name: kube
        backend: mk8s
        mk8s:
          cpu_nodes: {count: 1, platform: cpu-d3, preset: 8vcpu-32gb}
```

One-entry soperator:

```yaml
apiVersion: npa.fleet/v0.0.1
name: one-slurm
region: us-central1
projects:
  - project_id: <project-id>
    clusters:
      - name: slurm
        backend: soperator
        soperator:
          workers:
            - {name: cpu, platform: cpu-d3, preset: 8vcpu-32gb}
```

Mixed backends:

```yaml
apiVersion: npa.fleet/v0.0.1
name: mixed-example
region: us-central1
projects:
  - project_id: <project-id>
    clusters:
      - name: kube
        backend: mk8s
        mk8s:
          cpu_nodes: {count: 1, platform: cpu-d3, preset: 8vcpu-32gb}
      - name: slurm
        backend: soperator
        soperator:
          workers:
            - {name: cpu, platform: cpu-d3, preset: 8vcpu-32gb}
```

Flat legacy mk8s settings may not be mixed with an explicit `mk8s:` envelope,
and mk8s fields beside `backend: soperator` fail at spec validation with the
exact unsupported field names. The fleet/project envelope owns tenant, project,
region, and subnet selection; embedding conflicting identity inside
`soperator:` is rejected before provider access.

## RTX PRO 6000 MIG

The same pinned MIG policy is available in a fleet and standalone:

```yaml
backend: mk8s
mk8s:
  gpu_nodes:
    count: 2
    platform: gpu-rtx6000
    preset: 1gpu-24vcpu-218gb
    disk_size_gib: 128
    capacity_block_group: <capacity-block-group-id>
  mig: {enabled: true, strategy: mixed, config: all-balanced}
```

```bash
npa cluster up --gpu-nodes 2 --gpu-platform gpu-rtx6000 \
  --gpu-preset 1gpu-24vcpu-218gb --capacity-block-group "$NPA_CAPACITY_BLOCK_GROUP" \
  --mig --mig-strategy mixed --mig-config all-balanced
```

Both resolve the operator driver, Kubernetes version, component pins, 128 GiB
boot disk, strict reservation, health timeout, ordered reconciliation, exact
per-node MIG resources, absence of whole-GPU capacity, and representative CUDA
MIG allocation through the shared mk8s desired-state/materialization contract.

## Ownership and safe destroy

New fleet inventory and per-target sidecars record `backend` per
`(project_key, cluster_name)`. Old
inventory without that field is read as mk8s. Status, reconcile, and destroy
compare the spec backend with persisted ownership and fail closed on a mismatch;
NPA never guesses ownership from a resource name. Each Soperator fleet target
has an isolated backend state root, and Soperator physical/context names must be
unique fleet-wide; duplicate names across projects fail spec validation.
Targeted destroy calls only the recorded adapter
and retains incomplete state for a scoped retry. Fleet-created project networks
are reclaimed only after inventory and canonical backend state prove that no
mk8s or soperator target remains in that project. A backend mismatch,
noncanonical state root, incomplete teardown, or unreadable ownership record
fails closed.

`npa fleet status` asks each adapter for native status and merges it with durable
inventory; mk8s status retains the last verified health and MIG evidence when a
live probe is not requested. Use `npa fleet verify-mig --wait --reconcile` for an explicit
MIG recheck. Destroying a one-entry or mixed fleet uses each target's recorded
backend and never infers ownership from cluster names.

Corrupt or unreadable fleet inventory fails closed. A successful provider apply
is not reported as fleet-owned success unless the backend discriminator and
target identity can be written durably. Destroy requires the exact persisted
provider ID, treats only authoritative NotFound as absence, and retains backend
state on authentication, permission, network, schema, or cleanup uncertainty.

Standalone `npa cluster up` supplies an explicit validation policy to the
shared mk8s backend. The default validates the exact Ready-node count, the
recipe-appropriate default StorageClass, and—when present—whole-GPU health and
CUDA or the full MIG convergence/CUDA gate. `--skip-validate` skips all of those
post-deploy checks, but never skips desired-state validation, capacity/quota
preflight, Terraform apply safety, kubeconfig creation, or durable identity
persistence. Fleet deploys explicitly retain their historical policy: GPU and
MIG targets are validated, while CPU fleet targets are not made dependent on a
standalone CLI flag.

Legacy standalone state supports a narrow residual recovery case. If an exact
persisted cluster was deleted out of band, or a prior destroy removed the
cluster resource before its Terraform-owned network/auxiliary resources,
`npa cluster down` may finish the retained Terraform destroy only after the
requested project/context/cluster identity agrees, the retained state has valid
lineage and managed residuals, no different cluster is present in state, and the
provider authoritatively reports the exact persisted cluster absent. Missing,
malformed, conflicting, or unreadable evidence fails closed and keeps the state.

Soperator destroy first reconciles the exact auxiliary IDs recorded from
Terraform. After exact cluster absence, it may additionally delete a CCM-
recreated filesystem or VPC allocation only when its provider type, exact
Terraform-derived name, project parent, and cluster-bound persisted ownership
all agree uniquely. Prefix-only matching is never deletion authority;
ambiguity or unreadable evidence retains state for retry.

A successful Soperator Terraform apply is not reported as a provisioning
failure when the following authoritative state pull is unreadable or lacks an
exact cluster ID/name. It returns `deployed-state-capture-failed`, preserves the
Terraform state and pre-apply ownership sidecar, and instructs the operator to
retry the identical deploy after restoring state readability. Cleanup never
acts on resources whose ownership was not captured. `soperator destroy
--timeout-minutes` is one deadline shared by Terraform destroy, exact cluster-absence
convergence, and auxiliary-resource reconciliation; each phase reports where
that deadline was exhausted while retaining recovery state.

### Legacy mk8s execution compatibility seam

`npa.fleet.lifecycle._call_legacy_execution` exists only for downstream tests
and integrations that monkeypatch helpers which moved into
`npa.cluster_backends.mk8s_execution`. Production dispatch uses the registered
backend adapter. The seam is deprecated and scheduled for removal with the next
breaking API revision after downstream callers migrate to injected process and
provider collaborators. New production-route tests must patch only those
external boundaries and assert dispatch through `get_backend("mk8s")`; no new
caller may depend on legacy global swapping.
