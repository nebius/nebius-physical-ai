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
