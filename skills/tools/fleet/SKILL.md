---
name: fleet
description: Use to deploy or operate a fleet of Nebius Managed Kubernetes (k8s-training) clusters across one or many projects in a tenant from an npa.fleet/v0.0.1 spec — identical and/or custom clusters, create-on-demand projects, and a k8s-training recipe source that can consume the latest upstream changes.
---

# Fleet (multi-cluster, multi-project Kubernetes)

## When To Use

Use when a customer wants **more than one** Managed Kubernetes cluster stood up
across **one or many projects** in a single Nebius tenant — e.g. per-team or
per-tenant training clusters — and wants npa to drive it from one declarative
file. `npa fleet` wraps the public `nebius/nebius-solutions-library`
`k8s-training` recipe (the same recipe `npa cluster up` uses) once per cluster,
and can create the target projects on demand.

For a single cluster, prefer `npa cluster up`. For Slurm-on-Kubernetes, use
`npa soperator`.

Three-tier contract:
- **CLI**: `npa fleet plan|deploy|destroy|status --spec <fleet.yaml>`.
- **SDK**: `npa.sdk.fleet.deploy(spec)` / `destroy` / `plan` / `status` with
  `FleetSpec` / `ProjectSpec` / `ClusterSpec` / `NodePoolSpec`.
- **YAML / agent**: `apiVersion: npa.fleet/v0.0.1` spec; workflow
  `toolRef: infra.fleet.deploy` (config key `fleet_spec`).

## Spec (npa.fleet/v0.0.1)

A `defaults` cluster profile is deep-merged under every cluster, so **identical**
fleets are just project entries with no overrides. Projects may declare custom
`clusters` (overrides and/or several clusters), and identical + custom may be
freely **mixed**. Projects reference an existing `project_id` or a `name` that is
created on demand as `project_prefix` + `name`.

```yaml
apiVersion: npa.fleet/v0.0.1
name: fleet1-test
tenant_id: ""                # resolved from ~/.nebius + ~/.npa when empty
region: us-central1
project_prefix: "fleet1-test-"
defaults:
  cpu_nodes: { count: 1, platform: cpu-d3, preset: 48vcpu-192gb }
  gpu_nodes: { count: 1, platform: gpu-rtx6000, preset: 1gpu-24vcpu-218gb }
  enable_filestore: true
projects:
  - name: a                  # -> project fleet1-test-a (identical profile)
  - name: b                  # -> project fleet1-test-b (identical profile)
  # - name: c                # custom: overrides + a second cluster
  #   clusters:
  #     - name: train
  #       gpu_nodes: { count: 2, platform: gpu-h200-sxm, preset: 8gpu-128vcpu-1600gb }
  #       enable_gpu_cluster: true
  #       infiniband_fabric: us-central1-a
  #     - name: infer         # inherits defaults
  # - project_id: project-existing123   # deploy into an existing project by id
  #   clusters: [ {} ]
```

Example spec: `npa/examples/fleet/fleet1-test.yaml`.

## Procedure

1. Keep committed files public-safe: never hardcode tenant/project/registry IDs
   or SSH keys. The spec resolves tenant/region from `~/.nebius/config.yaml` and
   `~/.npa/config.yaml` when its fields are empty; the SSH public key comes from
   `ssh_public_key` or `~/.ssh/id_ed25519.pub` / `id_rsa.pub`.
2. **Plan first** (no infra): `npa fleet plan --spec fleet.yaml` shows the
   projects (create vs existing) and per-cluster node config.
3. **enable_gpu_cluster is auto**: GPU clustering (InfiniBand fabric) is only
   valid on fabric-capable 8-GPU SXM presets. Single-GPU presets (e.g. RTX PRO
   6000 `1gpu-24vcpu-218gb`) auto-set `enable_gpu_cluster=false`; set it `true`
   only with an 8-GPU preset **and** `infiniband_fabric`.
4. **Preflight quotas per target project.** Each cluster needs, in the *owning
   project's* region: `compute.instance.count` (nodes + etcd), the vCPU quota
   for the CPU/GPU presets, GPU quota for the GPU preset (on-demand GPU quota is
   often 0), `compute.disk.*`, and shared-filesystem SSD quota when
   `enable_filestore`. **Freshly created projects start with zero quota** — raise
   quotas (or deploy into projects that already have quota) before applying.
   Read with `nebius quotas quota-allowance get-by-name --parent-id <project> --region <region> --name <quota>`.
5. **Deploy** (asks for confirmation): `npa fleet deploy --spec fleet.yaml`. It
   prints the projects/clusters it will create/update and prompts before acting;
   pass `--yes`/`-y` for non-interactive runs. Missing projects are created via
   the `nebius` CLI unless `--no-create-projects`. Deploy runs per cluster and
   continues past a failing target (`--fail-fast` to stop); a JSON summary lists
   deployed vs failed clusters with kube contexts.
6. **Consume the latest recipe**: `--k8s-training-ref main` clones
   `nebius-solutions-library` and uses its `k8s-training` (or `--k8s-training-dir`
   for a local checkout). Omit both to use the repo-vendored, tested copy.
7. **Status / teardown**: `npa fleet status --spec fleet.yaml`; `npa fleet
   destroy --spec fleet.yaml` (prompts; `--yes`/`-y` or `--force` to skip).

## Add / remove clusters and projects

The fleet is spec-driven and idempotent, so growing or shrinking it is targeted:

- **Add** one or many: put the new project/cluster in the spec and deploy just
  those — `npa fleet deploy --spec fleet.yaml --only-projects c,d` or
  `--only-clusters train,infer`. Existing clusters are reconciled in place and
  untouched clusters are left alone (the persisted summary is merged, not
  overwritten).
- **Remove** one or many: `npa fleet destroy --spec fleet.yaml --only-clusters
  train` (or `--only-projects c`). Destroy tears down each **spec-declared**
  cluster that has local state, reclaims any VPC network the fleet created for
  it, and drops the removed cluster's local state so `status` reflects the
  removal. It does not enumerate clusters via the API, so a cluster created
  out-of-band is not reclaimed. Omitting `--only-*` tears down the whole fleet.

Both `deploy` and `destroy` confirm before acting (bypass with `--yes`/`-y`;
`destroy` also accepts `--force`).

## Gotchas

- **Per-cluster isolation**: each `(project, cluster)` gets its own Terraform
  install dir + local state under `~/.npa/fleet/<name>/<project>/<cluster>` and
  an env sidecar so `destroy` can rebuild the required `TF_VAR_*`. The sidecar's
  `status` starts as `provisioning` and is promoted to `deployed` only after a
  successful apply, so `status` never mislabels a half-applied cluster.
- **Region domain**: the recipe's `provider.tf` domain is patched to
  `api.nebius.cloud` for non-EU regions automatically (EU uses
  `api.eu.nebius.cloud`). If the upstream recipe drifts (renames `provider.tf`,
  moves the provider block, or changes the default domain), the patch becomes a
  no-op and the deploy logs a loud `WARNING` rather than silently talking to the
  wrong endpoint.
- **Latest-recipe coupling** (`--k8s-training-ref`/`--k8s-training-dir`): the
  rendered `terraform.tfvars` targets the recipe's *current* variable surface --
  the pinned `filesystem_csi.chart_version`, the `loki`/observability toggles,
  and the o11y/kuberay/gatekeeper `enable_*` flags. Pulling a newer recipe whose
  variables changed can require updating `fleet/tfvars.py`; validate with
  `npa fleet plan` + a `terraform plan` before a fleet-wide apply.
- **Filesystem quota**: `enable_filestore: true` creates one shared filesystem
  per cluster and consumes tenant `compute.filesystem.count` +
  `compute.filesystem.size.network-ssd` quota. Set `enable_filestore: false`
  (or raise quota) if the tenant is at its filesystem limit.
- **Auto-created VPC on destroy**: when a target project has no subnet, deploy
  creates a `<cluster>-net` + `<cluster>-subnet`; `destroy` reclaims exactly
  those (subnet then network). A *reused* pre-existing subnet is left untouched,
  and created *projects* are never deleted.
- **Parallelism** (`--concurrency N` / `-j N`, default 1 = sequential): applies/
  destroys N clusters at once. Each cluster has isolated terraform state, so there
  is no lock contention; the provider plugin cache is pre-warmed once (a single
  `init`) to avoid concurrent-init corruption, and each cluster streams to its own
  `<install_dir>/deploy.log` (or `destroy.log`). Wall-clock drops from `sum` to
  ~`max` of the applies. Parallel runs assume one cluster per project subnet —
  concurrent creates in a shared network race on the same `/16` CIDR pool, so give
  each cluster its own `subnet_id` (or its own project) when packing several into
  one project. Cap N to stay under Nebius API rate limits / GPU quota.
- **Stale IAM token**: a stale ambient `NEBIUS_IAM_TOKEN` shadows the profile
  exec-plugin; npa strips it for `nebius`/`terraform` calls unless
  `NPA_REUSE_IAM_TOKEN` is set (CI injecting a short-lived token).
- **terraform >= 1.12**: the recipe's modules use `ephemeral` blocks and a
  `>= 1.12` version constraint; set `NPA_TERRAFORM_BIN` if the system terraform
  is older.

## Verify

```bash
npa fleet plan --help
npa fleet deploy --help
npa/.venv/bin/python -m pytest npa/tests/unit/test_fleet_cli.py -q
```
