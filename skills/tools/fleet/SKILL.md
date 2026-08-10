---
name: fleet
description: Use to deploy or operate a fleet of Nebius Managed Kubernetes (k8s-training) clusters across one or many projects in a tenant from an npa.fleet/v0.0.1 spec — including strict capacity-block-backed GPU pools, identical and/or custom clusters, create-on-demand projects, and a k8s-training recipe source that can consume the latest upstream changes.
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
profile: ""                  # ~/.nebius profile to authenticate as; "" = active
project_prefix: "fleet1-test-"
defaults:
  cpu_nodes: { count: 1, platform: cpu-d3, preset: 48vcpu-192gb }
  gpu_nodes:
    count: 1
    platform: gpu-rtx6000
    preset: 1gpu-24vcpu-218gb
    # Optional runtime-only ID; renders STRICT and never falls back to PAYG.
    capacity_block_group: ""
  enable_filestore: true
  filestore_disk_size_gibibytes: 1024
  filestore_mount_path: /mnt/data
  filestore_mount_tag: npa-shared-fs
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

## Targeting another tenant (`profile`)

A Nebius service account belongs to exactly one tenant, so deploying into a
second tenant means authenticating as *that* tenant's principal. Set the spec's
`profile:` (or pass `--profile <name>`, which wins) to name a `~/.nebius`
profile; every `nebius` CLI call, the minted terraform `TF_VAR_iam_token`, and
the generated kubeconfig's exec-credential args are pinned to it. The machine's
active profile is never mutated, so concurrent fleets in different tenants stay
independent.

With a profile set, `tenant_id` resolves from **that** profile's `tenant-id`
(never the active profile's) and a profile with no `tenant-id` is a hard error
instead of a silent deploy into the wrong tenant. `destroy` falls back to the
profile recorded in each cluster's env sidecar at deploy time, so a teardown
always authenticates as the principal that created the cluster.

Register a service-account profile non-interactively:

```bash
nebius profile create <name> --endpoint api.nebius.cloud \
  --service-account-id <sa-id> --public-key-id <public-key-id> \
  --private-key-file-path ~/.nebius/<name>.pem \
  --parent-id <sa-parent-project> --tenant-id <tenant> --skip-auth
nebius --profile <name> iam get-access-token >/dev/null   # verify
```

`nebius profile create` also *activates* the new profile; re-activate the
previous one (`nebius profile activate <prev>`) if other tooling on the host
depends on it.

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
4. **Bind reserved GPU capacity explicitly when required.** Set
   `gpu_nodes.capacity_block_group` to a runtime-supplied Capacity Block Group
   ID. Fleet renders `gpu_nodes_reservation_policy = { policy = "STRICT", ... }`,
   so an unavailable or incompatible block fails instead of falling back to
   ordinary on-demand capacity. Never commit a live capacity block ID.

   Discover and verify reservations read-only with:

   ```bash
   nebius --profile <p> capacity capacity-block-group list \
     --parent-id <tenant> --all --format json
   nebius --profile <p> capacity capacity-interval list \
     --parent-id <capacity-block-group> --all --format json
   nebius --profile <p> capacity capacity-block-group list-resources \
     --id <capacity-block-group> --format json
   nebius --profile <p> capacity resource-advice list \
     --parent-id <tenant> --all --format json
   ```

   Preflight requires the named block to be active, in the target tenant and
   region, and matched to the GPU platform and InfiniBand fabric. It checks the
   aggregate GPU requirement against remaining reserved capacity. Only after
   that validation does it exclude those GPUs from ordinary GPU quota; all
   node, boot-disk, GPU-cluster, Kubernetes, and storage quotas still apply.
5. **Preflight quotas at the tenant, before anything else.** Each cluster needs,
   in the target region: `compute.instance.count` (worker nodes only; the
   managed control plane is service-owned),
   `compute.instance.non-gpu.vcpu` for the CPU preset,
   `compute.instance.gpu.<family>` for the GPU preset (on-demand GPU quota is
   frequently **0**), `compute.disk.count`/`compute.disk.size.network-ssd`,
   `compute.gpucluster.count` when `enable_gpu_cluster`, and
   `compute.filesystem.count` + `compute.filesystem.size.network-ssd` when
   `enable_filestore`. List them all at once with
   `nebius --profile <p> quotas quota-allowance list --parent-id <tenant> --format json`
   (each item carries `metadata.name`, `spec.region`, `spec.limit`).

   `deploy` does this automatically (`--preflight`, on by default) and refuses to
   apply when a capacity block or tenant limit cannot cover the in-scope
   clusters; `--no-preflight` attempts it anyway. Ordinary quota checks subtract
   exact usage when reported, otherwise the live wire's 0..1 fractional
   `status.usage_percentage`. Filesystem size is byte-valued and must report
   `status.unit: byte`; an incompatible/missing unit fails closed.

   Project-level allowances only *subdivide* the tenant allowance, so a tenant
   limit of 0 cannot be worked around by creating a project quota: raising a
   tenant allowance is a `root-g00root` operation and a tenant-scoped service
   account gets `PermissionDenied ... resource ID: root-g00root`. A new tenant
   therefore needs its GPU/filesystem quotas raised by the Nebius account team
   before any GPU or shared-filesystem cluster can be applied.
6. **Deploy** (asks for confirmation): `npa fleet deploy --spec fleet.yaml`. It
   prints the projects/clusters it will create/update and prompts before acting;
   pass `--yes`/`-y` for non-interactive runs. Missing projects are created via
   the `nebius` CLI unless `--no-create-projects`. Deploy runs per cluster and
   continues past a failing target (`--fail-fast` to stop); a JSON summary lists
   deployed vs failed clusters with kube contexts.
7. **Consume the latest recipe**: `--k8s-training-ref main` clones
   `nebius-solutions-library` and uses its `k8s-training` (or `--k8s-training-dir`
   for a local checkout). Omit both to use the repo-vendored, tested copy.
8. **Status / teardown**: `npa fleet status --spec fleet.yaml`; `npa fleet
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
  cluster that has local state, reclaims any project VPC network the fleet
  created after the last cluster state is gone, and drops local state only after
  authoritative Terraform teardown succeeds. An incomplete destroy reports
  `destroy-incomplete`, retains its exact state, and prints a scoped retry
  command. It does not enumerate clusters via the API, so a cluster created
  out-of-band is not reclaimed. Omitting `--only-*` tears down the whole fleet.

Both `deploy` and `destroy` confirm before acting (bypass with `--yes`/`-y`;
`destroy` also accepts `--force`).

## Gotchas

- **Per-cluster isolation**: each `(project, cluster)` gets its own Terraform
  install dir + local state under `~/.npa/fleet/<name>/<project>/<cluster>` and
  an env sidecar so `destroy` can rebuild the required `TF_VAR_*`. The sidecar's
  `status` starts as `provisioning` and is promoted to `deployed` only after a
  successful apply and kubeconfig write. Credential failures report
  `deployed-credentials-failed` and retain Terraform/cloud state for recovery.
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
- **Filesystem boot safety**: the filesystem is attached `READ_WRITE` to every
  CPU and GPU node-group template with `filestore_mount_tag`, and cloud-init
  persists the same tag at `filestore_mount_path` as virtiofs with
  `defaults,nofail`. Nebius warns that omitting `nofail` can prevent a node from
  booting after the filesystem is missing. The bundled CSI chart version is
  `0.1.6`, matching the current official filesystem-over-CSI guide.
- **Auto-created VPC on destroy**: when a target project has no subnet, deploy
  resolves one project network/subnet before parallel cluster applies and stores
  ownership in `.npa-fleet-network.json`. All clusters without explicit
  overrides share that authoritative subnet. `destroy` reclaims exactly that
  pair (subnet then network) after every project cluster state is gone. A
  *reused* pre-existing subnet is left untouched, and created *projects* are
  never deleted.
- **Parallelism** (`--concurrency N` / `-j N`, default 1 = sequential): applies/
  destroys N clusters at once. Each cluster has isolated terraform state, so there
  is no lock contention; the provider plugin cache is pre-warmed once (a single
  `init`) to avoid concurrent-init corruption, and each cluster writes restrictive
  per-cluster diagnostics to `<install_dir>/deploy.log` or the surviving
  `<fleet_root>/.logs/<project>/<cluster>/destroy.log`. JSON mode uses the same
  logs even for sequential or one-target runs so stdout remains one JSON document.
  Wall-clock drops from `sum` to ~`max` of the applies. Project network resolution
  remains sequential and single-flight, eliminating duplicate network/subnet
  creation in a fresh shared project; explicit per-cluster `subnet_id` values are
  preserved.
- **Stale IAM token**: a stale ambient `NEBIUS_IAM_TOKEN` shadows the profile
  exec-plugin; npa strips it for `nebius`/`terraform` calls unless
  `NPA_REUSE_IAM_TOKEN` is set (CI injecting a short-lived token). This is also
  why `--profile` must be threaded through rather than relying on the ambient
  token: the token, not the profile, decides the principal.
- **Default StorageClass depends on `enable_filestore`**: the recipe installs the
  filesystem CSI (`csi-mounted-fs-path-sc`, `ReadWriteMany`) only when a shared
  filesystem is attached. Without it the only class is
  `compute-csi-default-sc` (provisioner `compute.csi.nebius.com`, `ReadWriteOnce`
  disk-over-CSI); a PVC naming `csi-mounted-fs-path-sc` then sits `Pending` with
  `ProvisioningFailed: storageclass ... not found` and its pod never schedules.
- **A quota-starved node group looks like a hang, not an error**: mk8s *accepts* a
  node group whose instances it cannot create, then retries forever while compute
  rejects each one. Terraform only prints `Still creating...` until the timeout,
  and the real reason is visible solely in the node group's own events:
  `nebius --profile <p> mk8s node-group get --id <ng> --format json` shows
  `QuotaFailure` with `quota`, `limit`, and `requested`. This is exactly why the
  quota preflight exists — the node group's k8s-side state is `PROVISIONING`, not
  `FAILED`, so nothing else surfaces the wall.
- **`AUTO` is not an explicit reservation guarantee**: Nebius may fall back to
  ordinary on-demand capacity. Fleet's `capacity_block_group` surface therefore
  always renders `STRICT` and validates that exact block before apply.
- **terraform >= 1.12**: the recipe's modules use `ephemeral` blocks and a
  `>= 1.12` version constraint. Deploy and destroy assert machine-readable
  `terraform version -json` output before lifecycle work; set
  `NPA_TERRAFORM_BIN` if the system terraform is older.

## Verify

```bash
npa fleet plan --help
npa fleet deploy --help
npa/.venv/bin/python -m pytest npa/tests/unit/test_fleet_cli.py -q
```
