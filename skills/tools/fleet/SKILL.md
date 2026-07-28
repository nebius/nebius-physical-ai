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
5. **Deploy**: `npa fleet deploy --spec fleet.yaml`. Missing projects are created
   via the `nebius` CLI unless `--no-create-projects`. Deploy runs per cluster
   and continues past a failing target (`--fail-fast` to stop); a JSON summary
   lists deployed vs failed clusters with kube contexts.
6. **Consume the latest recipe**: `--k8s-training-ref main` clones
   `nebius-solutions-library` and uses its `k8s-training` (or `--k8s-training-dir`
   for a local checkout). Omit both to use the repo-vendored, tested copy.
7. **Status / teardown**: `npa fleet status --spec fleet.yaml`;
   `npa fleet destroy --spec fleet.yaml --force`.

## Gotchas

- **Per-cluster isolation**: each `(project, cluster)` gets its own Terraform
  install dir + local state under `~/.npa/fleet/<name>/<project>/<cluster>` and
  an env sidecar so `destroy` can rebuild the required `TF_VAR_*`.
- **Region domain**: the wrapper provider domain is patched to `api.nebius.cloud`
  for non-EU regions automatically (EU uses `api.eu.nebius.cloud`).
- **Stale IAM token**: a stale ambient `NEBIUS_IAM_TOKEN` shadows the profile
  exec-plugin; npa strips it for `nebius`/`terraform` calls unless
  `NPA_REUSE_IAM_TOKEN` is set (CI injecting a short-lived token).
- **Old terraform**: the recipe needs a recent terraform; set `NPA_TERRAFORM_BIN`
  if the system terraform is too old.

## Verify

```bash
npa fleet plan --help
npa fleet deploy --help
npa/.venv/bin/python -m pytest npa/tests/unit/test_fleet_cli.py -q
```
