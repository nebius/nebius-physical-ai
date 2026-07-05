---
name: soperator
description: Use to deploy or operate a Nebius soperator (Slurm-on-Kubernetes) cluster from npa — the npa.soperator/v0.0.1 spec, multi-preset worker pools, per-pool Docker/Enroot image cache, quota preflight, and post-deploy fixes.
---

# Soperator (Slurm-on-Kubernetes)

## When To Use

Use when a customer wants a managed **Slurm** cluster on Nebius (foundation-model
pretraining, large eval sweeps, HPC batch) instead of, or alongside, SkyPilot —
and wants npa to drive it. `npa soperator deploy` wraps the public
`nebius/nebius-solutions-library` soperator Terraform recipe from a compact
declarative spec, so customers get a working Slurm cluster without hand-editing
the recipe's large tfvars.

Three-tier contract:
- **CLI**: `npa soperator deploy --spec <cluster.yaml>`, `npa soperator status --name <n>`, `npa soperator destroy --name <n>`.
- **SDK**: `npa.sdk.soperator.deploy(spec)` / `destroy(name)` with `SoperatorSpec` / `WorkerPoolSpec`.
- **YAML / agent**: `apiVersion: npa.soperator/v0.0.1` spec; workflow `toolRef: infra.soperator.deploy`.

## Spec (npa.soperator/v0.0.1)

Multiple worker pools with different presets are first-class; each pool can
enable a node-local Docker/Enroot image cache disk (`NETWORK_SSD_IO_M3`) so
large GPU tool images don't thrash the boot disk.

```yaml
apiVersion: npa.soperator/v0.0.1
name: npasop                 # company_name; kube context = nebius-<name>-slurm
region: us-central1          # or resolved from ~/.npa config
control_plane:
  system: { min_size: 3, preset: 8vcpu-32gb }   # min_size >= 3 (recipe rule)
  controller: { preset: 8vcpu-32gb }
  login: { preset: 16vcpu-64gb }                 # login needs >= 16vcpu (sufficiency)
workers:
  - name: cpu8
    platform: cpu-d3
    preset: 8vcpu-32gb
    docker_cache: true          # node-local IO_M3 image cache
    docker_cache_gib: 930       # divisible by 93
  - name: gpu
    platform: gpu-b200-sxm
    preset: 8gpu-160vcpu-1792gb # GPU workers must be fabric-capable (8-GPU SXM)
    size: 2
    fabric: us-central1-b       # required for GPU presets; 1-GPU can't cluster
    preemptible: true           # on-demand GPU quota is often 0; preemptible works
    docker_cache: true
```

## Procedure

1. Keep committed files public-safe: never hardcode project/tenant/registry IDs
   or SSH keys in the skill or spec templates. The spec resolves region/tenant/
   project from `~/.npa/config.yaml` when its fields are empty.
2. **Preflight quotas** (the deploy hits these in order; raise before applying):
   - `compute.instance.count` — ~7 instances for a 2-pool cluster.
   - `compute.instance.non-gpu.vcpu` — sum of all node vCPUs.
   - `compute.disk.count` — boot disks + one IO_M3 cache disk per docker-cache pool + NFS PVC (~10).
   - `compute.disk.size.network-ssd-io-m3` — NFS PVC + docker-cache disks.
   - GPU on-demand quota is commonly 0; use `preemptible: true` for GPU pools.
   Read with `nebius quotas quota-allowance get-by-name --parent-id <tenant> --region <region> --name <quota>`.
3. Deploy: `npa soperator deploy --spec cluster.yaml --terraform-dir <solutions-lib>/soperator`
   (omit `--terraform-dir` to clone the library). Requires terraform >= 1.12
   (set `NPA_TERRAFORM_BIN` if the system terraform is older).
4. `--apply-fixes` (default) applies the 4.1.0-stable post-deploy fixes:
   prometheus-operator CRDs (operator chart needs ServiceMonitor even with
   telemetry off), the `plugStackConfig.ncclInspectorPreConf` CRD
   preserve-unknown-fields patch, and the cluster-name-prefixed
   `<ns>-slurm-scripts` configmap.
5. Verify: `npa soperator status --name <name>` runs `sinfo` on the controller.

## Gotchas

- **AppArmor**: the custom localhost profile is not loaded by SPO in 4.1.0-stable;
  the spec defaults `use_default_apparmor_profile: false` (unconfined) so
  login/worker sshd start. Do not flip it on unless your build loads the profile.
- **Worker registration is not fully automatic** in 4.1.0-stable: slurmrestd
  (the `rest` nodeset) is not deployed by the operator, so soperator's dynamic-
  node registration can stall. If `sinfo` shows 0 nodes with the worker pod
  Running, on the controller run: `scontrol update NodeName=<w> NodeAddr=<w>.soperator-nodeset-svc.soperator.svc.cluster.local State=RESUME`.
- **Region domain**: the recipe hardcodes the EU API domain; the deploy patches
  it to `api.nebius.cloud` for non-EU regions automatically.
- **Job I/O**: submit from the login node chrooted into `/mnt/jail`; write batch
  `--output` to a shared jail path (e.g. `/root/...`), not node-local `/tmp`.
- **GPU workers**: only 8-GPU SXM presets are fabric-capable; 1-GPU SXM presets
  return "does not support GPU clustering" and cannot be soperator GPU workers.

## Verify

```bash
npa soperator deploy --help
npa/.venv/bin/python -m pytest npa/tests/unit/test_soperator_cli.py -q
```
