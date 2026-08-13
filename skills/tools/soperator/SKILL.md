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
  system: { min_size: 3, max_size: 24 }         # preset omitted: derive XS..XL upstream
  controller: {}                                # preset omitted: derive with same tier
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
accounting: false
# omitted REST preserves the legacy default (follows accounting); GPU checks still run directly
slurm_operator_version: "4.1.6"
k8s_version: "1.34"
node_group_version: "72"
```

## Procedure

1. Keep committed files public-safe: never hardcode project/tenant/registry IDs
   or SSH keys in the skill or spec templates. The spec resolves region/tenant/
   project from `~/.npa/config.yaml` when its fields are empty. Login access is
   deliberately named as a root grant: one `root_login_ssh_public_key` record,
   `--root-login-ssh-public-key-file`,
   `NPA_SOPERATOR_ROOT_LOGIN_SSH_PUBLIC_KEY[_FILE]`, the compatible
   `NPA_SSH_PUBLIC_KEY` path, then `~/.ssh/id_ed25519.pub` / `id_rsa.pub` /
   `id_ecdsa.pub`. NPA validates one OpenSSH record and logs only its source and
   SHA256 fingerprint. The legacy one-element `ssh_public_keys` list remains
   accepted; multiple records fail early.
2. **Preflight quotas** (the deploy hits these in order; raise before applying):
   - `compute.instance.count` — ~7 instances for a 2-pool cluster.
   - `compute.instance.non-gpu.vcpu` — sum of all node vCPUs.
   - `compute.disk.count` — boot disks + one IO_M3 cache disk per docker-cache pool + NFS PVC (~10).
   - `compute.disk.size.network-ssd-io-m3` — NFS PVC + docker-cache disks.
   - GPU on-demand quota is commonly 0; use `preemptible: true` for GPU pools.
   Read with `nebius quotas quota-allowance get-by-name --parent-id <tenant> --region <region> --name <quota>`.
3. Deploy: `npa soperator deploy --spec cluster.yaml --terraform-dir <solutions-lib>/soperator`
   (omit `--terraform-dir` to clone the library). The default source is immutable
   solutions-library commit `7046fb3c68314a940cdb47ff5c4fd23c01a6711e`, not
   moving `main`; `--ref` accepts only a full commit SHA. Before any provider
   mutation NPA verifies the checkout SHA, sizing thresholds, REST/accounting
   inputs, template patch targets, example Slurm chart `4.1.6`, Kubernetes
   `1.34`, and node bundle `72`. Requires terraform >= 1.12 (set
   `NPA_TERRAFORM_BIN` if the system terraform is older).
4. Control-plane sizing follows the pinned upstream worker-count tiers:
   XS `<10`, S `<100`, M `<500`, L `<2000`, XL `>=2000`. Omitted system,
   controller, and accounting presets inherit the tier. Explicit presets remain
   compatible when large enough; NPA rejects component-specific undersizing
   before clone/apply (system minimums: 16/16/16/32/64 vCPU across XS..XL).
   The system nodeset defaults to autoscaling from `min_size` through 24.
5. REST and accounting are explicit but the verified runtime has an important
   boundary: although the pinned Terraform module accepts the switches
   independently, the exact Slurm operator `4.1.6` implementation skips REST
   reconciliation without an accounting database. Omitted REST therefore keeps
   the compatible legacy default (it follows accounting), explicit
   `slurm_rest_enabled: false` is accepted, and explicit REST-on/accounting-off
   fails before mutation with an actionable error. This does **not** remove GPU
   creation validation: after every deploy NPA submits an exclusive Slurm step
   through the login-node jail on every GPU worker, allocates every GPU, and
   requires `deviceQuery`, `vectorAdd`, `simpleMultiGPU`, and
   `p2pBandwidthLatencyTest` to report `PASS`. Accounting+REST GPU specs may also
   use the upstream `dev` ActiveChecks; all other specs use `essential`.
6. `--apply-fixes` (default) applies the pinned-contract fixes:
   the `monitoring-system` namespace and prometheus-operator CRDs (charts need
   both even with telemetry off), recovery for a dashboards HelmRelease whose
   remediation retries were exhausted before those prerequisites existed, the
   `plugStackConfig.ncclInspectorPreConf` CRD preserve-unknown-fields patch, and
   the cluster-name-prefixed `<ns>-slurm-scripts` configmap. If a previous
   ActiveChecks Helm generation is demonstrably still Progressing while a newer
   generation is pending, NPA removes only the superseded Helm-owned wait hook
   and requests reconciliation; current-generation installs are never
   interrupted. For the default
   unconfined worker profile, NPA also extends the node configurator's sysctls so
   Ubuntu's AppArmor user-namespace gate does not block Enroot/Pyxis jobs. The
   template mutation is indentation-bounded to `nodeConfigurator.values` and is
   tied to the verified chart contract. Post-apply monitoring repair is
   best-effort: RBAC/transient failures are returned as diagnostics and do not
   turn an otherwise healthy Terraform reconciliation into a failed deploy.
   The direct GPU creation check is a required validation, not a best-effort
   repair; it still runs with `--skip-fixes`, and a missing node/GPU or failed
   CUDA result fails deployment.
7. Verify: `npa soperator status --name <name>` runs `sinfo` on the controller.

## Gotchas

- **AppArmor**: the custom localhost profile is not loaded by the verified chart;
  the spec defaults `use_default_apparmor_profile: false` (unconfined) so
  login/worker sshd start. Non-default chart versions are permitted only when the
  userns override is not needed (`use_default_apparmor_profile: true`); the
  explicit override allowlist is currently `4.1.6` and `4.1.7`.
- **Worker registration can race readiness.** Post-deploy
  repair sets the nodeset-service FQDN and resumes down nodes. If diagnosing by
  hand, the equivalent is `scontrol update NodeName=<w>
  NodeAddr=<w>.soperator-nodeset-svc.soperator.svc.cluster.local State=RESUME`.
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
