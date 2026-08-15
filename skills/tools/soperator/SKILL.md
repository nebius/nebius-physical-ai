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
- **CLI**: `npa soperator plan|deploy --spec <cluster.yaml>`, `npa soperator status --name <n>`, `npa soperator destroy --name <n>`.
- **SDK**: `npa.sdk.soperator.plan(spec)` / `deploy(spec)` / `destroy(name)` with `SoperatorSpec` / `WorkerPoolSpec`.
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
    # For reserved capacity, set preemptible: false and exactly one runtime
    # selector. Never commit a live ID/name:
    # capacity_block_group: <capacity-block-group-id>
    # capacity_block_group_name: <unique-capacity-block-group-name>
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
2. Plan with `npa soperator plan --spec cluster.yaml`. The public-safe output
   labels every worker pool as `on-demand`, `preemptible`, or `reserved` without
   echoing reservation selectors. Applied `status` output reads local Terraform
   state and reports the same capacity modes.
3. **Preflight quotas** (the deploy hits these in order; raise before applying):
   - `compute.instance.count` — ~7 instances for a 2-pool cluster.
   - `compute.instance.non-gpu.vcpu` — sum of all node vCPUs.
   - `compute.disk.count` — boot disks + one IO_M3 cache disk per docker-cache pool + NFS PVC (~10).
   - `compute.disk.size.network-ssd-io-m3` — NFS PVC + docker-cache disks.
   - GPU on-demand quota is commonly 0; use `preemptible: true` for GPU pools.
   Read with `nebius quotas quota-allowance get-by-name --parent-id <tenant> --region <region> --name <quota>`.
4. **Bind reserved B200 capacity explicitly when required.** Set exactly one
   per-pool selector: `capacity_block_group` for an immutable ID (fleet-compatible)
   or `capacity_block_group_name` for an exact tenant-scoped name. Reserved pools
   must be GPU pools with `preemptible: false`. Before Terraform rendering,
   deploy verifies the selected project belongs to the selected tenant/region,
   resolves names uniquely, and requires the group to be tenant-owned, active,
   region/platform/fabric-compatible, and large enough for the additional GPUs.
   Existing applied STRICT workers are credited so idempotent reconciles do not
   demand duplicate capacity. Missing, cross-tenant, inactive, ambiguous,
   incompatible, unreadable, or insufficient reservations fail before provider
   mutation. The renderer passes the exact upstream contract:
   `reservation_policy = { policy = "STRICT", reservation_ids = [...] }` and
   never combines it with `preemptible`. `AUTO` is not accepted because it can
   fall back to on-demand capacity.
5. Deploy: `npa soperator deploy --spec cluster.yaml --terraform-dir <solutions-lib>/soperator`
   (omit `--terraform-dir` to clone the library). The default source is immutable
   solutions-library commit `7046fb3c68314a940cdb47ff5c4fd23c01a6711e`, not
   moving `main`; `--ref` accepts only a full commit SHA. Before any provider
   mutation NPA verifies the checkout SHA, sizing thresholds, REST/accounting
   inputs, template patch targets, example Slurm chart `4.1.6`, Kubernetes
   `1.34`, and node bundle `72`. Requires terraform >= 1.12 (set
   `NPA_TERRAFORM_BIN` if the system terraform is older).
   Existing default checkouts, including legacy shallow clones on moving
   `main`, are reconciled in place under a filesystem lock: NPA fetches the
   exact missing commit and checks it out detached while preserving untracked
   installations, Terraform state, and operator-owned files. Tracked changes
   or conflicting untracked paths fail with recovery guidance; NPA never
   deletes or recreates an installation to repair source. Destroy uses the
   same resolver. `deploy --source-preflight-only` and
   `destroy --source-preflight-only` exercise their real source/install
   boundaries without Terraform initialization or provider mutation/deletion.
   For an existing installation, its owner-only environment sidecar is the
   authoritative project/tenant/region/subnet identity; an ambient default can
   never override it, and an incomplete/corrupt sidecar fails closed. Sidecar
   replacement is atomic. Before each deploy apply, NPA saves a Terraform plan,
   inspects its machine-readable actions, refuses every provider replacement,
   pure delete, and unexpected destructive action, and applies only that exact
   inspected plan. The immutable source patch stabilizes the cluster-context
   and login-IP triggers. During its one-time migration, only the three exact
   audited local-only refresh resources may replace; subsequent plans contain
   zero replacements. The sidecar is checkpointed only after the guard passes
   and immediately before provider mutation.
6. Control-plane sizing follows the pinned upstream worker-count tiers:
   XS `<10`, S `<100`, M `<500`, L `<2000`, XL `>=2000`. Omitted system,
   controller, and accounting presets inherit the tier. Explicit presets remain
   compatible when large enough; NPA rejects component-specific undersizing
   before clone/apply (system minimums: 16/16/16/32/64 vCPU across XS..XL).
   The system nodeset defaults to autoscaling from `min_size` through 24.
7. REST and accounting are explicit but the verified runtime has an important
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
8. `--apply-fixes` (default) applies the pinned-contract fixes:
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
   CUDA result fails deployment. Its independent
   `--gpu-creation-check-timeout` defaults to 1,800 seconds and bounds the
   whole gate, Slurm queue wait (`srun --immediate`), Slurm wall time, and the
   local kubectl process. The deploy `--timeout` remains Terraform-only and is
   never silently reused for GPU validation. Timed-out/failed gate jobs are
   cancelled by a unique name and verified absent from `squeue`. If the
   Terraform apply created/reconciled the cluster but this gate fails, the SDK
   raises `SoperatorDeploymentValidationError` with a public-safe `result`;
   text/JSON CLI output retains the install directory, kube context, and pool
   metadata, reports `degraded-validation`, and exits nonzero.
9. Verify: `npa soperator status --name <name>` runs `sinfo` on the controller
   and reports each applied worker pool's capacity mode without reservation IDs.

## Compatibility and migration

- **Slurm operator 4.1.0:** this former default is rejected because it is not in
  the verified runtime contract. Replace it with `4.1.6` for the default
  unconfined Enroot/Pyxis setup. `4.1.7` is accepted only with
  `use_default_apparmor_profile: true` after separately validating that loaded
  profile on the nodes. Do not re-enable `4.1.0`.
- **System autoscaling ceiling:** older NPA output fixed
  `control_plane.system.max_size` to `min_size`. Omission now renders the pinned
  upstream maximum of **24** (or `min_size` when it is greater than 24), which
  can increase capacity and cost if autoscaling is exercised. Set an explicit
  `control_plane.system.max_size` to retain a reviewed lower ceiling, provided
  it is at least `min_size`. Rendered tfvars, deploy results, and agent
  validation/dry-run output expose the effective numeric maximum.
- **Immutable source migration:** default legacy clones are upgraded in place;
  preserve a clean tracked checkout and keep installations/state untracked.
  Offline migration works once the pinned object has been fetched. If it is
  missing, restore access to the checkout's `origin` and retry.

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
npa soperator plan --help
npa soperator deploy --help
npa/.venv/bin/python -m pytest npa/tests/unit/test_soperator_cli.py -q
```
