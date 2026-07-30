# Changelog

Releases are git tags `vX.Y.Z` matching `npa/pyproject.toml`; artifacts are
built and attached by `.github/workflows/release.yml`. See `docs/releasing.md`
for the release process. Entries accumulate under "Unreleased" and move under
a versioned heading when a release is cut.

## Unreleased

### Default Managed Kubernetes cluster is now a small FTUE / PAIDF shape

- **`deploy/cluster` defaults shrank from a 2×8-GPU farm to 1 GPU + 1 CPU node,
  and Shared Filesystem is now off by default.** `npa provision-if-absent` /
  `npa cluster up` (which inherit the Terraform `variables.tf` defaults) now
  create `gpu_nodes_count = 1` on `gpu-rtx6000` with the `1gpu-24vcpu-218gb`
  preset plus one small `cpu-d3` / `4vcpu-16gb` node, and `enable_filestore =
  false`. npa.workflow stages (including the Physical AI Data Factory) hand off
  artifacts over S3 URIs, so the default cluster needs no cross-node `/mnt/data`
  and **no Shared Filesystem SSD quota** — the default provision succeeds with
  zero SFS quota.
- The CLI Shared-Filesystem quota preflight and the post-apply default-
  StorageClass validation now key off `enable_filestore`: the quota check is
  skipped and the platform block-storage StorageClass is accepted when the
  shared filesystem is not opted into (it still enforces the filesystem CSI
  StorageClass when `enable_filestore = true`).
- **Larger GPU presets / multi-node InfiniBand and Shared Filesystem remain
  explicit opt-ins** via `deploy/cluster` tfvars/`TF_VAR_*`/`-var`
  (`gpu_nodes_count`, a multi-GPU `gpu_nodes_preset`, `enable_gpu_cluster`,
  `enable_filestore`, `existing_filestore`) — documented in
  `deploy/cluster/README.md` and `terraform.tfvars.example`.

### First-run walkthrough fixes (README → agent → Physical AI Data Factory)

- **Agent deploy SSH-timeout now fails with one clear line, not a dumped bash
  script.** When the new VM is `RUNNING` with a public IP but its `tcp/22` is
  unreachable from the deploy host (corporate VPN / split-tunnel / firewall),
  `null_resource.wait_for_cloud_init` used to die opaquely under `set -e`, so
  Terraform printed the entire local-exec provisioner body. The SSH-wait now
  emits an explicit reachability error, and `npa agent deploy` adds a concise
  post-rollback diagnosis (`_agent_deploy_failure_hint`) distinguishing an
  unreachable SSH port from a failed cloud-init bootstrap.
- **`npa workbench workflow submit` preflight verifies the `--infra k8s/<context>`
  context exists** in your kubeconfig before `sky jobs launch`. A missing context
  (e.g. after purging a stale controller with no cluster provisioned) previously
  failed late with a long SkyPilot stack (`Context <name> not found ... Available
  contexts: []`); it is now one prerequisite line with the available contexts and
  the fix.
- **Stale-controller remedy and docs no longer suggest `sky status --all`**,
  which SkyPilot 0.12 rejects (`Did you mean --all-users?`). They now use
  `sky status -r` / plain `sky status`.
- **`npa configure` object storage now defaults to yes and warns when skipped.**
  After project discovery the prompt defaults to `[Y/n]`, and declining prints
  that `npa agent setup` and the Physical AI Data Factory both need an S3 bucket
  + access key — so a first run does not silently finish without the storage the
  next steps require.
- **Docs align on `npa agent setup`.** The Physical AI Data Factory deploy guide
  now leads with `npa agent setup` (matching the README), keeps `fresh-setup` as
  the scripted path, and both call out the VPN/firewall SSH-reachability caveat.
  The README adds a note to bind a federation profile's `tenant-id`/`parent-id`
  before `npa configure`.
- **`npa configure` project discovery no longer pins a project-local registry.**
  When picking a project from discovery, configure saved whatever container
  registry the project happened to have; a project whose only registry is in
  another region (e.g. `us-central1`) got a project-local registry that does not
  hold the `npa-*` workbench images, breaking later workbench deploys with
  image-not-found. Discovery now shares the manual-entry path's logic
  (`_preferred_container_registry`): it only adopts a discovered registry when it
  is in eu-north1 (where the workbench images live), otherwise it falls back to
  the eu-north1 first-party default. The project's own region is still preserved
  (placement follows the project).
- **`npa agent setup` no longer leaks Typer defaults into Terraform.** `setup`
  calls `fresh-setup`, which calls `deploy`, as plain Python functions, so every
  omitted option arrived as a `typer.models.OptionInfo` sentinel:
  `server_port`/`ssh_user`/`extra_ingress_ports` were rendered as literal
  `"<typer.models.OptionInfo object at 0x...>"` Terraform vars, and
  `no_public_https` (an OptionInfo, therefore truthy) **silently disabled HTTPS**.
  New `npa.cli._typer_defaults.resolve_typer_defaults` resolves declared defaults
  on direct calls; a guardrail test fails when a Typer command is called as a
  function without it.
- **`npa configure` recovers the discovery tenant.** Profiles with only
  `parent-id` (federation profiles, single-project profiles) silently skipped
  project discovery. It now derives the tenant from the profile's project, then
  from the listable tenants, and explains why when it still cannot. New
  `nebius.get_project_tenant_id()` / `get_project_name()` / `set_profile_project()`.
- **`npa configure` on a pipe no longer trips `GetPassWarning`**, offers to point
  the Nebius CLI profile at the selected project, and derives the local alias
  from the Nebius **project name** instead of the region.
- **`credentials.yaml` accepts `token_factory: {api_key}` / `huggingface: {token}`**
  — the shapes the Physical AI Data Factory guide documented and the loader
  silently ignored. `tokens:` remains canonical and wins.
- **`npa skypilot bootstrap` persists `skypilot.sky_bin`** to `~/.npa/config.yaml`
  (`--save`, default on), so a new shell no longer fails with "SkyPilot CLI
  executable is not configured".
- **Stale SkyPilot jobs controllers explain themselves.** A controller cached
  against a missing kubeconfig now reports the path plus `sky status -r`,
  `sky down sky-jobs-controller-<id>`, `npa provision-if-absent` and
  `--infra k8s/<context>` instead of a raw traceback.
- **New `npa workbench workflow stage-src`** (and `submit --stage-src`) publishes
  the local `npa` package to `s3://<bucket>/npa-src/npa/` for image-less steps —
  previously `NPA_SRC_S3_URI` was required with no command to produce it.
- **`submit` reports every missing prerequisite at once** (SkyPilot CLI, npa
  source, placeholder bucket) with the fix for each; `--skip-preflight` bypasses.
- **`plan-spec` and `run-spec` accept `--var KEY=VALUE`**; all three commands warn
  when planning against the spec's `example-bucket` placeholder.
- **Docs: one ordered green path** from README to a real submit, a complete
  Physical AI Data Factory quickstart (`provision-if-absent` → `skypilot
  bootstrap` → `stage-src` → submit) with a failure-recovery table, and a single
  documented venv path (`.venv` for users, `npa/.venv` for repo validation),
  guarded by tests.

### Setup re-architecture: project discovery + simple agent deploy

- **`npa configure` discovers projects instead of asking you to type ids.** It
  now enumerates the Nebius projects your CLI profile can reach (via
  `nebius iam tenant list` + `iam project list`) and lets you pick one or more
  from a list, auto-deriving tenant id, project id, and region. New client
  helpers: `npa.clients.nebius.list_tenants()`, `list_projects_in_tenant()`,
  `list_accessible_projects()`.
- **npa is multi-project.** `configure` can select several projects and writes
  each as its own stanza under `projects:` with a chosen `default_project`;
  switch with `-p <alias>`. Falls back to the manual tenant/project/region
  prompts when discovery is unavailable (no CLI / not authenticated / no
  results).
- **Object storage is opt-in in the discovery path.** `configure` sets up the
  Nebius connection and optional model/inference tokens; it only provisions an S3
  bucket + access key when you ask, so first-run is lighter.
- **New `npa agent setup`: a simple interactive deploy.** After `configure`, it
  picks one of your configured projects (prompting when there is more than one)
  and deploys the agent VM — no `--project-id`/`--tenant-id`/`--region` to type.
  `npa agent fresh-setup` remains for scripted deploys.
- **Documented the agent-VM AI Cloud credential model.** The VM authenticates to
  Nebius AI Cloud via an **attached `npa-agent` service account** (granted the
  tenant `editors` role); code on the VM mints short-lived IAM tokens from the
  Nebius metadata endpoint on demand — key-less and auto-rotating. No static AI
  Cloud key is stored on the VM.
- **Agent VM uses its attached service account for IAM, not a copied operator
  token.** Deploy no longer stages the operator's short-lived IAM token onto the
  long-lived agent VM (removed `NEBIUS_IAM_TOKEN` / `NPA_NEBIUS_IAM_TOKEN` /
  `TF_VAR_iam_token` / `NPA_REUSE_IAM_TOKEN` from `/opt/npa-agent/nebius.env`, the
  `/root/.npa/nebius-token` write, and the `agent-bootstrap` profile). The VM
  self-mints fresh IAM tokens from the attached SA's metadata/token-file sources
  via `get_iam_token()`, so it no longer goes stale and needs re-bootstrap. When
  the SA can't be attached (compute PermissionDenied retry), deploy now emits a
  loud warning that the VM can't self-mint tokens. **Unchanged on purpose:** the
  SA's S3 access keys (Nebius object storage is HMAC-based — a bearer IAM token
  cannot replace them) and the independent service API keys (Token Factory / HF /
  NGC). The shared `get_iam_token()` chain is untouched, so no-agent workbench/CI
  flows still resolve IAM via their CLI profile or injected token.
- **`npa configure` no longer prompts for a project alias.** The alias is derived
  automatically (existing default alias on re-run, otherwise the region);
  multi-project users rename in `~/.npa/config.yaml`. Discovered projects already
  used the Nebius project name as the alias.
- **Surface (don't swallow) a denied `editors`-group grant during bootstrap.**
  `bootstrap_environment`/`bootstrap_agent_environment` create/reuse the service
  account and, when adding it to the tenant `editors` group is denied, now emit a
  clear WARNING (with the SA id) instead of continuing silently. Without that
  role the SA — including the one attached to an agent VM — can authenticate but
  cannot manage Nebius AI Cloud resources; a tenant admin must grant it. Found
  via live dev-VM testing of the agent SA-creation path (SA + access-key creation
  confirmed working; only the tenant-level group grant needs admin rights).

### First-time-user cold-start fixes

- **`npa --version` (and `-V`) is now ~6x faster** (~0.72s → ~0.11s). The console
  script now points at a lightweight entry (`npa.cli.entry:main`) that answers a
  bare version request before importing `npa.cli.main`, which eagerly pulls in
  the whole command tree (boto3 / paramiko / rerun / numpy …). In addition,
  `npa/__init__.py` now imports its SDK convenience submodules lazily (PEP 562
  `__getattr__`) instead of eagerly, so any bare `import npa` — which every CLI
  invocation triggers — no longer pays for the full SDK surface. `import npa` and
  `from npa import convert` still work unchanged; the historical
  `NPA_SKIP_EAGER_IMPORTS` flag is now a no-op (imports are always lazy).
- `npa configure --interactive` no longer exits 0 having written nothing. When it
  cannot proceed (no authenticated Nebius CLI profile for provisioning) or is
  cancelled mid-flow (EOF/Ctrl-C), it now exits **non-zero** with actionable
  guidance. **Behavior change:** wrappers/CI that treated a cancelled or aborted
  `npa configure` as success will now see a failure. Setup guidance and the
  interactive prompts also link where to obtain the Hugging Face and NGC keys.
- Removed the dead **Nebius AI Cloud key (`NEBIUS_AI_CLOUD_KEY`)** credential. It
  had no consumer — no code ever used it as an auth header; Nebius AI Cloud
  compute/storage authenticates through the Nebius CLI profile (short-lived IAM
  access token) and the S3 access keys, and hosted inference uses the Token
  Factory key. `npa configure` no longer prompts for it, the `--show` template no
  longer lists it, and it is no longer staged into agent-VM credentials. A stale
  `NEBIUS_AI_CLOUD_KEY` already in `~/.npa/credentials.yaml` is harmless and left
  untouched (nothing reads it), and the `CredentialsConfig.ai_cloud_api_key` /
  `nebius_api_key` aliases and `resolve_ai_cloud_key` helper are removed.
- Added `npa workbench health preflight`: a PASS/WARN/FAIL/SKIP check over
  Hugging Face, NVIDIA NGC, Nebius object storage (S3), and Token Factory
  credentials (`--checks`, `--offline`, `--warn-only`, `--json`). Replaces the
  deprecated hidden `npa workbench health sim2real` in the README preflight
  guidance.
- Added `npa agent preflight` and moved the terraform-binary and SSH-key-pair
  checks (plus the Token Factory 503 warning) ahead of any cloud IAM side effects
  in `npa agent deploy`, so Route C prerequisites fail fast instead of mid-run.

### Agent deploy & README-path fixes

- **`npa configure` project discovery is now scoped to the active profile's
  tenant.** Enumerating every accessible tenant was O(tenants) serial CLI calls
  (hundreds of tenants → minutes and thousands of projects dumped). Discovery now
  lists projects in the profile's current tenant, offers a name/id filter with a
  display cap for large tenants, and defaults to the current project (never
  `all`). Other tenants are reachable by switching the Nebius profile.
- **The CPU agent VM no longer installs a workbench in cloud-init.** Deploy used
  `workbench_type=lerobot` on a `cpu-d3` host, whose cloud-init built LeRobot +
  EGL and failed on a driverless CPU image. A new minimal `workbench_type="agent"`
  branch does no workbench install (the agent stack is bootstrapped over SSH), and
  the VM boots on `ubuntu24.04-driverless` (region-portable; the CUDA default is
  absent in several regions).
- **Terraform now fails when cloud-init ends in `error`** instead of reporting a
  green deploy over a broken bootstrap (the `wait_for_cloud_init` poller exits
  non-zero on `error`, so the instance rolls back).
- **`npa agent preflight` checks the public-IPv4 quota** gate that deploy
  enforces, and **`npa agent destroy` can reclaim orphan agents** (no local
  record/state) by instance name + S3 remote state as long as the project is
  configured. `--token-factory-key` now continues the rest of `configure` instead
  of storing only the key and returning.
- **Chat memory is scoped per agent.** *Migration note:* the S3 chat-session
  prefix moved from `.../tenants/<tenant>/chat-sessions/` to
  `.../tenants/<tenant>/agents/<project>/<name>/chat-sessions/` to stop cross-VM
  context bleed. Chat history saved under the old (tenant-global) prefix is not
  deleted but **will not be listed** by an upgraded agent.
- Smaller deploy fixes: robust JSON-string `extra_ingress_ports` Terraform var
  (`jsondecode`, no intermittent "Invalid expression"); expanded `~` in the
  Terraform SSH key path/outputs; deploy resolves the project's real region
  (placement follows the project); `--all` pagination for compute-instance and
  bucket listings; and `npa agent verify-live` runs its local test gate with the
  current interpreter from the repo root, so it works from any cwd in the source
  checkout.

### SkyPilot submit robustness

- **Pin the isolated SkyPilot venv to a supported Python.** `npa skypilot
  bootstrap` created the venv with the current/`python3` interpreter with no
  version guard; on a fresh image where that is Python 3.14, `skypilot[kubernetes]
  ==0.12.2` installs a kubernetes client whose typing/imports fail, so submits
  broke. Bootstrap now validates the interpreter against SkyPilot's supported
  range (3.9–3.12): an explicit `--python`/`NPA_SKYPILOT_PYTHON` that is too new
  is rejected with guidance, and an unsupported default auto-selects a supported
  `python3.x` on PATH (or errors clearly when none exists). A version it can't
  determine is passed through unchanged.
- **A STOPPED (autostopped) managed-jobs controller no longer blocks submit.**
  The pre-launch health check treated anything but `UP` as unhealthy, so a
  stale/autostopped controller — which `sky jobs launch` simply restarts — burned
  the whole preflight timeout and failed the submit. STOPPED is now treated as
  ready (only transient states like `INIT` block), and the timeout error points
  at `sky down <controller>` to clear a genuinely stuck controller.

### Repo hardening

- The base `pip install -e npa` is now fully capable: the previous
  `data`/`lancedb`/`viz`/`server` extras (dataframe/reporting, LanceDB, Rerun
  viewer, FastAPI eval/agent server) are folded into the default install, so
  there is no longer an `npa` vs `npa[full]` split. Only GPU/simulation wheels
  (`npa[genesis]`, `npa[groot]`, `npa[sonic]`) and the optional agent adapters
  (`npa[agent-eval]`, `npa[agent-trace]`) remain opt-in. The `full`, `data`,
  `lancedb`, `viz`, and `server` extras are retained as empty no-op aliases so
  existing `npa[full]`/`npa[server]` commands keep working. Supersedes the
  earlier "base install is lightweight" split below.
- Shipped SkyPilot examples and cookbooks now use the `<your-registry-id>`
  placeholder instead of the first-party registry ID; a guardrail test keeps
  concrete registry IDs out of shipped examples.
- The base `pip install npa` is now lightweight (offline paths only); heavy
  dependencies moved to `npa[data]`, `npa[lancedb]`, `npa[viz]`, with
  `npa[full]` covering the previous monolithic install. Over-narrow version
  pins were relaxed and the previously undeclared `pydantic` dependency is
  declared.
- Added `npa.__version__`, a tag-driven Release workflow that builds and
  attaches sdist/wheel artifacts, and `docs/releasing.md`.

### Cosmos e2e

- Validated Cosmos end-to-end on Nebius via serverless `train --smoke`.
  Run ID: `w13-cosmos-e2e-20260521T233523Z`. Output artifact:
  `s3://${NPA_S3_BUCKET}/w13-cosmos-e2e/w13-cosmos-e2e-20260521T233523Z/checkpoint.json`.
- Closes the 7/8 -> 8/8 Workbench tool verification matrix gap for the
  artifact-bearing Cosmos CLI workflow.
- Known constraints remain documented in `docs/testing/e2e-serverless.md`:
  NIM/Triton are not implemented, `finetune` is a placeholder, and deferred
  visual-generation/rendering paths still depend on the container EGL/DRI gap.

- Validated Isaac Lab bring-your-own-fork path: image override (Run ID:
  `w10-byof-image-only-20260520T232650Z`) and image+command override (Run ID:
  `w10-byof-image-and-cmd-20260520T233113Z`). Worked example at
  `docs/workbench/cookbooks/byof-isaac-lab/`. Checkpoint + sentinel:
  `s3://${NPA_S3_BUCKET}/checkpoints/isaac-lab-byof/w10-byof-image-and-cmd-20260520T233113Z/`.
- Fixed Isaac Lab train command construction to call the RSL-RL training script with `--num_envs` and `--max_iterations`; added SkyPilot single-job and parallel sweep YAMLs plus the Isaac Lab RL runner.
- Added BYOVM post-deploy SSH endpoint strategy persistence and transient SSH tunnel routing for live workbench commands; fixed GR00T S3 env injection/auditing, shortened BYOVM auto public health fallback, printed normal-deploy Hugging Face access status, suppressed successful FiftyOne readiness curl noise, and made template tests cwd-independent.
- Implemented demo pre-staging CLI fixes for shared credential injection, shell-safe and Docker-safe env files, BYOVM project storage inheritance, Hugging Face gated-model validation, BYOVM SSH health fallback, live status/readiness reporting, Cosmos progress output, GR00T gated-model fail-fast handling, FiftyOne video ingestion, deploy dry-runs, credential env audits, and cross-tool smoke-test scaffolding.
- Preserved Genesis BYOVM staging fixes with tests: EGL fallback for multi-GPU demo generation, Docker group/device access for Genesis containers, and BYOVM storage credential reuse.
- Added structured implementation prompts for the 14 NPA CLI demo pre-staging fixes.

## W9-W10 - Workbench maturity sequence

- fix(sonic): default serverless training to H100, not L40S (W12 condensed commit)
- feat(skypilot): `npa skypilot bootstrap/status/verify` with isolated venv
  pattern (W11 condensed commit)
- Isaac Lab SkyPilot orchestration validated end-to-end via BYOF runs
  (W10 condensed commit; see `docs/workbench/cookbooks/byof-isaac-lab/`)
- BYOF mechanism validated: image override and command override surfaces;
  worked example with verified S3 artifacts (run IDs in cookbook)
- Removed SONIC routing entry from `CONTRIBUTING.md` Known Deviations
