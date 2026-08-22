---
name: nebius-infra
description: Use for Nebius runtime configuration, provision-if-absent setup, cluster, registry, storage, GPU routing, and credential assumptions that affect NPA runs.
---

# Nebius Infrastructure

## When To Use

Use this skill before running NPA workloads that need Nebius runtime settings,
object storage, container registry access, Kubernetes, or GPU routing. Also use
it when reviewing setup changes touching `npa configure`, `npa
provision-if-absent`, `~/.npa/config.yaml`, `~/.npa/credentials.yaml`, or
workflow environment variables.

## Procedure

1. Keep committed files public-repo safe. Never hardcode project IDs, tenant IDs,
   registry IDs, bucket names, VM IPs, private endpoints, or secrets.
2. Capture runtime configuration with `npa configure`. In CI or scripted
   contexts, use `--show` for the schema and write `~/.npa/config.yaml` plus
   `~/.npa/credentials.yaml` from placeholders:

   ```bash
   npa configure --show
   ```

   Interactive runs auto-provision S3 when a Nebius profile is present. Use
   `--no-provision` to supply existing object-storage credentials instead.

3. Ensure runtime resources with the additive-only setup command:

   ```bash
   npa provision-if-absent --project ci --dry-run --skip-validate --output-format json
   ```

   Real runs may ensure S3 and Kubernetes. Dry runs only resolve settings and
   print intended actions. The command must not teardown or replace resources.
   Storage setup declares `GetObject`, `HeadObject`, `PutObject`, `DeleteObject`,
   and `ListObjectsV2`. The minimum provider binding is bucket-scoped
   `storage.object-editor`, assigned through the project-scoped
   `npa-storage-object-editors-<exact-project-id>` group. A provider-verified existing tenant
   `editors` membership is accepted for backward compatibility. Creating that
   broader membership is allowed only as an explicit fallback after the provider
   reports the narrow role unsupported; unknown or unreadable IAM fails closed
   before key creation or probing. A newly created or changed binding enables
   typed propagation convergence even when an active key is reused. Persistent
   authorization denial is terminal and must retain redacted phase evidence.
   The explicit opt-in is `NPA_ALLOW_EDITORS_STORAGE_FALLBACK=1`; never set it
   merely to bypass an authorization or inventory failure.
   Credentials and IAM generations are authoritative only under
   `project_credentials.projects.<exact-project-id>`; top-level fields are a
   selected-project compatibility view. Migrate legacy global state only with
   exact ownership proof.
   Its canonical whole-path quota plan treats
   `compute.disk.size.network-ssd` as bytes and reports exact bytes plus GiB,
   independently of `compute.disk.count`. Unknown, missing, malformed, or
   contradictory disk-capacity evidence blocks mutation. The default cluster is
   1,151 GiB (128 GiB CPU + 1,023 GiB GPU); adding the default 100 GiB agent root
   disk makes the README whole path 1,251 GiB.

4. Use `--skip-s3` or `--skip-k8s` when the operator only wants one side
   checked. Use `--sky-smoke` only when live GPU validation is explicitly
   requested.

## Full Teardown

Run project-scoped cloud deletion before forgetting the project, then use the
explicit full local scope:

```bash
npa workflow cancel <run-id> --project <alias> --json
npa agent destroy --project <alias> --name <name> --yes
npa skypilot cleanup-controller --project <alias> --context <context> --yes
npa cluster down --project <alias> --force
npa storage bucket delete --project <alias> --yes --wait
npa storage service-account delete --project <alias> --dry-run
# If ownership provenance is missing, verify and explicitly attest the exact ID:
npa storage service-account reconcile --project <alias> --id <exact-id> --dry-run
npa storage service-account reconcile --project <alias> --id <exact-id> \
  --reason '<legacy NPA setup evidence>' --attest-npa-created --yes
npa storage service-account delete --project <alias> --dry-run
npa storage service-account delete --project <alias> --yes
# Optional: omit this to retain the project (the safe default).
npa destroy --project <alias> --all --delete-project --yes --json
npa configure --forget-project <alias>
npa cleanup --full --yes --project <alias>
```

The storage service-account command is ownership-gated: it only deletes the
exact `lerobot-training` identity whose successful create call NPA recorded for
that project, either in committed `storage_iam` or the crash-safe setup journal
written before the next provider step. Bucket credentials and storage IAM
provenance have separate
lifecycles: bucket deletion preserves the dedicated `storage_iam` record until
the account is deleted or conclusively absent, while a familiar name or legacy
saved ID remains evidence but is not proof of ownership. Agent bootstrap may
change the generic `nebius.service_account_id` without changing this record.
For legacy NPA-created residue, `service-account reconcile` verifies the exact
immutable ID, expected name, project, tenant, and selected profile, then stores
non-secret operator/when/reason attestation. It never deletes the resource;
the existing guarded `delete` command remains the only deletion path. Unresolved
evidence is journaled in the project stanza and blocks project forgetting.
Project deletion is separately opt-in and ownership-gated. `npa destroy --all`
retains the project unless `--delete-project --yes` is explicit. The deletion
adapter verifies exact project/tenant/region identity, requires one durable
`provider-create-response` NPA ownership record, inventories every NPA-managed
child class provider-side, writes deletion intent before mutation, deletes by
exact ID, and verifies NotFound afterward. Any external/shared or unproven
identity, remaining child, unsupported/unreadable/schema-invalid inventory,
permission failure, or pre-mutation receipt failure stops safely.
After an alias has been forgotten, the narrow recovery form
`npa destroy --receipt <id> --all --delete-project --yes --json` reads the exact
project/tenant/region/profile identity from the durable receipt and runs only
the same ownership-gated, provider-inventoried project phase. It never treats a
deleted Terraform backend or missing bucket credentials as live infrastructure.

Plain `npa cleanup --yes` keeps credentials; `--full --yes` additionally removes
the locally saved Hugging Face, Token Factory, and NGC entries and prunes only
empty NPA-owned local state plus exactly validated NPA Terraform residue. It does
not delete cloud resources, but full cleanup performs read-only storage-IAM
verification. Verified deletion/absence exits 0; missing trustworthy ownership
or provider/auth verification failure is partial cleanup and exits 2.

## Three-Tier Contract

- CLI: `npa configure` writes project/storage config and credentials; public
  workbench images default to GHCR, while `NPA_REGISTRY` and existing saved
  overrides select custom/private images. `npa
  provision-if-absent` ensures missing S3/Kubernetes resources or reports the
  dry-run plan.
- SDK: `npa.provisioning.provision_if_absent` and project settings via
  `npa.clients.config.resolve_project_storage` / `resolve_environment`.
- YAML: workflow YAML reads runtime values through environment variables such as
  `NPA_PROJECT_ID`, `NPA_TENANT_ID`, `NPA_REGION`, `NPA_REGISTRY`,
  `NPA_S3_BUCKET`, `NPA_STORAGE_ENDPOINT`, and AWS S3 keys. NPA-owned image
  defaults are GHCR-based and independent of Nebius project identity.

## GPU Routing

- H100: general training, CLIP embedding, detection, MJLab evaluation, Cosmos
  inference that does not need RT cores, and non-render throughput work.
- L40S: Isaac Lab and SONIC render validation on VM hosts.
- RTX PRO 6000 Blackwell on Kubernetes: Isaac Lab and SONIC render validation
  with NVIDIA GPU Operator mounted drivers.
- H100/H200 do not provide RT cores; do not route Isaac Lab or render validation
  there unless the task explicitly avoids rendering.
- Preemptibility changes the GPU capacity pool only. It never reduces node boot
  disk count or `compute.disk.size.network-ssd` byte requirements.

## Gotchas

- `nebius iam v2 access-key list --format json` may disclose access-key secret
  material in the external CLI's ordinary list response. NPA inventory uses
  CLI-side JSONPath to select identifiers/metadata before stdout capture and
  redacts provider errors; never replace that helper with a raw list or recommend
  raw JSON in diagnostics/docs.
- A default security group cannot be deleted directly. Full teardown deletes its
  parent network only when the NPA Terraform state proves ownership (`npa agent
  destroy` / `npa cluster down`). Existing, reused, shared, and unproven networks
  are preserved for their owner.
- Use `https://storage.eu-north1.nebius.cloud` for the current primary region.
- `npa cluster down` uses the selected cluster's saved kubeconfig for its
  best-effort PDB preview, sets exec auth to non-interactive, and adds Nebius
  `--no-browser`. Authentication/RBAC/API/kubeconfig preview failures are
  explained and never masquerade as verified drain safety. Preview uses one
  cluster-wide node/pod/controller/PDB inventory and the same eviction selector
  and placement semantics for every namespace. It reports cilium/CoreDNS/
  autoscaler/metrics-server and future matching blockers, including the one-node
  CPU-pool shape. It requests normal eviction first. Only an explicitly
  confirmed whole-cluster destroy with an exact provider-verified NPA
  project/context may temporarily remove the exact kube-system
  cilium-operator/CoreDNS/CoreDNS-autoscaler/metrics-server blockers. It records
  each decision and restores the exact specs if destroy aborts while the cluster
  remains. Shared clusters, node-pool operations, unverified contexts, and
  user/application PDBs are never weakened or force-deleted.
- Shared-controller cleanup requires the selected NPA project and its exact
  saved context (explicit flags take precedence), verifies stable provider and
  local identities, proves remote absence, durably checkpoints it, and only then
  removes matching local metadata. Never use an ambient kube current-context,
  the first SkyPilot profile, or a stale unrelated row.
- Teardown receipts live under `~/.npa/teardown-receipts/`, contain no secrets,
  and survive removal of project config/caches. Managed jobs must be audited and
  receipted before SkyPilot operational state is removed. List receipts with
  `npa cleanup --list-receipts`; prune only terminal aged receipts with the
  explicit `--prune-receipts --receipt-retention-days <days> --yes` path.
- Receipt v2 is the durable recovery identity after a project stanza is removed.
  Use the opaque ID printed before `configure --forget-project` with `--receipt`;
  exact flags override receipt fields, receipts override live config, and every
  overlap conflict fails closed before mutation. Never pass an arbitrary path.
- Alias-free agent, cluster, storage-IAM, and controller reconciliation journals
  into the existing project-ID-keyed receipt namespace; it never recreates an
  alias. Exact NotFound is absence, while RBAC/auth/network/parse uncertainty is
  unresolved and nonzero.
- An NPA-created disposable project's unique provider default topology can be
  removed with `npa network delete-project-default`; extra, shared, or
  non-default network inventory fails closed. Run it before guarded project
  deletion.
- With no cluster state/inventory and no NPA kubeconfig, `npa cluster down` is a
  no-op before binary lookup, authentication, Terraform init/provider download,
  or Kubernetes/RBAC calls. Real apply/destroy uses marked ephemeral
  `TF_DATA_DIR` scratch, never source `deploy/cluster/.terraform`, and keeps the
  tracked lock read-only. Checksum mismatch is an actionable hard failure; verify
  the provider source and reconcile with reviewed `terraform providers lock`
  output rather than bypassing checksum verification.
- Cleanup reports distinguish operational residue, retained audit receipts, and
  unresolved verification. A receipt file alone never makes a fully cleaned
  machine report operational residue.
- On Nebius VMs with an attached service account, IAM token resolution can use
  service-account token sources (`/mnt/cloud-metadata/token` and IMDS) even
  when `~/.nebius/config.yaml` is absent. Keep this as fallback behavior, not a
  substitute for explicit operator-machine profile setup. `get_iam_token()` keeps
  its full first-match chain — CLI profile → `NPA_NEBIUS_IAM_TOKEN`/
  `NEBIUS_IAM_TOKEN` → token files → metadata IMDS — so workbench/`configure`/CI
  on machines with **no** attached SA still resolve via the profile or an injected
  token. The attached-SA/metadata path is one option, never the only one; never
  couple workbench credential resolution to the agent.
- The **`npa-agent` VM** relies on this attached-SA path as its intended default:
  deploy attaches the SA (Terraform `main.tf`) and does **not** copy the
  operator's short-lived IAM token onto the VM (it would go stale). S3 access keys
  stay staged (object storage is HMAC-based; a bearer IAM token cannot replace
  them).
- For human authentication on a remote operator/dev VM, use
  `skills/atomic/vm-nebius-auth/SKILL.md`; the callback completes a CLI profile,
  after which IAM mints access tokens. Never transfer those tokens through chat.
- Official NPA GHCR development and release tags pull anonymously. Operator-
  controlled private registries require explicit exact-host
  `SKYPILOT_DOCKER_SERVER`, `SKYPILOT_DOCKER_USERNAME`, and
  `SKYPILOT_DOCKER_PASSWORD` credentials. Kubernetes users pre-create and
  explicitly reference a standard Docker config secret. NPA never mints a
  registry token and never creates or refreshes a provider-specific pull secret.
- SkyPilot task pods run in `default`; deployed workbench services run in
  `workbench`.
- Cached kubeconfig reuse is a success path for `provision-if-absent`; absence
  of a cached kubeconfig triggers Terraform only outside dry-run mode.
- A green cluster/GPU snapshot is not sufficient evidence for the later
  Kubernetes jobs-controller creation boundary. Workflow submit probes the exact
  selected context using the same `KUBECONFIG` environment SkyPilot receives and
  requires a stable `/readyz` series. Auth, RBAC, missing/wrong context,
  certificate/config, and identity failures stop immediately; only centrally
  classified transport, API 429, and appropriate API 5xx warm-up failures can
  enter reconciled bounded recovery.

## Verify

Run the CI-backed dry-run example:

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

That test invokes `npa configure --show` and `npa
provision-if-absent --dry-run --output-format json` against temporary config
paths and asserts the S3/Kubernetes actions are reported without live writes.
