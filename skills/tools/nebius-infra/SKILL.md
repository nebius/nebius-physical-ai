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

4. Use `--skip-s3` or `--skip-k8s` when the operator only wants one side
   checked. Use `--sky-smoke` only when live GPU validation is explicitly
   requested.

## Full Teardown

Run project-scoped cloud deletion before forgetting the project, then use the
explicit full local scope:

```bash
npa storage bucket delete --project <alias> --yes --wait
npa storage service-account delete --project <alias> --dry-run
npa storage service-account delete --project <alias> --yes
npa configure --forget-project <alias>
npa cleanup --full --yes
```

The storage service-account command is ownership-gated: it only deletes the
exact `lerobot-training` identity whose successful create call NPA recorded for
that project. Bucket credentials and storage IAM provenance have separate
lifecycles: bucket deletion preserves the dedicated `storage_iam` record until
the account is deleted or conclusively absent, while a familiar name or legacy
saved ID remains evidence but is not proof of ownership. Agent bootstrap may
change the generic `nebius.service_account_id` without changing this record.
Plain `npa cleanup --yes` keeps credentials; `--full --yes` additionally removes
the locally saved Hugging Face, Token Factory, and NGC entries and prunes only
empty NPA-owned local state. It does not delete cloud resources.

## Three-Tier Contract

- CLI: `npa configure` writes runtime config and credentials; `npa
  provision-if-absent` ensures missing S3/Kubernetes resources or reports the
  dry-run plan.
- SDK: `npa.provisioning.provision_if_absent` and project settings via
  `npa.clients.config.resolve_project_storage` / `resolve_environment`.
- YAML: workflow YAML reads runtime values through environment variables such as
  `NPA_PROJECT_ID`, `NPA_TENANT_ID`, `NPA_REGION`, `NPA_REGISTRY`,
  `NPA_REGISTRY_ID`, `NPA_S3_BUCKET`, `NPA_STORAGE_ENDPOINT`, and AWS S3 keys.

## GPU Routing

- H100: general training, CLIP embedding, detection, MJLab evaluation, Cosmos
  inference that does not need RT cores, and non-render throughput work.
- L40S: Isaac Lab and SONIC render validation on VM hosts.
- RTX PRO 6000 Blackwell on Kubernetes: Isaac Lab and SONIC render validation
  with NVIDIA GPU Operator mounted drivers.
- H100/H200 do not provide RT cores; do not route Isaac Lab or render validation
  there unless the task explicitly avoids rendering.

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
  explained and never masquerade as verified drain safety. Full-cluster deletion
  relaxes only the exact managed `kube-system` PDBs for `coredns`,
  `cilium-operator`, and `metrics-server`; user/unknown PDBs are never patched.
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
- Nebius IAM registry tokens expire. If Kubernetes image pulls fail with `401
  Unauthorized` / `403 Forbidden` / `ErrImagePull`, refresh the registry pull
  secret in the namespace that owns the pod. **Do not hand-mint or hand-test the
  token per run** — use the shared, reusable helper below (any workflow can call
  and rely on it; it is the single source of truth).

### Reusable registry token refresh (`npa.clients.nebius_auth`)

`npa.clients.nebius_auth.mint_nebius_iam_token(...)` is the canonical way to get
a fresh short-lived Nebius IAM token. It is robust to a stale/ambient
`NEBIUS_IAM_TOKEN` (or `NEBIUS_IAM_TOKEN_FILE`) in the environment — which
otherwise makes the bare `nebius iam get-access-token` skip a real exchange
("token from NEBIUS_IAM_TOKEN env is used"), silently leaving pull secrets stale
and causing later `403 Forbidden` pulls. The helper strips the ambient token,
performs a profile-scoped exchange (`NPA_NEBIUS_PROFILE` / `NEBIUS_PROFILE`), and
only falls back to the ambient token if the exchange fails.

- `registry_auth.mint_nebius_registry_token` (sim2real / BYOF),
  `workbench.sonic.workflow._mint_nebius_registry_token`, and
  `ServerlessClient._mint_registry_token` all delegate here — fix once, fixed
  everywhere.
- To refresh a Kubernetes pull secret before a job, call
  `npa.workflows.sim2real.registry_auth.ensure_registry_pull_secret_for_images(
  *images, namespace=..., kubeconfig=..., k8s_context=...)`.
- Operators no longer need to `unset NEBIUS_IAM_TOKEN` before a BYOF / sim2real
  run; the refresh mints a fresh profile token regardless.
- Dev/operator VM Docker access to a private Nebius Container Registry image
  does not automatically authenticate fresh SkyPilot worker VMs. Direct Nebius
  burst jobs need SkyPilot Docker-login env/secrets injected before image pull;
  `npa burst submit-yaml` handles this for `cr.*.nebius.cloud` images by minting
  a short-lived IAM token with `nebius iam get-access-token`.
- SkyPilot task pods run in `default`; deployed workbench services run in
  `workbench`.
- Cached kubeconfig reuse is a success path for `provision-if-absent`; absence
  of a cached kubeconfig triggers Terraform only outside dry-run mode.

## Verify

Run the CI-backed dry-run example:

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

That test invokes `npa configure --show` and `npa
provision-if-absent --dry-run --output-format json` against temporary config
paths and asserts the S3/Kubernetes actions are reported without live writes.
