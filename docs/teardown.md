# Tear it all down

Stopping spend is an **ordered sequence**, and skipping a step leaves a hung
job, a credential, or a cache behind:

> cancel managed jobs → destroy the agent → remove the shared controller →
> destroy the cluster → delete the bucket → remove NPA-owned storage IAM →
> drop the project entry → clear local state

`npa cleanup` prints a report plus the exact runbook for your machine. Start
there if you are not sure what is still running.

## Two ways in

**One project, one plan.** `npa destroy --project <alias> --all` is the
project-scoped path. It is read-only unless you pass `--yes`. Execution journals
the immutable project identity and the complete phase plan, continues
independent cleanup when one phase fails, and blocks dependent phases rather
than guessing. It **retains the Nebius project by default**.

**Command by command.** The sequence below is the exact recovery surface when a
combined run stopped partway.

```bash
npa workflow cancel <run-id> --project <alias> --json
npa agent destroy --project <alias> --name <name> --yes
npa skypilot cleanup-controller --project <alias> --context <context> --yes
npa cluster down --project <alias> --force
npa storage bucket delete --project <alias> --yes --wait
npa storage service-account delete --project <alias> --dry-run
# Only when the previous command reports missing ownership provenance:
npa storage service-account reconcile --project <alias> --id <exact-id> --dry-run
npa storage service-account reconcile --project <alias> --id <exact-id> \
  --reason '<legacy NPA setup evidence>' --attest-npa-created --yes
npa storage service-account delete --project <alias> --dry-run
npa storage service-account delete --project <alias> --yes
# If validation created a project-local registry, delete its exact artifact DAG
# and registry using the immutable ID/name recorded at creation:
npa registry delete --project <alias> --project-id <project-id> \
  --tenant-id <tenant-id> --id <registry-id> --name <registry-name> --yes
# NPA-created disposable projects may contain one provider-created default
# topology. This command refuses any extra, shared, or non-default topology:
npa network delete-project-default --project <alias> --project-id <project-id> \
  --tenant-id <tenant-id> --yes
# Optional and ownership-gated; omit to retain the project (the safe default):
npa destroy --project <alias> --all --delete-project --yes --json
npa cleanup --full --yes --project <alias>
npa configure --forget-project <alias>
```

## Cloud spend vs local clutter

`npa cleanup` only touches **local** state. Plain `npa cleanup --yes` keeps your
credentials. The explicit `npa cleanup --full --yes` scope also removes saved
Hugging Face, Token Factory, and NGC credentials, removes only exactly validated
NPA Terraform caches, and prunes an empty `~/.npa` tree. It performs a read-only
storage-IAM verification but **never deletes cloud resources**.

Cloud deletion is always a separate, explicit command.

## Finishing a teardown you can no longer address by alias

`npa configure --forget-project` durably writes and prints an opaque **receipt
id** before it rewrites configuration. If teardown must continue past that
point, stay inside NPA and select the same immutable identity explicitly:

```bash
RECEIPT=<id printed by npa configure --forget-project>
npa agent destroy --receipt "$RECEIPT" --name <name> --yes
npa skypilot cleanup-controller --receipt "$RECEIPT" --context <context> --yes
npa cluster down --receipt "$RECEIPT" --context <context> --force
npa storage service-account delete --receipt "$RECEIPT" --id <exact-id> --dry-run
npa workflow cancel <run-id> --receipt "$RECEIPT" --json
# Optional, after every exact child cleanup has converged:
npa destroy --receipt "$RECEIPT" --all --delete-project --yes --json
```

If the alias was already forgotten and only the project deletion remains,
`npa destroy --receipt <id> --all --delete-project --yes` exposes just that
narrow phase and recovers the exact project/tenant/region identity from the
durable receipt. It does not reopen a deleted Terraform backend.

**Identity precedence is deterministic:** exact flags, then the selected
receipt, then live configuration. Any overlapping mismatch is unsafe and fails
before provider or Terraform mutation. NPA never substitutes a default alias, a
current Kubernetes context, or an unrelated SkyPilot profile.

## The confirmation contract

`agent destroy`, `storage bucket delete`, and `storage service-account delete`
share one contract:

| Situation | Behavior |
| --- | --- |
| Interactive terminal, no `--yes` | Prompts |
| Non-interactive, no `--yes` | Refuses, exit 1 |
| Explicit `--yes` | Proceeds without prompting |
| Read-only and `--dry-run` paths | Always available, no confirmation |

A `--json` confirmation refusal contains one machine-readable document.

## What each phase guards

### Deleting a project

`--delete-project --yes` deletes the exact project id **only** when one unique
durable NPA provider-create record proves ownership *and* strict provider
inventories prove every managed child class empty. External, shared, or unproven
projects, nonempty inventories, unreadable or schema-invalid evidence,
permission failures, and identity conflicts are all refused. `NotFound` is
repeat-safe verified absence. Deletion waits for eventual provider absence
instead of treating the first still-visible post-delete observation as failure.

Registry and default-network deletion require that **same** unique durable
creation proof. Registry teardown inventories and removes immutable artifact ids
before deleting the exact registry. Default-network teardown accepts only one
`default-network`, its one linked `default-subnet-*`, and its provider-marked
`default-security-group-*`; mixed or additional inventory fails closed.

### Storage IAM

`npa storage service-account delete` removes `lerobot-training` only when the
successful create response is present in NPA's final ownership record or its
crash-safe setup journal. A display-name match, a legacy id, a reused account, a
conflicting record, or a user-managed account is never enough.

Results are explicit: verified absence or deletion exits 0. Missing trustworthy
ownership, or a provider/auth verification failure, reports `Partial cleanup`
and **exits 2**. A project-scoped, non-secret `storage_iam_verification_required`
journal keeps the exact candidate visible and blocks `--forget-project` until
there is provider-verified absence or a guarded deletion. **Do not treat exit 2
as success.**

`reconcile` verifies the immutable id, expected name, project, tenant, and
selected CLI profile, then records a non-secret operator/when/reason
attestation. It never deletes IAM itself and never treats a display name as
ownership. The `delete` that follows still performs the access-key inventory and
guarded delete. Both are restart-safe.

### Bucket deletion

Deleting the bucket removes secret material, so it first writes a
project-scoped, **non-secret cleanup tombstone** containing immutable
service-account and access-key ids, ownership evidence, and the storage creation
outcome. That provenance survives until the exact IAM identity is deleted or
verified absent.

### The shared jobs controller

Controller cleanup has shared blast radius, so it accepts only an explicit (or
unambiguously selected) NPA project plus that project's exact saved context. It
cross-checks immutable project/cluster identity, deletes remotely through the
SkyPilot abstraction, independently proves the controller pods absent, writes
the remote-absence checkpoint, and only then converges local SkyPilot metadata.

Authentication, RBAC, connectivity, stale, mismatched, or ambiguous identity all
preserve local state for an exact retry. An unrelated current context or a stale
SkyPilot profile is never used as a fallback.

The shared controller has one global immutable owner. The accelerator-gated
`npa provision-if-absent` transaction **binds it automatically**, after the exact
project/context/provider cluster identity is durable and before GPU readiness or
submission. `npa skypilot bind-controller --project <alias> --context <context>`
is therefore only for adopting an already-live cluster outside that flow. It runs
the same provider identity checks and rejects missing, destroyed, rolled-back, or
replaced clusters. Cross-project use is refused. `--rebind` is allowed only after
the managed-job queue is proven terminal; changing an alias for the same
project/cluster ids is not a rebind.

### Cluster teardown

`npa cluster down` uses the kubeconfig saved for the selected NPA cluster and
forces its credential plugin into non-interactive/no-browser mode for the
best-effort drain preview. It distinguishes authentication, RBAC, kubeconfig,
and API failures, and still attempts the Terraform destroy.

For a full managed-cluster deletion it takes one cluster-wide inventory of
nodes, pods, controllers, and PodDisruptionBudgets with eviction-relevant
selector/placement semantics. This catches system workloads such as
`cilium-operator`, CoreDNS, the CoreDNS autoscaler, and `metrics-server` —
including the common one-CPU-node-pool case where a replacement simply cannot be
scheduled. NPA first requests normal eviction. Only for an explicitly confirmed
full destroy, whose exact NPA project/context/cluster identity is verified, may
it temporarily remove those exact four `kube-system` PDBs; it snapshots their
specs and restores them if destroy aborts while the cluster still exists. Shared
clusters, node-pool operations, unverified contexts, and every user or
application PDB are never weakened or force-deleted.

When no cluster state or inventory and no NPA kubeconfig exist, `cluster down`
is a **true no-op**: it does not authenticate, initialize Terraform, download
providers, or call Kubernetes. Real Terraform runs place provider/module data in
exact NPA-owned temporary scratch and remove it on success or failure, so they
do not populate `deploy/cluster/.terraform`. `npa cleanup --full --yes` detects
both a failed scratch cleanup and the legacy source-checkout cache. A provider
checksum mismatch stays a hard failure: NPA keeps `.terraform.lock.hcl`
read-only and prints a reviewed `terraform providers lock` reconciliation
command rather than bypassing verification.

With `--receipt` or exact `--project-id` / `--cluster-id`, the same no-state
decision happens before Terraform: provider-verified absence exits 0,
insufficient identity fails once with the required selectors, and a present
cluster without recoverable owned state fails closed.

### Cancelling a run

Workflow cancellation reports `NOT_SUBMITTED` only from durable planned or
reserved evidence. If submission began and S3 or SkyPilot verification is gone,
it stays `VERIFICATION_UNAVAILABLE` with exit 2 — never silently "already gone".

## Audit receipts

Every destructive phase writes a versioned, atomic, non-secret receipt under
`~/.npa/teardown-receipts/` **before** deleting the local evidence needed to
audit it. Managed jobs are checked and receipted before SkyPilot state is
removed; active or uncertain jobs preserve that state. Receipts survive project
and config removal, are not operational residue, and keep completed phases from
reverting to `unknown` on an idempotent retry.

```bash
npa cleanup --list-receipts
npa cleanup --prune-receipts --receipt-retention-days <days> --yes
```

Prune only old, fully terminal receipts, and only explicitly.

JSON reports expose `operational_residue_present`, `audit_receipts_retained`,
and `verification_unresolved` **separately**. A retained receipt alone never
turns `local_state: fully_cleaned` into `residue_present`; an unresolved action
recorded inside it is operator action, not local operational residue.

## Removing npa itself

Ordinary cleanup deliberately leaves the invoking NPA environment alone. To
remove only a supported repository-local `.venv`, preview `npa uninstall`; the
actual deferred removal requires both `--remove-environment --yes`. A one-time
helper waits for NPA to exit and revalidates the exact path, inode, marker, and
receipt nonce before deleting it. Source, `.git`, credentials, user data, and
unrelated caches stay outside the plan.

## Related

- [Known footguns](workbench/troubleshooting/known-footguns.md) — the teardown symptoms people actually hit
- [Physical AI Data Factory deploy runbook §8](workbench/guides/physical-ai-data-factory-deploy.md) — teardown in the context of one full run
- [teardown-and-cost skill](../skills/atomic/teardown-and-cost/SKILL.md) — the operator/agent version of this page
- [Run lifecycle](run-lifecycle.md) — run identity and status semantics that cancellation depends on
