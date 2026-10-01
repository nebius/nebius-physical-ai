# Plan GPU quota and capacity

[Workbench docs](../README.md)

This workflow prevents the two expensive GPU-launch mistakes: starting before
all quota families have headroom, and treating a capacity advisor row as a
guarantee. It covers both NPA clusters and direct Nebius VMs.

## Define the demand first

For each candidate topology, collect:

| Requirement | Quota | Unit |
| --- | --- | --- |
| GPU | `compute.instance.gpu.<model>` | GPU count |
| VM | `compute.instance.count` | VM count |
| Disk | `compute.disk.count` | disk count |
| Network SSD | `compute.disk.size.network-ssd` | bytes |
| Public node IP | `vpc.ipv4-address.public.count` | IP count |

Nebius maps the model from the platform: `gpu-h100-sxm` uses
`compute.instance.gpu.h100`; `gpu-h200-sxm` uses `compute.instance.gpu.h200`.
An `8gpu-...` preset consumes 8 units of the matching GPU quota.

Check region support and pricing in the
[Nebius VM type](https://docs.nebius.com/compute/virtual-machines/types),
[Compute quota](https://docs.nebius.com/compute/resources/quotas-limits), and
[quota workflow](https://docs.nebius.com/overview/quotas) documentation. Treat
those defaults as context, not as your tenant's current value.

## Prove access and resolve scope

Run the NPA Nebius check before any provider query or mutation:

```bash
npa workbench health preflight --checks nebius --json
```

Resolve the profile that NPA will use. For a non-default profile, export it:

```bash
export NPA_NEBIUS_PROFILE="<profile>"
```

Set shell variables for the tenant, project, region, platform, and preset you
are testing. Region support follows the project's real placement scope; do not
trust a stale config stanza if quota evidence and project metadata disagree.

## Read the three evidence layers

The tenant quota is the approved regional ceiling. The project allowance is a
local assignment from that ceiling. The capacity advisor reports current
on-demand, preemptible, and reserved availability by platform and preset.

```bash
nebius quotas quota-allowance list --parent-id "$TENANT_ID" --all \
  --profile "$NPA_NEBIUS_PROFILE" --format json
nebius quotas quota-allowance list --parent-id "$PROJECT_ID" --all \
  --profile "$NPA_NEBIUS_PROFILE" --format json
nebius capacity resource-advice list --parent-id "$TENANT_ID" --all \
  --profile "$NPA_NEBIUS_PROFILE" --format json
```

Filter quota rows to the target region and the quota families in the table.
Filter advice rows to the exact region, platform, and preset. An exact-preset
match is stronger evidence than another preset for the same GPU family.

Record these fields before deciding: snapshot time, profile, tenant scope,
project scope, region, fabric, platform, preset, allocation class, quota name,
required amount, usage, limit, and availability level. Redact live identifiers
and addresses before sharing the record.

## Select a target

A candidate is viable when every quota family has enough headroom and the
advisor reports usable capacity for the intended allocation class. Preserve the
GPU family, GPU count, preset, boot-disk size, execution mode, and image
requirements when comparing regions.

On-demand is the default for long, interactive, or benchmark workloads.
Preemptible can provide another capacity pool, but the provider can reclaim it
at any time and it does not reduce disk or IP requirements. A reserved capacity
block must match the platform, fabric, and preset, and its current usage must
leave room for the required GPUs.

For NPA clusters, run the normal whole-path preflight:

```bash
npa cluster up --project "<project-alias>"
```

For a direct VM or one-off launch, repeat the same quota/capacity check before
the create call; NPA's cluster preflight does not run for a manual `nebius
compute instance create`.

## Request only the missing quota

Calculate the shortfall after subtracting usage and any project assignment.
Request the minimum amount that covers the planned topology plus the operator's
explicit retry margin. Do not copy another tenant's quota value.

In the Nebius web console, use **Administration -> Limits -> Quotas ->
Change quota**. Users in a group with the `admin` role can submit the request.
After approval, create a project allowance only when the project needs a local
restriction or an explicit assignment:

```bash
nebius quotas quota-allowance create \
  --parent-id "$PROJECT_ID" \
  --region "$REGION" \
  --name "$QUOTA_NAME" \
  --limit "$LIMIT" \
  --profile "$NPA_NEBIUS_PROFILE"
```

Re-read the tenant and project snapshots after approval. Keep a durable record
of the requested value, approval time, and assigned project allowance.

## Classify the failure

| Evidence | Meaning | Action |
| --- | --- | --- |
| `QuotaFailure` with a quota name | Quota, not physical capacity | Raise the named quota, free quota, or move to a region with headroom |
| `NotEnoughResources` or scheduler timeout with sufficient quota | Placement capacity | Use another region, fabric, allocation class, or wait; do not raise quota |
| `LIMIT_REACHED` advice | On-demand capacity is exhausted now | Prefer another region or allocation class; do not assume a launch will succeed |
| Auth, RBAC, image pull, checkpoint, runtime, or timeout | Not quota or capacity | Use the matching auth, image, checkpoint, or runtime failure path |

For repeated typed on-demand failures and compatible preemptible capacity, use
the [GPU allocation fallback](../../../skills/atomic/gpu-allocation-fallback/SKILL.md)
workflow. It requires explicit operator consent and an action-digest-bound
confirmation before changing allocation class.
