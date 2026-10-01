---
name: gpu-capacity-quota-plan
description: Use before choosing a Nebius GPU region or allocation class, asking for a quota increase, provisioning a GPU VM or node group, or classifying QuotaFailure versus NotEnoughResources.
---

# GPU Capacity and Quota Planning

## When To Use

Use this workflow before a GPU deploy, a direct VM launch, a cluster scale-up,
or a multi-region capacity search. It also applies when NPA reports a quota
shortfall, when the provider reports insufficient physical capacity, or when a
run fails only after Terraform or the scheduler has started.

## Procedure

1. Prove provider access with the existing health gate:

   ```bash
   npa workbench health preflight --checks nebius --json
   ```

2. Resolve the real profile, project, tenant, and region. Quota reads use the
   active or `NPA_NEBIUS_PROFILE` profile; a config stanza's stated region is
   not authoritative.
3. Compute the exact demand before reading quotas:
   - GPU quota counts GPUs, not nodes. One `8gpu-...` preset needs 8 units.
   - Map `gpu-<model>...` to `compute.instance.gpu.<model>`, for example
     `gpu-h200-sxm` to `compute.instance.gpu.h200`.
   - Check `compute.instance.count`, `compute.disk.count`,
     `compute.disk.size.network-ssd` in bytes, and
     `vpc.ipv4-address.public.count` when public node IPs are requested.
   - Keep GPU, instance, disk, Network SSD, and public-IP requirements together.
     A preemptible GPU does not reduce disk or IP quota.
4. Read tenant, project, and regional evidence. Do not infer a quota from the
   public default value:

   ```bash
   nebius quotas quota-allowance list --parent-id "$TENANT_ID" --all \
     --profile "$NPA_NEBIUS_PROFILE" --format json
   nebius quotas quota-allowance list --parent-id "$PROJECT_ID" --all \
     --profile "$NPA_NEBIUS_PROFILE" --format json
   nebius capacity resource-advice list --parent-id "$TENANT_ID" --all \
     --profile "$NPA_NEBIUS_PROFILE" --format json
   ```

   Filter the first two responses by quota name and region, and the third by
   region, platform, and preset. Prefer the exact preset over a preset fallback.
5. Select a launch target only when the required quota has headroom **and** the
   allocation-class advice is usable. A capacity advisor row is evidence, not a
   guarantee. Physical scheduling can still return `NotEnoughResources`.
6. For a cluster, run NPA's whole-path preflight through `npa cluster up`,
   `npa agent preflight`, or the relevant dry-run surface. NPA reuses the same
   resolved plan for quota arithmetic and mutation, rather than checking one
   subset before apply and another during execution.
7. If provider quota is short, request only the exact missing headroom. The web
   console path is **Administration -> Limits -> Quotas -> Change quota**. This
   provider request requires an admin role and is separate from project
   allowance assignment.
8. After provider approval, set a project allowance only when the project needs
   a local restriction or an explicit assignment. A null project limit normally
   means the tenant limit governs. Record every allowance change before the next
   mutation.
9. Classify failure evidence before changing the target:
   - `QuotaFailure` plus a quota name: raise quota, free quota, or move to a
     region with headroom.
   - `NotEnoughResources` / scheduler timeout with sufficient quota: capacity
     placement, not quota.
   - Auth, RBAC, image pull, checkpoint, runtime, and timeout failures: do not
     count as quota or capacity failures.
10. For repeated typed on-demand failures, load
    `skills/atomic/gpu-allocation-fallback/SKILL.md`. Never switch allocation
    class without operator consent and an action-digest-bound confirmation.

## Evidence Contract

Retain the snapshot timestamp, provider command, requested platform, preset,
region, fabric, allocation class, quota name, required amount, usage, limit,
and outcome. Redact live project, tenant, instance, bucket, registry, endpoint,
IP, and credential values before placing evidence in a repository.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```
