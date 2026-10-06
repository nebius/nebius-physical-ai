---
name: gpu-capacity-quota-plan
description: Use before choosing a Nebius region, preset, or allocation type for a GPU VM, requesting a quota change, retrying a GPU launch across regions, classifying QuotaFailure versus NotEnoughResources, or cleaning up VMs, disks, and IP allocations left by failed launches.
---

# GPU Capacity and Quota Planning

The commands, `jq` filters, and observed API behavior are in the human guide:
[Plan GPU quota and capacity](../../../docs/workbench/gpu-capacity-quota-plan.md).
Follow its sections in this order.

## When To Use

- Before creating or starting a GPU VM directly with the `nebius` CLI, or when a
  launch needs quotas or capacity advice beyond what `npa cluster up` checks.
- When a launch fails and you need to decide between raising quota, waiting,
  moving region, or changing preset.
- When planning a retry loop across regions, and after it, to clean up.

## Procedure

1. **Size the demand** across every quota a GPU VM uses (guide: "Size the
   demand"), including `vpc.allocation.count` and the public IPv4 quota.
2. **Check headroom** in the tenant rows and in any project row with a limit.
   Demand must fit both. Stopped VMs keep their instance and disk quota until
   deleted.
3. **Read capacity advice** for the exact platform and preset. `available`
   decides; 0 or missing means a launch is unlikely at any availability level.
   Use only `DATA_STATE_FRESH` rows. Check quota usage before treating
   `AVAILABILITY_LEVEL_LIMIT_REACHED` as a hardware shortage. Advice is not a
   guarantee.
4. **Pick a target** only when quota and fresh advice both fit. Otherwise ask
   for quota (step 5), try another region or preset, or stop.
5. **Quota requests** go to a tenant admin through the web console. Hand the user
   the quota name, region, current limit, and requested limit. A
   `quota-allowance create` call only sets a project allowance; it never raises
   tenant quota.
6. **Launch with a run label** and record IDs. A launch succeeds only when the
   operation has finished without an error code and `instance get` reports
   `RUNNING`.
7. **Classify failures from operation details**, never from the status code:
   `QuotaFailure` names the quota to free or raise; `NotEnoughResources` means
   capacity, so retry later or move, and do not raise quota. Auth, image,
   cloud-init, and workload errors are neither.
8. **Retry within bounds**: one region at a time, at most one stopped VM per
   region (start it instead of creating another), three rounds unless the user
   sets a deadline, stop at the first confirmed `RUNNING` VM.
9. **Clean up by ID** after the user confirms the exact list: VMs, standalone
   disks, and allocations they no longer need. Verify each returns NotFound.

Switching from on-demand to preemptible goes through
`skills/atomic/gpu-allocation-fallback/SKILL.md` and needs the user's consent.
Clean up NPA-managed resources with `skills/atomic/teardown-and-cost/SKILL.md`.

## Gotchas

- `AVAILABILITY_LEVEL_LIMIT_REACHED` also appears when your quota is used up.
- JSON output omits zero values: no `available` field means zero VMs.
- A create that fails can leave a stopped VM behind; find it by its run label.
- A stopped VM still holds instance and disk quota, and its disks keep billing.
- Never delete by name, and never delete resources the user hasn't confirmed.

## Evidence Contract

Retain the snapshot time, commands, platform, preset, region, fabric,
allocation type, quota rows, advice rows, operation IDs with their detail codes,
and cleanup results. Redact project, tenant, instance, disk, allocation, IP, and
credential values before placing evidence in a repository.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py \
  npa/tests/guardrails/test_documentation_examples.py -q
```
