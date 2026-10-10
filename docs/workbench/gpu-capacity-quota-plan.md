# Plan GPU quota and capacity on Nebius AI Cloud

[Workbench docs](README.md)

Use this guide before you launch a GPU virtual machine (VM) on Nebius AI Cloud.
It shows how to check that a region can take the VM, request only the quota you
are missing, retry across approved regions without piling up stopped VMs, tell
a quota failure from a capacity failure, and clean up after failed attempts.

The commands use the `nebius` CLI and `jq`. For Workbench deploys and their
preemptible defaults, see [preemptible VMs](preemptible-vms.md); for NPA
clusters, see [NPA clusters](#npa-clusters). Behavior that the Nebius docs don't
describe is collected in [observed API responses](#observed-api-responses).

## Before you start

- Install and configure the [Nebius AI Cloud CLI](https://docs.nebius.com/cli/install).
- Find your tenant ID with `nebius iam tenant list`. Reading capacity advice
  requires the `viewer` role in the tenant, and changing quotas requires the
  `admin` role
  ([capacity advisor](https://docs.nebius.com/compute/virtual-machines/capacity-advisor),
  [quotas](https://docs.nebius.com/overview/quotas)).
- Pick a project in each region authorized for the search. Platforms and presets
  differ by region and project
  ([platforms and presets in a project](https://docs.nebius.com/compute/virtual-machines/list-platforms)).

Record the authorized regions, projects, fabrics, platforms, presets, and
allocation type. Propose alternatives with read-only checks; obtain approval
before launching outside that scope. Reuse authorization already given for the
search and its cleanup.

Set the values you are testing:

```bash
TENANT_ID="<tenant-id>"
PROJECT_ID="<project-id>"
REGION="<region>"
PLATFORM="<platform>"        # for example gpu-h200-sxm
PRESET="<preset>"            # for example 8gpu-128vcpu-1600gb
RUN="<run-name>"             # label value for VMs this search creates
```

## Size the demand

One GPU VM draws on several quotas at once:

| Quota | What counts against it |
| --- | --- |
| `compute.instance.gpu.<model>` | GPUs of one model. The model is the second part of the platform name: `gpu-h200-sxm` uses `compute.instance.gpu.h200`. An `8gpu-...` preset uses 8. Each row's description says which VMs count. |
| `compute.instance.count` | VMs |
| `compute.instance.preemptible.count` | Preemptible VMs |
| `compute.disk.count` | Boot and secondary disks |
| `compute.disk.size.<disk-type>` | Disk capacity in bytes for one disk type, for example `network-ssd` |
| `compute.gpucluster.count` | GPU clusters for InfiniBand |
| `vpc.allocation.count` | IP address allocations. Each IP address assigned to a VM is an allocation, so a VM with a public IP uses two. |
| `vpc.ipv4-address.public.count` | Public IPv4 addresses |

For example, one 8-GPU H200 VM with a 1 TiB Network SSD boot disk and a public
IP needs 8 of `compute.instance.gpu.h200`, 1 of `compute.instance.count`, 1 of
`compute.disk.count`, 1099511627776 bytes of `compute.disk.size.network-ssd`, 2
of `vpc.allocation.count`, and 1 of `vpc.ipv4-address.public.count`. The row
`compute.instance.non-gpu.vcpu` applies only to VMs without GPUs.

The default public IPv4 quota is small
([Virtual Networks quotas](https://docs.nebius.com/vpc/resources/quotas-limits)),
so a search that gives every attempt a public IP can run out of addresses before
it runs out of GPUs.

## Read quota headroom

List the rows for your region:

```bash
nebius quotas quota-allowance list --parent-id "$TENANT_ID" --all --format json \
  | jq -r --arg region "$REGION" '.items[]
      | select(.spec.region == $region)
      | select(.metadata.name | test("^compute\\.instance\\.(count|preemptible\\.count|gpu\\.)|^compute\\.disk\\.(count|size\\.)|^compute\\.gpucluster\\.count|^vpc\\.(allocation|ipv4-address\\.public)\\.count"))
      | [.metadata.name, .status.unit, (.spec.limit // "none"), (.status.usage // "0"), .status.description]
      | @tsv'
```

Run it again with `--parent-id "$PROJECT_ID"` for the project rows. The columns
are name, unit, limit, usage, and description. JSON output leaves out fields
whose value is zero, so the filter prints a missing usage as `0`. The filter
prints `none` when a row carries no limit at all.

Headroom is a row's limit minus its usage. The additional demand of the next
operation must fit the tenant headroom, and also the project headroom for any
project row that has a limit.
Headroom equal to the demand is enough for one VM. Each additional VM you create
while searching needs its own headroom until it is deleted: VMs count against
`compute.instance.count` and their disks count against the disk quotas until
the VM is deleted, even when it is stopped
([VM lifecycle](https://docs.nebius.com/compute/virtual-machines/lifecycle)).

For a new VM, count its instance, new disks and IP allocations, and compute
resources. For a start of a retained stopped VM, its existing instance, disks,
and retained allocations are already included in usage: do not count them
again. Check headroom for the compute resources acquired on start, including
GPUs, and any new resources you add. For example, zero free instance or disk
quota does not by itself rule out starting a VM that already owns those
resources.

## Read capacity advice

The capacity advisor reports, for each region, fabric, platform, and preset, how
many VMs you can launch based on your quotas and current physical capacity. It is
calculated for the tenant:

```bash
nebius capacity resource-advice list --parent-id "$TENANT_ID" --all --format json \
  | jq -r --arg platform "$PLATFORM" --arg preset "$PRESET" '.items[]
      | select(.spec.compute_instance.platform == $platform)
      | select(.spec.compute_instance.preset.name == $preset)
      | [.spec.region, .spec.fabric,
         (.status.on_demand.available // 0), (.status.on_demand.limit // 0),
         .status.on_demand.availability_level, .status.on_demand.data_state,
         .status.on_demand.effective_at]
      | @tsv'
```

Replace `on_demand` with `preemptible` or `reserved` for the other allocation
types. For each row:

- `available` is the maximum number of VMs you can launch. It decides the
  question: when it is 0 (or missing), a launch is unlikely to succeed, whatever
  `availability_level` says. If you try anyway, follow the retry procedure below.
- `limit` is your quota.
- `availability_level` is `AVAILABILITY_LEVEL_HIGH`, `AVAILABILITY_LEVEL_MEDIUM`,
  `AVAILABILITY_LEVEL_LOW`, or `AVAILABILITY_LEVEL_LIMIT_REACHED` ("launch
  impossible"). Because `available` is capped by your quota, check your quota
  usage before you treat `AVAILABILITY_LEVEL_LIMIT_REACHED` as a hardware
  shortage.
- `data_state` is `DATA_STATE_FRESH`, `DATA_STATE_STALE`, or `DATA_STATE_UNKNOWN`.
  Act only on fresh rows, and read the advice again just before you launch.

Advice describes availability at `effective_at`; it doesn't guarantee capacity
when you create the VM. If you can't read advice for the tenant, ask someone with
the `viewer` role for the rows you need, or rely on the quota reads and the
retry procedure below. Treat unavailable or stale advice as unknown; record
that limitation before an attempt within the authorized scope.

## Choose a target

A region is a candidate when the additional demand of the planned create or
start fits every quota and its fresh advice row shows `available` at or above
the number of VMs you need. For a start, exclude resources already charged to
the stopped VM as described in [quota headroom](#read-quota-headroom).

If no region qualifies:

- If quota is the limit, [request more quota](#request-more-quota).
- If capacity is the limit, try another region or a similar preset, as the
  [Not enough resources](https://docs.nebius.com/compute/virtual-machines/not-enough-resources)
  page suggests, or contact Nebius sales to reserve capacity. Check alternatives
  against the authorized scope before launching. You can also read advice again
  during the [retry procedure](#retry-across-regions).
- Preemptible VMs have their own quota and their own advice rows, and Compute can
  stop them at any time
  ([preemptible VMs](https://docs.nebius.com/compute/virtual-machines/preemptible)).
  Switch allocation type only when the workload tolerates interruption and the
  person who owns it agrees.

## Request more quota

Only users in a group with the `admin` role can change tenant quotas, through
the web console as described on the [quotas](https://docs.nebius.com/overview/quotas)
page. If you are not an admin, send a tenant admin the quota name, region,
current limit, requested limit, and the VMs you plan to run.

`nebius quotas quota-allowance create --parent-id "$PROJECT_ID" --region "$REGION" --name <quota> --limit <value>`
sets a project's allowance for one quota. It does not raise the tenant quota.

Request the shortfall for your planned VMs plus any retry margin the operator
has authorized.
Nebius can automatically reduce quotas that stay well below their limit for a
long period, after an email notice and a grace period.

## Launch and confirm

Label every VM you create during a search so you can find it again, and record
its ID:

```bash
nebius compute instance create ... --labels purpose=gpu-search,run="$RUN"
```

By default the CLI waits for the operation to finish; with `--async` it returns
the operation instead. A failed create can leave a stopped VM behind (see
[observed API responses](#observed-api-responses)).
List the VMs that carry your run label to recover their IDs:

```bash
nebius compute instance list --parent-id "$PROJECT_ID" --all --format json \
  | jq -r --arg run "$RUN" '.items[] | select(.metadata.labels.run == $run)
      | [.metadata.id, .metadata.name, .status.state] | @tsv'
```

Count a launch as successful only when its operation has finished without an
error code and `nebius compute instance get --id "$INSTANCE_ID"` reports
`RUNNING`.

## Classify a failed launch

Read the VM's operations and classify from the error details:

```bash
nebius compute instance operation list --resource-id "$INSTANCE_ID" --all --format json \
  | jq -r '.operations[]
      | [.created_at, .description,
         (if .finished_at == null then "IN_PROGRESS"
          elif (.status.code // 0) == 0 then "OK"
          else ([.status.details[]?.code // empty] | join(",")) as $codes
            | if $codes == "" then "code \(.status.code)" else $codes end
          end)]
      | @tsv'
```

If no VM was created, read the same fields from the create operation with
`nebius compute instance operation get --id "$OPERATION_ID"`. To see what each
failure asked for:

```bash
nebius compute instance operation list --resource-id "$INSTANCE_ID" --all --format json \
  | jq -r '.operations[] | select((.status.code // 0) != 0) | .status.details[]?
      | (.quota_failure.violations[]? | "QuotaFailure quota=\(.quota) limit=\(.limit) requested=\(.requested)"),
        (.not_enough_resources.violations[]? | "NotEnoughResources \(.resource_type) \(.requested): \(.message)")'
```

| Detail code | Meaning | Next step |
| --- | --- | --- |
| `QuotaFailure` | A quota is too small. The violation names the quota, its limit, and the amount requested. | Free usage of that quota, raise it, or use a region with headroom. Retrying unchanged won't help. |
| `NotEnoughResources` | The region lacks hardware for the preset. | Retry later, or try another region, fabric, or preset. Don't raise quota for this. |

Classify from the detail code, not the top-level status code. A VM in `ERROR`
can't be started again; delete it and create a new one
([VM lifecycle](https://docs.nebius.com/compute/virtual-machines/lifecycle)).
Authentication, permission, image, cloud-init, and workload failures are neither
quota nor capacity problems.

## Retry across regions

Use the operator's retry cadence and any limits they explicitly set. Do not
introduce time, cost, or attempt limits when none were requested. Keep an
inventory of attempts and reuse stopped VMs to avoid accumulating resources:

1. Read quota and advice again at the start of each round.
2. Work through candidate regions in the authorized scope, one region at a time.
3. Reuse a stopped VM from an earlier attempt in that region, and start it
   (`nebius compute instance start --id "$INSTANCE_ID"`) instead of creating
   another. Check start demand rather than charging its retained resources
   twice. The [Not enough resources](https://docs.nebius.com/compute/virtual-machines/not-enough-resources)
   page covers restarting a stopped VM. A stopped VM still holds quota, and its
   disks keep billing.
4. After a `QuotaFailure`, skip that region until the named quota has headroom
   again, through lower usage or a higher limit.
5. Refresh advice before another attempt, follow the operator's retry cadence,
   and honor any explicit stop condition or cancellation.

Write down the interval, authorized regions, platform, preset, allocation type,
and any operator-supplied limits before you start. Append one line per attempt
with the time, region, instance ID, operation ID, and detail code. Run the loop
under a process supervisor or in a terminal session that stays open; a background
process started from a short-lived session can be killed when that session ends.
Stop at the first `RUNNING` VM and confirm its state again before you hand it to
a workload. If attempts ever run in parallel, check every other recorded VM
after the first success, and stop or delete any extra VM that is starting or
running.

## Clean up after a search

Keep the successful VM and the resources it needs. Clean up unused attempts
within the run's authorized scope. Stopped VMs keep their instance and disk
quota, and their disks keep billing
([deleting a VM](https://docs.nebius.com/compute/virtual-machines/delete)). Build
the list from your attempt log and the run-label listing above, and use IDs, not
names. Exclude the successful VM and any other VM still in use, even if they
carry the run label.

For each VM, run `nebius compute instance get --id "$INSTANCE_ID"` and note:

- its state;
- its disks: `status.disk_attachments[]` lists each attached disk's `id` and
  whether it is VM-managed (`is_managed`);
- its IP allocation IDs (`status.network_interfaces[].ip_address.allocation_id`
  and `.public_ip_address.allocation_id`). An automatically assigned address is
  released when the VM is deleted. An existing allocation that you assigned when
  you created the VM stays after the VM is deleted
  ([IP addresses](https://docs.nebius.com/compute/virtual-machines/network)).

Check the exact IDs against the attempt log and ownership evidence. Use cleanup
authorization already granted for this search; ask the owner only when an ID
falls outside that scope or its ownership is unclear. Preserve pre-existing or
shared disks and allocations unless their deletion was explicitly authorized.
For resources selected for deletion, delete by ID:

```bash
nebius compute instance delete --id "$INSTANCE_ID"
```

VM-managed disks are deleted with the VM. Standalone disks and filesystems are
not ([deleting a VM](https://docs.nebius.com/compute/virtual-machines/delete)).
Delete a standalone disk you no longer need with
`nebius compute disk delete --id "$DISK_ID"`; a disk with deletion protection
can't be deleted until protection is off
([managing volumes](https://docs.nebius.com/compute/storage/manage)). Delete an
allocation you no longer need with
`nebius vpc allocation delete --id "$ALLOCATION_ID"`.

If one deletion fails, record the error and continue checking the remaining
owned resources. Do not report cleanup as complete while residue remains.
Then confirm the cleanup:

- `nebius compute instance get --id "$INSTANCE_ID"` returns NotFound for each
  deleted VM, and `nebius compute disk get --id "$DISK_ID"` returns NotFound for
  each VM-managed disk and each standalone disk you deleted.
- `nebius vpc allocation get --id "$ALLOCATION_ID"` returns NotFound for each
  allocation ID you recorded, unless it was an existing allocation you chose to
  keep.
- Quota usage has dropped. In a tenant shared with other people, compare
  project-level reads, because other projects change the tenant totals.

## NPA clusters

Before it applies changes, `npa cluster up` uses cumulative whole-path preflight
to check the additional cluster resources against
`compute.instance.count`, `compute.disk.count`, `compute.disk.size.network-ssd`,
and `vpc.ipv4-address.public.count` when public node IPs are requested. It checks
ordinary GPU-family quota for on-demand GPU pools without a capacity block;
reservation-backed pools have separate reservation checks. A quota failure
blocks apply. Read capacity advice separately using this guide before proposing
an allocation change. A fallback follows the
[GPU allocation fallback](../../skills/atomic/gpu-allocation-fallback/SKILL.md)
skill and requires the operator's consent. `npa workbench health preflight`
checks credentials only. For NPA-managed clusters, agent VMs, controllers, and
buckets, clean up with the
[teardown and cost](../../skills/atomic/teardown-and-cost/SKILL.md) skill.

## Observed API responses

The Nebius docs don't describe the following. They come from API responses in
one tenant between October 1 and October 6, 2026, and can change:

- `QuotaFailure` and `NotEnoughResources` arrive in `status.details[].code`, and
  both failures carried the same top-level status code (8).
- A `NotEnoughResources` failure carried the message "VM schedule timeout, most
  likely due to insufficient hardware resources". While Compute tried to place
  the VM it showed `STARTING`, and it returned to `STOPPED` when the operation
  failed. A create that failed this way left the VM in place, stopped.
- A `QuotaFailure` for GPU quota was returned by a start operation on a stopped
  VM, so GPU quota is checked when a VM starts.
- An advice row's `limit` was expressed in VMs of that preset: an 8-GPU preset
  with 8 GPUs of quota showed a limit of 1. Rows showed
  `AVAILABILITY_LEVEL_LIMIT_REACHED` when the quota for the preset was used up.
- Every tenant quota row carried a limit; `none` from the headroom filter
  appeared only on project rows.
