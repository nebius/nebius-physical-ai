# Plan GPU quota and capacity on Nebius AI Cloud

[Workbench docs](README.md)

Use this guide before you launch a GPU virtual machine (VM) on Nebius AI Cloud.
It shows how to check that a region can take the VM, request only the quota you
are missing, retry across regions without piling up stopped VMs, tell a quota
failure from a capacity failure, and clean up after failed attempts.

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
- Pick a project in each region you want to try. Platforms and presets differ by
  region and project
  ([platforms and presets in a project](https://docs.nebius.com/compute/virtual-machines/list-platforms)).

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

Headroom is a row's limit minus its usage. Your demand must fit the tenant
headroom, and also the project headroom for any project row that has a limit.
Headroom equal to the demand is enough for one VM. Each additional VM you create
while searching needs its own headroom until it is deleted: VMs count against
`compute.instance.count` and their disks count against the disk quotas until
the VM is deleted, even when it is stopped
([VM lifecycle](https://docs.nebius.com/compute/virtual-machines/lifecycle)).

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
  `availability_level` says. If you try anyway, use the bounded retry below.
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
bounded retry below.

## Choose a target

A region is a candidate when every quota you need has enough headroom and its
fresh advice row shows `available` at or above the number of VMs you need.

If no region qualifies:

- If quota is the limit, [request more quota](#request-more-quota).
- If capacity is the limit, try another region or a similar preset, as the
  [Not enough resources](https://docs.nebius.com/compute/virtual-machines/not-enough-resources)
  page suggests, or contact Nebius sales to reserve capacity. You can also read
  the advice again on the [retry cadence](#retry-across-regions) and stop when
  that bound runs out.
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

Request the shortfall for your planned VMs plus any retry margin you choose.
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

A bounded search keeps cost and leftovers under control. The following cadence
is a recommendation, not a provider requirement:

1. Read quota and advice again at the start of each round.
2. Make one attempt per candidate region per round, one region at a time.
3. Keep at most one stopped VM per region from earlier attempts, and start it
   (`nebius compute instance start --id "$INSTANCE_ID"`) instead of creating
   another. The [Not enough resources](https://docs.nebius.com/compute/virtual-machines/not-enough-resources)
   page covers restarting a stopped VM. A stopped VM still holds quota, and its
   disks keep billing.
4. After a `QuotaFailure`, skip that region until the named quota has headroom
   again, through lower usage or a higher limit.
5. Run at most three rounds, waiting 5 minutes after the first and 10 minutes
   after the second, unless the person who owns the search sets a longer
   deadline.

For a longer search, write down the deadline, interval, regions, platform,
preset, and allocation type before you start. Append one line per attempt with
the time, region, instance ID, operation ID, and detail code. Run the loop under
a process supervisor or in a terminal session that stays open; a background
process started from a short-lived session can be killed when that session ends.
Stop at the first `RUNNING` VM and confirm its state again before you hand it to
a workload. If attempts ever run in parallel, check every other recorded VM
after the first success, and stop or delete any extra VM that is starting or
running.

## Clean up after a search

Stopped VMs keep their instance and disk quota, and their disks keep billing
([deleting a VM](https://docs.nebius.com/compute/virtual-machines/delete)). Build
the list from your attempt log and the run-label listing above, and use IDs, not
names.

For each VM, run `nebius compute instance get --id "$INSTANCE_ID"` and note:

- its state;
- its disks: `status.disk_attachments[]` lists each attached disk's `id` and
  whether it is VM-managed (`is_managed`);
- its IP allocation IDs (`status.network_interfaces[].ip_address.allocation_id`
  and `.public_ip_address.allocation_id`). An automatically assigned address is
  released when the VM is deleted. An existing allocation that you assigned when
  you created the VM stays after the VM is deleted
  ([IP addresses](https://docs.nebius.com/compute/virtual-machines/network)).

Agree the exact list of IDs with the person who owns them, then delete by ID:

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

Before it applies changes, `npa cluster up` checks the planned cluster against
`compute.instance.count`, `compute.disk.count`, `compute.disk.size.network-ssd`,
and `vpc.ipv4-address.public.count`, and checks GPU quota for on-demand GPU
pools. When GPU quota is short, it reads capacity advice to suggest a
preemptible pool. For a direct VM launch, or for other quotas and capacity
advice, use the steps on this page. `npa workbench health preflight` checks
credentials only. For NPA-managed clusters, agent VMs, controllers, and buckets,
clean up with the [teardown and cost](../../skills/atomic/teardown-and-cost/SKILL.md)
skill.

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
