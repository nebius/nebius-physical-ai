# Temporary Nebius CPU runners

Use this pool to reserve CPU capacity for existing CI jobs. It does not change
required checks or merge-queue concurrency. Start with priority checks when cloud
quota cannot support the full eight-shard test suite concurrently.

Each GitHub just-in-time (JIT) registration accepts at most one job. Its VM and
managed boot disk are then deleted and replaced from a clean image. Workers have
automatic VM recovery disabled, and deletions run independently so one slow
deletion cannot block other workers from being replaced. Workers have
no Nebius service account, operator credentials, inbound network access, or
shared writable disks. Their runner events go to the VM serial log; GitHub keeps
the workflow logs. The controller retains private creation/deletion receipts.

The temporary controller runs on the operator machine using its existing `gh`
and `nebius` authentication. Neither credential is copied to a worker. Keep that
machine online while the pool is enabled. On macOS, `up` uses a launch agent and
prevents idle sleep while connected to power. On other platforms it starts a
detached process; `serve` can instead run under an operator-managed supervisor.

## Operate an installed pool

Run from this checkout using its Python environment. Set `RUNNER_STATE` to the
private state directory saved during setup; the default is
`~/.npa/ci-runners/default`. Do not commit that directory.

```bash
RUNNER_STATE="$HOME/.npa/ci-runners/default"
npa/.venv/bin/python npa/scripts/ci_cpu_runners.py status --state-dir "$RUNNER_STATE"
npa/.venv/bin/python npa/scripts/ci_cpu_runners.py up --state-dir "$RUNNER_STATE"
```

`up` maintains the configured worker count without changing repository routing.
After every worker is online, enable the selected existing repository variable:

```bash
npa/.venv/bin/python npa/scripts/ci_cpu_runners.py enable --state-dir "$RUNNER_STATE"
```

Validate a real PR or queue candidate. Check that the expected jobs ran on the
new runner names and that completed workers disappeared from both GitHub and
Nebius. Do not infer success from registration alone.

## Remove CPU capacity

One command restores the previous runner variable and starts a safe drain:

```bash
make ci-runners-down RUNNER_STATE="$RUNNER_STATE"
```

With state at the default location, use just `make ci-runners-down`.
`make ci-runners-status` reports readiness and the remaining workers.

New workflows use the previous routing immediately. The controller keeps serving
PR and merge-queue admission workflows that were active when routing changed,
including their later dependent jobs. Unrelated publishing and main-branch
audits do not delay removal. It then removes the GitHub registrations, VMs, and
managed boot disks and exits. `status` reports the remaining workers. The command requests this work
asynchronously; keep the controller online until it reports zero workers.
There is no forced cancellation or drain deadline. Repeating `down` is safe.

`disable` restores routing while keeping the pool running, which is useful for
diagnosis. `up` followed by `enable` starts a drained pool again. External changes
to the routing variable are preserved instead of overwritten.

`down` retains the dedicated project, firewall, private base image, and receipts
for a later restart. Only worker compute and boot-disk spend ends. Delete the
retained image and project separately if permanent removal is required, after
verifying that their inventories contain no other resources.

## Prepare a pool

Use a dedicated project in the operator's chosen tenant and region. Check live
CPU, VM, disk-count, disk-byte, and public-address quotas before selecting capacity.
The controller checks tenant and project headroom before allocating replacements.
Reserve one VM and a 64 GiB managed boot disk per worker. No GPU is used.

1. Save the provider's project creation response outside the repository. Label
   the project and its owned image/security group with `npa-purpose=ci-runners`
   and one unique `npa-owner` value, at most 32 characters.
2. Build a private Ubuntu 24.04 driverless image with [image.sh](image.sh). It
   installs patched OS packages and checksum-verified GitHub runner/CLI releases.
   Before imaging the stopped builder, verify no runner registration or cloud
   credentials exist; remove bootstrap SSH keys and host keys and run
   `cloud-init clean --logs --machine-id --seed`. Delete the builder afterwards.
3. Use a subnet in that project and a dedicated worker security group. Allow
   outbound traffic and explicitly deny all inbound IPv4 traffic with an `ANY`
   rule from `0.0.0.0/0`. Do not add inbound allow rules or attach the default
   security group alongside it. The controller verifies these boundaries.
4. Create a private directory (`chmod 700`) and a `config.json` (`chmod 600`) with
   the following fields. All provider identities stay in this private file.

| Field | Value |
| --- | --- |
| `repository` | GitHub `owner/repository` |
| `tenant_id`, `project_id`, `region` | Exact identities from project creation |
| `project_create_response` | Complete JSON object returned by project creation |
| `owner` | The unique `npa-owner` label |
| `profile` | Optional local Nebius CLI profile |
| `network_id`, `subnet_id`, `worker_security_group_id` | Dedicated project networking |
| `image_id` | Verified private runner image |
| `workers` | Desired VM capacity, selected from available quota |
| `preset` | An available `cpu-d3` CPU preset, for example `4vcpu-16gb` |
| `label` | Unique GitHub runner label for this pool |
| `variable` | `NPA_CI_PRIORITY_RUNNER` or `NPA_CI_TEST_RUNNER` |

Repository administration access is enough to register these runners; an
organization runner group is not required. Keep GitHub's existing public-PR
approval policy enabled. For unattended long-term operation, move the controller
to a trusted management host with dedicated credentials; do not put it on a
worker or reuse a worker VM for another job.
