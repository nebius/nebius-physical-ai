# Disposable Nebius CPU runners

Use this pool to reserve CPU capacity for existing CI jobs. It does not change
required checks or merge-queue concurrency. For a small pool, configure
`NPA_CI_SECURITY_RUNNER` to offload independent confidentiality and source/dependency
scans. Leave `NPA_CI_PRIORITY_RUNNER` and `NPA_CI_TEST_RUNNER` unset so admission,
test shards, and final aggregation can use GitHub-hosted capacity without waiting
for a fresh CPU VM. Other PR branches adopt this routing after updating their
workflow files to include it.

Each GitHub just-in-time (JIT) registration accepts at most one job. Its VM and
managed boot disk are then deleted and replaced from a clean image. Workers have
automatic VM recovery disabled, and deletions run independently so one slow
deletion cannot block other workers from being replaced. Workers have
no Nebius service account, operator credentials, inbound network access, or
shared writable disks. Their runner events go to the VM serial log; GitHub keeps
the workflow logs. The controller retains private creation/deletion receipts.

The cloud controller runs on a separate Nebius CPU VM under systemd. It uses an
attached service account with `editor` access to the dedicated CI project and a
GitHub App installed only on this repository. Installation tokens are refreshed
automatically and restricted to that repository with administration write (runner
registration), Actions read (draining), and variables write (routing rollback).
Its private App key stays on the controller. Neither that key nor cloud-management
credentials reach workers. No personal CLI login is transferred, and the
operator's Mac can be offline.

The controller never registers as a runner or checks out PR code. Its installed
scripts come from an explicit operator-prepared bundle. The worker login and sudo
access are removed, and systemd restricts writes to private controller state.
Controller SSH permits only the operator's address and pins host keys from
authenticated provider logs. Initial provisioning checks tenant and project
quota. During replacement the project-scoped identity checks project quota;
Nebius enforces inherited tenant limits on every allocation. No tenant-wide
role is granted.

Legacy local operation remains available for bootstrap using existing `gh` and
`nebius` authentication, and requires that machine to stay online. An installed
private `remote-controller.json` routes operator commands to the cloud VM instead.

## Operate an installed pool

Run from this checkout using its Python environment. Set `RUNNER_STATE` to the
private state directory saved during setup; the default is
`~/.npa/ci-runners/default`. Do not commit that directory.

```bash
RUNNER_STATE="$HOME/.npa/ci-runners/default"
npa/.venv/bin/python npa/scripts/ci_cpu_runners.py status --state-dir "$RUNNER_STATE"
npa/.venv/bin/python npa/scripts/ci_cpu_runners.py up --state-dir "$RUNNER_STATE"
```

For a cloud pool, `up` starts a stopped controller VM through the operator's
Nebius CLI and starts its systemd service over pinned SSH. It maintains the
configured worker count without changing repository routing.
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
managed boot disks. The command requests this work asynchronously. A cloud
controller completes the drain independently of the operator machine, then stops
its own VM through the Nebius API, ending its CPU runtime spend.
There is no forced cancellation or drain deadline. Repeating `down` is safe.

`disable` restores routing while keeping the pool running, which is useful for
diagnosis. `up` followed by `enable` starts a drained pool again. External changes
to the routing variable are preserved instead of overwritten.

`down` retains the stopped controller VM and its boot disk, dedicated project,
firewalls, private base image, and receipts for restart. Worker VM and disk spend
ends; controller disk and retained-image storage remain. Delete the retained
controller, image and project separately for permanent removal, after
verifying that their inventories contain no other resources.

## Prepare a pool

Use a dedicated project in the operator's chosen tenant and region. Check live
CPU, VM, disk-count, disk-byte, and public-address quotas before selecting capacity.
The local bootstrap checks tenant and project headroom before provisioning.
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
| `variable` | `NPA_CI_SECURITY_RUNNER` for a small pool; priority/test variables require separately sized capacity |

## Move the controller into Nebius

Register the repository-scoped App using the checked-out setup helper:

```bash
npa/.venv/bin/python npa/scripts/ci_cpu_runner_app_setup.py \
  --state-dir "$RUNNER_STATE" --repository '<owner>/<repository>'
```

Open the URL in private `app-setup.json`, create the App in the repository's
organization, and install it only on the selected repository. GitHub requires
this signed-in browser step; organization policy may require an owner. The
helper validates the callback state, saves the private key as mode `0600`, and
verifies a token restricted to exactly the selected repository. Never paste the
key into chat, commit it, or put it in a worker image. No additional GitHub
Actions secret is needed for this architecture.

Provision a separate controller with the private runner image, a dedicated
project service account and controller firewall. The request builder in
`npa/scripts/ci_cpu_runner_controller.py` gives it two vCPUs, automatic recovery,
an operator SSH key, and no runner service. Preserve its provider creation
receipt. Account for its public address and disk before allocating it.

Prepare an explicit code/state bundle outside the repository:

```bash
npa/.venv/bin/python npa/scripts/ci_cpu_runner_controller.py \
  --state-dir "$RUNNER_STATE" --instance-receipt '<private-controller-receipt.json>' \
  --output-path '<private-controller-bundle.tar.gz>'
```

Transfer it over pinned SSH, extract it as root into `/opt/npa-ci-controller`,
and run `.github/ci-runners/controller-install.sh` there. The installer verifies
the pinned Nebius CLI and configures the VM's rotating metadata token. It enables
the service but blocks activation with a `controller-stopped` marker.

For the final handoff, stop the old supervisor and verify its lock is released;
transfer a fresh bundle of its worker records and routing receipt; write the
operator's private `remote-controller.json` with the controller `instance_id`
and `host`; retain `controller-key` and `controller-known-hosts` beside it. Then
run `up` against that state directory. Never run two controllers for one pool.
Prove a real CI job and worker replacement with the old controller stopped,
then prove controller restart and drain/restart before enabling normal routing.

Cloud runtime configuration adds `github_auth: app`, `supervisor: systemd`,
`quota_scope: project`, `profile: ci-controller`, and `controller_instance_id`.
Private `github-app.json` contains `app_id`, `installation_id`, `repository`, and
`private_key_file`; the bundle rewrites the latter to the controller's private
state directory. Missing or invalid App configuration fails without falling back
to a personal token.

Repository administration access is enough to register runners; an organization
runner group is not required. Keep GitHub's existing public-PR approval policy
enabled. Do not put the controller on a worker or reuse a worker VM for another job.
