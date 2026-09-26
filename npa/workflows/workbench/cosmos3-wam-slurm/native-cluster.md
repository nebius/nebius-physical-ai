# Native Slurm deployment used for the B200 campaign

The benchmark uses dedicated Nebius GPU VMs. The first worker also runs the
Slurm controller, MariaDB accounting and NFS server. This compact research
deployment has no controller failover. The Soperator specification in
`cluster.py` is a separate deployment option; results from this deployment do
not validate Soperator.

## Create the dedicated workers

Use an operator-owned project and subnet in `us-central1`, with explicit
STRICT binding to the selected B200 reservation. Each worker uses
`gpu-b200-sxm`, preset `8gpu-160vcpu-1792gb`. Two workers require sixteen
available reserved GPUs. Check CPU, disk-count and disk-capacity quotas too.

Create a [Nebius GPU cluster](https://docs.nebius.com/compute/clusters/gpu) on
the reservation's `us-central1-b` InfiniBand fabric and attach both VMs **when
creating them**. GPU-cluster membership cannot be added to an existing VM.
Retain boot/data disks explicitly if recreating a preparation VM. Use the
matching Ubuntu 24.04 CUDA 13 managed image on both workers. The measured
runtime has NVIDIA driver 580.173.02 and an active NVIDIA fabric manager;
capture the selected image ID privately with the creation requests.

Use a 2 TiB `NETWORK_SSD` boot/data disk for the first worker and a 256 GiB
boot disk for the second. The prepared inputs, Python environments and run
outputs live on the first disk and are exported over NFS to the second worker
at exactly the same absolute path. Preserve identical operator usernames,
UIDs and home paths because the shared virtualenvs reference managed Python
interpreters under the operator's home directory.

Use provider-verified SSH host keys. Keep creation requests, IDs, addresses,
SSH configuration and accounting records in private operator storage.

## Bootstrap the controller

Copy this recipe directory to the first worker. Install `uv` for the operator
user, then choose private names and run:

```bash
export NPA_SLURM_CLUSTER_NAME="$WAM_CLUSTER_NAME"
export NPA_SLURM_ACCOUNT="$WAM_ACCOUNT_NAME"
bash "$WAM_RECIPE/bootstrap-controller.sh"
```

The script requires sudo and a fresh Slurm installation. It installs Slurm
23.11.4, enables cgroup v2 CPU/memory/device containment and exposes eight GPU
resources. It retains sixteen GiB of host RAM outside the schedulable amount.
It also sets `RemoveIPC=no` for systemd-logind on both workers: a Slurm job can
outlive the operator's login session, so logout must not remove its shared
memory. NVIDIA documents the same requirement for
[Slurm and NCCL workloads](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/user-guide/docs/troubleshooting/runtime_and_mpi_issues.html).
The campaign observed a PyTorch data-loader failure under the default setting,
then reproduced the removal with an independent background shared-memory probe.
The controller creates a private accounting password and stores it with mode
0600. A dedicated firewall permits SSH, loopback, established connections and
the worker's own address. `persist-firewall.sh` preserves it across reboots.

The final `srun` must enumerate all eight B200s. Verify `nvidia-smi topo -m`,
`systemctl is-active nvidia-fabricmanager` and `ibstat`: all GPU pairs should
have NVLink connectivity, and the InfiniBand ports should be active. These
checks establish device availability; the training recipe additionally runs
a real NCCL collective and verifies rank placement.

To verify the IPC setting, choose a fresh systemd unit name and receipt path
in `WAM_IPC_PROBE_UNIT` and `WAM_IPC_PROBE_OUTPUT`, then run:

```bash
sudo systemd-run --unit "$WAM_IPC_PROBE_UNIT" --uid "$USER" --property Type=exec \
  /usr/bin/python3 "$WAM_RECIPE/ipc_probe.py" \
  --output-path "$WAM_IPC_PROBE_OUTPUT"
```

Close all SSH sessions for that user, wait thirty seconds, then reconnect and
read the JSON receipt. Require `survived_logout: true`. Keeping another SSH
session open would prevent this test from exercising logout cleanup. The
thirty-second observation window is a host-readiness test, not a training limit.

Install the pinned framework and stage data/model inputs following the main
[runbook](../../../../docs/workbench/cookbooks/cosmos3-wam-slurm.md#2-install-the-pinned-native-training-environment).
Set `WAM_RUN_ROOT` beneath `WAM_SHARED_ROOT` so the NFS export covers both
prepared inputs and run directories. Run the eight-GPU baseline on this worker.

## Join the second worker

Set `WAM_WORKER_NAME`, `WAM_WORKER_ADDRESS`, `WAM_WORKER_BUNDLE` and
`WAM_SHARED_ROOT` to private operator values. The address must be the second
worker's private IPv4 address. On the controller:

```bash
sudo /usr/bin/python3 "$WAM_RECIPE/add_worker.py" \
  --worker-name "$WAM_WORKER_NAME" --worker-address "$WAM_WORKER_ADDRESS" \
  --shared-root "$WAM_SHARED_ROOT" --output-dir "$WAM_WORKER_BUNDLE"
```

This adds the node to Slurm without restarting the running controller, exports
the shared directory only to that peer and permits its private traffic. The
new bundle contains Slurm configuration and a **secret Munge key**. Transfer
the bundle and recipe directory to the second worker over verified SSH; keep
the bundle directory private and every contained file at mode 0600. Do not
commit, print or publish it. No accounting database password is transferred.

Install `uv` for the same operator user on the second worker. Set
`WAM_WORKER_BUNDLE` to its local transferred bundle and run:

```bash
bash "$WAM_RECIPE/bootstrap-worker.sh"
```

The worker installs Python 3.13.15 and 3.10.21 at the paths needed by the shared
environments, preserves the controller's Slurm service UID/GID, installs the
same Slurm version, verifies eight B200s and mounts NFS with hard TCP mounts.
Munge uses the shared key; only the controller and worker private addresses
are permitted to reach cluster services. Hostname resolution is recorded on
both workers so torchrun's rendezvous address resolves consistently.

Verify both nodes register with `sinfo`. A down/drained node requires inspection
of `scontrol show node` and `journalctl -u slurmd`, followed by correction and
an explicit resume; do not hide a resource mismatch by overriding hardware
counts. The two-node training preflight must report sixteen ranks on two
hosts and an all-reduce sum of 136 before training starts.

## Evidence and cleanup

Record `sacct` elapsed time, exit status and allocated TRES for each job. Keep
one-second GPU telemetry, raw NCCL transport logs and package versions private.
The public report contains counts, versions, timings and hashes, without
resource names or endpoints. Keep profile jobs separate from throughput jobs.

Cancel this campaign's remaining Slurm jobs before stopping its VMs. Verify
`squeue` is empty, archive complete checkpoints and reports, then stop or delete
only the dedicated workers. Preserve the first disk until its artifacts have
been verified in durable storage. Delete its GPU-cluster resource only after
all attached campaign workers have been removed. Audit the dedicated project
for remaining billable disks and VMs.
