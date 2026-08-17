# RTX PRO 6000 hardware MIG fleets

NPA supports true NVIDIA MIG on Nebius Managed Kubernetes as an additive
`npa.fleet/v0.0.1` policy. The first supported tuple is deliberately narrow and
fail-closed:

- Nebius `gpu-rtx6000`, NVIDIA RTX PRO 6000 Blackwell Server Edition (GB202,
  PCI device `0x2BB5`, 96 GB), one GPU per worker;
- Kubernetes 1.34, NVIDIA GPU Operator `v26.3.3`, driver `580.173.02`, NVIDIA
  device plugin and GFD `v0.19.3`, and MIG Manager `v0.14.2`;
- `mixed` strategy and the official `all-balanced` profile, which advertises
  exactly two `nvidia.com/mig-1g.24gb` and one
  `nvidia.com/mig-2g.48gb` per worker;
- a runtime-supplied Capacity Block Group, rendered as `STRICT`, and a 128 GiB
  GPU boot disk. NPA rejects PAYG fallback, another preset, another profile, or
  an oversized boot disk for this supported offering.

Omitting `mig` preserves the existing whole-GPU behavior. NPA does not accept
component-version overrides in the public spec: the tuple is a tested product
invariant and is asserted against the live ClusterPolicy and DaemonSets.

## Deploy

Copy [the example](../npa/examples/fleet/rtxpro-mig.yaml), replace only the
runtime project and capacity-block placeholders, then plan and deploy:

```bash
npa fleet plan --spec rtxpro-mig.yaml
npa fleet deploy --spec rtxpro-mig.yaml --yes
```

The node-group template receives `nvidia.com/mig.config=all-balanced`, so a
managed replacement converges without an out-of-band label. MIG-enabled fleets
use the pinned custom GPU Operator path and reject an explicit
`gpu_driver_mode: managed-image`; non-MIG fleets retain the repository's normal
automatic managed-image/operator driver selection.

An alternate `--k8s-training-ref`, `--k8s-training-dir`, or package-fallback
recipe is checked before quota, project, subnet, or Terraform mutation. MIG
deploy fails if that resolved recipe does not declare every required Operator,
driver, device-plugin, GFD, MIG Manager, reboot, and RDMA input; non-MIG recipes
retain their previous compatibility behavior.

Deployment does not become `deployed` merely because Terraform and Helm return.
NPA waits for two exact consecutive snapshots of:

- ClusterPolicy and driver/MIG Manager/GFD/device-plugin/toolkit pins and health;
- Ready, schedulable nodes plus fully observed/current GPU Operator Deployment
  and DaemonSet generations;
- PCI `0x2BB5`, supported vBIOS, driver/CUDA identity, Enabled/Enabled MIG mode,
  three distinct MIG UUIDs, and exact `1g.24gb`, `1g.24gb`, `2g.48gb` GI/CI
  profiles per worker;
- `nvidia.com/mig.config.state=success` and the expected product labels;
- capacity **and** allocatable values of 2/2 for `mig-1g.24gb`, 1/1 for
  `mig-2g.48gb`, and 0/0 for `nvidia.com/gpu` on every GPU worker.

The two snapshots, any driver-pod reconciliation, and the required final CUDA
smoke share the cluster's `gpu_health_timeout_minutes` deadline. The smoke asks
the scheduler for one `nvidia.com/mig-1g.24gb` device with equal requests and
limits, runs vectorAdd, proves an in-container `1g.24gb` MIG UUID with
`nvidia-smi -L`, checks its 24 GiB framebuffer identity (including CDI's
`NVIDIA_VISIBLE_DEVICES=void` mode), and waits for deletion of its uniquely
named pod. An enabled MIG policy rejects `gpu_cuda_smoke: false` because a
control-plane snapshot alone does not prove a usable allocation.

If geometry is successful but kubelet retains stale resources, NPA waits for a
GFD restart to complete before restarting the NVIDIA device plugin, then waits
again. It never patches Node status. Authentication and API failures stop
immediately with retained Terraform/cloud state for retry. Verifier subprocesses
ignore stale ambient Nebius IAM tokens unless the caller explicitly sets
`NPA_REUSE_IAM_TOKEN`.

A managed replacement can inherit the temporary
`nvidia.com/gpu=mig-not-ready:NoSchedule` taint even after MIG Manager reports
success. Verification treats that state as a scheduling failure. With
`--reconcile`, NPA uses race-closed JSON Patch tests to remove only that exact
obsolete taint from a successful GPU worker; it preserves every unrelated
taint.

The GPU Operator driver DaemonSet uses `OnDelete`. When a new driver pod template
is pending, `--reconcile` refuses to proceed while any running application pod
holds an `nvidia.com/*` resource: delete those workloads explicitly first. With
the cluster workload-free, NPA cordons one worker, rechecks the workload gate,
replaces and waits for that worker's driver pod, uncordons it, and only then
moves to the other worker. NPA never deletes application workloads implicitly.

## Verify and diagnose

```bash
npa fleet verify-mig --spec rtxpro-mig.yaml --output json
npa fleet verify-mig --spec rtxpro-mig.yaml --wait --reconcile
```

For a spec containing several MIG clusters, add `--project <key> --cluster
<name>`. `--kubeconfig` can verify a separately managed copy of the same
declarative target.

Treat any nonzero whole-GPU value as failure, including the known stale split
`capacity=1, allocatable=0`. A ClusterPolicy `ready` state alone is insufficient:
it does not prove GI/CI geometry, kubelet registration, scheduling, exhaustion,
or reuse.

## Lifecycle expectations

Blackwell MIG mode and GI/CI instances must be reconciled after driver unload
and reboot. Safely cordon and drain one worker at a time for driver changes,
cold reboot, or managed replacement; wait for exact state before uncordoning.
Existing pods may terminate and must be recreated according to Kubernetes
semantics. Do not infer silent reassignment from a replacement pod receiving a
different MIG UUID.

Production qualification must exercise both nodes: every resource profile,
all six slices concurrently with real CUDA kernels, exhaustion and Pending
events for both resource types, cleanup/reuse, individual operand restarts,
existing-pod behavior, one-at-a-time cold reboot, managed worker replacement,
idempotent fleet deployment, and a final repeated matrix.

## Terms and redistribution

GPU Operator installation/use is governed by NVIDIA's Product-Specific Terms
for NVIDIA AI Products and License for Customer Use of NVIDIA Software. Operator
consent is run-scoped and must never be stored in defaults. This deployment
pulls NVIDIA-delivered images directly; NPA does not bake or republish them. A
future mirror or redistributed image requires a fresh licensing classification.

## Compatibility references

- [GPU Operator 26.3 platform and component matrix](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/26.3/platform-support.html)
- [GPU Operator 26.3 release notes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/26.3/release-notes.html)
- [NVIDIA MIG prerequisites and RTX PRO Blackwell requirements](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/getting-started-with-mig.html)
- [NVIDIA's default MIG Manager profiles](https://github.com/NVIDIA/gpu-operator/blob/main/assets/state-mig-manager/0400_configmap.yaml)
- [Kubernetes device-plugin lifecycle](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
