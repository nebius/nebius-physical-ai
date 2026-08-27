---
name: gpu-cluster-provisioning
description: Use when creating or repairing a GPU Kubernetes cluster for npa — the managed-image vs GPU-Operator driver decision, why operator mode is unsafe on NVSwitch, the post-apply health gates (fabric, CUDA vectorAdd, stability window), and triage for nodes that come up without working GPUs.
---

# GPU cluster provisioning and driver strategy

Getting GPU nodes is the easy part. Getting nodes whose GPUs actually work — and
knowing *before* you submit a job that they do — is what this skill covers. It
complements `skills/atomic/gpu-selection/SKILL.md` (which GPU to ask for) and
`skills/tools/nebius-infra/SKILL.md` (config, storage, teardown) with the driver
and readiness decisions made at provisioning time.

Source of record: `docs/workbench/mk8s-gpu-driver-strategy.md`.

## The driver decision: default to managed, do not reach for operator

One policy covers both direct `npa cluster` provisioning and `npa fleet`: **GPU
node groups use a Nebius managed-driver image by default**, and CPU-only node
groups get no GPU driver settings at all.

| `--gpu-driver-mode` | Behavior | Use when |
|---|---|---|
| `auto` (default) | Managed-driver image whenever the recipe/provider supports it | Almost always |
| `managed-image` | Explicitly require the managed-driver image | Pinning the operational contract |
| `operator` | In-cluster NVIDIA GPU Operator driver path | Supported diagnostics or a recipe that requires it |

```bash
npa cluster up --gpu-driver-mode auto --managed-driver-preset cuda13.0
```

The default managed preset is `cuda13.0`. The same values exist on
`npa provision-if-absent`, and in a fleet spec under `defaults` or a single
cluster (`gpu_driver_mode`, `managed_driver_preset`).

**Operator mode is unsafe on NVSwitch topologies.** Fabric Manager can start
before Network Operator/MOFED exposes the host InfiniBand management devices,
and the result is nodes that exist but cannot run CUDA. It presents as:

- `NebiusGPUError=True`
- Fabric status `In Progress` or `N/A`
- CUDA `system not yet initialized`
- Fabric Manager / NVLSM `umad_open_port()` or `IB_ERROR` failures

NPA identifies an NVSwitch-risk topology from an explicit GPU cluster or a
multi-GPU SXM/NVL preset and **rejects operator mode** unless you pass
`--allow-unsafe-nvswitch-operator`. That flag is a diagnostics acknowledgement,
not a workaround: if a deploy fails and the suggested fix is that flag, the fix
is almost always `auto` instead.

The policy is topology-independent — expected capacity is derived from the
requested node count and GPU preset, not from a GPU SKU or an assumption of
eight devices per node.

## Provision with the health gates on

```bash
npa cluster up \
  --project <alias> \
  --gpu-nodes 2 --gpu-platform <platform> --gpu-preset <preset> \
  --cpu-nodes 1 --cpu-platform <platform> --cpu-preset <preset> \
  --gpu-driver-mode auto --managed-driver-preset cuda13.0 \
  --gpu-health-stability-seconds 120 \
  --validation-timeout 60 --timeout 120
```

Defaults are deliberately strict, and every one of them is on for a reason:

- `--validate` (default) checks stable nodes, GPU capacity, fabric, driver
  components, CUDA vectorAdd, and the default StorageClass.
- `--gpu-cuda-smoke` (default) runs NVIDIA's CUDA vectorAdd **on every requested
  GPU node**, which is the cheapest proof that drivers actually work.
- `--gpu-health-stability-seconds` (default 120) requires nodes, boot IDs,
  fabric, capacity, and components to stay healthy for that window. It exists
  because GPU nodes routinely look healthy for a few seconds during labelling.
- `--sky-smoke` (default) runs a SkyPilot Kubernetes GPU task and cleans it up.

Do not reach for `--skip-validate` / `--skip-gpu-cuda-smoke` to make a deploy
"succeed" faster. Skipping them moves the failure to your first real job, where
it costs more and is harder to attribute. `-1` on `--gpu-nodes` / `--cpu-nodes`
keeps the configured value rather than meaning zero.

For reserved capacity, `--capacity-block-group-id` selects a private capacity
block for strict GPU node-group reservation. `--preemptible` is often the only
way to get several GPUs at once, but a reclaim stops nodes mid-run — keep CPU
stages on the CPU pool, and note that preemptibility changes the capacity pool
only, never the disk or IP quota requirements.

When a cluster enables a shared filestore (or attaches an existing one), set
`TF_VAR_filesystem_csi_chart_repository` to the operator-approved Shared
Filesystem CSI Helm repository. NPA intentionally has no provider-private
default and now fails before apply if the repository is absent; creating the
filesystem without its CSI driver leaves validation waiting for a StorageClass
that cannot appear.

## Node groups after the fact

```bash
npa cluster node-group list --project <alias>
npa cluster node-group status --project <alias>
npa cluster node-group add --project <alias> ...
npa cluster node-group add-cpu --project <alias> ...
npa cluster node-group remove --project <alias> ...
```

GPU additions accept the same `--gpu-driver-mode auto|managed-image|operator`
and `--managed-driver-preset` contract as initial provisioning. Use `operator`
only when the workload needs an operator-managed capability absent from the
managed image and the selected topology is not NVSwitch-class.

Add a CPU pool rather than running CPU stages on GPU nodes: it is cheaper and it
keeps preemptible GPU reclaims from killing coordination work.

## After provisioning: what the cluster calls its GPUs

A healthy cluster is not yet a submittable one. SkyPilot addresses accelerators
by the name the cluster advertises, which comes from node labels and **changes
while the GPU operator is still labelling** (`nebius.com/gpu-name` first,
`nvidia.com/gpu.product` after):

```bash
npa cluster status --project <alias>
npa workbench workflow gpus --cluster <name> --json
npa skypilot verify --cluster <name> --output-format json
```

Run `workflow gpus` once after provisioning and note the requestable quantity per
node. A name mismatch fails as `FAILED_PRECHECKS` or "cluster does not contain
any instances satisfying the request", which reads like a capacity shortage and
is not one.

## Triage: nodes exist but GPUs do not work

1. **`NebiusGPUError=True`, fabric not ready, CUDA "system not yet initialized"**
   → operator-mode/Fabric-Manager ordering on NVSwitch. Redeploy the GPU node
   group with `--gpu-driver-mode auto`. Do not paper over it with the unsafe
   acknowledgement flag.
2. **Nodes Ready but zero schedulable `nvidia.com/gpu`** → drivers or device
   plugin have not landed. Re-run validation rather than submitting; the
   stability window exists precisely for this state.
3. **Accelerator name not found by SkyPilot** → labelling is still in progress or
   the spec uses a different spelling. Use `workflow gpus`.
4. **CUDA vectorAdd fails on one node only** → that node is bad; remove and
   re-add the node group rather than debugging the whole cluster.
5. **Deploy succeeded but jobs stall in `ImagePullBackOff`** → not a GPU problem
   at all. See `skills/atomic/debug-failed-run/SKILL.md`.

An image can also be architecturally incapable of running on the GPU you
provisioned — `sm_120` (RTX PRO 6000) and `sm_100`/`sm_103` (B200/B300) binaries
are mutually incompatible across the CUDA major boundary. That is an image
question, not a cluster question: see `docs/workbench/image-gpu-compatibility-matrix.md`
and `skills/atomic/gpu-selection/SKILL.md`.

## Gotchas

- **`npa cluster` is not raw MK8s administration.** For edit, update, upgrade,
  operation inspection, version listing, and the compatibility matrix, use
  `nebius mk8s` directly.
- **`npa cluster down` going silent for minutes is expected** — it previews the
  PodDisruptionBudgets that hold up the node drain first.
- **Stale `NEBIUS_IAM_TOKEN` breaks Terraform and the provider** even when the
  `nebius` CLI works. `unset NEBIUS_IAM_TOKEN NPA_NEBIUS_IAM_TOKEN` first.
- **SkyPilot task pods run in `default`; deployed workbench services run in
  `workbench`.** A pull secret in the wrong namespace helps nothing.
- **Provisioning is additive, teardown is not.** `provision-if-absent` never
  replaces resources; `cluster down` destroys. See
  `skills/atomic/teardown-and-cost/SKILL.md` for the ordering.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```
