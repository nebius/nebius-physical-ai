---
name: cluster
description: Use when deploying or managing NPA Workbench Managed-Kubernetes cluster targets and their GPU/CPU node groups via `npa cluster`.
---

# Cluster Targets And Node Groups

`npa cluster` manages NPA Workbench Managed-Kubernetes targets and the node
groups attached to them. It wraps `nebius mk8s` with NPA aliases, preset
validation, and a local state cache under `~/.npa`. Pairs with `nebius-infra`
(infra facts and GPU routing) and `skypilot-workflows` (what runs on the nodes).

## Interfaces

```bash
npa cluster deploy | destroy | up | down | status | list
npa cluster node-group add       # GPU node group
npa cluster node-group add-cpu    # CPU node group
npa cluster node-group remove | status | list
```

## GPU vs CPU node groups

- `add` is GPU-only and requires `--gpu-type` (`h100`, `h200`, `l40s`,
  `rtx6000`); the platform/preset come from `GPU_TYPE_DEFAULTS`.
- `add-cpu` attaches a GPU-free group with `--platform` (`cpu-e2`/`cpu-d3`) and
  `--preset` (e.g. `8vcpu-32gb`, `16vcpu-64gb`), validated against
  `SUPPORTED_NODE_PRESETS`.
- Route CPU-only workloads (motion retargeting, batched inference, SkyPilot
  pods that only call in-cluster services) to CPU node groups so they do not
  consume GPU nodes. H100/H200 are compute-only and do not provide RT cores;
  Isaac Lab / SONIC render validation needs L40S or RTX PRO 6000.

## Autoscaling and cost

- `--autoscaling-min/--autoscaling-max` set a fixed range; `--autoscaling-min 0`
  scales the group to zero when idle (cold start on the first pending pod), which
  is the cheap default for bursty batched CPU work. Omit both for a
  `--node-count` fixed group.

## Conventions

- Node-group create needs a VPC subnet. The CLI uses the saved cluster state's
  subnet; pass `--subnet-id` when no local state exists (e.g. a cluster created
  out of band).
- `add`/`add-cpu`/`status`/`list`/`remove` share the local node-group state
  cache, so CPU and GPU groups appear together in `status`/`list`.
- Keep node-group orchestration in the `npa.cluster` API/CLI; it is the source of
  truth that `MK8sClient` translates into `nebius mk8s` calls.
