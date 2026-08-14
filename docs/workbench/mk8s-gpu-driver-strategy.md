# Managed Kubernetes GPU driver strategy

NPA uses one GPU driver policy for both direct `npa cluster` provisioning and
`npa fleet`: GPU node groups use a Nebius managed-driver image by default, while
CPU-only node groups and clusters receive no GPU driver settings. The policy is
topology-independent; it derives expected capacity from the requested node
count and GPU preset instead of naming a GPU SKU or assuming eight devices.

## Modes

| Mode | GPU behavior | Intended use |
| --- | --- | --- |
| `auto` | Select the managed-driver image whenever the active recipe/provider supports it. | Safe default. |
| `managed-image` | Explicitly require the managed-driver image. | Pin the operational contract. |
| `operator` | Use the recipe's in-cluster NVIDIA GPU Operator driver path. | Supported diagnostics or a recipe-specific requirement only. |

The default managed preset is `cuda13.0`. Set `managed_driver_preset` to another
value supported by the selected recipe/provider. NPA validates the value and
does not copy it into unrelated platform-specific rules.

Operator mode is unsafe on an NVSwitch topology when Fabric Manager can start
before Network Operator/MOFED exposes the host InfiniBand management devices.
That failure presents as `NebiusGPUError=True`, Fabric `In Progress`/`N/A`, CUDA
`system not yet initialized`, and Fabric Manager/NVLSM `umad_open_port()` or
`IB_ERROR` failures. NPA identifies an NVSwitch-risk topology from an explicit
GPU cluster or a multi-GPU SXM/NVL preset and rejects operator mode unless the
operator sets `allow_unsafe_nvswitch_operator` deliberately.

Direct configuration:

```hcl
gpu_driver_mode                  = "auto"
managed_driver_preset            = "cuda13.0"
allow_unsafe_nvswitch_operator   = false
```

The same values can be supplied with `npa cluster up --gpu-driver-mode ...`
and `--managed-driver-preset ...`. Fleet puts them in `defaults` or an
individual cluster:

```yaml
defaults:
  gpu_driver_mode: auto
  managed_driver_preset: cuda13.0
  allow_unsafe_nvswitch_operator: false
```

## Recipe compatibility

The vendored `k8s-training` recipe exposes both the managed-image selection and
configurable preset. Fleet inspects that same capability after materializing a
vendored, `--k8s-training-ref`, or `--k8s-training-dir` source:

- a compatible recipe receives the selected image flag and preset;
- a recipe with a fixed managed preset may be used only when it matches the
  requested preset;
- a GPU recipe that cannot represent the selected strategy fails before
  Terraform apply with the missing input/wiring in the error;
- CPU-only clusters do not require those GPU recipe capabilities.

An alternate recipe therefore cannot silently fall back to the operator path.

## Fail-closed deployment health

After Terraform and kubeconfig creation, both direct and Fleet paths use the
same Kubernetes health gate. For a requested GPU topology it requires:

1. At least the expected total nodes are Ready and the exact expected GPU-node
   count is observable.
2. Total `nvidia.com/gpu` allocatable equals `GPU nodes × GPUs encoded by the
   requested preset`.
3. No requested GPU node has `NebiusGPUError=True`.
4. NVSwitch Fabric is `Completed`/`Success` wherever Kubernetes node metadata,
   conditions, or `nvidia-smi -q` exposes its state.
5. The provider device-plugin components (`managed-image`) or GPU Operator
   components (`operator`) are Running and Ready.
6. Node names and boot IDs do not change during the stabilization interval.
7. When live CUDA validation is enabled, NVIDIA vectorAdd reports `Test PASSED`
   on every requested GPU node; the pod also captures `nvidia-smi -q`.

Direct provisioning saves the local cluster state as `VALIDATING` before the
gate and `RUNNING` only afterward. Fleet records `validating-gpu-health`, then
`deployed`; failure is `deployed-validation-failed`. Fleet retains the cluster,
kubeconfig, Terraform state, and a permission-restricted `gpu-health.json`
under its per-cluster install directory so an operator can inspect evidence and
retry reconciliation. CUDA smoke pods are deleted after each attempt.

Fleet settings are `gpu_health_stabilization_seconds`,
`gpu_health_timeout_minutes`, `gpu_cuda_smoke`, and `gpu_cuda_smoke_image`.
Direct equivalents are `--gpu-health-stabilization-seconds`,
`--validation-timeout`, `--gpu-cuda-smoke/--skip-gpu-cuda-smoke`, and
`--gpu-cuda-smoke-image`. Skipping validation is an explicit diagnostic choice,
not the success default.

## Migrating existing affected pools

A code upgrade changes future rendered node-group configuration; it cannot
replace the boot image or driver stack on an already-running node. Existing GPU
pools that booted with the operator driver and exhibit the recovery loop need a
controlled rolling node-group update or recreation after the spec is set to
`auto` or `managed-image`.

Plan replacement capacity and reservation binding first, cordon/drain only
within the workload's disruption policy, replace nodes in controlled batches,
and require this health gate before returning the pool to service. Do not treat
an unchanged node that was merely power-cycled as migrated. Preserve the failed
health evidence until the replacement pool has remained stable.
