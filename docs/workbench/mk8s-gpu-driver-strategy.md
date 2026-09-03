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

## RTX rendering workload profile

RTX rendering is the deliberate exception to the managed-image default. Use the
single explicit profile instead of assembling independent platform, preset, and
driver flags:

```bash
npa cluster up --gpu-workload-profile rtx-rendering
# or, for additive project setup:
npa provision-if-absent --gpu-workload-profile rtx-rendering
```

`rtx-rendering` selects `gpu-rtx6000` / `1gpu-24vcpu-218gb`, defaults to one
GPU node, and selects the supported NVIDIA GPU Operator mounted-driver path.
It also makes graphics readiness mandatory. After the ordinary stability and
per-node CUDA vectorAdd gates, NPA runs an immutable, payload-clean RTX image
with `runtimeClassName: nvidia` and `NVIDIA_DRIVER_CAPABILITIES=all` on every
requested GPU node. The pod must dynamically load `libGLX_nvidia.so.0` and
`libEGL_nvidia.so.0`, create a Vulkan instance, and enumerate an NVIDIA physical
device. A missing runtime class, library mount, ICD, or device fails deployment.

Fleet and SDK use the same contract:

```yaml
defaults:
  gpu_workload_profile: rtx-rendering
```

```python
ClusterSpec(name="render", gpu_workload_profile="rtx-rendering")
```

The empty profile retains all historical defaults. Conflicting platform,
preset, zero-GPU, or managed-image selections fail closed. The existing
NVSwitch operator rejection is unchanged; this profile cannot be applied to an
SXM/NVL topology and does not enable the unsafe acknowledgement.

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
8. For `gpu_workload_profile: rtx-rendering`, GLX and EGL load through the
   operator mount and Vulkan enumerates an NVIDIA physical device on every GPU
   node.

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
