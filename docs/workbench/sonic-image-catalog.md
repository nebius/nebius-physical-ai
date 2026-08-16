# SONIC Image Catalog

The machine-readable source of truth is
`npa/src/npa/deploy/sonic_image_manifest.json`. Resolvers, workflow
materializers, and publishers all consume that manifest.

## Active image

Only `sonic-k8s-host-mounted` is active and publicly publishable. It is the
scanned CUDA 13 runtime-fetch image for RTX PRO 6000 Blackwell Kubernetes nodes
whose NVIDIA GPU Operator mounts driver-matched userspace:

| Variant | Tag | Driver provisioning | Use for | Why |
| --- | --- | --- | --- | --- |
| `sonic-l40s-baked` | `npa-sonic:0.1.2` | `baked` | L40S VM or compute-only host driver targets | The host does not mount the NVIDIA graphics userspace needed by Isaac Lab, so the image carries the matching NVML, GL, and Vulkan libraries. |
| `sonic-k8s-host-mounted` | `npa-sonic:0.1.2-k8s-runtime` | `host-mounted` | RTX PRO 6000 Blackwell on Kubernetes with the NVIDIA GPU Operator | The GPU Operator mounts driver-matched NVML, GL, and Vulkan libraries from the node, so the image must not carry conflicting driver libraries. |

Use `${NPA_REGISTRY}/npa-sonic:<tag>` for a concrete registry reference:

```bash
export NPA_REGISTRY=ghcr.io/nebius/nebius-physical-ai
```

It is the default only for supported RTX PRO Kubernetes routing. Select it
explicitly when useful:

```bash
npa workbench workflow submit \
  npa/workflows/workbench/npa-workflows/sonic-train.yaml \
  --registry "${NPA_REGISTRY}" \
  --gpu-target gpu-rtx6000 \
  --image-variant sonic-k8s-host-mounted \
  --accelerators RTXPRO-6000-BLACKWELL-SERVER-EDITION:1
```

## Quarantined images

`sonic-l40s-baked` and `sonic-mujoco-h100-mvp` remain in the manifest only as
explicit quarantine records. The former inherits an old
`nvcr.io/nvidia/isaac-lab` base and bakes NVIDIA driver libraries; the latter
inherits those same restricted bytes. Resolvers reject both variants. An EULA
credential or runtime flag cannot repair redistribution of bytes already baked
into an image.

sonic.submit_workflow(
    Path("npa/workflows/workbench/npa-workflows/sonic-train.yaml"),
    run_id="sonic-smoke",
    registry="<your-registry>/<namespace>",
    gpu_target="gpu-rtx6000",
    s3_endpoint="https://storage.eu-north1.nebius.cloud",
    s3_bucket="<bucket>",
    secret_envs=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
)
```

There is therefore no built-in compute-only image for the default serverless
L40S path, nor for H100/H200. `npa workbench sonic train --runtime serverless`
fails before provisioning unless the operator passes an independently built and
validated compute-only `--image`. The active host-mounted image must not be used
as that substitute: it depends on Kubernetes GPU Operator driver mounts.

## Build and publication

The retargeting workflow uses `NPA_RETARGETING_IMAGE` for the CPU preprocess
image. The committed default is
`ghcr.io/nebius/nebius-physical-ai/npa-retargeting:0.1.1`, a pushed
image that installs this repository's `npa` package, CPU preprocess
dependencies, and pinned upstream SONIC data-process scripts.

MJLab workflows use `NPA_WORKBENCH_IMAGE` for the generic Workbench CLI image.
The committed default remains
`ghcr.io/nebius/nebius-physical-ai/npa-genesis:0.4.6`.

## Build Commands

Baked L40S variant:

```bash
npa/docker/workbench/sonic/build.sh --registry "${NPA_REGISTRY}" --push --variant baked
```

Kubernetes host-mounted variant:

```bash
npa/docker/workbench/sonic/build.sh \
  --registry "${NPA_REGISTRY}" \
  --push \
  --variant k8s \
  --tag <new-additive-runtime-fetch-tag>
```

Publication selects the exact active tag from the manifest. The public image
must contain no Isaac Sim, Isaac Lab, Omniverse Kit, NVIDIA driver userspace, or
baked consent. Isaac dependencies are acquired at runtime only after the
operator supplies NVIDIA's documented, run-scoped `ACCEPT_EULA=Y`.

The current CLI defaults Isaac acceptance for non-interactive execution and
supports `--no-accept-eula`. The former `--accept-nvidia-eula VALUE` manual gate
is retired.
