# SONIC Image Catalog

The machine-readable source of truth is
`npa/src/npa/deploy/sonic_image_manifest.json`. Resolvers, workflow
materializers, and publishers all consume that manifest.

## Active image

Only `sonic-k8s-host-mounted` is active and publicly publishable. It is the
scanned CUDA 13 runtime-fetch image for RTX PRO 6000 Blackwell Kubernetes nodes
whose NVIDIA GPU Operator mounts driver-matched userspace:

```text
npa-sonic:cuda13-b300-0.1.2-k8s-runtime-sm80-sm90-sm100-sm103-sm120-20260803T034152Z
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

To restore either target, build a new runtime-fetch image, scan the built bytes
with `npa/scripts/scan_image_omniverse_payload.py`, record an active manifest
entry with an immutable digest, and validate it on the actual target GPU. Do not
overwrite or republish the quarantined tags.

There is therefore no built-in compute-only image for the default serverless
L40S path, nor for H100/H200. `npa workbench sonic train --runtime serverless`
fails before provisioning unless the operator passes an independently built and
validated compute-only `--image`. The active host-mounted image must not be used
as that substitute: it depends on Kubernetes GPU Operator driver mounts.

## Build and publication

Build the active host-mounted variant with an additive tag:

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
