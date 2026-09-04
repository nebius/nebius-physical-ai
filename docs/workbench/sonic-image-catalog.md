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
| `sonic-k8s-host-mounted` | `npa-sonic:0.1.2-k8s-runtime` | `host-mounted` | RTX PRO 6000 Blackwell on Kubernetes with the NVIDIA GPU Operator | The GPU Operator mounts driver-matched NVML, GL, and Vulkan libraries from the node, so the image must not carry conflicting driver libraries. |

The supported image is in the public GHCR namespace; no registry environment
variable or pull secret is required:

```bash
docker manifest inspect \
  ghcr.io/nebius/nebius-physical-ai/npa-sonic:<tag>
```

It is the default only for supported RTX PRO Kubernetes routing. Select it
explicitly when useful:

```bash
npa workbench workflow submit \
  npa/workflows/workbench/npa-workflows/sonic-train.yaml \
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

## Public MuJoCo release

`sonic-mujoco-runtime-fetch` is a new release, not a relabel of the legacy
digest. It builds independently on a digest-pinned official Python base from the
pinned Apache-2.0 SONIC source, a hash-locked MuJoCo/PyTorch closure, and Debian
EGL/GL libraries. The CUDA Toolkit runtime object files are retained only under
the redistribution grant in their included NVIDIA SDK terms. Isaac Sim, Isaac
Lab, Omniverse Kit, NGC/NLC layers, driver userspace, weights, credentials, and
accepted terms are absent. Isaac-facing modes retain the existing runtime-fetch
refusal and require caller-supplied acceptance.

The supported tag is bound to the exact public development digest that passed
the full Omniverse/layer/history scans plus a real B200 MuJoCo rollout with
artifact and metric checks. Future bytes require a new accepted digest.

The public source checkout intentionally skips every Git-LFS object. When the
upstream G1 mesh paths are LFS pointers, headless evaluation retains the
upstream joints, actuators, mass, and inertia but substitutes primitive collision
proxies and records `geometry_mode=primitive-proxy-no-lfs-payload`. This prevents
unclassified robot assets from silently entering the image; it is not a visual
or mesh-fidelity validation claim.

```python
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

## Build and publication commands

Do not rebuild or push either quarantined variant. Official NPA-owned image
publication runs only through the guarded workflow, which creates an immutable
`dev-<full-git-sha>` tag after its pre-publication gates:

```bash
gh workflow run publish-public-images.yml \
  --ref <prepared-branch> \
  -f development_sha=<full-git-sha> \
  -f build_development_tools=sonic-mujoco \
  -f dry_run=true
```

An operator may use `build.sh` with an operator-controlled generic registry for
BYOF interoperability. Such an image is not an NPA release. Public NPA images
must contain no Isaac Sim, Isaac Lab, Omniverse Kit, NVIDIA driver userspace, or
baked consent. Isaac dependencies are acquired at runtime only after the
operator supplies NVIDIA's documented, run-scoped `ACCEPT_EULA=Y`.

The current CLI defaults Isaac acceptance for non-interactive execution and
supports `--no-accept-eula`. The former `--accept-nvidia-eula VALUE` manual gate
is retired.
