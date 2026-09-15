# SONIC Image Catalog

[Workbench docs](README.md)

The machine-readable source of truth is
`npa/src/npa/deploy/sonic_image_manifest.json`. Resolvers, workflow
materializers, and publishers all consume that manifest.

## Active images

`sonic-k8s-host-mounted` and `sonic-mujoco-runtime-fetch` are active. The former is the
scanned CUDA 13 runtime-fetch image for RTX PRO 6000 Blackwell Kubernetes nodes
whose NVIDIA GPU Operator mounts driver-matched userspace:

| Variant | Tag | Driver provisioning | Use for | Why |
| --- | --- | --- | --- | --- |
| `sonic-k8s-host-mounted` | `npa-sonic:cuda13-b300-0.1.2-k8s-runtime-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | `host-mounted` | RTX PRO 6000 Blackwell on Kubernetes with the NVIDIA GPU Operator | The GPU Operator mounts driver-matched NVML, GL, and Vulkan libraries from the node, so the image must not carry conflicting driver libraries. |
| `sonic-mujoco-runtime-fetch` | `npa-sonic-mujoco:0.2.0-runtime` | `host-mounted` | MuJoCo checkpoint evaluation; B200 automatic selection or explicit variant | Isaac rendering and training are not published capabilities of this variant. |

Resolvers intersect GPU selection with the requested workload. ONNX
`sonic eval --backend container` requests `isaac-render`; choosing the MuJoCo
variant fails before evaluation. MuJoCo's separate `mujoco-eval` entrypoint
reads `SONIC_EVAL_CHECKPOINT_PATH`, not the ONNX/sidecar input contract. Passing
`--container-arg mujoco-eval` does not adapt those inputs.

Fine-tune and training callers request their own workload. A separately
validated custom workflow `--image` can supply a runtime outside the published
variants. Generic Blackwell labels containing B200 or B300 never change the
requested accelerator into RTX PRO 6000.

The supported image is in the public GHCR namespace; no registry environment
variable or pull secret is required:

```bash
docker manifest inspect \
  "ghcr.io/nebius/nebius-physical-ai/npa-sonic:<tag>"
```

It is the default only for supported RTX PRO Kubernetes routing. Prepare the
spec's actual GPU resource profile and required host mounts using the
[training runbook](cookbooks/sonic-train-runbook.md). `--gpu-target` selects image
routing; it does not rewrite `npa.workflow` resource profiles.

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

There is therefore no built-in compute-only image for the default serverless
L40S path, nor for H100/H200. `npa workbench sonic train --runtime serverless`
fails before provisioning unless the operator passes an independently built and
validated compute-only `--image`. The active host-mounted image must not be used
as that substitute: it depends on Kubernetes GPU Operator driver mounts.

## Build and publication

Legacy raw SkyPilot materializers use `NPA_RETARGETING_IMAGE` for the CPU preprocess
image. The committed default is
`ghcr.io/nebius/nebius-physical-ai/npa-retargeting:0.1.1`, a pushed
image that installs this repository's `npa` package, CPU preprocess
dependencies, and pinned upstream SONIC data-process scripts.

Those materializers use `NPA_WORKBENCH_IMAGE` for MJLab's generic Workbench CLI image.
The committed default remains
`ghcr.io/nebius/nebius-physical-ai/npa-genesis:0.4.6`.

## Build and publication commands

Do not rebuild or push either quarantined variant. Official NPA-owned image
publication runs only through the guarded workflow, which creates an immutable
`dev-<full-git-sha>` tag after its pre-publication gates:

```bash
gh workflow run publish-public-images.yml \
  --ref "<prepared-branch>" \
  -f development_sha="<full-git-sha>" \
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
