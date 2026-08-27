---
name: content-agents
description: Use when packaging, validating, or operating the NVIDIA Content Agents workflow that enriches a self-contained USD object with real Material Agent, Physics Agent, OVRTX, and Validation Agent stages and emits a narrow Isaac rigid-object handoff.
---

# Content Agents

Use the Tier 1 `content-agents-rigid-object.yaml` workflow. Do not add a
first-class service or present upstream Docker Compose as orchestration. NPA's
declarative workflow and SkyPilot own jobs; the public zero-vendor-payload image
runs upstream CLIs and a runtime-fetched isolated OVRTX subprocess.

## Boundary

- Accept one customer-owned self-contained USD/USDZ object, or the generated
  rigid-cube fixture.
- Produce a readable self-contained USDZ, upstream validation evidence,
  provenance, and `sim_asset_manifest.json` for one Isaac rigid object.
- Do not claim arbitrary scenes, articulated robots, joints, generated textures,
  or Genesis direct-USD consumption.
- Run Material Agent, Physics Agent, and Validation Agent through their real
  upstream entrypoints. An echo manifest or NPA reimplementation is invalid.

## Source and licensing boundary

Use NVIDIA Content Agents v0.5.2 at
`36dbf3f274f8e256637230a05a085853f65cc175`. Antioch commit
`9611fc17a899dee1a2fbf4837cce300019ad7210` was reviewed; do not port it unless a
new failing test proves a current upstream gap.

Content Agents is Apache-2.0. OVRTX 0.3.0.312915 is a proprietary runtime SDK,
not model weights, and must be absent from every public-image layer. On first
render use, fetch the exact architecture-specific wheel directly from NVIDIA's
anonymous package index through upstream's reviewed lock (complete SHA-256
`ed582577175e4a5b32f8b69ef9cdbfc3d7337f3786051d8b076e30a2652f6fa5`).

NVIDIA's current Omniverse page says downloading or using signifies agreement.
Link the official terms for review, but do not add an NPA acceptance flag,
prompt, stored consent record, or replacement `ACCEPT_*` variable:

- `https://docs.omniverse.nvidia.com/connect/latest/common/NVIDIA_Omniverse_License_Agreement.html`
- `https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/`
- `https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/`

The image may carry only the downloader and reviewed lock. Scene Optimizer Core,
OvPhysX, OVRTX, Omniverse runtime payload, material/sample libraries, weights,
caches, credentials, and customer data must remain absent. Scan the final
rootfs, every layer, nested archives, and OCI history before publication.

Use the standard model/runtime-cache wiring. The immutable identity includes
OVRTX version, architecture, and complete lock digest; use one filesystem
writer lock, a unique sibling temporary install, verification, and atomic ready
publication. A configured durable PVC is shared across render jobs: ReadWriteOnce
is sufficient for this sequential workflow, and ReadWriteMany is needed only for
concurrent cross-node readers. The fallback beneath XDG cache is
pod/node-ephemeral and may download once per job.
Never include credentials in cache paths, markers, metadata, logs, or artifacts.

## Operate

1. Run `npa workbench health preflight --checks s3,token_factory --json`. Stop on
   any required S3, Token Factory, registry, or cluster failure. OVRTX is
   anonymous and requires neither HF nor NGC; Content Agents still requires the
   Token Factory key. Generic preflight authenticates configured HF/NGC values,
   while `health access` proves entitlement for selected gated artifacts.
2. Provision the single-GPU PCIe RTX cluster with `--gpu-driver-mode operator`.
   OVRTX requires the GPU Operator's host-mounted Vulkan/GLX userspace; the
   managed CUDA image exposes compute devices but not the required graphics
   libraries. This exception is specific to the RTX render recipe and must not
   be copied to NVSwitch clusters.
3. Resolve the active project's registry and bucket through NPA config; never
   paste their identifiers into repository files.
4. Resolve the candidate image to `@sha256:`, byte-scan that exact digest, and
   verify labels and the running pod image ID before workflow submission.
5. Validate and plan
   `npa/workflows/workbench/npa-workflows/content-agents-rigid-object.yaml` with
   the private bucket and digest-pinned candidate image.
6. Submit through `npa workbench workflow submit ... --runtime`, forwarding
   `NEBIUS_TOKEN_FACTORY_KEY` and S3 credentials through `--secret-env`.
7. Require non-empty material/physics/validation renders, a passing upstream
   `validation_result.json`, a reopenable USD/USDZ, and all rigid/collision/
   non-null authored mass-or-density/material-binding checks in the manifest.
   Friction requires at least one authored static or dynamic coefficient in the
   Isaac handoff range; restitution alone is not friction.

All render-bearing states must use `RTXPRO6000:1`. B200/B300 lack RT cores and
are invalid OVRTX targets. Hosted VLM inference is already zero-GPU from NPA's
perspective; do not create a B200 job merely to exercise capacity.

SkyPilot replaces image entrypoints during Kubernetes bootstrap. The workflow
renderer therefore bootstraps the exact runtime cache, starts the image's
reviewed Xvfb wrapper in each render stage, and fails before upstream execution
unless `libGLX_nvidia.so.0` is available from the GPU Operator mount. CPU-only
acquire/package stages must not fetch OVRTX.

## Failure and cleanup

Follow `$debug-failed-run` from status to stage logs, S3 evidence, pod reason,
image pullability, and scheduling. Resume only when prior immutable artifacts
remain valid. Cancel before destroy. Remove transient jobs, services,
controllers, and the dedicated cluster; retain only the exact public image
digest and access-controlled run evidence needed for reproducibility. Apply
`$protect-nebius-infra-details` before commits or collaboration text.

Full contracts and commands: `docs/workbench/content-agents.md`.
