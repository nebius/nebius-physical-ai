---
name: content-agents
description: Use when packaging, validating, or operating the NVIDIA Content Agents workflow that enriches a self-contained USD object with real Material Agent, Physics Agent, OVRTX, and Validation Agent stages and emits a narrow Isaac rigid-object handoff.
---

# Content Agents

Use the Tier 1 `content-agents-rigid-object.yaml` workflow. Do not add a
first-class service or present upstream Docker Compose as orchestration. NPA's
declarative workflow and SkyPilot own jobs; the restricted image runs upstream
CLIs and a local isolated OVRTX subprocess.

## Boundary

- Accept one customer-owned self-contained USD/USDZ object, or the generated
  rigid-cube fixture.
- Produce a readable self-contained USDZ, upstream validation evidence,
  provenance, and `sim_asset_manifest.json` for one Isaac rigid object.
- Do not claim arbitrary scenes, articulated robots, joints, generated textures,
  or Genesis direct-USD consumption.
- Run Material Agent, Physics Agent, and Validation Agent through their real
  upstream entrypoints. An echo manifest or NPA reimplementation is invalid.

## Source and license gate

Use NVIDIA Content Agents v0.5.2 at
`36dbf3f274f8e256637230a05a085853f65cc175`. Antioch commit
`9611fc17a899dee1a2fbf4837cce300019ad7210` was reviewed; do not port it unless a
new failing test proves a current upstream gap.

Content Agents is Apache-2.0, but hash-locked OVRTX 0.3.0.312915 is proprietary.
The complete image is `restricted`, operator-built, and private-registry-only.
Before build/use, confirm the operator accepted the current NVIDIA Software
License Agreement and Product Specific Terms for NVIDIA AI Products (which
incorporate the former Omniverse product-specific terms), then set only in the
build shell. Official terms:

- `https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/`
- `https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/`

Set only in the build shell:

```bash
export NPA_CONTENT_AGENTS_ACCEPT_NVIDIA_OMNIVERSE_TERMS=YES
npa/docker/workbench/content-agents/build.sh --push
```

Never commit acceptance, tokens, configuration, customer assets, or live
infrastructure identifiers. Never push this image to GHCR or another public
registry. Scene Optimizer Core, OvPhysX, material/sample libraries, weights, and
caches must remain absent from the image.

## Operate

1. Run `npa workbench health preflight --output json`. Stop on any required S3,
   Token Factory, registry, or cluster failure.
2. Provision the single-GPU PCIe RTX cluster with `--gpu-driver-mode operator`.
   OVRTX requires the GPU Operator's host-mounted Vulkan/GLX userspace; the
   managed CUDA image exposes compute devices but not the required graphics
   libraries. This exception is specific to the RTX render recipe and must not
   be copied to NVSwitch clusters.
3. Resolve the active project's private registry and bucket through NPA config;
   never paste their identifiers into repository files.
4. Resolve the pushed image to `@sha256:` and verify labels, running pod image
   ID, and private-registry pull before workflow submission.
5. Validate and plan
   `npa/workflows/workbench/npa-workflows/content-agents-rigid-object.yaml` with
   the private bucket and digest-pinned image.
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
renderer therefore starts the image's reviewed Xvfb wrapper in each render
stage and fails before upstream execution unless `libGLX_nvidia.so.0` is
available from the GPU Operator mount.

## Failure and cleanup

Follow `$debug-failed-run` from status to stage logs, S3 evidence, pod reason,
image pullability, and scheduling. Resume only when prior immutable artifacts
remain valid. Cancel before destroy. Remove transient jobs, services,
controllers, and the dedicated cluster; retain only the private image digest
and run-scoped S3 evidence needed for reproducibility. Apply
`$protect-nebius-infra-details` before commits or collaboration text.

Full contracts and commands: `docs/workbench/content-agents.md`.
