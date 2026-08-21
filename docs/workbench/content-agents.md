# NVIDIA Content Agents rigid-object workflow

NPA packages NVIDIA Content Agents as a Tier 1 declarative workflow for one
self-contained, non-articulated USD object. It runs the upstream Material Agent,
Physics Agent, OVRTX renderer, and Validation Agent, then publishes a rigid-ready
USD/USDZ and an explicit Isaac Stage 2 handoff to S3-compatible object storage.
This is not a general scene converter or a robot-rigging claim.

## Upstream selection

The image fetches NVIDIA's maintained `v0.5.2` release at immutable commit
`36dbf3f274f8e256637230a05a085853f65cc175`. The Antioch fork was reviewed at
`9611fc17a899dee1a2fbf4837cce300019ad7210` (based on upstream v0.3.10).

No Antioch patch is carried. Current upstream has replaced the old material-copy
pipeline, handles PreviewSurface texture overrides and custom endpoint keys, and
supports portable USDZ hydration. Antioch's remaining internal model defaults
are unsuitable for a public integration. Generated-library material binding and
custom OpenAI-compatible endpoint behavior are covered by NPA tests so this
decision can be revisited if upstream regresses.

## Accepted contract

Input is either `generated://rigid-cube` or a customer-owned, self-contained
`.usd`, `.usda`, `.usdc`, or `.usdz` object. Each stage exchanges only named
objects under `s3://<configured-bucket>/content-agents/<run-id>/`:

1. `acquire` normalizes the source to USDA.
2. `materials` runs `material-agent run`, real hosted VLM classification, and
   local OVRTX before/after renders.
3. `physics` runs `physics-agent run` and authors `RigidBodyAPI`, `CollisionAPI`,
   mass/density, and friction/restitution material properties.
4. `validate` runs `validation-agent validate` with the upstream `render_valid`
   and `physics_sane` profiles and fresh OVRTX evidence.
5. `package` reopens the output, fails closed on missing schemas, creates a
   self-contained USDZ, and emits provenance, reports, `scene_spec.json`, and
   `sim_asset_manifest.json`.

The packaged SceneSpec preserves the authored USD mass choice without inventing
one: it contains either a positive `mass` or a positive `density`, never an
accepted null mass. The legacy `friction` scalar is always a finite authored
static or dynamic coefficient in the Isaac handoff range (`0.1` through `2.0`);
`friction_source` records which USD attribute supplied it, and any authored
`static_friction` and `dynamic_friction` values are preserved separately.
Restitution without either friction attribute is not rigid-ready.

The accepted consumer is one Isaac rigid object. Arbitrary scenes, articulated
robots, joints, and generated textures are not claimed. Genesis does not consume
USD directly through this handoff, so no Genesis compatibility is claimed.
Scene Optimizer and OvPhysX tuning are disabled because their separate runtime
dependencies are neither needed nor present.

## Licensing and packaging boundary

Content Agents source is Apache-2.0. OVRTX `0.3.0.312915` declares NVIDIA
Proprietary Software and is installed from upstream's SHA-256-locked wheel list
in an isolated environment. The operator must accept the current
[NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the
[Product Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/),
which incorporate the former Omniverse product-specific terms, before building
or using it. The build gate records that decision only in the invoking shell:

```bash
export NPA_CONTENT_AGENTS_ACCEPT_NVIDIA_OMNIVERSE_TERMS=YES
npa/docker/workbench/content-agents/build.sh --push
```

The complete image is `restricted`: build it only into the active operator's
private Nebius registry and retain its immutable digest. Never publish it to
GHCR or another anonymous registry. Acceptance, credentials, endpoint IDs,
customer data, and generated assets are not Docker build inputs.

The image separately excludes Scene Optimizer Core, OvPhysX, NVIDIA/default
material libraries, upstream samples, model weights, and caches. The workflow
generates a minimal PreviewSurface library at run time and calls a hosted
OpenAI-compatible VLM using `NEBIUS_TOKEN_FACTORY_KEY`; no model bytes are baked.
NVIDIA driver userspace is also not baked. OVRTX requires the host-mounted
Vulkan/GLX libraries supplied by NVIDIA GPU Operator with `compute`, `utility`,
`graphics`, and `display` capabilities.

## Validate and run

```bash
SPEC=npa/workflows/workbench/npa-workflows/content-agents-rigid-object.yaml
npa cluster up ... --gpu-driver-mode operator
npa workbench health preflight --output json
npa workbench workflow validate-spec "$SPEC"
npa workbench workflow plan-spec "$SPEC" --run-id content-agents-check \
  --var bucket=<private-bucket> \
  --var runtime_image=<private-registry>/npa-content-agents@sha256:<digest>
npa workbench workflow submit "$SPEC" --runtime \
  --run-id content-agents-check \
  --var bucket=<private-bucket> \
  --var runtime_image=<private-registry>/npa-content-agents@sha256:<digest> \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

Operator mode is intentional for this single-GPU PCIe RTX render cluster: the
Nebius managed CUDA image exposes CUDA but does not mount the graphics userspace
OVRTX needs. Do not copy this setting to NVSwitch clusters, where operator-mode
driver startup is unsafe. SkyPilot also replaces Docker entrypoints; the
rendered material, physics, and validation jobs explicitly start the reviewed
Xvfb wrapper and fail early if `libGLX_nvidia.so.0` is unavailable.

Use the NPA-configured S3 endpoint and credentials; the spec contains no bucket,
registry, project, or endpoint identifiers. Inspect status, logs, and artifacts
with `npa workbench workflow status|logs|artifacts`. Success requires non-empty
renders, a readable USDZ, a passing upstream validation result, and an adapter
whose physics inspection reports every required schema/property.

OVRTX path tracing requires RT cores, so the three render-bearing stages request
`RTXPRO6000:1`. B200/B300 have no RT cores and are deliberately rejected as
render targets. The hosted VLM is zero-GPU from NPA's perspective; there is no
honest B200 stage in this workflow.

Cancel the workflow before destroying its dedicated cluster. Preserve only the
private registry digest and run-scoped S3 evidence needed for reproducibility;
remove transient jobs, services, controllers, and clusters.
