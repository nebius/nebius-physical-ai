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
Proprietary Software, but it is not in the public image. The image contains only
the reviewed upstream runtime lock and NPA bootstrap. On the first render use,
the operator receives the exact architecture-specific SDK directly from
NVIDIA's anonymous package index into operator-owned cache storage. The complete
lock SHA-256 is
`ed582577175e4a5b32f8b69ef9cdbfc3d7337f3786051d8b076e30a2652f6fa5`;
the x86_64 OVRTX wheel SHA-256 is
`a6b2b3c357f6487451c8d71e96cc4f83156c08fd9747d10e1b65f3866bed4b8f`.

NVIDIA's current [Omniverse licensing page](https://docs.omniverse.nvidia.com/connect/latest/common/NVIDIA_Omniverse_License_Agreement.html)
states that downloading or using Omniverse signifies agreement. The governing
[NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and [Product Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/)
are linked for review. NPA does not add a prompt, acceptance environment
variable, stored consent record, or self-certification mechanism.

Builds use the additive `0.5.2-npa2` tag and may be published only after the
exact built digest passes the byte/layer scan and real RTX workflow:

```bash
npa/docker/workbench/content-agents/build.sh --push
npa/.venv/bin/python npa/scripts/scan_content_agents_image.py \
  <candidate-image>@sha256:<digest> --expected-npa-source-sha "$(git rev-parse HEAD)"
```

The accepted public image is
`ghcr.io/nebius/nebius-physical-ai/npa-content-agents:0.5.2-npa2`, OCI index
`sha256:c64aaf6201bdaa013f9d16e8497290cf166907932f036297d7abaa430cbad7db`.
An unauthenticated manifest/config read found one `linux/amd64` manifest, one
bound attestation manifest, the `ubuntu` user, and the expected `public` and
`runtime-fetch` labels. The exact digest's specialized scan covered three nested
archives with zero findings; the general scanner covered 28,471 entries with
zero payload/history hits; Trivy reported zero critical vulnerabilities and
zero secrets.

The same digest completed the five-stage workflow on one NVIDIA RTX PRO 6000
Blackwell Server Edition. It emitted 6 material, 6 physics, and 1 validation
render plus 37 artifacts (1,808,557 bytes); upstream validation passed, rigid
body/collision/mass/friction checks were non-null, and both USD and USDZ reopened
independently. The immutable numeric record lives in
`npa/src/npa/deploy/content_agents_image_manifest.json`; future image changes
need a new additive tag and new evidence.

The scanner walks the final root filesystem, every individual image layer,
nested archives, and OCI history. It fails on OVRTX/Omniverse runtime bytes,
graphics-driver userspace, model weights, samples, populated caches,
credentials, customer data, or a build-time runtime fetch. Credentials,
endpoint IDs, customer data, and generated assets are never Docker build inputs.

The image separately excludes Scene Optimizer Core, OvPhysX, NVIDIA/default
material libraries, upstream samples, model weights, and caches. The workflow
generates a minimal PreviewSurface library at run time and calls a hosted
OpenAI-compatible VLM using `NEBIUS_TOKEN_FACTORY_KEY`; no model bytes are baked.
Optional telemetry exporters are not installed and OpenTelemetry SDK/exporters
are disabled by default.
NVIDIA driver userspace is also not baked. OVRTX requires the host-mounted
Vulkan/GLX libraries supplied by NVIDIA GPU Operator with `compute`, `utility`,
`graphics`, and `display` capabilities.

### Runtime cache and credentials

The bootstrap publishes only after exact-lock and import verification, under an
immutable identity containing the version, architecture, and complete lock
digest. A filesystem lock gives one writer; installation occurs in a unique
sibling temporary directory and the ready directory is renamed atomically.
Invalid content at that identity is never overwritten and render execution
fails closed.

The standard NPA cache wiring sets `NPA_CONTENT_AGENTS_RUNTIME_CACHE` beneath
`/opt/npa-model-cache/runtimes/content-agents`. A configured durable PVC is
shared by the material, physics, and validation jobs: ReadWriteOnce is sufficient
for this sequential workflow, while a ReadWriteMany claim also supports
concurrent readers on multiple nodes. The accepted RTX run used one bound
ReadWriteOnce claim and retained one unchanged ready identity across all three
jobs. Without a mounted cache, the fallback is beneath
`$XDG_CACHE_HOME/npa/runtime-cache/content-agents`; in a SkyPilot pod that is
node/pod-ephemeral, so a later job may download again. NPA never stores a token
or credential in the ready marker or cache metadata.

OVRTX is anonymous and uses neither `HF_TOKEN` nor `NGC_API_KEY`. Those
credentials remain relevant only to capabilities that actually fetch gated HF
or NGC artifacts. This workflow still requires `NEBIUS_TOKEN_FACTORY_KEY` for
its hosted VLM plus S3 credentials for stage handoff. `npa configure` and
`npa workbench health preflight` authenticate configured HF/NGC credentials,
while `health access` probes each selected gated artifact; neither flow grants
redistribution rights or invents an NPA EULA boolean.

## Validate and run

```bash
SPEC=npa/workflows/workbench/npa-workflows/content-agents-rigid-object.yaml
npa cluster up ... --gpu-driver-mode operator
npa workbench health preflight --checks s3,token_factory --json
npa workbench workflow validate-spec "$SPEC"
npa workbench workflow plan-spec "$SPEC" --run-id content-agents-check \
  --var bucket=<private-bucket>
npa workbench workflow submit "$SPEC" --runtime \
  --run-id content-agents-check \
  --var bucket=<private-bucket> \
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

Cancel the workflow before destroying its dedicated cluster. Preserve the exact
public digest and access-controlled run evidence needed for reproducibility;
remove transient jobs, services, controllers, and task-created clusters.
