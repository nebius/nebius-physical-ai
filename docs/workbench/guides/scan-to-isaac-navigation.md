# Prepare a captured scene for Isaac navigation

[Workflow catalog](../../../workflows/README.md) · [NuRec reconstruction](neural-reconstruction.md)

The [scan-to-Isaac workflow](../../../workflows/testing/scan-to-isaac-navigation.yaml)
combines an existing reconstructed visual scene with a supplied collision mesh,
preserves explicit scale and alignment, and exports a portable USDZ. A second
stage opens that package in Isaac Sim and measures native PhysX ray hits against
operator-specified distances. The intended runtime is Nebius Kubernetes with one
RTX PRO 6000 GPU; the scene assembly stage runs on CPU.

This is a static scene handoff. It does not train or evaluate a navigation policy,
construct a navigation map, qualify robot clearance, or prove safe navigation.
CPU assembly does not establish physics or NuRec visual-rendering readiness.
The committed live test is opt-in. The shared native probe passed on a complete
public RGB-D reconstruction in the RTX Isaac runtime; this separate supplied
visual/collision workflow has not been qualified end to end. See its
[readiness record](../../../workflows/testing/scan-to-isaac-navigation.readiness.json).

## Start from a capture

For calibrated metric RGB-D with measured poses, use the new
[RGB-D scan reconstruction workflow](rgbd-scan-to-isaac.md). It derives both
colored appearance and collision geometry from real Open3D TSDF integration,
checks held-out depth frames, and feeds the same portable USDZ/PhysX handoff.
It does not require the separate visual/collision inputs described below.

Use the existing [NCore reconstruction workflow](../../../workflows/main/nurec-reconstruct.yaml)
or [COLMAP ingestion and reconstruction workflow](nurec-colmap-reconstruct.md).
Both publish `reconstruction/last.usdz`. Retain the source capture and its
reconstruction provenance in operator-controlled storage. Stage that USDZ,
collision geometry, and the input manifest described below under one prefix.
The same contract also accepts an existing USD visual scene.

The existing public NuRec PPISP reference exercises visual reconstruction. It
does not provide a navigation collision qualification. Proprietary scans remain
operator inputs and are never shipped as example assets.

Gaussian splats do not supply collision geometry. NVIDIA's
[mono-camera robotics guide](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html)
describes separate physics geometry and ground alignment. The
[stereo-camera robotics guide](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_stereo.html)
uses nvblox to reconstruct a triangle mesh from aligned RGB, dense depth, and
poses. That upstream route needs calibrated capture data beyond this workflow's
existing NuRec USDZ input. This adapter therefore requires an explicit static USD
triangle mesh; it does not infer walls or obstacles from splats. A mesh produced
externally by the documented nvblox route can be converted to USD and supplied
with its measured transform. The metric RGB-D workflow is a separate direct
surface route; it does not infer collision from a Gaussian splat artifact.

## Input contract

`input_path` must be a worker-readable S3 prefix containing `scene.json` and all
referenced files. Direct module use also accepts a local directory. The default
empty value deliberately fails with an input error. The minimal directory is:

```text
scene.json
reconstruction/last.usdz
collision.usda
```

```json
{
  "schema": "npa.nurec.navigation_input.v1",
  "visual_asset": "reconstruction/last.usdz",
  "collision_asset": "collision.usda",
  "capture_provenance": {"sha256": "<sha256-of-capture-provenance>"},
  "visual_to_world": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "collision_to_world": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "ray_probes": [
    {"origin": [0,0,1], "direction": [0,0,-1], "min_distance": 0.99, "max_distance": 1.01}
  ]
}
```

Replace the provenance placeholder with 64 hexadecimal characters identifying
the retained capture provenance. Identity transforms and the downward ray are
illustrative only: they are correct only for assets already aligned to a floor at
world Z=0. Choose probe origins, directions, and distance intervals from measured
site surfaces, covering floors and relevant obstacles.

Each source must have one Xformable default prim containing all its content and
must explicitly author `metersPerUnit` and `upAxis`. The visual default prim
must contain actual geometry or a volume. Each
required 4×4 matrix is a finite rigid row-vector transform: translation occupies
the final row, in meters. Source units are converted to meters before applying
the matrix. Matrices map the source axes into the output's Z-up world; `upAxis`
metadata alone does not rotate an asset. Scale belongs in `metersPerUnit`, not in
the rigid transform. A monocular reconstruction needs independent scale
calibration before it can be a physical scene input.

Sources are static: time samples, value clips, instances, `resetXformStack`, and
preexisting physics bodies, joints, or scenes are rejected. Collision input is
triangulated USD mesh geometry with `subdivisionScheme = "none"` and no mesh
holes. Assembly validates mesh topology and finite coordinates, authors `UsdPhysics.CollisionAPI` and
triangle-mesh collision properties, and packages the visual dependencies. All
dependencies must be contained in the staged input tree or a contained USDZ.
Nested contained USDZ packages are checked recursively. Relative paths cannot
contain `..`, even when traversal would stay inside the input tree. External
references, missing payloads, and unsafe paths are rejected. Copy every
required local dependency into the input prefix before submission.

The raw Sdf layer audit also rejects
[OmniScripting attachments](https://docs.omniverse.nvidia.com/kit/docs/omni.usd.schema.omniscripting/1.0.1/Overview.html)
and [OmniGraph declarations](https://docs.omniverse.nvidia.com/kit/docs/omni.graph.docs/latest/dev/Usd.html)
before stage composition or Kit loading. It checks authored schema, type, and
execution-property opinions even in inactive prims, unselected variants,
overridden layers, and unused layers in nested USDZ archives. Hidden time
samples and value clips are rejected too. These checks preserve the static
scene contract; they do not establish a general sandbox for renderer plugins.
Ordinary material shaders, USD shading node graphs, and contained NuRec volume
payloads remain supported.

## Validate and submit

Use the [standard setup](../getting-started.md) for project, storage, credentials,
and Kubernetes configuration. Run the existing health preflight and verify the
selected images before submitting. The Isaac image fetches its proprietary
runtime under the existing run-scoped EULA contract; see
[Isaac Lab](../../../skills/tools/isaac-lab/SKILL.md). The CPU image provides
OpenUSD only for this stage; no Content Agents inference or OVRTX rendering runs.

```bash
npa workbench workflow validate-spec workflows/testing/scan-to-isaac-navigation.yaml --json
npa workbench workflow plan-spec workflows/testing/scan-to-isaac-navigation.yaml --run-id preview --json

npa workbench workflow submit workflows/testing/scan-to-isaac-navigation.yaml \
  --runtime --max-wait-seconds 0 --run-id "$RUN_ID" --project "$NPA_PROJECT" --infra "$NPA_INFRA" \
  --var "bucket=$NPA_OUTPUT_BUCKET" --var "input_path=$NPA_SCENE_INPUT_URI" \
  --var "isaac_image=$NPA_ISAAC_IMAGE" \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

Set the variables to your configured project, RTX Kubernetes target, fresh run
ID, writable bucket, and prepared input prefix. `NPA_ISAAC_IMAGE` must name an
immutable Isaac Lab image as `registry/repository@sha256:<64-hex-digest>`.
The default `tool://isaac-lab` is for planning; physics execution refuses it with
an instruction to select an immutable image. The workflow uses
`source_overlay: true` so generic submission stages the checkout's NPA source
with the image dependencies. `assembly_image` defaults to the catalog's Content
Agents image and can also be overridden with an exact digest through `--var`.
The `isaac_image` value selects the GPU worker image and is passed unchanged to
the probe's `--runtime-image` argument. Do not use a
global `--image` override: the two stages need different runtimes.

The development extra pins CPU OpenUSD 26.8 so offline USD regressions run in CI.
For local CPU preparation with that OpenUSD-enabled interpreter:

```bash
npa/.venv/bin/python -m npa.workbench.nurec.navigation_scene \
  --input-path "$NPA_LOCAL_SCENE_INPUT" --output-path "$NPA_LOCAL_SCENE_OUTPUT"
```

## Outputs and evidence

All outputs are under `s3://<bucket>/scan-to-isaac-navigation/<run-id>/` unless
`prefix` is overridden:

| Artifact | Meaning |
| --- | --- |
| `assembled/scene.usdz` | Portable visual scene plus explicitly authored static colliders |
| `assembled/provenance.json` | Input lineage, scene SHA-256, transforms, collider/triangle counts, requested probes; physics and visual-render validation remain false |
| `reports/physics_validation.json` | Native PhysX measured hits bound to scene and exact assembly-provenance SHA-256; runtime versions and declared image identity |

Each publication retains `.npa-navigation-claim.json` and
`.npa-navigation-complete.json` sidecars. The permanent claim and conditional S3
writes prevent reuse of an output prefix; the completion seal lists the published
file hashes. The Isaac consumer verifies this seal before reading the assembled
bundle. An incomplete or previously used output prefix is rejected. Retry with
a fresh run/output prefix; keep the sidecars with retained evidence.

Treat the scene as physically checked only when the report states
`physics_validated: true`, its `scene_sha256` matches the downloaded USDZ,
`assembly_provenance_sha256` matches the exact downloaded `provenance.json`
bytes, and every required probe passed. Replacing capture lineage or probe
expectations changes that provenance digest even when scene bytes are unchanged.
The report reads the Isaac Sim version from the running application's version
API and records the enabled `omni.physx` extension version when available;
unavailable extension metadata is explicitly reported. `runtime_image` is an
**operator declaration**, with `running_image_identity_verified: false`.
It does not attest the container's actual registry digest or detect a runtime
image replacement. A missing runtime, missing input, invalid geometry,
or wrong/missing hit fails the stage; no stock plane or robot substitutes for
the requested scene. Visual rendering remains unverified by this physics test.
Follow NVIDIA's [NuRec-in-Isaac guidance](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/assets/usd_assets_nurec.html)
to inspect the scene's NuRec volume with the compatible rendering extension.

## Opt-in live qualification

The live test submits the generic workflow through `--runtime`, retrieves the
actual S3 USDZ and reports, reopens the package with OpenUSD, and checks the
package digest and measured probes. Its local Python environment needs `usd-core`.
It has no synthetic fallback and is skipped unless explicitly enabled. Provide
existing storage credentials and operator-prepared scene inputs:

```bash
NPA_INTEGRATION_E2E=1 NPA_SCAN_TO_ISAAC_LIVE=1 \
NPA_SCAN_TO_ISAAC_INPUT_URI="$NPA_SCENE_INPUT_URI" \
NPA_SCAN_TO_ISAAC_BUCKET="$NPA_OUTPUT_BUCKET" \
NPA_SCAN_TO_ISAAC_INFRA="$NPA_INFRA" \
NPA_SCAN_TO_ISAAC_PROJECT="$NPA_PROJECT" \
NPA_SCAN_TO_ISAAC_ISAAC_IMAGE="$NPA_ISAAC_IMAGE" \
npa/.venv/bin/python -m pytest npa/tests/e2e/test_scan_to_isaac_navigation_live.py -q -s
```

`NPA_SCAN_TO_ISAAC_ISAAC_IMAGE` requires the immutable Isaac image digest.
`NPA_SCAN_TO_ISAAC_ASSEMBLY_IMAGE` optionally selects an exact assembly image;
otherwise the workflow catalog default applies.
The submit live matrix registers this as a real GPU workflow and excludes it
from unattended rotation because site assets, calibrated transforms, and probe
expectations require operator preparation. No live infrastructure is accessed by
the offline workflow tests.
