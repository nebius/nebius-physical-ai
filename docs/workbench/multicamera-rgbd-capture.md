# Calibrated multicamera RGB-D capture

[The workflow](../../workflows/testing/multicamera-rgbd-capture.yaml) renders a
camera rig in a static USD scene with **Isaac Sim 6.0.1.0**, then downloads and
validates the resulting S3 dataset in a separate CPU stage. No robot is required.
The existing Franka RGB capture adapter is unchanged. This is an executable
implementation with CPU contract tests; **GPU acceptance is pending**. See the
[readiness record](../../workflows/testing/multicamera-rgbd-capture.readiness.json).

The default `procedural://four-camera-room` input creates a repository-authored
room with four walls, a floor, a crate, lighting, four outward-facing calibrated
cameras, and three sampled rig poses. It contains no vendor or proprietary
assets. Its intended validation scope is camera synchronization and geometry.
It does not represent an industrial scan, navigation task, robot policy, or
encoder-training quality assessment. Supplied scenes never fall back to it.

## Run the standard workflow

Use a configured Workbench operator environment, staged branch source, S3
credentials, and an RT-core GPU (L40S or RTX PRO 6000). The capture toolRef routes
to the existing `isaac-lab` image and its `/isaac-sim/python.sh` interpreter.
That image fetches the pinned Isaac runtime through the existing operator
bootstrap; do not run this adapter with the system Python interpreter. The
supported Isaac Lab image is generation 3 beta; this adapter calls Isaac Sim
Replicator directly and does not instantiate a Lab robot environment.

```bash
npa workbench workflow validate-spec workflows/testing/multicamera-rgbd-capture.yaml
npa workbench workflow plan-spec workflows/testing/multicamera-rgbd-capture.yaml --run-id preview
npa workbench workflow submit workflows/testing/multicamera-rgbd-capture.yaml \
  --run-id '<unique-run-id>' --stage-src \
  --var 'bucket=<your-bucket>' \
  --var 'capture_input_uri=s3://<your-bucket>/<input-prefix>/request.json' \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

Omit `capture_input_uri` to render the explicit procedural fixture. The YAML
owns both stages; no custom outer orchestrator or workflow-specific CLI is
required. The existing bootstrap acceptance/opt-out behavior applies. Set the
storage endpoint through the normal Workbench credential configuration. Source
staging, runtime cache, credentials, image availability, and GPU compatibility
must be verified before a live submission. Schema validation does not prove any
of those prerequisites. The submit matrix includes this executable GPU case
and excludes it from automatic daily rotation until live qualification.

`config.source_overlay: true` is required for this new adapter. `--stage-src`
uploads the submitting checkout through the standard source staging path;
an already verified `NPA_SRC_S3_URI` can be used instead. The rendered GPU task
sets `NPA_SRC_OVERLAY=1`, installing that source into the Isaac interpreter even
if the existing image already imports an older NPA. The CPU validator receives
the same source URI. A baked image alone does not prove the new module exists.

The only workflow-specific configuration keys are `capture_input_uri`,
`capture_uri` (default run-scoped capture prefix), and `validation_uri` (default
run-scoped report object). `bucket` and `prefix` select the usual S3 destination.
Use a fresh run prefix for every capture; committed captures and validation
reports cannot be overwritten.

## Input bundle

An S3 object named `request.json` contains the calibration and sampled
trajectory. Every USD layer and texture it needs is another object relative to
that manifest's prefix, declared by SHA256 in `files`. The input adapter fetches
only those explicit objects and verifies every hash. It does not list buckets,
extract archives, or fetch dependencies from arbitrary USD URLs.

Generate an editable example locally, without Isaac or cloud access:

```bash
npa/.venv/bin/python -m npa.workflows.isaac_rgbd fixture --output-path rig-input
```

Replace its scene, cameras and trajectory, update the asset SHA256 values, and
stage the complete bundle to your input prefix through normal S3 tooling.
`validate_request` in `npa.workflows.isaac_rgbd.contract` validates the decoded
request on CPU. Example shape with one camera (the generated fixture has four):

```json
{
  "schema": "npa.isaac.rgbd.request.v1",
  "scene": "scene.usda",
  "files": {"scene.usda": "<lowercase SHA256 of the supplied layer>"},
  "pointcloud": true,
  "cameras": [{
    "id": "front", "width": 320, "height": 240,
    "intrinsics": [[240, 0, 159.5], [0, 240, 119.5], [0, 0, 1]],
    "T_rig_camera": [[0, 0, 1, 0.15], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    "depth_range_m": [0.1, 20.0]
  }],
  "trajectory": [{
    "timestamp_ns": 0,
    "T_world_rig": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.5], [0, 0, 0, 1]]
  }]
}
```

There are no implicit calibration, timestamp, or missing-pose defaults. Unknown
fields are rejected. IDs are unique lowercase identifiers. Transforms must be
finite, right-handed rigid 4x4 matrices. Timestamps are nonnegative integer
nanoseconds, strictly increasing and at most 2^53 for numeric representability.
They are simulation time relative to the capture origin, not Unix time. Every
sample is rendered; the adapter does not interpolate or check path feasibility.
Camera counts and resolutions are configurable independently. Pinhole focal
lengths must be positive; principal points must lie inside the image; skew and
lens distortion are unsupported. Input focal lengths and principal points are
in pixels. Non-square pixels and off-center principal points are supported.

Scene layers must declare `metersPerUnit=1` and `upAxis="Z"`. Supported USD
layers are `.usd`, `.usda`, and `.usdc`; packaged USD/USDZ is unsupported anywhere
in the bundle, including nested dependencies. Preflight uses USD package-path
and file-format inspection plus the USD ZIP reader to reject renamed packages;
ordinary texture files remain supported. Author asset paths
as plain contained relative POSIX paths without upward traversal, network
locations, hidden components, or expansion tokens. All referenced layers,
payloads, value clips and textures must be declared. Time-sampled properties
and executable OmniGraph nodes are rejected. Raw Sdf layer, prim and property
preflight runs before Kit opens the scene, including inactive prims and all
unselected variants. It rejects `OmniScriptingAPI`, `omni:scripting:*`, graph
schemas/properties, time samples and value clips. The opened stage must also
have no composition errors; a hash-valid layer referencing a missing prim is
still invalid. These are supported-content checks, not evidence of a reproduced
script exploit. This captures a static scene at
explicit camera poses: no physics advancement, rolling shutter, motion blur,
sensor noise, distortion, SLAM, robot controller or navigation evaluation is
provided. Supply a trusted USD bundle; this is not a sandbox for arbitrary USD
plugins or shaders. Existing lighting is used as authored.

## Output and geometry contract

`capture/manifest.json` is the commit marker. It records the complete request,
input asset hashes, both source and canonical request hashes, Isaac and
Replicator versions, procedural/supplied scope, and every frame/view record.
Artifacts are under a unique `capture/captures/<attempt>/frames/` prefix. A
failed or concurrent upload can leave uncommitted attempt objects, but cannot
modify data belonging to an existing committed manifest. Remove those unused
attempts using your normal retention policy. There is no success manifest until
all local bytes decode and validate and every artifact upload completes.

Each camera at every sample has:

| Artifact or field | Meaning |
| --- | --- |
| `rgb.png` | Decoded RGB uint8 PNG, H×W×3 |
| `depth.npy` | Float32 H×W optical-axis Z depth in **meters**, from `distance_to_image_plane` |
| `mask.npy` | Boolean H×W; true only for finite depths inside the camera's inclusive near/far interval |
| `points.npz` (optional) | `xyz_world_m`: float64 N×3; `rgb`: uint8 N×3; valid pixels in row-major order |
| `T_world_camera` | Column-vector optical-to-world matrix, `T_world_rig @ T_rig_camera` |
| `render_calibration` | Renderer-reported aperture, offset, focal length, view transform and resolution |
| `render_reference_time` | Exact numerator/denominator from that render product's `ReferenceTime` annotator |

Depth zero means invalid; NaN, infinity, negative, zero and out-of-range raw
depths are masked to zero. A view with no valid depth pixels fails the capture.
Axial depth is **not Euclidean ray range**. The optical frame is +X right, +Y
down, +Z forward; the world frame is the supplied USD Z-up meter frame. The
rig frame is operator-defined by its transforms. Pixel centers use integer
coordinates `u=0..W-1`, `v=0..H-1`, with `(0,0)` at the first pixel center.
Backprojection is `p_camera = depth * inverse(K) @ [u,v,1]`, then
`p_world = T_world_camera @ [p_camera,1]`. USD camera film offsets include the
half-pixel conversion from this convention to the physical image window.

Frames follow input trajectory order; cameras follow sorted IDs. Every camera's
render product participates in one blocking Replicator step per sample, with
`delta_time=0`, a paused timeline and four rendering subframes. All annotator
buffers are copied before changing any pose. The adapter checks the actual
timeline against the requested time, equal rational reference times across
cameras, and strictly advancing reference times across samples. Renderer camera
metadata must agree with the calibrated poses and intrinsics. Ordering and
sample times are deterministic; bit-identical RTX pixels across drivers or
hardware are not promised.

The CPU stage downloads every declared object again, verifies hashes, decodes
all images/arrays with NumPy pickle loading disabled, checks exhaustive unique
frame ownership and cross-camera alignment, and independently recomputes every
optional colored point cloud. It writes `validation.json` with actual frame,
camera, view and valid-depth-pixel counts plus the manifest SHA256. It does not
train an encoder or qualify simulation-to-real performance. Missing or corrupt
data, unsupported runtime versions and S3 failures propagate as nonzero stage
failures. The shared Franka Kit lifecycle ensures shutdown cannot hide them.

## Opt-in live acceptance

Run the procedural workflow through standard submit, then verify its retained
S3 artifacts. This test checks all data and the known 4.75 m central wall depth
for each camera, alongside non-flat fixture RGB. It does not launch additional
infrastructure or grant model/software terms on the operator's behalf.

```bash
NPA_INTEGRATION_E2E=1 \
NPA_RGBD_LIVE_MANIFEST_URI='s3://<your-bucket>/<run-prefix>/capture/manifest.json' \
NPA_RGBD_LIVE_REPORT_URI='s3://<your-bucket>/<run-prefix>/live-validation.json' \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_multicamera_rgbd_live.py -q
```

The first variable enables live tests; the second selects the procedural capture;
the third must be a fresh report object. With no opt-in, the test skips without
using storage. Retain GPU, driver, immutable-image and staged-source evidence
externally. No industrial scene or calibration has been qualified by this PR.

## API sources

The adapter uses original integration code against these official references;
it does not vendor runtime code or assets:

- [Isaac Sim v6.0.1 multicamera example](https://github.com/isaac-sim/IsaacSim/blob/045ca8b59622b99a408092124377c66346e8d9c2/source/standalone_examples/api/isaacsim.replicator.examples/multi_camera.py)
  establishes SimulationApp-first startup and per-render-product annotators.
- [The same pinned release's timed capture example](https://github.com/isaac-sim/IsaacSim/blob/045ca8b59622b99a408092124377c66346e8d9c2/source/standalone_examples/api/isaacsim.replicator.examples/custom_fps_writer_annotator.py)
  demonstrates timeline control and frozen-time Replicator stepping.
- [Versioned Replicator annotator API](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.13.30/source/extensions/omni.replicator.core/docs/API.html)
  specifies `distance_to_image_plane`, `CameraParams` and `ReferenceTime`.
  The installed extension version is recorded in every capture.
- [OpenUSD dependency extraction](https://openusd.org/release/api/dependencies_8h.html)
  documents nonrecursive enumeration of layer, payload and asset references.
- [OmniScripting schema 1.0.1](https://docs.omniverse.nvidia.com/kit/docs/omni.usd.schema.omniscripting/1.0.1/Overview.html)
  documents the applied API and script asset attribute rejected by preflight.

CPU regression tests parse actual USD fixtures with `usd-core==26.8`, included
in `npa[dev]`; it is not added to the deployed runtime, which uses Isaac's USD.
