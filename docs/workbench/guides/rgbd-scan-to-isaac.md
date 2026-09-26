# Reconstruct a metric RGB-D scan for Isaac

[Workflow catalog](../../../workflows/README.md) · [Existing USD scene handoff](scan-to-isaac-navigation.md)

The [RGB-D scan workflow](../../../workflows/testing/rgbd-scan-to-isaac.yaml)
turns calibrated depth, color, and measured camera poses into a colored surface
with matching static collision geometry. Open3D integrates observed surfaces;
OpenUSD packages them; Isaac Sim checks intersections using CPU PhysX queries
inside the RTX PRO 6000 runtime. It does not require a separately authored
collision mesh.

The geometry stages run on CPU using the existing SONIC image's
`/opt/npa/venv/bin/python`, without starting SONIC or downloading Isaac. The first
uses its Open3D runtime. Assembly installs `usd-core==26.8` without dependencies
into a fresh temporary directory, appends it after the selected source overlay,
and removes it after the stage. The image's Python environment stays intact.
The GPU stage uses the Isaac image. Pin the three image settings to independently
verified immutable images when qualifying a run. This adds no image or baked dataset.

Reconstruction covers measured surfaces only. Missing surfaces and unseen space
stay unknown; the workflow does not invent floors, close holes, reconstruct from
unposed photographs, or infer metric scale from arbitrary COLMAP units. A
navigation task can consume the resulting `scene.usdz` through the
[verified training handoff](#continue-into-native-navigation-training), but robot
clearance, policy learning, and held-out navigation success require separate native runs.
An indoor public capture does not establish industrial-scene or customer-data
quality. A complete public capture has passed the three-stage managed workflow
and native collision queries inside the RTX Isaac runtime. Learned navigation
remains a separate qualification. See the
[readiness record](../../../workflows/testing/rgbd-scan-to-isaac.readiness.json).

## Measured public reference

On September 25, 2026, the complete associated TUM RGB-D
`fr3/long_office_household` capture completed reconstruction, assembly and native
Isaac validation as one standard managed workflow. Both CPU stages used the
pinned SONIC image below; assembly added OpenUSD 26.8 in an isolated target.
The native stage used an RTX PRO 6000 Blackwell Server Edition with driver
580.173.02. Actual image digests were independently verified on all three workers.

| Check | Measured result |
| --- | --- |
| Capture split | 1,984 integration frames; 501 held-out frames |
| Excluded source frames | 100, with missing bounded pose or RGB/depth association recorded |
| Reconstructed geometry | 513,598 vertices; 921,561 triangles |
| CPU reconstruction | Open3D 0.19.0; 56.97 seconds |
| Held-out depth | 80,703 rays; 99.903% coverage; 94.118% within 10 cm |
| Depth error over hits | Mean 4.12 cm; 95th percentile 12.25 cm |
| Native collision handoff | All 501 measured-depth probes passed in Isaac Sim 6.0.1 / PhysX 110.1.13 |

The verified scene SHA-256 is
`cdd2744ce8cc4c2ca6fa7b5a0ec457344f76793fca91a8d1d155ac0bdd36011b`;
the native physics report SHA-256 is
`22f8cc9a0d53d193be34f3041ab330b714ea4f0f3e60f8a511465256be0ca81f`.
All three completion seals, original capture and reconstruction report bytes,
exact scene/provenance hashes, and every probe's measured distance interval were
independently checked. The public assembly bootstrap also passed an exact-image
CPU check against the full published reconstruction. Operational evidence remains
private. These are CPU PhysX scene queries inside the RTX Isaac runtime; this
qualification contains no rendered image or learned-policy result.

Reference attribution: J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers,
*A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012;
[TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset), CC-BY-4.0.

## Calibrated input

Upload `capture.json` and the referenced images under one private S3 prefix.
Every RGB image is three-channel uint8; depth is registered, undistorted uint16
axial Z depth, with zero marking missing observations. Both images use the same
pinhole calibration. `camera_to_world` matrices use **column vectors** with
optical x-right, y-down, z-forward camera axes and a metric Z-up world. This is
distinct from the existing USD handoff's row-vector registration matrices.
Camera poses must already be measured or estimated in that calibrated world.

The manifest schema is `npa.navigation.rgbd_capture.v1`:

| Field | Required content |
| --- | --- |
| `world` | `{"meters_per_unit": 1, "up_axis": "Z"}` |
| `camera_convention` | `optical_x_right_y_down_z_forward` |
| `intrinsics` | Positive integer `width`, `height`; `fx`, `fy`, `cx`, `cy`; `distortion: [0, 0, 0, 0, 0]` |
| `depth_units_per_meter` | Calibration scale; e.g. 5000 for TUM PNG depth |
| `depth_max_m` | Sensor's usable maximum measurement distance |
| `voxel_size_m`, `sdf_trunc_m` | TSDF spatial resolution and truncation distance, with truncation larger than one voxel |
| `validation` | `pixel_stride`, `distance_tolerance_m`, `min_coverage`, `min_inlier_fraction`, `max_mean_error_m` |
| `frames` | Complete selected input inventory; each record has `id`, `timestamp_s`, `split`, `camera_to_world`, `rgb`, `depth`, `rgb_sha256`, `depth_sha256` |

Frame image paths are relative contained files. `split` is `integration` or
`validation`; both are required. Identical RGB/depth pairs cannot cross the
split. Select quality thresholds and split membership before running. The stage
checks every hash, including held-out frames, before integrating. No frame-count
limit or implicit subsampling is applied to integration frames. Keep capture
license, attribution, calibration assumptions, temporal association errors, and
excluded-source-frame reasons in additional manifest fields; the exact manifest
travels with the output.

The stage calls real
[Open3D ScalableTSDFVolume](https://www.open3d.org/docs/release/python_api/open3d.pipelines.integration.ScalableTSDFVolume.html)
and extracts its measured triangle surface. It raycasts that surface against a
regular pixel grid in every held-out validation depth frame. Coverage is
hits/valid depth observations, and inlier fraction is depth-matching hits/all
valid observations;
misses cannot disappear from the denominator. Mean and 95th-percentile errors
are measured over hits. Failing the configured coverage, inlier fraction, or mean
error prevents publication. One representative inlier per held-out frame becomes
a native PhysX probe bounded by the **measured depth** tolerance. These probes
verify the collision handoff; the complete held-out score qualifies geometry.

## Execute on Nebius

Use the configured project, storage, and RT-core runtime after the normal health
and image preflights. Before creating an isolated runtime API, run
`npa workbench workflow stage-src --project example-project --bucket example-bucket`
from the reviewed checkout and retain its returned source URI. Pass that exact
URI to the submit process: a previously configured source prefix can predate
these modules. Replace all example values with private operator settings:

```bash
npa workbench workflow validate-spec workflows/testing/rgbd-scan-to-isaac.yaml --json
npa workbench workflow plan-spec workflows/testing/rgbd-scan-to-isaac.yaml --run-id preview --json
export NPA_SRC_S3_URI="s3://example-bucket/npa-src/npa/<verified-source-fingerprint>/"
export NPA_SCAN_CPU_IMAGE="ghcr.io/nebius/nebius-physical-ai/npa-sonic@sha256:c9ba0996b28f54b013e36da689638b386a7ef9c0c8c4413fc4b3c72ff1a808bb"
export NPA_SCAN_ISAAC_IMAGE="ghcr.io/nebius/nebius-physical-ai/npa-isaac-lab@sha256:e321e8631c7e318b5012dad210d9cd1001b7dc833cbff0369e420c5c12657ab6"
npa workbench workflow submit workflows/testing/rgbd-scan-to-isaac.yaml \
  --runtime --no-stage-src --project example-project --infra example-rtx-runtime \
  --var bucket=example-bucket \
  --var input_path=s3://example-bucket/captures/metric-room \
  --var "reconstruction_image=$NPA_SCAN_CPU_IMAGE" \
  --var "assembly_image=$NPA_SCAN_CPU_IMAGE" \
  --var "isaac_image=$NPA_SCAN_ISAAC_IMAGE" \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

Optional `reconstruction_image` and `assembly_image` overrides must retain the
`/opt/npa/venv/bin/python` interpreter, Open3D and Python dependency contracts.
Assembly also requires package-index access for the pinned OpenUSD wheel.
The workflow overlays the selected source URI;
enabling the overlay alone does not select or verify a particular checkout.
Retain its content fingerprint and verify the required module bytes before GPU
submission. Keep the control CLI checkout, Python import path, and configuration
fixed while sharing an isolated API; select each workflow's immutable payload
with its explicit source URI. The public SONIC
image supplies Open3D; local module testing needs `open3d==0.19.0`, Pillow, and
NumPy, and scene assembly additionally needs OpenUSD. Public reference inputs
must be downloaded at run time. Open3D is MIT-licensed; the separately downloaded
[TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) is
CC-BY-4.0 and requires its attribution when sharing derived assets.

## Evidence and reuse

`reconstructed/` contains the actual `surface.npz`, `reconstruction.json`, and
exact `capture.json`. `assembled/` contains `scene.usdz`, `provenance.json`, and
copies of both source reports. `reports/physics_validation.json` records actual
Isaac and PhysX versions and native ray hits. Every stage uses the existing
permanent conditional publication claim and completion seal; retries require a
fresh prefix. Reports bind input, surface, scene, and provenance hashes. The
declared runtime image remains an operator declaration, not pod attestation.

Python integrations use `reconstruct_capture(input_path, output_path)` from
`npa.workbench.nurec.navigation_reconstruction`, followed by
`prepare_scene(surface_prefix, scene_prefix)` from
`npa.workbench.nurec.navigation_scene`. Keep separate container stages joined by
S3 in a workflow. A single-process integration must supply both dependency sets.
Do not publish private captures, manifests, scene screenshots, storage locations,
or raw infrastructure logs as public PR evidence.

## Continue into native navigation training

The scan workflow above ends at native collision queries. To train a navigation
policy in its exact reconstructed scene, combine this change with the companion
[native navigation implementation, PR #805](https://github.com/nebius/nebius-physical-ai/pull/805).
The public helper below calls that implementation's existing `reference_bundle`
builder, strict `Recipe` reader and immutable artifact publisher. It does not
implement a second trainer. Until both changes are present, the helper's native
navigation imports are an unmet dependency. Use one reviewed combined checkout
to build inputs and select the runtime source overlay; do not edit a checkout
already serving active jobs.

Provide a local JSON file containing exactly `train_cases`, `eval_cases`, and
`probe`, measured for the reconstructed scene. Each case specifies `id`, `seed`,
`position_m: [x,y,z]`, `heading_rad`, and `goal_m: [x,y]`. Train/evaluation IDs,
seeds and physical resets must be disjoint; evaluation needs exactly `num_envs`
cases. `probe` supplies `free`, `parked`, `obstacle`, the native `actions` sequence,
and optional `tolerance`. Choose supported root heights and collision-free
starting footprints from measured geometry, and a positive obstacle control
that really contacts that geometry. The built-in reference uses a public
quadruped and three velocity actions; it cannot infer safe routes, fill missing
floor, or supply a private robot/controller integration. The authoritative types
are `Case`, `Probe` and `Recipe` in `npa.workflows.navigation.contract`.

After the scan's assembly and native physics stages have completed, run:

```bash
npa/.venv/bin/python -m npa.workbench.nurec.navigation_handoff \
  --input-path "$NPA_SCAN_ASSEMBLED_URI" \
  --physics-path "$NPA_SCAN_PHYSICS_URI" \
  --cases-file "$NPA_SCAN_NAVIGATION_CASES" \
  --output-path "$NPA_NAVIGATION_HANDOFF_URI" \
  --image "$NPA_NAVIGATION_IMAGE" --num-envs 4000 \
  --iterations "$NPA_PPO_ITERATIONS" \
  --episode-steps "$NPA_EVALUATION_STEPS" > "$NPA_NAVIGATION_HANDOFF_JSON"
export NPA_NAVIGATION_INPUT_URI="$(npa/.venv/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["workflow_input_uri"])' \
  "$NPA_NAVIGATION_HANDOFF_JSON")"
```

`NPA_SCAN_ASSEMBLED_URI` and `NPA_SCAN_PHYSICS_URI` are the scan run's `assembled/`
and `reports/` prefixes. `NPA_NAVIGATION_HANDOFF_URI` must be a fresh private S3
prefix; local directories are also supported for offline handoff checks.
`NPA_NAVIGATION_IMAGE` is an exact Isaac image digest, not `tool://isaac-lab`.
Use the native runtime, controller-asset access and graphics preflights from the
companion navigation guide. Iterations and evaluation horizon remain explicit
experiment inputs, not automatic workload limits.

The helper verifies both scan publication seals, exact scene and report hashes,
the original capture manifest, and every native probe's expected geometry. It
rejects case files that override scene, image or source bindings. Its output
contains `recipe.json`, the unchanged `scene.usdz`, `scan-lineage.json`, and exact
scan/case report bytes under `scan/`. The builder pins the reviewed navigation
source inventory and public low-level controller. S3 publication uses the
navigation publisher's conditional claim and readback. The returned
`workflow_input_uri` selects the completed immutable attempt containing the raw
bundle; use that URI, not its logical publication parent, for workflow preparation.
This operation seals inputs; `navigation_learning_verified` remains false.

The public handoff was exercised against these exact managed outputs using live
S3 publication and independent readback, all 501 native probe results, and 4,000
measured training plus 4,000 held-out cases. The real companion recipe builder
and preparation stage preserved the exact scene and report bytes. Its recipe
SHA-256 is `197f4e05fe273cf9d9348896d27fc5738b46c52107fd5694a36b3f33f40c73e4`.
This CPU input-preparation check supplies no navigation learning or GPU population
acceptance result; its native-runtime and learning verification flags remain false.

With PR #805 installed, set `NPA_NAVIGATION_WORKFLOW` to its checked-in
`shared-scene-navigation.yaml` in the testing workflow catalog. This specification
belongs to the companion change and is required before these commands can run.
Validate, plan and submit that existing native workflow:

```bash
test -f "$NPA_NAVIGATION_WORKFLOW"
npa workbench workflow validate-spec "$NPA_NAVIGATION_WORKFLOW" --json
npa workbench workflow plan-spec "$NPA_NAVIGATION_WORKFLOW" --run-id preview --json
npa workbench workflow submit "$NPA_NAVIGATION_WORKFLOW" \
  --runtime --no-stage-src --project "$NPA_PROJECT_ALIAS" \
  --infra "$NPA_RTX_RUNTIME" --run-id "$NPA_NAVIGATION_RUN_ID" \
  --var "bucket=$NPA_S3_BUCKET" --var "input_uri=$NPA_NAVIGATION_INPUT_URI" \
  --var "byof_image=$NPA_NAVIGATION_IMAGE" \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

Before that submission, explicitly stage and select the same combined source
using `NPA_SRC_S3_URI` as described above. The workflow then runs prepare → native
isolation controls → RSL-RL PPO training → reload/evaluate on held-out cases in
the reconstructed scene. Retain the handoff publication separately: native
training/evaluation carry their recipe and scene, while the handoff retains the
capture lineage. Require `training.json` and `evaluation.json` to report the
handoff's exact scene hash in `runtime.scene_sha256`, its recipe hash in
`recipe_sha256`, and the same image/source/controller bindings. Actual learning
requires a finite positive parameter update, the saved checkpoint, native
learning curves and a passed held-out evaluation; input publication or PhysX
ray probes alone establish none of those results.

For replaying a recorded field failure, resuming a real baseline and comparing
against an independent held-out scene, use the separate
[field-loop implementation, PR #802](https://github.com/nebius/nebius-physical-ai/pull/802).
That path additionally requires the real baseline and recorded failure inputs.
Public scene/controller experiments demonstrate the mechanism and do not stand
in for those private inputs or a validated private robot deployment.

## Scan-only live test

The [opt-in native live test](../../../npa/tests/e2e/test_scan_to_isaac_navigation_live.py)
also qualifies this route. Set `NPA_SCAN_TO_ISAAC_INPUT_KIND=rgbd` with the
existing scene-handoff live-test variables documented in the USD guide; its
`NPA_SCAN_TO_ISAAC_INPUT_URI` must then reference the calibrated capture prefix.
`NPA_SCAN_TO_ISAAC_RECONSTRUCTION_IMAGE` optionally overrides the reconstruction
image. The test verifies scan and reconstruction hashes, held-out quality, and
the exact native PhysX artifact rather than treating a successful submit as proof.
