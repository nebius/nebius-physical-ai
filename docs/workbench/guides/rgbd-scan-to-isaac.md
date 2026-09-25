# Reconstruct a metric RGB-D scan for Isaac

[Workflow catalog](../../../workflows/README.md) · [Existing USD scene handoff](scan-to-isaac-navigation.md)

The [RGB-D scan workflow](../../../workflows/testing/rgbd-scan-to-isaac.yaml)
turns calibrated depth, color, and measured camera poses into a colored surface
with matching static collision geometry. Open3D integrates observed surfaces;
OpenUSD packages them; Isaac Sim checks native PhysX intersections on an RTX
PRO 6000 GPU. It does not require a separately authored collision mesh.

The geometry stages run on CPU. The first uses the existing SONIC image's
Open3D runtime directly, without starting SONIC or downloading Isaac. The second
uses the existing Content Agents image's OpenUSD runtime. The GPU stage uses
the Isaac image. Override the three image settings with independently verified
immutable images when qualifying a run. This adds no image or baked dataset.

Reconstruction covers measured surfaces only. Missing surfaces and unseen space
stay unknown; the workflow does not invent floors, close holes, reconstruct from
unposed photographs, or infer metric scale from arbitrary COLMAP units. A
navigation task can consume the resulting `scene.usdz`, but robot clearance,
policy learning, and held-out navigation success require separate native runs.
An indoor public capture does not establish industrial-scene or customer-data
quality. Native GPU qualification remains pending in the
[readiness record](../../../workflows/testing/rgbd-scan-to-isaac.readiness.json).

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
regular pixel grid in every excluded depth frame. Coverage is hits/valid depth
observations, and inlier fraction is depth-matching hits/all valid observations;
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
npa workbench workflow submit workflows/testing/rgbd-scan-to-isaac.yaml \
  --runtime --project example-project --infra example-rtx-runtime \
  --var bucket=example-bucket \
  --var input_path=s3://example-bucket/captures/metric-room \
  --var isaac_image=registry.example.invalid/isaac@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

Optional `reconstruction_image` and `assembly_image` overrides must retain their
interpreter/dependency contracts. The workflow overlays the selected source URI;
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

The [opt-in native live test](../../../npa/tests/e2e/test_scan_to_isaac_navigation_live.py)
also qualifies this route. Set `NPA_SCAN_TO_ISAAC_INPUT_KIND=rgbd` with the
existing scene-handoff live-test variables documented in the USD guide; its
`NPA_SCAN_TO_ISAAC_INPUT_URI` must then reference the calibrated capture prefix.
`NPA_SCAN_TO_ISAAC_RECONSTRUCTION_IMAGE` optionally overrides the reconstruction
image. The test verifies scan and reconstruction hashes, held-out quality, and
the exact native PhysX artifact rather than treating a successful submit as proof.
