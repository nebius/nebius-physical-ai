# Lyra reconstruction, collision geometry and executed actions

The [Lyra workflow](../../../workflows/testing/lyra-reconstruction.yaml) runs
the released Lyra 2 reconstruction on captured RGB video. It preserves the
Gaussian PLY, native rendered video, estimated depth, intrinsics and camera
poses. The scene adapter calibrates those predictions, fuses a colored triangle
surface and imports the same triangles as static Isaac collision geometry.
The physical augmentation task then executes new Franka actions across object
position, mass and friction changes.

This is an internal R&D integration. Live qualification is required separately
for reconstruction, collision behavior and robot execution. A validated YAML,
a rendered Gaussian scene or a passing CPU test does not establish native
manipulation success. Customer-specific scene, embodiment and wrist-camera
quality remain separate validation work.

## Prepare captured video

The public `npa-lyra2:2.0-rtfetch2` image packages the CUDA compiler, Python and
hash-locked NPA dependencies independently of Isaac Lab. The pinned Lyra/DA3
source, inference dependencies and reconstruction checkpoint are fetched into
a private temporary environment on each job. The image contains no weights or
capture data. The workflow pins the accepted image digest and baked NPA source;
it does not overlay local payload code. The exact image passed native RTX PRO
6000 reconstruction and artifact verification: 128 views, 8,658,944 Gaussians,
320 rendered frames and an offline HTML review. Required SkyPilot startup tools
are baked, so missing packages cannot block startup during a snapshot outage.
See its [RTX qualification record](../validation/lyra2-rtx-20261009.json) and
[packaging record](../../../npa/docker/workbench/lyra2/REDISTRIBUTION.md).

Use a configured Workbench project with private object storage and an RTX PRO
6000 execution context. The standalone workflow defaults to `RTXPRO6000:1`;
the recorded run used preemptible capacity. The earlier `2.0-rtfetch1` image
retains its [B200 qualification](../validation/lyra2-b200-20261008.json).
That result does not qualify the repaired image on B200. Native Isaac camera
rendering and action execution use the separate scene workflow.

```bash
docker pull ghcr.io/nebius/nebius-physical-ai/npa-lyra2:2.0-rtfetch2
```

```bash
npa/.venv/bin/python -m npa.workflows.lyra_capture \
  --input-path ./capture.mp4 --output-path ./lyra-input
```

The command verifies that the video decodes and writes a checksummed bundle.
Add `--calibration-path ./calibration.json` when measured camera poses accompany
the video. The file uses schema `npa.lyra-camera-calibration.v1`,
`camera_convention: optical_x_right_y_down_z_forward`,
`world: {meters_per_unit: 1, up_axis: Z}`, and a `frames` list containing one
`camera_to_world` proper 4×4 transform per decoded frame, in playback order.
Pose translations are meters. The adapter checks matching frame counts and
rigid rotations; it does not infer or verify the external calibration itself.
Depth remains optional for this video-plus-poses path.

Alternatively, pass a directory following
the [calibrated capture contract](rgbd-scan-to-isaac.md). That path selects an
explicit excerpt (`--start-frame`, default 0; `--frame-count`, default 320),
preserves camera poses and reserves measured depth for validation. RGB remains
the only image input to Lyra. The public reference capture uses 30 Hz playback;
original capture timestamps remain in its manifest.

For an existing Lyra Gaussian export, rerun the original capture through this
wrapper to retain the predicted depth needed by the collision adapter. A Gaussian
PLY alone does not contain a validated collision mesh or movable object bodies.

Publish the bundle to a private S3 prefix through your configured project's
storage client, then submit:

```bash
npa workbench workflow submit workflows/testing/lyra-reconstruction.yaml \
  --project "$NPA_PROJECT" --infra "k8s/$NPA_CLUSTER" --stage-src \
  --run-id "$NPA_RUN_ID" --var "bucket=$NPA_S3_BUCKET" \
  --var "input_uri=$NPA_LYRA_INPUT_URI" \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

The stage uses upstream `vipe_da3_gs_recon --no_vipe`: Lyra's released two-pass
DA3 camera-estimation and Gaussian-reconstruction mode with the actual Lyra
reconstruction checkpoint. It does not run Lyra image-to-video generation,
VIPE, Cosmos appearance augmentation, or a world action model. `views` defaults
to 128 and `resolution` to 640, matching the selected reconstruction shape.

Source is pinned to `nv-tlabs/lyra` revision
`9fffc9adc37004091ecf26ef03abfb3abdf4d59a`. The model is
`nvidia/Lyra-2.0`, revision `c178c3fcf12b63cf98f6749999e6ecb63901669f`,
file `checkpoints/recon/model.pt`, SHA-256
`d26380a2d2ecceb6c7ed8ccdb6c53d2664259132ddc946c6c189cf29151c8042`.
The wrapper calls upstream inference and additionally retains its predictions;
it does not substitute another checkpoint or fabricate a scene.

## Calibrate and bind the scene

Download the completed reconstruction bundle. Collision extraction requires
matching metric camera poses in `capture.json`; raw video alone cannot establish
the robot's metric workspace. The adapter fits a proper similarity transform
to the calibrated camera trajectory, rejects degenerate motion and rejects
camera residuals above 10 cm RMS. That gate is a coarse registration check,
not a claim of millimeter contact accuracy.

```bash
npa/.venv/bin/python -m npa.workflows.lyra_geometry \
  --input-path ./reconstruction --output-path ./geometry
```

Open3D fuses the predicted depth at 7.5 mm voxel spacing and 30 mm TSDF
truncation. The lowest confidence quintile in each view is excluded; no unseen
floor or filled collision volume is added. These settings determine extraction
resolution, not measured reconstruction accuracy. The result remains a
collision candidate until native PhysX behavior is checked.

When the prepared capture includes measured depth, extraction also raycasts the
surface against that independent sensor data, which Lyra never receives. The
report includes coverage, inlier fraction, mean error and 95th percentile error.
Scene assembly rejects a failure against the capture's original thresholds.
These RGB views are not held out; this is independent depth validation.

Prepare `mounting.json` with `world_to_task`, a proper 4×4 rigid transformation
using **column-vector convention**. It maps calibrated scene coordinates into
the fixed-base Franka task frame. Translation is in meters. Select the mount
so the intended tabletop is at task Z=0 and the manipulation workspace near
X=0.5, Y=0 has real reconstructed support. An identity transform is appropriate
only when the scene already has this alignment.

```bash
npa/.venv/bin/python -m npa.workflows.lyra_scene_assembly \
  --input-path ./geometry --binding-path ./mounting.json \
  --output-path ./prepared-scene --run-id "$NPA_RUN_ID"
```

Assembly creates a self-contained `scene.usdc`, task recipe and checksum manifest.
Publish the prepared directory and checksummed geometry directory to private
storage before collection. The importer replaces the procedural studio table and
floor. It adds a fixed-base Franka and an explicitly inserted 5 cm cube; it
does not identify movable captured objects, infer their mass/friction or
recover robot actions from pixels. Manipulating an existing reconstructed
object requires segmentation, separate rigid/articulated geometry, physics
properties and its own task/controller binding.

Submit the [scene action workflow](../../../workflows/testing/lyra-scene-actions.yaml)
with the three matching bundles. It collects native actions, exports reports,
and generates the combined HTML automatically:

```bash
npa workbench workflow submit workflows/testing/lyra-scene-actions.yaml \
  --project "$NPA_PROJECT" --infra "k8s/$NPA_RTX_CLUSTER" --stage-src \
  --run-id "$NPA_ACTION_RUN_ID" --var "bucket=$NPA_S3_BUCKET" \
  --var "isaac_image=$NPA_ISAAC_IMAGE" \
  --var "prepared_uri=$NPA_LYRA_PREPARED_URI" \
  --var "reconstruction_uri=$NPA_LYRA_RECONSTRUCTION_URI" \
  --var "geometry_uri=$NPA_LYRA_GEOMETRY_URI" \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

Each condition executes fresh actions and retains every
attempt. The unchanged acceptance criteria require at least 10 cm lift,
at most 3 cm/s object speed and 30 consecutive stable control steps, together
with the existing tool-distance and drift checks. Accepted workspace-camera
demonstrations export to LeRobot/Rerun. The imported-scene path also records a
320×240 native wrist camera for close inspection; it does not establish the
quality of a particular real wrist reconstruction.

## Standalone Lyra HTML

Reconstruction can be reviewed independently of the collision and Isaac action
workflow. Its completed output bundle includes `index.html`, a single offline
file. To regenerate the viewer from a downloaded bundle:

```bash
npa/.venv/bin/python -m npa.workflows.lyra_reconstruction_demo \
  --input-path ./reconstruction --output-path ./lyra-standalone.html
```

This viewer verifies native artifact hashes, embeds captured and rendered videos,
and backprojects sampled model depth into an orbitable point preview with relative
confidence coloring. It requires complete reconstruction outputs; there are no
placeholder simulation panels. The point preview uses relative model coordinates
and does not claim metric collision accuracy or executed robot actions. The
separate native Gaussian video is produced by upstream Lyra's rasterizer.

The dedicated GHCR image reconstructed 128 views from a 320-frame public capture,
exported 8,658,944 Gaussians and rendered all 320 frames at 640×480 and 30 FPS.
The standalone HTML contains 176,128 sampled depth points. Desktop and mobile
Chromium checks verified offline playback, orbit/zoom controls, confidence
coloring and no external asset requests. The cold-start workflow completed
dependency installation, checkpoint verification, inference, HTML generation
and S3 publication without intervention. All 11 output files were independently
downloaded and hash-verified; both videos fully decoded.

Separately, metric extraction produced 679,534 triangles. Camera registration
RMS was 7.3 mm; independent measured-depth comparison found 97.74% coverage,
85.17% inliers and 7.67 cm mean absolute error. These passed the capture's
preselected coarse thresholds, not manipulation contact tolerances. Imported
scene physics and wrist rendering still require native RTX qualification.

## Combined reconstruction and action HTML

```bash
npa/.venv/bin/python -m npa.workflows.lyra_demo \
  --input-path ./lyra-input --reconstruction-path ./reconstruction \
  --geometry-path ./geometry --actions-path ./reports \
  --output-path ./lyra-demo/index.html --require-complete
```

The file works offline. It includes source/reconstruction playback, an orbitable
collision mesh, physical-condition replays and actual action/outcome telemetry.
The report checks that the reconstruction belongs to the input video, geometry
belongs to that reconstruction, and robot replay belongs to that geometry.
`--require-complete` refuses to write a finished review when reconstruction,
collision geometry, videos or executed trials are missing. The scene action
workflow enables this check automatically. Without this flag, an intentionally
partial diagnostic review labels missing stages as pending.

## Wrist reconstruction quality

Retain sharp, overlapping views of the manipulation/contact region and inspect
held-out wrist views. Moving robot links and hands need explicit treatment;
they should not silently become static environment colliders. Native Lyra
also exposes chunked reconstruction for longer videos, which upstream suggests
when long-sequence reconstruction blurs. This adapter currently uses the
single-pass reconstruction entry point in its two-pass pose-estimation mode;
it does not claim that chunking or appearance augmentation fixes a particular
capture. A Gaussian scene can look plausible while having incorrect contact
surfaces, scale or occluded geometry.

## Model and artifact boundaries

| Boundary | Delivery and scope |
| --- | --- |
| Source | Upstream Apache-2.0 code and pinned submodule are fetched at runtime; source attribution remains upstream. |
| Baked runtime | Dedicated digest-pinned CUDA/Python/NPA bootstrap, independent of Isaac; Lyra source, inference dependencies and checkpoint remain runtime fetches. |
| Weights | Exact Lyra reconstruction checkpoint is fetched anonymously from its official immutable URL and SHA-256 verified. |
| Input data | Operator-owned private capture, or separately attributed public TUM sample; no customer capture is committed. |
| Cache | Ephemeral per-stage source/environment/checkpoint directory; model bytes are not exported in run artifacts or an image. |
| Outputs | Private non-production internal R&D reconstruction and simulation evidence; no production or customer redistribution permission is claimed. |

Lyra's [source license](https://github.com/nv-tlabs/lyra/tree/main/Lyra-2)
is separate from the
[NVIDIA Internal Scientific Research and Development Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/).
Runtime fetch does not expand model/output rights. This workflow has no public
publishing or serving stage. A production/customer deployment must establish
its applicable rights independently. Isaac follows the existing Workbench
runtime-fetch policy and preserves an explicit EULA opt-out.
