# Reconstruct a capture with COLMAP and gsplat

Run this reference on an operator-controlled Linux CUDA environment on Nebius
with a GPU compatible with the selected PyTorch and gsplat build. Camera
registration precedes Gaussian training; preserve both stages when assessing
reconstruction quality. This is a manual upstream reference, not a registered
Workbench trainer or a reproduction of an undisclosed pipeline. Its commands
have been checked against upstream interfaces; this capture-to-splat sequence
has not been GPU-qualified by this PR.

The reference uses [COLMAP's CLI](https://colmap.github.io/cli.html) and
[gsplat's example trainer](https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/examples/simple_trainer.py)
at revision `937e29912570c372bed6747a5c9bf85fed877bae` (v1.5.3).
Record the installed COLMAP version and its command help with each run.
The example's [requirements](https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/examples/requirements.txt)
include additional packages and native extensions. Install in a separate training
environment with compatible CUDA-enabled PyTorch, then retain its dependency
lock or installed package inventory. A source pin alone does not lock that environment.

## Capture and image preparation

Retain the original capture privately. Export sharp, overlapping images with
consistent color and exposure. Record calibration, stabilization, stitching,
frame selection, projection, resolution and any HDR-to-SDR conversion. Moving
objects and stitching artifacts require explicit treatment; neither the camera
change nor Gaussian training guarantees their removal.

For this recipe, supply **undistorted perspective images** with fixed dimensions,
field of view and intrinsics. Obtain them from calibrated camera software or a
verified panorama reprojection. Keep zero-padded filenames and the mapping back
to source frames in private evidence. Use the original image exports when
available, avoiding another lossy video round trip for the splatting baseline.

Raw fisheye, dual-fisheye and equirectangular images are different projections.
Do not label them `PINHOLE` or carry original fisheye intrinsics into a resized
perspective export. This recipe uses one fixed virtual view per sequence;
multiple views require camera/rig handling beyond this reference. For Lyra's
MP4 input preparation, use the separate [360° capture command](lyra-360-capture.md).

Freeze a list of selected frames and an appearance evaluation split before
training. The pinned gsplat example reserves every Nth registered image in
sorted filename order for evaluation. This split holds images out of Gaussian
training, while COLMAP may have used those images for camera registration.
It is not independent pose validation. Save the resolved frame lists after
registration, including images that failed to register.

## Recover camera poses and prepare training data

Choose a fresh private run directory. Supply these environment variables with
operator-local paths on the execution machine; keep their actual values out of
public examples and reports:

| Variable | Required value |
| --- | --- |
| `NPA_SPLAT_RUN` | New private directory for database, sparse models and logs |
| `NPA_SPLAT_IMAGES` | Selected calibrated perspective image directory |
| `NPA_SPLAT_INTRINSICS` | Exported `fx,fy,cx,cy`, in pixels at these exact dimensions |
| `NPA_SPLAT_MODEL` | Sparse model directory selected after registration review |
| `NPA_SPLAT_TRAIN_DATA` | New directory for undistorted images and sparse model |
| `NPA_GSPLAT_SOURCE` | Checkout of the pinned upstream gsplat revision |
| `NPA_SPLAT_PYTHON` | Absolute interpreter path in the dedicated CUDA training environment |
| `NPA_SPLAT_RESULT` | New private training result directory |
| `NPA_SPLAT_TEST_EVERY` | Frozen evaluation stride, at least 2; upstream defaults to 8 |

The intrinsics must come from the selected export/projection calibration. Do
not guess them from the original lens or reuse them after changing dimensions.
For mixed cameras, varying crops or changing zoom, stop using this single-camera
recipe and configure the appropriate per-image cameras or rig instead.

```bash
umask 077
: "${NPA_SPLAT_RUN:?set a new private run directory}"
: "${NPA_SPLAT_IMAGES:?set the selected perspective image directory}"
: "${NPA_SPLAT_INTRINSICS:?set exported PINHOLE fx,fy,cx,cy}"
mkdir -p "$NPA_SPLAT_RUN/sparse"

colmap feature_extractor \
  --database_path "$NPA_SPLAT_RUN/database.db" \
  --image_path "$NPA_SPLAT_IMAGES" \
  --ImageReader.camera_model PINHOLE \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_params "$NPA_SPLAT_INTRINSICS"

colmap exhaustive_matcher --database_path "$NPA_SPLAT_RUN/database.db"

colmap mapper \
  --database_path "$NPA_SPLAT_RUN/database.db" \
  --image_path "$NPA_SPLAT_IMAGES" \
  --output_path "$NPA_SPLAT_RUN/sparse"
```

Inspect registered versus submitted images, connected components, reprojection
errors and camera trajectories before training. A successful command exit does
not establish adequate scene coverage. If COLMAP produces several models,
select and record the appropriate component; do not silently assume model `0`
contains the complete capture. Weak registration needs correction before splat
optimization. Exhaustive matching is a reference choice; retain the settings
if substituting a sequence-aware matcher for a longer capture.

Set `NPA_SPLAT_MODEL` to the reviewed component, then prepare the dataset:

```bash
: "${NPA_SPLAT_MODEL:?set the reviewed sparse model directory}"
: "${NPA_SPLAT_TRAIN_DATA:?set a new training-data directory}"
colmap image_undistorter \
  --image_path "$NPA_SPLAT_IMAGES" \
  --input_path "$NPA_SPLAT_MODEL" \
  --output_path "$NPA_SPLAT_TRAIN_DATA" \
  --output_type COLMAP
```

The pinned trainer expects `images/` plus a sparse camera/image/point model
under `sparse/0/` or `sparse/`. Use the intrinsics and image dimensions produced
by this export together. Preserve the original database and reviewed model.

## Fit and export Gaussian splats

Prepare the pinned checkout and install its library and example dependencies
in the dedicated CUDA environment. Inspect dependency terms separately; gsplat's
[Apache-2.0 source license](https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/LICENSE)
does not establish rights for every dependency, evaluation weight or capture.
The trainer's LPIPS metric can fetch pretrained scoring weights at runtime.
Retain their provenance and terms with private run evidence.

```bash
: "${NPA_GSPLAT_SOURCE:?set the upstream checkout path}"
: "${NPA_SPLAT_PYTHON:?set the dedicated CUDA environment interpreter}"
: "${NPA_SPLAT_RESULT:?set a new result directory}"
: "${NPA_SPLAT_TEST_EVERY:?set the frozen evaluation stride}"
git -C "$NPA_GSPLAT_SOURCE" checkout --detach \
  937e29912570c372bed6747a5c9bf85fed877bae
"$NPA_SPLAT_PYTHON" -m pip install "$NPA_GSPLAT_SOURCE"
"$NPA_SPLAT_PYTHON" -m pip install \
  -r "$NPA_GSPLAT_SOURCE/examples/requirements.txt"

"$NPA_SPLAT_PYTHON" "$NPA_GSPLAT_SOURCE/examples/simple_trainer.py" default \
  --data-dir "$NPA_SPLAT_TRAIN_DATA" \
  --data-factor 1 \
  --result-dir "$NPA_SPLAT_RESULT" \
  --test-every "$NPA_SPLAT_TEST_EVERY" \
  --disable-viewer --save-ply
```

`--data-factor 1` preserves the prepared dimensions rather than applying the
example's default additional downsampling. Training otherwise uses upstream
defaults; record the resolved configuration and any overrides. The example
writes checkpoints, evaluation statistics and rendered images, and the PLY
option exports Gaussian appearance. Preserve the native checkpoint and its
configuration alongside the PLY. Record coordinate normalization and units;
SfM scale and a Gaussian PLY alone do not establish metric collision geometry.

## Review and compare with Lyra 2

Review held-out appearance images and the reported PSNR, SSIM and LPIPS together
with haze/floaters, thin edges, holes, seams and view consistency. Retain failed
registration, weak regions and losing comparisons. Define acceptance criteria
before choosing configurations; a training-view replay alone is insufficient.

Compare the same capture subset, projection, visible region and render viewpoint
first. A higher-resolution splatting run and a 640-pixel Lyra run are different
conditions: record both dimensions and report that difference rather than
attributing a gain solely to the algorithm. Match the input conditions for a
controlled algorithm comparison, then assess each pipeline's chosen quality
configuration separately.

The current Lyra wrapper does not expose arbitrary held-out render poses or an
explicit train/evaluation split. Its native trajectory replay supports visual
inspection, but does not automatically reproduce gsplat's held-out evaluation.
A strict paired benchmark requires additional pose alignment, split handling
and rendering work; do not report its absence as a quality result.

## Record the recipe privately

Copy this lab-notes template into private run evidence. It documents the run;
it is not an `npa.workflow` spec or an executable training configuration. Fill
measurements from actual outputs. Leave unknown fields empty.

```yaml
capture:
  projection: perspective
  exporter_and_version: null
  image_dimensions: null
  calibration_hash: null
  selection_manifest_hash: null
  color_processing: null
poses:
  colmap_version: null
  registered_images: null
  submitted_images: null
  selected_component: null
  sparse_model_hash: null
training:
  gsplat_revision: 937e29912570c372bed6747a5c9bf85fed877bae
  environment_lock_hash: null
  resolved_configuration_hash: null
  checkpoint_hash: null
  gaussian_ply_hash: null
  coordinate_convention_and_scale: null
evaluation:
  split_manifest_hash: null
  pose_estimation_used_evaluation_images: true
  acceptance_criteria: null
  measured_results: null
  decision: null
```

Keep original names, locations, metadata, images, poses, commands with private
paths, and storage identifiers in access-controlled evidence. Public handoffs
may describe generic stages, source revisions and non-identifying aggregate
measurements. Check reusable prose and configuration for confidential details
before sharing; removing EXIF does not remove information visible in pixels.
