# Prepare 360° captures for Lyra 2

Use the existing [Lyra reconstruction workflow](lyra-physical-augmentation.md)
on Nebius RTX PRO 6000 after preparing a perspective input bundle locally.
High-resolution panoramas need reprojection: resizing alone leaves spherical
distortion intact.

The pinned [upstream reconstruction entry point](https://github.com/nv-tlabs/lyra/blob/9fffc9adc37004091ecf26ef03abfb3abdf4d59a/Lyra-2/lyra_2/_src/inference/vipe_da3_gs_recon.py)
estimates pinhole intrinsics and applies `upper_bound_resize` at the configured
processing resolution. Workbench defaults to 128 reconstruction views and
`resolution=640`. Camera resolution does not become reconstruction resolution,
and upstream decodes frames before model resizing. Preparing a smaller
perspective sequence reduces that input memory requirement. Input compatibility
does not establish reduced haze or better geometry.

## Export and prepare

Export **stitched, mono, upright 2:1 equirectangular** MP4/MOV/MKV video or
JPG/PNG images using the camera's calibrated stitching software. Raw dual-fisheye,
`.insv`, `.insp`, DNG, stereo panoramas and arbitrary fisheye images are unsupported.
A 2:1 aspect ratio alone does not prove the projection; `--input-projection` is
the operator's explicit declaration. Inspect stitching seams and exposure
discontinuities first. Image rotation metadata is not applied.

Run from a current Workbench checkout with its installed Python dependencies:

```bash
npa/.venv/bin/python -m npa.workflows.lyra_panorama_capture \
  --input-path ./stitched-frames --input-projection equirectangular \
  --output-path ./lyra-panorama-input \
  --width 640 --height 480 --h-fov 90 --yaw 0 --pitch 0 --fps 30
```

For video, replace `./stitched-frames` with `./stitched-video.mp4` and omit
`--fps`. Preparation streams one frame at a time through PyAV's native FFmpeg
`v360` filter, supersamples at twice the output dimensions, reduces with Lanczos,
and encodes an RGB-only MP4. This local step requires no GPU, weights, camera SDK
or separate FFmpeg executable. PyAV must provide `v360`, `scale` and `libx264`.

| Option | Meaning and default |
| --- | --- |
| `--input-projection` | Required `equirectangular` declaration; no projection guessing |
| `--width`, `--height` | Perspective dimensions, 640×480; even values from 2 to 16382, reflecting the filter's 2× working-size ceiling |
| `--h-fov` | Horizontal FOV, 90°; finite and strictly between 0° and 180°; vertical FOV follows the aspect ratio |
| `--yaw`, `--pitch` | Fixed FFmpeg view rotation, both 0°; yaw within ±180°, pitch within ±90° |
| `--fps` | Image presentation cadence, 30; not recovered capture timestamps; video uses its source average frame rate |
| `--start-frame` | Zero-based source index, 0 |
| `--frame-count` | Optional exact excerpt length, at least two; otherwise all remaining frames |
| `--output-path` | New local directory or configured private S3 prefix |

Image directories use lexicographic filename order; use zero-padded names such
as `frame-000001.jpg`. Selected frames must have identical 2:1 dimensions.
An unavailable excerpt, invalid settings or undecodable input fails before
publication. Existing local output directories are never overwritten.

Keep one fixed view direction per sequence, with sufficient overlap and
translation between capture positions for pose estimation. Rotation around a
single point cannot establish metric geometry. For another direction, prepare a
separate bundle with a different yaw or pitch. Alternating cube faces as video
frames introduces artificial temporal jumps; this command does not join virtual
cameras or merge reconstructions.

The bundle contains `capture.mp4`, `capture.json`, generic attribution and
`checksums.json`, matching the existing input contract. It records dimensions,
projection settings, versions, selected source indices and hashes. Original video
frame times remain in the manifest; playback uses a constant cadence, including
for variable-rate input. Source filenames, absolute paths, EXIF and source
container metadata are omitted. Pixels and resulting artifacts remain private.

Publish the bundle to your private run-scoped input prefix, then follow the
[existing submit instructions](lyra-physical-augmentation.md#prepare-captured-video).
The pinned reconstruction image can consume this bundle unchanged; run the new
preparation command from the updated checkout. Larger output dimensions also
require explicitly selecting the workflow's `resolution` to test higher model
processing resolution. That setting remains an unqualified experiment.

## Compare quality and retain the splatting baseline

Keep an independently validated photogrammetry/splatting pipeline as the
comparison baseline. Its implementation must be supplied and validated before
Workbench can reproduce it. The
[COLMAP → gsplat reference runbook](gaussian-splatting-reference.md) provides
an independently specified sequence, exact upstream training revision, image
and camera handoffs, and a private recipe template. It is not a claim about
the tools used in an unshared pipeline. Record these details privately:

- Capture format, lens calibration, stitching/export settings and color handling.
- Image ordering, selection, reprojection, resize dimensions and intrinsics.
- Pose estimation, reconstruction/training commands, versions and parameters.
- Gaussian export format, coordinate convention, scale and render-camera settings.
- Evaluation images, acceptance criteria and measured outcomes.

Compare identical capture excerpts and matching render viewpoints. Reserve
views before fitting, then inspect haze/floaters, edges, seams, unseen surfaces
and pose consistency on those views. If haze is unacceptable, retain the
baseline and report that outcome. More pixels, a different camera or reprojection
does not establish a quality win.

Preparation is CPU-validated with synthetic direction-coded panoramas, image
and video sequences, and 8192×4096 images. These checks establish projection,
dimensions, ordering, metadata handling and bundle integrity. They do not
qualify panorama captures through Lyra GPU inference or measure reconstruction
quality. Existing RTX qualification used a separate perspective capture.
The panorama bundle has `metric_calibration=false` and cannot serve as calibrated
collision geometry. Existing Lyra model/output rights and internal R&D scope
still apply.
