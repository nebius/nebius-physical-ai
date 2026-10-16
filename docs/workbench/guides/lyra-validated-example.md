# Inspect a validated 360° reconstruction

The [2026-10-16 qualification](../validation/lyra2-panorama-rtx-20261016.json)
records a completed native Lyra 2 run on an RTX PRO 6000, using a real 5.7K
Insta360 One X capture from the public
[OmniPhotos dataset, version 2](https://researchdata.bath.ac.uk/948/).
This is a new panorama reconstruction, using normal 128-view settings and the
immutable Workbench image. It is separate from the earlier perspective example.

| Check | Actual result |
| --- | --- |
| Camera input | Entire provided Ballintoy stitched video, 5760×2880, 827 frames at 30000/1001 Hz |
| Preparation | Fixed 90° horizontal FOV, yaw 90°, pitch 0°; native `v360`, 2× supersampling and Lanczos reduction to 640×480 |
| Selection | Full video; view direction chosen before inference to exclude the default view of the operator and selfie stick |
| Native inference | 128 views, processing resolution 640, pinned Lyra source and checkpoint |
| Gaussian output | 8,658,944 native primitives; every floating-point property checked for finiteness |
| Render | All 827 frames, 640×480 at the upstream renderer's rounded 30 Hz |
| Geometry | 128 depth/confidence maps at 644×476, estimated cameras and intrinsics; numeric arrays finite |
| Transfer | Complete S3 readback matched every published SHA-256 |
| Replay fidelity | 19.692545947578196 dB PSNR over all paired decoded RGB frames |
| Offline browser | Desktop and mobile playback, frame cadence synchronization, confidence controls and exact YAML download checked; zero external requests |

The larger camera input is processed before inference. Lyra received a 640×480
perspective video; this run does not establish inference at native 5.7K or 8K
resolution. The model's processing grid is recorded separately from the input
and rendered video dimensions.

The standalone review embeds both actual videos, an orbitable preview of
176,128 model-depth samples, validation evidence and workflow downloads. The
point preview is backprojected depth, separate from the native Gaussian
rasterization shown in the video. No sharpening, dehazing, masking or geometric
alignment is applied to the comparison.

## Reproduce the processing path

Use a stitched, upright mono equirectangular SDR export. Follow
[360° capture preparation](lyra-360-capture.md), then submit the existing
[Lyra reconstruction workflow](../../../workflows/testing/lyra-reconstruction.yaml)
through the [standard Workbench runtime](lyra-physical-augmentation.md).
The qualified example used the whole provided video:

```bash
npa/.venv/bin/python -m npa.workflows.lyra_panorama_capture \
  --input-path ./stitched-video.mp4 --input-projection equirectangular \
  --output-path ./lyra-panorama-input \
  --width 640 --height 480 --h-fov 90 --yaw 90 --pitch 0
```

Keep the original source, prepared input, native outputs and independent
verification privately. A new execution needs its own qualification record;
the committed record identifies the exact completed example's bytes.

## Package the recorded GPU result

Given the exact native standalone `index.html` retained from this qualified run:

```bash
npa/.venv/bin/python -m npa.workflows.lyra_validated_example \
  --input-path ./qualified-native-index.html \
  --qualification-path docs/workbench/validation/lyra2-panorama-rtx-20261016.json \
  --workflow-path workflows/testing/lyra-reconstruction.yaml \
  --workflow-path workflows/testing/gaussian-splatting-reference.yaml \
  --output-path ./private-review/lyra-360-gpu-proof.html
```

The packager checks the recorded HTML, embedded native provenance and media
hashes, then fully decodes both videos. It pairs frames by index and validates
each stream against its own qualified cadence: source 30000/1001 Hz, native
render 30 Hz. Playback uses the corresponding rate ratio. RGB replay PSNR is
`10 log10(255² / mean((source − render)²))`, pooled over every RGB value in every
frame. Packaging itself is a CPU operation; the qualification records the actual
GPU execution that produced its inputs. HTML and adjacent validation JSON are
owner-only files.

## Quality scope and earlier evidence

This proves full panorama preparation and native GPU reconstruction on the
declared public capture. It does not establish reduced haze, held-out novel-view
quality, raw-fisheye support, Insta360 X5 performance or a matched comparison
with the [COLMAP → gsplat reference](gaussian-splatting-reference.md).
Inspect the actual input and native render before accepting scene quality.

A real COLMAP attempt on 128 frozen frames from this prepared capture did not
pass camera registration: its exported model contained 2 registered images and
1 3D point. The full-coverage gate stopped before gsplat training. The native
database and model archive were retained and read back with matching SHA-256;
their identities are in the qualification record. This failed baseline is not
evidence of a trained splat or a comparison between reconstruction quality.

The [earlier RTX qualification](../validation/lyra2-rtx-20261009.json) used a
public TUM RGB-D perspective office capture. Its
[retained-artifact review](../validation/lyra2-reference-review-20261015.json)
measured 19.492875897550903 dB over 320 frames; that packaging did not rerun GPU
inference and is not the new panorama proof.

OmniPhotos attribution is embedded under its CC BY 4.0 terms. Lyra's recorded
scope remains private, non-production internal R&D. Keep generated media out of
the public repository; the repository ships the renderer and sanitized evidence.
