# Inspect a validated Lyra reconstruction

Start with the result: the retained native RTX PRO 6000 run reconstructed a
public TUM RGB-D office capture into 8,658,944 Gaussian primitives using 128
views. Its source and rendered videos each contain 320 frames at 640×480 and
30 Hz. The [original qualification](../validation/lyra2-rtx-20261009.json)
records the immutable image, checkpoint, native outputs and independent artifact
readback. This is a completed perspective reconstruction, not a panorama or
fisheye qualification.

The refreshed offline review puts synchronized source/render playback first,
followed by an orbitable preview of 176,128 actual model-depth samples, validation
evidence and workflow downloads. It embeds all media and makes no network
requests. The point preview is backprojected depth, separate from the native
Gaussian rasterization shown in the video.

## Revalidate the retained example

Keep the original qualified standalone HTML in private storage. The following
command requires that file's exact recorded SHA-256, matches its embedded native
provenance and video hashes, fully decodes both videos, checks frame dimensions
and presentation timestamps, and computes full-frame RGB replay fidelity:

```bash
npa/.venv/bin/python -m npa.workflows.lyra_validated_example \
  --input-path ./retained-lyra-review.html \
  --qualification-path docs/workbench/validation/lyra2-rtx-20261009.json \
  --workflow-path workflows/testing/lyra-reconstruction.yaml \
  --workflow-path workflows/testing/gaussian-splatting-reference.yaml \
  --output-path ./private-review/lyra-validated-example.html
```

The HTML and adjacent `.validation.json` are owner-only files. `--input-path`
is the qualified retained review; it is not arbitrary HTML or raw camera video.
`--qualification-path` supplies the trusted recorded identities. Repeat
`--workflow-path` for each YAML to embed verbatim. `--output-path` selects the
new single-file review. No GPU job is launched and no model is fetched.

The [refreshed validation record](../validation/lyra2-reference-review-20261015.json)
reports **19.492875897550903 dB PSNR** from all 320 paired decoded RGB frames.
The measurement is `10 log10(255² / mean((source − render)²))`, pooled across
every RGB value in every frame. No masking, geometric alignment, sharpening or
tone adjustment is applied. This measures replay of the original camera
trajectory; it is not held-out novel-view evaluation, an acceptance threshold,
or evidence of reduced haze. Native GPU inference remains the run dated
2026-10-09; refreshing the review does not rerun it.

## What remains unqualified

The [360° preparation path](lyra-360-capture.md) still needs a real panoramic
capture run. Its synthetic projection tests do not transfer this qualification
to high-resolution 360° or fisheye inputs. The [COLMAP → gsplat reference](gaussian-splatting-reference.md)
also needs its own compatible runtime and native training/evaluation evidence.
No matched comparison between the two pipelines is claimed.

TUM's [RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
is attributed in the generated file under its CC BY 4.0 terms. Lyra's recorded
scope remains private, non-production internal R&D. Keep generated media out of
the public repository; the repository ships the renderer and sanitized evidence.
