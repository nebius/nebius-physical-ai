# SeedVR2 video restoration

[Workbench docs](README.md)

SeedVR2 restores a low-resolution MP4 for human review or visual annotation.
The original sensor clip remains authoritative: SeedVR2 is generative, so its
output is derived review media and can contain unsupported detail.

The candidate image is `npa-seedvr2:0.1.0-cu130-unbuilt`. It remains
publication-quarantined until built-image scans, a real H100 workflow run,
objective preservation measurements, calibrated VLM review, and independent
review pass for the same commit and image digest.

## Run the workflow

Upload one MP4 to your own S3 bucket, then submit the four-stage workflow:

```bash
npa workbench workflow validate-spec workflows/testing/seedvr2-video-restoration.yaml
npa workbench workflow submit workflows/testing/seedvr2-video-restoration.yaml \
  --var bucket="<your-bucket>" \
  --var seedvr2_input_uri="s3://<your-bucket>/inputs/low-resolution.mp4"
```

The stages are:

1. `probe`: fully decode and hash the exact input.
2. `restore`: run the official pinned SeedVR2-3B one-step entrypoint on one H100.
3. `verify`: read the uploaded result back, hash it, and decode it independently.
4. `review`: render a non-blended bicubic-left/SeedVR2-right MP4, fixed-frame
   contact sheet, JSON manifest, and browser-readable HTML page.

The output prefix contains `restored.mp4`, `upstream.log`, and `result.json`.
The review prefix contains `comparison.mp4`, `contact-sheet.png`,
`review.json`, and `index.html`. Existing output objects are never overwritten.

The direct batch interface uses the same implementation:

```bash
npa workbench seedvr2 restore \
  --input-path "s3://<your-bucket>/inputs/low-resolution.mp4" \
  --output-path "s3://<your-bucket>/runs/<run-id>/restoration/" \
  --run-id "<run-id>" \
  --output-height 480 \
  --output-width 640 \
  --seed 666
```

`--input-path` must identify one MP4 object and `--output-path` must be an S3
prefix. Output dimensions must be divisible by 16. `--dry-run` prints the exact
official `torchrun` invocation without fetching the model or touching storage.
The Python SDK exposes `probe`, `restore`, `verify`, and `review`.

## Identity and packaging

- Source: `ByteDance-Seed/SeedVR`
  `e4de8c24441a67e1b7df56abea10645059bb1185`, Apache-2.0.
- Model: `ByteDance-Seed/SeedVR2-3B`
  `37255ff8cccfb01071b87f635a5948ca8d53117c`, Apache-2.0.
- The four required model payloads are public, runtime-fetched, and checked
  against fixed sizes and SHA-256 hashes before inference.
- No weights, input videos, output videos, credentials, or populated model
  cache are baked into the image.

The optional service requires `SEEDVR2_TOKEN` and explicit
`SEEDVR2_ALLOWED_S3_ROOTS`. It serializes GPU operations and removes storage and
service credentials from the upstream inference process.

See the [redistribution record](../../npa/docker/workbench/seedvr2/REDISTRIBUTION.md)
and [operator skill](../../skills/tools/seedvr2/SKILL.md) for exact payload
identities and validation gates.

## Limits

SeedVR2 does not recover unobserved truth, metric geometry, camera calibration,
or policy success. Upstream specifically warns about incomplete restoration
under heavy degradation or large motion, unpleasant generated detail, and
oversharpening on light degradation or small inputs. Preserve and compare the
source, use matched frame timing, and do not silently substitute restored clips
for training ground truth or metrology.
