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
3. `verify`: on the same digest-bound H100 image, recompute runtime identity,
   read the uploaded result back, hash it, and decode it independently.
4. `review`: render a non-blended bicubic-left/SeedVR2-right MP4, fixed-frame
   contact sheet, JSON manifest, and browser-readable HTML page.

The output prefix contains `restored.mp4`, `upstream.log`, and `result.json`.
The review prefix contains `comparison.mp4`, `contact-sheet.png`,
`review.json`, and `index.html`. Existing output objects are never overwritten.
After an interrupted multi-object publication, retain the partial prefix as
failure evidence and retry with a new run ID and output prefix.

The direct batch interface uses the same implementation. Bind restoration to a
probe from the exact input bytes:

```bash
npa workbench seedvr2 probe \
  --input-path "s3://<your-bucket>/inputs/low-resolution.mp4" \
  --output-path "s3://<your-bucket>/runs/<run-id>/probe.json" \
  --run-id "<run-id>"

npa workbench seedvr2 restore \
  --input-path "s3://<your-bucket>/inputs/low-resolution.mp4" \
  --probe-path "s3://<your-bucket>/runs/<run-id>/probe.json" \
  --output-path "s3://<your-bucket>/runs/<run-id>/restoration/" \
  --run-id "<run-id>" \
  --output-height 480 \
  --output-width 640 \
  --seed 666
```

`--input-path` must identify one MP4 object and `--output-path` must be an S3
prefix. `--probe-path` is mandatory for every non-dry restoration, and the run
fails if the probe run ID, URI, byte hash, or decoded media no longer matches
the input. Source and output frames may contain at most 1920x1080 pixels;
output dimensions must also be divisible by 16 and preserve the source aspect
ratio. Non-dry execution requires exactly one full-memory H100, a digest-bound
`NPA_TASK_IMAGE`, and the full NPA source revision baked into that image; a tag
or caller-supplied revision is rejected. `--dry-run`
prints the exact official `torchrun` invocation without fetching the model or
touching storage. The Python SDK exposes `probe`, `restore`, `verify`, and
`review`.

## Identity and packaging

- Source: `ByteDance-Seed/SeedVR`
  `e4de8c24441a67e1b7df56abea10645059bb1185`, Apache-2.0.
- Model: `ByteDance-Seed/SeedVR2-3B`
  `37255ff8cccfb01071b87f635a5948ca8d53117c`, Apache-2.0.
- The four required model payloads are public, runtime-fetched, and checked
  against fixed sizes and SHA-256 hashes before inference.
- No weights, input videos, output videos, credentials, or populated model
  cache are baked into the image.
- CUDA extension compilation occurs in a devel stage. The final image uses the
  digest-pinned cuDNN runtime base; wheel SDK headers and static archives are
  removed before the runtime-only environment is copied, with a retained
  hash inventory for the remaining shared libraries and license notice. The
  same boundary removes NVSHMEM headers, device archive, and bitcode while
  retaining inventoried shared runtimes, the wheel license, and the exact
  v3.4.5-0 product license containing bundled third-party notices.
- Source builds record bounded compiler fanout in OCI labels. The reviewed
  defaults are two outer FlashAttention jobs with one NVCC thread each and two
  Apex jobs; `build.sh` exposes positive-integer overrides for a differently
  sized trusted builder. Python dependency installation, FlashAttention
  compilation, and Apex compilation are separate cacheable layers.
  FlashAttention's own `FLASH_ATTN_CUDA_ARCHS` input is pinned to `90`; its
  upstream default is a multi-architecture `80;90;100;120` build and does not
  honor `TORCH_CUDA_ARCH_LIST`.

The optional service requires `SEEDVR2_TOKEN` and explicit
`SEEDVR2_ALLOWED_S3_ROOTS`. It serializes GPU operations. Media is forced
through the local MP4 demuxer with a restricted protocol list, and media/model
subprocesses receive purpose-specific environment allowlists rather than the
service environment. Storage, service, workload-identity, and metadata
credentials are not copied through environment variables. This is process
hygiene, not a sandbox: the child still shares the container filesystem and
network namespace, so keep the service internal and use pod-level identity and
egress policy. Verification and review re-read the probe and bind the same
workflow run, image digest,
baked source revision, model files, command, media hashes, and artifact URIs.
The verification document proves artifact/runtime consistency, not who
produced the restore bytes; workload acceptance additionally requires the
independently retained platform workflow receipt for the actual producer pod.

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
