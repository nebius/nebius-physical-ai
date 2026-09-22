# SeedVR2 video restoration

[Workbench docs](README.md)

SeedVR2 restores a low-resolution MP4 for human review or visual annotation.
The original sensor clip remains authoritative: SeedVR2 is generative, so its
output is derived review media and can contain unsupported detail.

The candidate image is `npa-seedvr2:0.1.0-cu130-unbuilt`. It remains
publication-quarantined until built-image scans, a real workflow on the declared GPU,
objective preservation measurements, calibrated VLM review, and independent
review pass for the same commit and image digest.

| Evidence gate | Current result |
| --- | --- |
| Source and packaging | Source supports explicit H100/sample defaults and an opt-in B200/posterior-mode path; each new source requires its own qualified immutable image |
| CUDA applicability | B200 requires the existing additive `90;100` FlashAttention and `9.0;10.0` Torch/Apex build options, actual native extension measurement, and kernel execution; Hopper-only images are rejected |
| Actual 3B quality | The retained H100 sample run failed the fixed LPIPS-improvement and temporal-preservation gates; this change does not erase those failures |
| Posterior-mode quality | Unproven hypothesis; CPU conditioning/RNG controls are not encoder, GPU, or quality acceptance |
| Other models and GPUs | Private 7B diagnostics do not qualify this 3B tool. H200, L40S, RTX PRO 6000, and B300 remain outside this runtime contract |
| VLM evidence | Descriptive review cannot override failed objective preservation gates |

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
2. `restore`: run the official pinned SeedVR2-3B one-step entrypoint on one declared H100 or B200.
3. `verify`: on the same digest-bound GPU image, recompute runtime identity,
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
ratio. Non-dry execution requires exactly one full-memory, non-MIG device matching `--expected-gpu`
(`H100` by default, or explicitly `B200`), a digest-bound
`NPA_TASK_IMAGE`, and the full NPA source revision baked into that image; a tag
or caller-supplied revision is rejected. `--dry-run`
prints the exact official `torchrun` invocation without fetching the model or
touching storage. The Python SDK exposes `probe`, `restore`, `verify`, and
`review`.

## Explicit conditioning and hardware

`restore --conditioning-mode sample` keeps the upstream default and copies the
original pinned `configs_3b/main.yaml` bytes unchanged. The experimental
`--conditioning-mode posterior-mode` changes only `vae.use_sample` to false in
an owned working-directory copy; the upstream entrypoint and encoder remain
unmodified. The selected mode and source/effective configuration hashes are
recorded in `result.json` and checked during verification and review. These
hashes establish config consistency, not an independent producer attestation.

The pinned encoder wrapper still makes its posterior sample draw in both
branches. CPU controls exercise that exact control flow and reject changed RNG
consumption; real encoder/CUDA equality must be checked before a paired quality
experiment. The entrypoint still overrides the YAML to one diffusion step,
CFG scale 1.0 and rescale 0.0, with seed 666 by default. Posterior mode has no
established quality benefit, and the default remains `sample`.

`--expected-gpu B200` validates assigned hardware; it never allocates a GPU.
The workflow keeps hardware allocation in its resource profile and binds that
profile to the same declaration using `--var seedvr2_gpu=B200`. It keeps
sample conditioning unless `--var seedvr2_conditioning_mode=posterior-mode`
is also explicit. The SDK exposes the same `expected_gpu` and
`conditioning_mode` arguments. B200 needs a separately qualified SM90+SM100
image; selecting a name does not qualify compiled kernels or restoration.

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
  FlashAttention's own `FLASH_ATTN_CUDA_ARCHS` input defaults to `90`; use
  `--flash-attn-cuda-archs '90;100' --torch-cuda-arch-list '9.0;10.0'`
  for the B200-capable private build. The two target lists must agree. Its
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

## Check nonroot packaging locally

The final COPY instructions normalize license and source permissions so UID 1000
can read them even when the source context was created with umask `077`.
The native Docker regression creates disposable fixture images from an existing
local SeedVR image that already has the runtime notice directory. The image
must have a pre-existing tag, which is verified before and after temporary tag
cleanup; untagged inputs are rejected before any Docker mutation. The check
does not pull images, use GPUs, or verify restoration.
Set `NPA_E2E_SEEDVR_READABILITY_BASE_IMAGE` to that local `sha256:<64-hex>` image ID
(no default), then run:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_E2E_SEEDVR_READABILITY_BASE_IMAGE="sha256:<local-image-id>" \
npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_seedvr2_image_readability.py -q
```

Both umasks `022` and `077` must pass with UID 1000 reading the copied notice,
metadata, and nested source while retaining executable-file permissions.
Each test removes only its unique temporary image tag. The actual final image
still requires its own nonroot runtime, payload, and qualification checks.

## Limits

SeedVR2 does not recover unobserved truth, metric geometry, camera calibration,
or policy success. Upstream specifically warns about incomplete restoration
under heavy degradation or large motion, unpleasant generated detail, and
oversharpening on light degradation or small inputs. Preserve and compare the
source, use matched frame timing, and do not silently substitute restored clips
for training ground truth or metrology.


### Compiled CUDA extension targets

The default build targets Hopper `sm_90` for both FlashAttention and Apex.
To prepare an additive Hopper/B200 image, use
`--flash-attn-cuda-archs '90;100' --torch-cuda-arch-list '9.0;10.0'`.
The build rejects mismatched or duplicate targets and checks every CUDA-bearing
extension against the exact selected SASS set. PyTorch wheel architecture flags
alone do not prove extension compatibility. The historical Hopper-only image
contains no extension PTX and cannot run those kernels on B200.

These options prepare image bytes; they do not establish B200 functional
acceptance. A successor still needs exact-image security, licensing, delivery,
native kernel and complete workload validation on the selected GPU. Models,
library versions, inference backends and evaluation thresholds stay unchanged.
