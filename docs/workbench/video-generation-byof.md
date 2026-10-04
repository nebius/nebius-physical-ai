# Native GPU video and perception workflows

Mochi 1, CogVideoX 2B, and Wan 2.1 14B have separate text-to-video
workflow recipes. Each invokes its native Diffusers pipeline on a B200,
fetches an immutable checkpoint at runtime, decodes every output frame, and
publishes an MP4, capability JSON, and `npa_byof_summary.json` to your bucket.
The [catalog](oss-solution-catalog.md) records packaged GPU qualification status.
These are Tier 1 workflow integrations; they do not add standalone model servers.

## Curated public runtime packaging

Three source-only job runtimes package these integrations in public GHCR:

| Image under `ghcr.io/nebius/nebius-physical-ai/` | Release | Native capabilities |
| --- | --- | --- |
| `npa-diffusers` | `0.38.0-rtfetch-20260916` | Mochi 1, CogVideoX-2B, Wan 2.1 14B and Depth Anything V2 Small |
| `npa-lingbot-world` | `a43bec7-rtfetch-20260916` | LingBot World v1 camera-conditioned generation |
| `npa-sam2` | `2.1-rtfetch-20260916` | SAM 2.1 Small video-mask propagation |

The [container catalog](container-image-catalog.md) records accepted digests.
The [qualification record](validation/studio-public-models-20260916.json) binds
all six native capabilities to the exact public development images, actual
B200 execution, complete output hashes and full video decode. B300 remains
unvalidated for these images. Source revisions, NPA build SHA and model checkpoint
revisions remain distinct records.

Copy the matching `workflows/testing/byof-<model>.yaml` to your project. Its
`config.base_image` already selects the immutable public digest. Set your bucket,
prompt and seed, or the input URI and complete SHA256 for perception and LingBot.
Run credential/access preflight, validate and plan the spec, then submit to your
B200 cluster through the standard NPA runtime with `--stage-src`. Configure
Nebius, storage and model access externally. Public pulls need no registry
credential or Kubernetes `imagePullSecrets`:

```bash
docker pull ghcr.io/nebius/nebius-physical-ai/npa-diffusers:0.38.0-rtfetch-20260916
```

Each image provides `model-runtime health` (source metadata), `ensure` (the
hash-locked CUDA runtime), `status`, `exec COMMAND`, and `golden OPTIONS`.
`golden` runs native inference and validates every decoded output frame.
For example, on a CUDA-enabled container with a writable output mount:

```bash
model-runtime golden --model cogvideox-2b --output-path /outputs/cog
model-runtime golden --model depth-anything-v2 \
  --input-path /inputs/video.mp4 --output-path /outputs/depth
model-runtime golden --model sam2.1 --input-path /inputs/video.mp4 \
  --box 240 100 700 500 --output-path /outputs/masks
model-runtime golden --model lingbot-world --input-path /inputs/image.png \
  --gpus 4 --output-path /outputs/camera
```

Select the matching image for each capability. Optional `--prompt` and
`--seed` default to a warehouse-robot scene and 42. `--gpus` selects two or four
devices for LingBot; four is the default. Inputs are local mounted files.
The existing `NPA_WAN_RUNTIME_CACHE` and `NPA_WAN_RUNTIME_OFFLINE` settings
control the shared CUDA cache, and Hugging Face's standard cache and credential
settings control model downloads. No tenant or registry credential is baked.

## Operator-built alternative

Build the shared image in your own registry. The accepted base supplies pinned
OSS CPU dependencies and a hash-locked runtime-fetch mechanism for CUDA/PyTorch.
The child replaces the baked source with Diffusers 0.38.0; weights, credentials,
CUDA Python packages, and operator caches are absent from the build context.
Diffusers and the three selected checkpoints are Apache-2.0. Fetching the runtime
remains subject to its upstream terms. This operator BYOF image is not a new
official GHCR release.
Custom Mochi, CogVideoX and LingBot runtimes must include Protobuf 6.33.6 for
native SentencePiece tokenizer conversion. The public images already contain
that CPU dependency.

```bash
npa/.venv/bin/python npa/scripts/run_byof_repo.py \
  --repo-url https://github.com/huggingface/diffusers.git \
  --repo-ref 275869dcae4ebcfee6a80253fdabc56033335020 \
  --base-profile ubuntu \
  --base-image ghcr.io/nebius/nebius-physical-ai/npa-wan2-2@sha256:5780959ca6c6e7eb77ee7ea7d005fcf0f56db50783ce798dddee2809185eb837 \
  --registry '<your-registry>/<namespace>' \
  --image '<your-registry>/<namespace>/byof-diffusers:<your-immutable-build-tag>' \
  --build-command 'ln -s /workspace/.cache/npa/wan2-2/runtime/current/venv /opt/byof/.venv && ln -s LICENSE /opt/byof/LICENSE.txt && /opt/wan-base/bin/python -m pip install --no-deps /opt/byof' \
  --workload container-verify --skip-run --skip-push
```

Inspect the built image under the repository packaging and security procedures,
push to your authorized private registry, and resolve the pushed OCI digest.
Copy the desired `workflows/testing/byof-{mochi-1,cogvideox-2b,wan2.1-14b}.yaml`
outside the checkout. Set `config.base_image` to that exact digest, `config.bucket`
to your own bucket, and choose a prompt and seed. Configure standard registry
authentication, storage credentials, and Hugging Face access outside the recipe.
For a private registry, supply a Kubernetes `imagePullSecrets` reference through
your cluster configuration. A desktop Docker config that only names a credential
helper cannot serve as the pull secret: resolve the credential into a private
`kubernetes.io/dockerconfigjson` secret, and verify a pull from the target cluster.
Run credential/access preflight, validate and plan the spec, then submit through
the standard NPA workflow runtime to your B200 cluster. Use `--stage-src` when
submitting from this checkout: an older saved source URI does not contain these
adapters, and this flag explicitly stages the current source. The worker needs
the checked-out `npa.solutions.video_generation` adapter.

Provide enough writable cache space for the checkpoint and CUDA runtime: the
recipes request 128 GB for Mochi/Wan and 256 GB for LingBot. On Kubernetes,
configure that capacity in the node or mounted cache volume; a SkyPilot
`disk_size` request reserves ephemeral storage; it does not create a Kubernetes
persistent volume. Account for concurrent jobs' aggregate reservations. The LingBot
checkpoint cache alone occupied approximately 150 GiB during qualification.

Admission requires the allocated worker to pull those exact image bytes,
`smoke_exit_code: 0`, the named capability JSON, and a fully decoded nonblank,
moving video. The JSON records checkpoint revision, prompt, seed, native pipeline,
runtime package versions, real CUDA devices, frame measurements, and file SHA256.
Generated video is synthetic model output; this recipe makes no claim of physical
calibration, robot action conditioning, successful task execution, or training.

Scene rendering and narration belong to Studio. They do not change a model's
source or turn an editorial overlay into inference evidence.

## Camera-conditioned worlds and perception

The same BYOF builder supports the following additional pinned native sources.
Use the accepted base above, your own image destination, and the source/build
settings below. Retain `--skip-run --skip-push` until image inspection passes;
then push, resolve its digest, and bind the corresponding workflow. No checkpoint
is downloaded by these build commands.

| Runtime | Source repository and immutable revision | Build command |
| --- | --- | --- |
| LingBot World v1 | `https://github.com/Robbyant/lingbot-world.git` at `a43bec7f8091c83e9b30b16b912f6fc906236fa6` | `ln -s /workspace/.cache/npa/wan2-2/runtime/current/venv /opt/byof/.venv && test -f /opt/byof/LICENSE.txt && /opt/wan-base/bin/python -m pip install --no-deps scipy==1.15.3 easydict==1.13` |
| SAM 2 | `https://github.com/facebookresearch/sam2.git` at `2b90b9f5ceec907a1c18123530e92e794ad901a4` | `ln -s /workspace/.cache/npa/wan2-2/runtime/current/venv /opt/byof/.venv && ln -s LICENSE /opt/byof/LICENSE.txt && /opt/wan-base/bin/python -m pip install --no-deps hydra-core==1.3.2 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 iopath==0.1.10 portalocker==3.2.0` |

`byof-lingbot-world.yaml` takes an S3 image URI and its complete SHA256. It
invokes upstream `generate.py` with authored camera poses, approximate intrinsics,
FSDP and Ulysses on four B200 GPUs. The adapter enables NCCL implicit launch
ordering for their concurrent collective streams. Per-rank evidence must show actual attention
and all-to-all calls. This pins the v1 camera model; the separate World Infinity
successor, action control, training and real-time performance are outside its
validated scope.

### LingBot World Base (Cam) controlled-continuation contract

[`lingbot-world-controlled-continuation.yaml`](../../workflows/testing/lingbot-world-controlled-continuation.yaml)
is the durable base contract for SwitchWorld integrations. It is deliberately a
camera/control-conditioned video contract, not a robot interface: one verified RGB
context frame is paired with 161 OpenCV `poses.npy` records shaped `[161, 4, 4]`
and `intrinsics.npy` records shaped `[161, 4]`. The five connected stages prepare
the context and paired trajectories, make a prescribed native continuation, make a
matched alternative-control continuation while consuming the first result, decode
and measure the two videos, then publish an all-frame synchronized comparison MP4,
RRD and provenance. The two generation stages call the pinned upstream
`generate.py` path through its documented eight-rank FSDP/Ulysses topology. The
upstream command does not prescribe an accelerator product; NPA resolves the
eight-GPU product through workflow configuration for the selected target. They are
not a metadata wrapper or a training job.

The upstream Base (Cam) README prescribes eight local FSDP/Ulysses ranks, and
this contract rejects any other degree before it fetches stage artifacts or
starts inference. The pre-existing `byof-lingbot-world.yaml` four-rank image
smoke remains a separate historical qualification of that legacy capability. It
does **not** qualify this controlled-continuation graph, its topology, quality,
throughput, or scaling.

The input, both controls, model result, decode facts, and final media artifacts are
bound by SHA-256 in run-scoped S3 manifests. Preparation also requires a separate
`input_provenance_uri`: a JSON record that binds the exact selected image bytes to
the source URL, author, license, source hash, disclosed transformation, and
redistribution decision. The preparation artifact copies that record into its new
run-scoped prefix. This records actual input terms and credit; it is not a new
EULA, checkbox, or acceptance environment variable. Do not substitute an upstream
example image unless its asset-level provenance is recorded in that input sidecar.

The evaluation reports decoded frame counts, media dimensions and motion statistics,
pose-trajectory divergence, and paired RGB difference. A nonzero RGB difference
only establishes an observed rendered response to different authored controls; it
does not establish calibrated pose accuracy, robot-action dynamics, physical-world
accuracy, robot success, or real-time performance. The RRD timeline is
generated-video frame time, not a physical-camera clock.

#### Attribution, terms, and redistribution boundary

The upstream source is [Robbyant/lingbot-world at
`a43bec7f8091c83e9b30b16b912f6fc906236fa6`](https://github.com/Robbyant/lingbot-world/tree/a43bec7f8091c83e9b30b16b912f6fc906236fa6),
credited to the Robbyant Team and distributed under its
[Apache-2.0 license](https://github.com/Robbyant/lingbot-world/blob/a43bec7f8091c83e9b30b16b912f6fc906236fa6/LICENSE.txt).
The runtime checkpoint is [LingBot World Base (Cam) at
`6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd`](https://huggingface.co/robbyant/lingbot-world-base-cam/tree/6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd),
which labels the release Apache-2.0 and supplies the diffusion, VAE, and uMT5
encoder weights. The runtime fetch also pins tokenizer configuration and
SentencePiece assets from
[`google/umt5-xxl` at `66cb9e7e85526fe440a945569e42c72fb6cbc0ad`](https://huggingface.co/google/umt5-xxl/tree/66cb9e7e85526fe440a945569e42c72fb6cbc0ad).
Credit the upstream citation, including its full author list, exactly as
[published in the pinned source BibTeX](https://github.com/Robbyant/lingbot-world/blob/a43bec7f8091c83e9b30b16b912f6fc906236fa6/README.md#-citation):
*Robbyant Team et al., Advancing Open-source World Models, arXiv:2601.20540
(2026)*. Retain the upstream acknowledgement of the Wan2.2 team. NPA's
modification is limited to durable artifact wiring, matched-control evaluation,
and visualization; inference remains upstream `generate.py`.

The 2026-10-03 exact-revision review found the source license and Base (Cam)
model-card license above, with a public model page and no separate documented
click-through or output-use term on those exact pages. Therefore this integration
adds no NPA EULA, acceptance flag, telemetry, or duplicate attestation. The public
OCI image is a redistributable source-and-OSS bootstrap: it retains the Apache
license and upstream notices, but not model weights, context media, credentials,
or populated caches. CUDA/PyTorch/NCCL runtime payloads are fetched under their
respective provider terms; Base (Cam) weights and tokenizer assets are fetched
under their recorded Apache-2.0 releases; no training or evaluation dataset is
used by this inference contract. User-provided context media, their S3 retention,
and downstream generated outputs remain the operator's separate data-governance
responsibility and must be represented by the checksum-bound provenance sidecar.
Recheck these upstream pages before publishing a different source or checkpoint
revision; access to a payload alone is not a redistribution grant.

`byof-depth-anything-v2.yaml` uses the shared Diffusers/Transformers image.
The native Transformers `AutoModelForDepthEstimation` runs the pinned Depth
Anything V2 Small checkpoint on CUDA. `byof-sam2.1.yaml` uses the SAM image and
native `build_sam2_video_predictor` to propagate an operator's first-frame box.
Both require a video URI and complete SHA256. SAM boxes use an explicit 960×540
viewport; input frames are resized to that viewport. Both publish raw prediction
arrays and a decoded MP4 derived from those arrays. The depth visualization uses
one clip-wide scale. Neither workflow claims metric depth, ground-truth masks,
or successful robot task execution.

All these workflows use the public runtimes by default, with credentials and
cache configuration supplied externally. Source staging keeps orchestration
code aligned with the checkout. Custom images remain available through an
explicit `config.base_image` override and require fresh qualification.
