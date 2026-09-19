# SAM 3.1 container

`npa-sam3` is a Linux amd64 job image for SAM 3.1 text-prompted video tracking.
The public image contains the NPA runner and a neutral Python bootstrap.
SAM source, the gated checkpoint and CUDA packages are fetched into an operator
cache at runtime. No model payload, credential or accepted-terms receipt is baked.

This is a **development candidate**. Real GPU qualification is pending model
access. `health` proves the bootstrap only; it does not prove segmentation.
The existing SAM 2.1 release remains separately available.

## Access

1. Request access to [facebook/sam3.1](https://huggingface.co/facebook/sam3.1)
   while signed in to the account that owns your token.
2. In [token settings](https://huggingface.co/settings/tokens), grant that token
   read access to the approved gated repository. Model approval and token
   permission are separate requirements.
3. Supply `HF_TOKEN` through your job's secret environment. Do not put it in
   a Dockerfile, image build argument, command argument, workflow or commit.

## Run

Use the exact public development reference from the publication receipt:

```bash
export SAM3_IMAGE="ghcr.io/nebius/nebius-physical-ai/npa-sam3:dev-<full-git-sha>"
docker run --rm "$SAM3_IMAGE" health
docker run --rm -e HF_TOKEN "$SAM3_IMAGE" access
docker run --rm --gpus all -e HF_TOKEN \
  -v sam3-cache:/workspace/.cache/npa/sam3 \
  -v "$PWD/inputs:/inputs:ro" -v "$PWD/outputs:/outputs" \
  "$SAM3_IMAGE" segment --input-path /inputs/video.mp4 \
  --prompt 'robot arm' --output-path /outputs/sam31-run
```

The output directory must not already exist and its parent must be writable by
UID 1000. Choose a video with at least two frames and a visible prompted object.
The runner preserves source frame count and average frame rate, saves object masks per frame,
and writes an H.264 `overlay.mp4`, `frames.json` and `result.json`. Audio is omitted.
It fails if frame counts disagree, decoding fails or fewer than two frames have
nonempty masks. Human review is still required for segmentation quality.

`sam3-runtime ensure` prepares the cache; `sam3-runtime exec <python arguments>`
runs a Python command in it. `NPA_SAM3_CACHE` optionally changes the default cache
directory `/workspace/.cache/npa/sam3`. Cache installs are locked and published
atomically, and the checkpoint hash is checked before each use. The inference
child receives neither the HF token nor network-enabled Hugging Face loading.
Health, help and missing-token refusal do not populate the model cache.

## Reproducibility and qualification

`pins.json` records the exact upstream source revision, model revision and
checkpoint SHA-256. `requirements.lock` pins the Linux Python 3.12 runtime
closure with hashes; CUDA-enabled PyTorch is installed only at runtime. The
runner uses the upstream `build_sam3_multiplex_video_predictor` API, without
FlashAttention 3 or compilation, and does not download SAM 3.0 as a fallback.
Source and output hashes plus model, dependency and GPU metadata are recorded.
The runtime uses PyTorch 2.13 and setuptools 84 with security fixes. A checked
compatibility patch replaces the removed `pkg_resources` tokenizer lookup with
`importlib.resources`; original and patched file hashes are recorded in the
pins and output provenance. No model computation is changed by that patch.

Build through the repository's trusted `publish-public-images.yml` workflow.
The canonical Dockerfile uses context `npa/`, and immutable development tags
are `dev-<full committed SHA>`. Both pre-push and pushed-byte scans must pass
`npa/scripts/scan_image_sam3_payload.py`, in addition to the standard image gates.
Public development availability does not qualify a supported release: run real
text-prompted mask propagation on that exact public digest first.

See [redistribution classification](REDISTRIBUTION.md) and
[container contribution guidance](../../../../docs/workbench/contributing-a-containerized-solution.md).
