# npa-ltx2 third-party notices

## Baked into the image

| Component | Version | License |
| --- | --- | --- |
| `python:3.12-slim-bookworm` (digest-pinned) | 3.12 | PSF-2.0 + Debian base (OSS) |
| `uv` | 0.9.8 | Apache-2.0 OR MIT |
| `huggingface_hub[cli]` | 0.36.0 | Apache-2.0 |
| ffmpeg, git, libgl1, libglib2.0, openssh-server, rsync, sudo, util-linux | Debian bookworm | Debian OSS (LGPL/GPL/BSD/MIT as packaged) |
| `video_check.py`, `validate_video.py`, `ltx_runtime.sh`, `entrypoint.sh`, `smoke.sh` | this repo | Apache-2.0 |

## Fetched at run time, never baked

These are delivered by their vendors to the operator, under the operator's own
entitlement. They are listed for transparency; their presence in a running
container is the operator's licensed copy, not something this image redistributes.

| Component | Source | License | Gate |
| --- | --- | --- | --- |
| `ltx-core`, `ltx-pipelines` (+ `natten` extra) | `github.com/Lightricks/LTX-2` @ `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` | LTX-2.x Community License Agreement, 2026-08-11 — **not OSI** | operator's own `HF_TOKEN`: Section 1.9 makes the source licensed material too |
| LTX-2.5 weights (DiT, Gemma 4 12B LTX text encoder, video/audio VAEs, spatial upscaler) | `huggingface.co/Lightricks/LTX-2.5` (**gated**) | LTX-2.x Community License Agreement, 2026-08-11 | operator's own `HF_TOKEN` with gated-repo scope |
| `torch`, `torchaudio` (CUDA 13.2 build) | `download.pytorch.org/whl/cu132` | BSD-3-Clause + NVIDIA CUDA EULA for the bundled NVIDIA components | `NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS` |
| `natten` | `whl.natten.org` | MIT | resolved with the above |
| remaining Python closure (transformers, accelerate, safetensors, scipy, av, …) | PyPI, resolved by upstream's own pins | assorted OSS | resolved with the above |

The text encoder is a Gemma 4 12B checkpoint fine-tuned by Lightricks and
redistributed inside the gated LTX-2.5 repository. Google's Gemma terms may apply
to it in addition to the LTX agreement; the operator accepts whatever the gated
repository presents at download time. Stock Gemma 4 is not a substitute — loading
checks the encoder version against `gemma4-12b-ltx-v1`.

## Attribution

LTX-2 and LTX-2.5 are products of Lightricks Ltd. This image is not endorsed by
or affiliated with Lightricks, and Section 8 of the agreement grants no trademark
rights; "LTX-2.5" is used here only to identify the upstream software the
operator chooses to fetch.
