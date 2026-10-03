# SeedVR2 image third-party notices

The built image retains package-level license files. This inventory records the
principal runtime components and immutable source identities.

| Component | Identity | License |
| --- | --- | --- |
| ByteDance SeedVR / SeedVR2 inference source | `ByteDance-Seed/SeedVR@e4de8c24441a67e1b7df56abea10645059bb1185` | Apache-2.0; `/opt/seedvr2/LICENSE` |
| NVIDIA Apex | `NVIDIA/apex@8a6508aaad6e75a2b939e33f308cd63d745d97f1` | BSD-3-Clause; installed distribution license |
| FlashAttention | `flash-attn==2.8.3.post1` | BSD-3-Clause; installed distribution license |
| PyTorch | `torch==2.13.0` | BSD-3-Clause; installed distribution license |
| TorchVision | `torchvision==0.28.0` | BSD-3-Clause; installed distribution license |
| PyAV | `av==16.0.1` | BSD-3-Clause; installed distribution license |
| Hugging Face Diffusers | `diffusers==0.38.0` | Apache-2.0; installed distribution license |
| Hugging Face Safetensors | `safetensors==0.8.0` | Apache-2.0; installed distribution license |
| NVIDIA NVSHMEM runtime | `nvidia-nvshmem-cu13==3.4.5`; product notice `NVIDIA/nvshmem@v3.4.5-0` | `LicenseRef-NVIDIA-Proprietary` plus bundled third-party terms; installed wheel license and `/usr/share/doc/npa-seedvr2/NVSHMEM-License-v3.4.5-0.txt` |
| NVIDIA CUDA base/runtime | CUDA 13.0.2 cuDNN-runtime final base plus the hash-locked PyTorch CUDA closure | NVIDIA CUDA Toolkit End User License Agreement and component notices shipped in the base/wheels |

The CUDA/cuDNN devel base is a build stage only and is absent from the final
image manifest. The cuDNN wheel inventory is validated before final-stage copy;
SDK headers and static archives are omitted while shared runtime libraries and
the license notice remain hash-recorded in
`/usr/share/doc/npa-seedvr2/cudnn-runtime.json`.
The PyTorch closure also installs the NVSHMEM wheel. Its SDK headers, device
static archive, and device bitcode are omitted; only reviewed shared runtime
objects remain. The wheel's license plus the exact upstream v3.4.5-0 product
license (SHA-256
`1f5b7ada702926bc73327e6eb02dc2d41facc844cc4512ac900451bda06a459e`)
are retained. The latter carries the DF-NVSHMEM, Sandia OpenSHMEM, Argonne,
RDMA Core, and libfabric notices missing from the wheel license. Their
identities are recorded in
`/usr/share/doc/npa-seedvr2/nvshmem-runtime.json`.

The pinned upstream inference entrypoint imports TorchVision's deprecated
`read_video` API. Public TorchVision 0.28.0 wheels delegate that module to an
unpublished internal package, so the image replaces that one exact import with
an NPA-authored PyAV adapter. The patch refuses any upstream source identity
other than the reviewed entrypoint SHA-256 and records the original, patched,
and adapter hashes in
`/usr/share/doc/npa-seedvr2/video-io-compat.json`. The adapter only performs the
full-file RGB input decode used by SeedVR2. It changes that decoder boundary but
does not modify model configuration, weights, model execution, or output
post-processing; unusual media can therefore still produce decoder-dependent
inputs and generated output.

SeedVR2-3B model payloads are not image members. Runtime fetch accepts only
Hugging Face revision `37255ff8cccfb01071b87f635a5948ca8d53117c` and verifies:

- `seedvr2_ema_3b.pth`:
  `6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b`
- `ema_vae.pth`:
  `c7df8a67e68b7f9aca3d5d2153d2ce8ab4373687741a0f9ce87cb356ace51cac`
- `pos_emb.pt`:
  `fa07a14844314772266b66c3b95deb0027696d8fe7065721263db5176f45d799`
- `neg_emb.pt`:
  `6a43e5800ef2354f1c156d27535834da055cbec8248298b8923492bba2076581`

The model repository declares Apache-2.0 and is public, non-gated. No model
payload, sensor clip, generated output, token, object-storage credential, or
populated cache is included in the image.

## MCAP Python 1.4.0

The service environment includes `mcap==1.4.0`, copyright Foxglove Technologies
Inc, under the MIT license. The exact upstream grant is delivered as
`/usr/share/doc/npa-seedvr2/MCAP-1.4.0-LICENSE.txt` (SHA-256
`da11235665c17d4c1634072dae92b8ba1b38d6fdde2ccf19a6bbede33253f58d`).
Source: [the MCAP 1.4.0 release license](https://github.com/foxglove/mcap/blob/b33fa682a5c517b1d213faeabd118e0b4f9d9d93/LICENSE).
The upstream wheel's metadata names MIT but does not deliver this notice.
