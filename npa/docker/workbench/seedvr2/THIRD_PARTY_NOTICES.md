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
| Hugging Face Diffusers | `diffusers==0.38.0` | Apache-2.0; installed distribution license |
| Hugging Face Safetensors | `safetensors==0.8.0` | Apache-2.0; installed distribution license |
| NVIDIA CUDA base/runtime | CUDA 13.0.2 base plus the hash-locked PyTorch CUDA closure | NVIDIA CUDA Toolkit End User License Agreement and component notices shipped in the base/wheels |

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
