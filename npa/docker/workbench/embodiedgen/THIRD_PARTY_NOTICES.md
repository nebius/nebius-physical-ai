# Third-party notices

- EmbodiedGen V2.1.0, commit `f0124197888c2b733e4eaa65acd81ad9cfda3b79` — Apache-2.0, runtime fetched.
- TRELLIS, commit `55a8e8164b195bbf927e0978f00e76c835e6011f` — MIT, runtime fetched.
- `microsoft/TRELLIS-image-large`, revision `25e0d31ffbebe4b5a97464dd851910efc3002d96` — MIT, runtime fetched.
- `openbmb/MiniCPM-V-4_5` — Apache-2.0, hosted runtime request only.
- NVIDIA CUDA 12.8 development base — NVIDIA CUDA Toolkit EULA; retained only in the operator-private image.
- Validation runtime closure — `numpy==1.26.4` (BSD-3-Clause),
  `scipy==1.14.1` (BSD-3-Clause; bundled wheel notices apply),
  `Pillow==11.3.0` (MIT-CMU), `trimesh==4.11.1` (MIT),
  `plyfile==1.0.3` (GPL-3.0-or-later), `tifffile==2024.8.30` (BSD),
  `contourpy==1.3.0` (BSD-3-Clause), `imageio==2.37.4` (BSD-2-Clause),
  `imageio-ffmpeg==0.6.0` (BSD-2-Clause wrapper), and `pybullet==3.2.7`
  (zlib), all runtime fetched. The pinned ImageIO FFmpeg wheel may carry its
  platform executable; it remains in the operator runtime cache rather than the
  image, and its applicable binary terms still govern any redistribution.

The pinned EmbodiedGen `requirements.txt` hash, validation package list, and
full runtime dependency closure are recorded with each run's receipt.
