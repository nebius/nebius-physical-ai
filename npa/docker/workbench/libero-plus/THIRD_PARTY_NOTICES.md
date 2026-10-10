# LIBERO-Plus source-admission image notices

This private, unvalidated qualification image is intentionally source-free. It
does **not** contain the `sylvestf/LIBERO-plus` repository, the
`Sylvest/LIBERO-plus` Hugging Face asset, simulator/model/runtime packages,
weights, task assets, textures, demonstration data, rendered media, or a
populated cache. It must not be represented as a runnable LIBERO-Plus benchmark
image.

The image contains NPA's Apache-2.0 workflow adapter and the following upstream
metadata for attribution only:

- LIBERO-Plus authors: Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai,
  Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing
  Gong, and Xipeng Qiu. Citation: *LIBERO-Plus: In-depth Robustness Analysis of
  Vision-Language-Action Models*, arXiv:2510.13626 (2025).
- The referenced source revision is
  `4976dc30028e805ff8094b55501d532c48fec182`. It has no reviewed `LICENSE`,
  `NOTICE`, or `COPYING` declaration and is not copied into this image.
- The separately referenced asset revision is
  `dd2bd61b7d9a6fef1abc52d606e983b41886a149`; its card declares MIT, but the
  asset archive is not copied into this image. That declaration does not grant
  permission for the separate source tree.

The base is Docker Official Images `python:3.10-slim-bookworm` for linux/amd64,
manifest `sha256:999137905e8718de681744822ccd965e1950e1baba089035060418e05e1d7496`.
CPython is under the Python Software Foundation License. Debian packages are
pinned by the copied `libero/debian-packages.lock`; their corresponding-source
locations and package copyright notices remain authoritative. The image omits
NVIDIA, CUDA, MuJoCo, PyTorch, robomimic, and LIBERO-Plus bytes.
