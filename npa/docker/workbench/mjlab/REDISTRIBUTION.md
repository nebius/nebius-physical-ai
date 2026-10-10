# MJLab image inventory

This image is **publication-quarantined**. The current CUDA 13.0 operator build
has executed G1 training, checkpoint resume and rendered evaluation on RTX PRO
6000. Historical B200 and RTX acceptance uses a separate CUDA 12.8 build; exact
artifacts and measured scope are recorded in `docs/workbench/mjlab.md`. GPU
execution does not establish redistribution or security approval for the built
dependency closure.

MJLab 1.6.0 and MuJoCo are Apache-2.0; MuJoCo Warp and Warp carry their own
upstream notices. The robot assets installed by MJLab retain the package's
per-asset licenses. PyTorch and its NVIDIA CUDA/cuDNN dependency wheels are
separate runtime license boundaries. The Python base is pinned by digest.

The current recipe pins Debian package resolution to the signed 2026-10-10
snapshot. Python distributions retain their installed license files and Debian
packages retain `/usr/share/doc/*/copyright`. The installed cuDNN 9.20.0.48
wheel uses the same reviewed runtime-only filter as cuRobo before the dependency
layer commits: SDK headers/static files are removed, shared runtime libraries
and the full wheel license remain, and retained hashes are recorded in
`/usr/share/doc/npa-mjlab/cudnn-runtime.json`.

The image retains hash-verified notices omitted by the mcap 1.4.0 and NVSHMEM
3.4.5 wheel distributions. These are the same exact notices already used by
the repository's Open3D and cuRobo recipes. System FFmpeg replaces
imageio-ffmpeg's bundled executable in the dependency install layer. These
controls narrow the shipped payload; publication still requires the actual
built-layer audit and corresponding-source review.

Recipients receive source for every original and installed Debian package under
`/usr/share/doc/npa-mjlab/ubuntu-sources`, with exact descriptor checksums in
`bootstrap-sources.json`. The directory names follow the shared bootstrap source
delivery helper; these are Debian sources fetched through signed snapshot APT
metadata. The older source-only snapshot covers superseded parent-image packages.
The source verifier checks the complete saved-image ancestry against that bundle,
including packages upgraded later.

PyAV 17.1.0 bundles a separate GPL-enabled FFmpeg 8.1.1 build. The image also
delivers its exact upstream build recipe, all locked native component source
archives, PyAV source and CPython 3.12.14 source under `python-sources`, retaining
the archives' original license texts alongside them. The PyAV release pins
pyav-ffmpeg revision `b84838a003b1626152ef530750d5c3e0022c3db9`; source URLs,
sizes and hashes are checked in `python-sources.lock.json`. These files give
recipients the corresponding source rather than a general upstream homepage.
The original dynamic libraries remain replaceable; the container adds no
restriction on modification or reverse engineering for debugging changes.

The image contains no trained weights, motion datasets, operator credentials,
accepted-terms files, or populated model/kernel caches. Training produces native
RSL-RL checkpoints; an operator supplies any tracking motion NPZ at runtime.
Kernel caches live under `/workspace/.cache` and are ephemeral unless mounted.

Before public publication, inspect every built layer and exact dependency wheel,
retain all required third-party notices and corresponding-source obligations,
resolve CUDA/cuDNN redistribution requirements for the actual shipped members,
and pass the repository's security, payload, bootstrap and real GPU capability
gates. No public-release redistribution approval is claimed here.

Source references:

- https://github.com/mujocolab/mjlab/tree/v1.6.0
- https://github.com/google-deepmind/mujoco/blob/main/LICENSE
- https://github.com/google-deepmind/mujoco_warp/blob/main/LICENSE
- https://github.com/NVIDIA/warp/blob/main/LICENSE.md

Regenerate the Linux Python 3.12 lock with `uv pip compile npa/pyproject.toml
npa/docker/workbench/mjlab/build-requirements.in --extra mjlab --python-version 3.12 --python-platform x86_64-manylinux_2_28
--extra-index-url https://download.pytorch.org/whl/cu130
--index-strategy unsafe-best-match --constraint npa/docker/workbench/mjlab/constraints.txt
--generate-hashes --output-file npa/docker/workbench/mjlab/requirements.lock`.
The constraint file pins `torch==2.13.0+cu130` and requires `setuptools>=83.0.0`.
These replace the older CUDA 12.8 qualification image's vulnerable Torch and
setuptools versions; historical GPU records remain bound to their original bytes.

The locked libpng 1.6.58 gzip archive uses the Debian HTTPS mirror to avoid
regional SourceForge redirects. All 669 archive members (types, link targets,
and file hashes) were compared with the upstream xz archive and match.
The gzip archive has its own reviewed size and SHA-256 in the source lock.
