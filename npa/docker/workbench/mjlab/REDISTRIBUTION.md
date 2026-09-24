# MJLab image inventory

This image is **unbuilt and publication-quarantined**. The Dockerfile and
hash-locked dependency inventory are build inputs, not built-byte validation.

MJLab 1.6.0 and MuJoCo are Apache-2.0; MuJoCo Warp and Warp carry their own
upstream notices. The robot assets installed by MJLab retain the package's
per-asset licenses. PyTorch and its NVIDIA CUDA/cuDNN dependency wheels are
separate runtime license boundaries. The Python base is pinned by digest.

The image contains no trained weights, motion datasets, operator credentials,
accepted-terms files, or populated model/kernel caches. Training produces native
RSL-RL checkpoints; an operator supplies any tracking motion NPZ at runtime.
Kernel caches live under `/workspace/.cache` and are ephemeral unless mounted.

Before public publication, inspect every built layer and exact dependency wheel,
retain all required third-party notices and corresponding-source obligations,
resolve CUDA/cuDNN redistribution requirements for the actual shipped members,
and pass the repository's security, payload, bootstrap and real GPU capability
gates. No built-image redistribution or hardware qualification is claimed here.

Source references:

- https://github.com/mujocolab/mjlab/tree/v1.6.0
- https://github.com/google-deepmind/mujoco/blob/main/LICENSE
- https://github.com/google-deepmind/mujoco_warp/blob/main/LICENSE
- https://github.com/NVIDIA/warp/blob/main/LICENSE.md

Regenerate the Linux Python 3.12 lock with `uv pip compile npa/pyproject.toml
npa/docker/workbench/mjlab/build-requirements.in --extra mjlab --python-version 3.12 --python-platform x86_64-manylinux_2_28
--extra-index-url https://download.pytorch.org/whl/cu128
--index-strategy unsafe-best-match --constraint npa/docker/workbench/mjlab/constraints.txt
--generate-hashes --output-file npa/docker/workbench/mjlab/requirements.lock`.
The constraint file pins `torch==2.11.0+cu128`.
