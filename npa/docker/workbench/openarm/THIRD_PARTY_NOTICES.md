# OpenArm image third-party inventory

| Component | Shipped | License / packaging decision |
| --- | --- | --- |
| Enactic openarm_mujoco 2.2.0 (`a8c979…`) | yes | Apache-2.0 source, MJCF, meshes, and package; upstream license retained. |
| Enactic openarm_isaac_lab (`bad82e…`, extension 0.1.0) | yes | Apache-2.0 source and OpenArm USD assets; upstream license retained. |
| MuJoCo 3.6.0 | yes | Apache-2.0 binary wheel; exact hashes in `mujoco-requirements.txt`. |
| FastAPI 0.136.1 / Starlette 1.6.0 / Pydantic 2.13.5 / Uvicorn 0.53.0 | yes | MIT-licensed API, ASGI toolkit, schema, and server runtime; exact hashes and transitive dependencies are recorded in `mujoco-requirements.txt`. OpenArm overrides the shared Isaac 2 closure's vulnerable Starlette release. |
| GitPython 3.1.62 / gitdb 4.0.12 / smmap 5.0.3 | yes | BSD-3-Clause helper and dependencies. OpenArm's exact lock upgrades the shared closure's vulnerable GitPython 3.1.57 before publication. |
| absl-py 2.5.0 / etils 1.14.0 | yes | Apache-2.0. |
| glfw 2.10.2 | yes | zlib/libpng-style license. |
| PyOpenGL 3.1.10 | yes | BSD-3-Clause. |
| NumPy 1.26.4 / fsspec 2026.4.0 / typing-extensions 4.15.0 / zipp 4.1.0 | yes | BSD-3-Clause, BSD-3-Clause, PSF-2.0, and MIT respectively. |
| NVIDIA CUDA/cuDNN development base | yes | Proprietary redistributable container, pinned by digest. Redistribution is conditioned by the NVIDIA Deep Learning Container License and applicable component notices. |
| CUDA 12 runtime wheels required by PyTorch | yes | Exact-version NVIDIA CUDA components. The CUDA Toolkit EULA permits only the listed distributable portions and imposes application, access, protective-terms, and notice conditions. Separately packaged headers/static archives are removed. |
| cuDNN 9.10.2.21 runtime libraries | yes | NVIDIA proprietary; only runtime `.so`/`.dll` portions are distributable under the cuDNN supplement. Wheel headers are removed and its license is retained. |
| NCCL 2.27.5, NVSHMEM 3.3.20, and NVTX 12.8.90 | yes | Exact package metadata and retained license files identify BSD-3-Clause or Apache-2.0 code together with any NVIDIA package terms. |
| Isaac Sim 5.1.0.0 and Isaac Lab 2.3.2.post1 | no | NVIDIA proprietary wheels, hash-pinned and operator-fetched at runtime only after the shared acceptance/refusal preflight. |
| Isaac Lab source (`37ddf6…`) | no image layer; runtime cache | BSD-3-Clause checkout fetched by the shared bootstrap to provide official scripts. |
| Credentials, caches, customer data, checkpoints, and outputs | no | Runtime-only operator material. |

The shared Isaac prerequisite closure contains both open-source packages and the
NVIDIA runtime components listed above. Ubuntu package copyright files remain in
the image. `components.json` records source and bake boundaries; the built image
generates `python-license-inventory.json` with the installed distribution's exact
license signals and retained license-file hashes. The release SBOM is
authoritative for all transitive bytes in the published digest.

Official governing sources reviewed for this packaging decision:

- <https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license>
- <https://docs.nvidia.com/cuda/archive/12.8.1/eula/index.html>
- <https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html>
- <https://github.com/NVIDIA/nccl/blob/v2.27.5-1/LICENSE.txt>
- <https://pypi.org/project/nvidia-nvshmem-cu12/3.3.20/>
