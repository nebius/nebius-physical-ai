# OpenArm image third-party inventory

| Component | Shipped | License / packaging decision |
| --- | --- | --- |
| Enactic openarm_mujoco 2.2.0 (`a8c979…`) | yes | Apache-2.0 source, MJCF, meshes, and package; upstream license retained. |
| Enactic openarm_isaac_lab (`bad82e…`, extension 0.1.0) | yes | Apache-2.0 source and OpenArm USD assets; upstream license retained. |
| MuJoCo 3.6.0 | yes | Apache-2.0 binary wheel; exact hashes in `mujoco-requirements.txt`. |
| absl-py 2.5.0 / etils 1.14.0 | yes | Apache-2.0. |
| glfw 2.10.2 | yes | zlib/libpng-style license. |
| PyOpenGL 3.1.10 | yes | BSD-3-Clause. |
| NumPy 1.26.4 / fsspec 2026.4.0 / typing-extensions 4.15.0 / zipp 4.1.0 | yes | BSD-3-Clause, BSD-3-Clause, PSF-2.0, and MIT respectively. |
| NVIDIA CUDA base | yes | Official redistributable CUDA container; base is pinned by digest. |
| Isaac Sim 5.1.0.0 and Isaac Lab 2.3.2.post1 | no | NVIDIA proprietary wheels, hash-pinned and operator-fetched at runtime only after the shared acceptance/refusal preflight. |
| Isaac Lab source (`37ddf6…`) | no image layer; runtime cache | BSD-3-Clause checkout fetched by the shared bootstrap to provide official scripts. |
| Credentials, caches, customer data, checkpoints, and outputs | no | Runtime-only operator material. |

The shared Isaac OSS closure and Ubuntu package copyright files remain the
complete runtime inventory. `components.json` records source and bake boundaries;
the built image generates `python-license-inventory.json` from the installed
distribution metadata. The release SBOM is authoritative for all transitive
bytes in the published digest.
