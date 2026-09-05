# cuRobo V2 packaging record

This candidate is eligible for public redistribution but remains in publication
quarantine until exact-image scans and actual GPU capability validation pass.
No published image or measured performance is claimed by this record.

- Source: NVIDIA cuRobo V2, revision
  `8e734f3ced1df898990bcd92de40abce475907db`, Apache-2.0. The image retains its
  `LICENSE`, `LICENSE_ASSETS` and all source/asset attributions under `/opt/curobo`.
  V1's research license does not apply to this separately pinned V2 artifact.
- Packaging correction: that release's `project.classifiers[5]` names the
  unregistered `Topic :: Scientific/Engineering :: Robotics` classifier. The
  pinned setuptools/Trove validator rejects it. `correct_package_metadata.py`
  verifies the source revision and original `pyproject.toml` SHA256, replaces
  that one classifier with the registered `Topic :: Scientific/Engineering`
  parent, and verifies the complete corrected metadata SHA256. It adds a
  prominent changed-file comment while preserving upstream copyright/license
  notices. `/usr/share/doc/npa-curobo/metadata-correction.json` records the pinned
  archive hash and metadata hashes before/after the correction. The normal
  validator and all dependency pins remain enabled and unchanged.
- Robot assets: Franka assets are Apache-2.0; included UR/Unitree assets carry
  BSD-3-Clause terms described by upstream `LICENSE_ASSETS`. No asset notice is
  removed during packaging.
- Benchmark data: robometrics revision
  `81e3d1d605de84100d8ab880b43096aba221a48b` is MIT; its `Licenses` explicitly
  records Motion Policy Networks under MIT and MotionBenchMaker under
  BSD-3-Clause. Both files remain under `/opt/robometrics`. The raw dataset
  loaders run directly from that verified source tree with
  NumPy/PyYAML; no robometrics distribution or optional evaluator is installed.
  This avoids falsely claiming its older NumPy<2 package constraint is compatible
  with the current Pinocchio NumPy 2 runtime.
- Baked runtime: digest-pinned NVIDIA CUDA 13 development base **without cuDNN**,
  PyTorch CUDA 13 wheels, NVIDIA cuda-core/runtime, Warp, Pinocchio, Rerun and
  NPA. CUDA's Linux-specific supplement (section 2.3) permits redistribution of
  Linux components with unmodified object code; Attachment A also enumerates
  runtime/JIT libraries and runtime compilation headers. NVIDIA drivers are
  host-injected. CUDA and container license notices remain intact.
- cuDNN: the current official supplement permits only runtime `.so`/`.dll`
  distribution. The locked wheel's bundled older supplement additionally lists
  `.h`; packaging deliberately uses the narrower runtime boundary that satisfies
  both, without depending on which version governs header redistribution.
  The locked cuDNN wheel contains fourteen SDK headers as well as shared libraries.
  `filter_cudnn_runtime.py` removes headers/static archives in the **same RUN** as
  pip installation, before any final-image layer exists. It rejects unexpected
  or unrecorded payloads, preserves unmodified ELF runtime libraries and license
  metadata, and writes their hashes to `/usr/share/doc/npa-curobo/cudnn-runtime.json`.
  The image has no cuDNN development base ancestor; deleting those files in a
  later layer would leave restricted bytes in the image and is insufficient.
  `runtime-payload.json` records the independently SHA-verified source wheel,
  all fourteen excluded header paths, eight retained runtime hashes and both
  required license hashes. `verify_image.py` binds a Docker save to the inspected
  image ID and every config diff ID, scans all historical layers without a
  member-size cap, and accounts for final whiteouts and type changes. It rejects
  relocated known SDK/runtime bytes and requires the exact retained payloads.
  The trusted publication workflow runs this check before push and again on the
  pulled immutable digest, alongside the generic payload and security gates.
  The complete Python closure is version/hash locked. This record is an artifact
  classification, not evidence that an image has been built or scanned.
- Other NVIDIA dependencies: cuSPARSELt 0.8.0 retains one header and its shared
  library under its product supplement, which permits `.h` and `.so` files as
  application components. NCCL 2.27.7 retains its header, library and full
  BSD-3-Clause license, including the NVTX notice reference. NVSHMEM 3.3.24
  includes 45 headers, a device archive, device bitcode and twelve shared
  libraries; all 59 payloads match its official CUDA 13 Linux distribution.
  Its product-specific supplement permits any SDK portion subject to the
  application distribution requirements. The build adds that distribution's
  full license and third-party notices at
  `/usr/share/doc/npa-curobo/NVSHMEM-LICENSE.txt`, verified against SHA256
  `43a87c0ff94ce3196011ff75e17fbee96933c9e1d511557659ece8a326f95e8f`,
  alongside the wheel's bundled CUDA license. Wheel metadata alone is not a
  license grant. Built-image inventory and notice hashes must verify these
  boundaries before publication.
- Weights: none. No model checkpoint, HF/NGC credential or gated data is needed.
- Caches: CUDA/Warp runtime compilation and working artifacts are node-local,
  ephemeral `/workspace` state unless the operator explicitly mounts storage.
  No populated runtime cache or operator agreement acceptance is baked.
- Outputs: operator planning requests remain private. Benchmark trajectories and
  metrics preserve source/dataset provenance. No additional model-output license
  applies; this is numerical motion planning, not a generative model service.

Official sources: [cuRobo license](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/LICENSE),
[robot assets](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/LICENSE_ASSETS),
[benchmark licenses](https://github.com/fishbotics/robometrics/blob/81e3d1d605de84100d8ab880b43096aba221a48b/Licenses),
[CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html),
[cuDNN EULA and supplement](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html),
[cuSPARSELt SLA](https://docs.nvidia.com/cuda/cusparselt/license.html),
[NCCL release license](https://github.com/NVIDIA/nccl/blob/v2.27.7-1/LICENSE.txt),
[NVSHMEM release manifest](https://developer.download.nvidia.com/compute/nvshmem/redist/redistrib_3.3.24.json),
[NVSHMEM product license](https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/LICENSE.txt),
[container license](https://gitlab.com/nvidia/container-images/cuda/-/blob/master/NGC-DL-CONTAINER-LICENSE).

Metadata sources: [pinned upstream package metadata](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/pyproject.toml),
[PyPA classifier specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/#classifiers),
[PyPI classifier registry](https://pypi.org/classifiers/).

The source install makes only the documented package metadata correction;
vendor runtime code, robot assets and dataset bytes are unchanged. NPA invokes
upstream's benchmark configuration loader and MotionPlanner, and records its own factual metrics;
it does not reproduce upstream's placeholder end-effector path statistics or
convert inverse-dynamics failures to zero energy.
