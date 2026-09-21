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
- Baked runtime: digest-pinned NVIDIA CUDA 13 runtime base **without cuDNN**,
  PyTorch 2.13.0 CUDA 13 wheels, NVIDIA cuda-core/runtime, Warp, Pinocchio, Rerun and
  NPA. CUDA's Linux-specific supplement (section 2.3) permits redistribution of
  Linux components with unmodified object code; Attachment A also enumerates
  runtime/JIT libraries and runtime compilation headers. NVIDIA drivers are
  host-injected. CUDA and container license notices remain intact.
  The minimal `base` image supplies cudart; pinned cuda.core/NVRTC, CUDA header
  wheels and Warp provide runtime compilation. The upstream CUDA-core backend
  discovers cudart/NVRTC headers through cuda-pathfinder. The image does not
  install the optional pybind backend or need the full CUDA development image,
  its unused Nsight profiler, or operating-system development headers. Native
  GPU execution remains a required qualification gate.
- OpenMP runtime: Pinocchio requires `libgomp.so.1`, so the image installs
  Ubuntu Noble `libgomp1=14.2.0-4ubuntu2~24.04.1` from the same immutable
  `20260920T000000Z` snapshot as the rest of the distro closure. The exact
  `libgomp1` deb is 148062 bytes with SHA256
  `e8a95ec58125b4933597f30ff56c2ae10edf90f287262e366d4b6edea3019144`;
  its 352304-byte `libgomp.so.1.0.0` has SHA256
  `135f3c8f006d2fe5e68e51281c7974cb991a03de3bfb3593d68d174dfcf854d1`.
  The final-image verifier also requires `libgomp.so.1` to remain an exact
  symbolic link to `libgomp.so.1.0.0`.
  Its exact `gcc-14-base` dependency is 51014 bytes with SHA256
  `b95c172411a7fdae70307cf33a9f5320ba5e056b556454543dd5b679d5ce1c4f`.
  The matching `gcc-14-base` copyright file is retained at
  `/usr/share/doc/gcc-14-base/copyright` (69004 bytes, SHA256
  `20390f8a6f3b1e4d7cb45dd8652dabb259bbef688cbad839bcdb0b9ba7252f79`)
  together with `/usr/share/common-licenses/GPL-3` (35149 bytes, SHA256
  `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986`).
  libgomp is `GPL-3.0-or-later WITH GCC-exception-3.1`. Matching corresponding
  source remains available from that immutable snapshot as
  [the upstream archive](https://snapshot.ubuntu.com/ubuntu/20260920T000000Z/pool/main/g/gcc-14/gcc-14_14.2.0.orig.tar.gz)
  (SHA256 `768c314c11eeab56ccebb91eb42ec4a41122fa94f0d83400126401942622197b`),
  [Ubuntu packaging](https://snapshot.ubuntu.com/ubuntu/20260920T000000Z/pool/main/g/gcc-14/gcc-14_14.2.0-4ubuntu2~24.04.1.debian.tar.xz)
  (SHA256 `cfece214c2fb790ef5f3baffb9a53e40618e7ae12d053610b251e94d77d08ade`)
  and [the source descriptor](https://snapshot.ubuntu.com/ubuntu/20260920T000000Z/pool/main/g/gcc-14/gcc-14_14.2.0-4ubuntu2~24.04.1.dsc)
  (SHA256 `50950080874a6ec6780dd60c243e21d9cda9d736bb32bca98d16095d27cc01b5`).
  Public release remains blocked if those source or license bytes cannot be
  delivered. `runtime-payload.json` and the complete-layer verifier bind these
  identities; the image build additionally imports the real upstream benchmark
  through Pinocchio and checks all 800 MotionBenchMaker and 1800 MPiNets rows.
- Reproducible Python bytecode: the build helper and trusted workflow pass the
  exact source commit epoch as a build-only `SOURCE_DATE_EPOCH` argument. The
  pinned CPython compiler then uses PEP 552 checked-hash bytecode for newly
  installed packages. This removes installation-time header variation while
  preserving the complete compiled code payload; it does not modify inherited
  image layers or change runtime authentication. Verify actual built-layer
  headers and full bytecode hashes against the exact source and compiler before
  accepting image findings. The argument is not retained in runtime environment.
- Dependency source correction: scikit-image 0.26.0 includes a historical
  download recipe containing bearer material whose continued usability cannot
  be established. `remove_scikit_image_recipe.py` checks the exact installed
  version and source hash, removes only that second, inert string expression
  from `grass()`, and verifies the resulting source hash. The executable module
  AST, primary documentation and image-loading call remain identical; original
  notices are retained. The same installation RUN removes the affected old
  bytecode, regenerates it from the corrected source and updates the wheel's
  `RECORD`. `/usr/share/doc/npa-curobo/dependency-source-correction.json` records
  the source, resulting source, bytecode and before/after metadata hashes.
  A later-layer deletion would leave the original bytes distributed, so this
  correction must finish before the installation layer commits. It is not a
  scanner-rule exception, and the resulting image still requires all scans.
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
  Generated PyTorch ATen cuDNN operator headers are BSD-licensed and permitted
  only by their exact pinned wheel paths/content hashes, with the complete wheel
  notice retained; this grants no general CUDA/cuDNN SDK exemption.
  The patched PyTorch wheel splits its upstream BSD license from 97 bundled
  third-party notice files. The inventory binds every separate notice by its
  exact path, size and SHA256; the verifier requires all of them, including
  PyTorch's project license, in the final image. The 52 reviewed generated
  operator headers and cuDNN 9.20.0.48 runtime/header inventories were refreshed
  from the complete hash-verified upstream wheels.
  The trusted publication workflow runs this check before push and again on the
  pulled immutable digest, alongside the generic payload and security gates.
  The complete Python closure is version/hash locked. This record is an artifact
  classification, not evidence that an image has been built or scanned.
- Other NVIDIA dependencies: cuSPARSELt 0.8.1 retains one header and its shared
  library under its product supplement, which permits `.h` and `.so` files as
  application components. NCCL 2.29.7 retains its header, library and full
  BSD-3-Clause license, including the NVTX notice reference. NVSHMEM 3.4.5
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
[NCCL release license](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/LICENSE.txt),
[NVSHMEM release manifest](https://developer.download.nvidia.com/compute/nvshmem/redist/redistrib_3.4.5.json),
[NVSHMEM product license](https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/LICENSE.txt),
[container license](https://gitlab.com/nvidia/container-images/cuda/-/blob/master/NGC-DL-CONTAINER-LICENSE).

Metadata sources: [pinned upstream package metadata](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/pyproject.toml),
[PyPA classifier specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/#classifiers),
[PyPI classifier registry](https://pypi.org/classifiers/).

The cuRobo source install makes only the documented package metadata correction;
its runtime code, robot assets and dataset bytes are unchanged. The separate
scikit-image correction removes an inert historical example while preserving
its executable AST. NPA invokes
upstream's benchmark configuration loader and MotionPlanner, and records its own factual metrics;
it does not reproduce upstream's placeholder end-effector path statistics or
convert inverse-dynamics failures to zero energy.
