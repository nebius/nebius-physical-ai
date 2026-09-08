# OpenPI image redistribution

This image contains pinned Apache-2.0 OpenPI source, its frozen Python
dependencies with documented compatibility/security overlays, and NPA's policy
inference, training and evaluation adapters. NVIDIA dependencies retain their
own terms; they are not relicensed as Apache-2.0.

The final CUDA 12.8.1 runtime base does not contain cuDNN or inherit the compiler
stage. The build stage contributes NPA's SM100/SM120 device-kernel probes and
unmodified Linux `cuobjdump`, covered by the
[CUDA 12.8.1 supplement, section 2.3](https://docs.nvidia.com/cuda/archive/12.8.1/eula/index.html).
The CUDA runtime and its upstream notices remain in the image.

The installed `nvidia-cudnn-cu12==9.10.2.21` wheel has fourteen development
headers and eight shared libraries. Its bundled license (SHA-256
`49cf79bdb35734b52fe6203013b3bd759f81e998cd32aa2c65c51db9a88c61d2`)
permits headers as well as shared libraries; the current
[cuDNN supplement](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html)
lists runtime shared libraries. Packaging chooses the narrower boundary:
`filter_cudnn_runtime.py` removes headers/static archives in the same installation
RUN, before any final-image layer commits. It rejects unknown files, missing
libraries, changed license bytes, and unrecorded or escaped paths. Unmodified
ELF libraries and notices remain; their SHA-256 identities are recorded at
`/usr/share/doc/npa-openpi/cudnn-runtime.json`. Removing a development base's
headers in a later layer would not satisfy this boundary.

`runtime-payload.json` records independently computed hashes from complete,
hash-verified upstream wheels. `verify_image.py` binds the saved config/manifest
and every ordered layer to the inspected image ID, reads every regular file,
and rejects changed/relocated runtimes, cuDNN headers (including renamed copies),
cached cuDNN wheels, or missing/changed notices. Later whiteouts cannot erase a
bad ancestor finding. Exact PyTorch 2.7.1 adapter declarations qualify under
their retained complete license; these 53 generated adapter headers are not
NVIDIA cuDNN SDK headers. The general archive/security scanner remains required
for nested payloads and other licensing/security boundaries.

The three compatibility vendor wheels are SHA-256 locked in
`vendor-requirements.txt`. NCCL 2.27.5 retains its header, shared library and
complete BSD-3-Clause license, including its NVTX attribution reference, under
[the release license](https://github.com/NVIDIA/nccl/blob/v2.27.5-1/LICENSE.txt).
All 56 NVSHMEM 3.2.5 wheel payloads match NVIDIA's official CUDA 12 Linux archive
identified by [its release manifest](https://developer.download.nvidia.com/compute/nvshmem/redist/redistrib_3.2.5.json).
The [NVSHMEM product supplement](https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/LICENSE.txt)
permits SDK portions as application components subject to its distribution
requirements. The wheel's CUDA-only notice is supplemented with the full
product license and third-party notices at
`/usr/share/doc/npa-openpi/NVSHMEM-LICENSE.txt`, pinned to SHA-256
`43a87c0ff94ce3196011ff75e17fbee96933c9e1d511557659ece8a326f95e8f`.
These dependencies serve the image's policy workloads; their vendor restrictions
remain applicable and do not confer an independent SDK distribution license.

It does not contain Gemma/pi0.5 weights, DROID records, credentials, terms
acceptance, trained checkpoints, or populated model/data caches. Those artifacts
remain runtime-only under the operator's own access and applicable terms.

This packaging record is not evidence of a successful image scan or workload.
Each public development digest still requires complete pre-publication security,
license, confidentiality, provenance and bootstrap checks, anonymous registry
readback, and substantive GPU validation. A successful single-GPU development
workload does not qualify the eight-node full-DROID release.
