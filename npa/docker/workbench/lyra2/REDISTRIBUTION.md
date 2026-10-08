# Lyra 2 reconstruction image

The public image is a dedicated reconstruction bootstrap. It has no dependency
on the Isaac Lab image and ships no Lyra model, source checkout, input video,
inference output or populated runtime cache. Source and model delivery are
separate from permission to redistribute the image.

| Boundary | Delivery and terms |
| --- | --- |
| Source | NPA is Apache-2.0. Lyra source is fetched from `nv-tlabs/lyra` at `9fffc9adc37004091ecf26ef03abfb3abdf4d59a`; its Apache-2.0 license stays in the runtime checkout. The pinned DA3 submodule is `1ed6cb8eee386a3c94077d907b09c7aa1c312cd8`. |
| Baked runtime | Digest-pinned NVIDIA CUDA 12.8.1 devel Ubuntu 24.04 base, Ubuntu packages and hash-locked NPA CPU dependencies. The CUDA base has no cuDNN installation. Ubuntu and Python package license files remain in the image. CUDA redistribution follows the CUDA Toolkit Linux grant; cuDNN and PyTorch are delivered directly at runtime. |
| Weights | `nvidia/Lyra-2.0` revision `c178c3fcf12b63cf98f6749999e6ecb63901669f`, `checkpoints/recon/model.pt`, SHA-256 `d26380a2d2ecceb6c7ed8ccdb6c53d2664259132ddc946c6c189cf29151c8042`. Anonymous upstream delivery; no invented token or consent boolean. The NVIDIA Internal Scientific Research and Development Model License governs use. Never bake or republish the checkpoint. |
| Data | Operator-supplied checksummed RGB capture. No sample data enters the image. Input rights and any attribution remain with the operator's bundle. |
| Cache | Unique per-job temporary directory; fresh source/environment/checkpoint on each run. Model digest verification precedes inference. Failed or partial downloads remain isolated and are removed by the temporary-directory context. No durable cache or cross-operator reuse is implied. |
| Outputs | Native Gaussian PLY, rendered MP4, predicted geometry, provenance and offline HTML are published to the operator's selected output path. The model license's internal R&D use restriction remains applicable; this image does not grant production, resale or redistribution rights for model use. |

The existing bounded internal R&D run scope applies to the reference validation;
there is no new acceptance mechanism. Anonymous access establishes technical
availability, not rights beyond the upstream terms. The source checkout is
verified against its immutable commit and unmodified working tree. Checkpoint
SHA-256 is verified before upstream inference. Keep generated artifacts private
for the reference validation.

Authoritative terms:

- [Lyra source license](https://github.com/nv-tlabs/lyra/blob/9fffc9adc37004091ecf26ef03abfb3abdf4d59a/LICENSE)
- [Lyra model card and license](https://huggingface.co/nvidia/Lyra-2.0/tree/c178c3fcf12b63cf98f6749999e6ecb63901669f)
- [NVIDIA Internal Scientific Research and Development Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/)
- [CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html)

Publication requires the repository's exact-image payload, secret, license,
vulnerability, non-root, revision and bootstrap gates. A dedicated image name or
successful dependency install is not evidence of native reconstruction. Keep the
image quarantined from supported releases until a real GPU produces and decodes
the declared artifacts from that exact digest.
