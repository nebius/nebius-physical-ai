# Wan 2.2 container redistribution record

Review date: 2026-08-08. This is an engineering classification, not a legal
opinion or substitute for counsel.

## Three separately classified layers

1. **Shipped source.** `Wan-Video/Wan2.2` is checked out at immutable commit
   `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`. Its Apache-2.0 `LICENSE.txt`
   and notices remain in `/opt/byof`. The Debian/Python base and baked Python
   dependency closure are OSS; the complete closure is pinned in
   `/opt/npa/wan2-2/baked-constraints.txt` and its installed identity is
   recorded in `/opt/byof/npa_baked_python_inventory.txt`. Two checked-in,
   Apache-2.0 patches select only TI2V exports and replace the historical
   LGPL-3.0 EasyDict dependency; the S2V-only LGPL audio closure is absent.
2. **Shipped runtime.** Only the digest-pinned official Python base, Debian
   packages, CPU-only BSD-3-Clause PyTorch, and OSS Wan dependencies are image
   bytes. No CUDA, cuDNN, NCCL, TensorRT, `nvidia-*` Python distribution,
   NVIDIA SDK, checkpoint, tokenizer, credential, or cache is shipped.
3. **Runtime-fetched software, weights, and data.** CUDA-enabled PyTorch 2.7.1
   and its NVIDIA dependencies are delivered into the operator's writable
   volume only after `NPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS=YES`; current terms:
   <https://docs.nvidia.com/cuda/eula/index.html> and
   <https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/>.
   Wan TI2V-5B (`921dbaf…`) and UMT5 (`66cb9e7…`) are separately downloaded at
   runtime under their upstream terms and the operator's credentials/network.
   User inputs and generated artifacts are never image inputs.

Release qualification requires scanning the pushed digest with
`npa/scripts/scan_image_wan_payload.py`, separately inspecting the BuildKit SPDX
attestation and SLSA provenance bound to the exact platform manifest, reviewing
the dependency license inventory, and independently confirming the precise
source/model/vendor terms remain current. The scanner proves prohibited-byte
absence; it does not generate or review an SBOM or make a legal determination.
Passing automation proves byte absence and contract consistency, not human legal
approval.

The container starts as UID 1000 and the runtime/model caches are owned by that
user. This is an ownership and accidental-write boundary, not a security
sandbox: the image retains passwordless sudo because the shared SkyPilot image
bootstrap contract may need to create SSH host keys and perform runtime setup.
Operators requiring a privilege boundary must enforce an admission-approved pod
security context outside the image and validate it against SkyPilot bootstrap;
the default image alone does not provide that isolation.
