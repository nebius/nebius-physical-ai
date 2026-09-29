# CUDA 13 base for Blackwell, including RTX PRO 6000

**Adopting FA4 on RTX PRO 6000? Start with the
[application-image guide](../../../../../docs/workbench/guides/rtx6000-fa4.md).**
It covers the local build, application Dockerfile, dependency constraints,
explicit model integration, GPU checks and deployment by digest.

`cuda13-blackwell` is the canonical shared image-family name for RTX PRO 6000,
B200 and B300 build targets. Select the physical GPU in the workload's resource
configuration. Architecture coverage is separate from a measured capability
result on that GPU; the current FA4 qualification is specific to RTX PRO 6000.

This directory builds `npa-base:cuda13-blackwell-dev-<full-source-sha>` when
`build.sh --tag dev-<full-source-sha>` is used. That is a local artifact until
published through the appropriate registry process. The base contains the
Python/CUDA/FA4 environment and validation scripts; add application code,
dependencies, launch commands and scheduler bootstrap in the derived image.

For existing consumers, `base/cuda13-b300/build.sh` forwards to this build script.
Use this canonical directory for the Dockerfile and validation scripts.
The build script tags the same image as both `cuda13-blackwell-<suffix>` and
`cuda13-b300-<suffix>`. `--registry` adds both registry references, and `--push`
pushes both, using `DOCKER_CONTEXT` when set. Already-published image references
and recorded validation identities are unchanged; the rename publishes nothing.

See [FA4 pins, restrictions and measured evidence](../../../../../docs/workbench/flash-attention.md)
and the [public image catalog](../../../../../docs/workbench/container-image-catalog.md)
for the distinction between source recipes and accepted published images.

## Standalone FA2 comparison variant

`build.sh --attention-backend fa2 --tag dev-<full-source-sha>` builds a separate
`npa-base:cuda13-blackwell-fa2-dev-<full-source-sha>` image. It compiles FA2 from
the same pinned upstream revision and common CUDA/PyTorch layers, with
`FA2_CUDA_ARCHS=120` by default. Override that environment variable only when
building and qualifying other architectures. FA4 remains the build default;
the two attention packages are never installed together. `NPA_ATTENTION_BACKEND`
and `NPA_FLASH_ATTN_COMMIT` identify the baked selection and source for the GPU
checker and benchmark workers.
Set `FA2_NVCC_THREADS=1` when compiling on a host with limited RAM; the default
is upstream's four compiler threads. This affects build memory and throughput,
not the GPU kernel configuration or attention benchmark iteration count.
The new FA2 variant uses only its canonical tag; it has no historical B300
alias to preserve and makes no B300 qualification claim.

See [FA2/FA4 comparison and inference tile experiments](../../../../../docs/workbench/guides/fa2-fa4-comparison.md)
for correctness checks, full SDXL generations and repeated timing. The
[measured RTX comparison](../../../../../docs/workbench/fa2-fa4-validation.md)
records the exact FA2/FA4 image identities and 108 complete SDXL generations.
Selected tuned FA4 inference calls are faster; full-model generation is
effectively tied with FA2. Prior receipts do not validate replacement bytes.

The FA4 variant installs `flash_attn.rtx.make_inference_attention`, an explicit
SM120 inference factory with a small shape-qualified tile profile. It rejects
training use and a different upstream revision; other shapes use native FA4.
The normal root/CuTe APIs are unchanged. See the comparison guide before opting
a model into this profile; kernel timings alone do not qualify an application.

## Qualify the intended GPU

The `base-cuda13-b300` golden-eval entry asserts capability **10.3**. It has an
explicit `unlimited` timeout because cold CuTe compilation for the complete
output/gradient matrix has not been timed on the new datacenter dependency set.
Run this entry with local `--execute` inside the selected image on B300;
the golden-eval runner rejects serverless execution for unlimited entries.
For RTX PRO 6000, B200 or H100, invoke `/npa/gpu_capability_smoke.py` directly
with `--expect-capability 12.0`, `10.0` or `9.0`, respectively.

Use `--json-output` to retain the numerical checks, matrix-derived
`expected_cases`, total `elapsed_seconds` and each completed case's
`elapsed_seconds`. Total time includes imports, controls, reference checks and
cold JIT compilation; it is qualification duration, not an attention benchmark.
The earlier receipts predate these timing fields and remain bound to their
recorded image and source hashes. B200, B300 and H100 qualification of the new
pins remains pending; each downstream rebuild also needs its own workload test
before release.
