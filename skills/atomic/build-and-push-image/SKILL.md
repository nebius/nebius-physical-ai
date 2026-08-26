---
name: build-and-push-image
description: Use when building, tagging, validating, or publishing NPA workbench container images through immutable full-SHA public development tags and digest-identical supported GHCR releases.
---

# Build And Push Image

Load and obey `skills/atomic/secure-image-build/SKILL.md` before every image
build, tag, push, copy, or promotion. That skill defines the mandatory order and
refusal conditions; this skill records NPA-specific build and GPU details.

## Build Contract

1. Build only a checked-in Dockerfile under `npa/docker/workbench/<tool>/`.
2. Resolve the exact 40-character commit and use the one official namespace,
   `ghcr.io/nebius/nebius-physical-ai`. A pre-release tag is exactly
   `dev-<full-git-sha>` on the normal `npa-<tool>` package.
3. Require `redistribution: public` before any official push. Build restricted
   images, including `cosmos3-serving`, only into an operator-controlled
   registry; neither a private package nor a development tag changes licensing.
4. Run every pre-publication security, packaging, payload, provenance, SBOM,
   vulnerability, secret, non-root, base-pin, and bootstrap-contract gate before
   pushing the public development tag.
5. After push, resolve the immutable digest, repeat exact-digest checks, and
   verify anonymous pullability. Use that digest for functional GPU validation.
6. Promote only the real-GPU-validated digest to the supported release tag with
   `npa.deploy.publish_public`; verify digest parity and anonymous pull afterward,
   then record the accepted release digest for anonymous read-only health checks.

Use `NPA_PUBLIC_REGISTRY` only to select the configured official namespace.
`NPA_REGISTRY` remains the generic execution override. Do not introduce a second
official source registry or a separate pre-release package naming convention.

## Packaging And Runtime Fetch

Read `docs/workbench/container-packaging.md` and update
`npa/docker/workbench/packaging-contract.yaml` when an image or tier changes.
Make the final stage non-root, pin resolvable bases by digest, keep credentials
and generated data out of layers, and make service health machine-checkable.

Runtime-fetch images must prove absence on the built artifact. Use
`scan_image_omniverse_payload.py` for Isaac-family images and
`scan_image_ltx_payload.py` for LTX. LTX source and weights both remain runtime
fetches under the operator's entitled `HF_TOKEN`; never fetch either at build
time or cache acceptance in the image.

For SkyPilot workflow images, prove the versioned bootstrap behavior before
adding its OCI label. A label is an attestation of tested behavior, not a switch
that makes the behavior true.

## GPU Architecture Coverage

Choose the GPU with `skills/atomic/gpu-selection/SKILL.md`. Prebuilt Torch wheels
ship fixed architecture sets; `TORCH_CUDA_ARCH_LIST` affects only source-built
extensions. Check wheel flags with `torch._C._cuda_getArchFlags()` and validate
custom kernels on the target hardware.

`npa-base`'s CUDA 13 contract covers `sm_80`, `sm_90`, `sm_100`, `sm_103`, and
`sm_120`; its wheel must report `sm_80`, `sm_90`, `sm_100`, and `sm_120`.
Validate datacenter Blackwell images with
`npa/scripts/validate_blackwell_image.sh <image> --target b200|b300 --gpu` and
record truthful additive tags in `npa/docker/workbench/blackwell-dc-images.json`.

## Verify

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/docker/ npa/tests/deploy/ \
  npa/tests/guardrails/test_secure_image_build_skill.py \
  npa/tests/guardrails/test_skills_index.py -q
```
