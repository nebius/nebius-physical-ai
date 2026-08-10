# Cosmos3 serving redistribution decision

Decision: `restricted` / build-your-own. Do not mirror
`npa-cosmos3-serving` to an anonymous public registry.

## Three-layer classification

- Source wrapper: repository code under this repository's license.
- Baked runtime: `vllm/vllm-omni` at manifest-list digest
  `sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587`.
  The upstream vLLM-Omni project declares Apache-2.0, but the actual amd64
  image embeds `/NGC-DL-CONTAINER-LICENSE` from its NVIDIA CUDA base.
- Weights/data: none baked. `nvidia/Cosmos3-Super` and guardrail weights are
  fetched at runtime with the operator's own Hugging Face credential and
  license acceptance.

## Why the image is restricted

The NVIDIA Deep Learning Container License grants distribution of a compatible
derived container only subject to conditions that include material additional
primary functionality and downstream terms at least as protective as NVIDIA's
license; it also prohibits distributing the container as a standalone product.
This image is a thin preflight/configuration wrapper around the upstream serving
runtime, and an anonymous GHCR package does not establish those downstream
terms. That is not a sufficiently clear basis for public redistribution.

The safe path is for each operator to build the checked-in Dockerfile into its
own registry and use the vendor base under its own acceptance. A future public
release requires human legal approval of a distribution mechanism that satisfies
the vendor terms, or a replacement base whose public redistribution is clear.

## Authoritative evidence

- NVIDIA Deep Learning Container License:
  https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf
- vLLM-Omni package license metadata:
  https://github.com/vllm-project/vllm-omni/blob/main/pyproject.toml

The packaging contract and `OMNIVERSE_RESTRICTED_TOOLS` enforce this decision.
