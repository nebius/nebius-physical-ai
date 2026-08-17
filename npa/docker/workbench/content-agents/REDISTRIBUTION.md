# NVIDIA Content Agents runtime boundary

`npa-content-agents` is deliberately `restricted` and build-your-own. The
NVIDIA Content Agents source at release `v0.5.2` / commit
`36dbf3f274f8e256637230a05a085853f65cc175` is Apache-2.0. The isolated OVRTX
`0.3.0.312915` wheel is labeled `NVIDIA Proprietary Software` and contains
native NVIDIA rendering payloads. Those bytes make the complete image
ineligible for the NPA public mirror even though the worker source is open.

Before building, the operator must review and accept the current NVIDIA
Software License Agreement
(`https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/`)
and Product Specific Terms for NVIDIA AI Products
(`https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/`),
which incorporate the former Omniverse product-specific terms. Then set
`NPA_CONTENT_AGENTS_ACCEPT_NVIDIA_OMNIVERSE_TERMS=YES` only in the build shell.
The marker is consumed by `build.sh`; it is not passed into Docker or baked.

Artifact classification is separate:

- Content Agents source: Apache-2.0, immutable source commit, baked.
- OVRTX runtime: NVIDIA Proprietary Software, hash-locked, baked; private
  operator registry only.
- OvPhysX: not installed. Physics simulation tuning is not claimed.
- Scene Optimizer Core: not installed or fetched. Optimization is disabled.
- Material library: generated at run time from a small USD PreviewSurface; no
  NVIDIA sample/material asset library is baked or used by the accepted path.
- Model weights: not installed or cached. Material and physics inference calls
  a configured OpenAI-compatible hosted endpoint with the operator's secret.
- Customer USD and generated artifacts: run time only, under the configured
  private S3 prefix.

The accepted consumer is one rigid, non-articulated Isaac object. Texture
generation, joint authoring, arbitrary scenes, articulated robots, and direct
Genesis USD consumption are outside this contract.
