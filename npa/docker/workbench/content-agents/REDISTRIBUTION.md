# NVIDIA Content Agents public-image boundary

`npa-content-agents:0.5.2-npa2` is a public, zero-vendor-payload adapter image.
It bakes the Apache-2.0 NVIDIA Content Agents release `v0.5.2` at immutable
commit `36dbf3f274f8e256637230a05a085853f65cc175`, NPA's downloader, and the
reviewed upstream runtime lock. It contains no OVRTX wheel, installed runtime,
or OVRTX native payload.

OVRTX `0.3.0.312915` is a proprietary runtime SDK, not model weights. On first
render use the operator receives its exact architecture-specific wheel directly
from NVIDIA's anonymous package index. The complete upstream lock SHA-256 is
`ed582577175e4a5b32f8b69ef9cdbfc3d7337f3786051d8b076e30a2652f6fa5`.
The bootstrap verifies the lock, platform entry, upstream ready marker, and
installed version before atomically publishing an immutable cache identity.

NVIDIA states that downloading or using Omniverse signifies agreement. NPA
links the terms for review and adds no prompt, acceptance flag, or consent
record:

- https://docs.omniverse.nvidia.com/connect/latest/common/NVIDIA_Omniverse_License_Agreement.html
- https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/
- https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/

Artifact classification:

- Content Agents source: Apache-2.0, immutable source commit, baked.
- OVRTX runtime: NVIDIA Proprietary Software, exact anonymous runtime fetch by
  the operator, absent from every image layer.
- OvPhysX and Scene Optimizer Core: not installed or fetched.
- Material library: generated at run time from a minimal PreviewSurface; no
  NVIDIA sample/material library is baked.
- Model weights: not installed or cached. The accepted path calls hosted Token
  Factory using the operator's runtime secret.
- Customer USD and generated artifacts: run time only under configured S3.
- Graphics/display driver userspace: host-injected by NVIDIA GPU Operator, not
  baked. Required capabilities remain `compute,utility,graphics,display`.

The accepted consumer is one rigid, non-articulated Isaac object. Texture
generation, joint authoring, arbitrary scenes, articulated robots, and direct
Genesis USD consumption remain outside this contract.
