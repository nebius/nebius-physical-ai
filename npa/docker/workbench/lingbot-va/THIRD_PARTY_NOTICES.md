# Third-party notices: LingBot-VA

## LingBot-VA

- Source: https://github.com/Robbyant/lingbot-va
- Revision: `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`
- License: Apache-2.0 (`/opt/lingbot-va/LICENSE.txt` in the image)
- Copyright: 2024-2025 The Robbyant Team Authors.
- Citation: *LingBot-VA: Causal World Modeling for Robot Control*, Lin et al.,
  arXiv:2601.21998 (2026), as supplied by the upstream repository.

NPA modifications are limited to source-only runtime packaging, object-store
artifact wiring, a no-telemetry configuration overlay, and workflow provenance.
They are not presented as original LingBot-VA research.

## Inherited components

LingBot-VA credits Wan-Video and Mixture-of-Transformers (MoT). The image
inherits the neutral OS/bootstrap layers of the pinned `npa-wan2-2` parent; its
existing redistribution records remain in `/usr/share/doc/npa-wan2-2`. The
LingBot-VA runtime uses its own PyTorch 2.9/CUDA 12.6 environment and does not
claim compatibility with the parent's CUDA 13 runtime.

The pinned source imports Flash Attention even though the official LIBERO path
uses Flex Attention for training and Torch attention for server inference. NPA
softens only that unused optional import so an unavailable Flash extension does
not prevent these two documented modes from importing; selecting `flashattn`
remains unsupported by this candidate.

## Runtime-only components

- LingBot-VA base and official LIBERO-Long checkpoint: Apache-2.0 model cards
  at the exact revisions documented in `REDISTRIBUTION.md`.
- LIBERO source: https://github.com/Lifelong-Robot-Learning/LIBERO at
  `8f1084e3132a39270c3a13ebe37270a43ece2a01`; fetched only into the
  operator-owned runtime cache.
- LeRobot 0.3.3 is isolated to this runtime because upstream LingBot-VA requires
  it; this is distinct from newer NPA LeRobot tool runtimes.
- `robbyant/libero-long-lerobot` data: CC BY-NC-SA 4.0, operator-staged only,
  never copied into public image layers or fetched by NPA.

Dependency licenses and redistribution rights remain governed by their own
authoritative distributions. This notice neither requests nor records consent.
