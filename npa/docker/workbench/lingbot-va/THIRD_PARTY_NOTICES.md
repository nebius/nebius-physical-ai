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
LingBot-VA runtime uses its own security-maintained PyTorch 2.13/CUDA 12.6
environment, within the upstream project's declared Torch lower bound, and
does not claim compatibility with the parent's CUDA 13 runtime. The pinned
source has no TorchAudio or Accelerate import; both are intentionally absent
from the runtime closure. Native GPU qualification is still required.

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
  it; this is distinct from newer NPA LeRobot tool runtimes. The contiguous
  training-subset regression executes the cumulative-index body from upstream
  [`lerobot.datasets.utils.get_episode_data_index`](https://github.com/huggingface/lerobot/blob/b883328e6c95681ca90a18b102e4ae5e1f91e2bf/src/lerobot/datasets/utils.py)
  at `b883328e6c95681ca90a18b102e4ae5e1f91e2bf` (Apache-2.0), with a
  test-only CPU tensor carrier; production uses the installed native package.
- Raw data: [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero)
  at `affa19c0de0f6bce2a7edd26dddef8a532e7e6f6`, whose immutable card declares
  CC-BY-4.0. It is the HuggingFace VLA Team's LeRobot v2.1 conversion of the
  original LIBERO data and is operator-staged only, never copied into image
  layers. The preparation stage selects source task IDs 0–9, decodes raw image
  bytes, and creates fresh run-scoped MP4/latent outputs; it does not use the
  separate Robbyant CC-BY-NC-SA latent dataset.
- LIBERO data credit: Liu, Zeng, Patil, Mu, Xu, Liu, Wu, Liu, Tenenbaum, et
  al., *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*,
  arXiv:2306.03310 (2023), as requested by the exact raw-source card.

Dependency licenses and redistribution rights remain governed by their own
authoritative distributions. This notice neither requests nor records consent.
