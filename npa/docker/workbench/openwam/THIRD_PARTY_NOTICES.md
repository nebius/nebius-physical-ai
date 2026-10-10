# OpenWAM third-party notices

## OpenWAM

- Source: <https://github.com/OpenWAM-Official/OpenWAM>
- Immutable source revision: `48bd67b89d489b14d03b8d92bc66e65d306df32e`
- License: Apache-2.0; the authoritative text is retained at
  `/opt/openwam/LICENSE`.
- Citation: OpenWAM Official Team, *OpenWAM: An Open, Modular Exploration
  Towards Systematic World-Action Model Pretraining*, arXiv:2609.07398 (2026).
  Preserve the complete author list and BibTeX from `/opt/openwam/CITATION.cff`.
- Modification: NPA adds only the external workflow adapter and packaging;
  training, serving, and LIBERO bridge entrypoints remain upstream-native.

## OpenWAM-alpha and inherited Wan 2.2 component

- Foundation model: <https://huggingface.co/OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model>
  at `52df4e66c82c5c8b480adcc8d01f4db7415dfb56`, public Apache-2.0 model-card
  revision, fetched at runtime only.
- Video backbone: <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B> at
  `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`, public Apache-2.0 model-card
  revision, fetched at runtime only. Credit the Wan authors and retain the
  upstream model-card citation with every derived artifact.

## LIBERO

- Code: <https://github.com/Lifelong-Robot-Learning/LIBERO> at
  `8f1084e3132a39270c3a13ebe37270a43ece2a01`, MIT. The installed source retains
  its license at `/opt/openwam-libero-source/LICENSE`.
- Benchmark data origin: LIBERO, CC-BY-4.0. The OpenWAM mirror used for the
  training-format asset is `OpenWAM/LIBERO@bcb2eaf1121ae4cbd324f8862807abce282500e8`.
  Attribute the original LIBERO authors and CC-BY-4.0 source independently of
  the mirror card label.

The Dockerfile's upstream locked dependency environment includes additional
components. The exact image scan/SBOM for each candidate digest is the
authoritative inventory for their resolved licenses and notices; this notice
does not substitute for it.
