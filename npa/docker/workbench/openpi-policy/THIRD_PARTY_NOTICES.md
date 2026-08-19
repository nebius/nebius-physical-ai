# Third-party inventory inputs: npa-openpi-policy

Engineering review date: 2026-08-19. The built-image SBOM is the exact installed
inventory; package license files remain in their Python distribution metadata and
Debian copyright directories.

| Component | Immutable input | Shipped? | Classification |
| --- | --- | --- | --- |
| OpenPI source | `Physical-Intelligence/openpi@15a9616a00943ada6c20a0f158e3adb39df2ccac` | yes | Apache-2.0; upstream `LICENSE` remains in `/opt/byof`. |
| LeRobot source dependency | `huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` via the pinned OpenPI `uv.lock` | yes | Apache-2.0. |
| CUDA base | CUDA 12.8.1 cuDNN Ubuntu 24.04 build/runtime stages, digest-pinned in the Dockerfile | yes | Redistributable CUDA container runtime under NVIDIA's CUDA/container terms; it is not Isaac, Omniverse Kit, or a host driver. |
| Python dependency closure | exact upstream `uv.lock` SHA-256 `793488b5…37d74` | yes | Individual OSS licenses retained in installed metadata and inventoried by the release SBOM. |
| DeepDiff security override | `deepdiff==8.6.2` | yes | Exact patched release replaces the lock's vulnerable 8.5.0 while preserving the upstream dependency contract. |
| W&B Python client | lock-resolved Python package; optional `wandb-core` helper removed | Python library only | The policy server does not perform experiment logging. The unused prebuilt Go helper is removed because its embedded runtime has fixed critical vulnerabilities; policy-config import is asserted after removal. |
| Python runtime | uv-managed CPython 3.11 selected by OpenPI's pinned `.python-version` | yes | Python Software Foundation license; the interpreter is copied with the lock-resolved environment so no builder-only symlink remains. |
| ImageIO FFmpeg wrapper | lock-resolved `imageio-ffmpeg`; bundled executable deleted | yes, Python wrapper only | BSD-2-Clause wrapper. Its wheel-bundled static FFmpeg is prohibited by the payload scanner and replaced with Debian's dynamically linked package. |
| Debian FFmpeg | Ubuntu 24.04 repository package and dynamically linked closure | yes | Redistributable LGPL/GPL components with copyright records retained under `/usr/share/doc`; corresponding source is available from Ubuntu/Debian source archives. |
| pi0.5-DROID Polaris / Gemma-derived checkpoint | pinned GCS object-generation manifest `8b97388a…85218` | no; runtime cache only | Operator-scoped use under the Gemma Terms of Use and Prohibited Use Policy. No checkpoint, tokenizer/model payload, or populated cache is an image input. |
| Credentials, operator data, generated actions | operator supplied/runtime generated | no | Never build inputs or image-layer contents. |

The OpenPI checkout includes `LICENSE_GEMMA.txt` because it documents the terms
that govern separately downloaded model material. Its presence does not mean the
image contains that material or grant model redistribution rights.
