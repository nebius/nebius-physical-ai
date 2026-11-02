# Third-party inventory inputs: npa-wan2-2

Engineering review date: 2026-08-13. Exact installed package versions are
machine-readable in `/opt/byof/npa_baked_python_inventory.txt`; distribution license
texts remain under `/usr/share/doc/*/copyright`, Python wheel license files
remain in their `.dist-info` directories, and release SBOM generation inventories
the built digest. This record does not replace those complete artifact records.

| Component class | Immutable input / version | Shipped? | License source and conclusion |
| --- | --- | --- | --- |
| Wan source | `Wan-Video/Wan2.2@42bf4cfaa384bc21833865abc2f9e6c0e67233dc` | yes | Upstream `LICENSE.txt`, Apache-2.0: <https://github.com/Wan-Video/Wan2.2/blob/42bf4cfaa384bc21833865abc2f9e6c0e67233dc/LICENSE.txt> |
| Runtime base | `ubuntu:26.04@sha256:42527649dbc4f9197a6073f426f488cbff0e9cc9f2afce32d91c383e4875634a` | prepared filesystem | Official Ubuntu image and redistributable distribution packages; installed copyright records remain in `/usr/share/doc`. The unused inherited Pebble service manager and superseded packages are excluded before the final filesystem copy. See <https://ubuntu.com/legal/intellectual-property-policy>. |
| Python donor | `python:3.11.16-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84` | repaired `/usr/local` only | Official Python interpreter and standard library, including `/usr/local/lib/python3.11/LICENSE.txt`. Debian donor layers/libraries remain in a disposable stage. Source/build provenance: <https://github.com/docker-library/python>. |
| Repaired installer | `pip==26.2.1+npa.1`, source and vendor pins in `common/secure_pip/inputs.json` | yes, including ensurepip bundle | Source-built OSS derivative; original licenses, vendor BOM, and repair provenance remain installed. See [the source repair record](../common/secure_pip/README.md). Build-stage tools are not copied into the final image. |
| Wan Python closure | complete pins in `baked-constraints.txt`; exact installed closure in baked inventory/SBOM | yes | Apache/BSD/MIT/ISC/MPL and redistributable copyleft components. License/notice files are retained. No proprietary classification is accepted. |
| Wan EasyDict compatibility | checked-in `easydict_compat.py` | yes | NPA-authored Apache-2.0 implementation. It replaces the historical LGPL-3.0 `easydict` distribution; the built-image scanner fails if that distribution returns. |
| Capability-scoped Wan exports | checked-in `wan_init_ti2v.py` | yes | NPA-authored Apache-2.0 patch retaining the advertised TI2V-5B path while omitting the unadvertised S2V import and its LGPL audio dependency closure. |
| ImageIO FFmpeg wrapper | `imageio-ffmpeg==0.6.0`, installed from its source distribution | yes | BSD-2-Clause wrapper: <https://github.com/imageio/imageio-ffmpeg/tree/v0.6.0>. Its PyPI wheels bundle a separate static FFmpeg executable, so this image refuses that wheel and points the wrapper at `/usr/bin/ffmpeg`; the byte scanner fails if a bundled executable returns. |
| Ubuntu FFmpeg runtime | Resolute `ffmpeg` and its dynamically linked Ubuntu closure; exact installed versions in the candidate SBOM | yes | The distribution build enables GPL components; package copyright texts are shipped under `/usr/share/doc`, and corresponding source is available from the signed Ubuntu archive and <https://launchpad.net/ubuntu/+source/ffmpeg>. These are redistributable copyleft components with notice/source obligations. |
| CUDA PyTorch closure | torch/vision `2.13.0/0.28.0`, CUDA 13.0 dependencies, complete hashes in `runtime-requirements.txt` | no; runtime volume | PyPI metadata names the explicit `cuda-toolkit` and NVIDIA package closure. Operator must accept current CUDA/NVIDIA terms before download: <https://docs.nvidia.com/cuda/eula/index.html>. |
| TI2V-5B model | `Wan-AI/Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e` | no; runtime cache | Upstream model card declares Apache-2.0: <https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/921dbaf3f1674a56f47e83fb80a34bac8a8f203e>. |
| UMT5 tokenizer | `google/umt5-xxl@66cb9e7e85526fe440a945569e42c72fb6cbc0ad` | no; runtime cache | Upstream model card declares Apache-2.0: <https://huggingface.co/google/umt5-xxl/tree/66cb9e7e85526fe440a945569e42c72fb6cbc0ad>. |
| Operator data / credentials / generated video | operator supplied | no | Never build inputs or shipped layers. |

The artifact scanner fails on proprietary NVIDIA/CUDA payloads, weights,
caches, credentials, and the historical LGPL Python distributions removed by
this TI2V-only design in both the flattened root filesystem and every
individual layer, including bytes later hidden by whiteouts. Publication still
requires a human to review the SBOM/license findings and current vendor terms.
