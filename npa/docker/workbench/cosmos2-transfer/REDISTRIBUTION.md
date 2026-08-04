# Cosmos Transfer 2.5 redistribution evidence

This document records an engineering redistribution classification. It is not legal advice.
It applies only to the image built by the adjacent Dockerfile from the
immutable inputs below. Evidence was accessed on 2026-08-03. A built-image audit
is mandatory before the packaging contract may classify the image `public`.

## Immutable provenance

- Image tag: `npa-cosmos2-transfer:2.5.1-skypilot-ready-20260801T053000Z`
- Upstream source: `https://github.com/nvidia-cosmos/cosmos-transfer2.5`
- Upstream revision: `67d56b7d550a3911024a32dc23ae0bae5258e633`
- CUDA build base: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`
- CUDA build linux/amd64 manifest: `sha256:3986465b3dd3b4d602c07061f2cff417e0bfb24810129408d4eb12e111015a6c`
- CUDA final runtime base: `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`
- CUDA runtime linux/amd64 manifest: `sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9`
- uv build/runtime image: `ghcr.io/astral-sh/uv:0.8.12`
- uv linux/amd64 manifest: `sha256:e1d7fa999ae39871fc2fd6c9c572d970cc877a239d2a16238a204c23772d1c0e`
- Python: uv-managed CPython `3.10.18`
- Ubuntu archive snapshot: `20260801T053000Z`
- Python closure: upstream `uv.lock`, installed with `uv sync --locked --no-dev
  --no-editable --extra=cu128`, plus the hash-pinned vulnerability fixes in
  `security-overrides.txt`

## Artifact classification

| Artifact category | Evidence and classification | Image treatment |
|---|---|---|
| Cosmos Transfer source | The pinned repository declares Apache-2.0 in [`LICENSE`](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/67d56b7d550a3911024a32dc23ae0bae5258e633/LICENSE) and `pyproject.toml`. | Included at the full SHA. `LICENSE` and `ATTRIBUTIONS.md` are retained. |
| Upstream `assets/` and Git LFS objects | The pinned `.gitattributes` routes `assets/**` and media suffixes through LFS. The tree contains 444 LFS paths (258,036,941 declared bytes), including third-party example and post-training media. No explicit license covering every media origin was found. Classification: ambiguous, therefore not redistributable here. | `GIT_LFS_SKIP_SMUDGE=1` is set for fetch and checkout. `.git`, `.git/lfs`, and the entire `assets/` tree are removed in the checkout layer and forbidden by a build assertion. |
| Cosmos Transfer model weights | NVIDIA publishes the gated model under the [NVIDIA Open Model License](https://www.nvidia.com/content/dam/en-zz/Solutions/license-agreements/enterprise-software/nvidia-open-model-license-agreement-16-6-2025.pdf). Operators must accept its terms for themselves. | No checkpoint is baked. `HF_TOKEN` is accepted only at runtime; inference refuses before download when it is absent. The cache is a runtime volume. |
| CUDA/cuDNN base runtime | The public NGC CUDA page identifies the container and its governing [NVIDIA Deep Learning Container License](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf). That license expressly permits distributing a compatible derived whole container for running the distributor's application, subject to its notice, NVIDIA-GPU-use, ownership, and protective-terms conditions. This image adds the complete Cosmos application and is not a standalone redistribution of extracted NVIDIA components. The [CUDA Toolkit EULA](https://docs.nvidia.com/cuda/archive/12.0.1/eula/index.html) separately identifies distributable runtime components when incorporated into an application with material additional functionality. | The devel image is a discarded build stage; the final image inherits the smaller digest-pinned runtime base. `NGC-DL-CONTAINER-LICENSE` must remain present. The OCI license/source labels and this notice make the NVIDIA conditions visible to recipients. |
| Python dependencies | Direct source dependencies are fixed by the pinned lockfile. The only direct Git dependency is [Video-Depth-Anything at `f70048c9599cc9221e7d70b098621909316fa4a4`](https://github.com/jeanachoi/Video-Depth-Anything/tree/f70048c9599cc9221e7d70b098621909316fa4a4), whose repository license is Apache-2.0. Registry artifacts are hash-pinned by `uv.lock`; installed package license metadata/files and the upstream attribution bundle are subject to the final Trivy/SBOM audit. PyPI's primary records identify [NLTK 3.10.0](https://pypi.org/project/nltk/3.10.0/) and [msgpack 1.2.1](https://pypi.org/project/msgpack/1.2.1/) as Apache-2.0, [defusedxml 0.7.1](https://pypi.org/project/defusedxml/0.7.1/) as PSF-2.0, and [pip 26.2](https://pypi.org/project/pip/26.2/) plus [setuptools 83.0.0](https://pypi.org/project/setuptools/83.0.0/) as MIT; their exact wheel URLs and SHA-256 fragments are checked in. | The production, non-dev locked closure is included. Four security replacements plus pip required by SkyPilot's source-overlay bootstrap are installed from hash-verifying PEP 503 wheel URLs using `--no-deps`; the build imports or invokes them and asserts every exact installed version. The unused pip/setuptools bundled with the standalone base interpreter are removed, while the venv retains only the current audited packaging tools it needs. The upstream lock contains two package-metadata conflicts (Megatron's NumPy range and the local `cosmos-oss` version declaration), so a whole-environment `uv pip check` would reject NVIDIA's otherwise unchanged lock and is not used as a substitute for real inference. Matplotlib's unnecessary `mpl-data/sample_data`, the unused scikit-image data/fetch module, W&B's unused native services, and NumPy/SciPy binary test fixtures are removed before the build assertion. NumPy's single BSD-licensed `_core/tests/_natype.py` source helper remains because NumPy's own `numpy.testing` import requires it through SciPy/Transformers; a build-time SigLIP/SciPy import proves that dependency. SciPy 1.15.3's [`_sobol_direction_numbers.npz`](https://github.com/scipy/scipy/blob/v1.15.3/scipy/stats/_sobol_direction_numbers.npz) is runtime algorithm data governed by SciPy's [BSD license](https://github.com/scipy/scipy/blob/v1.15.3/LICENSE.txt), not media or a model, and is narrowly allowed. SentencePiece's small `.bin` files are Unicode normalization tables governed by its Apache-2.0 package license, not learned weights. `_virtualenv.pth` and the venv's `distutils-precedence.pth` are text startup hooks, not checkpoints. Those exact paths are narrowly allowed and every other `.pth`/`.npz` remains forbidden. Any other unclassified or incompatible license discovered in the actual registry image blocks the `public` classification. |
| Python interpreter | The uv-managed CPython 3.10.18 archive is supplied by [Astral's python-build-standalone project](https://github.com/astral-sh/python-build-standalone), which publishes its build logic under MPL-2.0 and preserves the PSF and bundled-library license material in the distribution. CPython itself is governed by the [Python Software Foundation License](https://docs.python.org/3/license.html). | The exact version is installed under `/opt/cosmos/uv-python`; the registry filesystem audit must confirm its bundled license material is present. |
| uv | [uv is dual Apache-2.0/MIT](https://github.com/astral-sh/uv#license). The pinned upstream Cosmos runtime invokes `uvx hf` for gated checkpoint retrieval rather than embedding a model client or credential. | The digest-pinned `uv` and `uvx` executables are included in the final image so operator-authorized model downloads work in every orchestrator environment; no package or model bytes are fetched at build time. |
| Ubuntu packages | Runtime packages come from Ubuntu's official [snapshot service](https://snapshot.ubuntu.com/) at immutable snapshot `20260801T053000Z`; Ubuntu documents snapshots as the supported mechanism for reproducible package states. Before `apt-get update`, every inherited APT source (including the mutable CUDA source) is removed and one exact snapshot-only deb822 source is written. Exact installed versions and license files are captured by the registry-image SBOM and Trivy license scan. Ubuntu publishes the matching source packages and copyright records; GPL/LGPL and other reciprocal components remain redistributable only while their corresponding-source, notice, and relinking obligations are met. | The inherited runtime is upgraded from the same immutable snapshot before CA certificates, FFmpeg, netcat, OpenSSH, procps, rsync, and sudo are installed. A license gap or unmet reciprocal-license obligation in the built closure blocks redistribution. |
| Live-test input | `src/npa/workbench/cosmos/fixture.py` authors a moving grid and boxes using FFmpeg `lavfi`; it copies no external pixel, video, model, or dataset bytes. Its JSON follows the pinned Apache-2.0 `InferenceArguments`/`EdgeConfig` schema and requests four diffusion steps. | The generator is included; the generated MP4 and spec exist only in run scratch/S3 input, never in image layers. |

## Enforced build boundary

The Dockerfile accepts no secret build argument or environment variable. The build
script removes `HF_TOKEN`, `NGC_API_KEY`, and Nebius IAM token variables from the
BuildKit process. Public source, Ubuntu, PyPI, PyTorch, NVIDIA Cosmos dependency,
and GitHub endpoints are the only build-time network sources.

`assert_no_forbidden_payload.sh` fails the build for:

- `.git` or Git LFS metadata;
- the upstream `assets/` tree or source-side media;
- any checkpoint/weight-like suffix, regardless of file size;
- Hugging Face or Torch model caches; and
- unexpected source files larger than 10 MiB outside the locked venv.

The guard runs once in the source build stage and again in the final stage. Release
qualification must additionally inspect the registry manifest, config/history,
exported filesystem, and every layer; run the Omniverse payload scanner; run Trivy
vulnerability, secret, and license scans; and retain an SBOM. Dockerfile inspection
alone is not evidence that the built bytes satisfy this classification.

The cross-stage environment copy assigns UID/GID 1000 directly. This preserves the
non-root runtime ownership without a second copy-on-write layer containing the full
dependency tree.

`openssh-server` is present only for SkyPilot bootstrap compatibility. Its package
post-install step generates random host keys, so the same build layer deletes every
`/etc/ssh/ssh_host_*` file. SkyPilot generates ephemeral keys during live bootstrap;
no reusable host private key is embedded in the image.

## Runtime obligations

- Run only on NVIDIA GPUs as required by the inherited NVIDIA runtime terms.
- Preserve the Apache-2.0 license, upstream attributions, this notice, and the
  inherited NVIDIA container license.
- Supply `HF_TOKEN` only at runtime after the operator has accepted the model
  license. Never embed or forward the token into image history, artifacts, or logs.
- Mount `/opt/cosmos/model-cache` for gated weights. Do not promote that cache into
  a derived image or published artifact.
- Keep NLTK's path-security enforcement enabled. `NLTK_DATA` authorizes only the
  dedicated `/opt/cosmos/model-cache` volume because the gated Guardrail model
  resolves its NLTK tables through Hugging Face blob links inside that volume.
- Preserve package copyright files and make corresponding source available as
  required by GPL/LGPL and other reciprocal licenses identified by the SBOM.
- Re-run the complete classification and built-byte audit whenever the source,
  base digest, lockfile, system packages, or fixture changes.
