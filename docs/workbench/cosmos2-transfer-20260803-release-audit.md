# Cosmos Transfer 2.5 release audit — 2026-08-03

> **Historical release only — not a current SkyPilot bootstrap attestation.**
> The exact index below predates `skypilot-0.12.2-v1`; its linux/amd64 OCI
> config does not carry
> `org.nebius.npa.skypilot-bootstrap-contract=skypilot-0.12.2-v1`. Its SBOM and
> provenance attest what was built, but do not attest the worker bootstrap
> contract. Current workflow preflight must reject this digest. Keep the digest
> immutable for provenance; publish a separately named, additive release only
> after the new digest passes the bootstrap and redistribution gates.

This is an engineering redistribution and release record, not legal advice. It
qualifies only the exact registry bytes below. Artifact-level license evidence
and runtime obligations are recorded in
`npa/docker/workbench/cosmos2-transfer/REDISTRIBUTION.md`.

## Candidate identity and provenance

| Field | Qualified value |
|---|---|
| Run | `npa-cosmos2-legal-20260803T025147Z` |
| Private source tag | `npa-cosmos2-transfer:2.5.1-skypilot-ready-20260801T053000Z` |
| OCI index digest | `sha256:9b4c5eb505353aa3dea37284c662f3cff306fa7c902f040e559f7939173345cc` |
| Linux/amd64 image digest | `sha256:cda3588d417518aec14d66a292f2dd4984739bc922492f2e78cae2bee377e450` |
| Config digest | `sha256:1935be2a054d17aa9188415c9899b49951d1708b43bce49e93a0983bb5aafe4c` |
| Attestation manifest | `sha256:d43d579cb8e1e60afe546cb5746a7454e3d5c9bc7f6a806a5cf12c6ddaaa0540` |
| Cosmos Transfer revision | `67d56b7d550a3911024a32dc23ae0bae5258e633` |
| Build base | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:3986465b3dd3b4d602c07061f2cff417e0bfb24810129408d4eb12e111015a6c` |
| Final base | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9` |
| uv image (build and runtime tools) | `ghcr.io/astral-sh/uv:0.8.12@sha256:e1d7fa999ae39871fc2fd6c9c572d970cc877a239d2a16238a204c23772d1c0e` |
| Python / Ubuntu snapshot | CPython `3.10.18` / `20260801T053000Z` |
| Build credentials | none |
| Packaging contract | `tier: job`, `redistribution: public` |

The pushed index has one `linux/amd64` image plus its SBOM/provenance
attestation. The image has 22 compressed layers totaling 8,663,394,825 bytes;
the merged filesystem declares 17,323,651,113 bytes. The config selects user
`ubuntu`, an exec-form `/opt/cosmos2-transfer/entrypoint.sh`, a runtime volume
at `/opt/cosmos/model-cache`, and limits NLTK's operator-authorized download
root to that volume. `LD_LIBRARY_PATH` covers the CUDA runtime directory whose
unversioned `libcudart.so` compatibility link targets the inherited
`libcudart.so.12`. Its revision label matches the full source SHA. Build logs
contained no mutable APT endpoint, credential name, credential-shaped value,
or actual runtime/registry secret value.

## Registry-byte audit

All checks below addressed the registry digest, not the local Docker daemon.

| Check | Result |
|---|---|
| Manifest and config | one `linux/amd64` child; non-root `ubuntu`; 67 history entries; zero credential-pattern matches in config/history |
| Every-layer inventory | 22/22 layers read; 68,362 members and 58,002 files; zero Git/LFS paths and zero sensitive/cache payloads |
| Merged filesystem | 55,735 files, 7,587 directories, 65,318 members; zero `.git`/LFS paths and zero sensitive paths |
| Large files | 90 files over 10 MiB, totaling 15,876,898,970 bytes; all classified CUDA/Python libraries or executables (including the Apache-2.0/MIT `uv` runtime tool), no media/checkpoints |
| Media | 40 package-owned test/icon/documentation files, 365,651 bytes total; largest 122,772 bytes; no upstream Cosmos `assets/` media |
| Weight-like suffixes | seven classified non-model files: two text startup hooks, one SciPy BSD Sobol table, and four Apache SentencePiece Unicode tables |
| Model caches | only empty declared runtime cache directories; no Hugging Face, NGC, Torch, or checkpoint bytes |
| `/root` | only inherited `.bashrc` and `.profile`; no application, cache, key, or credential payload |
| License/notice inventory | 622 merged-filesystem license/notice paths |
| Omniverse payload scan | 65,318 entries; clean; zero payload and history hits |
| SBOM | in-toto SPDX attestation for the exact child: 596 packages, 5,072 files, and 8,331 relationships |
| Provenance | in-toto SLSA v1 attestation for the exact child; five digest-pinned builder/base materials; no credential-shaped content |

## Trivy triage

The installed Trivy version scanned the exact registry child for OS and Python
vulnerabilities, secret patterns, license coverage, and configuration findings.
It reported 492 vulnerabilities: 0 critical, 53 high, 398 medium, and 41 low;
87 have an upstream fixed version and 405 do not. It found zero secrets.

Two of the high findings are non-executable metadata references rather than
installed vulnerable code. pip 26.2 intentionally carries its own dependency
SBOM at `pip/_vendor/bom.cdx.json`; that document names msgpack 1.1.2 and
setuptools 70.3.0, so BuildKit and Trivy report one high for each. The same SPDX
attestation separately identifies the actual installed msgpack 1.2.1 and
setuptools 83.0.0. The Docker build imports those versions exactly and proves
the standalone base interpreter has neither package. The old SBOM records are
retained as attribution evidence rather than deleted to hide scanner output.
They correspond to [GHSA-6v7p-g79w-8964](https://github.com/advisories/GHSA-6v7p-g79w-8964),
[GHSA-5rjg-fvgr-3xxf](https://github.com/advisories/GHSA-5rjg-fvgr-3xxf), and
[GHSA-h35f-9h28-mq5c](https://github.com/advisories/GHSA-h35f-9h28-mq5c).

The 51 executable-code high findings are: GitPython 3.1.45 (13 fixed), Pillow 11.3.0 (13
fixed), OpenEXR 3.4.2 (5 fixed), pyasn1 0.6.1 (5 fixed), urllib3 2.5.0 (4
fixed), PyJWT 2.10.1 (2 fixed), cryptography 46.0.3 (2 fixed), diffusers
0.35.2 (2 fixed), transformers 4.51.3 (2 fixed), protobuf 6.33.0 (1 fixed), pyarrow 22.0.0 (1
fixed), and FlashAttention 2.7.3 (1 without a fixed version).

The 50 real fixable highs remain in NVIDIA's immutable, hash-locked inference
closure; replacing that many intertwined findings across 12 packages without an upstream lock
would weaken reproducibility and inference compatibility. They are therefore a
documented residual release risk, not a silent waiver, and should be addressed
by advancing the complete upstream lock after compatibility testing. The sole
unfixed high, CVE-2026-31253, concerns unsafe pickle checkpoint loading in a
FlashAttention training helper. This image uses the FlashAttention inference
kernel and operator-authorized NVIDIA gated weights; it does not invoke that
training helper or accept untrusted checkpoint uploads. The installed code is
still present, so the finding remains a security risk for any out-of-scope use.

Trivy emitted 8,057 license records: 2 critical, 844 high, 18 medium, 6,610
low, and 583 unknown. Both critical records are the same attribution copied in
PyArrow's license bundle and Cosmos's `ATTRIBUTIONS.md`: Apache Arrow 22 states
that vendored `whereami` is dual licensed under MIT and WTFPLv2, and reproduces
the MIT grant. The MIT option is selected; the duplicate WTFPL detections are
not an ambiguity. The 852 high records are reciprocal/restricted classifications,
not missing notices. Their package copyright/license files remain in the image,
and redistribution retains corresponding-source, notice, and LGPL relinking
obligations. The authoritative dual-license text is Apache Arrow's tagged
[`LICENSE.txt`](https://raw.githubusercontent.com/apache/arrow/apache-arrow-22.0.0/LICENSE.txt).

## Functional release gates

Four superseded private candidates were rejected by the real GPU/workflow gates rather
than waived: the first revealed that the SigLIP/scientific-Python import path reaches
`numpy.testing`, whose pinned NumPy 2.2.6
`numpy/testing/_private/utils.py` unconditionally imports `pd_NA` from
`numpy/_core/tests/_natype.py`. Test-tree pruning therefore retains that one small
BSD-licensed source helper, and the Docker build's SigLIP/SciPy import is the checked
runtime assertion for it. The second candidate reached the prompt guardrail and
showed that NLTK PathSec needed the dedicated mounted model-cache root; the
third passed both guardrails and reached the first diffusion step, where
Transformer Engine exposed the CUDA runtime image's missing unversioned
`libcudart.so` linker name. A fourth passed raw inference but failed the managed
workflow after source setup because SkyPilot had masked upstream's runtime
`uvx` dependency. The final image ships `uv` and `uvx` from the same
digest-pinned, Apache-2.0/MIT image used by NVIDIA upstream. Each correction has
a build-time assertion. Only the final registry digest and successful live
results below qualify.

The catalog's serverless golden declaration remains H100. The live Kubernetes
cluster had no H100 node, so the GPU-selection procedure selected the available
96 GiB RTX PRO 6000 Blackwell: it exceeds the measured 62,822 MiB requirement
and exercises the image's CUDA 12.8 / compute-capability 12.0 FlashAttention
path. Managed job 362 used the exact registry child above and passed as UID
1000 with torch 2.7.0+cu128, `uvx 0.8.12`, CUDA available, and a real
FlashAttention kernel. Removing `HF_TOKEN` exited 78 before a download and the
logs contained no credential. With the token injected only at run time, the
repository-authored fixture drove `examples/inference.py` for four diffusion
steps in 341.127 seconds. The output was a decodable 1280x720 MP4 with 93 frames,
5.8125 seconds duration, 5,537,600 bytes, and SHA-256
`b85870845a816dda3d62bd906f8bc58250b1b0d1a90c90054865c30ac1491eb7`.
The mounted cache held 17,436 files / 36,448,258,402 bytes; no model or token
entered the image.

Three deliberately conservative packaging details remain coupled to this pinned
dependency closure. The forbidden-payload guard treats every unreviewed `.pth` as
checkpoint-like and permits only the two known venv startup hooks; adding another
`.pth` requires an explicit artifact, runtime, and license review plus a narrow path
allowlist update. It must not be accepted by inspecting file contents. The global
`/usr/local/bin/python3` venv shim is also intentional: SkyPilot and
orchestrator-supplied commands invoke `python3`, while `/usr/bin/python3` remains for
system tooling. Finally, `smoke_functional.sh` safely captures fixture JSON from
stdout because the generator emits exactly one JSON object after success and its
ffmpeg subprocess uses `-hide_banner -loglevel error`, which keeps diagnostics on
stderr; a failed generator terminates the shell before JSON parsing.

Managed workflow job 363 (`npa-wf-gpu-cosmos2-transfer-ca6a03fa`) then used
that same child digest through the repository's live submit path. The actual
Kubernetes pod reported the qualified digest as its image ID and remained alive
while SkyPilot completed its non-root SSH, rsync, and sudo bootstrap. The
recorded interpreter was `/opt/cosmos/cosmos-transfer2.5/.venv/bin/python`.
`workbench.cosmos2.transfer_execute` fetched gated weights only into the mounted
runtime cache and completed the production 35-step edge-conditioned inference.
SkyPilot reported `SUCCEEDED` after 13 minutes 7 seconds of job execution (14
minutes 19 seconds including managed-job setup).

The result was downloaded back from object storage and independently inspected.
Its `npa.cosmos2.transfer.v1` manifest reported `status: executed`, a distinct
non-control generated video, and eight uploaded PNG frames. `ffprobe` decoded
the H.264 output as 1280x720 at 16 fps with 93 frames and 5.8125 seconds of video.
The file is 5,635,256 bytes with SHA-256
`5c245706db69ece6e7c300d4c215eea4834cd02a3ed5b8428b735dba449f8da8`.
The manifest's byte count, generated-video URI, clip, conditioning metadata,
and frame inventory agreed with the downloaded artifacts.

## Publication boundary

This run pushes only the additive private source-registry tag above. It does not
publish or copy any image to GHCR, alter package visibility, or dispatch the
`Publish public images` workflow. A later public publication must independently
revalidate this exact digest, satisfy the runtime/reciprocal-license obligations,
and accept the documented vulnerability risk.
