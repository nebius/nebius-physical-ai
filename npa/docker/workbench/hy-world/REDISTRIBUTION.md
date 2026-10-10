# HY-World 2.0 runtime-fetch redistribution record

Review date: 2026-10-04. This is an engineering packaging record, not legal
advice. The image is an **unbuilt, publication-quarantined candidate**. It has
no OCI digest, payload scan, or GPU capability result yet.

## Official identities and terms

- Source: [`Tencent-Hunyuan/HY-World-2.0@df9988efb87bfc0f4947eb3889411cf957478b06`](https://github.com/Tencent-Hunyuan/HY-World-2.0/tree/df9988efb87bfc0f4947eb3889411cf957478b06).
- Tencent terms: [`License.txt` at that source ref](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/df9988efb87bfc0f4947eb3889411cf957478b06/License.txt), the Tencent HY-WORLD 2.0 Community License Agreement, dated 2026-04-15.
- Tencent model repository: `tencent/HY-World-2.0@d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0`.
- WorldStereo component: `hanshanxue/WorldStereo@ac2ad97ecb043fe80c2f19cd1898006becb9d66e` (model card: MIT).
- Image-conditioning backend: `Qwen/Qwen-Image-Edit-2509@d3968ef930e841f4c73640fb8afa3b306a78167e` (model card: Apache-2.0).
- External trajectory VLM: `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (model card: Apache-2.0), served by an operator-owned vLLM deployment.
- Trajectory dependencies: ZIM `naver-iv/zim-anything-vitl@667e2d7c233f6f1cacd12ccc64bdf6cc7b5aa16d` (CC-BY-4.0), Grounding DINO `IDEA-Research/grounding-dino-tiny@a2bb814dd30d776dcf7e30523b00659f4f141c71` (Apache-2.0), MoGe `Ruicheng/moge-2-vitl-normal@cb0e8bbd6b1e243589717c78e750b1ba4c093acf` (MIT), and Uni3C `ewrfcas/Uni3C@fca895f7fb454f4dfed43956cab09180d0946318` (Apache-2.0).
- Meta SAM 3: `facebook/sam3@3c879f39826c281e95690f02c7821c4de09afae7` (model card: `other`, gated; the operator must complete Meta/Hugging Face approval separately).
- Runtime native dependencies: `facebookresearch/pytorch3d@88e182f989c80836f4bd744e0d9cb1852762ce01` (BSD-3-Clause) and `flash-attn==2.8.3` (BSD-3-Clause). The upstream source's custom gsplat and NavMesh extensions are compiled only in the operator cache from the pinned HY-World source and recursively pinned Recast submodule.

The Tencent agreement defines HY-WORLD 2.0 to include its models, trained
weights, software/algorithms, and inference/training code. It grants no licence
for EU, UK, or South Korea use, and an entity with more than 1,000,000 MAU at
release needs a Tencent licence. Section 5(b) forbids using HY-World works or
Outputs to train or improve another AI model. These are operator facts; an
`ACCEPT=YES` environment variable would establish none of them and is not used.

## Six boundaries

| Boundary | Delivery and decision |
| --- | --- |
| Tencent source | Not baked. `hy-world-runtime ensure` fetches the exact Git commit and recursive gitlinks into the operator cache only after the private terms preflight. |
| Baked runtime | Digest-pinned CUDA 12.8 development bootstrap, compiler toolchain, ffmpeg, git, uv, Hugging Face client, and NPA-owned validation/report scripts. It contains no Python environment, model, source tree, extension binary, cache, input, or output. The component closure has not yet received a built-byte review, so the candidate is `unvalidated`, not publicly redistributable. |
| Weights | Tencent HY-World and WorldStereo, Qwen Image, ZIM, Grounding DINO, SAM 3, MoGe, and Uni3C are never baked. Runtime fetch pins exact Hub revisions and switches Hub access offline afterwards. |
| Data/assets | No input, sample scene, panorama, mesh, or camera data is baked. The operator supplies one image through the declared private S3 input object. |
| Caches | Source, virtualenv, checkpoint, and Hub blobs are operator-owned runtime bytes beneath the existing model-cache mount. They never enter OCI layers or output metadata. |
| Outputs | Generated panoramas, videos, 3DGS PLY/SPZ assets, mesh if emitted, cameras, and RRD remain run artifacts. They may be inspected/rendered, but this workflow has no trainer or robot-policy consumer because Tencent Section 5(b) forbids using Outputs to improve another AI model. |

## Delivery shape and limits

The image is a neutral bootstrap only. It does not make Tencent content publicly
redistributable and must not be pushed to the public registry until the exact
built artifact passes source/license closure, secret, layer/history, SBOM, and
payload-absence checks. The runtime does not start a service. It consumes an
operator-provided private vLLM endpoint for WorldNav and does not create or keep
one running.

The accepted future path is **image-conditioned world generation** using the
released lighter Qwen-Image-Edit HY-Pano backend, then WorldNav, WorldStereo,
GS-data preparation, and 3DGS training. The workflow intentionally does not
claim text-to-panorama/text-to-world, a WorldMirror-only reconstruction, metric
scale, mesh collision validity, or robot-policy simulation.

Upstream's dependency files do not provide a complete hash lock for every
ordinary Python package. This candidate replaces the mutable PyTorch3D source
line with an immutable commit and pins FlashAttention, then records the full
resolved package inventory and recursive submodule state from the actual run.
That is run provenance, not a claim of a fully hash-locked third-party closure.
