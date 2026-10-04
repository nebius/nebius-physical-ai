# HY-World 2.0 image-to-world candidate

This is a **candidate-only** BYOF workflow for the released Tencent
HY-World 2.0 image-conditioned path. It expands one supplied image into a
panorama, plans and renders a trajectory, builds Gaussian-splat training data,
and composes a persistent scene. The workflow then verifies the generated
WorldStereo video, upstream camera matrices, PLY/SPZ scene assets, a camera
render, and a Rerun recording made from those verified assets.

It is not an accepted image or a claim of live GPU compatibility. It does not
support the distinct text-to-panorama route, collision validity, calibrated
metric scale, or robot-policy simulation. World coordinates are the upstream
scene frame and scale is uncalibrated.

## Pinned component closure

| Component | Pinned source | Delivery |
| --- | --- | --- |
| HY-World source | `Tencent-Hunyuan/HY-World-2.0@df9988efb87bfc0f4947eb3889411cf957478b06` | Operator runtime fetch |
| HY-World weights | `tencent/HY-World-2.0@d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0` | Operator runtime fetch |
| WorldStereo | `hanshanxue/WorldStereo@ac2ad97ecb043fe80c2f19cd1898006becb9d66e` | Operator runtime fetch |
| Image-conditioned panorama | `Qwen/Qwen-Image-Edit-2509@d3968ef930e841f4c73640fb8afa3b306a78167e` | Operator runtime fetch |
| Trajectory VLM | `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Existing private operator vLLM |
| Trajectory matting | `naver-iv/zim-anything-vitl@667e2d7c233f6f1cacd12ccc64bdf6cc7b5aa16d` | Operator runtime fetch |
| Trajectory grounding | `IDEA-Research/grounding-dino-tiny@a2bb814dd30d776dcf7e30523b00659f4f141c71` | Operator runtime fetch |
| Trajectory segmentation | `facebook/sam3@3c879f39826c281e95690f02c7821c4de09afae7` | Operator runtime fetch after Meta/Hugging Face approval |
| Panorama geometry | `Ruicheng/moge-2-vitl-normal@cb0e8bbd6b1e243589717c78e750b1ba4c093acf` | Operator runtime fetch |
| WorldStereo control | `ewrfcas/Uni3C@fca895f7fb454f4dfed43956cab09180d0946318` | Operator runtime fetch |
| Native 3D renderer | `facebookresearch/pytorch3d@88e182f989c80836f4bd744e0d9cb1852762ce01` (BSD-3-Clause) | Compiled in operator runtime cache |
| Attention kernel | `flash-attn==2.8.3` (BSD-3-Clause) | Compiled in operator runtime cache |

The public bootstrap contains none of those source, model, cache, input, or
output bytes. It records the source/model/image references and SHA-256 hashes
of accepted outputs in `hy_world_image_to_world.json`.

## Operator prerequisites and terms gate

Tencent's HY-WORLD 2.0 Community License Agreement (2026-04-15) covers source,
algorithms, inference/training code, and weights. It excludes the EU, UK, and
South Korea, requires a Tencent licence for an entity with more than one
million monthly active users at release, and prohibits using HY works or their
outputs to train or improve another AI model. A private operator record must
confirm the applicable territorial/MAU condition or provide a Tencent grant
before fetching Tencent material. The runtime prints those facts; it does not
pretend an environment variable constitutes Tencent acceptance.

Use `npa workbench health preflight --checks nebius` and the exact component
access checks before building, provisioning, or submitting. Supply the image
and private vLLM endpoint only through protected operator configuration; neither
belongs in a workflow file, public artifact, or PR. The workflow creates no
persistent service.

Meta's `facebook/sam3` model card is separately gated: the operator must agree
to its own terms and share contact information with Meta through Hugging Face.
That approval and its exact revision-access probe are independent from Tencent
eligibility. ZIM (CC-BY-4.0), Grounding DINO (Apache-2.0), MoGe (MIT), and
Uni3C (Apache-2.0) remain runtime-only component bytes as well. The runtime
also pins PyTorch3D (BSD-3-Clause) and FlashAttention (BSD-3-Clause) before it
compiles the upstream `gsplat_maskgaussian` and NavMesh extensions in the
operator cache; no compiled extension is baked into the candidate image.

## Planned execution and evidence

After all prerequisite gates and an immutable candidate image digest exist,
submit `workflows/testing/byof-hy-world.yaml` with a private input image and
vLLM configuration. The released upstream instructions recommend at least four
GPUs and were tested upstream on eight H20 GPUs. NPA's planned eight-rank B200
profile is an unvalidated compatibility target, not a B200 claim.

The complete capability gate must execute the actual Qwen-image HY-Pano,
WorldNav, WorldStereo, GS-data, and `world_gs_trainer` composition sequence. A
plausible image, a reconstruction-only result, or a JSON manifest does not pass
the gate. Required retained artifacts are the generated panorama/trajectory
outputs, decoded WorldStereo and rendered-camera MP4s, `camera.json`, PLY/SPZ,
the evidence JSON, and the verified `.rrd` plus RRD manifest.
