# Public Workbench Container Images

These public images are published under:

```text
ghcr.io/nebius/nebius-physical-ai/<image>:<tag>
```

External consumers can pull them anonymously from GHCR:

```bash
export NPA_REGISTRY=ghcr.io/nebius/nebius-physical-ai
docker pull "${NPA_REGISTRY}/npa-retargeting:0.1.1"
```

The catalog was verified against the public GHCR tag and OCI manifest APIs on
2026-08-11. Every repository and tag below resolved. **Built** is the UTC build
date of the newest listed variant; reproducible images that intentionally zero
their OCI `created` field use the timestamp in the immutable tag and
`npa.build_ts` label.

Prefer a full timestamped tag when selecting a hardware-specific variant. OCI
tags can be moved, so resolve and retain the manifest digest as well when strict
reproducibility is required.

Rows are ordered by **Built** date, then by friendly name.

| Friendly name | Image (`ghcr.io/nebius/nebius-physical-ai/...`) | Published tag(s) | Built | What it does |
| --- | --- | --- | --- | --- |
| SONIC Retargeting 0.1.1 | `npa-retargeting` | `0.1.1` | 2026-06-16 | CPU-only motion retargeting and motion-library conversion feeding SONIC locomotion training. A slim `python:3.11` image for the inexpensive preprocessing stage before GPU work. |
| Rerun 0.31.4 | `npa-rerun-viewer` | `0.31.4` | 2026-07-01 | Rerun viewer/server on port 9090 for `.rrd` robotics traces produced by workflow stages. Uses `python:3.11-slim` and runs as `nobody`. |
| LeRobot Policy Server 0.1.1 | `npa-lerobot-policy` | `0.1.1` | 2026-07-10 | Serves a trained LeRobot policy over HTTP for closed-loop inference (default `lerobot/diffusion_pusht`). This is the BYO-policy contract endpoint called by other workflow stages. |
| BDD100K Detection Training | `npa-detection-training` | `bdd100k-golden-eval-smoke-20260614T210000Z` | 2026-07-22 | Object-detector train/eval service on port 8790 with torchvision detectors and COCO metrics. It provides the re-label and measurement stage in the data-factory loop. |
| Lichtblick 1.26.0 | `npa-lichtblick` | `1.26.0` | 2026-07-23 | Fully open-source (MPL-2.0), Foxglove-compatible MCAP/ROS log viewer served by Caddy on port 8080. No account or proprietary component is required. |
| FiftyOne 1.15.0.post1 (Voxel51) | `npa-fiftyone` | `1.15.0.post1` | 2026-08-13 | Dataset curation and visualization UI on port 5151, including uniqueness, similarity, and embedding visualization. Bundles a `mongod` binary so FiftyOne can launch its own metadata database. |
| GR00T N1.7-3B | `npa-groot` | `0.1.0` | 2026-08-01 | NVIDIA Isaac-GR00T humanoid foundation-model inference using `nvidia/GR00T-N1.7-3B`; weights are pulled at runtime with the operator's Hugging Face token. GR00T inference itself does not require Isaac or EULA acceptance. |
| Isaac Lab 2.3.2 (Isaac Sim 5.1) | `npa-isaac-lab` | `2.3.2.post1` | 2026-08-01 | Isaac Lab RL simulation. Contains no NVIDIA Isaac bytes: Isaac Sim and Isaac Lab are fetched from `pypi.nvidia.com` on first use under the operator's EULA. Isaac startup requires `OMNI_KIT_ACCEPT_EULA=YES` and `ISAACSIM_ACCEPT_EULA=YES`; expect an approximately 4.5 GB first-run download. |
| Cosmos 1.0 Diffusion 7B (Predict) | `npa-cosmos` | `1.0.9`, `cu128-torch27-sm100-1.0.9-20260803T002017Z` | 2026-08-03 | Cosmos world-model generation with `Cosmos-1.0-Diffusion-7B-Text2World`, plus the default self-hosted VLM image for workflows. Uses Torch 2.7 and CUDA 12.8 with flash-attn, NATTEN, and Transformer Engine. |
| Cosmos Curator 0.1.2 | `npa-cosmos-curate` | `0.1.2-skypilot-v1-20260813T164700Z` | 2026-08-13 | Runs real `cosmos-curate` stages in process: download, fixed-stride extraction, clip transcode, motion-vector decode, motion filtering, and clip writing. GPU-stage models are fetched at runtime with the operator's Hugging Face token. |
| Cosmos Evaluator 0.1.2 | `npa-cosmos-evaluator` | `0.1.2-skypilot-v1-20260813T164700Z` | 2026-08-13 | Runs the upstream `HallucinationProcessor` quality gate on generated video using classical computer vision and no weights. Attribute verification calls an OpenAI-compatible endpoint; the LFS/EULA-gated obstacle checker is deliberately not fetched. |
| Cosmos Reason 2 / Predict 2.5 (3.0.1) | `npa-cosmos3-reason` | `3.0.1-genuine-sm120`, `cuda13-b300-3.0.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | 2026-08-03 | VLM reasoning over video/images with `Cosmos-Reason2-8B` or `Cosmos-Reason2-2B`, serving as a judge/critic stage. Also wires Predict 2.5, Transfer 2.5, and Cosmos-Guardrail1 model IDs on a Blackwell-capable CUDA 13 base. |
| Cosmos Transfer 2.5 | `npa-cosmos2-transfer` | `2.5.1-skypilot-ready-20260801T053000Z` | 2026-08-03 | Cosmos Transfer 2.5 Sim2Real video augmentation, built from source at an immutable commit with hash-locked dependencies. Gated weights are fetched at runtime with `HF_TOKEN`; baked-byte scans are a release gate. |
| Foxglove Embed SDK 0.58.0 | `npa-foxglove-embed` | `0.58.0` | 2026-08-03 | Static host for the pinned `@foxglove/embed` browser SDK (MIT) and shared NPA glue module used by the agent UI, on port 8099. Serves operator-mounted MCAP/bag recordings with CORS and byte ranges; the Foxglove app is not redistributed. |
| Genesis 0.4.6 | `npa-genesis` | `0.4.6`, `cuda13-b300-0.4.6-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | 2026-08-03 | Genesis physics simulator for interactive simulation and development. It is the base image for the Sim2Real family: environment generation, evaluation, policies, and VLM-RL. |
| LanceDB 0.30.3 + CLIP | `npa-lancedb` | `0.30.3`, `cuda13-b300-0.30.3-sm80-sm90-sm100-sm103-sm120-20260803T031514Z` | 2026-08-03 | CLIP embedding and LanceDB vector service on port 8686: the query index behind dataset-of-record search. It uses a thin FastAPI layer on the shared CUDA/PyTorch base. |
| LeRobot 0.5.1 | `npa-lerobot` | `0.5.1`, `cuda13-b300-0.5.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | 2026-08-03 | Hugging Face LeRobot training/evaluation service on port 8080 for manipulation policies. Includes CUDA and MuJoCo/EGL headless rendering; checkpoints and job state live on mounted volumes. |
| LeRobot VLM-RL 0.1.1 | `npa-lerobot-vlm-rl` | `0.1.1`, `cuda13-b300-0.1.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | 2026-08-03 | RL loop in which a VLM supplies reward or shaping signals for LeRobot policies. It is built on the Genesis image so simulation and policy execution share one container. |
| Sim2Real Control 0.1.2 | `npa-sim2real-control` | `0.1.2` | 2026-08-11 | Private/runtime CPU stage adapter for the compositional Sim2Real workflow. It bakes exact NPA source into site-packages path discovery, a pinned S3 dependency closure, and the non-root SkyPilot Kubernetes bootstrap prerequisites on digest-pinned Python slim with snapshot-pinned Debian packages; it contains no CUDA, simulator, model, or trainer runtime. Built from `sim2real-control/Dockerfile`. |
| Sim2Real EnvGen 0.1.2 | `npa-envgen` | `0.1.2`, `cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | 2026-08-03 | Generates randomized Sim2Real environments and scenes on the Genesis base. Exact-source workflow builds also bake the snapshot-pinned non-root SkyPilot Kubernetes bootstrap closure (`sudo`, SSH, and rsync); this is required before a standard workflow task can start. It is the parent image for BYO policy containers and is built from `sim2real-envgen/Dockerfile`. |
| Sim2Real Loop Eval 0.1.3 | `npa-loop-eval` | `0.1.3-genuine-sm120`, `cuda13-b300-0.1.3-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | 2026-08-03 | Batched closed-loop policy evaluation in Genesis (default 16 environments and 240 steps), providing the scoring stage of the Sim2Real loop. Exact-source workflow builds bake the same snapshot-pinned non-root SkyPilot Kubernetes bootstrap closure as EnvGen so Stage 14 can start without a privileged or moving bootstrap image. Built from `sim2real-eval/Dockerfile`; the tool key is `loop-eval`. |
| Sim2Real Reference Policy 0.1.2 | `npa-reference-policy` | `0.1.2`, `cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | 2026-08-03 | Reference BYO-compatible Sim2Real action policy and worked example of the policy-container contract. Includes the policy functional smoke for comparison with custom images. |
| SONIC (GR00T-WholeBodyControl) | `npa-sonic` | `0.1.2` (L40S), `cuda13-b300-0.1.2-k8s-runtime-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | 2026-08-03 | Whole-body humanoid locomotion training and evaluation using `gear_sonic` (Apache-2.0 at a pinned commit). Isaac is fetched on first use; NVIDIA Open Model License weights are downloaded at runtime and are never baked into the image. |
| Cosmos 3 (`cosmos-framework` 1.2.2) | `npa-cosmos3` | `1.2.2-cu130`, `1.2.2-cu130-r2` | 2026-08-08 | Cosmos 3 omni-model generation: text-to-image, image-to-image, text-to-video, image-to-video, and video-to-video. Contains OpenMDW-1.1 source and a CUDA 13 venv only; checkpoints, Wan VAE, and guardrails download at runtime. |
| Wan 2.2 TI2V-5B | `npa-wan2-2` | historical accepted tag: `2.2-ti2v5b-rtfetch-cu128-20260809T011658Z-r7`; current unpublished closure: Torch 2.13.0/CUDA 13.0/NCCL 2.29.7 | 2026-08-09 | Wan 2.2 text/image-to-video generation from Apache-2.0 source on an OSS dependency base. CUDA PyTorch, `nvidia-*` wheels, models, and tokenizers are runtime-fetched after operator acceptance. The listed tag carries the prior live evidence; the security-fixed closure requires new single- and four-GPU qualification before any tag publication or promotion. |

## Intentionally not published as separate images

- **`npa-cosmos3-serving`** is `restricted` and build-your-own only. Its pinned
  vLLM-Omni base embeds a runtime under NVIDIA's Deep Learning Container License;
  the thin wrapper and anonymous GHCR distribution do not establish that
  license's derived-distribution conditions. Operators build it into their own
  registry; see [Cosmos3-Super serving](cosmos3-super-serving.md).
- **`npa-sonic-mujoco`** is a SONIC variant, not a separate public-publish tool.
  It ships through the `sonic` tool and SONIC image manifest rather than getting
  an independent row in the public publishing plan.

## Verification scope

The registry verification confirms repository visibility, exact tag spelling,
manifest resolution, platform metadata, selected OCI labels, exposed ports,
entrypoints, and build timestamps. It is descriptive, not a packaging-policy
pass: for example, `npa-sonic:0.1.2` declares the `ubuntu` OCI user, while the
timestamped Kubernetes pin declares `root` because its build sets
`NPA_RUNTIME_USER=root`. Capability descriptions are cross-checked against the
corresponding Dockerfiles, packaging contracts, golden evaluations, and workflow
integrations on `main`.

This catalog does not imply that every image supports every GPU. Use the
[B300 validation matrix](../b300-validation-matrix.md) when choosing a
hardware-specific variant, and use the
[container packaging contract](container-packaging.md) for security and
redistribution requirements.
