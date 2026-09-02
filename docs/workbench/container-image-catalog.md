# Public Workbench Container Image Catalog

Repository-selected runtime images use the public mirror by default:

```text
ghcr.io/nebius/nebius-physical-ai/<image>:<tag>
```

Consumers can pull published entries anonymously from GHCR without configuring
a registry in NPA:

```bash
docker pull ghcr.io/nebius/nebius-physical-ai/npa-retargeting:0.1.1
```

`NPA_REGISTRY` remains a build/BYOF destination for private or locally modified
images. It and existing saved `container_registry` values do not repoint these
repository-owned runtime defaults; select custom bytes with a complete image
reference or an explicit workflow `--registry`.

The accepted-release manifest was verified against the public GHCR tag and OCI
manifest APIs on 2026-09-02. All 31 recorded release digests resolved
anonymously. **Built** is the UTC build date of the newest listed variant;
reproducible images that
intentionally zero their OCI `created` field use the timestamp in the immutable
tag and `npa.build_ts` label.

Prefer a full timestamped tag when selecting a hardware-specific variant. OCI
tags can be moved, so resolve and retain the manifest digest as well when strict
reproducibility is required.

Rows are ordered by **Built** date, then by friendly name.

## 2026-09-02 private-registry isolation audit

All 31 accepted release tags and recorded digests resolved through anonymous
GHCR manifest requests. The repository-wide reference audit found that ambient
`NPA_REGISTRY` and legacy saved registry values could nevertheless redirect
repository-owned workload defaults to operator registries, where missing pull
permission surfaced as HTTP 403 and Kubernetes `ImagePullBackOff`. Runtime
defaults are now isolated from those build/BYOF settings; explicit image and
workflow `--registry` selections remain available for intentional custom bytes.

Third-party authoritative references were classified separately. NVIDIA NRE
remains on its anonymously pullable NGC `nre-ga` channel, the CUDA vector-add
health image remains on its anonymously pullable upstream NGC channel, and
Docker Hub base/runtime images remain upstream. None is an NPA image suitable
for blind relocation to GHCR.

## 2026-08-29 main publication audit

The main-branch publishing plan, packaging contract, accepted-release manifest,
and catalog were compared with GHCR without credentials. All 31 exact plan tags
resolved anonymously, so no image build or registry write was required. The
accepted-release manifest's stale LeIsaac pending marker was replaced with its
already-published exact digest. That digest matched the earlier publication
record, ran as the non-root `ubuntu` user, and passed the full layer/history
payload scanner across 98,792 filesystem entries with zero restricted-payload
findings. Public contract entries outside the tool publishing plan remain
intentionally excluded rather than being treated as missing releases.

## 2026-08-19 main publication audit

The main-branch publishing plan was compared with GHCR without credentials. Its
24 existing tags were left unchanged, and the two absent tags were published:

- `npa-alpamayo2-super:0.1.0-cu128` — OCI index
  `sha256:2164450f8baf57d8798f64063ea27bf11611f5b695c467de0c2e319e3134ebd5`,
  containing the accepted `linux/amd64` image and its bound attestation.
- `npa-leisaac:0.4.0-20260817T231825Z` — `linux/amd64` manifest
  `sha256:82069eb74a18a88f77ad3149b6c5ed220c4eed33b1d550c26d361947805e8280`,
  rebuilt from main and full-layer scanned before publication.

An independent anonymous verification then resolved all 26 exact plan tags.
LTX-2.5 remains excluded because its required GPU validation is not complete;
`cosmos3-serving` and `sonic-mujoco` remain excluded by the redistribution
guard.

## 2026-08-22 Content Agents publication

`npa-content-agents:0.5.2-npa2` was published as OCI index
`sha256:c64aaf6201bdaa013f9d16e8497290cf166907932f036297d7abaa430cbad7db`,
containing one `linux/amd64` manifest and one bound attestation manifest. An
independent unauthenticated manifest and config read resolved that exact digest,
the `ubuntu` user, and the `public` / `runtime-fetch` packaging labels.

The exact published bytes passed the Content Agents scanner across three nested
archives with zero findings, the general layer/history scanner across 28,471
entries with zero payload or history hits, and Trivy with zero critical
vulnerabilities and zero secrets. A digest-pinned RTX PRO 6000 workflow produced
6 material, 6 physics, and 1 validation render; upstream validation passed, all
rigid-physics checks were non-null, and both the USD and USDZ reopened
independently. These measurements establish built, payload-clean, GPU-validated,
published, and anonymously pullable status for this exact digest only.

| Friendly name | Image (`ghcr.io/nebius/nebius-physical-ai/...`) | Published tag(s) | Built | What it does |
| --- | --- | --- | --- | --- |
| Alpamayo 2 Super 34B | `npa-alpamayo2-super` | `0.1.0-cu128` | 2026-08-18 | Real surround-view VLA trajectory inference through NVIDIA's Apache-2.0 source. OpenMDW-1.1 weights and the separately gated/non-transferable PhysicalAI-AV sample data are fetched only at runtime under the operator's Hugging Face identity. The payload-clean image and real workflow were validated independently on B200 and RTX PRO 6000. See the [operator guide](alpamayo2-super.md). |
| NVIDIA Content Agents 0.5.2 | `npa-content-agents` | `0.5.2-npa2` | 2026-08-22 | Public rigid-object material/physics/validation adapter containing Apache-2.0 source and zero OVRTX payload. Exact OVRTX 0.3.0.312915 is fetched directly from NVIDIA into the operator runtime cache. The exact public digest passed byte/layer scanning, anonymous resolution, and a real RTX PRO 6000 workflow. See [Content Agents](content-agents.md). |
| SONIC Retargeting 0.1.1 | `npa-retargeting` | `0.1.1` | 2026-06-16 | CPU-only motion retargeting and motion-library conversion feeding SONIC locomotion training. A slim `python:3.11` image for the inexpensive preprocessing stage before GPU work. |
| Rerun 0.31.4 | `npa-rerun-viewer` | `0.31.4` | 2026-07-01 | Published Rerun viewer/server on port 9090 for `.rrd` robotics traces. Current source defines an as-yet-unpublished non-root `ubuntu` SkyPilot worker with ports 9090/9876 and an exact-source Sim2Real Stage 14 runtime; it bakes no models, datasets, credentials, or runtime caches. |
| LeRobot Policy Server 0.1.1 | `npa-lerobot-policy` | `0.1.1` | 2026-07-10 | Serves a trained LeRobot policy over HTTP for closed-loop inference (default `lerobot/diffusion_pusht`). This is the BYO-policy contract endpoint called by other workflow stages. |
| BDD100K Detection Training | `npa-detection-training` | `bdd100k-golden-eval-smoke-20260614T210000Z` | 2026-07-22 | Object-detector train/eval service on port 8790 with torchvision detectors and COCO metrics. It provides the re-label and measurement stage in the data-factory loop. |
| Lichtblick 1.26.0 | `npa-lichtblick` | `1.26.0` | 2026-07-23 | Fully open-source (MPL-2.0), Foxglove-compatible MCAP/ROS log viewer served by Caddy on port 8080. No account or proprietary component is required. |
| GR00T N1.7-3B | `npa-groot` | `0.1.0` | 2026-08-01 | NVIDIA Isaac-GR00T humanoid foundation-model inference using public `nvidia/GR00T-N1.7-3B`; weights are pulled anonymously at runtime by default, with an optional Hugging Face token for rate limits or private overrides. GR00T inference itself does not require Isaac or EULA acceptance. |
| Isaac Lab 3.0 beta 2 patch 1 (Isaac Sim 6.0.1) | `npa-isaac-lab` | `3.0.0b2.post1` | 2026-08-25 | Payload-clean Isaac Lab RL simulation image: every proprietary NVIDIA runtime wheel is hash-pinned and fetched from `pypi.nvidia.com` only after the operator's run-scoped EULA acceptance. The exact public digest passed layer and flattened payload scans, critical-CVE/secret/license scanning, anonymous resolution, a real RTX PRO 6000 four-variant RSL-RL workflow, and the paired generation 2 benchmark. Upstream still labels this release beta; see [Isaac Lab 3](isaac-lab-3.md) for the measured cold-start tradeoff. |
| Cosmos 1.0 Diffusion 7B (Predict) | `npa-cosmos` | `1.0.9`, `cu128-torch27-sm100-1.0.9-20260803T002017Z` | 2026-08-03 | Cosmos world-model generation with `Cosmos-1.0-Diffusion-7B-Text2World`, plus the default self-hosted VLM image for workflows. Uses Torch 2.7 and CUDA 12.8 with flash-attn, NATTEN, and Transformer Engine. |
| Cosmos Reason 2 / Predict 2.5 (3.0.1) | `npa-cosmos3-reason` | `3.0.1-genuine-sm120`, `cuda13-b300-3.0.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | 2026-08-03 | VLM reasoning over video/images with `Cosmos-Reason2-8B` or `Cosmos-Reason2-2B`, serving as a judge/critic stage. Also wires Predict 2.5, Transfer 2.5, and Cosmos-Guardrail1 model IDs on a Blackwell-capable CUDA 13 base. |
| Cosmos Transfer 2.5 | `npa-cosmos2-transfer` | `2.5.1-skypilot-ready-20260801T053000Z` | 2026-08-03 | Cosmos Transfer 2.5 Sim2Real video augmentation, built from source at an immutable commit with hash-locked dependencies. Gated weights are fetched at runtime with `HF_TOKEN`; baked-byte scans are a release gate. |
| Foxglove Embed SDK 0.58.0 | `npa-foxglove-embed` | `0.58.0` | 2026-08-03 | Static host for the pinned `@foxglove/embed` browser SDK (MIT) and shared NPA glue module used by the agent UI, on port 8099. Serves operator-mounted MCAP/bag recordings with CORS and byte ranges; the Foxglove app is not redistributed. |
| Genesis 0.4.6 | `npa-genesis` | `0.4.6`, `cuda13-b300-0.4.6-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | 2026-08-03 | Genesis physics simulator for interactive simulation and development. It is the base image for the Sim2Real family: environment generation, evaluation, policies, and VLM-RL. |
| LanceDB 0.30.3 + CLIP | `npa-lancedb` | `0.30.3`, `cuda13-b300-0.30.3-sm80-sm90-sm100-sm103-sm120-20260803T031514Z` | 2026-08-03 | CLIP embedding and LanceDB vector service on port 8686: the query index behind dataset-of-record search. It uses a thin FastAPI layer on the shared CUDA/PyTorch base. |
| LeRobot 0.5.1 | `npa-lerobot` | `0.5.1`, `cuda13-b300-0.5.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | 2026-08-03 | Hugging Face LeRobot training/evaluation service on port 8080 for manipulation policies. Includes CUDA and MuJoCo/EGL headless rendering; checkpoints and job state live on mounted volumes. |
| LTX-2.5 2.5 | `npa-ltx2` | `2.5-rtfetch-20260817` | 2026-08-17 | Lightricks LTX-2.5 text-to-video, shipped with zero Lightricks bytes: source and gated weights are operator-entitled runtime fetches. The accepted digest passed the exact-layer payload scan, entitlement refusal, and real GPU text-to-video plus decoded-MP4 validation. |
| LeRobot VLM-RL 0.1.1 | `npa-lerobot-vlm-rl` | `0.1.1`, `cuda13-b300-0.1.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | 2026-08-03 | RL loop in which a VLM supplies reward or shaping signals for LeRobot policies. It is built on the Genesis image so simulation and policy execution share one container. |
| Sim2Real EnvGen 0.1.2 | `npa-envgen` | `0.1.2`, `cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | 2026-08-03 | Generates randomized Sim2Real environments and scenes on the Genesis base. Exact-source workflow builds also bake the snapshot-pinned non-root SkyPilot Kubernetes bootstrap closure (`sudo`, SSH, and rsync); this is required before a standard workflow task can start. It is the parent image for BYO policy containers and is built from `sim2real-envgen/Dockerfile`. |
| Sim2Real Loop Eval 0.1.3 | `npa-loop-eval` | `0.1.3-genuine-sm120`, `cuda13-b300-0.1.3-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | 2026-08-03 | Batched closed-loop policy evaluation in Genesis (default 16 environments and 240 steps), providing the scoring stage of the Sim2Real loop. Exact-source workflow builds bake the same snapshot-pinned non-root SkyPilot Kubernetes bootstrap closure as EnvGen so Stage 14 can start without a privileged or moving bootstrap image. Built from `sim2real-eval/Dockerfile`; the tool key is `loop-eval`. |
| Sim2Real Reference Policy 0.1.2 | `npa-reference-policy` | `0.1.2`, `cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | 2026-08-03 | Reference BYO-compatible Sim2Real action policy and worked example of the policy-container contract. Includes the policy functional smoke for comparison with custom images. |
| SONIC (GR00T-WholeBodyControl) | `npa-sonic` | `cuda13-b300-0.1.2-k8s-runtime-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` (active RTX PRO Kubernetes); `0.1.2` (quarantined L40S) | 2026-08-03 | Whole-body humanoid locomotion training and evaluation using `gear_sonic` (Apache-2.0 at a pinned commit). The public active image runtime-fetches Isaac and requires GPU Operator driver mounts. The old L40S and combined H100/H200 MuJoCo images are restricted and rejected; compute-only serverless use requires a separately validated custom image. |
| SONIC MuJoCo | `npa-sonic-mujoco` | `0.2.0-runtime` | 2026-08-17 | Independently rebuilt from pinned Apache-2.0 SONIC source on a digest-pinned public Python base with a hash-locked PyTorch/MuJoCo closure. The exact accepted digest passed the real B200 Unitree G1 rollout and payload gates. |
| Cosmos3-Super serving | `npa-cosmos3-serving` | `0.2.0-oss` | 2026-08-17 | Zero-payload non-root bootstrap on a digest-pinned public Python base. The serving closure, models, and guardrails are operator runtime fetches after terms and entitlement checks; the exact accepted digest passed guarded multi-GPU service boot and real inference. |
| LeIsaac 0.4.0 | `npa-leisaac` | `0.4.0-20260817T231825Z` | 2026-08-17 | Browser teleoperation for the real upstream SO-101 LiftCube and PickOrange tasks, with secure agent-relay transport and immutable LeRobot episode recording. The image contains Apache-2.0 LeIsaac source and OSS dependencies only; Isaac Sim/Lab, NVIDIA's browser client, and task assets are runtime-fetched under the shared `ACCEPT_EULA` contract and are never baked into the image. Revalidate a digest before use. |
| Cosmos 3 (`cosmos-framework` 1.2.2) | `npa-cosmos3` | current: `1.2.2-cu130-r6`; historical provenance: `1.2.2-cu130`, `1.2.2-cu130-r2`, `1.2.2-cu130-r5` | 2026-08-21 | Cosmos 3 omni-model generation: text-to-image, image-to-image, text-to-video, image-to-video, and video-to-video. Contains OpenMDW-1.1 source and a CUDA 13 venv only; checkpoints, Wan VAE, and guardrails download at runtime. The additive r6 image retains the attested SkyPilot worker bootstrap and adds auditable source-motion-preserving PAIDF publication while keeping every raw Cosmos video. |
| Cosmos3-Nano native Ray Serve | `npa-cosmos3-ray-serve` | `ray1-cu130` | 2026-08-26 | Persistent authenticated Cosmos3-Nano serving through cosmos-framework's native `OmniModelDeployment` and `@ray.serve.batch` path. The image contains pinned OpenMDW source and the CUDA/Ray closure but no model or guardrail weights. The exact release digest passed payload/security/SBOM/provenance gates and independent guarded two-sample generation with durable S3 evidence on B200 `sm_100` and RTX PRO 6000 `sm_120`. |
| Wan 2.2 TI2V-5B | `npa-wan2-2` | `2.2-ti2v5b-rtfetch-cu130-20260817` | 2026-08-17 | Wan 2.2 text/image-to-video generation from Apache-2.0 source on an OSS dependency base. CUDA PyTorch and `nvidia-*` wheels are runtime-fetched under their upstream package terms. The accepted exact digest passed the zero-payload, SPDX/SLSA, vulnerability, and single-GPU TI2V/MP4/Rerun gates; the current four-GPU path remains deferred. |
| Cosmos Curator 0.1.2 | `npa-cosmos-curate` | `0.1.2-skypilot-v1-20260813T164700Z` | 2026-08-13 | Runs real `cosmos-curate` stages in process: download, fixed-stride extraction, clip transcode, motion-vector decode, motion filtering, and clip writing. GPU-stage models are fetched at runtime with the operator's Hugging Face token. |
| Cosmos Evaluator 0.1.2 | `npa-cosmos-evaluator` | `0.1.2-skypilot-v1-20260813T164700Z-r2` | 2026-08-21 | Runs the upstream `HallucinationProcessor` quality gate on generated video using classical computer vision and no weights. The additive r2 image exposes the deterministic ranking/holdout attribute-sample policy consumed by PAIDF. Attribute verification calls an OpenAI-compatible endpoint; the LFS/EULA-gated obstacle checker is deliberately not fetched. |
| FiftyOne 1.15.0.post1 (Voxel51) | `npa-fiftyone` | `1.15.0.post1` | 2026-08-13 | Dataset curation and visualization UI on port 5151, including uniqueness, similarity, and embedding visualization. Bundles a `mongod` binary so FiftyOne can launch its own metadata database. |

## Validated source-registry candidates pending public release

The supported worker defaults currently select additive Cosmos Transfer,
FiftyOne, and Rerun releases with the immutable SkyPilot Kubernetes bootstrap
closure. Those candidate tags are present in the maintainer source registry but
were not anonymously available in the 2026-08-17 audit. The public resolver and
publisher therefore retain the prior verified tags shown above. Moving any of
these candidates to GHCR requires the separately authorized publication workflow
and a successful unauthenticated manifest check; private availability and
redistribution eligibility are not evidence of publication.

## Intentionally not published as separate images

- **`npa-sim2real-control`** is an internal workflow artifact, not a public-mirror
  tool. Its packaging contract permits redistribution, but it has no entry in
  `CONTAINER_IMAGE_NAMES` and is therefore outside `publicly_publishable_tools()`;
  anonymous resolution of `npa-sim2real-control:0.1.2` was denied during the
  2026-08-14 audit. Eligibility is not evidence of publication.
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
pass: for example, the active runtime-fetch `npa-sonic` image declares the `ubuntu` OCI user, while the
timestamped Kubernetes pin declares `root` because its build sets
`NPA_RUNTIME_USER=root`. Capability descriptions are cross-checked against the
corresponding Dockerfiles, packaging contracts, golden evaluations, and workflow
integrations on `main`.

Maintainers should use `skills/atomic/audit-container-docs/SKILL.md` to repeat
the repository-inventory and anonymous-registry audit after container changes.

This catalog does not imply that every image supports every GPU. Use the
[B300 validation matrix](../b300-validation-matrix.md) when choosing a
hardware-specific variant, and use the
[container packaging contract](container-packaging.md) for security and
redistribution requirements.
