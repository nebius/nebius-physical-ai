---
name: oss-solution-registry-onboard
description: Use when evaluating and onboarding an open-source Physical AI solution into the NPA registry/catalog with documented capabilities, BYOF packaging, smoke tests, and live Nebius validation.
---

# OSS Solution Registry Onboard

Use this skill when an agent is asked to turn a public Physical AI repository
into a registry/catalog candidate for NPA. This is stricter than generic BYOF:
the agent must discover **that solution's** real documented capabilities, test
those capabilities with solution-specific commands, and produce validation
evidence before calling the solution registry-ready.

Do **not** force capabilities into a shared taxonomy. Each OSS project has its
own APIs, assets, and hello-worlds; name and test them as the upstream project
does.

## When To Use

- Onboard a public GitHub/GitLab Physical AI repo into the NPA registry/catalog
- Promote a BYOF image from "containerized repo" to "discoverable NPA solution"
- Evaluate partner or OSS robotics, simulation, perception, policy-training,
  synthetic-data, or evaluation projects for Workbench inclusion
- Create registry metadata, workflow specs, docs, and validation evidence for an
  OSS solution

If the task is only "build and run this fork," load
`skills/workflows/byof-onboard/SKILL.md`. If the task asks for registry/catalog
admission, load this skill and then delegate build/run mechanics to BYOF.

## Required Companion Skills

Load these as needed before making decisions:

- `skills/workflows/byof-onboard/SKILL.md` — containerize, push, and run OSS repo
  workloads through BYOF.
- `skills/workflows/author-npa-workflow/SKILL.md` — write and validate
  `npa.workflow/v0.0.1` specs and `toolRef` usage.
- `skills/atomic/architecture/SKILL.md` — respect the Workbench marketplace and
  solution namespace boundary.
- `skills/atomic/testing-conventions/SKILL.md` — run validation with
  `npa/.venv/bin/python` and report exact evidence.
- `skills/atomic/solution-licensing/SKILL.md` — satisfy the License admission
  gate below: classify what the image actually bakes, decide whether it may be
  redistributed, and record it in the packaging contract.
- `skills/atomic/audit-container-docs/SKILL.md` — reconcile
  `docs/workbench/container-image-catalog.md` for every new or changed
  image-backed solution without confusing registry admission with public-mirror
  publication.
- Relevant tool skill (`skills/tools/isaac-lab`, `lerobot`, `genesis`,
  `cosmos`, `groot`, `sonic`, `fiftyone`, `lancedb`, `mjlab`, or
  `retargeting`) when the upstream repo depends on that stack.

When the solution depends on Hugging Face LeRobot, pin an explicit supported
workbench version (`0.5.1` default or `0.6.0` additional) via
`skills/tools/lerobot/SKILL.md` and
`npa/src/npa/deploy/lerobot_version_manifest.json`. Do not assume
`pip install lerobot` without extras on 0.6.0 — use
`lerobot[training,evaluation,...]==0.6.0`. Record the chosen pin in the
capability table (`gpu_or_assets` / runtime notes).

## Non-Negotiable Agent Contract

Do not invent capabilities from repo names, README badges, or marketing copy.
Before authoring registry metadata, the agent must read upstream documentation
and identify real user-facing capabilities that can be tested **for that
solution**.

For each claimed capability, record:

- capability id unique to this solution (use upstream names: env ids, config
  names, script entrypoints, dataset ids)
- upstream doc path or URL
- command, API, example, or config that demonstrates it
- required runtime profile (`ubuntu`, `isaac-lab`, custom base, service image)
- required accelerator and assets
- input artifact contract and output artifact contract
- NPA mapping: BYOF workload, Workbench tool, `toolRef`, workflow state, or docs
  only
- validation command and result
- status: `accepted` (live smoke passed), `deferred` (blocker recorded), or
  `rejected`

If a capability cannot be tested on available Nebius infrastructure, mark it
`deferred` with the precise blocker. Do not list deferred capabilities as
registry-ready.

## Capability Testing Built Into Onboarding

When **creating or onboarding any new solution**, agents must follow this
procedure. Do not skip to Docker build.

### 1. Discover this solution's native capabilities

Read upstream README/docs/examples. Produce a capability table with columns:

`capability_id`, `upstream_doc`, `command_or_api`, `runtime`, `gpu_or_assets`,
`artifact_name`, `status`.

Use the project's own vocabulary. Examples of good ids:

- ManiSkill: `pickcube_cpu_step`, `pickcube_parallel_envs`
- MuJoCo Playground: `mjx_cartpole_step`, `train_jax_ppo_cartpole_smoke`
- RoboCasa: `kitchen_task_registration`, `download_kitchen_assets_lw`
- OpenPI: `pi05_droid_jointpos_polaris_direct_infer`,
  `pi05_droid_jointpos_polaris_cross_pod_serve`,
  `pi05_droid_jointpos_polaris_lora_optimizer_smoke`,
  `pi05_droid_jointpos_polaris_heldout_evaluate`
- DROID: `rlds_config_generator_contract`, `droid_100_config_gen`

### 2. Choose a golden hello-world per accepted claim

For each capability marked for admission:

- Prefer the smallest documented upstream command that proves that claim.
- Require a JSON artifact named for the solution + capability, written to
  `$NPA_SMOKE_OUTPUT_DIR`.
- Artifact must include at least: `solution`, `capability`, and one
  capability-specific proof field (env id, reward, config name, checkpoint
  path, dataset keys, etc.).
- A single `solution-smoke` may exercise several capabilities for one image;
  write one primary artifact and optional per-capability JSON files. List every
  exercised capability in the primary artifact.

### 3. Encode into BYOF + workflow

Author `npa/workflows/workbench/npa-workflows/byof-<solution>.yaml` with:

```yaml
config:
  workload: solution-smoke
  build_command: "<pinned install>"
  smoke_command: |
    # must write $NPA_SMOKE_OUTPUT_DIR/<smoke_artifact_name>
  solution_name: "<slug>"
  capability_name: "<primary-capability-id>"
  smoke_artifact_name: "<solution>_<capability>.json"
  resource_profile_yaml: "npa/src/npa/workflows/byof/profiles/byof-container-smoke-rtxpro.yaml"
  # use npa/src/npa/workflows/byof/profiles/byof-solution-smoke-rtxpro-gpu.yaml when CUDA/EGL/Vulkan is required
```

Run via:

```bash
npa/.venv/bin/python npa/scripts/run_byof_repo.py \
  --repo-url <url> \
  --repo-ref <pinned-tag-or-sha> \
  --base-profile ubuntu \
  --base-image <if-required> \
  --build-command '<install>' \
  --workload solution-smoke \
  --smoke-command '<solution-specific hello-world>' \
  --solution-name <slug> \
  --capability-name <capability_id> \
  --smoke-artifact-name <artifact.json> \
  --project <project-alias> \
  --run-id byof-<slug>-smoke \
  --cleanup
```

### 4. Live infra gate (mandatory)

Registry admission requires all of:

| Check | Pass criteria |
| --- | --- |
| Build/push | Image in Nebius registry with `npa_source_metadata.json` |
| K8s pull | Pod starts from pushed image (`sky launch --down` path) |
| Capability smoke | `smoke_command` exit 0 |
| Artifact | Named JSON present under smoke output dir and uploaded to S3 |
| Summary | `npa_byof_summary.json` includes `solution_name`, `capability_name`, `smoke_exit_code: 0` |

`container-verify` alone is **not** registry admission. Use `solution-smoke`.

### 5. Document accepted vs deferred

Update `docs/workbench/oss-solution-catalog.md` with **this solution's**
capability table. Mark only live-passing capabilities as accepted. Keep deferred
blockers explicit (assets, Vulkan, GCS, dataset size, VRAM).

Then load `skills/atomic/audit-container-docs/SKILL.md` and reconcile
`docs/workbench/container-image-catalog.md`. Add the solution's image to the
public table only if repository publication policy selects its resolved pin and
anonymous registry inspection proves that exact tag is available. A BYOF-only,
restricted, deferred, private-registry, or not-yet-published solution belongs in
its solution documentation, not in the public-image table.

## Current Onboarded Solutions

Catalog: `docs/workbench/oss-solution-catalog.md`.
Specs: `npa/workflows/workbench/npa-workflows/byof-<solution>.yaml`.

Keep each solution's capability list and smoke command unique. When promoting a
deferred capability, change that solution's smoke (or add a second workflow
spec) rather than mapping it onto a generic family label.

### ManiSkill (`byof-maniskill.yaml`)

Pinned: `mani-skill/ManiSkill` `v3.0.1` · base `maniskill/base:latest`

Required smoke capabilities (encoded in `byof-maniskill.yaml`):

- `gymnasium_pickcube_registration` (required / accepted gate)
- `pickcube_cpu_step` (attempted in isolated subprocess; may defer on SAPIEN segfault)
- `pickcube_parallel_envs` (attempted in isolated subprocess)
- `pickcube_gpu_rgb_render` (attempted in isolated subprocess)

Follow-up: RL/IL baselines (`mani_skill.examples.*`), asset download / real2sim.

### MuJoCo Playground (`byof-mujoco-playground.yaml`)

Pinned: `google-deepmind/mujoco_playground` `v0.2.0`

Required smoke capabilities:

- `mjx_cartpole_step`
- `mjx_cheetah_run_step`
- `train_jax_ppo_cartpole_smoke` (live-accepted; brax PPO train API, jax&lt;0.8.1)

### RoboCasa (`byof-robocasa.yaml`)

Pinned: `robocasa/robocasa` `v1.0`

Hard-gate capability: `kitchen_task_registration`.

Also exercised in the same smoke (live-accepted with S3 evidence):

- `download_kitchen_assets_lw` (IIFAN lightwheel fixtures/objects + git accessory restore)
- `kitchen_egl_env_reset` (post-download subprocess so `OBJ_CATEGORIES` sees mjcf paths)
- `kitchen_random_rollout` (`run_random_rollouts` with mp4; pin `gymnasium==0.29.1` and bind `env.sim`)

### OpenPI (`byof-openpi.yaml` + `openpi-pi05-four-mode.yaml`)

Pinned: `Physical-Intelligence/openpi` `15a9616a00943ada6c20a0f158e3adb39df2ccac`

The builder's historical hard gate is
`pi05_droid_jointpos_polaris_served_infer` using the upstream WebSocket
policy server/client in one B200 (`sm_100`) pod. The connected four-mode gate
must additionally pass all of:

- `pi05_droid_jointpos_polaris_cross_pod_serve`: digest-pinned upstream server
  Deployment, private ClusterIP Service with readiness/liveness, and two valid
  requests from a distinct client pod
- `pi05_droid_jointpos_polaris_lora_optimizer_smoke`: supported upstream pi0.5
  LoRA configuration, real forward/backward/AdamW update, changed trainable
  state, and reloadable Orbax checkpoint
- `pi05_droid_jointpos_polaris_heldout_evaluate`: exact trained-checkpoint
  reload, disjoint held-out upstream model loss plus action MAE/MSE, and a valid
  reloaded trajectory

Also hard-gated in the same smoke:

- `pi05_droid_jointpos_polaris_checkpoint_download` from the runtime-only GCS source
- `pi05_droid_jointpos_polaris_direct_infer` (`create_trained_policy` + `policy.infer`)
- finite joint-position action chunks shaped `[T>=5,8]` from both paths

Four-mode live acceptance requires the canonical isolated B200 (`sm_100`) gate: build the
pinned source, execute the declared editable-install and CUDA-compile commands,
push it to the private project registry, resolve and pull the immutable digest,
then run a separate invalid-terms workload that exits 64 before checkpoint/model
loading. Only after that negative gate passes may accepted stages fetch the 27
objects / 12,434,530,837 bytes at runtime. Direct and both cross-pod service
requests must be finite `float64[T>=5,8]`. Training and held-out evaluation must
consume machine-verifiably disjoint samples, and evaluation must consume the
exact independently read-back training checkpoint.

This checkpoint contains Gemma-derived material. Require the exact run-scoped
`NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES` gate before build or download; forward it
only through the secret channel and never bake/persist it. The image contains
the pinned Apache-2.0 source and CUDA/JAX runtime, not checkpoint bytes. A tiny
deterministic compatible dataset is valid only for the real optimizer and
held-out offline operational gate. It is not convergence evidence. Do not claim
physical Franka success, external Ingress, or robot success from offline
evaluation. The builder's legacy served check remains same-pod loopback; only
the connected service capability may claim cross-pod ClusterIP transport.

### DROID policy learning (`byof-droid-policy-learning.yaml`)

Pinned: `droid-dataset/droid_policy_learning` `9a29c832b4c81bf38401111f5e4cdddaca217581`

Hard-gate capability: `rlds_config_generator_contract`.

Also exercised in the same smoke (live-accepted with S3 evidence):

- `droid_100_download`
- `droid_100_config_gen`

Follow-up: full / debug `train.py` once data is staged.

### Open Dreamer (`byof-open-dreamer.yaml`)

Pinned: `next-state/open-dreamer` `2b10640` · base `ubuntu` + system `python3.11`
+ `uv sync` (CUDA-12 JAX/Flax). This is a **world-model** solution (a JAX/Flax
Dreamer 4 pipeline) and the reference example for the multi-GPU BYOF path: its
accepted capability requires a genuine **>=2 GPU** device mesh, so it uses the
`byof-solution-smoke-rtxpro-2gpu.yaml` resource profile
(`RTXPRO-6000-BLACKWELL-SERVER-EDITION:2`) instead of the single-GPU profile.

Hard-gate capabilities (all must pass; the driver `raise SystemExit`s if any is
missing, so a green smoke means the dream actually ran):

- `jax_two_gpu_data_parallel_mesh` (`dreamer.parallel.build_parallel("data")`
  builds a `{data: 2, model: 1}` mesh over `jax.devices()`; fails on <2 GPUs)
- `dreamer4_tokenizer_train_two_gpu` (real `scripts/train_tokenizer.py`
  entrypoint trains the causal video tokenizer sharded across the mesh on a
  **real Minecraft/VPT** video subset to legibility)
- `dreamer4_action_conditioned_dream_rollout` (the marquee payoff:
  `dreamer.sampler.sample_video` dreams future gameplay from context frames +
  future actions; reports dream PSNR — transitively gates the whole loop)

Also exercised in the same smoke:

- `minecraft_vpt_video_dataloader` (real `dreamer.data.build_iterator`
  `minecraft_vpt` MP4 path — decord decode + VPT action parse — with device
  sharding)
- `dreamer4_latent_tokenization` (`scripts/tokenize_minecraft_dataset.py`
  encodes the episodes into real latent ArrayRecords + `latent_stats`, carrying
  the real 27-binary / 121-categorical VPT actions)
- `dreamer4_dynamics_train_two_gpu` (action-conditioned latent dynamics trained
  on those Minecraft latents via `scripts/train_dynamics.py`; the core Dreamer
  world-model loop)
- `world_model_rerun_visualization` (emits `open_dreamer_world_model.rrd` with
  synchronized `world/observation` (GT), `world/dream` (predicted),
  `world/gt_decoded` (tokenizer ceiling), and `world/tokenizer_reconstruction`
  streams, loadable in the NPA agent Rerun viewer)

The run trains the tokenizer and dynamics for real on a real Minecraft/VPT
gameplay subset and dreams action-conditioned future frames, so it is a real
multi-stage GPU run with viewable visualizations, not an import-only or
synthetic smoke.

Data note: the smoke trains on a real **Minecraft/VPT** contractor-gameplay
subset (OpenAI VPT `.mp4` + `.jsonl`), center-cropped and resized to 128x128 and
staged as `minecraft_vpt` ArrayRecords to the run bucket under
`datasets/minecraft_vpt_128_64/`, pulled at run time (no dataset paths, buckets,
or IDs are hardcoded in the spec). Latent records carry the minecraft `latent`
action layout (27 binary / 121 categorical) required by `train_dynamics.py`.

Follow-up: FVD evaluation (`scripts/eval_fvd.py`, needs I3D weights) and a
larger training budget / dataset for a sharper, longer-horizon dream.

### Alibaba Wan 2.2 (`byof-wan2.2.yaml`, `byof-wan2.2-multigpu.yaml`)

Pinned official source:
`Wan-Video/Wan2.2@42bf4cfaa384bc21833865abc2f9e6c0e67233dc`; official
TI2V-5B checkpoint:
`Wan-AI/Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e`.
The candidate uses one RTX PRO 6000 Blackwell (`sm_120`), native
`wan.WanTI2V.generate`, the security-fixed PyTorch 2.13.0 CUDA 13.0 wheel line
with an explicit `sm_120` architecture check, and run-time model acquisition.
No weights are baked, and the upstream native PyTorch SDPA fallback is used.

Accepted historical hard-gate evidence, validated by a prior private record on
one RTX PRO 6000 Blackwell (`sm_120`) using Torch 2.7.1/CUDA 12.8:

- `wan2.2_ti2v_5b_text_to_video` (real 1280x704 MP4)
- `wan2.2_decoded_mp4_validation` (decode all frames; dimensions/count/fps and
  conservative non-uniform-content checks)

Accepted historical distributed evidence, validated by a prior private record
on one node with four B200s (`sm_100`, world/local world size 4) using NCCL
2.27.7. The current NCCL 2.29.7 closure requires fresh operator-accepted live
qualification:

- `wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses`
  (`torch.distributed.run` launches an instrumentation wrapper on four ranks;
  the wrapper executes pinned official `generate.py` as `__main__` with NCCL,
  T5 and DiT FULL_SHARD FSDP, and Ulysses size 4)
- `wan2.2_distributed_rank_topology_validation` (four unique GPU hashes;
  ranks/local ranks 0–3; NCCL sum 10/10; 480 distributed-attention and 1,920
  all-to-all calls per rank; upstream and observer final barriers)
- `wan2.2_decoded_mp4_validation` (2,809,770-byte H.264 MP4; 1280x704,
  17 frames, 24 fps; spatial stddev 71.9485, pixel range 255, temporal delta
  9.714725, SHA-256 `9574f79c…94865`)

The primary artifact is `wan2_2_ti2v_5b_text_to_video.json`; the MP4 is
`wan2_2_ti2v_5b.mp4`, and the actual pulled image emits
`wan2_2_runtime_inventory.json` with installed package/license metadata and a
baked-checkpoint scan. All three are present in S3 for the acceptance run; its
2,923,858-byte H.264 MP4 (SHA-256 `60001084…92328`) decoded as 17 1280x704
frames at 24 fps and passed the non-uniform-content gates. Kubernetes observed
the immutable accepted image digest as the running `imageID` in both fresh
single- and four-GPU proofs; each fresh RRD embeds the exact generated MP4.

Deferred: TI2V image-to-video until its own live input/output evidence; T2V and
I2V A14B, S2V-14B, Animate-14B, and official training as separate contracts.
Stock Wan action prediction is rejected as an upstream claim. Successful Wan
runs are postprocessed into a verified Rerun recording that embeds the exact
MP4 alongside static run evidence; see `skills/tools/wan2-2/SKILL.md` and
`docs/workbench/wan2.2.md`.

### Lightricks LTX-2.5 (`byof-ltx2.yaml`)

Pinned upstream source:
`Lightricks/LTX-2@fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`; gated checkpoint
set: `Lightricks/LTX-2.5`. **No capability here is live**: the image has been
built, pushed, and scanned by digest, but no GPU has run it. The entry exists so
the contract is reviewable before evidence, not so it can be mistaken for
evidence.

Read this one before onboarding any non-OSI model, because it breaks the habit
the other entries teach. The LTX-2.x Community License Agreement (2026-08-11)
licenses the **source** as well as the weights (Section 1.9 covers the
accompanying source code), so "bake the code, fetch the weights" would have made
the image non-redistributable. `npa-ltx2` bakes neither, and both fetches refuse
without the operator's own `HF_TOKEN`:

- `ltx2_5_text_to_video` (real `python -m ltx_pipelines.distilled` generation)
- `ltx2_5_decoded_mp4_validation` (decode the pixels; reject an unreadable
  container, a flat render, and one still repeated)

The primary artifact is `ltx2_5_text_to_video.json`. Before either fetch, the
run proves the refusal on the image it is actually running (`ltx-runtime
assert-refusal`: exit 78, naming *which* gate refused, with both caches still
empty) — a property of the image rather than a capability of the model.

The licence acceptance is not ours to collect. It binds by conduct, and
`Lightricks/LTX-2.5` is a gated repository, so a token that can read it is
checkable evidence that a human accepted Lightricks' terms — strictly better
than a `NPA_LTX_ACCEPT_COMMUNITY_LICENSE=YES` variable, which an earlier version
of this entry required and which never formed the contract. Compliance with the
Agreement, including Attachment A(18) (no training other models on the Outputs
for commercial use, and a robot policy is another machine learning model), is
the operator's own responsibility; the pipeline therefore stops at curation
rather than training. Not claimed: image-to-video, audio-to-video, and LoRA
fine-tuning. See `npa/docker/workbench/ltx2/REDISTRIBUTION.md` and
`docs/workbench/ltx2.md`.

### Multi-GPU solutions

When a solution's accepted capability is only meaningful across multiple GPUs
(distributed / sharded / model-parallel training, multi-GPU inference), request
`>=2` accelerators through a dedicated resource profile
(`byof-solution-smoke-rtxpro-2gpu.yaml`) and make a "device mesh sees N GPUs"
check a hard gate so a single-GPU scheduling fallback cannot masquerade as a
passing multi-GPU run. Open Dreamer is the reference example.
Wan 2.2 is the inference reference: the dedicated `B200:4` profile must prove
that all ranks participate in one official generation through sharding and
sequence parallelism; four scheduled/visible GPUs or four replica outputs are
not evidence.

## Capability Discovery Procedure

1. **Read upstream docs first.**
   - Inspect README, docs site, examples, install guide, quickstarts,
     configuration examples, model/data download instructions, and license.
   - Prefer docs and maintained examples over source-code guessing.
   - Capture the exact upstream refs used: repo URL, commit/ref, docs paths, and
     example names.

2. **Classify the solution (for NPA mapping only).**
   - Domain and runtime help choose base image / GPU profile.
   - NPA surface: BYOF image, registry entry, workflow, future Workbench tool,
     or future top-level solution namespace.
   - Do not collapse distinct upstream capabilities into shared family labels.

3. **Select capability tests.**
   - Include at least one smoke per registry claim.
   - For multi-capability repos, test the smallest representative command for
     each major claim, not a single generic import check.
   - Favor documented example commands with reduced dataset/model sizes or smoke
     flags. Do not add artificial time, cost, or job-count limits unless the
     operator asks.

4. **Map artifacts.**
   - Define S3-style inputs and outputs for every workflow-stage claim.
   - Record schemas when known; otherwise create a conservative artifact
     manifest and mark schema stabilization as follow-up.

## Registry Admission Gates

A solution is registry-ready only after all applicable gates pass:

| Gate | Requirement |
| --- | --- |
| Documentation | Upstream docs read and cited for every claimed capability |
| License | Upstream license and asset/model/data restrictions recorded, and the image's redistribution class set per `skills/atomic/solution-licensing/SKILL.md` |
| Packaging | BYOF image builds and includes `npa_source_metadata.json` |
| Registry | Image pushed to the resolved Nebius registry; no hardcoded registry IDs |
| Contract | Inputs, outputs, runtime, GPU, credentials, and failure modes documented |
| Workflow | NPA workflow validates/plans if a workflow is part of the registry entry |
| Smoke | Capability-level smoke commands pass in the container or service |
| Container E2E | The registry image is pulled and exercised by a real NPA/SkyPilot/Kubernetes E2E workflow, not only by local Docker |
| Live Infra | Required GPU/K8s/SkyPilot path runs on live Nebius infrastructure |
| Hygiene | No secrets, project IDs, tenant IDs, bucket names, private endpoints, or customer identifiers committed |
| Docs | NPA registry/catalog docs and validation report are linkable |

Build-only validation is not sufficient for registry admission.

## Implementation Flow

1. **Evidence brief**
   - Summarize upstream docs and selected testable capabilities.
   - Reject or defer unsupported, undocumented, or license-blocked claims.

2. **BYOF package**
   - Use `npa/scripts/run_byof_repo.py` from the BYOF skill.
   - Pick `--base-profile ubuntu` for generic repos.
   - Pick `--base-profile isaac-lab` for Isaac Lab/LeIsaac sim, datagen, or RL.
   - Use `--base-image <ref>` only when upstream runtime requirements demand it.
   - For registry candidates with documented install/run commands, prefer
     `--workload solution-smoke --build-command <install> --smoke-command <smoke>`
     plus `--solution-name`, `--capability-name`, and
     `--smoke-artifact-name` so the pushed image is tested through the live BYOF
     workflow and writes an inspectable capability artifact.

3. **Capability smoke matrix**
   - Add or document smoke commands for each claim.
   - Include container-local smokes and live SkyPilot/Kubernetes smokes where the
     capability needs GPU or cluster resources.
   - Treat local container smokes as preflight only. The same pushed registry
     image must be pulled by an NPA/SkyPilot/Kubernetes workflow and run through
     at least one representative end-to-end path that consumes declared inputs
     and writes declared outputs.
   - A solution-smoke command must do more than import modules: it must execute a
     documented capability hello-world and write the named JSON artifact under
     `$NPA_SMOKE_OUTPUT_DIR`.
   - Keep commands grounded in upstream docs.

4. **NPA contract**
   - If the solution is workflow-shaped, author a YAML under
     `npa/workflows/workbench/npa-workflows/` and validate with:
     ```bash
     npa/.venv/bin/npa workbench workflow validate-spec <spec.yaml> --json
     npa/.venv/bin/npa workbench workflow plan-spec <spec.yaml> --run-id <run-id> --json
     ```
   - If the solution should remain BYOF-only, document the BYOF command and
     registry metadata without adding a new CLI namespace.
   - Add a first-class Workbench tool only when there is stable user-facing
     behavior, docs, tests, and a maintained contract.

5. **Live Nebius validation**
   - Use resolved project, registry, storage, and Kubernetes config from
     `~/.npa/config.yaml` and `~/.npa/credentials.yaml`.
   - Never hardcode infrastructure identifiers.
   - Validate the actual registry image inside the real E2E path.
   - Run the relevant live path:
     ```bash
     export NPA_E2E_PROJECT=<project-alias>
     export NPA_BYOF_LIVE_PIPELINE=1
     bash npa/scripts/verify_byof_onboarding_live.sh
     ```
   - For repo-specific validation, set `NPA_BYOF_REPO_URL`,
     `NPA_BYOF_REPO_REF`, `NPA_BYOF_BASE_PROFILE`, and the matching live flags
     from `byof-onboard`.

6. **Registry report**
   - Produce a concise report with:
     - upstream repo/ref/license
     - docs consulted
     - accepted capabilities
     - deferred capabilities and blockers
     - image URI or placeholder
     - workflow/toolRef/CLI/docs paths
     - smoke and live validation commands
     - exact pass/fail output summaries

## Promotion Rules

- **BYOF image**: repo builds, image pushes, and at least one documented
  capability smoke passes inside the built container.
- **Registry/catalog entry**: BYOF image plus capability matrix, docs, artifact
  contract, hygiene, live Nebius validation, and an E2E workflow that pulls and
  runs the pushed registry image.
- **Workbench workflow**: registry entry plus validated/planned
  `npa.workflow/v0.0.1` spec and live workflow evidence.
- **First-class Workbench tool**: workflow or service has stable API/CLI,
  tool-specific docs, unit/smoke/live tests, and a maintenance owner.
- **New top-level solution namespace**: only when the capability is a durable
  product surface, per `docs/architecture/solutions-model.md`.

## Required Validation Commands

Run local guardrails after changing skills, docs, workflow specs, or catalog
entries:

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
npa/.venv/bin/python -m pytest npa/tests/workflows/test_byof_solution_smokes.py -q
```

When adding workflow specs:

```bash
npa/.venv/bin/python -m pytest npa/tests/orchestration/npa_workflow/ \
  npa/tests/smoke/test_npa_workflow_smoke.py \
  npa/tests/smoke/test_all_workflow_yamls.py -q --tb=no
```

When claiming live readiness, include the BYOF live verification or a
tool-specific live e2e command that pulls the registry image and runs a real
workflow path. If live infrastructure is unavailable, report the exact precheck
failure and keep the solution out of registry-ready status.

## Gotchas

- A Docker build is evidence of packageability, not a capability test.
- A local `docker run` is still only preflight; registry readiness requires the
  pushed image to run inside the same kind of NPA/SkyPilot/Kubernetes E2E
  workflow users will invoke.
- An upstream README capability is not an NPA registry capability until it has a
  passing smoke or a documented live-infra blocker.
- Generic import checks do not prove simulation, training, datagen, serving, or
  evaluation behavior.
- Shared capability "families" are not part of this skill; do not invent a
  cross-solution taxonomy that erases upstream-specific APIs.
- Narrowing a smoke to a stable contract is valid when fuller paths are blocked;
  record the fuller path as `deferred`, never as accepted.
- Do not create new skills under `.agents/skills` or `.claude/skills`; update
  only the root `skills/` tree and `skills/index.yaml`.
- Do not add hidden infrastructure defaults. Let project, registry, Kubernetes,
  and storage resolve through NPA config.
