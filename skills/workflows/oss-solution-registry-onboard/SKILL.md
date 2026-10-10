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
- LIBERO: `libero_spatial_bc_rnn_train_reload_heldout`
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

Author `workflows/testing/byof-<solution>.yaml` with:

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
| Build/push | Image in the authorized registry with `npa_source_metadata.json` |
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
Specs: `workflows/testing/byof-<solution>.yaml`.

Keep each solution's capability list and smoke command unique. When promoting a
deferred capability, change that solution's smoke (or add a second workflow
spec) rather than mapping it onto a generic family label.

### Habitat-Sim (`habitat-sim-smoke.yaml`)

Pinned: `facebookresearch/habitat-sim`
`57ee4941dc4765240f0f91f70b2c97a919bf9038` (MIT). Upstream explicitly warns
that beyond v0.3.4, Meta internal teams do not officially maintain releases or
provide active development.

The public `npa/docker/workbench/habitat-sim/Dockerfile.bootstrap` ships a neutral
runtime-fetch launcher. Its retained development proof is recorded in
[`habitat-sim-development-image-manifest.json`](../../../docs/workbench/validation/habitat-sim-development-image-manifest.json):
19 RGB/depth frame pairs, 19 Bullet steps and 2.2466 metres of traversal on
one RTX PRO 6000 Blackwell. Supported release selection remains quarantined.
The result is scoped to that historical producer/digest and does not qualify
the separate baked recipe or prove a long benchmark or policy-training result.

The separate legacy `npa/docker/workbench/habitat-sim/Dockerfile` recipe pins linux/amd64 Ubuntu
22.04 by digest and a signed immutable package snapshot. It materializes only
the source projection required by the headless RGB-D, pathfinding, Bullet, and
EGL build. The unused `rlr-audio-propagation` gitlink is CC BY-NC 4.0 and is
excluded with audio; GUI and docs gitlinks are excluded too. Install the exact
hash-locked build and runtime wheel closures with dependency resolution and
build isolation disabled. Keep the candidate unbuilt and publication-
quarantined until actual OCI bytes pass the complete byte, layer, license,
SBOM, provenance, vulnerability, payload-absence, and non-root checks.

Hard-gate capabilities (all must pass in one live pod for the selected digest):

- `skokloster_castle_rgb_depth_bullet_traversal`: actual upstream greedy-follower
  agent actions, saved RGB/depth observations, and nonzero displacement
- `headless_nvidia_egl_rgb_depth_render`: live NVIDIA GL strings and NVIDIA EGL
  libraries loaded by the renderer process
- `bullet_physics_world_step`: Bullet-enabled backend with advancing world time
- `greedy_geodesic_agent_traversal`: a real pathfinder/navmesh traversal rather
  than direct state teleportation

Fetch only the official Meta `habitat-test-scenes.zip` archive referenced by the
pinned Habitat-Sim `examples/settings.py`. Treat its URL as mutable: require the
94,590,970-byte archive SHA-256
`1231420c6482e79e25beea7ab25121e0421a5fd67b68dd9502145442c288db06`,
then extract only `skokloster-castle.glb` and `skokloster-castle.navmesh` after
their exact member names, sizes, CRC32 values, and SHA-256 hashes pass. Delete
the archive before simulator creation and never bake or extract another member.
The pinned source README and original asset record identify The King's Hall
under CC BY 4.0. Preserve creator/scan attribution, license and original-asset
links, and the Habitat-ready modification provenance in the proof.

Use `workflows/testing/habitat-sim-smoke.yaml` on exactly one RTX PRO 6000
Blackwell (`sm_120`). The owner-only runtime evidence must prove
that cluster's Capacity Block is bound `STRICT`; the resource profile alone is
not that proof. The live gate requires an owner-only, run-bound manager receipt,
verifies a hashed provider reservation readback, rejects the public registry
default, and compares Kubernetes `containerStatuses[].imageID` with the pushed
digest before exact-run teardown. It also downloads and re-hashes every declared
RGB/depth and inventory object. This renderer must never use B200. A future
trusted public workflow must rebuild the exact reviewed full Git SHA; that new
digest requires complete repeated byte scans and genuine RTX qualification, and
must not transport privately built OCI bytes.
Defer proprietary/gated datasets, semantic annotations, and distributed
Habitat-Lab training.
### LIBERO (`byof-libero.yaml`)

Pinned: `Lifelong-Robot-Learning/LIBERO`
`8f1084e3132a39270c3a13ebe37270a43ece2a01`.

Hard-gate capability: `libero_spatial_bc_rnn_train_reload_heldout`.

The public candidate is a quarantined, unbuilt neutral bootstrap: it contains
only the pinned Python/Debian base, snapshot-locked bootstrap packages, NPA
scripts, and immutable manifests. It must contain no LIBERO, GPU runtime,
model, demonstration, task/render asset, cache, checkpoint, credential, or
output bytes. Do not add it to the public table or release manifest until a
trusted exact-SHA build passes complete-byte and independent base-provenance
scans, anonymous pull, and the exact-digest hard gate.
The neutral development build does not require customer authorization or prior
runtime qualification. Its local complete-byte scan records the ordered-layer,
flattened-rootfs, and config identity from the inspected build; post-publication
scanning must reproduce that identity and an anonymous pull must fetch the
actual image layers. Customer acknowledgement remains mandatory before runtime
fetch, and real exact-digest B200 evidence is required for supported release
promotion and public catalog validation.

The qualifying smoke must runtime-fetch and SHA-256-verify the exact official
LIBERO-Spatial demonstration pinned in the spec, bind it to the reviewed BDDL
and initial-state hashes, split whole trajectories into disjoint train and
held-out partitions, execute exactly eight nonzero upstream
`Sequential.observe` optimizer steps with `BCRNNPolicy`, record and validate
that exact count, save and strictly reload an upstream checkpoint, and report
finite held-out loss plus finite reloaded actions. It
must run headlessly on exactly one STRICT-bound B200 (`sm_100`) and record the
Pod-observed immutable digest of the qualified candidate, matching the
anonymously resolved public-development digest; private or historical image
digests are not qualification evidence. Imports, BDDL parsing, dataset
inventory, or zero-step training are not acceptance evidence.

The source is MIT and the upstream LIBERO publisher declares its datasets CC BY
4.0. Preserve the publisher's license when a mirror card conflicts. Keep the
demonstration in a run-scoped runtime cache outside `$NPA_SMOKE_OUTPUT_DIR` and
never bake it. The selected task names Google Scanned Objects and a HOPE
distractor; the headless qualification's runtime sparse checkout never fetches
the unused render-asset tree while retaining the hash-bound MIT task
definitions. Require a short-lived authorization signed directly by the
customer-controlled key, bound to the customer, run, exact terms, runtime
manifest, and immutable qualified image when available, before any cache or
network mutation. The manager/control plane may authenticate, transport, and
validate that evidence but never accept, acknowledge, issue, or sign the
customer's terms decision. An optional transported copy of the signer public key
is owner-private; when supplied, it must byte-match the customer-signed evidence.
The authenticated caller assertion remains the required signer-fingerprint binding.
No customer signer trust root or fingerprint is baked into the neutral image. Keep the payload on its
pods/get-only account and require the separately precreated, non-wildcard
controller Role to pass exact namespaced and no-ClusterRoleBinding checks
before submission.
Never invent an `ACCEPT_*` variable, automate a vendor acceptance action, or
treat fetch/authentication as permission.
Rendered closed-loop sweeps, all 130
tasks, lifelong-algorithm comparison, and physical-robot use remain deferred.

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

### EmbodiedGen V2 (`byof-embodiedgen.yaml`)

Pinned: `HorizonRobotics/EmbodiedGen` `f0124197888c2b733e4eaa65acd81ad9cfda3b79`;
the upstream-supported TRELLIS gitlink
`55a8e8164b195bbf927e0978f00e76c835e6011f`; and
`microsoft/TRELLIS-image-large@25e0d31ffbebe4b5a97464dd851910efc3002d96`.

The hard gate is `img3d-cli_trellis_image_to_urdf_pybullet`: the real upstream
TRELLIS backend must produce a mesh and URDF, the exported collision meshes must
decode and remain non-degenerate, and that exact URDF must load, contact a plane,
settle under gravity, and produce a fully decoded PyBullet view MP4. The artifact
is `embodiedgen_image_to_rigid_object.json`. It must name actual GPU facts and
the source/model/image revisions and hashes. VLM-derived physical values are
estimates, not calibrated ground truth. This operator-private CUDA candidate does
not assert articulated generation or policy improvement.

### Gymnasium-Robotics (`byof-gymnasium-robotics.yaml`)

Pinned: `Farama-Foundation/Gymnasium-Robotics`
`4d1ebecbc6436806cfbc0e42ebc36f594d05844e` · MuJoCo `3.12.0` · base
`ubuntu:noble-20260905@sha256:a61567bd31828687156d735ea8eb01ba4e37636e225dd6a48ba94136a70d9d61`

The single hard gate is the upstream registered
`HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1` environment. One
strictly reserved RTX PRO 6000 Blackwell must exercise all of:

- `registered_shadow_hand_environment`
- `mujoco_physics_steps`
- `continuous_touch_sensor_response`
- `mujoco_contacts`
- `egl_rgb_rendering`
- `rtx_pro_6000_blackwell_execution`

The smoke must write exactly `gymnasium-robotics-smoke.json`, prove real
MuJoCo steps, contacts or touch response, state/orientation change, distinct
RGB frame hashes, NVIDIA EGL, and equality between the immutable image digest
and the pod-observed digest. The public candidate is a neutral bootstrap with
an empty workflow build command: upstream source, Shadow assets, MuJoCo/Python
workload, and populated caches are fetched only into an operator-owned runtime
cache after complete hash locks. Fetch changes delivery only, not use,
derivative, output, or service rights. The image has an immutable payload-free
development-build path but remains release-quarantined until every exact
publication and capability gate passes. Historical private evidence never
qualifies redesigned or public bytes. This is not a first-class tool. It uses
no model, external dataset, gated asset, or terms acceptance. RL sweeps, expert
scores, other environment families, and physical-robot transfer remain
deferred. The operator qualification contract requires exactly one strictly
reserved RTX PRO 6000 Blackwell; never route this task to B200. Other hardware,
including B200/B300, is unqualified/deferred for this capability, not proven
incapable solely by the absence of RT cores. MuJoCo's EGL raster rendering is
distinct from ray tracing; any future qualification needs exact driver/EGL,
image and real-workload evidence. This rationale does not relax placement or
publication exclusions and makes no new hardware-support claim.

### RoboCasa (`byof-robocasa.yaml`)

Pinned: `robocasa/robocasa` `v1.0`

Hard-gate capability: `kitchen_task_registration`.

Also exercised in the same smoke (live-accepted with S3 evidence):

- `download_kitchen_assets_lw` (IIFAN lightwheel fixtures/objects + git accessory restore)
- `kitchen_egl_env_reset` (post-download subprocess so `OBJ_CATEGORIES` sees mjcf paths)
- `kitchen_random_rollout` (`run_random_rollouts` with mp4; pin `gymnasium==0.29.1` and bind `env.sim`)

### RoboTwin 2.0 (`byof-robotwin.yaml`)

Pinned source: `RoboTwin-Platform/RoboTwin`
`96c1feab536306b50c26af200044fcdf126e8904`. Pinned runtime assets:
`TianxingChen/RoboTwin2.0`
`785feb15aa4a4f532395ad2b1d2be5f28cb561ad`.

Hard-gate capability:
`beat_block_hammer_successful_seed_replay_collection`. It is accepted only when
the official `demo_clean` path searches for one successful seed, replays that
seed through SAPIEN/Vulkan on exactly one RTX PRO 6000 Blackwell (`sm_120`), and
emits native RoboTwin HDF5 plus decoded MP4 evidence with hashes, sizes, action
and frame counts, observed GPU/image identity, task success, and exit status.
Renderer startup or task registration alone is not evidence.

The image is a zero-vendor-payload public-bootstrap candidate. A full-SHA
development image completed one operator-run RTX collection/replay workload;
see [the operator evidence scope](../../../docs/workbench/byof-robotwin.md#retained-operator-evidence-and-readiness).
The public BYOF CLI and normal-submit worker bridge remain blocked, so registry
admission and supported-release publication remain quarantined. CuRobo v0.7.8
stays runtime-only and its
noncommercial research/evaluation field-of-use restriction remains binding on
use and service claims. Official assets remain a
runtime fetch. The standalone operator path requires an explicit customer-terminal
decision; the existing hosted authorization boundary remains a separate interface.
Both bind the customer, run, manifest, terms, activity, issuance, expiry, nonce,
and assertion. Manager context alone is not customer authentication, and an
operator result does not attest the blocked worker bridge. The repository MIT
card classifies the two exact
locked runtime members, `embodiments.zip` and `objects.zip`, at revision
`785feb15aa4a4f532395ad2b1d2be5f28cb561ad`. No additional restriction was
found in the inspected authoritative terms for the five declared output
classes: native HDF5 action/state data, decoded MP4, rendered frames, smoke
JSON, and summary JSON. CuRobo's noncommercial research/evaluation limit still
binds their generating workload and all use/service claims; hosted service is
not approved. Exact provider/artifact/revision payload probes remain required
before provisioning, with the customer's runtime-only credential when an
artifact is gated and no generic NPA terms boolean. The harness must scan the
pushed exact digest's rootfs and every layer for asset/cache/output bytes and
launch only that scanned digest. Runtime fetch, a credential, a private
registry, or a passing byte scan does not grant permission. Trusted full-SHA
development publication is separate from admission to the supported public
image catalog; keep the latter excluded. Other embodiments, unselected
future task assets, the full 50-task sweep, policy training/evaluation, physical
deployment, and their outputs remain independently deferred.

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

### evo trajectory evaluation (`byof-evo.yaml`)

Pinned: `MichaelGrupp/evo`
`8dd6cfe0ec1747f9e1b5b569edd82c54d1a3f422` (`v1.35.1`).
This is a GPL-3.0-or-later operator-built BYOF candidate; do not describe it as
an NPA-published image without a separate conveyance/compliance decision.

Hard-gate capability:

- `evo_ape_rpe_trajectory_evaluation`: native `evo_ape`, `evo_rpe`, and
  `evo_traj` commands must save decodable result archives and matched plots.
  Exact and bounded-error controls must pass the predeclared APE/RPE thresholds;
  nonlinear drift and malformed-input controls must fail, with zero false
  positives and zero false negatives.

Also exercised in the same live-qualified smoke:

- `decoded_plot_validation`: decode the selected native metric/trajectory plots
  and their derived review copies, require at least 800x600 pixels and
  non-uniform RGB content, and retain their hashes across the result and capture
  records. This is an auxiliary reviewability check, not a trajectory-quality
  metric. It covers the six review-facing plots plus eight additional
  `evo_ape` raw/control plots; only the six review-facing plots enter the
  capture/review set. It does not cover the five auxiliary `evo_traj`
  RPY/XYZ/speed plots. `synthetic_controls_speeds.png` is 651x491 and is the
  only residual plot below the stated 800x600 floor.

The same smoke evaluates the pinned upstream KITTI 00 ground truth, ORB, and
S-PTAM examples. Treat those as representative compatibility and reviewability
evidence, not as threshold calibration or proof of navigation success. Retain
each error distribution, trajectory-input hash, native plot hash, derived
review-plot hash, and decode result. The live-qualified candidate passed the
digest-pinned CPU Kubernetes pull and execution path, all four frozen gate
controls, and ten independent archive decodes. Its first unlabeled APE visual
review was invalidated when exact-prompt ORB/S-PTAM cross-swaps false-passed;
retain that failure. The corrected smoke preserves each native map and emits a
hash-linked review copy with visible estimate/reference identity. Require
identify-first cross-swap calibration with zero false positives before scoring
those final plots. In `npa.evo.capture-manifest.v1`, unsuffixed
`kitti_*_ape` keys identify the labeled review copies and `*_native` keys
identify untouched evo maps; follow each review entry's `source_path` and
`source_sha256` instead of guessing from the suffix. The operator-built image
remains unpublished. Before
registry admission, require independent review of the candidate commit and
retained evidence; a later conveyance still needs an explicit GPL compliance
decision.

### AprilTag 3 (`byof-apriltag.yaml`)

Pinned: `AprilRobotics/apriltag`
`94be783968e5091bcc9972c72c84fd63efce2935` (`v3.4.5`).

The detector source is BSD-2-Clause. Its three representative upstream
Swarmathon photographs and expected-corner labels are separately identified by
upstream as CC-BY-SA-2.0. Keep this an operator-built BYOF candidate unless a
later publication review accounts for both source and data terms.

Hard-gate capability: `apriltag_real_image_fiducial_detection`.

- Build and rerun `native_apriltag_ctest`; an import is not acceptance.
- Evaluate all three pinned real photographs and their upstream tag-ID/corner
  records. Every case must reproduce the recorded ID set (precision and recall
  1.0), with aggregate corner RMSE at most 0.1 pixel and no coordinate residual
  above 0.1 pixel. This is upstream regression parity. The records store four
  decimal places, so a roughly `2.74e-5 px` residual is their quantization
  floor, not detector localization accuracy.
- Run blank and fixed-seed random-noise controls. Both must produce zero
  detections.
- Emit `fiducial_observations.json` with source image hash, family, tag ID,
  center, ordered corners, decision margin, and hamming distance for a later
  camera-localization or calibration stage.
- Preserve each source image and native measurement. Annotated review PNGs must
  link back to the source hash and distinguish expected from detected corners.

Calibrate visual review with a correct overlay, a deliberately shifted overlay,
a blank image, and a mismatched annotation before scoring final exact-run
captures. Visual alignment is review evidence; objective corner metrics are
the upstream-parity gate. Do not call image-space corners camera-pose accuracy.
Pose claims require calibrated intrinsics, known tag geometry, and separate
translation / rotation error against ground truth.

### robomimic (`byof-robomimic.yaml`)

Pinned: `ARISE-Initiative/robomimic`
`d309eaecc18acf4152a830a895a6984b8ac71b05` · official Lift PH low-dimensional
dataset revision `robomimic/robomimic_datasets`
`74fa018461f479cd9fd15b924a16103012096203`.

The Phase A image is a quarantined neutral candidate only: bake the pinned MIT
source and exact non-CUDA lock, with no torch, NVIDIA/CUDA runtime, weights,
dataset, populated cache, credential, or output. Consume CUDA/PyTorch only from
an independently prepared exact-inventory read-only operator mount after the
customer creates an unexpired noncommercial-use record bound to its identity,
run, exact runtime lock and inventory, and official terms. The bootstrap verifies
or refuses; it must not fetch, install, warm, populate, or accept terms. An
environment flag, credential, manager signature, private registry, or runtime
fetch never supplies permission.

The hard gate must pass all three solution-specific capabilities on exactly one
STRICT-reserved B200 (`sm_100`):

- `lift_ph_lowdim_bc_train` — the upstream `scripts/train.py` BC entrypoint must
  perform nonzero optimizer work and serialize the Adam step state;
- `lift_ph_lowdim_heldout_validate` — upstream HDF5 train/valid masks must be
  nonempty and disjoint, and validation must produce a finite loss;
- `lift_ph_lowdim_checkpoint_reload_action` — reload the exact saved checkpoint
  with `policy_from_checkpoint` and infer a finite, in-range action from a
  held-out trajectory.

Require `robomimic-smoke.json`, the immutable source and dataset identities, the
dataset file hash and trajectory/sample counts, split hashes/counts, losses,
checkpoint hash, action proof, observed B200 identity, pod-observed image digest,
and exit status. Dataset inspection, imports, CPU fallback, a mutable image, or a
zero-step training config fails the gate. CUDA and cuDNN remain governed by the
NVIDIA CUDA Toolkit EULA, NVIDIA Software License Agreement, and cuDNN Software
License Agreement. They have no vendor token probe, so present the exact notice
and offer customer-controlled accept, decline, and resume actions. Acceptance
must be time-limited and bind customer, run, runtime lock, and inventory; reject
missing, declined, stale, or mismatched records before external action. It is
not a redistribution/publication grant. Keep the dependent capability private;
download the official dataset at runtime only after every separate gate passes.
Defer image-policy sweeps, simulator rollouts, and the full algorithm matrix.

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

Accepted current single-GPU evidence, bound to the exact public digest in
`npa/src/npa/deploy/wan2_2_image_manifest.json`, on one RTX PRO 6000 Blackwell
(`sm_120`) using Torch 2.13.0/CUDA 13.0:

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
baked-checkpoint scan. The current single-GPU acceptance record binds the
runtime inventory, MP4, and verified RRD to the accepted digest. Its
2,807,385-byte H.264 MP4 decoded as 17 1280x704 frames at 24 fps and passed
the non-uniform-content gates. The prior four-GPU proof remains historical
and does not qualify the current release's distributed path.

Deferred: TI2V image-to-video until its own live input/output evidence; T2V and
I2V A14B, S2V-14B, Animate-14B, and official training as separate contracts.
Stock Wan action prediction is rejected as an upstream claim. Successful Wan
runs are postprocessed into a verified Rerun recording that embeds the exact
MP4 alongside static run evidence; see `skills/tools/wan2-2/SKILL.md` and
`docs/workbench/wan2.2.md`.

### Lightricks LTX-2.5 (`byof-ltx2.yaml`)

Pinned upstream source:
`Lightricks/LTX-2@fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`; gated checkpoint
set: `Lightricks/LTX-2.5`. The accepted `2.5-rtfetch-20260817` release passed
payload and entitlement-refusal gates plus real text-to-video generation and
independent MP4 decoding on one RTX PRO 6000. The exact digest and 1536×1024,
121-frame, 1,994,625-byte result are recorded in
`npa/src/npa/deploy/ltx2_image_manifest.json`; new image bytes require new proof.

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
| Registry | Official public development image passes all pre-publication gates and uses `dev-<full-git-sha>`; BYOF or restricted images use only an operator-controlled registry |
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
     `workflows/testing/` and validate with:
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


## Diffusers video candidate contracts

- `byof-mochi-1.yaml`: `mochi-1_text_to_video` emits `mochi_1_text_to_video.json`; invoke the native pipeline and validate every decoded MP4 frame.
- `byof-cogvideox-2b.yaml`: `cogvideox-2b_text_to_video` emits `cogvideox_2b_text_to_video.json`; invoke the native pipeline and validate every decoded MP4 frame.
- `byof-wan2.1-14b.yaml`: `wan2.1-14b_text_to_video` emits `wan2_1_14b_text_to_video.json`; invoke the native pipeline and validate every decoded MP4 frame.

Catalog status remains authoritative. These recipes use an operator-built image; no official public release is implied.


### LingBot World v1 camera candidate

`byof-lingbot-world.yaml` exercises `lingbot_world_camera_conditioned_video` using pinned upstream `generate.py`, FSDP, and Ulysses on four B200 GPUs. It emits `lingbot_world_camera_conditioned_video.json`, per-rank execution evidence, authored camera controls, the source image, and a fully decoded MP4. See the catalog for packaged GPU qualification status. Camera trajectories and approximate intrinsics are authored, not measured. This v1 recipe does not claim robot action input, training, real-time performance, or support for the separate World Infinity successor.

- `byof-depth-anything-v2.yaml`: `relative_depth_video` emits `depth_anything_v2_relative_depth.json`, raw prediction arrays and a fully decoded GPU-derived video. See the catalog for packaged GPU qualification status. Model checkpoints are immutable runtime fetches; video inputs require a complete SHA256. No metric-depth, ground-truth-mask, or robot-success claim.

- `byof-sam2.1.yaml`: `prompted_video_mask_propagation` emits `sam2_1_video_mask_propagation.json`, raw prediction arrays and a fully decoded GPU-derived video. See the catalog for packaged GPU qualification status. Model checkpoints are immutable runtime fetches; video inputs require a complete SHA256. No metric-depth, ground-truth-mask, or robot-success claim.
