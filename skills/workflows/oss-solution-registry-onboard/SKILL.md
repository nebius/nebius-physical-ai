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
- Relevant tool skill (`skills/tools/isaac-lab`, `lerobot`, `genesis`,
  `cosmos`, `groot`, `sonic`, `fiftyone`, `lancedb`, `mjlab`, or
  `retargeting`) when the upstream repo depends on that stack.

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
- OpenPI: `policy_config_materialization`, `pi05_droid_checkpoint_infer`
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
  resource_profile_yaml: "npa/workflows/workbench/skypilot/byof-container-smoke-rtxpro.yaml"
  # use byof-solution-smoke-rtxpro-gpu.yaml when CUDA/EGL/Vulkan is required
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
- `train_jax_ppo_cartpole_smoke` (attempted; may remain deferred if the
  upstream train script's video/EGL path fails independently)

### RoboCasa (`byof-robocasa.yaml`)

Pinned: `robocasa/robocasa` `v1.0`

Hard-gate capability: `kitchen_task_registration`.

Also attempted in the same smoke (may remain deferred with evidence):

- `download_kitchen_assets_lw`
- `kitchen_egl_env_reset`
- `kitchen_random_rollout` (best-effort after reset)

### OpenPI (`byof-openpi.yaml`)

Pinned: `Physical-Intelligence/openpi` `15a9616a00943ada6c20a0f158e3adb39df2ccac`

Hard-gate capability: `policy_config_materialization` (`get_config("pi05_droid")`).

Also attempted in the same smoke (may remain deferred with evidence):

- `pi05_droid_checkpoint_download`
- `pi05_droid_checkpoint_infer` (`create_trained_policy` + `policy.infer`)

Follow-up: LoRA / fine-tune recipes.

### DROID policy learning (`byof-droid-policy-learning.yaml`)

Pinned: `droid-dataset/droid_policy_learning` `9a29c832b4c81bf38401111f5e4cdddaca217581`

Hard-gate capability: `rlds_config_generator_contract`.

Also attempted in the same smoke (may remain deferred with evidence):

- `droid_100_download`
- `droid_100_config_gen`

Follow-up: full / debug `train.py` once data is staged.

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
| License | Upstream license and asset/model/data restrictions recorded |
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
