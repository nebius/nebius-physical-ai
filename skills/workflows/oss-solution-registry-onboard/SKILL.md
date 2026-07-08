---
name: oss-solution-registry-onboard
description: Use when evaluating and onboarding an open-source Physical AI solution into the NPA registry/catalog with documented capabilities, BYOF packaging, smoke tests, and live Nebius validation.
---

# OSS Solution Registry Onboard

Use this skill when an agent is asked to turn a public Physical AI repository
into a registry/catalog candidate for NPA. This is stricter than generic BYOF:
the agent must discover the upstream project's real documented capabilities,
test those capabilities, and produce validation evidence before calling the
solution registry-ready.

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
and identify real user-facing capabilities that can be tested.

For each claimed capability, record:

- upstream doc path or URL
- command, API, example, or config that demonstrates it
- required runtime profile (`ubuntu`, `isaac-lab`, custom base, service image)
- required accelerator and assets
- input artifact contract and output artifact contract
- NPA mapping: BYOF workload, Workbench tool, `toolRef`, workflow state, or docs
  only
- validation command and result

If a capability cannot be tested on available Nebius infrastructure, mark it
`deferred` with the precise blocker. Do not list deferred capabilities as
registry-ready.

## Capability Discovery Procedure

1. **Read upstream docs first.**
   - Inspect README, docs site, examples, install guide, quickstarts,
     configuration examples, model/data download instructions, and license.
   - Prefer docs and maintained examples over source-code guessing.
   - Capture the exact upstream refs used: repo URL, commit/ref, docs paths, and
     example names.

2. **Classify the solution.**
   - Domain: robotics, manipulation, locomotion, sim, synthetic data,
     perception, model serving, policy training, evaluation, visualization.
   - Runtime: CPU batch, CUDA batch, Isaac Lab, Mujoco, ROS, web service,
     dataset tool, model server, or multi-stage workflow.
   - NPA surface:
     - BYOF image only
     - Workbench registry/catalog entry
     - `npa.workflow/v0.0.1` workflow
     - future first-class Workbench tool
     - future top-level solution namespace

3. **Select capability tests.**
   - Include at least one smoke per registry claim.
   - For multi-capability repos, test the smallest representative command for
     each major capability family, not a single generic import check.
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
     documented capability hello-world (for example create/reset/step a sim env,
     generate a tiny synthetic-data artifact, materialize a policy/training
     config, or run a reduced inference/eval) and write the named JSON artifact
     under `$NPA_SMOKE_OUTPUT_DIR`.
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
   - Validate the actual registry image inside the real E2E path. For workflow
     candidates, this means a checked-in or generated workflow that pulls the
     image, runs the documented capability, writes artifacts to object storage,
     and can be inspected through NPA workflow status/logs.
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
- Do not create new skills under `.agents/skills` or `.claude/skills`; update
  only the root `skills/` tree and `skills/index.yaml`.
- Do not add hidden infrastructure defaults. Let project, registry, Kubernetes,
  and storage resolve through NPA config.
