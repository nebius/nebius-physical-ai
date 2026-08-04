---
name: skypilot-workflows
description: "Use when running or debugging how the engine renders and submits SkyPilot from an npa.workflow spec: invocation, SkyPilot limits, JobGroups, multi-node tasks, runner scripts, and cleanup. For authoring, use author-npa-workflow."
---

# SkyPilot Workflows

SkyPilot is the workflow **execution engine** in this repo. Argo is deprecated;
do not add or revive Argo workflows.

SkyPilot is not the repository authoring surface. Pipelines are
`npa.workflow/v0.0.1` specs, and the engine plans those specs and renders
SkyPilot tasks. Use `skills/workflows/author-npa-workflow/SKILL.md` to author a
pipeline. The retired raw-task catalog must not return; raw SkyPilot input is
still accepted for customer-provided tasks and guarded tool-specific examples.

## Invocation

SkyPilot lives in an isolated virtualenv outside NPA's main Python environment. Invoke it through `NPA_SKYPILOT_BIN`; never rely on `sky` from `PATH`.

Use `npa skypilot bootstrap` to create or reuse the pinned SkyPilot `0.12.2`
venv, then set `NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"`.

The Kubernetes controller is the default path (`W9-skypilot-k8s-controller`). The VM controller exists only as a fallback.

## Known SkyPilot 0.12.2 Limits

- Raw SkyPilot `envs` does not support self-referencing variable interpolation.
  Repository specs use resolved `config` tokens instead.
- `sky jobs launch` has no dry-run flag. For a spec, use `workflow submit
  --plan-only` to inspect the rendered plan without launching.
- Mixed serial/parallel task groups are expressed in the spec. The runtime
  driver emits a SkyPilot JobGroup for each `parallel:` wave and submits the
  following state as its barrier.
- Managed-job Python API `Dag` support is effectively single-task for this repo's burst path. Use `npa burst submit-yaml` only for rendered single-task SkyPilot YAMLs; use `npa workbench workflow submit` for multi-stage workbench YAMLs.
- Direct Nebius burst jobs pull `resources.image_id` before YAML `setup` runs. For private `cr.*.nebius.cloud` images, the submitter must inject SkyPilot Docker login config (`SKYPILOT_DOCKER_SERVER`, `SKYPILOT_DOCKER_USERNAME`, `SKYPILOT_DOCKER_PASSWORD`) into task secrets before launch. `npa burst submit-yaml` does this by minting a short-lived Nebius IAM token when the submitter has Nebius credentials.

## What the Renderer Emits

`npa/src/npa/orchestration/npa_workflow/skypilot_render.py` turns a planned spec
into SkyPilot documents:

- A `parallel:` wave becomes a SkyPilot JobGroup; `maxConcurrency` splits a
  larger group into batches.
- `num_nodes` is emitted at the SkyPilot task level. SkyPilot gang-schedules the
  pods and exports `SKYPILOT_NODE_RANK` and `SKYPILOT_NODE_IPS`.
- Package extras, third-party requirements, source staging, and vendor
  interpreters are selected from the toolRef.
- A self-hosted service that must survive into the task command belongs in the
  run preamble, not in `setup`.
- Isaac-routed stages carry the operator-controlled NVIDIA EULA gate and fail
  closed when acceptance is absent.

## Live Debugging Traps

- A vendor image may put a stale npa tree on `PYTHONPATH`; the renderer stages
  the selected source first and runs the recorded interpreter.
- Operator wrappers must set `NPA_LIVE_WT` to an existing, durable worktree and
  fail before submission when it is missing. Never fall back to an ambient
  checkout: a vanished `/tmp` worktree once made the harness test another
  contributor's tree and report a misleading skip.
- The `NPA_SRC_S3_URI` overlay has no embedded provenance. Re-stage it after a
  source change before diagnosing a renamed flag in a live pod.
- An unresolved placeholder in rendered setup is rejected before submission.

## Reference Pattern

- Canonical spec: `npa/workflows/workbench/npa-workflows/bdd100k-pipeline.yaml`.
- Runner script pattern: `npa/scripts/run_bdd100k_pipeline.py`, a thin wrapper around `npa.orchestration.skypilot.submit_workflow`.
- Isaac Lab runners follow the same shape through `npa/scripts/run_isaac_lab_rl.py`.

## Commit And Cleanup

Acquire `/tmp/npa-commit-lock/workflows-skypilot` before committing workflow files in parallel-run contexts.

Cleanup is best-effort and must not raise. `also_teardown_controller=False` is the safe default; only opt into controller teardown when no other run can be using it.
