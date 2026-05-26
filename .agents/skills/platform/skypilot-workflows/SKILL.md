---
name: skypilot-workflows
description: Use when authoring, reviewing, running, or debugging NPA SkyPilot workflows and runner scripts.
last_verified: 2026-05-26
owner: platform
version: 1.0.0
---

# SkyPilot Workflows

SkyPilot is the sole workflow orchestrator in this repo. Argo is deprecated; do not add or revive Argo workflows.

## Invocation

SkyPilot lives in an isolated virtualenv outside NPA's main Python environment. Invoke it through `NPA_SKYPILOT_BIN`; never rely on `sky` from `PATH`.

Use `npa skypilot bootstrap` to create or reuse the pinned SkyPilot `0.12.2`
venv, then set `NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"`.

The Kubernetes controller is the default path (`W9-skypilot-k8s-controller`). The VM controller exists only as a fallback.

## Known SkyPilot 0.12.2 Limits

- `envs` does not support self-referencing variable interpolation. Use explicit comment blocks for alternatives, following the `BDD100K_LABEL_MAP` pattern in `npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml`.
- `sky jobs launch` has no dry-run flag. Use mock-endpoint mode for validation before live submission.
- Mixed serial/parallel task groups in one YAML are not fully supported. Serialize the workflow if needed.

## Reference Pattern

- Canonical YAML: `npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml`.
- Runner script pattern: `npa/scripts/run_bdd100k_pipeline.py`, a thin wrapper around `npa.orchestration.skypilot.submit_workflow`.
- Isaac Lab runners follow the same shape through `npa/scripts/run_isaac_lab_rl.py`.

## Commit And Cleanup

Acquire `/tmp/npa-commit-lock/workflows-skypilot` before committing workflow files in parallel-run contexts.

Cleanup is best-effort and must not raise. `also_teardown_controller=False` is the safe default; only opt into controller teardown when no other run can be using it.

## Changelog

- 2026-05-26: Added frontmatter metadata (last_verified, owner, version) and Changelog section per skill-authoring.
