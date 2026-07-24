---
name: submit-workflow
description: Use when submitting, validating, or debugging NPA SkyPilot workflow YAMLs and workflow runner paths.
---

# Submit Workflow

## When To Use

Use this skill for workflow launch, YAML validation, runner scripts, and
SkyPilot submission behavior.

## Procedure

1. Read `skills/tools/skypilot-workflows/SKILL.md` for SkyPilot version and
   cleanup constraints.
2. Prefer `npa.workflow/v0.0.1` specs under
   `npa/workflows/workbench/npa-workflows/`. Parse / `validate-spec` locally
   before launch.
3. Use `NPA_SKYPILOT_BIN` or `npa skypilot status --bin-path`; do not assume
   `sky` from `PATH`.
4. Submit through `npa workbench workflow submit` (accepts npa.workflow specs
   and legacy SkyPilot YAML) or the shared workflow submission helper.
5. Keep cleanup best-effort and avoid tearing down a shared controller unless
   the operator explicitly requests it.

## Three-Tier Contract

- CLI: `npa workbench workflow --help` and tool-specific `workflow` commands.
- SDK: use shared workflow submission helpers rather than shelling out from
  application logic.
- YAML: prefer `npa.workflow/v0.0.1` specs under
  `npa/workflows/workbench/npa-workflows/` for authoring. `npa workbench
  workflow submit` accepts those specs (plans → renders → SkyPilot) and still
  accepts raw SkyPilot YAML under `npa/src/npa/workflows/skypilot/` for
  operator/runtime and SkyPilot-only exceptions (parallel, burst, runbook).

## Gotchas

- SkyPilot `envs` does not support self-referencing interpolation. The
  npa.workflow renderer resolves images and config before submit so rendered
  YAML has no `${VAR}` placeholders.
- `sky jobs launch` does not provide a reliable dry-run path in the pinned
  version; use `npa workbench workflow submit --plan-only` for npa.workflow
  specs, or mock submission before live launch.
- Mixed serial and parallel task groups can be fragile; serialize when behavior
  must be deterministic. Parallel sweeps stay SkyPilot-only in v0.0.1.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

The smoke test invokes workflow help and parses representative workflow YAML.
