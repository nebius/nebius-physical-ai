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
2. Parse the YAML locally before launch.
3. Use `NPA_SKYPILOT_BIN` or `npa skypilot status --bin-path`; do not assume
   `sky` from `PATH`.
4. Submit through `npa workbench workflow` or the shared workflow submission
   helper.
5. Keep cleanup best-effort and avoid tearing down a shared controller unless
   the operator explicitly requests it.

## Three-Tier Contract

- CLI: `npa workbench workflow --help` and tool-specific `workflow` commands.
- SDK: use shared workflow submission helpers rather than shelling out from
  application logic.
- YAML: SkyPilot YAML under `npa/workflows/workbench/skypilot/` is the
  executable source of truth for resources, env, and task order.

## Gotchas

- SkyPilot `envs` does not support self-referencing interpolation.
- `sky jobs launch` does not provide a reliable dry-run path in the pinned
  version; test YAML parsing and mocked submission before live launch.
- Mixed serial and parallel task groups can be fragile; serialize when behavior
  must be deterministic.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

The smoke test invokes workflow help and parses representative workflow YAML.
