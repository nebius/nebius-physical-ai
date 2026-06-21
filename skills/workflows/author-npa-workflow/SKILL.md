---
name: author-npa-workflow
description: Use when authoring, validating, or reviewing NPA workflow specs (apiVersion npa.workflow/v0.0.1) — declarative state machines that invoke workbench tools via SkyPilot.
---

# Author NPA Workflow

## When To Use

Load when creating or editing **NPA workflow YAML** under
`npa/workflows/workbench/npa-workflows/`, wiring tool stages, loops, or
transitions, or when helping agents/users convert SkyPilot bash pipelines into
specs.

## Spec Contract

- **Guide:** `docs/workbench/npa-workflow-guide.md` (canonical examples + verify commands).
- **apiVersion:** `npa.workflow/v0.0.1` only (beta).
- **kind:** `Workflow`
- **States:** declarative nodes with `run` (shell/argv), `toolRef`, or `sequence`.
- **Tokens:** `{{config.key}}`, `{{run.id}}`, `{{run.prefix}}`, `{{state.NAME.uri}}` — no Jinja, no eval.
- **Predicates:** closed set: `promote_checkpoint`, `loop_back`.
- **Loops:** `loop.max: "{{config.attr}}"` or integer; `loop.until` for dynamic exit.
- **I/O:** `inputs` / `outputs` with `uri` + optional `schema` (documentation + future validation).

## Tool References

Use `toolRef` from the catalog in
`docs/workbench/npa-workflow-tool-catalog.md` and
`npa/src/npa/orchestration/npa_workflow/catalog.py`. Prefer toolRef over
inventing shell when a catalog entry exists.

## Commands (agent harness)

```bash
npa/.venv/bin/npa workbench workflow validate-spec <spec.yaml> --json
npa/.venv/bin/npa workbench workflow plan-spec <spec.yaml> --run-id demo --json
npa/.venv/bin/npa workbench workflow run-spec <spec.yaml> --plan-only --scheduler-plan --persist-state --json
```

For live infra (optional):

```bash
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest npa/tests/e2e/test_npa_workflow_live_e2e.py -q
```

## Authoring Rules

1. One workflow file = one variant; do not add sim2real-specific Python orchestrators.
2. Keep **terminal: true** on leaf completion states.
3. Use `--assume-decision` when planning dynamic loops (`loop_back` vs `promote_checkpoint`).
4. SkyPilot submits each planned step; the spec does not call `engine.py` or schedulers per workflow.
5. Cross-stage data uses S3 URIs in config — tools are stateless.

## Examples

| Spec | Purpose |
| --- | --- |
| `npa-workflows/vlm-eval-single.yaml` | Single-tool minimal |
| `npa-workflows/tokenfactory-rollout-judge.yaml` | Serial two-tool |
| `npa-workflows/sim2real-vlm-rl.yaml` | Fixed + dynamic loops |

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/orchestration/npa_workflow/ npa/tests/smoke/test_npa_workflow_smoke.py -q --tb=no
```
