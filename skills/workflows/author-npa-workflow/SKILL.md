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

For **new creative pipelines**, also load `skills/workflows/generate-npa-workflow/SKILL.md`.

## Spec Contract

- **Guide:** `docs/workbench/npa-workflow-guide.md` (canonical examples + verify commands).
- **Catalog:** `docs/workbench/npa-workflow-tool-catalog.md` +
  `npa/src/npa/orchestration/npa_workflow/catalog.py`.
- **apiVersion:** `npa.workflow/v0.0.1` only (beta).
- **kind:** `Workflow`
- **States:** declarative nodes with `run` (shell/argv), `toolRef`, or `sequence`.
- **Tokens:** `{{config.key}}`, `{{run.id}}`, `{{run.prefix}}`, `{{state.NAME.uri}}` — no Jinja, no eval.
- **Predicates:** closed set: `promote_checkpoint`, `loop_back`.
- **Loops:** `loop.max: "{{config.attr}}"` or integer; `loop.until` for dynamic exit.
- **Decision states:** `writesDecision: true` when the state writes `config.decision_uri`.
- **needs:** ordering hints only (validated acyclic; not enforced at runtime).
- **I/O:** `inputs` / `outputs` with `uri` + optional `schema`.

## Validation Hardening (v0.0.1)

| Check | When |
| --- | --- |
| Unknown `toolRef` / predicate | `validate-spec` |
| Unbounded transition cycles | `validate-spec` (loops do **not** whitelist cycles) |
| Missing `{{config.*}}`, bad loop max | `validate-spec` via token resolution |
| Forward `{{state.*}}` refs | Allowed at validate; resolved during plan/execute |
| Execution depth | Guarded at `--execute` (no stack blowups) |
| `run.shell` | Resolves config tokens; spec authors are trusted (injection risk if config is untrusted) |

## Commands

```bash
npa/.venv/bin/npa workbench workflow validate-spec <spec.yaml> --json
npa/.venv/bin/npa workbench workflow plan-spec <spec.yaml> --run-id demo --json
npa/.venv/bin/npa workbench workflow run-spec <spec.yaml> --plan-only --scheduler-plan --json
```

Dynamic branches: add `--assume-decision promote_checkpoint|loop_back`.

Live infra (required before merge):

```bash
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_npa_workflow_live_e2e.py \
  npa/tests/e2e/test_npa_workflow_live_infra.py -q
```

Tmux full matrix (all golden YAMLs, real S3, credential leak checks):

```bash
./scripts/npa-workflow-real-infra-tmux.sh
```

## Authoring Rules

1. One workflow file = one variant; do not add sim2real-specific Python orchestrators.
2. Keep **terminal: true** on leaf completion states.
3. Use `--assume-decision` when planning specs with `transitions`.
4. SkyPilot submits each planned step; the spec does not call `engine.py` per workflow.
5. Cross-stage data uses S3 URIs in `config` — tools are stateless.
6. Group config: runtime knobs first, then `*_uri` keys under `config.prefix`.

## Golden Examples

| Spec | Purpose |
| --- | --- |
| `vlm-eval-single.yaml` | Single-tool minimal |
| `tokenfactory-rollout-judge.yaml` | Serial two-tool |
| `sim2real-vlm-rl.yaml` | Nested loops + dynamic gate |
| `bdd100k-pipeline.yaml` | AV failure-mode LanceDB → train → eval |
| `av-night-scene-hardening.yaml` | AV night-scene fan-out — two per-view detector train→eval branches |
| `cosmos-synth-fanout-curation.yaml` | Cosmos Transfer 2.5 synthetic fan-out → Voxel51 (FiftyOne) curation |
| `tokenfactory-cosmos-gate.yaml` | Creative reason → augment → VLM gate loop |

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/orchestration/npa_workflow/ \
  npa/tests/smoke/test_npa_workflow_smoke.py \
  npa/tests/smoke/test_all_workflow_yamls.py -q --tb=no
```

Tmux matrix: `./scripts/npa-workflow-creative-tmux.sh`
