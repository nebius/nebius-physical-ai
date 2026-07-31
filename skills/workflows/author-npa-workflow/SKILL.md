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
npa/.venv/bin/npa workbench workflow submit <spec.yaml> --run-id demo --plan-only
npa/.venv/bin/npa workbench workflow submit <spec.yaml> --run-id demo
```

`submit` accepts both `npa.workflow/v0.0.1` specs and legacy SkyPilot YAMLs.
For npa.workflow specs it plans → renders serial SkyPilot YAML → `sky jobs launch`.
Use `--plan-only` to inspect the rendered YAML without launching. Dynamic
branches still need `--assume-decision`.

Live infra (required before merge):

```bash
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_npa_workflow_live_e2e.py \
  npa/tests/e2e/test_npa_workflow_live_infra.py -q
```

Live **submit** matrix (operator VM with SkyPilot + registry; burns GPU-hours):

```bash
# CPU-only first
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=cpu ./scripts/npa-workflow-submit-live-e2e.sh

# Full cpu+gpu+multi
./scripts/npa-workflow-submit-live-e2e.sh
```

Tmux full matrix (all npa.workflow YAMLs, real S3, credential leak checks):

```bash
./scripts/npa-workflow-real-infra-tmux.sh
```

## Authoring Rules

1. One workflow file = one variant; do not add sim2real-specific Python orchestrators.
2. Keep **terminal: true** on leaf completion states.
3. Use `--assume-decision` when planning specs with `transitions`.
4. `npa workbench workflow submit <npa.workflow.yaml>` plans the graph, renders
   a serial SkyPilot multi-doc YAML, and submits it. The spec does not call
   `engine.py` per workflow. Parallel fan-out stays on raw SkyPilot YAMLs.
5. Cross-stage data uses S3 URIs in `config` — tools are stateless.
6. Group config: runtime knobs first, then `*_uri` keys under `config.prefix`.
7. A `run.shell` state may import an npa module for logic that needs the package
   (npa is pip-installed in the task): `python3 -c "from npa.workflows.<mod>
   import <fn>; <fn>('{{config.a}}', '{{config.b}}')"` — prefer this over inlining
   heavy logic. Put testable logic in a real module (e.g.
   `npa/src/npa/workflows/data_factory_viz.py`) with a unit test.
8. A `toolRef` argv template must match the tool's **actual CLI option names**
   (e.g. `--input-uri`/`--output-uri`, not `--input-path`) and include required
   flags like `--run-id`; a mismatch validates/plans fine but crashes on real
   submit. Keep `catalog.py` and `docs/workbench/npa-workflow-tool-catalog.md` in
   sync.
9. **Live-infra is a priority** (`skills/atomic/testing-conventions/SKILL.md`): a
   new spec must be registered in `SUBMIT_LIVE_MATRIX`
   (`npa/src/npa/orchestration/npa_workflow/submit_matrix.py`); if it has a
   dynamic gate/loop, also add it to `DYNAMIC_SPECS` in
   `npa/tests/e2e/npa_workflow_live_helpers.py`. Use `plan_only=True` for
   stub/GPU-wasteful stages. Don't stop at smoke.

## Reference Examples

| Spec | Purpose |
| --- | --- |
| `vlm-eval-single.yaml` | Single-tool minimal |
| `tokenfactory-rollout-judge.yaml` | Serial two-tool |
| `sim2real-vlm-rl.yaml` | Nested loops + dynamic gate |
| `bdd100k-pipeline.yaml` | AV failure-mode LanceDB → train → eval |
| `av-night-scene-hardening.yaml` | AV night-scene fan-out — two per-view detector train→eval branches |
| `cosmos-synth-fanout-curation.yaml` | Cosmos Transfer 2.5 synthetic fan-out → Voxel51 (FiftyOne) curation |
| `tokenfactory-cosmos-gate.yaml` | Creative reason → augment → VLM gate loop |
| `physical-ai-data-factory.yaml` | NVIDIA Physical AI Data Factory (no OSMO): annotate → Cosmos augment → evaluate/validate gate → re-label → FiftyOne curate → Rerun visualize; toolRefs + `run.shell` glue; dynamic gate |

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/orchestration/npa_workflow/ \
  npa/tests/smoke/test_npa_workflow_smoke.py \
  npa/tests/smoke/test_all_workflow_yamls.py -q --tb=no
```

Tmux matrix: `./scripts/npa-workflow-creative-tmux.sh`
