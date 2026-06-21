---
name: generate-npa-workflow
description: Use when inventing a new npa.workflow/v0.0.1 pipeline from the tool catalog — creative stage graphs, loops, gates, and golden YAML output.
---

# Generate NPA Workflow

## When To Use

Load **after** `author-npa-workflow` when the task is to **design a new pipeline**
(not edit an existing golden spec). Use for creative mashups, customer demos, and
SkyPilot-to-spec conversions.

## Design Recipe

1. **Pick tools first** — only use `toolRef` values from
   `npa/src/npa/orchestration/npa_workflow/catalog.py`. Add catalog entries before
   inventing shell.
2. **Name the story** — one sentence in `metadata.description` (what flows where).
3. **Config layout** (beauty convention):
   - `bucket`, `prefix`, runtime knobs (`vlm_backend`, iteration counts)
   - blank line
   - URI keys grouped (`*_uri`) built from `s3://{{config.bucket}}/{{config.prefix}}/…`
4. **Graph patterns**:
   | Pattern | YAML shape |
   | --- | --- |
   | Linear chain | `next:` edges |
   | Fan-in deps | `needs:` (ordering hints only) |
   | Fixed repeat | `loop.max: "{{config.attr}}"` |
   | Dynamic exit | `loop.until: promote_checkpoint` |
   | Runtime branch | `transitions` + `writesDecision: true` on the decision state |
5. **Decision states** — any state that writes `config.decision_uri` must set
   `writesDecision: true` (never rely on a magic state name like `decide`).
6. **Terminal** — every completion leaf needs `terminal: true`.
7. **Validate early** — missing `{{config.*}}` and bad loop bounds fail at
   `validate-spec`, not at plan/execute.

## Beauty Checklist

- `apiVersion: npa.workflow/v0.0.1` + `kind: Workflow` at top
- Fold long descriptions with `>` under `metadata.description`
- One blank line between `config` runtime keys and URI keys
- `resources` profiles referenced by `states.*.resources`
- State `description` on every node
- `inputs` / `outputs` with `uri` + `schema` labels when artifacts cross stages
- Prefer `toolRef` over `run.shell`

## Creative Example (golden)

`npa/workflows/workbench/npa-workflows/tokenfactory-cosmos-gate.yaml` — Token Factory
reason → Cosmos augment → VLM critique loop with promote / re-augment gate.

## Generate + Verify

```bash
# 1. Write YAML under npa/workflows/workbench/npa-workflows/
# 2. Validate structure + tokens + cycles
npa/.venv/bin/npa workbench workflow validate-spec <new.yaml> --json

# 3. Plan (use --assume-decision when transitions exist)
npa/.venv/bin/npa workbench workflow plan-spec <new.yaml> \
  --run-id creative-demo --assume-decision loop_back --json

# 4. Register in tests — add filename to:
#    - npa/tests/orchestration/npa_workflow/test_spec.py parametrize
#    - npa/tests/smoke/test_npa_workflow_smoke.py parametrize
#    - skills/index.yaml npa_workflow_yaml smoke list (optional)

npa/.venv/bin/python -m pytest npa/tests/orchestration/npa_workflow/ \
  npa/tests/smoke/test_npa_workflow_smoke.py npa/tests/smoke/test_all_workflow_yamls.py -q
```

## Anti-Patterns

- Do not add sim2real `engine.py` stages — specs invoke catalog tools only.
- Do not use Jinja, `eval`, or shell for control flow.
- Do not create transition cycles — validation rejects unbounded graphs.
- Do not hardcode bucket/project IDs — use `example-bucket` placeholders.
