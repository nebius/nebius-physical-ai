---
name: diagram-to-npa-workflow
description: Use when turning an architecture diagram plus a step-by-step write-up into a working npa.workflow/v0.0.1 YAML — parse boxes/arrows/decision-diamonds and numbered steps into states, loops, gates, and catalog toolRefs, then validate/plan until green. Generalizes across sim2real, AV perception, RL, and Cosmos pipelines.
---

# Diagram → NPA Workflow

## When To Use

Load when the input is a **picture of a pipeline** (architecture diagram, boxes +
arrows + decision diamonds) **and/or a numbered step write-up**, and the output
must be a runnable `npa.workflow/v0.0.1` spec under
`npa/workflows/workbench/npa-workflows/` (or a skill/example directory).

This skill is the **front end** that produces the graph. It composes two existing
skills — read them too:

- `skills/workflows/author-npa-workflow/SKILL.md` — spec contract + validation hardening.
- `skills/workflows/generate-npa-workflow/SKILL.md` — beauty conventions + design recipe.

Reference material for this skill:

- `reference/mapping.md` — keyword→`toolRef` table, control-flow patterns, the
  sim2real worked example, and discrepancy-reconciliation rules.
- `examples/sim2real-vlm-rl-from-diagram.yaml` — the 13-step sim2real diagram rendered as a spec.
- `examples/av-failure-mode-from-diagram.yaml` — a second, differently-shaped diagram (linear, multi-resource, no loop) proving the method generalizes.

## Inputs → Output Contract

| Input | You extract |
| --- | --- |
| Diagram image | Nodes (boxes), edges (arrows), decision diamonds, back-edges (loops), fan-in/out |
| Step write-up | Per-step: actor/tool, input artifact, output artifact, loop membership, gate/threshold |
| Domain knowledge | Which catalog `toolRef` implements each node; where real GPUs are needed |

Output: one `apiVersion: npa.workflow/v0.0.1`, `kind: Workflow` document that
passes `validate-spec` and `plan-spec`.

## Procedure

Work the seven steps in order. Do not skip validation.

### 1. Build the step table (data, not prose)

For every numbered step in the write-up, record a row:

`step# | short name | tool/actor | input artifact | output artifact | loop? | gate/threshold?`

Missing/duplicate step numbers are expected — the write-up and diagram will
disagree. Reconcile with the rules in `reference/mapping.md` ("Reconciling
discrepancies"): the **diagram is authoritative for topology** (what connects to
what, where the loops are); the **write-up is authoritative for intent**
(thresholds, why a step exists, which tool). Record every reconciliation as a
one-line `# note:` comment near the affected state so reviewers see the decision.

### 2. Read topology off the diagram

- Boxes → states. A rectangle is a `toolRef` (or `run`) state.
- Arrows → `next` edges (or `sequence` order inside a parent).
- Decision diamond → a state with `writesDecision: true` + `transitions`.
- Back-edge (arrow returning to an earlier box) → a **loop**, not a `next` cycle.
- Two nested back-edges → nested loops (inner fast, outer slow). See §4.
- Cylinders / buckets → artifact `inputs`/`outputs` URIs, not states.

### 3. Map each node to a `toolRef`

Use the keyword table in `reference/mapping.md`. Rules:

1. Prefer a cataloged `toolRef` from `npa/src/npa/orchestration/npa_workflow/catalog.py`.
2. If no tool matches, **add a catalog entry in Python first** (see
   `skills/workflows/author-npa-workflow`), then reference it — do not invent YAML fields.
3. Only fall back to `run.shell` / `run.argv` for genuinely ad-hoc glue with no
   reusable tool.
4. A step that is a data source/sink (a bucket, "place data in S3") is usually an
   artifact URI on a neighboring state, not its own state.

### 4. Encode control flow

| Diagram shape | YAML |
| --- | --- |
| A → B → C | `next:` edges |
| Ordered group under one parent | parent `sequence: [a, b, c]` |
| Fixed repeat (N iterations) | `loop.max: "{{config.iters}}"` on the parent |
| "Iterate until good" back-edge | `loop.until: promote_checkpoint` on the parent |
| Decision diamond (promote vs retry) | decision state `writesDecision: true` + `transitions` (`promote_checkpoint` → forward, `loop_back` → earlier state) |
| Fan-in ("needs both X and Y first") | `needs: [x, y]` (ordering hint only) |

**Nested loops** (the sim2real signature): an `outer` state with
`loop.max`/`loop.until` whose `sequence` contains an `inner` state that itself has
`loop.max`. Both parents get their own `sequence`.

**Loop-of-loops / real-world retrigger:** a back-edge that would return to the
*first* state (e.g. "retrigger Step 1 with real data") **cannot** be a graph edge —
`validate-spec` rejects unbounded control-flow cycles. Model it as a **terminal
record state** (a `retrigger`/`finalize` state that writes a manifest and sets
`terminal: true`). The real re-entry happens by launching a new run, not a YAML
edge. Note this explicitly (see the sim2real example's `finalize`).

### 5. Lay out `config` (beauty convention)

- First: `bucket`, `prefix: "<name>/{{run.id}}"`, runtime knobs (backends,
  iteration counts, thresholds).
- Blank line.
- Then every `*_uri` key, built from `s3://{{config.bucket}}/{{config.prefix}}/…`.
- Decision states need `decision_uri` + `default_decision` (for planning).

### 6. Emit the YAML

Follow the `author`/`generate` beauty checklist: `apiVersion` + `kind` on top,
folded `metadata.description` with `>`, one-line `description` per state,
`resources` profiles referenced by `states.*.resources`, `terminal: true` on every
leaf, `inputs`/`outputs` with `schema` labels where artifacts cross stages.

### 7. Validate, plan, iterate (mandatory)

```bash
npa/.venv/bin/npa workbench workflow validate-spec <spec>.yaml --json
npa/.venv/bin/npa workbench workflow plan-spec <spec>.yaml --run-id demo \
  --assume-decision loop_back --json          # add --assume-decision only if transitions exist
npa/.venv/bin/npa workbench workflow run-spec <spec>.yaml --run-id demo \
  --plan-only --scheduler-plan --json
```

Fix every error before moving on. Common failures and fixes are in
`reference/mapping.md` ("Validator error → fix").

## Generalization Checklist

The method is domain-agnostic. Confirm on a new diagram:

- [ ] Every box maps to a `toolRef` (or a newly-added catalog entry).
- [ ] Every decision diamond became a `writesDecision` + `transitions` state.
- [ ] Every back-edge became a `loop` (bounded) or a terminal retrigger record.
- [ ] No control-flow cycle survives (`validate-spec` passes).
- [ ] At least one `terminal: true` leaf on each branch.
- [ ] `plan-spec` emits a non-empty step plan.

## Sim2real: from spec to real learning

The v0.0.1 spec captures the **graph**; the sim2real GPU **engine** that produces
real weight updates is the staged runbook
`npa/workflows/workbench/sim2real/runbook.yaml` (see
`skills/workbench/sim2real-engine/SKILL.md` for the 14-stage map and
`skills/workflows/sim2real-operate/SKILL.md` to run it on a cluster). The produced
spec mirrors that engine one-to-one (augment → envgen → inner rollouts/VLM → heldout
eval → threshold gate → finalize).

"Real learning and progress" means: run the staged loop on GPUs and watch the
held-out `success_rate` (and reward trend) **climb across outer iterations** with
`trainer_source != reference` — a clean instant 1.0 is the stub, not success. Wire
a genuine trainer via `--byo-trainer-command` / `BYO_TRAINER_COMMAND`.

## Anti-Patterns

- Turning a retrigger/real-world back-edge into a `transitions` cycle (validator rejects).
- Inventing a `toolRef` instead of adding it to the catalog.
- Jinja, `eval`, or shell for control flow — use `loop`/`transitions`/`needs`.
- Modeling buckets/data stores as executable states.
- Hardcoding bucket/project/tenant IDs — use `example-bucket` + `{{config.*}}`.
