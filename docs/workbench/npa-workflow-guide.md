# NPA workflow guide (`apiVersion: npa.workflow/v0.0.1`)

Declarative state-machine specs for workbench tool pipelines. One format is consumed
three ways: YAML file, CLI, and Python SDK.

## Quick start

```bash
# Validate structure and closed toolRef / predicate registries
npa workbench workflow validate-spec npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml

# Expand loops/branches into a step plan (dry-run)
npa workbench workflow plan-spec npa/workflows/workbench/npa-workflows/sim2real-vlm-rl.yaml \
  --run-id demo --assume-decision loop_back

# Plan + optional scheduler hints + S3 run manifest
npa workbench workflow run-spec npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml \
  --plan-only --scheduler-plan --persist-state --json

# Submit an npa.workflow spec
npa workbench workflow submit npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml \
  --run-id demo --registry cr.eu-north1.nebius.cloud/<your-registry-id>

# Plan only (no submit) — inspect planned steps
# Token Factory (and other no-image tools) need NPA_SRC_S3_URI or --image
NPA_SRC_S3_URI=s3://<bucket>/npa-src/npa \
  npa workbench workflow submit npa/workflows/workbench/npa-workflows/token-factory-caption.yaml \
  --plan-only --run-id demo
```

Author and submit `npa.workflow/v0.0.1` specs under
[`npa-workflows/`](../../npa/workflows/workbench/npa-workflows/). See that
README for the full catalog.

**No-image tools** (Token Factory specs): set
`NPA_SRC_S3_URI=s3://bucket/prefix/npa` so the job can sync and install `npa`,
or pass `--image` to a workbench image that already includes it. `--plan-only`
does not mint or print live registry tokens.

Reference specs (all pytest-guarded):

| File | Shows |
| --- | --- |
| `vlm-eval-single.yaml` | Single `toolRef`, terminal state |
| `token-factory-caption.yaml` | Zero-GPU Token Factory caption |
| `tokenfactory-rollout-judge.yaml` | Serial two-tool chain with `inputs`/`outputs` |
| `sim2real-vlm-rl.yaml` | Nested loops + dynamic `transitions` |
| `bdd100k-pipeline.yaml` | AV failure-mode pipeline — ingest → backfill → train → eval |
| `av-night-scene-hardening.yaml` | AV night-scene hardening — fan-out into two per-view detector train→eval branches |
| `cosmos-synth-fanout-curation.yaml` | Fan-out Cosmos Transfer 2.5 synthetic-data shards → Voxel51 (FiftyOne) curation |
| `tokenfactory-cosmos-gate.yaml` | Creative reason → augment → VLM gate loop |
| `sonic-locomotion-finetuning.yaml` | Retarget → SONIC train → MJLab eval |
| `mjlab-eval.yaml` / `retargeting.yaml` / `sonic-*.yaml` / `cosmos3-reason.yaml` | Single-tool workbench specs |

## Document shape

```yaml
apiVersion: npa.workflow/v0.0.1
kind: Workflow

metadata:
  name: my-workflow

config:            # parameters; referenced by tokens
  bucket: my-bucket
  prefix: "runs/{{run.id}}"

resources:         # named profiles → scheduler hints
  gpu:
    cloud: kubernetes
    accelerators: H100:1

initial: first

states:
  first:
    toolRef: workbench.vlm_eval.run
    resources: gpu
    outputs:
      - uri: "s3://{{config.bucket}}/{{config.prefix}}/scores/"
    next: second

  second:
    terminal: true
```

## State mechanics

| Field | Purpose |
| --- | --- |
| `toolRef` | Cataloged workbench tool (preferred) |
| `run.shell` / `run.argv` | Ad-hoc command when no catalog entry exists |
| `next` | Linear edge to the next state |
| `sequence` | Ordered sub-states (optionally inside `loop`) |
| `loop.max` | Fixed iteration count (`int` or `config.attr`) |
| `loop.until` | Stop when predicate is true (`promote_checkpoint`) |
| `transitions` | Branch on predicates after the state runs |
| `needs` | Ordering hint only (validated acyclic; not enforced at runtime) |
| `writesDecision` | State writes `config.decision_uri`; engine reads S3 after this state |
| `inputs` / `outputs` | Artifact URIs + optional schema labels |
| `terminal: true` | End state |

## Tokens (no Jinja)

| Token | Meaning |
| --- | --- |
| `{{config.key}}` | Value from `config` |
| `{{run.id}}` | Run id from CLI/SDK |
| `{{run.prefix}}` | `{metadata.name}/{run.id}` or `config.prefix` |
| `{{state.NAME.uri}}` | Primary output URI recorded after state `NAME` runs |

## Predicates (closed registry)

| Name | True when |
| --- | --- |
| `promote_checkpoint` | Decision artifact says promote |
| `loop_back` | Decision artifact says loop back |

**Planning:** dynamic branches need `--assume-decision promote_checkpoint|loop_back` on
`plan-spec` / `run-spec --plan-only` because the full graph is not known until runtime.

**Execution:** with `--execute`, the interpreter walks the graph dynamically and reads
`config.decision_uri` from S3 after decision states (see `decisions.py`).

## Tool catalog

See `docs/workbench/npa-workflow-tool-catalog.md` and
`npa/src/npa/orchestration/npa_workflow/catalog.py`. Add new tools in Python, not by
inventing YAML fields.

## Runtime features (v0.0.1+)

| Flag / module | Behavior |
| --- | --- |
| `--persist-state` | Write `npa-workflow/manifest.json` + `status.json` under `config.prefix` |
| `--require-inputs` | Fail fast when declared input URIs are missing on S3 |
| `--scheduler-plan` | Emit portable per-step task docs (`resources`, `command`) |
| `run_workflow(..., execute=True)` | Dynamic traversal; not a static pre-built plan |
| `npa workbench workflow submit <npa.workflow.yaml>` | Plan the graph and launch the run |

`npa workbench workflow submit` on an `npa.workflow/v0.0.1` spec plans the graph
and launches it. Use `--plan-only` to inspect the plan without launching.
Parallel fan-out remains out of scope for v0.0.1.

### Live submit E2E

On an operator VM with Nebius credentials and `NPA_REGISTRY`:

```bash
# Cheap first: Token Factory CPU twins
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=cpu ./scripts/npa-workflow-submit-live-e2e.sh

# Full matrix
./scripts/npa-workflow-submit-live-e2e.sh
```

Matrix: `npa/src/npa/orchestration/npa_workflow/submit_matrix.py`
(19 twins across cpu / gpu / multi; stub twins are plan-only).

## SDK

```python
from npa.orchestration.npa_workflow import build_plan, load_spec, run_workflow

spec = load_spec("npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml")
plan = build_plan(spec, run_id="sdk-demo")
report = run_workflow(spec, run_id="sdk-demo", persist_state=True)
```

## Verify (same gates as CI / agent skill)

```bash
npa/.venv/bin/python -m pytest npa/tests/orchestration/npa_workflow/ -q
npa/.venv/bin/python -m pytest npa/tests/smoke/test_npa_workflow_smoke.py -q
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest npa/tests/e2e/test_npa_workflow_live_e2e.py -q
```

## What is intentionally out of scope (v0.0.1)

- Gang scheduling, parallel fan-out, runtime manifest-driven `foreach`
- JSON Schema validation of artifact payloads
- Unified `workflow status` for npa.workflow runs (sim2real path is separate today)

## YAML beauty conventions

- Group `config`: runtime knobs (`bucket`, `prefix`, backends, iteration counts), blank line, then `*_uri` keys.
- Fold long `metadata.description` with `>`.
- Every state gets a one-line `description`.
- Prefer `toolRef`; use `run.shell` only when no catalog entry exists.
- Decision states that write threshold JSON must set `writesDecision: true`.

`run.shell` resolves `config.*` tokens into `/bin/bash -lc` commands; treat spec files as trusted authored input.

Those advanced scheduling features belong in spec v0.0.2+ as explicit fields (`parallel`, `gang`, `foreach`), not Jinja.
