# NPA workflow guide (`apiVersion: npa.workflow/v0.0.1`)

Declarative state-machine specs for workbench tool pipelines. One format is consumed
three ways: YAML file, CLI, and Python SDK.

## Quick start

```bash
# Validate structure and closed toolRef / predicate registries
npa workbench workflow validate-spec npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml

# Expand loops/branches in the demo-only Sim2Real DSL fixture (dry-run)
npa workbench workflow plan-spec npa/workflows/workbench/npa-workflows/sim2real.yaml \
  --run-id demo --assume-decision loop_back

# Plan + optional scheduler hints + S3 run manifest
npa workbench workflow run-spec npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml \
  --plan-only --scheduler-plan --persist-state --json

# Submit an npa.workflow spec
npa workbench workflow submit npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml \
  --run-id demo --registry <your-registry>/<namespace>

# Plan only (no submit) — inspect planned steps
# Token Factory (and other no-image tools) need NPA_SRC_S3_URI or --image
NPA_SRC_S3_URI=s3://<bucket>/npa-src/npa \
  npa workbench workflow submit npa/workflows/workbench/npa-workflows/token-factory-caption.yaml \
  --plan-only --run-id demo
```

A successful submit prints the resolved run ID in text mode and returns it as
the top-level `run_id` in `--output-format json`. If the spec configures an S3
`bucket`, NPA also writes a run manifest under its resolved prefix. List those
runs later with the established durable-run command:

```bash
npa workbench workflow list \
  --s3-bucket <bucket> --workflow-s3-prefix <parent-prefix> --json
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
| `sim2real.yaml` | Canonical compositional 14-stage Sim2Real runtime; requires immutable component images and task-aligned S3 inputs for execution |
| `bdd100k-pipeline.yaml` | AV failure-mode pipeline — ingest → backfill → train → eval |
| `av-night-scene-hardening.yaml` | AV night-scene hardening — fan-out into two per-view detector train→eval branches |
| `cosmos-synth-fanout-curation.yaml` | Fan-out Cosmos Transfer 2.5 synthetic-data shards → Voxel51 (FiftyOne) curation |
| `tokenfactory-cosmos-gate.yaml` | Creative reason → augment → VLM gate loop |
| `sonic-locomotion-finetuning.yaml` | Retarget → SONIC train → MJLab eval |
| `groot-1-7-finetune.yaml` | GR00T N1.7 operational pipeline: deterministic real-data split, parameterized distributed optimizer smoke, immutable checkpoint, aligned offline inference, honest learning outcome, native RRD/MCAP, S3 publication, and deployed-agent viewer verification |
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
| `parallel` | Fan-out group: members launch concurrently (SkyPilot JobGroup) and the group's `next` state is the barrier |
| `maxConcurrency` | Cap on concurrent members of a `parallel` group (int or `{{config.attr}}`); larger groups are submitted in batches |
| `parallelCount` | Optional validate-time cardinality assertion (int or `{{config.attr}}`); rejects a config override that does not equal the explicit `parallel` member count before plan/submit |
| `params` | Per-state config overlay used when resolving that state's tokens (how N sweep members share one `toolRef`) |
| `trigger` | `{uri, pollSeconds, maxPolls, minObjects}` — the runtime driver waits for objects at `uri` before running this state |
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
| `npa workbench workflow submit <npa.workflow.yaml>` | Plan the graph, launch it, and return the resolved run ID |
| `plan-spec --waves` | Show the runtime wave shape (serial steps + parallel groups and their concurrency batches) |
| `submit --runtime` | Runtime orchestrator: submit each wave, poll it to terminal, read the real decision artifact from S3, then replan |

`npa workbench workflow submit` on an `npa.workflow/v0.0.1` spec plans the graph
and launches it. Use `--plan-only` to inspect the plan without launching.

### Runtime orchestrator (`--runtime`)

The default submit path is one-shot: it renders the flattened serial plan (loops
unrolled with `--assume-decision`) and launches it. That path is unchanged.

`--runtime` adds a driver that executes the graph wave by wave:

```bash
npa workbench workflow submit <spec.yaml> --run-id <id> --runtime \
  [--resume] [--poll-seconds 30] [--max-wait-seconds 3600] \
  [--retries 1] [--max-concurrency 2] [--no-cancel-on-timeout]
```

| Capability | Behaviour |
| --- | --- |
| Parallel fan-out | A `parallel:` group is rendered as a SkyPilot JobGroup (`execution: parallel`) so members run concurrently, batched by `maxConcurrency` |
| Barrier | The group's `next` state is submitted only after every member reached a terminal state |
| Real early-exit | After each loop iteration the driver re-reads `config.decision_uri` from S3; a promoting gate ends the loop instead of running the remaining budget |
| Data-dependent branching | `transitions` outside a loop body are resolved from the real decision artifact (`goto`) |
| Trigger / watch | A state's `trigger:` prefix is polled by the driver before its wave is submitted |
| Retry / resume | Every wave attempt is written to `<config.prefix>/npa-workflow/runtime.json` (`npa.workflow.runtime.v1`); `--retries` is the payload/terminal-wave retry count, while `--max-infrastructure-recoveries` is the separate finite typed-infrastructure recovery count (default 1, 0 disables it); `--resume` reconciles the exact durable history |
| Automated supervision | The CPU-side runtime observes the exact recorded SkyPilot job and Kubernetes pods, classifies stalls, cancels only an exact actionable configuration attempt, and recovers only a proven transient incomplete wave |
| Timeout | A positive `--max-wait-seconds` bounds each wave; `0` waits indefinitely. `--no-cancel-on-timeout` preserves a timed-out job as in-flight so `--resume` adopts it instead of submitting a duplicate |

Design notes: [`DESIGN.md`](../../DESIGN.md). Live evidence:
[`EVIDENCE.md`](../../EVIDENCE.md).

#### Durable run supervision and recovery

The supervisor is part of the standard workflow runtime, runs outside ephemeral
payload pods, and needs no GPU. SkyPilot remains the sole Kubernetes
orchestrator. The durable runtime ledger and content-addressed events under
`npa-workflow/supervisor/attempts/` are the source of truth; restarting the
driver with the same explicit run ID reconciles those records instead of relying
on process memory.

Before any initial or recovered launch, submit reuses the normal exact-image,
credential/access, accelerator-resolution, per-node GPU-shape, and gang-capacity
preflights. A recovered attempt is permitted only after all of those checks pass
again, the prior attempt's recorded workflow/source/image identity matches values
independently recomputed from the current spec, source selection, and digest pins,
declared S3 output evidence
is authoritative, and any live prior attempt is cancelled by exact provider ID
with terminal verification.

SkyPilot launch uses asynchronous API submission followed by exact-name/ID
reconciliation inside the crash-safe launch transaction. This allows production
supervision to observe genuinely Pending work instead of waiting inside the
submit command. Exact cancellation is also observed until terminal before a
recovery attempt may cross the provider boundary.

Machine-readable evidence distinguishes:

- `actionable_configuration`: image pull/auth/reference errors, missing
  Secrets/ConfigMaps, malformed pod configuration, and impossible accelerator or
  per-node GPU placement. Retry stops immediately and the exact attempt is
  terminalized with remediation.
- `transient_infrastructure`: provider interruption/preemption, node loss,
  capacity, Kubernetes transport/rate-limit/server failures. Recovery adopts an
  exact live attempt or records a new provider attempt for only the incomplete
  wave under the same NPA run ID. `--max-infrastructure-recoveries` bounds this
  path independently of `--retries`; exhaustion is a durable terminal decision.
- `payload`: the workload ran and failed, or claimed success without its declared
  outputs. Infrastructure retry is disabled.
- `unknown`: missing, conflicting, or ambiguous backend identity/evidence.
  Relaunch and fuzzy cancellation are blocked to prevent duplicates.

`npa workbench workflow status <run-id> --json` includes the latest supervisor
classification, recovery action, exact attempt identity, output/checkpoint
validation, preflight evidence, and remediation. Evidence is credential-redacted.
The shared Python contract also drives the existing production
`npa workbench genesis train-teacher --runtime serverless` Jobs path. That command
uses exact provider observation/cancellation, digest-resolved image identity,
content-addressed S3 supervisor history, declared-output validation, and the same
finite `--max-infrastructure-recoveries` policy. A recovery creates or adopts a
deterministically named provider attempt under the same logical run and verified
output/checkpoint prefix after process restart.

This does **not** enable per-stage mixed Kubernetes/Serverless routing in an
`npa.workflow/v0.0.1` graph. Workflow runtime waves remain SkyPilot/Kubernetes in
this change; the Serverless adapter is wired through the Genesis workbench command.

Checkpoint semantics are deliberately narrow. A completed wave may be reused
only when every declared non-empty S3 output validates. An incomplete wave is
restarted from its boundary by default. Mid-stage recovery is allowed only when
the tool explicitly supplies a compatible checkpoint loader and the checkpoint
is validated; tools without that implementation are reported as unsupported,
not checkpoint-resumable.

### Live submit E2E

On an operator VM with Nebius credentials and `NPA_REGISTRY`:

```bash
# Cheap first: Token Factory CPU twins
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=cpu ./scripts/npa-workflow-submit-live-e2e.sh

# Full matrix
./scripts/npa-workflow-submit-live-e2e.sh
```

Matrix: `npa/src/npa/orchestration/npa_workflow/submit_matrix.py`
(56 twins across cpu / gpu / multi; reviewed non-executable twins are plan-only).

## SDK

```python
from npa.orchestration.npa_workflow import build_plan, load_spec, run_workflow

spec = load_spec("npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml")
plan = build_plan(spec, run_id="sdk-demo")
report = run_workflow(spec, run_id="sdk-demo", persist_state=True)
```

## Grounded agent chat drafting

The NPA agent handles PAIDF and sim-to-real authoring on its deterministic,
zero-token grounded path. It selects a contract-aware template, applies values
from the natural-language request, then validates and plans the result before
returning YAML.

- PAIDF supports scenario count/GPU fan-out, augmentation subject, refinement
  passes, grade threshold, caption limits, and Cosmos Curator clip settings. Its
  graph invokes real Cosmos Transfer, Cosmos Evaluator, Cosmos Curator,
  FiftyOne, and Rerun stages.
- Sim-to-real supports robot/backend/task and input URIs, generated environment
  counts/splits, loop and rollout counts, held-out success threshold, and seed.
  It invokes `workbench.sim2real.run`, the maintained staged engine, instead of
  the legacy demo toolRefs.
- On a bootstrapped agent, `config.bucket`, Kubernetes target, and accelerator
  come from staged `~/.npa/config.yaml` and agent S3 settings. A user-requested
  accelerator that contradicts the configured profile fails closed. Missing S3
  or Kubernetes setup is surfaced as an explicit placeholder/warning; the agent
  does not invent a cluster or bucket.

Examples:

```text
Create PAIDF YAML for my robot clips: 6 variants on 4 GPUs, grade threshold 70%.
Create sim-to-real YAML for UR5e on Genesis with 12000 environments, 4 inner
iterations, 2 outer iterations, and an 82% held-out success threshold.
```

## Verify (same gates as CI / agent skill)

```bash
npa/.venv/bin/python -m pytest npa/tests/orchestration/npa_workflow/ -q
npa/.venv/bin/python -m pytest npa/tests/smoke/test_npa_workflow_smoke.py -q
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest npa/tests/e2e/test_npa_workflow_live_e2e.py -q
```

## What is intentionally out of scope (v0.0.1)

- Gang scheduling and runtime manifest-driven `foreach`
- Multi-step branches inside a `parallel:` group (members are leaf states)
- JSON Schema validation of artifact payloads
- A detached/daemonized `--runtime` service. The lightweight supervisor runs in
  the CPU-side runtime process and resumes durably through `--resume-run`; it is
  not deployed into ephemeral GPU payloads.

Parallel fan-out **is** supported as of the `parallel:` / `maxConcurrency` fields
above — the explicit-field direction this section originally deferred to v0.0.2.
They are optional and additive, so every pre-v0.0.1 spec is unaffected.

## YAML beauty conventions

- Group `config`: runtime knobs (`bucket`, `prefix`, backends, iteration counts), blank line, then `*_uri` keys.
- Fold long `metadata.description` with `>`.
- Every state gets a one-line `description`.
- Prefer `toolRef`; use `run.shell` only when no catalog entry exists.
- Decision states that write threshold JSON must set `writesDecision: true`.

`run.shell` resolves `config.*` tokens into `/bin/bash -lc` commands; treat spec files as trusted authored input.

Advanced scheduling stays in explicit fields (`parallel`, `maxConcurrency`,
`params`, `trigger`), never Jinja. `gang` and `foreach` remain unimplemented.
