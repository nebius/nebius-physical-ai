# Nebius Physical AI Workbench — Pipeline Authoring Guide

> Living document. Updated as new pipeline patterns are introduced.
> Last updated: 2026-08-03

## Overview

Repository Workbench pipelines are authored as **`npa.workflow/v0.0.1` specs**. A
spec is one YAML document that declares configuration, named resource profiles,
states, transitions, and artifact contracts. The workflow engine plans that
document and renders SkyPilot tasks for execution on Nebius.

SkyPilot remains the execution engine, not the repository authoring surface. The
old shipped multi-document SkyPilot workflow catalog is retired and guarded from
returning. `npa workbench workflow submit` still accepts customer-provided raw
SkyPilot YAML, and a few tool-specific single-task examples or resource profiles
remain in guarded locations, but new repository pipelines belong under:

```text
npa/workflows/workbench/npa-workflows/
```

The concise language reference is
[`docs/workbench/npa-workflow-guide.md`](workbench/npa-workflow-guide.md). This guide
uses
[`bdd100k-pipeline.yaml`](../npa/workflows/workbench/npa-workflows/bdd100k-pipeline.yaml)
as the longer service-backed example.

## Spec Structure

A minimal spec has an API version, kind, strict metadata, at least one state, and
an initial state when the graph has more than one possible entry:

```yaml
apiVersion: npa.workflow/v0.0.1
kind: Workflow

metadata:
  name: example-pipeline
  description: One tool stage followed by a terminal report.

config:
  bucket: example-bucket
  prefix: "example/{{run.id}}"
  vlm_backend: stub
  rollouts_uri: "s3://{{config.bucket}}/inputs/rollouts/"
  scores_uri: "s3://{{config.bucket}}/{{config.prefix}}/scores/"

resources:
  cpu:
    cloud: kubernetes
    cpus: 4
    memory: 16Gi

initial: score

states:
  score:
    toolRef: workbench.vlm_eval.run
    resources: cpu
    inputs:
      - uri: "{{config.rollouts_uri}}"
    outputs:
      - uri: "{{config.scores_uri}}vlm_eval_stub.json"
        schema: npa.workbench.vlm_eval.report.v1
    terminal: true
```

`metadata` rejects unknown fields. Keep historical lineage in `EVIDENCE.md`, not
in ad hoc metadata keys.

### Configuration and tokens

`config:` is the contract between the spec and its states. Values can refer to:

| Token | Meaning |
| --- | --- |
| `{{config.key}}` | Another configuration value |
| `{{run.id}}` | Run ID supplied by the CLI or SDK |
| `{{run.prefix}}` | Workflow/run prefix derived by the engine |
| `{{state.NAME.uri}}` | Primary output URI of an earlier state |

Tokens are deliberately limited; this is not Jinja and there is no `eval`.
Missing config keys fail validation.

Keep runtime knobs first in `config:`, then group the run-scoped `*_uri` values.
Cross-stage data always moves through object-storage URIs because stages do not
share a filesystem.

### States and toolRefs

Most states name a `toolRef` from
`npa/src/npa/orchestration/npa_workflow/catalog.py`. Its `argv_template` is the
command the engine resolves and runs. This keeps option names and required flags
auditable against the real CLI.

Use `run.argv` or `run.shell` only when there is no cataloged capability. Put
substantial logic in a tested package module and keep the state command small.

| Field | Purpose |
| --- | --- |
| `next` | Linear edge |
| `sequence` | Ordered child states, optionally inside a loop |
| `parallel` | Concurrent leaf states; the group’s `next` state is the barrier |
| `maxConcurrency` | Batch size for a `parallel` group |
| `params` | Per-state config overlay, useful for sweep members |
| `trigger` | Wait for objects at an input URI before submitting the state |
| `loop` | Bounded repetition, optionally ending on a decision predicate |
| `transitions` | Data-dependent branch selected from the decision artifact |
| `needs` | Validated acyclic ordering hint |
| `inputs` / `outputs` | Artifact URIs and optional schema labels |
| `terminal: true` | Successful leaf completion |

The closed decision predicates are `promote_checkpoint` and `loop_back`. A state
that writes `config.decision_uri` sets `writesDecision: true`.

### Serial groups and parallel fan-out

`sequence:` is ordered. The BDD100K reference groups three training states and
three evaluation states this way:

```yaml
train-models:
  needs: [curate-views]
  sequence: [train-rider, train-nighttime, train-distant]
  next: evaluate-models
```

For actual concurrency, use `parallel:` and submit with `--runtime`:

```yaml
sweep:
  parallel: [train-small, train-medium, train-large]
  maxConcurrency: 2
  next: rank
```

The runtime renderer emits a SkyPilot JobGroup for each parallel batch and waits
for every member before submitting the barrier state. Parallel members are leaf
states; multi-step branches remain explicit serial groups.

### Multi-node stages

Put `num_nodes` on a named resource profile:

```yaml
resources:
  gang:
    cloud: kubernetes
    accelerators: H100:1
    num_nodes: 2
```

The renderer emits the node count at the SkyPilot task level. SkyPilot
gang-schedules identical pods and supplies `SKYPILOT_NODE_RANK` and
`SKYPILOT_NODE_IPS`. See `multi-node-probe.yaml` for the smallest reference.

## BDD100K Reference

The BDD100K spec has thirteen states: eleven working states and two ordered group
states.

| Stage | States | Resource profile |
| --- | --- | --- |
| Ingest | `ingest` | `cpu` |
| CPU backfill | `backfill-cpu` | `cpu` |
| CLIP backfill | `backfill-clip` | `gpu-embed` |
| Materialized views | `curate-views` | `cpu` |
| Detector training | `train-models` → three `train-*` states | `gpu-train` |
| Detector evaluation | `evaluate-models` → three `eval-*` states | `gpu-eval` |
| Review | `review` | `cpu` |

The named profiles keep the repeated CPU/GPU shapes in one place. A stage says
what profile it needs; SkyPilot decides how to schedule that profile.

### Label-map configuration

Detection training is dataset-agnostic, so the category map belongs in the spec:

```yaml
config:
  detection_label_map: >-
    {"person":0,"rider":1,"car":2,"truck":3,"bus":4,"train":5,"motor":6,"bike":7,"traffic light":8,"traffic sign":9}
```

All three training and all three evaluation toolRefs read the same key. This is
important for BDD100K because `train` is a vehicle category; evaluation without
the map tries to interpret that label as an integer. For real BDD100K labels,
change the single map to use `pedestrian`, `motorcycle`, and `bicycle` in place
of `person`, `motor`, and `bike`.

`num_classes` is inferred as `len(label_map) + 1` for the background class. Only
set it explicitly when deliberately overriding that inference.

### Service endpoints

Service-backed stages use Kubernetes DNS:

```text
http://<service-name>.workbench.svc.cluster.local:<port>
```

| Tool | Service | Port |
| --- | --- | ---: |
| LanceDB | `npa-lancedb` | `8686` |
| Detection training | `npa-detection-training` | `8790` |

Deploy those services before a live BDD100K submit. ToolRef commands call the
real CLIs, which in turn call the services; specs do not embed `curl`/`jq`
request construction.

### Artifact paths

The spec derives every run-scoped URI from `bucket`, `prefix`, and `{{run.id}}`.
For example:

```text
s3://<bucket>/bdd100k-pipeline/<run-id>/lancedb/
s3://<bucket>/bdd100k-pipeline/<run-id>/training/<view>
s3://<bucket>/bdd100k-pipeline/<run-id>/eval/<view>/metrics.json
```

An `outputs:` entry is a promise to downstream consumers. Declare the path the
tool actually writes; when a tool exposes a `*_result_uri_for()` helper, use it
as the source of truth and extend `test_spec_declared_outputs.py` for new
toolRefs.

## Validate, Plan, and Submit

Run these from the repository root:

```bash
npa/.venv/bin/npa workbench workflow validate-spec <spec.yaml> --json
npa/.venv/bin/npa workbench workflow plan-spec <spec.yaml> --run-id preview --json
npa/.venv/bin/npa workbench workflow submit <spec.yaml> --run-id preview --plan-only
npa/.venv/bin/npa workbench workflow submit <spec.yaml> --run-id <run-id>
```

For a dynamic branch, add `--assume-decision promote_checkpoint` while planning.
For real parallel fan-out, triggers, or early exit based on an S3 decision
artifact, submit with `--runtime`. `--plan-only` never launches infrastructure.

The engine renders each planned state as a SkyPilot task. Setup is selected from
the toolRef, not copied into every spec: package extras, vendor interpreters,
source staging, image routing, and required run preambles are renderer concerns.

## Durable State

The spec runtime can persist `npa-workflow/manifest.json`, `status.json`, and the
runtime wave ledger under the run prefix. Use the generic workflow status/logs
and artifact commands instead of adding per-spec log upload code:

```bash
npa/.venv/bin/npa workbench workflow status "s3://<bucket>/<prefix>/"
npa/.venv/bin/npa workbench workflow logs "s3://<bucket>/<prefix>/" --stage <state>
npa/.venv/bin/npa workbench workflow artifacts "s3://<bucket>/<prefix>/"
```

## Isaac Lab: Spec Versus Resource Profile

The parallel sweep is an authored workflow spec:

```bash
npa/.venv/bin/npa workbench workflow submit \
  npa/workflows/workbench/npa-workflows/isaac-lab-rl-sweep.yaml \
  --run-id isaac-cartpole-sweep --runtime
```

The single-job files under `npa/src/npa/workflows/byof/profiles/` are guarded
BYOF resource profiles consumed by their runners, not a workflow catalog. They
remain valid inputs for those tool-specific paths, but should not be copied as
the starting point for a new pipeline.

Isaac Sim workloads require RT-core GPUs. Prefer L40S or RTX PRO 6000; H100 and
H200 do not provide the RT cores used by rendering/simulation.

## Adding a Pipeline

1. Start from the closest spec under
   `npa/workflows/workbench/npa-workflows/`.
2. Reuse an existing toolRef. If the needed capability is missing, add it to the
   workbench tool and catalog rather than embedding a second implementation.
3. Put runtime values and S3 URIs in `config:`; never depend on a repository path
   existing inside a task pod.
4. Declare only artifacts the implementation writes, with the implementation’s
   real schema.
5. Validate and plan the spec, including both assumed decisions when it branches.
6. Register it in `SUBMIT_LIVE_MATRIX`; use `plan_only=True` only for a stub,
   placeholder/reference component, or a separately covered onboarding flow.
   Do not use an arbitrary execution gap to avoid a live case: fix the gap or
   fail closed and exercise the real path. A reference implementation must not
   count as real GPU coverage. Add dynamic specs to `DYNAMIC_SPECS`, and seed
   inputs only for cases that actually execute.
7. Run focused offline tests, then the plan-only live-matrix preflight. Report a
   genuine environment blocker explicitly if a live launch is unavailable.

Do not add a raw workflow template under `npa/src/npa/workflows/skypilot/`; that
catalog is retired and its absence is guardrail-enforced.

## Related Documentation

- [`docs/workbench/npa-workflow-guide.md`](workbench/npa-workflow-guide.md)
- [`docs/workbench/npa-workflow-tool-catalog.md`](workbench/npa-workflow-tool-catalog.md)
- [`docs/workbench/cookbooks/bdd100k-pipeline.md`](workbench/cookbooks/bdd100k-pipeline.md)
- [`npa/workflows/workbench/npa-workflows/README.md`](../npa/workflows/workbench/npa-workflows/README.md)

## Changelog

| Date | Change |
| --- | --- |
| 2026-08-03 | Rewritten around the `npa.workflow/v0.0.1` authoring surface; SkyPilot is documented as the execution engine. |
| 2026-05-20 | Added Isaac Lab RSL-RL single-job and parallel sweep patterns. |
| 2026-05-16 | Initial guide and BDD100K label-map pattern. |
