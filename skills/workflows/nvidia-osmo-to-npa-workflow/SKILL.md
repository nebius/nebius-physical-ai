---
name: nvidia-osmo-to-npa-workflow
description: Use when semantically translating a pinned NVIDIA OSMO workflow definition, or a related Airflow DAG, into a reviewable npa.workflow/v0.0.1 graph without claiming control-plane equivalence.
---

# NVIDIA OSMO to NPA Workflow

## Scope

Use this skill to translate the intent and execution contract of an authoritative,
pinned NVIDIA OSMO workflow into `npa.workflow/v0.0.1`. It also covers an
Airflow-on-Kubernetes definition when that is the authoritative NVIDIA source.
This is a review procedure, not an automated converter.

The result is a semantic translation: it must preserve real components, data
dependencies, failure behavior, and evidence expectations while explicitly
recording control-plane differences. Passing YAML parsing alone does not prove
equivalence. Do not claim that SkyPilot executes OSMO or Airflow, and do not
claim one-to-one equivalence for features NPA does not implement.

Read these companion skills before changing a workflow:

- `skills/workflows/author-npa-workflow/SKILL.md`
- `skills/atomic/real-components/SKILL.md`
- `skills/atomic/solution-licensing/SKILL.md`
- `skills/atomic/gpu-selection/SKILL.md`
- `skills/atomic/health-preflight/SKILL.md`
- `skills/atomic/submit-workflow/SKILL.md`
- `skills/workflows/emit-reviewable-rrd/SKILL.md` when Rerun evidence is required
- `skills/tools/artifact-viz-share/SKILL.md` when media or viewer handoff is required
- `skills/atomic/protect-nebius-infra-details/SKILL.md` before publication

For PAIDF, also read `skills/workflows/physical-ai-data-factory/SKILL.md` and
`skills/NOTICE-NVIDIA-PAIDF`.

## 1. Freeze the authoritative source

Before interpreting the graph, record:

1. Repository URL and full commit SHA.
2. Exact workflow file path at that revision.
3. License for the workflow definition and separate licenses or terms for every
   image, model, weight set, dataset, SDK, and generated artifact.
4. Referenced component repositories, image tags or digests, configuration files,
   model revisions, and data contracts.
5. Whether the source is actually OSMO, Airflow, or another controller. For
   example, a repository may be related to an OSMO ecosystem but publish only
   Airflow DAGs at the pinned revision.

Study the raw files at the pinned revision, including helper scripts that define
barriers, retries, service lifecycle, payload schemas, or output locations. Never
translate a moving branch or infer missing values from an unpinned README.

Add or update the repository notice with the source/revision/license mapping.
Where the workflow contract requires it, make the first NPA state write a
run-scoped provenance record such as `reports/upstream.json`. That record must
say which upstream sources informed the translation, which orchestrator NPA
actually used, and that NPA did not execute the upstream controller.

## 2. Build a semantic ledger

Create a review ledger before authoring YAML. Use one row per upstream task or
task group:

| Field | Required interpretation |
| --- | --- |
| Upstream identity | Exact group/task/DAG name and pinned source location |
| Real component | Executable tool, service, model, or library doing the work |
| Inputs | Data bytes, configuration, model/cache, and metadata dependencies |
| Outputs | Durable artifacts, schemas, manifests, and completion signals |
| Control | Predecessors, fan-out/fan-in, branch, loop, sensor, or barrier behavior |
| Resources | CPU/GPU count and class, memory, node topology, and backend constraints |
| Attempts | Retry count/delay, timeout, idempotency key, and partial-output policy |
| Failure | Fatal, fail-closed quality result, tolerated advisory result, or cleanup-only |
| Secrets | Named credential requirement and the smallest task scope that receives it |
| Image | Immutable source/runtime image identity and redistribution classification |
| Evidence | Files and fields that prove the real task executed successfully |

A state is ready to translate only when every column is known or an explicit,
reviewable gap is recorded. Do not replace an unknown with a plausible default.

## 3. Map the control plane

### OSMO

Map OSMO concepts by behavior:

| OSMO concept | NPA representation |
| --- | --- |
| Workflow task | `states.<name>` with a real `toolRef` or narrowly scoped `run.argv` |
| Task output reference | Typed `outputs` URI consumed by downstream `inputs` |
| Data backend URL | Operator-owned `config.*_uri`, normally under one run-scoped S3 prefix |
| Dependency or barrier | `next`, `needs`, `sequence`, or a real artifact gate |
| Parallel tasks | Independent states joined by a state that validates every required output |
| Iterative quality loop | Bounded `loop` plus `writesDecision` and explicit transitions |
| Pool/platform/GPU request | Named NPA resource profile selected with the GPU-selection rules |
| Named credential | Submit-time secret environment reference; never a value in YAML |
| Monitor/retrieve command | Standard NPA submit, status, logs, artifact readback, and resume path |

OSMO helper barriers often coordinate rank zero or wait for all shards. A plain
`needs` edge is not enough if the downstream task requires complete bytes: add a
real join/validation state or use a component that atomically publishes the
manifest only after all current shards exist.

### Airflow

Map Airflow concepts separately:

| Airflow concept | NPA representation |
| --- | --- |
| DAG / TaskGroup | Workflow graph / `sequence` or grouped states |
| KubernetesPodOperator | Real toolRef or adapter running the same component contract |
| XCom payload | Durable typed S3 JSON or manifest passed by URI |
| Sensor/readiness task | Bounded readiness state that fails closed |
| Branch operator | Decision artifact plus explicit `transitions` |
| `retries` / `retry_delay` | Real adapter retry with bounded attempts and immutable attempt evidence |
| DAG/task timeout | Component or submit/runtime timeout with a documented failure result |
| `all_done` cleanup | Explicit cleanup state only when NPA can guarantee equivalent triggering |
| Success/failure terminal task | Terminal validation state based on actual artifacts |

If NPA cannot reproduce an Airflow trigger rule, callback, SLA, scheduler pool,
or cleanup guarantee, state the gap in the YAML comments and guide. Do not encode
a weaker behavior and label it equivalent.

## 4. Preserve real components

Map each advertised upstream component to the cataloged NPA toolRef that actually
runs it. Inspect the tool implementation, rendered argv, routed image, and output
validator. A manifest writer, echo, copied sample, fabricated media file, or
reimplemented stand-in is not evidence that the upstream component ran.

When no real toolRef exists, stop and onboard the component under the workbench
tool and licensing procedures before authoring the workflow. Keep control-plane
glue narrow; it may validate or transform metadata but must not masquerade as the
named model, evaluator, curator, detector, simulator, or service.

## 5. Translate data and handoffs

Use one operator-owned run prefix, normally
`s3://{{config.bucket}}/{{config.prefix}}/`, and give every cross-state artifact
an explicit URI and schema. Preserve upstream distinctions among input data,
configuration, model/cache assets, service request payloads, raw generated media,
labels, evaluation reports, curated datasets, and final reports.

Prefer durable S3 handoffs over controller-local state. Replace OSMO task-output
references and Airflow XCom values with small typed manifests containing URIs,
hashes, counts, and producer identity. A downstream stage must validate the bytes
it consumes, not only trust the presence of a marker.

For fan-out work, publish under attempt- or shard-scoped prefixes and commit a
stable manifest only after all expected outputs validate. Keep prior attempts
append-only so runtime resume does not turn partial output into success.

## 6. Select execution backends and GPUs

Derive resources from the actual component rather than copying an upstream pool
label. Use hosted inference only for supported hosted APIs. Route CUDA workloads
to compatible Nebius GPUs, honor topology and multi-node requirements, and keep
CPU orchestration/curation on CPU where supported. Validate the exact accelerator
name on the target before submission.

Do not silently substitute a cheaper/different model or GPU family. Record any
intentional backend substitution in the semantic ledger, workflow description,
guide, and provenance artifact. For gated models, prove access to the exact
payload before provisioning GPU capacity.

## 7. Preserve retry, resume, and failure semantics

Classify every upstream failure path:

- Infrastructure or component failure remains fatal.
- A quality rejection is a typed fail-closed decision, not infrastructure success
  that can be promoted.
- An advisory check stays advisory only when the authoritative source says it is
  non-blocking.
- Cleanup cannot erase run evidence needed for diagnosis.

`npa.workflow/v0.0.1` does not make an arbitrary OSMO/Airflow retry declaration
equivalent by syntax. Implement bounded retries inside the real adapter only when
the upstream contract and idempotency rules are preserved; otherwise document the
gap. Each managed attempt needs an identity, immutable evidence, and an atomic
winning commit. Resume must use the standard durable workflow ledger and skip
only completed states whose outputs still validate.

Every loop must be bounded. Every leaf must be terminal. Test both accepted and
rejected decisions, mid-run interruption, and zero-launch resume where the runtime
supports it.

## 8. Handle secrets and images securely

Declare only secret names in the submit contract. Inject credentials at runtime
into the smallest set of tasks that need them; never place values in YAML, argv,
S3 manifests, logs, images, evidence, docs, or examples. Do not forward model or
registry credentials into offline child processes after staging completes.

Use immutable full-SHA development tags or digest-pinned images according to the
repository policy. Classify public, gated, EULA-bound, and operator-built private
artifacts separately. If upstream bytes cannot be redistributed, use an approved
runtime-fetch bootstrap or operator-built private derivative and verify the built
image bytes. Do not publish a new image when an unchanged, already-validated
immutable image supplies the required component.

## 9. Validate the translation

Run checks cheapest first:

```bash
npa/.venv/bin/npa workbench workflow validate-spec <workflow>.yaml --json
npa/.venv/bin/npa workbench workflow plan-spec <workflow>.yaml \
  --run-id osmo-translation-review --assume-decision promote_checkpoint --json
npa/.venv/bin/npa workbench workflow plan-spec <workflow>.yaml \
  --run-id osmo-translation-reject --assume-decision loop_back --json
```

Then run focused schema, catalog, toolRef argv, real-component, provenance,
licensing, docs/source-drift, and confidentiality tests. Register a runnable
workflow in `SUBMIT_LIVE_MATRIX`; register a dynamic graph in `DYNAMIC_SPECS`.
Add readiness metadata bound to the exact YAML hash. Follow the repository's
pre-PR validation skill for the remaining gates.

## 10. Require real-workload evidence

Before claiming runnable or translated:

1. Pass service, infrastructure, storage, and exact gated-payload preflight.
2. Submit the committed spec through the standard NPA workflow runtime with the
   actual reviewed components and immutable images.
3. Read back every required manifest and validate component-specific fields,
   source/model/image revisions, nonempty data, and terminal disposition.
4. Independently decode generated media rather than trusting filename or size.
5. Independently decode and inspect the Rerun recording when the workflow claims
   RRD evidence.
6. Preserve raw MP4 and `.rrd` as separate artifacts when both exist. The RRD may
   reference or visualize media; it is not a replacement for the raw media file.
7. Store a sanitized evidence summary outside the clone. Never publish bucket,
   project, cluster, registry, signed URL, credential, or private endpoint values.

Artifact discovery should work from the run prefix without hard-coded path
allowlists. Final reports should link the typed outputs; they must not collapse a
raw media artifact and its RRD visualization into one ambiguous record.

## Completion checklist

- Source and every dependency are pinned and attributed.
- OSMO/Airflow versus NPA/SkyPilot boundary is explicit.
- Semantic ledger has no unreviewed gaps.
- Every advertised component is real and image-routed correctly.
- S3 handoffs are typed, validated, and resume-safe.
- GPU/backend decisions match the component contract.
- Retry, rejection, terminal failure, cleanup, and resume behavior are tested.
- Secrets are name-only and least-scoped; image policy and licenses are satisfied.
- Validate/plan, focused tests, readiness hash, and required PR gates pass.
- A representative standard-runtime run produced independently validated output,
  separate media and RRD evidence where applicable, and sanitized external proof.

If any item is unresolved, describe the integration as a draft translation and
do not claim automated conversion, operational parity, or live acceptance.
