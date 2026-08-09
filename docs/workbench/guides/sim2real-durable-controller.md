# Sim2Real durable controller and resume contract

Status: implementation contract for the canonical 14-stage controller. This is
an executable design, not a second workflow or a replacement for the real stage
engine.

## Canonical surface

`npa/workflows/sim2real.yaml` is a typed
`npa.sim2real/v1alpha1` controller specification. The existing workflow
submit detector validates that type and materializes one CPU controller Job.
The controller invokes the same preamble, outer/inner loop, and finalization
entrypoints used by tests. It dispatches the real GPU sibling Jobs; it is not a
catalog-tool facade and it is not accepted by the generic `npa.workflow` DSL.

The submitted controller and every sibling use an immutable registry reference
(`@sha256:`). Images contain the final NPA source and pinned runtime
dependencies. Runtime source archives, `git clone`, generated Python programs,
and best-effort `pip install` are forbidden. Image labels, Job annotations,
controller state, and ComponentRecords attest the same source SHA and runtime
digest.

## Durable identity and state

A controller execution is identified by this immutable tuple:

- run ID;
- Git source SHA;
- controller image digest;
- normalized controller-spec digest;
- task-contract digest.

Resume fails closed if any member differs. An explicit new attempt ID is needed
for different code or a different task contract.

The local state directory is a cache. S3 is the durable journal:

```text
<run-root>/state/controller/latest.json
<run-root>/state/controller/heartbeat.json
<run-root>/state/controller/checkpoints/sha256-<checkpoint-digest>.json
<run-root>/state/controller/records/{component,stage}/sha256-<record-digest>.json
<run-root>/state/controller/units/<unit>/<input-digest>.json
```

Each checkpoint is canonical JSON and content-addressed. It records the phase,
stage, outer and inner indices, input digest, output artifacts, immutable
ComponentRecords, Kubernetes Job references, checkpoint lineage, and the next
reconciliation target. A checkpoint object is uploaded before the mutable
`latest.json` pointer advances. A restarted controller downloads and verifies
the pointer and checkpoint, reconstructs required local files from their exact
S3 URIs, and continues at the first incomplete unit.

The reusable unit is intentionally smaller than an outer iteration:

1. the completed preamble (stages 1 through 6) as a workflow checkpoint;
2. Stage 7 rollout plus Stage 8 model results for one inner iteration;
3. Stage 9 signal conversion and PPO update as separate units;
4. each validation-checkpoint evaluation;
5. final gold Stage 10 and Stage 11 decision as separate units;
6. Stage 14 artifacts and the complete final report.

A completed unit is reusable only when its canonical payload SHA-256, immutable
controller identity, and input digest match the current unit. External artifact
URIs and their hashes remain in ComponentRecords. Stage 8 output can therefore survive a controller
restart and feed Stage 9 without repeating the model Job. Finalization resumes
after its last committed stage and never pairs metrics with a render directory
from another split or checkpoint.

## Kubernetes reconciliation

Normal Kubernetes state is read through the official Python client. Decisions
use typed Job and Pod fields:

- Job UID, generation, success/failure conditions and reasons;
- owning Job UID and deletion timestamp;
- PodScheduled condition and reason;
- container waiting reason;
- terminated reason, signal, and exit code;
- requested resources, selected node, actual image ID, and restart count;
- API exception status and reason;
- Kueue Workload admission conditions and assigned flavor.

The controller first looks up the deterministic Job name. A matching live Job
is watched; a matching completed Job is verified and reused; a name collision
whose UID/spec/source identity does not match fails closed. It never deletes and
recreates a healthy pending Job to probe capacity. The compatibility fallback
is limited to transport failures for which no HTTP status or API object exists;
it may retry the same read but may not infer scheduling or application state
from prose.

Every Job has a heartbeat/progress record. A zero Job deadline remains the
production default. Optional hang detection is operator-configurable and
observational until the threshold is exceeded; exceeding it records diagnostics
and stops the affected Job according to policy instead of silently holding a
GPU forever.

## Failure and retry policy

GPU Jobs use `restartPolicy: Never` because Kubernetes requires it with
`podFailurePolicy`, plus a nonzero native `backoffLimit`. The policy is:

- fail the Job immediately for a nonzero application-container exit;
- ignore a Pod with `DisruptionTarget=True` so a node/preemption disruption does
  not consume an application retry;
- count other infrastructure Pod failures up to the configured retry count;
- never add `activeDeadlineSeconds` unless the operator explicitly requests a
  deadline.

The CPU controller uses the same policy. In particular, a generic exit 1 is an
application failure and fails immediately; it is not relabeled as an
infrastructure retry merely because the CLI did not choose a specialized exit
code.

Missing/malformed data, image or model failures, checkpoint load errors, and
component contract failures are application failures and are never retried on a
different GPU product. Isaac candidates remain RTX PRO 6000 or L40S only.

## Kueue scheduling

The cluster uses the pinned Kueue release recorded in deployment evidence. A
ResourceFlavor selects the exact RTX PRO product label. A ClusterQueue owns the
real GPU quota plus CPU and memory quota for every resource the Jobs request, a
namespace LocalQueue is the submission surface, and a PriorityClass makes
controller-selected priority observable. Omitting CPU or memory from the queue
leaves an otherwise valid GPU Workload unadmittable and is a preflight failure.
GPU sibling Jobs carry the queue label and begin suspended for Kueue admission.
The controller watches the generated Workload and records admission, assigned
flavor, quota, and timestamps in provenance.

The controller service account therefore requires `list` on
`kueue.x-k8s.io/workloads` in the run namespace. The preflight checks that
permission explicitly. Authorization failures retain the Kubernetes API status
and reason and stop the run; they are not collapsed into an apparently missing
Workload or treated as contention.

Ordinary contention remains queued. Product fallback is permitted only after a
structured, terminal scheduling incompatibility for that flavor; queue waiting
or lack of current quota is not fallback evidence.

## Credential rotation

The long-lived controller does not depend on a registry credential captured at
process start. Its project-scoped Docker config Secret is mounted read-only at
`DOCKER_CONFIG`; both direct Docker `auths` entries and configured credential
helpers can therefore be materialized without requiring a `nebius` executable
or an injected short-lived token in the immutable controller image. Before each
sibling create, it reconciles that pull Secret through the Kubernetes API. A
rotated Secret is used by later Pods without restarting the controller.
Malformed direct credentials are not copied and fall through to a fresh-token
exchange. Storage credentials are mounted as a projected Secret and read for
new clients; an authentication failure triggers a single credential reload and
exact-operation retry. Tokens and secret bytes are never written to provenance.

## Visualization and evaluation lineage

Capture timestamps are `frame_index / configured_capture_fps` throughout RRD
and MCAP generation. Render lineage carries split, outer/inner/checkpoint
indices, checkpoint URI and SHA-256, scenario digest, and render artifact
digest. Gold metrics accept only a render manifest whose split is
`gold_heldout` and whose checkpoint/scenario lineage exactly matches the gold
report. Validation renders can never be selected lexicographically as a gold
fallback.

Gold remains sealed until the final configured Stage 10. Placement success is a
stable final placement strictly inside 5 cm; closest distance, reach, contact,
grasp, and lift are diagnostics only. A validation-only canary must show a
credible placement signal before the full 3x3 proof is submitted.

## Required integration ladder

Before a full run, the exact image and live target must prove:

1. API create/watch/delete and reconciliation from typed objects;
2. a running controller observing a registry Secret rotation and pulling a
   subsequent digest-pinned Job;
3. controller termination and restart between Stage 8 and Stage 9;
4. controller termination and restart during finalization;
5. Kueue saturation, pending Workload visibility, and later admission;
6. immutable image/source provenance;
7. validation/gold object and render separation;
8. configured-FPS RRD and MCAP timestamps;
9. nonzero, lineage-correct Stage 14 artifacts.

Mocks remain useful for pure state transitions, but production classification
tests consume real Kubernetes API objects and the ladder exercises the same
submit/controller path as the authoritative workflow.
