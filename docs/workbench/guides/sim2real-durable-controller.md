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
(`@sha256:`). Images contain the final NPA source and redistributable pinned
runtime dependencies. Runtime source archives, generated Python programs, and
best-effort `pip install` are forbidden. Image labels, Job annotations,
controller state, and ComponentRecords attest the same source SHA and runtime
digest. NVIDIA-proprietary Isaac wheels are the deliberate licensing exception:
the operator fetches them under its EULA in a CPU-only warm Job before the run,
never after a workload Pod receives a GPU.

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
URIs and their hashes remain in ComponentRecords. Before a unit-level commit,
each logical sibling execution also has a stable SHA-256-derived component I/O
prefix and Kubernetes Job base name. Re-entering an incomplete Stage 8 therefore
adopts the existing Job and exact output object instead of generating a new
attempt suffix. Stage 8 output can survive a controller restart and feed Stage 9
without repeating the model Job. Finalization resumes after its last committed
stage. In particular, it adopts the complete digest-verified Stage 7--11
ComponentRecord set instead of reconstructing those records from a prior Pod's
filesystem, and the Stage 11 durable unit embeds the candidate-checkpoint
handoff needed to recreate `candidate.json`. It never pairs metrics with a
render directory from another split or checkpoint.

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

The CPU controller distinguishes process termination used by Pod deletion,
eviction, and node shutdown from an application exit. Exit 137 or 143 is counted
against its bounded native restart allowance, after which the replacement Pod
resumes from the journal. Other nonzero exits fail immediately. This distinction
is controller-only: an Isaac/Cosmos process killed for out-of-memory remains a
component failure unless Kubernetes supplies a structured disruption condition.

Missing/malformed data, image or model failures, checkpoint load errors, and
component contract failures are application failures and are never retried on a
different GPU product. Isaac candidates remain RTX PRO 6000 or L40S only.
Runtime attestation and every scenario, robot asset, and resume-checkpoint stage
run under shell `errexit` before the trainer exit-code capture begins. A failed
stage therefore exits the Pod and cannot silently start a fresh policy.
Every resume download also carries the selected checkpoint's lowercase SHA-256.
An auto-discovered same-run checkpoint is hashed by the controller first; the GPU
Pod verifies those exact bytes while staging, and the update record preserves the
URI and digest as explicit lineage.

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

## Isaac runtime dependency closure

The public Isaac image contains no proprietary NVIDIA Isaac/Omniverse payload.
Before launch, an operator-accepted CPU Job runs the exact digest-pinned image
and atomically warms a `ReadWriteMany` PVC. Its cache key is the full SHA-256 of
the Isaac pins, wheel manifest, OSS dependency lock, Python ABI, and bootstrap
source. The published manifest records those digests and `.complete` is written
only after verification.

Preflight requires the named claim to be `Bound`, to have a concrete volume,
and to advertise `ReadWriteMany`. Every Isaac GPU sibling then mounts it
read-only at `/opt/isaac-cache` with bootstrap offline mode enabled. Runtime
attestation verifies the `current` symlink remains under `v/<sha256>`, the
completion marker and interpreter exist, and the manifest's bootstrap digest
equals the exact bootstrap in the running image. Provenance records the PVC,
cache stamp, bootstrap digest, and manifest digest. A missing, cold, mutable, or
wrong-version cache fails before simulation; the GPU Pod cannot download or
install a substitute.

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

Validation and gold evaluation receive distinct exact `envs.jsonl` S3 object
URIs. A replacement controller hydrates that object into its new local cache;
it never treats a previous Pod's local directory as durable state, discovers a
split by listing a prefix, or falls back from an unavailable gold object to a
validation directory. Reports preserve the exact scenario-record URI.

Gold remains sealed until the final configured Stage 10. Placement success is a
stable final placement strictly inside 5 cm; closest distance, reach, contact,
grasp, and lift are diagnostics only. A validation-only canary must show a
credible placement signal before the full 3x3 proof is submitted.

The scenario-bound Isaac task supplies a last-mile placement curriculum without
changing that verdict. It activates after a real 4 cm lift, transforms the
base-frame goal through the robot root exactly as Isaac's stock reward does,
provides an object-to-goal approach gradient only while the end effector remains
near the object, and limits the stillness incentive to the narrow target basin.
An episode-local signed potential rewards held-object approach and penalizes
departure; its first eligible sample is zero and reset returns it to infinity,
so a drop or reset cannot manufacture progress. Positive approach tapers to
zero over the final 5 cm before the strict boundary while negative departure
remains fully active. The exact `object_dropping` termination has a
time-step-scaled terminal penalty that ramps with first-pass goal difficulty.
The unchanged 5 cm / 0.03 m/s event advances an exact three-step dwell reward;
after the third step that reward remains saturated until the policy leaves the
strict event. Resumed exact-goal passes apply the full drop consequence
immediately. The first pass
interpolates each applied scenario from an 8 cm
lift directly above its real object pose to the exact recorded goal by 60% of
the configured steps; the remaining 40% and every resumed pass use only the
unmodified scenario goal. Training deliberately does not terminate after the
first three stable steps: strict validation requires the object to remain
stably placed at episode end, so continued saturated dwell reward teaches the
same sustained hold that validation and sealed gold measure. Immediate success
termination remains an explicit opt-in diagnostic only.
Its 0.35 m approach scale and weight 32 are deliberate: the first live validation
canary reached and contacted 3/3 objects and grasped/lifted 2/3, but its closest
target distances were still 0.205--0.364 m; the earlier 0.15 m, weight-8 term was
effectively flat there and left lift as the dominant objective. A follow-up
seven-checkpoint validation ladder proved that later training learned 3/3
reach/contact/grasp/lift and came within 0.057 m, but then moved away. A later
exact-goal resumed pass retained 3/3 reach/contact/grasp/lift and entered the
strict basin at 0.009 m, but all seven validation checkpoints still produced
zero stable placements: fast near-target motion merely lost the positive dwell
bonus. A negative narrow-basin term was tested and rejected because it taught
target avoidance. The current basin signal is positive-only, exposes braking
over the final 5 cm of approach, and combines a 0.20 m/s settling gradient with
the unchanged 0.03 m/s boundary. Departure is handled by the signed potential;
broad approach remains positive at transport distances and the strict evaluator
threshold is unchanged. The 0.15
m/s velocity gate added in the next canary was itself rejected by live evidence:
it suppressed reward while the object was transported and regressed validation
to 0/3 placement and 2/3 lift. A signed step-progress follow-up was also rejected:
drop/reset cycles exploited it, late training drop rate rose to 0.7866, and
validation regressed to 0/3 grasp and 1/3 lift. The explicit goal curriculum and
an unscaled drop penalty were then tested exactly at 500 iterations: the goal
curriculum reached fraction 1.0 with 35,412 true-goal assignments, but late drop
rate still reached 0.8269 and validation remained 0/3. Isaac's reward manager
multiplies every weight by the environment time step; the old weight `-50`
therefore contributed only about `-0.16` to late episode summaries and did not
counter the throw/drop shortcut. A subsequent from-scratch canary with weight
`-5000` eliminated late drops but also suppressed grasp/lift exploration and
collapsed to timeouts. The final schedule therefore starts the first pass at
zero, ramps to `-2000` with the goal curriculum, retains 20% of the dense
approach signal outside the hold gate, and accepts held-progress eligibility to
30 cm. The scheduled penalty subclasses Isaac Lab 2.3's stateful
`mdp.is_terminated_term` manager term, preserving its resolved Isaac termination
names and timeout filtering; it is never invoked as a free function. This
preserves early discovery while still making the measured late throw/drop
shortcut costly. The first PPO pass begins at the stock-like entropy coefficient
(`0.006`) and anneals to `0.0005` after 60% of its iterations, preserving early
grasp/lift discovery. A resumed policy has already crossed that exploration
wall, so it instead uses a dedicated convergence phase: `0.0005` to `0.0` after
20% of the pass with optimizer learning rate `0.0001`. All settings remain
operator-tunable and appear in runtime provenance. Validation-only Train16
confirmed the need for the sustained-hold contract: an immutable checkpoint
held the exact stable event for nine consecutive steps, then departed before
episode end and correctly scored zero strict success. Training now optimizes
the missing post-success interval rather than weakening that verdict.

The evaluator seals the first episode for each vector environment. Isaac resets
a completed environment inside `step()` before returning, so the returned scene
state can already belong to a second episode. The evaluator retains the last
pre-step sample for every newly terminal environment, stops all subsequent
metric updates for that index, and excludes post-reset frames from its render
manifest. Thus terminal strict metrics and their images always describe the
same validation or gold scenario rather than an arbitrary later auto-reset.

Validation-only Train17 then showed why completion and termination must be
separate controls. Its training telemetry contained nonzero strict dwell reward,
and checkpoint 4100 held a validation object inside 5 cm below 0.03 m/s for nine
steps, but moved again before the sealed episode terminal. Disabling success
termination had also removed the old termination-keyed completion bonus. The
task now always pays a one-shot bonus on the first exact three-step event without
resetting and keeps the saturated dwell reward while stable. A bounded
post-success departure penalty was tested as a retention aid; later live evidence
below showed that its delayed negative return instead made the rare success
avoidable, so it is no longer part of the task. Immediate success termination
remains an independent diagnostic opt-in; validation and gold still require the
unchanged stable terminal placement.

Validation-only Train19 then exercised that retention contract over another
12,288,000 training environment steps and evaluated seven immutable checkpoints
on all 64 validation scenarios. The last checkpoint entered the unchanged 5 cm
basin on 47 scenarios and ended inside it on 30, but no policy sustained two
consecutive sub-0.03 m/s steps. A bounded consequence for breaking an unfinished
one- or two-step dwell was tested next; it was zero during transport, before the
first stable step, while a partial dwell continued, after completion, and after
reset. This remained an experiment until live validation could distinguish
credit assignment from target avoidance.

Validation-only Train21 then trained another 12,288,000 environment steps with
that partial-break signal and evaluated seven immutable checkpoints over all 64
validation scenarios. Checkpoint 4400 reached the unchanged event and held it
for 16 consecutive steps, but left before the sealed episode terminal; the
remaining checkpoints still broke every partial event after one step. A follow-up
Train23 used the stronger event-gated consequence and a smaller `0.0001` resumed
PPO learning rate. All seven checkpoints still scored zero strict placements,
and every one regressed to a one-step maximum despite entering the 5 cm basin on
up to 54/64 scenarios. The negative transition therefore taught avoidance of
the low-speed state and is not part of the production reward.

Train25 removed that partial-break term and made the strict dwell reward
positive-only and quadratic over the unchanged
three-step counter: fractions `1/9`, `4/9`, and `1` at steps one through three,
then `1` for every continued stable step. Weight `4096` makes the second and
third steps materially more valuable than a fly-through while keeping the first
step a positive waypoint. Its seven-checkpoint validation64 sweep nevertheless
scored zero decomposed placements at every checkpoint. The first saved update
had already reduced Train21 checkpoint 4400 from a 16-step event to a one-step
maximum. The remaining exact-event-gated `-4096` post-completion consequence had
made all later unstable steps part of the return for reaching that rare success;
unless the policy could already hold to timeout, successful arrival was strongly
negative. That delayed consequence is therefore removed too. Retention is now
positive-only: saturated dwell pays each continued strict step, the one-shot
completion bonus remains larger, and the existing signed physical-progress term
still penalizes actual departure. The strict terminal evaluator contract is
unchanged. Legacy departure fields remain readable in durable PPO telemetry so
archived attempts retain their original meaning, but production no longer emits
or optimizes that reward term.

Train27 then isolated a different sampled-policy/served-policy gap. After the
positive-only retention change, exact PPO telemetry contained repeated
three-step completions, but the learned mean action-noise standard deviation
was still about `0.41`. RSL-RL samples that distribution during training while
validation and deployment execute the deterministic actor mean. Seven
validation64 checkpoints therefore still scored `0/64` strict terminal
placements (one checkpoint recorded a nonterminal placement event), despite
the stochastic completion samples. Resumed training now keeps its short
exploration segment, then sets and freezes the state-independent action noise
at the operator-tunable convergence value (default `0.05`) before the final PPO
segment. The exact update is verified at runtime and fails closed for an
unsupported policy representation. This makes the convergence rollouts train
near the policy that validation actually executes; it does not alter the sealed
split, 5 cm distance, 0.03 m/s speed, three-step dwell, or terminal-success
contract.

Train29 confirmed that convergence closed the sampling gap without closing the
physical last mile. Its final checkpoint brought 57/64 validation objects into
the unchanged 5 cm basin and four below 3 cm, but no trajectory retained the
0.03 m/s condition for three consecutive steps. Train31 reduced the configured
resume exploration radius again and evaluated seven immutable checkpoints over
the same canonical validation64 object. All seven remained `0/64` strict
terminal placements; the strongest checkpoint nevertheless held one object for
18 consecutive strict steps before driving it away. The learned actor had
therefore demonstrated the exact success state, while its unconstrained next
joint targets remained the source of terminal failure.

Eval32 and Eval33 rejected proximity-triggered retention as a valid solution.
Replaying the actor's last normalized action preserves an old moving target;
inverting Isaac's live affine action term does command the measured pose, but
doing so while the grasp is moving still disrupts the grasp. In Eval33 all 49
pre-success latches eventually left the goal, with a 22.2 cm median terminal
error. Those attempts remain non-authoritative and cannot satisfy the placement
canary.

Eval34 then delayed measured-pose retention until the learned actor itself had
completed the exact three-step event. The single qualifying validation row did
latch, but later ended 19.3 cm from goal; overall 51/64 objects ended inside 5 cm
without terminal stillness. Scripted retention is therefore removed completely.
Inference provenance and the validation canary now require `learned_actor_only`
and reject every post-actor controller. The remaining failure is addressed in
training through a positive near-goal arm-stillness objective that supplies a
directly controllable braking gradient without relaxing distance, speed, dwell,
split, or terminal-success semantics. The same immutable learned-actor path runs
in rollout, validation, and sealed gold Jobs.

Train32 then tested that objective from an independently hashed checkpoint for
500 PPO iterations (12.288 million environment steps) across all 512 training
scenario digests. The initial 0.15 rad/s `tanh` scale saturated at normal joint
motion: aggregate arm-stillness reward remained effectively zero. Even so,
actor-only checkpoint 5300 produced the first genuine validation placement:
one of 64 sealed-validation scenarios ended 5.7 mm from goal after 152
consecutive strict steps. The other six periodic checkpoints remained 0/64, so
that result is a valid signal, not sufficient efficacy. The scale is therefore
1.0 rad/s for the next training pass, retaining a positive-only gradient over
ordinary arm motion while leaving the 5 cm, 0.03 m/s, three-step verdict and
terminal evaluation semantics unchanged.

Train33 confirmed that the widened term was no longer saturated: its final
weighted contribution rose to about 10.0, 57/64 validation objects ended inside
5 cm, and 11/64 entered the strict speed basin. Checkpoint 5400 produced a
second genuine actor-only terminal placement on a different validation
scenario, but the canary rate remained only 1/64. The arm term was still weaker
than each broader placement contribution (about 28), so Train34 tested matching
its weight to the `4096` strict-dwell term. That setting made arm stillness
dominate at about 70 reward, regressed broad object-position error, and all
seven validation checkpoints fell to 0/64. The rejected run is preserved. The
final balance therefore retains weight `512`: it is the only tested setting
that both preserves transport and has repeatedly produced genuine actor-only
strict validation placements. This changes no policy action path or success
rule.

## Required integration ladder

Before a full run, the exact image and live target must prove:

1. API create/watch/delete and reconciliation from typed objects;
2. a running controller observing a registry Secret rotation and pulling a
   subsequent digest-pinned Job;
3. controller termination and restart between Stage 8 and Stage 9;
4. controller termination and restart during finalization;
5. Kueue saturation, pending Workload visibility, and later admission;
6. immutable image/source provenance;
7. CPU cache warm plus cross-node read-only Isaac cache attestation with no GPU
   network/bootstrap install;
8. validation/gold object and render separation;
9. configured-FPS RRD and MCAP timestamps;
10. nonzero, lineage-correct Stage 14 artifacts.

Mocks remain useful for pure state transitions, but production classification
tests consume real Kubernetes API objects and the ladder exercises the same
submit/controller path as the authoritative workflow.
