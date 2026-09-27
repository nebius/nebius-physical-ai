# Reusable BEHAVIOR campaigns

The BEHAVIOR campaign modules turn a baseline/candidate experiment into immutable
declarations, resumable workers, and complete-panel comparisons. Submit workers
through the standard Workbench workflow runtime; the internal stages retain
per-case progress and original artifacts in S3.

The main use case is a focused comparison on one or two tasks before spending
compute on a wider evaluation. The library accepts any nonempty subset of the
official 100-task registry, including all 100 tasks, so the same contracts scale
without changing case-selection rules.

## What a campaign freezes

`freeze_policy_identity()` requires immutable checkpoint and serving artifacts.
Each artifact has a SHA-256 digest and byte count. A change to either artifact
creates a different policy identity.

`declare_campaign()` freezes:

- the ordered official 100-task registry;
- the selected tasks in registry order;
- baseline and candidate policy identities;
- the existing public development cases, indices 10–19;
- the existing public reporting cases, indices 0–9;
- one deterministic worker partition for each policy panel; and
- a reviewed minimum paired-case threshold.

Development and reporting cases are always disjoint. Every task contributes all
ten prescribed cases to each split. The module has no API for removing a case
after observing its result.

```python
from npa.workflows.behavior_challenge.campaign import (
    aggregate_panel,
    bind_inspected_rollout,
    declare_campaign,
    freeze_policy_identity,
)

baseline = freeze_policy_identity(
    "stock-policy",
    {
        "checkpoint": {"sha256": "a" * 64, "bytes": 123},
        "serving": {"sha256": "b" * 64, "bytes": 456},
    },
)
candidate = freeze_policy_identity(
    "candidate-policy",
    {
        "checkpoint": {"sha256": "c" * 64, "bytes": 234},
        "serving": {"sha256": "d" * 64, "bytes": 567},
    },
)
declaration = declare_campaign(
    "radio-control-v1",
    official_task_ids,
    ["turning_on_radio"],
    baseline,
    candidate,
    worker_count=4,
    minimum_paired_cases=10,
)
```

The example hashes are synthetic. A real declaration must use the byte
identities verified during packaging.

Before allocating a simulator or policy process, an internal CPU stage can
check one frozen panel against its local checkpoint archive, serving source,
and policy settings:

```bash
python -m npa.workflows.behavior_challenge policy-identity-inspect \
  --panel-path /inputs/panel.json \
  --policy-kind official \
  --policy-archive /inputs/checkpoint.json
```

Pass the same policy flags that the campaign worker will receive. Add
`--train-experience` only for a prescribed non-reporting TRAIN panel. The
command emits one JSON identity receipt and does not create a workspace, claim
a case, start a simulator, or start a policy process. It verifies identity
only; campaign startup and admission checks still run in the worker.

## Prescribed non-reporting TRAIN panels

The `nonreporting_train` module freezes small TRAIN-only panels for descriptive
policy studies. Callers provide the task name, a typed TRAIN task-to-data
mapping with source, split, and mapping-manifest identities, prescribed
instances, an evaluator contract with a typed TRAIN argv identity, science
lineage, and an existing `npa.behavior.policy-identity.v1` binding. The module
does not derive cases from the development or reporting split tables. The
current one-rollout evaluator writes only rollout ID 0, so the declaration
rejects every other rollout ID until the public runner supports and tests a
different original-artifact layout.

The resulting protocol, panel, deterministic partition, case-record aggregate,
and study each carry content-derived identities. `aggregate_train_panel()` is a
pure declaration helper: it accepts only a complete set of inspection-shaped
records, but it does not open their files. Its receipts therefore say
`npa.behavior.caller-supplied-rollout-record.v1` and keep
`artifact_bytes_verified_by_aggregator` false. `aggregate_train_outputs()` is
the executable counterpart: it requires the exact case-ID-to-directory mapping,
opens every original JSON, fully decodes every video through `inspect_rollout()`,
and emits the `npa.behavior.inspect-rollout.v1` contract with the verification
flag true. The protocol binds source and argv artifact identities; constructing
it alone does not verify those bytes.

Before a TRAIN worker loads a policy or claims a case, it verifies the pinned,
clean BEHAVIOR checkout, requires the official
`omnigibson.eval.wrappers.RGBDFullResWrapper`, and matches the declared robot
configuration to `OmniGibson/omnigibson/eval/r1pro.yaml` byte for byte. Generic
dotted evaluator declarations remain valid as portable protocol data, but the
production TRAIN runner accepts only this source-bound official invocation.

`rank_train_study()` compares any declared number of complete, file-backed
policy aggregates. It rejects pure caller-supplied aggregates. It uses ascending
primary/secondary calibration loss and descending mean Q/success count, with
shared midranks for complete-key ties. Its Kendall tau-b, Spearman midrank
correlation, and Pareto relations are descriptive. The result cannot select or
promote a policy.

```python
from npa.workflows.behavior_challenge.nonreporting_train import (
    aggregate_train_outputs,
    declare_train_panel,
    declare_train_protocol,
    partition_train_panel,
)

protocol = declare_train_protocol(
    task="your_train_task",
    task_mapping=verified_task_mapping,
    prescribed_cases=[{"instance_id": 0, "rollout_id": 0}],
    science_lineage=verified_science_lineage,
    evaluator_contract=verified_evaluator_contract,
)
panel = declare_train_panel(protocol, complete_policy_binding)
partition = partition_train_panel(panel, worker_count=1)
aggregate = aggregate_train_outputs(panel, case_id_to_downloaded_output_directory)
```

This library surface is separate from existing development/report panel
validation. The existing internal `campaign-worker`, `campaign-aggregate`, and
`campaign-status` commands dispatch by panel schema. TRAIN panels reuse the same
`CaseStore` claim/start/complete transitions, raw failure preservation, and
immutable-original recovery. Each unstarted case gets a fresh managed-policy
context. The aggregate command re-downloads, hashes, and decodes every original
before it emits the file-backed TRAIN aggregate. Existing development and report
panels continue through their original validators and evaluator argv unchanged.

`train_evaluator_argv()` renders one declared instance with literal
`--mode train`, one rollout, video enabled, and no max-step override. TRAIN
execution accepts only the `comet-native` policy kind. Existing Comet and RLC
policy kinds retain their development/report split restrictions and fail before
simulator startup or case claims.

The native adapter requires a provider-read
`npa.behavior.comet-native-verified-checkpoint.v1`. That record comes from an
actual restore of the complete optimizer/TrainState and serving parameters, an
exhaustive typed architecture partition, equality of every frozen leaf with its
parent value, and a discarded finite probe. The adapter does not assume one
leaf count for all native checkpoints: action-expert and full-SFT receipts carry
their own exact path, shape, dtype, byte, and value-hash inventories. The
qualified checkpoint tree also includes the normalization assets required by
the pinned native loader under the selected manager step. Asset identifiers are
canonical relative paths, including upstream identifiers such as
`behavior-1k/2025-challenge-demos`; absolute paths, traversal, escapes, and
encoded separators are rejected. A private training
publisher must translate its own terminal and resume records into this generic
contract; it cannot label them as the public native milestone schema. Every
protocol and native input is opened and hashed before simulator startup,
`CaseStore`, or provider mutation.

Each case gets a discarded checkpoint-load qualification followed by a fresh
serving process. Its seed is derived from the frozen RNG contract and case
identity. The server assigns the JAX key explicitly, verifies the loaded
policy's real key, and requires exactly one key split per inference. An optional
serving-identity-bound JSONL trace records the actual sent 23-vector, command
indices 14 and 22, and the observed gripper proprioception values without image,
token, prompt, hidden-state, or parameter payloads. These traces are diagnostic
and do not affect scores, ranking, selection, or report eligibility. Finalization
revalidates every row against the immutable process identity, contiguous action
and inference ordinals, redundant gripper fields, and RNG
transitions; malformed partial diagnostics remain visible and cannot become a
successful final receipt. The server records successfully sent actions in one
append-only, fsynced process-progress JSONL file, so a long rollout does not
create one provider object per action. The immutable ready and final receipts
bind that journal. Real model execution still requires an
operator-qualified checkpoint bridge and runtime; the public adapter does not
manufacture that evidence.

The `comet-trained` policy kind serves a selected BF16 export after a separate
two-process GPU parity qualification. Its admission binds the parity terminal,
producer-valid selection, score, precision, runtime, and milestone receipts,
both raw action archives, every exported Orbax `params/` member, the exact
normalization asset, the RNG contract, and the adapter serving identity to
provider-read bytes. The operator materializes those admitted params under
`<checkpoint>/<manager-step>/params/` and the normalization file under
`<checkpoint>/<manager-step>/assets/<asset-id>/norm_stats.json`, then supplies the
provider-read parity originals with `--policy-trained-input-root`. Before
simulator startup or case claims, the worker checks the complete tree and the
frozen panel. DEV and REPORT retain their existing panels. TRAIN recording uses
a distinct non-reporting panel and binds the recorder sources into a new serving
identity. Each case performs a discarded real checkpoint load before starting a
fresh selected-policy process.

This serving tree is a derivative inference artifact. It is not a complete
training-manager checkpoint and does not claim optimizer state, `train_state`,
or resumability. It cannot be passed to the `comet-native` TRAIN admission path;
TRAIN recording reuses only its qualified inference admission.
Parity admission also does not claim rollout quality: candidate identity and
the exact panel must be frozen before any DEV case is read.

The selected DP4 producer emits milestone schema v2 with a compact partition
`state_contract`; the generic native-training milestone uses schema v1 with
per-leaf FP32 and AdamW maps. The trained adapter reuses the shared complete
checkpoint topology check, then validates the v2 cursor, partition totals,
provider-member closure, and source-bound static reconstruction directly. It
does not translate the selected receipt into the generic resumable-training
contract.

For managed workers, construct the serving artifact with
`serving_identity.serving_artifact(args)`. It hashes the actual adapter files,
policy kind, execution variant, per-episode process lifecycle, and optional
correlation/export receipts.
The worker recomputes that identity and checks the checkpoint archive bytes
before starting the policy. This prevents a changed controller from inheriting
an earlier policy's completed cases.

### Package frozen workflow code

Use `build_manifest_archive()` to package an already reviewed `MANIFEST.json`.
It copies only the declared payloads, checks their bytes and permissions, and
verifies the finished archive before publishing it to a new local path. Source
files and directories must be regular files and real directories; symlinks and
special files are rejected. The output does not include unlisted files, macOS
resource forks, local ownership, or timestamps. Identical manifest bytes,
payloads, permissions, and archive root produce identical archive bytes.

```python
from pathlib import Path
from npa.workflows.behavior_challenge.package_archive import build_manifest_archive

build_manifest_archive(
    Path("/private/reviewed-worker"),
    Path("/private/worker.tar.gz"),
    expected_root="reviewed-worker",
    expected_manifest_sha256=approved_manifest_sha256,
)
```

Freeze the returned archive's digest in the workflow inputs. The worker can then
call `extract_manifest_archive()` with that same explicit root and manifest
digest. A changed source file requires a new reviewed manifest; an existing
output is never replaced.

Released-specialist reporting adds independently frozen receipts and their
expected SHA-256 values. Before constructing the durable case store, the worker
downloads every development and baseline original, verifies its declared hash,
fully decodes every video, rebuilds each aggregate, and recomputes the frozen
baseline selection. Candidate selection must bind only that verified development
evidence; baseline unsealing and report authorization are later, separately
hashed steps. The worker also requires a free loopback policy port before any
claim. Receipt paths authorize the campaign; they are excluded from the
specialist runtime identity and never enter the policy command.

The opt-in negative live regression uses an owner-only JSON file selected by
`NPA_BEHAVIOR_SPECIALIST_ADMISSION_LIVE_CONFIG`. It invokes the internal
`campaign-worker` CLI against real panel and partition objects, supplies a fresh
empty S3 prefix, and proves missing or mismatched authorization exits before any
case state, worker receipt, policy process, or evaluator output. The config holds
the declaration URIs and the pinned upstream source path; the test does not read
sealed baseline results or create positive report authorization.

Create the private file outside the repository, make it owner-readable only,
and use the exact upstream source checkout:

```json
{
  "panel_uri": "s3://<bucket>/<prefix>/reporting-panel.json",
  "partition_uri": "s3://<bucket>/<prefix>/reporting-partition.json",
  "absence_prefix": "s3://<bucket>/<fresh-negative-test-prefix>",
  "upstream_root": "/path/to/BEHAVIOR-1K"
}
```

Run only the negative admission regression:

```bash
chmod 600 /path/to/specialist-admission-live.json
NPA_BEHAVIOR_SPECIALIST_ADMISSION_LIVE_CONFIG=/path/to/specialist-admission-live.json \
  NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_behavior_specialist_report_admission_live.py -q
```

## Conditional replanning

The [conditional replanning library](../../npa/src/npa/workflows/behavior_challenge/conditional_replan/README.md)
provides a training-data-only refresh decision for a native 32-action policy.
After 16 actions, it compares the remaining queue with a new proposal. Accepting
the proposal replaces the queue; rejecting it restores the policy's random
state so the original continuation remains reproducible.

Its fitting, feature extraction, artifact export, and controller are reusable
Python components. Callers supply the native policy adapter and verified input
artifacts. Bind the fitted artifact and controller source to a new policy
identity before evaluating it. Fit and serving parity checks establish numerical
consistency; complete development and reporting panels establish task performance.

## Durable execution and recovery

The internal `campaign-worker` stage takes `--panel-uri`, `--partition-uri`,
`--worker-index`, and the existing managed-policy/runtime arguments. Required
`--output-path` is a stable S3 campaign prefix that must remain the same across
resumes. `--workspace` is a persistent worker directory. Required
`--worker-receipt-uri` is a unique output location for this workflow invocation.
Run and monitor it with `npa workbench workflow submit`, `status`, `logs`, and
`artifacts`, as with other Workbench workflows.

When workers use separate workspace paths, give each `worker_slots` entry a
`simulator_startup` object containing either `spec` or `receipt`, with an
absolute container path. For example:

```json
{"simulator_startup": {"receipt": "/campaign/worker-0/simulator-startup.json"}}
```

The startup specification's `apps.owner_root` and the receipt's evaluator
context must name that worker's workspace. The worker checks this binding before
reading campaign inputs or preparing a policy. Pin the specification and receipt
identities in the enclosing workflow inputs. See the
[startup qualification contract](behavior-challenge.md) for creation and reuse.

Before scheduling simulator GPUs, use the CPU-only
[simulator source inspector](behavior-simulator-source-inspect.md) to observe
the retained installation's exact application files and extension directories.
Its JSON output supplies the startup spec's `apps` object and validates it with
the same source checker used by simulator startup.

The per-case ledger uses atomic conditional S3 writes:

1. A worker claims a prescribed case. A replaced pre-start owner cannot start it.
2. It starts a fresh policy process for that case, waits for readiness, then
   commits `started` immediately before invoking the unchanged evaluator.
3. After evaluator success, it validates the original JSON and fully decodes the
   MP4, stops the policy, and preserves case-specific policy and evaluator logs.
   It uploads original artifacts without overwriting conflicting bytes, then
   marks `complete`.
4. A resumed worker downloads and verifies completed cases. It can recover a
   started case from its original uploaded artifacts or successful evaluator
   output retained on the worker volume. It never invokes that case again.

If a started case has no recoverable original evidence, recovery stops and
identifies the case. Creating a new output prefix to repeat it would defeat the
protocol. Keep the original volume and logs for investigation.

When evaluation or result validation fails, the worker also preserves any
available prescribed metrics JSON and MP4 under the claim's `raw/` prefix.
Conditional uploads and full byte readback precede `raw/manifest.json`, which
records their hashes and explicitly marks them unvalidated. This diagnostic
manifest neither completes the case nor authorizes recovery or another rollout.
If storage fails, keep the original worker volume; logs report the failed upload.

Each case starts with the policy's native initial random state. A websocket
reset alone does not reset the RLC sampler's random generator. Fresh processes
keep the sampler state independent of preceding cases, worker assignments, and
resumes. A policy startup failure leaves the case reclaimable; recovering an
original completed rollout does not start a policy process.

Pure planning supports any official task subset. A managed worker currently
serves one task per panel; use separate task panels for multiple tasks.
The internal `campaign-aggregate` stage takes `--panel-uri`, `--output-path`,
`--workspace`, and `--receipt-uri`. It requires complete coverage, downloads and
hashes every original artifact, and fully decodes every video. Its verified-panel
envelope contains the pure aggregate plus explicit byte-verification evidence.

Live coverage is in `npa/tests/e2e/test_behavior_challenge_live.py`. Set
`NPA_BEHAVIOR_CAMPAIGN_LIVE_CONFIG` to a private JSON file containing the worker
arguments in an authorized simulator runtime. The test executes or recovers
that exact partition and checks every assigned case has a completion receipt.

### Repeated runtime preparation

Simulator and policy preparation may both request the same runtime in one
worker. `prepare_runtime` reuses an existing Python base only when its complete
file tree, contents, and permissions match the verified cache. It preserves
those files and rejects changed contents or symlinks. A fresh worker restores
the base from the cache; placing a virtual environment on a persistent volume
does not also persist its base interpreter outside that volume.

The CPU-only live transport test uses real S3 uploads and downloads with small
cache fixtures; it does not execute a policy or substitute for GPU evaluation.
Set `NPA_BEHAVIOR_RUNTIME_CACHE_LIVE_CONFIG` to a private JSON file with
`project` (a configured project alias) and `prefix` (an owned `s3://` test
prefix). The test creates a unique child prefix, checks unchanged reuse, rejects
tampering, restores a missing base, and removes its own uploaded objects:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_BEHAVIOR_RUNTIME_CACHE_LIVE_CONFIG=/private/runtime-cache-test.json \
  npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_behavior_runtime_cache_live.py -q
```

Use the read-only internal `campaign-status` command with the same `--panel-uri`
and `--output-path` to inspect durable case counts while a workflow runs. It
distinguishes unclaimed, claimed, started, and complete records. A started record
does not prove a worker is still alive; use workflow status with the run's exact
isolated controller for that check. Storage errors fail the query instead of
appearing as zero progress. Even when all records are complete, run the aggregate
stage to verify original artifacts before comparing scores.

A parallel workflow can fail while sibling tasks continue. Workflow status JSON
includes `scheduler_task_activity.active_stage_keys` for those tasks, independently
of the workflow outcome. Missing or ambiguously attributed task observations
appear in `unresolved_stage_keys`; they cannot establish that every task has
stopped. Duplicate names or task IDs, unexpected rows, and missing observations
prevent a terminal claim. A stage can be both active and unresolved when its
duplicate observations disagree. Use these fields only with the response's live-verification result.
`all_stage_tasks_terminal` describes scheduler tasks, not resource cleanup.
Task activity remains separate from outcome conflicts: explicit terminal task
observations can establish that tasks stopped while contradictory durable
outcomes keep the workflow status `UNKNOWN`.

The read-only live regression uses an existing terminal workflow. On its Linux
operator, set `NPA_WORKFLOW_TASK_ACTIVITY_LIVE_CONFIG` to a
private JSON file containing `run_id`, `project`, `isolated_config_dir`,
`expected_active_stage_keys`, `expected_all_tasks_terminal`,
`expected_task_states` (stage key to scheduler state), `expected_workflow_status`,
an optional `expected_raw_controller_state` (default `FAILED` for existing
configuration compatibility), `expected_outcome_conflicts` (stage keys), and a
private `output_path`. Run
`npa/tests/e2e/test_workflow_task_activity_live.py` with `NPA_INTEGRATION_E2E=1`.
It records the actual status response and checks the prescribed task states;
it submits, cancels, and reruns no jobs.

## Reusable panel identity

A panel ID covers the pinned evaluator upstream revision and wrapper, registry
identity, split, ordered cases, and immutable policy identity. It deliberately
does not include the campaign ID, candidate policy, worker count, or worker
assignment. A complete baseline panel can therefore be reused in a later
candidate comparison when every byte and prescribed case remains identical.

Worker partitions are separate immutable records. `partition_panel()` assigns
ordered cases round-robin, exactly once, and produces a partition digest. The
runtime claim ledger owns retries and recovery. In particular, an official case
that reached `started` without valid completion evidence must not be silently
rerun or reassigned by this pure module.

## From original artifacts to a complete panel

The existing `artifacts.inspect_rollout()` function opens the original evaluator
JSON, validates its metrics, fully decodes the MP4, and returns their hashes.
Pass that direct result to `bind_inspected_rollout()` before serialization. The
case receipt binds it to the panel and policy identity.

```python
case_receipt = bind_inspected_rollout(panel, inspected_rollout)
aggregate = aggregate_panel(panel, all_case_receipts)
```

`aggregate_panel()` checks identities, score ranges, video frame counts, and the
exact JSON/MP4 hash-key shape. It requires every prescribed case exactly once;
missing, duplicate, and extra cases fail instead of being imputed or dropped.
Optional `success` and `steps` measurements must be present for every case or for
none of them.

This is an explicit trust boundary: the pure aggregator cannot read artifact
bytes and cannot prove that a caller actually ran `inspect_rollout()`. Its output
therefore records:

```json
{
  "artifact_validation_contract": "npa.behavior.inspect-rollout.v1",
  "artifact_bytes_verified_by_aggregator": false
}
```

The runner must call `inspect_rollout()` on preserved original bytes before it
creates a case receipt. A self-declared digest is not verified evidence.

## TRAIN autonomous experience

For paired prompt diagnostics with a parity-qualified `comet-trained` policy,
`--policy-prompt-override` supplies one literal wrapper prompt. The option is
accepted only with `--train-experience` on a non-reporting TRAIN panel. Its
literal value and validation source are included in the serving identity before
startup or case claim. The admitted task slug remains the wrapper metadata
lookup key; the option changes the resulting `task_prompt` only. Omit it to use
the released instruction from that exact task mapping. DEV and REPORT reject
the override.

The internal campaign worker can record reusable Comet TRAIN experience with
`--train-experience`. The option is accepted only for a non-reporting TRAIN
panel served by `--policy-kind comet-native` or a parity-qualified
`--policy-kind comet-trained` export; development and report
workers reject it before starting the policy or evaluator. Add
`--train-experience-depth` to retain the three onboard depth arrays. Depth is
off by default because the current Comet training sample does not consume it.

The recorder keeps one lossless RGB/proprio observation at each 32-action model
decision and every action accepted by the official evaluator. It separately
stores the exact raw 32-by-23 model emission and joins every returned action to
its source chunk. The join verifies the pinned evaluator transport's explicit
float32 cast, then requires byte equality between that returned float32 action
and the action accepted by the official evaluator. Episode resets and
monotonic policy/apply timings are explicit. Post-apply observation identities
must equal the next policy input, including the intermediate steps whose large
image arrays are not duplicated. The manifest declares this cadence as
`model_decision_observation_with_all_applied_actions`.
The final post-apply allowed observation is retained once in a separate shard,
so a partial final action chunk still has a materialized terminal observation.

The successful case directory contains `train-experience/experience-manifest.json`
and bounded lossless NPZ shards. The final manifest is written only after both
the evaluator and policy terminals exist and its frame count equals the
official metrics. The bundle inventories an exact copy of those metrics, and
publication joins it back to the official evaluator output before upload.
Partial shards remain failure evidence and are never named by a success
manifest. The exact config identity produced by validated policy preparation is
committed in the durable case-start record before the evaluator runs.
Publication and recovery must match that identity, including the policy,
checkpoint, RNG, and source fields. Publication writes an immutable manifest
requirement before the primary case originals. A resumed worker must reconstruct
and verify every required experience member before it can complete the case; an
interrupted experience upload remains incomplete instead of silently dropping
the recording.

New recordings bind the wrapper's effective prompt. Policy preparation resolves
the admitted task slug through the exact `task_mapping.json`, applies any
TRAIN-only literal override, and requires the discarded-load process to report
the same `B1KPolicyWrapper.task_prompt`. The versioned experience config records
the effective text plus task-mapping, wrapper, tokenizer, transform, and
training-config source identities. The PaliGemma tokenizer model bytes remain a
training-projection input and must be bound where tokenization runs; these
source identities do not claim the downloaded tokenizer bytes were checked.

`AutonomousCometDataset` projects a finalized recording into the existing
Comet sample shape: float32 61-element state, three uint8 RGB images, a float32
32-by-23 future applied-action target, and terminal validity/padding masks.
It uses the recorded effective prompt and never substitutes the task slug.
Legacy v1 recordings require an explicit, non-mutating derivation receipt bound
to their original config, discarded-load qualification, clean source revision,
and prompt-source bytes. A legacy default also requires a preserved runtime log
containing the resolved prompt. Training projection also requires the
independently admitted canonical identity of the derivation receipt, so a
self-consistent replacement receipt cannot relabel an old recording.
Inspection remains supported without this
training-only derivation.
Depth is preserved when requested but excluded from this projection. Unknown or
privileged simulator leaves are excluded before recording and cannot enter the
policy-side stream or default Comet projection. Privileged TRAIN annotations
remain a separate optional offline-labeling boundary.

For v3.9.3 TRAIN recordings, the evaluator also preserves the official
post-step `goal_status` partition in `evaluator/goal-progress.jsonl`. Each row
is aligned to one officially applied action and records the satisfied and
unsatisfied goal-condition indices, integer counts and count delta, and the
official terminated/truncated flags. The first delta is `null` because the
evaluator does not evaluate a pre-action goal status. Before this annotation is
enabled, the terminal pins the exact upstream evaluator, behavior task, base
task, environment, predicate-goal, and condition-evaluation source bytes that
compute, nest, and return those fields.

`OfficialGoalProgressDataset` exposes these post-action labels and a derived
fractional potential for TRAIN-only progress or advantage modeling. The labels
never enter the evaluator observation, policy request, or default
`AutonomousCometDataset` projection. Older finalized recordings without this
optional annotation remain valid. Task 1 currently exposes goal-condition
satisfaction rather than an authored sequence of manipulation stages, so this
is a source-bound progress target and not a ForesightFlow reproduction or a
claim of improved task performance.

New v3 recordings also preserve the official `TaskMetric` Q value after every
applied action in `evaluator/official-q.jsonl`. This observer reads the same
reset-time predicate masks, current grounded-predicate masks, success flag, and
`compute_q_score` implementation used by v3.9.3 evaluation. Finalization
requires the last recorded Q to equal the unchanged official terminal metric.
`OfficialQProgressDataset` exposes these rows separately from the coarser
top-level `goal_status`; task 1 has one top-level goal while official Q can show
one-third increments and regressions among its three grounded predicates.

Official Q remains a TRAIN-only offline annotation. It never enters policy
inputs or the default `AutonomousCometDataset` projection. Any Q-weighted
projection must explicitly require `OfficialQProgressDataset`, so evaluator v1
and coarse-progress-only v2 recordings fail closed instead of receiving an
inferred Q label.

## TRAIN semantic labels

The [task-1 semantic monitor](../../npa/src/npa/workflows/behavior_challenge/semantic_monitor/README.md)
provides an offline labeler for `picking_up_trash`. It derives persistent grasp,
released placement, and failure labels from explicitly authorized TRAIN traces,
and splits episodes deterministically for fitting and calibration. Delayed stage
proposals use the current completion state; drops and re-grasping invalidate it.

The package also defines an allowed-observation inference interface. It does not
collect simulator traces, train a monitor, or modify policy execution. Real TRAIN
predicate collection, stage-to-object bindings, model calibration, and transport
integration are required before a learned monitor can be evaluated.

## Paired comparison and claim scope

`compare_panels()` accepts only the complete baseline and candidate aggregates
referenced by the campaign. Cases remain in declaration order, producing paired
Q deltas plus success-rate deltas when both policies provide success and step
measurements. The result also reports whether the reviewed paired-case threshold
was met.

A complete one-task or two-task comparison supports a narrow statement about
those prescribed task panels. It does not establish challenge-wide
competitiveness. Every declaration, aggregate, and comparison sets
`full_challenge_competitive_claim_allowed` to `false`. Even a complete public
100-task panel remains local reproducibility evidence rather than organizer
certification or a hidden-test leaderboard result.

The comparison contains no challenge score for focused panels. Report mean Q,
success rate, paired-case count, selected tasks, split, and immutable panel IDs
together so readers can see the exact scope.
