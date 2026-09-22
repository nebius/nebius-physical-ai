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

For managed workers, construct the serving artifact with
`serving_identity.serving_artifact(args)`. It hashes the actual adapter files,
policy kind, execution variant, per-episode process lifecycle, and optional
correlation/export receipts.
The worker recomputes that identity and checks the checkpoint archive bytes
before starting the policy. This prevents a changed controller from inheriting
an earlier policy's completed cases.

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
the declaration URIs and the existing specialist runtime paths; the test does not
read sealed baseline results or create positive report authorization.

Create the private file outside the repository, make it owner-readable only,
and use the existing authorized runtime paths:

```json
{
  "panel_uri": "s3://<bucket>/<prefix>/reporting-panel.json",
  "partition_uri": "s3://<bucket>/<prefix>/reporting-partition.json",
  "absence_prefix": "s3://<bucket>/<fresh-negative-test-prefix>",
  "upstream_root": "/path/to/BEHAVIOR-1K",
  "evaluator_python": "/path/to/evaluator-python",
  "data_root": "/path/to/dataset",
  "policy_root": "/path/to/rlc-source",
  "policy_python": "/path/to/policy-python",
  "policy_checkpoint": "/path/to/checkpoint_2",
  "policy_archive": "/path/to/specialist.zip"
}
```

Run only the negative admission regression:

```bash
chmod 600 /path/to/specialist-admission-live.json
NPA_BEHAVIOR_SPECIALIST_ADMISSION_LIVE_CONFIG=/path/to/specialist-admission-live.json \
  NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_behavior_specialist_report_admission_live.py -q
```

## Durable execution and recovery

The internal `campaign-worker` stage takes `--panel-uri`, `--partition-uri`,
`--worker-index`, and the existing managed-policy/runtime arguments. Required
`--output-path` is a stable S3 campaign prefix that must remain the same across
resumes. `--workspace` is a persistent worker directory. Required
`--worker-receipt-uri` is a unique output location for this workflow invocation.
Run and monitor it with `npa workbench workflow submit`, `status`, `logs`, and
`artifacts`, as with other Workbench workflows.

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

The read-only live regression uses an existing failed workflow. On its Linux
operator, set `NPA_WORKFLOW_TASK_ACTIVITY_LIVE_CONFIG` to a
private JSON file containing `run_id`, `project`, `isolated_config_dir`,
`expected_active_stage_keys`, `expected_all_tasks_terminal`,
`expected_task_states` (stage key to scheduler state), `expected_workflow_status`,
`expected_outcome_conflicts` (stage keys), and a private `output_path`. Run
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
