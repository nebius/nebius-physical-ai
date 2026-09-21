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
