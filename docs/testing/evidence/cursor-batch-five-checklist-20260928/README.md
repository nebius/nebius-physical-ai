# Resume stack: supplied checklist closed

The supplied chart described base `c660428`. These seven PRs now include current
base `eb5d1dbdfce2692b8a6a09ad5abce3408bb6cd95` and apply cleanly in the required order:
**#635 → #637 → #638 → #639 → #641 → #643 → #644**.

Main subsequently advanced to `3ca7a94c79f1fdc5a0d44c2ad50f1af6dd9b6245` with #762/#763. The same
seven heads also apply cleanly onto that newer base; its two affected suites
returned **353 passed, 2 warnings in 13.95s** on the combined tree. The complete-suite
and incremental compatibility evidence retain their own exact revisions.

| PR | Current head | Merge checks | Review |
| --- | --- | --- | --- |
| [#635](https://github.com/nebius/nebius-physical-ai/pull/635) | `36de4ac3e6` | Clean; required checks passed | [Recorded disposition](https://github.com/nebius/nebius-physical-ai/pull/635#pullrequestreview-5346224368) |
| [#637](https://github.com/nebius/nebius-physical-ai/pull/637) | `3063171b4e` | Clean; required checks passed | [Recorded disposition](https://github.com/nebius/nebius-physical-ai/pull/637#pullrequestreview-5346224701) |
| [#638](https://github.com/nebius/nebius-physical-ai/pull/638) | `0d41a1c675` | Clean; required checks passed | [Recorded disposition](https://github.com/nebius/nebius-physical-ai/pull/638#pullrequestreview-5346224994) |
| [#639](https://github.com/nebius/nebius-physical-ai/pull/639) | `619d4f1a08` | Clean; required checks passed | [Recorded disposition](https://github.com/nebius/nebius-physical-ai/pull/639#pullrequestreview-5346225379) |
| [#641](https://github.com/nebius/nebius-physical-ai/pull/641) | `9116e957f2` | Clean; required checks passed | [Recorded disposition](https://github.com/nebius/nebius-physical-ai/pull/641#pullrequestreview-5346225671) |
| [#643](https://github.com/nebius/nebius-physical-ai/pull/643) | `0dfdd2068e` | Clean; required checks passed | [Recorded disposition](https://github.com/nebius/nebius-physical-ai/pull/643#pullrequestreview-5346225983) |
| [#644](https://github.com/nebius/nebius-physical-ai/pull/644) | `b3ad9671a9` | Clean; required checks passed | [Recorded disposition](https://github.com/nebius/nebius-physical-ai/pull/644#pullrequestreview-5346226322) |

## Every requested gate

- **Code:** preserve #635's deliberate verified-cancellation AND terminal-provider condition, exact-int prelaunch proof, and unknown-decision reconciliation. #637 owns phase definitions at the producer and persists sanitized exception types. #639 rejects missing or changed replay identities. #641 binds complete image selection and per-state resolved images. #644 validates every branch before preparation and once per batch.
- **Tests:** core #641 coverage ships with its implementation; #643 retains supplemental precedence/resource tests. Current-head CI is complete for each PR. The combined Linux suite returned **33880 passed, 156 skipped, 1 xpassed, 458 warnings in 916.89s (0:15:16)**. Lint/precheck, documentation drift, and confidentiality also passed; [validation logs](validation.json) bind the exact integration revision.
- **Rebase/conflicts:** current main #760/#761 is included. The #551 helper resolution and both selector/live-gate checks remain intact. A seven-step squash simulation is recorded in [the report](report.json). The shared image explanation now lands in #641, avoiding a later documentation conflict.
- **Order/labeling:** #638's description identifies its cumulative dependency and incremental tests-only scope. #643 is supplemental after #641; the earlier missing-core-coverage window is closed. Merge in the displayed order through the existing queue.
- **#635 decision:** the helper behavior is an explicit operator-directed implementation choice recorded in its review, not an accidental conflict resolution. Terminal status without verified cancellation stays blocked. No separate human approval is invented.
- **Rollout:** use the [documented controller rollout](https://github.com/nebius/nebius-physical-ai/blob/961eabfc7660c13ae29bd90e7b52e96e5e68e0e6/docs/run-lifecycle.md#controller-rollout-with-existing-runs). Keep unfinished runs on their original controller, environment, source, images, and recorded inputs; upgrade separately for new work. Retire old controllers only after their runs are terminal. Do not rewrite ledgers or replace unfinished work merely by changing its run ID. This task did not update existing controllers and does not claim a global absence of active jobs.
- **Compatibility note:** the workflow guide's old promise of legacy digest-pin fallback has been removed. Guarded replay/reuse/recovery rejects insufficient old identity evidence; ordinary in-flight provider adoption remains a separate legacy path. #644 documents that unmatched selectors now fail and shared override sets must be scoped to each workflow.
- **Review provenance:** every PR has a formal GitHub COMMENT review linked above, with the independent Claude Code AI review and Codex follow-up attributed accurately. Author-posted mutation evidence remains preserved. The repository requires checks, signatures and its merge queue, not a second human approval; no rule was weakened.

## Real evidence and scope

[Prior independent AI review, full validation and real B200 proof](https://github.com/nebius/nebius-physical-ai/blob/2fca824c6a7f46871a7665f0f6c8d2ae55bb6cc8/docs/testing/evidence/cursor-batch-five-followup-20260928/README.md) remain
available at immutable revisions. The B200 run exercised real optimizer steps,
a CPU numerical oracle, and both injected crash boundaries; it is attributed to
its original source. This follow-up does not claim fresh GPU or VLM execution.
The workflow production code is byte-identical to the prior validated stack;
current-main changes and the documentation correction have fresh validation.

Public artifacts contain sanitized results and hashes. No PR was merged or
enqueued, no repository policy changed, and no customer credentials were
published. [Machine-readable report](report.json).
