# Sim2Real durability: standard workflow runtime

The former Sim2Real-specific controller is retired from the canonical path.
Durability now belongs to the standard `npa.workflow` runtime, not to a driver
pod that creates and watches child Jobs.

The canonical workflow persists its runtime ledger under the run's S3 prefix.
Each state also publishes immutable outputs and a content-addressed
ComponentRecord. A restarted invocation with the same run ID and exact source
and image identities uses `--resume` to:

1. verify the workflow digest and durable ledger;
2. adopt any managed SkyPilot job still running;
3. verify declared S3 outputs before reusing a completed wave;
4. restart only incomplete work; and
5. continue across nested-loop and finalization barriers.

The Stage 8 Reason wave therefore commits both lane outputs before Stage 9 can
consume them. Finalization similarly resumes from the last completed state and
retains the exact gold report/render lineage. Mutable image tags and mismatched
source SHAs fail closed at the stage boundary.

The archived direct-controller implementation remains importable for evidence
replay, but it is not selected by workflow detection or submission and is not
an operator surface. The full contract and module ownership are documented in
[`docs/architecture/sim2real-compositional-workflow.md`](../../architecture/sim2real-compositional-workflow.md).
