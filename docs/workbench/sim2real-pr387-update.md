# PR #387: unique scope after the current-main audit

The rebased PR retains the Sim2Real submit check that rejects rollout,
validation, or gold requests exceeding their deterministic sealed splits before
GPU work. Focused regression tests cover all three consumers and the supported
boundary. The [operator guide](guides/sim2real-workflow.md) explains split
sizing: 640 environments at a training fraction of 0.8 provide 512 training,
64 validation, and 64 gold rows, supporting 64 requested rows at each boundary.

The remaining access delta adds the Guardrail1 and Predict2.5-2B dependencies
to the direct `cosmos2` capability. PAIDF submission keeps its deliberately
narrow exact-checkpoint probe rather than inheriting the two additional
`cosmos2` dependencies; a submit/access regression test protects that scope.
Additive path-free classification tests cover the existing vendor diagnostics.

The following work is already merged to `main` and is excluded from this PR's
aggregate diff:

| Merged PR | Existing behavior preserved |
| --- | --- |
| #378 | Sim2Real Guardrail1 and Predict2.5-2B access checks. |
| #379 | Cosmos Transfer vendor-error diagnostics. |
| #380 | Existing-cluster adoption, registry credentials, `source_sha`, and resume/retry guidance. |
| #373 | Refreshed coherent images and runtime fixes, superseding the earlier image release associated with #387. |

The guide preserves current main's `SOURCE_SHA` export and explanation. The
shared workflow CLI also preserves #370's terminal plan-migration flags,
arguments, and behavior while applying the PAIDF exclusion. Historical audit
reports remain historical records; this update describes the reduced PR scope.

## Reused workload evidence and limits

The earlier PR #387 run completed all 21 rendered jobs: Cosmos Transfer,
eight EnvGen shards producing 640 environments, 64 Isaac rollout episodes,
64 hosted Cosmos evaluations, 200 PPO updates, 64 validation episodes, and
64 sealed gold episodes. Durable read-back found JSON, JSONL, MP4, checkpoint,
MCAP, and RRD artifacts. Strict 5 cm gold success was 0/64, and the workflow
correctly took its loop-back branch with early exit disabled. Completion
demonstrated orchestration, not policy efficacy.

This evidence predates #373's refreshed images and point-cloud fix. The rebase
reuses it without claiming a workload rerun of the current head or current
images. The prior record also notes target-specific cache-volume and Isaac
CUDA failures, and stale global storage configuration despite successful
project-scoped storage probes. No new GPU workload or image build is part of
this deduplication pass. Fresh repository validation is recorded in the PR's
checks and validation summary.
