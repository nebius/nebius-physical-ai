# Four-node Soperator deployment and GPU qualification

NPA deployed **four reserved workers with eight B200 GPUs each** in
`us-central1`. [Direct device inventory](hardware.json) verified 32 distinct
physical GPUs privately; the published receipt omits device identifiers.
The [deployment receipt](deployment.json) records the successful NPA reconcile
and mandatory CUDA checks on all four workers: `deviceQuery`, `vectorAdd`,
`simpleMultiGPU`, and `p2pBandwidthLatencyTest`.

This closes the previously unexecuted Soperator deployment path. It establishes
cluster readiness; it does not establish WAM training duration or policy quality.
The subsequent [live training receipts](../cosmos3-wam-live-training-32/README.md)
verify 32 attributed trainer processes, optimizer updates, InfiniBand and a fixed
two-minute observation window. The later [full schedule](../cosmos3-wam-full-32/README.md),
[repeated timing](../cosmos3-wam-timing-32/README.md) and
[checkpoint evaluations](../cosmos3-wam-quality-32/README.md) completed successfully.
The original [pre-execution protocol](../../../../npa/workflows/workbench/cosmos3-wam-slurm/benchmark-protocol-32.json)
and [capacity rejection](../cosmos3-wam-plan-32/preflight.json) retain their
historical bytes.

## Deployment failures and recovery

The first attempt encountered a stale local Helm repository cache. A private,
task-specific `HELM_REPOSITORY_CONFIG` and `HELM_REPOSITORY_CACHE` resolved it.
Provider resources were reused; NPA's replacement guard rejected destructive
provider changes.

The pinned Flux chart then exposed a real NPA defect: it does not forward
arbitrary `nodeConfigurator.values` fields. The old init-container patch never
reached the operator-managed NodeConfigurator, so Ubuntu's AppArmor host gate
blocked Enroot user namespaces. NPA now supplies the chart's supported,
watched `terraform-nodeconfigurator` ConfigMap through `valuesFrom`. The patch
preserves the complete default sysctl initializer and changes the AppArmor
gate only for the intentionally unconfined worker profile.

The failed container pre-pull check was rerun successfully. NPA reconciled the
fix through Terraform and Helm, with zero provider replacements, then repeated
the four-worker CUDA gate successfully. The committed
[live test](../../../../npa/tests/e2e/test_soperator_userns_live.py) independently
passed non-root user-namespace creation on all four workers after reconciliation.
Its documented command is in the [live-test README](../../../../npa/tests/e2e/README.md).
Original failures and complete deployment logs remain in private evidence;
the public receipt includes their hashes.

The receipt pins the exact executed NPA source revision. A subsequent registry
policy correction retains NPA's existing public BusyBox bootstrap image; that
image change is outside this historical deployment receipt's validation scope.
The measurement cluster stayed unchanged throughout the GPU campaign.
After all ten workload jobs and all CPU reductions completed, the current public
bootstrap was [reconciled and separately qualified](post-campaign-bootstrap.json)
through NPA, with zero provider replacements. All ten node-configurator init
containers completed with `docker.io/library/busybox:stable`, including the four
GPU hosts; exact image IDs are retained. NPA repeated its all-worker CUDA gate,
and the live user-namespace test passed on all four workers. This closes the
current public-image qualification gap without attributing that image to the
earlier benchmark deployment. It was an in-place reconciliation, not another
fresh initial deployment.

The first WAM launch passed the 32-rank collective, then blocked before any
optimizer update: Soperator's NCCL log FIFO reader had exited after the
preflight. The subsequent trainer waited for a reader when opening the same
FIFO. That attempt and its dependent jobs were cancelled; its logs are
preserved and excluded from timing and quality. The recipe now redirects
FIFO-based NCCL output to separate per-process files in the run directory.
It preserves regular-file logging and all transport settings. The complete
schedule restarted in a fresh run directory after this correction and finished
all 2,000 updates successfully.

## Comparison boundary

This cluster uses Soperator 4.1.6, Slurm 25.11.3, driver 580.159.04 and a
shared jail. The earlier 8/16-GPU measurements used native Slurm 23.11.4,
driver 580.173.02 and NFS. Preserve these differences when discussing the
32-GPU extension; GPU count is not the only changed condition.

## Completed cleanup

After every campaign job completed and the full Slurm queue was empty, the
checkpoint and runtime archive passed full-object read-back hash verification.
The dedicated cluster was then destroyed through `npa soperator destroy`.
The independent provider inventory found zero compute instances, managed
clusters, filesystems and IP allocations in the dedicated project. See
[teardown.json](teardown.json) for counts, archive totals and private-source hashes.

The private archive, project and two original native-VM evidence disks remain.
Both historical native VMs were already absent at the pre-cleanup audit.
Retained storage and reservation commitments are separate from compute-instance
cleanup; this is not a billing or reservation-cancellation receipt.
