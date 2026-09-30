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
two-minute observation window.
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

The first WAM launch passed the 32-rank collective, then blocked before any
optimizer update: Soperator's NCCL log FIFO reader had exited after the
preflight. The subsequent trainer waited for a reader when opening the same
FIFO. That attempt and its dependent jobs were cancelled; its logs are
preserved and excluded from timing and quality. The recipe now redirects
FIFO-based NCCL output to separate per-process files in the run directory.
It preserves regular-file logging and all transport settings. The complete
schedule restarts in a fresh run directory after this correction.

## Comparison boundary

This cluster uses Soperator 4.1.6, Slurm 25.11.3, driver 580.159.04 and a
shared jail. The earlier 8/16-GPU measurements used native Slurm 23.11.4,
driver 580.173.02 and NFS. Preserve these differences when discussing the
32-GPU extension; GPU count is not the only changed condition.
