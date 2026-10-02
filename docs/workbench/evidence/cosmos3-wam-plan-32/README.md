# Four nodes, 32 B200 GPUs: deployment preparation

The requested extension is **four workers with eight B200s each**, using the
existing `npa soperator` CLI/SDK/backend for cluster lifecycle. This page retains
the original planning and capacity-rejection history. A later
[deployment and all-worker CUDA qualification](../cosmos3-wam-soperator-32/README.md)
succeeded on all four workers after capacity became available. The subsequent
[full training run](../cosmos3-wam-full-32/README.md),
[three timing repetitions](../cosmos3-wam-timing-32/README.md) and
[four checkpoint evaluations](../cosmos3-wam-quality-32/README.md) completed
successfully. The original planning receipts below remain unchanged.
The [preflight receipt](preflight.json) records a successful provider-free plan
and verification of the immutable upstream deployment source. These checks do
not verify reservation availability, deploy Kubernetes/Slurm, or exercise GPUs.
The actual `npa soperator deploy` command then rejected the request before
provider mutation: 32 GPUs required, 21 reserved GPUs available, 11 short. Its
STRICT reservation gate disallowed on-demand/preemptible fallback. No resources
were provisioned by the attempt.

The [pre-execution protocol](../../../../npa/workflows/workbench/cosmos3-wam-slurm/benchmark-protocol-32.json)
fixes 2,000 updates, batch 2,048, sample cap 64, accumulation one, seed 42,
three timing repeats, separate profiles on ranks 0/8/16/24, and all four
500-trial checkpoint evaluations. GPU proof must cover every worker and rank.
Previous 8/16-GPU measurements used native Slurm. Runtime versions and any
storage/network differences must be recorded before pooling or comparing runs.

## Reproduce the preparation

Provide project, tenant and STRICT reservation selectors through private
`NPA_PROJECT_ID`, `NPA_TENANT_ID`, and `NPA_CAPACITY_BLOCK_GROUP` environment
variables. Choose an unused cluster name and a private output path. The helper
is intentionally scoped to this campaign's `us-central1` region and
`us-central1-b` B200 fabric; it is not a general regional discovery tool.

```bash
npa/.venv/bin/python npa/workflows/workbench/cosmos3-wam-slurm/cluster.py \
  --name "$WAM_CLUSTER_NAME" --nodes 4 --output "$WAM_CLUSTER_SPEC"
npa/.venv/bin/npa soperator plan --spec "$WAM_CLUSTER_SPEC" --output json
npa/.venv/bin/npa soperator deploy --spec "$WAM_CLUSTER_SPEC" \
  --root-login-ssh-public-key-file "$WAM_LOGIN_PUBLIC_KEY" \
  --source-preflight-only --output json
```

After reserved capacity and credential/access preflights pass, deploy through
the same command without `--source-preflight-only`. The NPA lifecycle verifies
the reservation before provider mutation and runs mandatory CUDA checks on all
workers. Then prepare the pinned runtime/inputs in the shared jail, validate
32-rank NCCL placement, and submit the recipe with `--nodes 4`. Plan/source
preflight records above are historical; the later deployment receipt establishes
actual cluster and GPU qualification.

Cancel owned Slurm jobs and verify an empty queue before preserving artifacts
and running `npa soperator destroy --name "$WAM_CLUSTER_NAME"`. The previous
native VM lifecycle is historical evidence, not the deployment path for this
extension. See the [Soperator skill](../../../../skills/tools/soperator/SKILL.md).
