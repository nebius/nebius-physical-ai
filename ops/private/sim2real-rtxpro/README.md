# RTX PRO Sim2Real — Private Operator Pack

**For Nebius RTX PRO 6000 cluster operators only.** This directory is safe to commit
(templates + scripts). **Secrets never go here.**

## Setup (once per machine)

```bash
# 1. Machine config (shareable — no secrets)
cp /path/to/rtxpro-cluster-config.yaml ~/.npa/config.yaml

# 2. Secrets (never commit)
npa configure   # writes ~/.npa/credentials.yaml

# 3. Kubeconfig
export KUBECONFIG=~/.npa/clusters/npa-rtxpro-mk8s/kubeconfig
kubectl --context npa-rtxpro-mk8s get nodes

# 4. Generate local operator files (gitignored)
./ops/private/sim2real-rtxpro/setup-local-operator.sh
```

This writes **`RUNBOOK.local.md`**, **`DEMO-WALKTHROUGH.local.md`**, and **`env.local`**
with your bucket, registry, and cluster context filled from `~/.npa/config.yaml`.
Credentials are referenced by path only.

## Run locally (no cluster)

Rehearse the full 13-stage artifact tree on **this machine only** (~1 s). No GPU, no S3, no kubeconfig.

```bash
./ops/private/sim2real-rtxpro/run-local-demo.sh
# → runs pipeline, verifies .rrd, starts local Rerun web viewer URL
```

See **`LOCAL-DEMO.md`** for the local walkthrough. Cluster/S3 paths are optional (appendix in private repo).

## Run staged workflow

```bash
source ops/private/sim2real-rtxpro/env.local
npa workbench health sim2real --checks all
npa workbench workflow submit \
  npa/workflows/workbench/sim2real/runbook.yaml \
  --env-file ops/private/sim2real-rtxpro/env.local
```

## View results

```bash
# Offline walkthrough (sync golden run from S3)
./ops/private/sim2real-rtxpro/prestage-offline-run.sh <run-id>
rerun /tmp/sim2real-prestage/<run-id>/reports/sim2real.rrd

# After live run — sync from S3 or read pod /tmp
rerun reports/sim2real.rrd
cat reports/sim2real-report.json | jq '.components[] | select(.name=="stage_14_rerun_viz")'
cat reports/sim2real-report.json | jq '.outer_loop.latest_decision'
```

Rerun `.rrd` is at `s3://<bucket>/sim2real-b/<run-id>/reports/sim2real.rrd` when
`stage_14_rerun_viz` tier is **WORKS** in the report JSON. Tier **WARN** (no
rerun-sdk) or **SEAM** (`NPA_SIM2REAL_RERUN=0`) means the object is absent — not
an upload bug.

See **`RUNBOOK.local.md`** (generated) for asset URIs, trigger paths, and accuracy baselines.

K8s deployment inventory (placeholders): [sim2real-architecture.md](../../../docs/workbench/guides/sim2real-architecture.md#kubernetes-deployment-inventory).

## Direct Kubernetes submit (RTX PRO)

SkyPilot on `npa-rtxpro-mk8s` is blocked by kubeconfig context mismatch. Use:

```bash
export KUBECONFIG=~/.npa/clusters/npa-rtxpro-mk8s/kubeconfig
INNER_ITERATIONS=2 OUTER_ITERATIONS=2 \
  ./ops/private/sim2real-rtxpro/submit-k8s-staged-job.sh
# Submit preflights registry-qualified images (orchestrator + sibling stages).
# Monitor auto-starts in tmux session sim2real-cluster-live
tmux attach -t sim2real-cluster-live
# Or read logs:
tail -f /tmp/sim2real-cluster/sim2real-<run-id>-monitor.log

# Clean up finished s2r-* sibling jobs (dry-run first)
./ops/private/sim2real-rtxpro/delete-stale-s2r-jobs.sh --dry-run
./ops/private/sim2real-rtxpro/delete-stale-s2r-jobs.sh --keep-run-id <active-run-id>
```

Held-out rollouts use **Isaac Lab** (`NPA_SIM2REAL_SIM_BACKEND=isaac`, `ISAAC_IMAGE`).
Override with `NPA_SIM2REAL_SIM_BACKEND=genesis` only for legacy debugging.

Logs: `/tmp/sim2real-cluster/`. The job clones `NPA_SOURCE_REF` (default: branch under test).
