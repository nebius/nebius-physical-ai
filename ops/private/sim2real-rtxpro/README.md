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

This writes **`RUNBOOK.local.md`** and **`env.local`** with your bucket, registry,
and cluster context filled from `~/.npa/config.yaml`. Credentials are referenced
by path only.

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
# After run completes — sync from S3 or read pod /tmp
rerun reports/sim2real.rrd
cat reports/sim2real-report.json | jq '.outer_loop.latest_decision'
```

See **`RUNBOOK.local.md`** (generated) for asset URIs, trigger paths, and accuracy baselines.

## Direct Kubernetes submit (RTX PRO)

SkyPilot on `npa-rtxpro-mk8s` is blocked by kubeconfig context mismatch. Use:

```bash
export KUBECONFIG=~/.npa/clusters/npa-rtxpro-mk8s/kubeconfig
INNER_ITERATIONS=2 OUTER_ITERATIONS=2 \
  ./ops/private/sim2real-rtxpro/submit-k8s-staged-job.sh
./ops/private/sim2real-rtxpro/monitor-k8s-job.sh sim2real-<run-id>
```

Logs: `/tmp/sim2real-cluster/`. The job clones `NPA_SOURCE_REF` (default: branch under test).
