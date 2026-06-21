---
name: sim2real-operate
description: Use when running, monitoring, or debugging the staged Sim2Real pipeline on a Kubernetes GPU cluster — the runbook, the direct-K8s submit path, preflight health checks, cluster storage secrets, and job monitoring.
---

# Sim2Real Operate

## When To Use

Use this skill to actually *operate* the staged Sim2Real VLM-to-RL pipeline on a
Kubernetes RTX PRO 6000 / L40S cluster: submitting a run, preflighting it,
watching the orchestrator and its sibling Jobs, and recovering from the recurring
cold-start blockers. For navigating or changing the engine *code* (the 14-stage
map), use `sim2real-engine` instead; for generic sim-to-real workflow design use
`sim-to-real`.

## Entry Points

- `npa/workflows/workbench/sim2real/runbook.yaml` — the standalone, materialized
  raw-SkyPilot runbook. Read its header first: it documents every env var, the
  trigger-bucket vs artifact-bucket split, and the S3-compatible endpoint map.
- `<private-operator-pack>/sim2real-rtxpro/submit-k8s-staged-job.sh` — the direct-Kubernetes
  submit path (bypasses the SkyPilot 0.12.2 getcwd/kubeconfig blocker). This is
  the route that actually reaches GPUs today. It applies a one-GPU orchestrator
  Job that clones `NPA_SOURCE_REF` and runs `python -m npa.workflows.sim2real run`,
  which fans out sibling Jobs (Isaac sim, VLM, eval, trainer, envgen).
- `npa workbench workflow submit npa/workflows/workbench/sim2real/runbook.yaml`
  and `sim2real.run` (SDK) wrap the same workflow; they do not gate it.

## Procedure

1. **Configure once.** `~/.npa/config.yaml` (bucket, endpoint, registry,
   `k8s_context`) + `~/.npa/credentials.yaml` (S3 HMAC, HF/NGC tokens). Generate
   operator files with `<private-operator-pack>/sim2real-rtxpro/setup-local-operator.sh`.
2. **Seed the trigger** on a new bucket: `seed-stock-trigger.sh`, then set
   `storage.sim2real_stock_trigger_uri`.
3. **Sync the cluster storage secret** so pods get the endpoint + keys:
   `<private-operator-pack>/sim2real-rtxpro/sync-cluster-storage-secret.sh`.
4. **Preflight:** `npa workbench health sim2real --checks all` (accepts `all` or
   a comma list: `config,coherence,s3,registry,tokens,cluster`). Expect PASS on
   s3, tokens, cluster; WARN on registry only when `NPA_REGISTRY` is unset.
5. **Submit:** `INNER_ITERATIONS=… OUTER_ITERATIONS=… submit-k8s-staged-job.sh`
   (or `run.sh trigger`). It registry-qualifies every image, refreshes the
   `npa-nebius-registry` pull secret, and preflights the trigger + S3 write.
6. **Monitor:** `<private-operator-pack>/sim2real-rtxpro/monitor-k8s-job.sh sim2real-<run-id>`
   or `npa workbench sim2real status <run-id> --watch`.
7. **View results:** `run.sh sync <run-id>` (Rerun), or read
   `reports/sim2real-report.json` (`.outer_loop.latest_decision`,
   `.inner_loop.reward_trend`, `.upload.status`).

## Gotchas

- **Exit codes are load-bearing.** `python -m npa.workflows.sim2real run` exits
  non-zero when an artifact upload was requested but `upload.status` is
  `blocked`/`failed`. Shell wrappers must check `$?` (do not print success
  unconditionally). `rerun_serve` blocked is a warning, not a failure.
- **Trigger bucket vs artifact bucket can differ** on S3-compatible object
  stores. `NPA_SIM2REAL_TRIGGER_DATASET_URI` is required at submit.
- **Stale IAM token → ImagePullBackOff 401.** The submit script refreshes
  `npa-nebius-registry` before apply; if a sibling Job still fails to pull,
  re-run the refresh. The refresh is per-registry-server, so it also covers the
  envgen image even though that image is set from `NPA_REGISTRY` at runtime.
- **GPU product is pinned** via `nodeSelector` /
  `NPA_SIM2REAL_K8S_GPU_PRODUCT` (default
  `NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition`). Wrong product → Pods stay
  Pending.
- **Isaac Lab needs RT-core GPUs** (L40S / RTX PRO). Genesis is the fallback
  backend (`NPA_SIM2REAL_SIM_BACKEND`).
- Keep `runbook.yaml`'s `envs:` literals and the `run:` block `${VAR:-default}`
  fallbacks in agreement — a cleared env var must not silently change behavior.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
npa workbench health sim2real --checks all
bash -n <private-operator-pack>/sim2real-rtxpro/submit-k8s-staged-job.sh
```
