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

- `npa/workflows/sim2real.yaml` — the standalone canonical direct-Kubernetes
  workflow. Read its header first: it documents every env var, the
  trigger-bucket vs artifact-bucket split, and the S3-compatible endpoint map.
- `npa workbench workflow submit npa/workflows/sim2real.yaml`
  detects this exact file and calls the in-repo direct-K8s materializer. It
  applies the immutable, CPU-only controller image, which fans out sibling GPU
  Jobs (Isaac, Cosmos Transfer/Reason, PPO, eval, envgen) through Kueue.

## Procedure

1. **Configure once.** `~/.npa/config.yaml` (bucket, endpoint, registry,
   `k8s_context`) + `~/.npa/credentials.yaml` (S3 HMAC, HF/NGC tokens). An
   isolated launcher instead sets `NPA_SIM2REAL_OPERATOR_CONFIG` to an absolute
   run-local `config.yaml` and `KUBECONFIG` to its isolated kubeconfig. Relative
   config paths fail closed; never redirect `HOME` to select a tenant.
2. **Seed the trigger** with a task-aligned Isaac lift-cube/Franka trajectory
   prefix containing `task-dataset-manifest.json` and its referenced actions and
   camera observations, then set `storage.sim2real_stock_trigger_uri`. PushT is
   incompatible and the real-required path fails closed on it.
3. **Sync the cluster storage secret** so pods get the endpoint + keys:
   ensure `npa-storage-credentials` and `hf-ngc-tokens` exist in the namespace.
4. **Preflight:** `npa workbench health sim2real --checks all` (accepts `all` or
   a comma list: `config,coherence,s3,registry,tokens,cluster`). Expect PASS on
   s3, tokens, cluster; WARN on registry only when `NPA_REGISTRY` is unset.
5. **Submit:** `npa workbench workflow submit npa/workflows/sim2real.yaml
   --run-id <id> --var ...`; pass explicit Isaac EULA acceptance and use the
   real-tier example in `docs/workbench/guides/sim2real-workflow.md`.
   `NPA_SIM2REAL_K8S_JOB_TIMEOUT_S=0` is the intentional uncapped default. Use
   a positive override such as `--var NPA_SIM2REAL_K8S_JOB_TIMEOUT_S=14400`
   only when the operator wants a deadline.
6. **Monitor:** `npa workbench workflow status <run-id> --watch` plus
   `kubectl get jobs -l sim2real.local/run-id=<run-id>` for sibling evidence.
7. **View results:** download the Rerun/MCAP objects, or read
   `reports/sim2real-report.json` (`.outer_loop.latest_decision`,
   `.inner_loop.reward_trend`, `.policy_access`, `.upload.status`). The canonical
   `reports/sim2real.rrd` includes the 14-stage timeline, every persisted
   outer/inner pass, reward/loss/success metrics, rollout cameras/actions, and
   checkpoint access instructions. `reports/sim2real-progress.rrd` is refreshed
   while enough stage records exist.

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
- **GPU placement is ordered and capacity-aware.** Set the preferred product
  with `NPA_SIM2REAL_K8S_GPU_PRODUCT` and optional ordered products with
  `NPA_SIM2REAL_K8S_GPU_CANDIDATES`; actual node labels are normalized and
  compatible discovered products are appended. A Job changes product only for
  concrete GPU capacity/selector evidence. Image pulls, credentials, model
  weights, container exits, and application failures fail on the selected
  product without retry.
- **Taints, cordons, and NotReady nodes are not product-capacity evidence.** A
  scheduling failure containing only those signals stays on the selected GPU
  product and fails closed. Product fallback is allowed only when the same
  scheduler evidence also names insufficient `nvidia.com/gpu` or a concrete
  node-selector/affinity mismatch. This boundary prevents a fallback from
  hiding a broken node pool or bypassing placement policy. Inspect `kubectl
  describe node`, Job/Pod events, readiness, `spec.unschedulable`,
  taints/tolerations, GPU Operator health, and the exact product label; remediate
  and resubmit instead of widening retry behavior.
- **Architecture markers are evidence, not marketing labels.** If an image tag
  advertises architectures, RTX PRO 6000 requires `sm120` SASS or `compute120`
  PTX. L40S accepts proven same-major `sm80`/`sm89` SASS or
  `compute80`/`compute89` PTX, never `sm90` SASS. Isaac always keeps the
  independent RT-core restriction and rejects H100/H200/B200/B300.
- **Isaac Lab needs RT-core GPUs** (L40S / RTX PRO). H100/H200 are always
  filtered for Isaac. Selecting Genesis is an explicit backend choice
  (`NPA_SIM2REAL_SIM_BACKEND`), never an automatic failure fallback in the real
  tier.
- Every retry preserves registry-qualified real-tier images and Kubernetes
  execution. Candidate exhaustion is reported with exact scheduler evidence;
  it never falls back to SEAM/reference/in-process behavior.
- An uncapped timeout does not mask terminal errors: failed Job counters, Job
  deletion, kubectl errors, image/runtime failures, and non-zero component exits
  still fail the run. `0` only omits a time-based deadline.
- Keep `npa/workflows/sim2real.yaml`'s `envs:` literals and the `run:` block `${VAR:-default}`
  fallbacks in agreement — a cleared env var must not silently change behavior.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
npa/.venv/bin/npa workbench health sim2real --checks all
npa/.venv/bin/npa workbench workflow submit npa/workflows/sim2real.yaml --run-id <id> --plan-only --var ...
```
