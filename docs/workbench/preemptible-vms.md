# Workbench preemptible VMs

**Short answer:** yes — Workbench GPU VM deploys support Nebius **preemptible**
(spot-style) instances. They are **on by default** for most tools. Use
`--no-preemptible` when you need a VM that stays up until you tear it down.

Preemptible VMs cost less but can be **stopped by the platform at any time**.
Design long jobs around checkpoints in S3 and idempotent resume.

## What supports preemptible

| Surface | Flag | Default | Notes |
| --- | --- | --- | --- |
| `lerobot deploy` | `--preemptible` / `--no-preemptible` | preemptible | Terraform VM |
| `genesis deploy` | same | preemptible | Terraform VM |
| `groot deploy` | same | preemptible | Terraform VM |
| `isaac-lab deploy` | same | preemptible | Terraform VM |
| `cosmos deploy` (`--runtime vm`) | same | preemptible | Terraform VM |
| `fiftyone deploy` (GPU) | same | preemptible | CPU-only deploys ignore preemptible |
| `cosmos deploy` / jobs (`--runtime serverless`) | `--preemptible` | off in API unless set | Nebius AI endpoint flag |
| `workflow submit` (SkyPilot VM tasks) | `--use-spot` / `--no-use-spot` | template-dependent | SONIC / materialized GPU tasks |

Under the hood, Terraform sets `enable_preemptible=true`, which maps to
`on_preemption = "STOP"` and `recovery_policy = "FAIL"` on the Nebius instance
(bundled module: `npa/src/npa/deploy/terraform/`).

## Quick start: cheap GPU workbench

Deploy a preemptible LeRobot VM (default — no extra flag needed):

```bash
npa workbench lerobot -p <project> -n cheap-h200 deploy \
  --gpu-type gpu-h200-sxm \
  --gpu-preset 1gpu-16vcpu-200gb
```

Force a **non-preemptible** VM for a long training block:

```bash
npa workbench lerobot -p <project> -n stable-h200 deploy \
  --gpu-type gpu-h200-sxm \
  --gpu-preset 1gpu-16vcpu-200gb \
  --no-preemptible
```

Same pattern works on `genesis`, `groot`, `isaac-lab`, and `cosmos` VM deploys.

Preview without provisioning:

```bash
npa workbench lerobot -p <project> -n preview deploy \
  --gpu-type gpu-h200-sxm --gpu-preset 1gpu-16vcpu-200gb \
  --preemptible --dry-run
```

## SkyPilot workflows

For workflow stages that materialize a Nebius GPU VM through SkyPilot, pass
spot/preemptible at submit time:

```bash
npa workbench workflow submit npa/workflows/workbench/skypilot/sonic-eval.yaml \
  --run-id sonic-spot \
  --use-spot
```

Use `--no-use-spot` when the stage must hold a non-preemptible VM for its full
runtime.

## After a preemption

1. **Checkpoints to S3** — train/export commands should write artifacts under
   `s3://<bucket>/...` (LeRobot, SONIC export, sim-to-real stages all use this
   pattern).
2. **Redeploy or resume** — redeploy the workbench alias, then rerun from the
   latest checkpoint path. Many tools accept `--input-path s3://...` on eval,
   serve, or resume flows.
3. **Teardown** — destroy the alias when finished so preemptible VMs are not
   left orphaned:

   ```bash
   npa workbench lerobot -p <project> -n cheap-h200 deploy --destroy
   ```

See also the preemptible H200 research flow in
[`research/lerobot-deploy/README.md`](../../research/lerobot-deploy/README.md)
for trap-on-exit S3 upload patterns.

## Verify a deployed VM

After a successful deploy, confirm preemptible on the instance (replace IDs
with yours):

```bash
# From ~/.npa/config.yaml workbench entry or terraform output
nebius compute instance get --id <instance-id> --format json \
  | jq '.spec.preemptible // .preemptible'
```

You should see a non-null preemptible block when `--preemptible` was used.

## When *not* to use preemptible

- Multi-hour unattended training without frequent checkpointing
- Interactive demos or hackathon dry-runs where interruption is costly
- Benchmarks that require stable wall-clock comparisons (`--no-preemptible`)

For production-ish runs, prefer `--no-preemptible` or managed Kubernetes /
serverless paths where your tool supports them.

## Validated on this repo

| Check | Result |
| --- | --- |
| CLI wiring (`enable_preemptible=true` on GPU deploy) | `pytest npa/tests/cli/test_fiftyone_cli.py::test_fiftyone_deploy_accepts_gpu_flags_and_installs_app` — pass |
| Preemptible / non-preemptible CLI regression | `pytest npa/tests/cli/test_preemptible_deploy.py` — pass |
| IAM-restricted bootstrap reuse | `bootstrap_environment()` reuses saved S3 credentials when access-key provisioning is blocked; `ensure_service_account()` parses the service-account id from restricted `get-by-name` responses |
| Dry-run deploy path (`lerobot`, `fiftyone`, `--preemptible`) | pass against configured project alias |
| Live deploy + instance verify (`agent-live`, `gpu-l40s-a`, eu-north1) | pass — `preemptible { on_preemption = STOP }` confirmed via `nebius compute instance get` |
| Full live deploy + instance verify | run `NPA_PREEMPTIBLE_E2E=1 pytest npa/tests/e2e/test_preemptible_live_e2e.py` on a profile with compute + IAM permissions |

Related: [getting-started.md](getting-started.md),
[guides/README.md](guides/README.md), [CLI.md](../../CLI.md).
