# Daily Tests On The Dev VM (SSH From GitHub)

GitHub-hosted runners live outside the operator's Nebius environment, so they
cannot reach real infrastructure to run e2e tests. See
[Live GPU E2E On An Operator Host](live-e2e.md) for why hosted runners cannot
validate live infra directly.

This workflow inverts the direction. Instead of running tests on the hosted
runner, it **SSHes into the dev/operator VM** — which already has
`~/.npa/credentials.yaml`, `~/.npa/config.yaml`, and SkyPilot — and runs the
suite there. A GitHub-hosted runner is only used as an orchestrator.

- Workflow: `.github/workflows/dev-vm-daily-tests.yml`
- Runner script (executes on the dev VM): `scripts/dev-vm-daily-tests.sh`

## Setup

Add these repository secrets under **Settings → Secrets and variables →
Actions**:

| Secret | Purpose |
| --- | --- |
| `DEV_VM_SSH_HOST` | Hostname or IP of the dev VM |
| `DEV_VM_SSH_USER` | SSH login user |
| `DEV_VM_SSH_PRIVATE_KEY` | Private key (PEM) authorized on the dev VM |
| `DEV_VM_SSH_PORT` | Optional; defaults to `22` |

Optional repository **variables** (not secrets) forwarded to the run:

| Variable | Purpose |
| --- | --- |
| `NPA_E2E_SERVERLESS_PROJECT` | Sandbox project id for the `e2e-serverless` tier |
| `NPA_E2E_PROJECT` | Override the S3 e2e project alias (harness auto-selects `eu-north1` otherwise) |

The dev VM must already have `git`, `python3`, and `make`, plus a reachable git
remote and valid `~/.npa` credentials. The runner script uses a dedicated CI
checkout (`~/npa-ci-daily` by default) with its own venv, so it never disturbs
the shared dev clone or other agents' worktrees.

## Schedule

The workflow runs daily at `07:00 UTC` and can also be triggered manually via
**Run workflow** (`workflow_dispatch`). Change the `cron` line to match the
operator's off-peak window.

## Test tiers

The runner script accepts a tier and a git ref: `dev-vm-daily-tests.sh [tier]
[git-ref]`.

| Tier | What runs | Cost / cleanup |
| --- | --- | --- |
| `unit` | Fast unit suite (`make test`) + ruff lint | No infra, no spend |
| `e2e` | `unit` plus the real-infra S3 e2e suite (`-m e2e`) | Real S3 buckets; self-cleaning and budget-bounded (see [e2e.md](e2e.md)) |
| `e2e-serverless` | Serverless endpoint/job e2e (`-m e2e_serverless`) | Real serverless spend; requires `NPA_E2E_SERVERLESS_PROJECT`; self-terminating (see [e2e-serverless.md](e2e-serverless.md)) |
| `live-gpu` | Delegates to `scripts/live-e2e.sh` (`-m "gpu and e2e"`) | **Opt-in, manual only** — real GPU clusters with verified teardown |

The **scheduled** run uses the tier in the workflow `env.SCHEDULED_TEST_TIER`
(default `e2e`). Manual dispatch chooses any tier from a dropdown.

## GPU guardrail

The GPU-spending `live-gpu` tier must **never** run unattended on a schedule: it
provisions real GPU clusters and can leak spend overnight
([live-e2e.md](live-e2e.md)). Two independent guards enforce this:

1. The workflow refuses to dispatch `live-gpu` from a `schedule` event and caps
   the scheduled tier at `env.SCHEDULED_TEST_TIER`.
2. The runner script only runs the `live-gpu` tier when
   `NPA_DAILY_ALLOW_LIVE_GPU=1`, which the workflow sets only for a manual
   `live-gpu` dispatch.

Even in the `live-gpu` tier the work is delegated to `scripts/live-e2e.sh`,
which keeps its existing pre-run and post-run SkyPilot teardown verification.

## Run it by hand on the dev VM

```bash
# Fast, no infra:
bash scripts/dev-vm-daily-tests.sh unit main

# Real S3 e2e (needs ~/.npa credentials):
bash scripts/dev-vm-daily-tests.sh e2e main

# Serverless e2e against a sandbox project:
NPA_E2E_SERVERLESS_PROJECT=project-eXXXXXXXXXXXX \
  bash scripts/dev-vm-daily-tests.sh e2e-serverless main

# GPU live path (explicit opt-in):
NPA_DAILY_ALLOW_LIVE_GPU=1 bash scripts/dev-vm-daily-tests.sh live-gpu main
```

Logs are written under `~/npa-daily-test-logs/` on the dev VM.
