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
| `NPA_E2E_CLUSTER_CONTEXT` | Exact disposable cluster/context for the manual `mutation-live` tier |
| `NPA_E2E_AGENT_NAME` | Exact disposable agent name for the manual `mutation-live` tier |
| `NPA_E2E_CONTROLLER_TRANSACTION_RUN_ID` | Unique run ID for the manual controller transaction regression |
| `NPA_REGISTRY_ID` | Nebius registry id, passed through for image resolution/inspection |
| `NPA_DAILY_E2E_SHARDS` | Days to spread the S3 e2e suite over (default 7) |
| `NPA_DAILY_ENABLE_GPU` | Set to `1` to have `e2e-daily` also run one rotating real-GPU workflow submit |
| `NPA_DAILY_AGENT_GPU_E2E` | Set to `1` with `gpu-daily` to run the agent-confirmed self-hosted VLM proof instead of the rotating case; requires a deployed agent record |

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
| `e2e` | `unit` plus the whole real-infra S3 e2e suite (`-m e2e`) | Real S3 buckets; self-cleaning and budget-bounded (see [e2e.md](e2e.md)) |
| `e2e-daily` | **Scheduled default.** Comprehensive-workflow coverage gate + plans every `npa.workflow` spec + inspects every workbench image in the registry + a **different rotating shard** of the S3 e2e suite each day + (when `NPA_DAILY_ENABLE_GPU=1`) one rotating real-GPU workflow submit (see below) | CPU + a fraction of the S3 e2e suite; optional single self-cleaning GPU job |
| `gpu-daily` | ONE rotating real-GPU workflow-submit E2E — a different GPU twin each day via a self-cleaning Nebius managed job (`cancel-on-timeout`) | Real GPU spend, bounded to one managed job |
| `e2e-serverless` | Serverless endpoint/job e2e (`-m e2e_serverless`) | Real serverless spend; requires `NPA_E2E_SERVERLESS_PROJECT`; self-terminating (see [e2e-serverless.md](e2e-serverless.md)) |
| `mutation-live` | PR lifecycle and controller-transaction tests against exact project/context/agent/run selectors | **Opt-in, manual only** — destructive mutation; rejected from scheduled events |
| `live-gpu` | Delegates to `scripts/live-e2e.sh` (the full `-m "gpu and e2e"` sky-cluster suite) | **Opt-in, manual only** — real GPU clusters with verified teardown |

The **scheduled** run uses the tier in the workflow `env.SCHEDULED_TEST_TIER`
(default `e2e-daily`). Manual dispatch chooses any tier from a dropdown.

## The `e2e-daily` tier

Every day it runs, in order:

1. **Comprehensive-workflow coverage gate** (`npa/scripts/daily_workflow_e2e.py check`).
   A "comprehensive" workflow is an `npa.workflow` spec with at least 4
   executable steps. The gate fails if a required workflow-reachable image
   stops being covered by any comprehensive workflow (regression guard); the
   source of truth is `npa.orchestration.npa_workflow.daily_coverage`.
2. **Plan every `npa.workflow` spec** (validate + render/plan, no GPU). This
   exercises the >= 4-step comprehensive workflow E2Es (e.g. the 11-step
   `bdd100k-pipeline` and the Physical AI Data Factory blueprint) every day.
3. **Inspect every workbench image** (`daily_workflow_e2e.py images --inspect`).
   Resolves all `CONTAINER_IMAGE_NAMES` to their pinned registry refs and checks
   presence with `crane`/`skopeo`/`docker` on the dev VM. Report-only by default;
   set `NPA_DAILY_REQUIRE_IMAGES=1` to fail when a required image is absent.
4. **A rotating shard of the S3 e2e suite.** The `-m e2e` tests are collected,
   sorted, and split into `NPA_DAILY_E2E_SHARDS` buckets (default 7); the bucket
   for today's day-of-year runs. Over `NPA_DAILY_E2E_SHARDS` days the whole S3
   e2e suite is covered, a different subset each day.

Inspect coverage and today's plan-set by hand:

```bash
npa/.venv/bin/python npa/scripts/daily_workflow_e2e.py report      # coverage table
npa/.venv/bin/python npa/scripts/daily_workflow_e2e.py plan-set    # today's >=4-step specs
npa/.venv/bin/python npa/scripts/daily_workflow_e2e.py images      # all image refs
```

Images not yet exercised by a >= 4-step workflow are tracked explicitly in
`daily_coverage.EXEMPT_IMAGE_TOOLS` (currently `sonic`, `retargeting`,
`cosmos3-reason`, `lerobot`, `genesis`, `groot`) so the gap is visible; shrink
that set by extending or authoring a comprehensive workflow, never grow it to
hide a regression. Every image — including the exempt ones — is still checked
for registry presence daily by step 3.

## Real GPU E2E (bounded + rotating)

Real GPU e2e is a first-class part of the daily run. Because unattended runs
must not leak GPU spend, it is implemented as **one self-cleaning managed-job
submit per day**, not the whole `gpu and e2e` suite:

- `daily_workflow_e2e.py gpu-case` picks today's GPU workflow twin from
  `submit_matrix.gpu_submit_cases()`, rotating by day-of-year so every twin is
  exercised over the window, and prints which live test can drive it
  (`<spec>\t<one-shot|runtime>`) — a twin with a parallel group or a
  decision-driven loop is only collected by the runtime test. `plan_only` stubs
  and `rotation_skip` twins (each with a documented `skip_reason`) are excluded
  so the rotation only picks a case that can actually pass standalone. Every
  GPU-launching twin has been run live on RTXPRO-6000; the rotation cycles the
  verified-passing set:

  Wall clock is the submit test end to end (provision, run, poll to terminal),
  which is what has to fit `NPA_DAILY_GPU_MAX_WAIT_SECONDS`:

  | Twin | Live wall clock |
  | --- | --- |
  | `sonic-train` | 3m41s |
  | `sonic-export` | 6m11s |
  | `sonic-locomotion-finetuning` | 7m01s |
  | `vlm-eval-single` | 8m23s |
  | `sonic-export-eval` | 8m28s |
  | `tokenfactory-cosmos-gate` | 11m40s |
  | `mjlab-eval` | not re-measured here |
  | `cosmos3-reason` | not re-measured here |
  | `tokenfactory-rollout-judge` | not re-measured here |
  | `vlm-eval-benchmark` | not re-measured here |
  | `isaac-lab-rl-sweep` | not re-measured here |

  Two twins stay out, each with an accurate `skip_reason` in
  `submit_matrix.py`: `sonic-eval` is a single consume-only stage whose input is
  an ONNX a previous export wrote (SONIC eval is covered in the rotation by the
  `sonic-export-eval` chain), and `bdd100k-pipeline` drives 11 sequential stages
  against workbench services deployed in-cluster.
- The runner submits just that twin via the sanctioned
  `test_npa_workflow_submit_live_reaches_terminal` path with a bounded wait and
  `NPA_E2E_NPA_WORKFLOW_SUBMIT_CANCEL_ON_TIMEOUT=1`, so a stuck job is cancelled
  rather than left running. Registry + accelerator remap come from the dev VM's
  `~/.npa/live-e2e.env` (e.g. `H100:1=RTXPRO6000:1`).
- Enable it in the daily schedule with the `NPA_DAILY_ENABLE_GPU=1` repo
  variable; run it on demand any time with the `gpu-daily` tier.

Bound the wait with `NPA_DAILY_GPU_MAX_WAIT_SECONDS` (default 2400),
`NPA_DAILY_GPU_POLL_SECONDS` (30), and `NPA_DAILY_GPU_PYTEST_TIMEOUT` (2600).

**Validated live on the dev VM.** Each twin above submitted a Nebius managed job
onto `1x RTXPRO-6000-BLACKWELL-SERVER-EDITION` (cluster `npa-rtxpro-mk8s`),
reached `SUCCEEDED`, and self-cleaned with no leftover cluster. The rotation has
already paid for itself in real product bugs it surfaced:

- The self-hosted vLLM server was never started by the render (`[Errno 111]
  Connection refused`); once started, a 7B VLM's cold start did not fit a
  bounded run; and once it fit, the engine died in warmup twice — `ninja` (a
  vLLM dependency) was not on the stage shell's PATH, and then FlashInfer's JIT
  sampler wanted an `nvcc` the image does not ship.
- `sonic train` had no runtime that trains in the job it is already running in;
  all three delegated to more infrastructure.
- `sonic export` / `sonic eval` could not exchange artifacts between stages,
  because each stage runs on its own cluster.
- The sim2real decision writer invoked a bare `python`, which is not on PATH in
  the image the rotation actually uses.
- The runner sourced `npa-cloud-env.sh` after `live-e2e.env`, and that script
  unsets `AWS_ACCESS_KEY_ID`, so every GPU twin silently **skipped**.

## Fresh checkout and separate process

Each run uses a **dedicated, isolated** checkout (`NPA_CI_REPO_DIR`, default
`~/npa-ci-daily`) with its own venv, so it never disturbs the shared dev clone
or other agents' worktrees. With `NPA_DAILY_FRESH=1` (default) the checkout is
hard-reset and cleaned to a pristine ref each run (the venv is preserved to
avoid reinstalling dependencies daily). With `NPA_DAILY_DETACH=1` (default) the
runner re-execs under `setsid` so the suite runs as its own session/process
group, decoupled from the invoking SSH TTY.

## GPU guardrail

Two distinct GPU paths, with different safety postures:

- **Bounded daily GPU** (`gpu-daily`, or the GPU phase of `e2e-daily` when
  `NPA_DAILY_ENABLE_GPU=1`): ONE rotating managed-job submit per day, bounded
  wait, `cancel-on-timeout`, self-cleaning. This is safe to run on a schedule
  and is the operator-authorized way to keep real GPU coverage daily.
- **Full `live-gpu` suite** (`scripts/live-e2e.sh`, the entire
  `-m "gpu and e2e"` set launching many sky clusters): still **manual only** and
  must **never** run unattended on a schedule. Two guards enforce this: the
  workflow refuses to dispatch `live-gpu` from a `schedule` event, and the
  runner only runs it when `NPA_DAILY_ALLOW_LIVE_GPU=1` (set solely for a manual
  `live-gpu` dispatch). It keeps its own pre/post SkyPilot teardown verification.

## Run it by hand on the dev VM

```bash
# Fast, no infra:
bash scripts/dev-vm-daily-tests.sh unit main

# The scheduled daily tier (coverage gate + plan all specs + image inspect +
# today's rotating S3 e2e shard):
bash scripts/dev-vm-daily-tests.sh e2e-daily main

# The whole S3 e2e suite at once (needs ~/.npa credentials):
bash scripts/dev-vm-daily-tests.sh e2e main

# Serverless e2e against a sandbox project:
NPA_E2E_SERVERLESS_PROJECT=project-eXXXXXXXXXXXX \
  bash scripts/dev-vm-daily-tests.sh e2e-serverless main

# Exact disposable lifecycle/controller mutation (all four selectors required):
NPA_E2E_PROJECT=alias NPA_E2E_CLUSTER_CONTEXT=context \
NPA_E2E_AGENT_NAME=agent NPA_E2E_CONTROLLER_TRANSACTION_RUN_ID=run-id \
  bash scripts/dev-vm-daily-tests.sh mutation-live main

# GPU live path (explicit opt-in):
NPA_DAILY_ALLOW_LIVE_GPU=1 bash scripts/dev-vm-daily-tests.sh live-gpu main
```

Logs are written under `~/npa-daily-test-logs/` on the dev VM.
