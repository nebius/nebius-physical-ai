# Live GPU E2E On The Dev VM

Live GPU e2e runs execute from the Nebius Dev VM, on demand, by a human
operator. They do not run in GitHub Actions and there is no cron, systemd timer,
nightly workflow, or other scheduled execution.

Use this path for the `gpu and e2e` pytest subset, including the VLM live GPU
tests, SONIC live SkyPilot test, and the spine e2e once its marker lands.

## Why Not GitHub Actions

GitHub-hosted runners are outside Nebius. The VMs launched during these tests
are reachable from the Nebius-connected Dev VM, but they are not reliably
reachable over SSH from GitHub-hosted runners. A hosted workflow can therefore
start infrastructure and still fail to validate or tear it down correctly.

Do not reintroduce a hosted `live-e2e.yml` workflow for this path. The Dev VM is
the runner location for live GPU validation.

## Why There Is No Nightly Job

Live GPU e2e tests provision real GPU resources. A scheduled unattended run can
spend money overnight and can leave a leaked cluster when nobody is watching.

Runs stay manual and on demand. Do not install a cron entry, systemd timer, PM2
job, GitHub Actions schedule, or any other automatic trigger for
`scripts/live-e2e.sh`.

## Prerequisites

Run from the Nebius-connected Dev VM checkout after installing the normal NPA
development environment:

```bash
cd /opt/nebius-physical-ai
npa/.venv/bin/python -m pip install -e npa
```

The runner uses credentials from the VM process environment and local VM config
files only. It does not read GitHub Actions secrets.

By default the script looks for local env files at:

- `/home/ubuntu/codex-runner/env`
- `/home/ubuntu/codex-runner/.env`
- `~/.codex-runner/env`
- `~/.codex-runner/.env`
- `~/.npa/live-e2e.env`

To use a specific file:

```bash
export NPA_LIVE_E2E_ENV_FILE=/path/to/dev-vm-live-e2e.env
```

The SkyPilot executable defaults to:

```bash
export NPA_SKYPILOT_BIN=/home/ubuntu/.npa/skypilot-venv/bin/sky
```

Override it only when the Dev VM has a different SkyPilot venv:

```bash
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"
```

## Run

Run manually:

```bash
bash scripts/live-e2e.sh
```

The script runs:

```bash
npa/.venv/bin/python -m pytest -m "gpu and e2e" npa/tests/ -q
```

The default GPU candidate order is H100 first:

```text
H100:1,H200:1,A100:1,L40S:1,RTX6000:1
```

Override the order for a manual run:

```bash
export NPA_LIVE_E2E_GPU_CANDIDATES=H100:1,H200:1,L40S:1
bash scripts/live-e2e.sh
```

## Logs And Notifications

Each run writes a timestamped log under:

```text
~/npa-live-e2e-logs/
```

Override the log directory with `NPA_LIVE_E2E_LOG_DIR`.

ntfy is optional but expected on the Dev VM. Configure either a full URL or a
topic name:

```bash
export NPA_NTFY_TOPIC_URL=https://ntfy.sh/<topic>
# or
export NPA_NTFY_TOPIC=<topic>
```

When `GITHUB_TOKEN` or `GH_TOKEN` is available in the Dev VM environment, the
script posts a GitHub commit status to the current commit with context
`live-e2e/dev-vm`. Disable that for a local dry run:

```bash
export NPA_LIVE_E2E_POST_GITHUB_STATUS=0
```

## Verified Teardown

Before pytest starts, and again after it exits, the runner clears matching
SkyPilot clusters and polls until none remain. The default prefixes are:

```text
npa-vlm-live npa-sonic-e2e npa-spine-e2e npa-live-e2e
```

The teardown path calls `sky down --yes <cluster>` for matching clusters and
then checks `sky status --refresh` until the prefix list is empty. The pytest
command itself does not pass a teardown flag to SkyPilot; teardown remains in the
script trap and the tests' own `finally` fixtures.

If the process is interrupted, the trap still runs teardown. If teardown cannot
verify an empty status before the timeout, the script exits non-zero and reports
the failure in ntfy and the commit status when those are configured.

## Post-Merge Validation

After this runner lands and the spine e2e marker fix merges, run
`bash scripts/live-e2e.sh` once on the Dev VM to validate the full selected live
set. Do not add a timer after that validation; keep future runs manual.
