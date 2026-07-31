---
name: testing-conventions
description: Use before running or interpreting NPA tests, lint checks, or validation reports.
---

# Testing Conventions

Use the repository virtualenv. Never use bare `python`; use `npa/.venv/bin/python`.

Correct command from repo root:

```bash
npa/.venv/bin/python -m pytest npa/tests/ --ignore=npa/tests/e2e --timeout=120 -q
```

Correct command from inside `npa/`:

```bash
cd npa
.venv/bin/python -m pytest tests/ --ignore=tests/e2e --timeout=120 -q
```

`ruff` is available in the venv:

```bash
npa/.venv/bin/python -m ruff check <files>
```

## Existing Failures And Gates

- Known pre-existing failure: `tests/smoke/test_cosmos_serverless_smoke.py` has 5 tests gated by `NPA_COSMOS_SERVERLESS_SMOKE=1`; they fail with `Unable to list Nebius VPC networks for project project-smoke: unsupported`.
- E2E tests are gated by `NPA_INTEGRATION_E2E=1` and excluded from standard runs with `--ignore=npa/tests/e2e`.
- Pipeline E2E tests use the `e2e_pipeline` pytest marker.
- Live Nebius Token Factory tests use the `token_factory_e2e` marker (in `npa/tests/e2e/test_token_factory_e2e.py`). They self-skip without a real `NEBIUS_TOKEN_FACTORY_KEY`; the marker is in conftest `_LIVE_MARKERS` so the key is not scrubbed. Run with `NEBIUS_TOKEN_FACTORY_KEY=... pytest npa/tests/e2e/test_token_factory_e2e.py`.

Expected baseline: 1242+ passed, 21 skipped, 1 xpassed, 0 failures.

Use evidence-based convergence: report numeric pass counts and exact failure messages, not subjective assessment.

## Unit Test Rules

Tests must not hit real infrastructure. Mock SSH, S3, Nebius APIs, GPUs, and network calls at the call site. CLI tests use `typer.testing.CliRunner` against `npa.cli.main:app`.

## Live-Infra Testing Is A Priority (not optional)

Smoke + mocked-unit tests are necessary but **not sufficient**. Any change to an
`npa.workflow` spec, a workbench tool / `toolRef`, or the agent/deploy
(Terraform / `provisioner`) path MUST also land committed **live-infra** coverage
— do not stop at smoke. Definition of done: the change is wired into a live path
and you report numeric results from running it.

- **New/changed npa.workflow spec:** register it in `SUBMIT_LIVE_MATRIX`
  (`npa/src/npa/orchestration/npa_workflow/submit_matrix.py`) with the right tier
  (`cpu` | `gpu` | `multi`). If it has a dynamic gate/loop, also add it to
  `DYNAMIC_SPECS` in `npa/tests/e2e/npa_workflow_live_helpers.py` so the runner
  supplies `--assume-decision`; if it actually executes (non-plan-only), seed
  inputs in `seed_live_workflow_inputs`. Use `plan_only=True` **only** when stages
  are stubs or a real run would burn a GPU on a stub (the repo convention:
  "do not burn GPUs on stubs"). Keep `test_submit_live_matrix.py` green.
  Run it:
  ```bash
  NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=<cpu|gpu|multi> \
    NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=<spec>.yaml \
    ./scripts/npa-workflow-submit-live-e2e.sh
  # plan-only preflight (no job launch):
  NPA_E2E_NPA_WORKFLOW_SUBMIT_PLAN_ONLY=1 ./scripts/npa-workflow-submit-live-e2e.sh
  ```
- **Live workflow render/infra:**
  ```bash
  NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest \
    npa/tests/e2e/test_npa_workflow_live_e2e.py \
    npa/tests/e2e/test_npa_workflow_live_infra.py -q
  ```
- **Token Factory-backed stages:** the `token_factory_e2e` live tests must pass
  with a real `NEBIUS_TOKEN_FACTORY_KEY`.
- **Agent/deploy changes:** validate a real `npa agent destroy` + `deploy` (and,
  for credential-path changes, that it is reproducible). Known trap: a stale
  ambient `NEBIUS_IAM_TOKEN` shadows the fresh `var.iam_token` in the Nebius
  Terraform provider (`PermissionDenied`/`Unauthenticated` even though the CLI
  works); `provisioner._run` scrubs it, but when reproducing by hand
  `unset NEBIUS_IAM_TOKEN NPA_NEBIUS_IAM_TOKEN` first.
- If a full live run is genuinely infeasible in the environment, say so
  explicitly and still commit the `plan_only` live-matrix entry — never silently
  ship smoke-only.
