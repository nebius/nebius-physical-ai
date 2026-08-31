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

**Run it in parallel.** The suite is xdist-safe (`pytest-xdist` is in the `dev`
extra); `-n auto` takes ~6 min down to ~2 min on 8 cores with an identical pass
count, which is the difference between running the whole suite each iteration and
only running a slice:

```bash
.venv/bin/python -m pytest tests/ -q -n auto
```

Use the serial form when a failure needs a readable, ordered traceback. CI still
runs serially with coverage, so treat a parallel pass as the fast signal, not as a
replacement for the gate.

**Docs drift is a required gate and is slow to re-run blind.** `scripts/build_docs.sh`
memoizes and prefetches its `npa --help` walk (~1 min, was ~5), but it still costs a
minute — so change CLI options first, then regenerate once, rather than checking
after each edit. Per-subcommand *options* do not change the generated pages; a new
top-level command adds `docs/cli/<name>.md` plus a README index line.

`ruff` is available in the venv:

```bash
npa/.venv/bin/python -m ruff check <files>
```

## Existing Failures And Gates

- Known pre-existing failure: `tests/smoke/test_cosmos_serverless_smoke.py` has 5 tests gated by `NPA_COSMOS_SERVERLESS_SMOKE=1`; they fail with `Unable to list Nebius VPC networks for project project-smoke: unsupported`.
- E2E tests are gated by `NPA_INTEGRATION_E2E=1` and excluded from standard runs with `--ignore=npa/tests/e2e`.
- Pipeline E2E tests use the `e2e_pipeline` pytest marker.
- Live Nebius Token Factory tests use the `token_factory_e2e` marker (in `npa/tests/e2e/test_token_factory_e2e.py`). They self-skip without a real `NEBIUS_TOKEN_FACTORY_KEY`; the marker is in conftest `_LIVE_MARKERS` so the key is not scrubbed. Run with `NEBIUS_TOKEN_FACTORY_KEY=... pytest npa/tests/e2e/test_token_factory_e2e.py`.

The gate for `make test` is **0 failures**. Most recent measurement:
`10836 passed, 37 skipped, 12 deselected, 1 xpassed`, ~14 min serial (2026-08-18 at
`1b89b3ba`).

Treat that pass count as a floor, never as an equality. It is stale by
construction — it rises whenever tests land, and it was previously recorded as a
closed total that went wrong at the next merge. A higher number is normal; only a
count that has *fallen* indicates tests stopped being collected, and the reliable
comparison is against your own merge base. Two things move it legitimately: a few
tests self-skip without `node`, `tmux`, or `docker`, and `make test` deselects the
live/GPU markers so it collects a different tree from `test.yml`.

The suite is hermetic: it needs no `kubectl`, no cluster, and no venv at a
particular path. A failure that names a missing binary or `ModuleNotFoundError:
No module named 'npa'` is a bug in the test or the script it drives, not a
missing prerequisite — three of those were fixed in `516396ec`.

A hermeticity fix needs a test that fails without it in *any* environment. CI
pip-installs the package, so a fix that only matters when the `npa` console script
is absent from `PATH` cannot regress-fail there: prune `PATH` inside the test, or
assert on the helper directly, rather than relying on the ambient environment.

Failures that appear only on an operator/dev VM are usually an ambient env var
the conftest does not scrub, not a real regression. `tests/conftest.py` scrubs
credential and infra-targeting variables for every non-live test precisely
because CI runs with them unset and a working machine does not. When a test
asserts on argv or rendered output, check that every env var the product reads to
build it is in that scrub list: `NPA_NEBIUS_PROFILE` / `NEBIUS_PROFILE` were
missing, and because product code prepends `--profile <name>` whenever either is
set, ten `mk8s` argv assertions failed on any machine where an operator had
selected a profile. Add the variable to the conftest tuple rather than working
around it in the test; the tests that exercise the variable's own behavior set it
with `monkeypatch.setenv` after the scrub runs.

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
