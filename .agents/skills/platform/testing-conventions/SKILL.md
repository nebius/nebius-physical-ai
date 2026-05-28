---
name: testing-conventions
description: Use before running or interpreting NPA tests, lint checks, or validation reports.
last_verified: 2026-05-26
owner: platform
version: 1.0.0
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

Expected baseline: 1242+ passed, 21 skipped, 1 xpassed, 0 failures.

Use evidence-based convergence: report numeric pass counts and exact failure messages, not subjective assessment.

## Unit Test Rules

Tests must not hit real infrastructure. Mock SSH, S3, Nebius APIs, GPUs, and network calls at the call site. CLI tests use `typer.testing.CliRunner` against `npa.cli.main:app`.

## Changelog

- 2026-05-26: Added frontmatter metadata (last_verified, owner, version) and Changelog section per skill-authoring.
