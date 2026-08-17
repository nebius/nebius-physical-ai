---
name: pre-pr-validation
description: Use before pushing an npa change to pick which gates apply and run them locally in cheapest-first order — the map from each CI job to its exact local command.
---

# Pre-PR Validation

Six workflows gate every pull request: `Lint / ruff`, `Lint / docs-drift`,
`Test / test (3.12)`, `harness guardrails`, `gitleaks`, and
`confidentiality scan`. A seventh, `image-security-scan`, is path-triggered on
Docker changes.

All of them are reproducible locally. Run them in cost order so the cheap ones
catch the common mistakes before you spend minutes on the full suite.

For how the test suite itself behaves — markers, hermetic fixtures, parallel
runs, live-infra expectations — use `skills/atomic/testing-conventions/SKILL.md`.
This skill is the gate map.

## Use The Repo Virtualenv

`npa/.venv/bin/python` (Python 3.12). Never bare `python`. The `make` targets
default `PYTHON` to bare `python` and `cd` into `npa/` first, so pass an
absolute path:

```bash
make test PYTHON=/workspace/npa/.venv/bin/python
```

## The Ladder

```bash
# 1. Lint — seconds. CI runs the narrower `src tests`; make lint checks all of npa/.
cd npa && ../npa/.venv/bin/python -m ruff check src tests

# 2. Onboarding smoke — ~20s.
make test-smoke PYTHON=/workspace/npa/.venv/bin/python

# 3. Guardrails — ~50s, ~2000 static contract assertions. Highest signal per second.
npa/.venv/bin/python -m pytest npa/tests/guardrails -q

# 4. The tests for what you touched — seconds to a minute.
npa/.venv/bin/python -m pytest npa/tests/cli/test_<tool>_cli.py -q

# 5. Full unit suite — minutes. This is the PR gate.
make test PYTHON=/workspace/npa/.venv/bin/python

# 6. Docs drift — ~1-2 min. Only when CLI commands or options changed.
bash scripts/build_docs.sh --check

# 7. Confidentiality, diff-scoped — seconds.
npa/.venv/bin/python -m npa.guardrails.confidentiality \
  --repo-root . --diff-range origin/main..HEAD --built-in-nebius-infra
```

Steps 1 through 3 catch most failures and cost about a minute together. Run them
after every meaningful edit; save 5 and 6 for before you push.

## Which Gates Apply To Your Change

| You changed | Also run |
|---|---|
| A CLI command or any of its options | `bash scripts/build_docs.sh --check`, plus the argv guardrails — a renamed option breaks every `toolRef` that passes it |
| A `toolRef` or the catalog | `test_tool_catalog_argv`, `test_module_toolref_argv`, `test_shown_workflow_catalog`, and the catalog doc sync test |
| An `npa.workflow` spec | `npa workbench workflow validate-spec <path>`, plus live-matrix registration per the testing conventions |
| A workbench tool's surface | `test_three_tier_contract`, and the tool's CLI and workbench tests |
| A Dockerfile or image tag | `npa/.venv/bin/python npa/docker/workbench/check_tag_consistency.py`, plus `npa/tests/docker/` |
| A `SKILL.md` or `skills/index.yaml` | `test_skills_index` and `test_develop_skills` |
| Terraform or the agent deploy path | `test_terraform_provisioner_shell`, plus a real destroy/deploy cycle |
| Docs only | Lint and `pytest --collect-only` as a smoke check |

## Gate Details Worth Knowing

**`make test` is not identical to CI.** It deselects live and GPU markers and
sets a 180s timeout; CI runs with coverage and enforces `--cov-fail-under=60`.
A local pass is a strong signal, not proof of the CI result.

**Docs drift is expensive to re-run blind.** Change all CLI options first, then
regenerate once with `bash scripts/build_docs.sh`. Per-subcommand option changes
do not alter the generated pages; a new top-level command adds a page and a
README index line.

**The confidentiality scan has two modes.** CI runs an operator denylist supplied
as a repository secret, which you will not have locally. The built-in Nebius
pattern set needs no secret and covers the deterministic leak classes.

Scope it to your diff. A `--tree` scan reads every tracked file and reports
pre-existing hits in files you never touched, which is noise for a PR check:

```bash
# Your change only — this is the one that should be clean.
npa/.venv/bin/python -m npa.guardrails.confidentiality \
  --repo-root . --diff-range origin/main..HEAD --built-in-nebius-infra
```

**Gitleaks needs full history.** CI checks out with `fetch-depth: 0` and scans
the PR range. Locally: `gitleaks detect --source . --config .gitleaks.toml
--redact --no-banner --log-opts="origin/main..HEAD"`.

**Typecheck is advisory.** The mypy workflow is manual-dispatch only and does not
block.

## Interpreting A Failure

Guardrail failures map to fixes through
`skills/atomic/guardrail-failures/SKILL.md`. Report numeric results — pass
counts and exact error messages — rather than a subjective assessment; the repo
convention is evidence-based convergence.
