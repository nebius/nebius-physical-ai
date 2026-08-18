---
name: pre-pr-validation
description: Use before pushing an npa change to pick which gates apply and run them locally in cheapest-first order — the map from each CI job to its exact local command.
---

# Pre-PR Validation

Six checks across five workflows gate every pull request: `Lint / ruff`,
`Lint / docs-drift`, `Test / test (3.12)`, `harness guardrails`, `gitleaks`, and
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
# 1. Lint — seconds. This mirrors CI; `make lint` checks all of npa/ instead.
npa/.venv/bin/python -m ruff check npa/src npa/tests

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

**Some local failures are missing host tools, not broken code.** The suite shells
out to real binaries that CI has and a bare workstation or container may not.
Before investigating a failure, check whether it is one of these:

- `Required executable not found: kubectl` in `npa/tests/test_provisioning.py`
  means `kubectl` is not installed. Install it or accept the skip.
- `FileNotFoundError: 'npa'` in `npa/tests/workflows/test_bdd100k_pipeline.py`
  means the `npa` console script is not on `PATH`. Fix it by prepending the
  venv: `PATH="$PWD/npa/.venv/bin:$PATH"`.

When a failure looks unrelated to your change, confirm it against a clean base
before spending time on it:

```bash
git worktree add /tmp/main-check origin/main
cd /tmp/main-check && npa/.venv/bin/python -m pytest <the failing test> -q
```

The shared dev/operator VM has both of those binaries, so the full suite passes
there with nothing deselected. It is also shared, so take an isolated worktree +
venv + tmux session rather than checking out in the common clone:

```bash
npa/scripts/dev_vm_isolated_session.sh start <branch> <run-id>
```

Give that session its own extras — `pip install -e "npa[dev,adapter]"` — before
running the suite, and never install into the shared venv. Do not set
`NPA_ISOLATED_FAST=1` for a full-suite run: fast mode skips the per-run venv and
borrows the shared one, which lacks those extras, so collection dies on
`jsonschema` and `av` before a single test executes.

**Docs drift is expensive to re-run blind.** Change all CLI options first, then
regenerate once with `bash scripts/build_docs.sh`. Per-subcommand option changes
do not alter the generated pages; a new top-level command adds a page and a
README index line.

**The confidentiality scan has two modes, and only one is reproducible locally.**
CI scans with an operator denylist supplied as a repository secret, which you do
not have. The built-in Nebius pattern set needs no secret and covers the
deterministic leak classes, so it is the local substitute — not the same check.

Scope it to your diff:

```bash
# Your change only — this is the one that must be clean.
npa/.venv/bin/python -m npa.guardrails.confidentiality \
  --repo-root . --diff-range origin/main..HEAD --built-in-nebius-infra
```

Do not run the built-in patterns over `--tree` and read the result as a verdict
on your change. That scan exits non-zero on this repository today, with dozens
of hits in long-standing files — including `.gitleaks.toml`, which necessarily
contains the patterns being searched for. Those are allowlisted in CI. A tree
scan tells you nothing about whether you introduced a leak; the diff scan does.

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
