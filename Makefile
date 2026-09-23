# Nebius Physical AI - developer workflow shortcuts.
#
# Works out of the box with the npa/.venv that CONTRIBUTING prescribes, activated
# or not. Override the interpreter for a venv kept elsewhere:
#   make test PYTHON=~/.venvs/npa/bin/python
#
# Targets run pytest from the npa/ package where the pytest config lives.

# Default to the repo venv, then to python3. Defaulting to a bare `python` meant
# every target failed with "python: not found" in the prescribed setup, because
# that venv is usually not activated and many systems ship only `python3`.
# The path must be absolute: recipes cd into npa/ before running $(PYTHON).
DEFAULT_PYTHON := $(shell \
	if [ -x "$(CURDIR)/npa/.venv/bin/python" ]; then echo "$(CURDIR)/npa/.venv/bin/python"; \
	elif command -v python3 >/dev/null 2>&1; then echo python3; \
	else echo python; fi)
PYTHON ?= $(DEFAULT_PYTHON)
PYTEST := cd npa && $(PYTHON) -m pytest

# Live/GPU/e2e markers. Deselecting by marker is more robust than ignoring a
# directory: gpu/e2e-marked tests also live under tests/workbench/ and will try
# to launch real infrastructure if a developer has SkyPilot/creds configured.
LIVE_DESELECT := -m "not e2e and not e2e_serverless and not e2e_skypilot and not e2e_pipeline and not gpu and not multi_gpu and not byovm_live and not ngc_e2e"

# The CLI reference is generated from live `npa --help`, so a PYTHON override
# should also select that interpreter's console script. Only export NPA_BIN when
# the derived script actually exists: build_docs.sh already resolves npa/.venv and
# then PATH, and handing it a path that is not there suppresses both. `command -v`
# is empty when $(PYTHON) is not on PATH at all, which is how an unconditional
# `dirname` produced "./npa" and broke every docs target.
NPA_BIN_FOR_PYTHON = NPA_BIN="$${NPA_BIN:-$$(bin=$$(command -v $(PYTHON) 2>/dev/null) \
	&& [ -x "$$(dirname "$$bin")/npa" ] && printf '%s' "$$(dirname "$$bin")/npa" || true)}"

.PHONY: help install-dev check-env test-prereqs test test-smoke test-all test-e2e \
	test-guardrails lint format format-check precheck merge-precheck docs docs-check check \
	ci-runners-status ci-runners-down

RUNNER_STATE ?= $(HOME)/.npa/ci-runners/default

help:
	@echo "Targets:"
	@echo "  install-dev      Install npa with the dev+adapter extras CONTRIBUTING.md requires"
	@echo "  check-env        Fail fast if PYTHON would import npa from another checkout"
	@echo "  test-prereqs     Report (non-blocking) which full-suite/CI-parity tools are missing"
	@echo "  test             Fast default: full unit suite, no live/GPU/network"
	@echo "  test-smoke       Quickest check: onboarding CLI smoke tests only"
	@echo "  test-all         Alias for 'test' (no live tests)"
	@echo "  test-guardrails  Repo guardrails (catalogs, specs, skills, docs, hygiene)"
	@echo "  test-e2e         Opt-in live suite (requires real Nebius infra + NPA_INTEGRATION_E2E=1)"
	@echo "  lint             Ruff lint"
	@echo "  format           Ruff autofix + format"
	@echo "  format-check     Ruff formatting check without changing files"
	@echo "  precheck         Fast local pins, lint, format, and CI contract checks"
	@echo "  merge-precheck   Check committed HEAD merged with fetched origin/main"
	@echo "  ci-runners-status  Show the temporary CPU runner pool"
	@echo "  ci-runners-down    Restore CI routing and drain/delete CPU workers"
	@echo "  docs             Regenerate the CLI reference under docs/cli/"
	@echo "  docs-check       Fail if docs/cli/ has drifted from 'npa --help'"
	@echo "  check            The reproducible PR gates: precheck, docs-check, test"
	@echo "                   (no coverage floor; see CONTRIBUTING.md)"
	@echo "Interpreter: $(PYTHON)"
	@echo "Override it with: make test PYTHON=/path/to/venv/bin/python"

install-dev:
	$(PYTHON) -m pip install -e "npa[dev,adapter]"

# A venv shared (e.g. symlinked) across worktrees/clones keeps whichever
# checkout last ran `pip install -e`, so $(PYTHON) can silently import a
# DIFFERENT checkout's npa while pytest collects THIS checkout's test files.
# Catch that before spending minutes on a run whose result would be
# meaningless. See npa/scripts/check_dev_environment.py for the exact failure
# modes this catches and how to fix each one.
check-env:
	$(PYTHON) npa/scripts/check_dev_environment.py --repo-root "$(CURDIR)"

# Non-blocking: reports which optional prerequisites are missing, and their
# verified real consequence (some self-skip; the adapter extra instead makes
# pytest fail to collect outright). Also reports temp-disk headroom. Never
# exits non-zero itself; see npa/scripts/check_test_prereqs.py.
test-prereqs:
	$(PYTHON) npa/scripts/check_test_prereqs.py

# Fast default: every unit test, with live/GPU/e2e markers deselected.
test: check-env
	$(PYTEST) tests/ --ignore=tests/e2e $(LIVE_DESELECT) --timeout=180 -q

# Tightest loop: just the first-time-user CLI smoke guards (sub-second).
test-smoke: check-env
	$(PYTEST) tests/cli/test_main.py tests/cli/test_onboarding_smoke.py -q

test-all: test

# The harness-guardrails PR gate. Adding a tool, spec, toolRef, image or skill
# usually lands here first, so it is worth running before the full suite.
test-guardrails: check-env
	$(PYTEST) tests/guardrails -q

# Opt-in: launches real Nebius infrastructure. Read docs/testing/ first.
test-e2e: check-env
	cd npa && NPA_INTEGRATION_E2E=1 $(PYTHON) -m pytest tests/e2e -q

lint:
	cd npa && $(PYTHON) -m ruff check .

format:
	cd npa && $(PYTHON) -m ruff check --fix . && $(PYTHON) -m ruff format .

format-check:
	cd npa && $(PYTHON) -m ruff format --check .

# Keep cheap failures ahead of docs generation and the full suite, even with -j.
precheck: check-env
	$(PYTHON) npa/scripts/ci_requirements.py --check
	$(MAKE) lint format-check
	$(PYTEST) tests/test_ci_requirements.py tests/test_ci_merge_precheck.py tests/test_ci_precheck.py tests/test_ci_queue_evidence.py tests/test_ci_test_scope.py tests/test_ci_cpu_runners.py tests/test_merge_queue_report.py tests/guardrails/test_ci_workflows.py tests/guardrails/test_ci_concurrency.py -q

# Fetch first; this deliberately reports exact commits and never edits the index.
merge-precheck:
	$(PYTHON) npa/scripts/ci_merge_precheck.py --base origin/main --head HEAD

ci-runners-status:
	$(PYTHON) npa/scripts/ci_cpu_runners.py status --state-dir "$(RUNNER_STATE)"

ci-runners-down:
	$(PYTHON) npa/scripts/ci_cpu_runners.py down --state-dir "$(RUNNER_STATE)"

# docs/cli/ is generated and drift-gated in CI. Regenerate and commit it whenever
# a command, flag or help string changes.
docs:
	$(NPA_BIN_FOR_PYTHON) bash scripts/build_docs.sh

docs-check:
	$(NPA_BIN_FOR_PYTHON) bash scripts/build_docs.sh --check

# Run the reproducible local subset in cost order; Linux CI also enforces coverage.
check: precheck
	$(MAKE) docs-check test
