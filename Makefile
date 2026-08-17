# Nebius Physical AI - developer workflow shortcuts.
#
# Activate your virtualenv first (see docs/quickstart.md), or override PYTHON:
#   make test PYTHON=~/.venvs/npa/bin/python
#
# Targets run pytest from the npa/ package where the pytest config lives.

PYTHON ?= python
PYTEST := cd npa && $(PYTHON) -m pytest

# Live/GPU/e2e markers. Deselecting by marker is more robust than ignoring a
# directory: gpu/e2e-marked tests also live under tests/workbench/ and will try
# to launch real infrastructure if a developer has SkyPilot/creds configured.
LIVE_DESELECT := -m "not e2e and not e2e_serverless and not e2e_skypilot and not e2e_pipeline and not gpu and not multi_gpu and not byovm_live and not ngc_e2e"

# The CLI reference is generated from live `npa --help`, so the docs targets need
# the console script that belongs to $(PYTHON) rather than whatever is on PATH.
NPA_BIN_FOR_PYTHON = NPA_BIN="$${NPA_BIN:-$$(dirname "$$(command -v $(PYTHON))")/npa}"

.PHONY: help install-dev test test-smoke test-all test-e2e test-guardrails \
	lint format docs docs-check check

help:
	@echo "Targets:"
	@echo "  install-dev      Install npa with dev/test tooling into the active venv"
	@echo "  test             Fast default: full unit suite, no live/GPU/network"
	@echo "  test-smoke       Quickest check: onboarding CLI smoke tests only"
	@echo "  test-all         Alias for 'test' (no live tests)"
	@echo "  test-guardrails  Repo guardrails (catalogs, specs, skills, docs, hygiene)"
	@echo "  test-e2e         Opt-in live suite (requires real Nebius infra + NPA_INTEGRATION_E2E=1)"
	@echo "  lint             Ruff lint"
	@echo "  format           Ruff autofix + format"
	@echo "  docs             Regenerate the CLI reference under docs/cli/"
	@echo "  docs-check       Fail if docs/cli/ has drifted from 'npa --help'"
	@echo "  check            Everything the PR gates block on: lint, docs-check, test"
	@echo "Override the interpreter with: make test PYTHON=/path/to/venv/bin/python"

install-dev:
	$(PYTHON) -m pip install -e "npa[dev]"

# Fast default: every unit test, with live/GPU/e2e markers deselected.
test:
	$(PYTEST) tests/ --ignore=tests/e2e $(LIVE_DESELECT) --timeout=180 -q

# Tightest loop: just the first-time-user CLI smoke guards (sub-second).
test-smoke:
	$(PYTEST) tests/cli/test_main.py tests/cli/test_onboarding_smoke.py -q

test-all: test

# The harness-guardrails PR gate. Adding a tool, spec, toolRef, image or skill
# usually lands here first, so it is worth running before the full suite.
test-guardrails:
	$(PYTEST) tests/guardrails -q

# Opt-in: launches real Nebius infrastructure. Read docs/testing/ first.
test-e2e:
	cd npa && NPA_INTEGRATION_E2E=1 $(PYTHON) -m pytest tests/e2e -q

lint:
	cd npa && $(PYTHON) -m ruff check .

format:
	cd npa && $(PYTHON) -m ruff check --fix . && $(PYTHON) -m ruff format .

# docs/cli/ is generated and drift-gated in CI. Regenerate and commit it whenever
# a command, flag or help string changes.
docs:
	$(NPA_BIN_FOR_PYTHON) bash scripts/build_docs.sh

docs-check:
	$(NPA_BIN_FOR_PYTHON) bash scripts/build_docs.sh --check

# Mirrors the blocking PR gates so a contributor can reproduce them in one command.
check: lint docs-check test
