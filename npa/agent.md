# agent.md — Add Unit Tests to the `npa` CLI

## Goal

Add comprehensive unit tests to the `npa` package (`npa/src/npa/`) using pytest. The repo currently has a single test file (`npa/tests/test_adapter.py`). Target: every module under `npa/src/npa/` has corresponding tests with ≥80% line coverage on pure-logic code (config parsing, data transforms, CLI argument handling). Network/SSH/GPU code gets interface tests with mocks.

---

## Repo Layout

```
nebius-physical-ai/
├── npa/
│   ├── pyproject.toml          # package def, deps: typer httpx paramiko pyyaml rich boto3 jinja2
│   ├── src/npa/
│   │   ├── cli/
│   │   │   ├── main.py         # Typer app, subcommands: workbench, adapter, workflow
│   │   │   ├── adapter/        # `npa adapter convert`
│   │   │   ├── workbench/      # `npa workbench lerobot ...`, `npa workbench genesis ...`
│   │   │   │   ├── lerobot.py
│   │   │   │   └── (genesis commands)
│   │   │   ├── workflow/       # `npa workflow run|status|logs|teardown|distill`
│   │   │   └── genesis/        # `npa workbench genesis train-teacher|generate-demos|...`
│   │   ├── adapter/            # data format conversion logic (SimToLeRobot etc.)
│   │   ├── clients/            # httpx/paramiko/boto3 wrappers for remote calls
│   │   ├── server/             # FastAPI server (npa-lerobot-server)
│   │   ├── lerobot/            # LeRobot integration helpers
│   │   ├── genesis/            # Genesis simulation helpers
│   │   ├── workflows/          # pipeline orchestration (distill_two_vm.py etc.)
│   │   ├── deploy/             # Terraform/SSH deploy logic
│   │   └── config/             # YAML config loading, sample_config.yaml, domain_random.yaml
│   └── tests/
│       └── test_adapter.py     # existing — has fixtures demo_dir, output_dir, ffmpeg skip
├── research/                   # research scripts, not in scope
└── CLAUDE.md
```

CLI entrypoint: `npa.cli.main:app_entry` (Typer). Three subcommand groups with ~25 commands total.

---

## Rules

### 1. Environment & Dependencies

- Work inside `npa/`. The venv is at `npa/.venv` — activate it or use `npa/.venv/bin/python`.
- Install test deps if missing:
  ```bash
  cd npa
  .venv/bin/pip install pytest pytest-cov pytest-mock
  ```
- Do NOT install GPU-only packages (lerobot, genesis, torch). Tests must run on CPU-only CI. Guard imports:
  ```python
  pytest.importorskip("lerobot")
  ```

### 2. Test File Conventions

- Mirror the source tree: `npa/tests/test_<module>.py` for flat modules, `npa/tests/<subpackage>/test_<module>.py` for nested ones.
- Target mapping:
  ```
  src/npa/adapter/       → tests/test_adapter.py (extend existing)
  src/npa/cli/main.py    → tests/cli/test_main.py
  src/npa/cli/adapter/   → tests/cli/test_adapter_cli.py
  src/npa/cli/workbench/ → tests/cli/test_workbench_cli.py
  src/npa/cli/workflow/  → tests/cli/test_workflow_cli.py
  src/npa/cli/genesis/   → tests/cli/test_genesis_cli.py
  src/npa/clients/       → tests/test_clients.py
  src/npa/config/        → tests/test_config.py
  src/npa/server/        → tests/test_server.py
  src/npa/lerobot/       → tests/test_lerobot.py
  src/npa/genesis/       → tests/test_genesis.py
  src/npa/workflows/     → tests/test_workflows.py
  src/npa/deploy/        → tests/test_deploy.py
  ```
- Add `__init__.py` to every new test subdirectory.
- Add a `npa/tests/conftest.py` with shared fixtures.

### 3. What to Test — Priority Order

**P0 — Pure logic (no mocks needed, test first):**
- Config loading: parse `sample_config.yaml`, validate keys, handle missing file, handle malformed YAML.
- Adapter transforms: the existing `test_adapter.py` covers some; extend for edge cases (empty dirs, missing fields, format round-trips).
- Any utility/helper functions (data format converters, path builders, config mergers).

**P1 — CLI commands (mock infra, test arg parsing + dispatch):**
- Use `typer.testing.CliRunner` to invoke commands.
- Mock everything below the CLI layer (SSH, HTTP, S3, subprocess).
- Verify: correct exit codes, expected output fragments, error messages for bad args, `--help` works for every command.
- Template for a CLI test:
  ```python
  from typer.testing import CliRunner
  from npa.cli.main import app

  runner = CliRunner()

  def test_workbench_lerobot_list(mocker):
      mocker.patch("npa.cli.workbench.lerobot.some_client_call", return_value=[...])
      result = runner.invoke(app, ["workbench", "lerobot", "list"])
      assert result.exit_code == 0
      assert "some_expected_output" in result.output
  ```

**P2 — Client wrappers (mock network boundary):**
- `clients/`: mock `httpx.Client`, `paramiko.SSHClient`, `boto3.client`. Verify request construction, header assembly, retry logic, error mapping.
- `deploy/`: mock subprocess/Terraform calls. Verify command strings, config file generation.

**P3 — Server endpoints (if FastAPI app exists):**
- Use `httpx.AsyncClient` with `app` from `npa.server` as transport.
- Mock any LeRobot/Genesis imports behind the endpoints.

**P4 — Workflow orchestration:**
- `workflows/distill_two_vm.py`: mock SSH + S3 calls, verify step ordering, error handling on partial failure.

### 4. Mocking Strategy

- **Patch at the call site**, not at the definition. Example: if `npa.cli.workbench.lerobot` imports `run_ssh_command` from `npa.clients`, patch `npa.cli.workbench.lerobot.run_ssh_command`.
- Use `pytest-mock`'s `mocker` fixture (cleaner than `unittest.mock.patch` decorators).
- For boto3: use `mocker.patch("npa.clients.boto3.client")` or consider `moto` if S3 logic is complex.
- For paramiko: mock at the `SSHClient` level — don't let tests open real connections.
- For httpx: mock the `Client.request` / `Client.get` / `Client.post` methods.

### 5. conftest.py Shared Fixtures

Create `npa/tests/conftest.py` with:

```python
import os
import pytest
import tempfile
from pathlib import Path

@pytest.fixture
def tmp_workspace(tmp_path):
    """A clean temp directory simulating a workspace."""
    return tmp_path

@pytest.fixture
def sample_config(tmp_path):
    """Write a minimal valid config YAML and return its path."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "tenant: test-tenant\n"
        "project: test-project\n"
        "region: eu-north1\n"
        "bucket: test-bucket\n"
    )
    return cfg

@pytest.fixture
def mock_ssh(mocker):
    """Patch paramiko.SSHClient universally."""
    mock_client = mocker.MagicMock()
    mock_client.exec_command.return_value = (
        mocker.MagicMock(),  # stdin
        mocker.MagicMock(read=lambda: b"ok\n"),  # stdout
        mocker.MagicMock(read=lambda: b""),  # stderr
    )
    mocker.patch("paramiko.SSHClient", return_value=mock_client)
    return mock_client

@pytest.fixture
def mock_s3(mocker):
    """Patch boto3 S3 client."""
    mock_client = mocker.MagicMock()
    mocker.patch("boto3.client", return_value=mock_client)
    return mock_client
```

### 6. Running Tests

```bash
cd npa
.venv/bin/python -m pytest tests/ -v --tb=short
.venv/bin/python -m pytest tests/ --cov=npa --cov-report=term-missing
```

### 7. Execution Plan

Do this in order. After each step, run `pytest` and confirm green before moving on.

1. **Setup**: Install test deps, create `conftest.py`, create test subdirs with `__init__.py`.
2. **Config tests** (`test_config.py`): Read each .py file in `src/npa/config/`, find all functions/classes, write tests. This is pure logic — no mocks.
3. **Adapter tests** (`test_adapter.py`): Read existing tests, read `src/npa/adapter/`, add missing coverage.
4. **CLI smoke tests** (`tests/cli/test_main.py`): `--help` for the top-level app and every subcommand group. Should be ~10 tests, all trivial.
5. **CLI command tests** (`tests/cli/test_*_cli.py`): One test file per subcommand group. Read the command source, mock infra calls, test happy path + one error path per command.
6. **Client tests** (`test_clients.py`): Read `src/npa/clients/`, mock network, test request construction.
7. **Workflow tests** (`test_workflows.py`): Read `src/npa/workflows/`, mock SSH+S3, test step sequencing.
8. **Deploy tests** (`test_deploy.py`): Read `src/npa/deploy/`, mock subprocess, test config generation.
9. **Server tests** (`test_server.py`): If FastAPI app exists, test endpoints with httpx test client.
10. **Coverage report**: Run `--cov` and list uncovered lines. Fill gaps on pure-logic code.

### 8. What NOT To Do

- Do NOT run tests that hit real Nebius infrastructure, SSH into VMs, or touch S3.
- Do NOT import `lerobot`, `genesis`, `torch`, or any GPU package at module level in tests. Use `pytest.importorskip()` or mock the import.
- Do NOT refactor source code. Tests only. If you find an untestable function (e.g., 200-line function mixing logic and I/O), note it in a comment but don't refactor.
- Do NOT add type stubs or mypy config. Out of scope.
- Do NOT modify `pyproject.toml` beyond adding `[tool.pytest.ini_options]` if needed.

### 9. pytest Configuration

Add to `npa/pyproject.toml` if not already present:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```

### 10. Success Criteria

- `pytest` passes with 0 failures.
- Every CLI subcommand has at least a `--help` smoke test.
- Every module under `src/npa/` has a corresponding test file (even if some are thin).
- Pure-logic modules (config, adapter) have ≥80% line coverage.
- No test imports GPU packages or opens network connections.
