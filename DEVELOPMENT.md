# Developer Guide

This guide gets a new developer productive on the `nebius-physical-ai`
repository. It focuses on the local dev loop: clone, install, run tests, lint,
and find the right place to make a change. For platform concepts and operator
workflows, start with the [README](README.md), the
[npa quickstart](docs/quickstart.md), and the
[Workbench getting started guide](docs/workbench/getting-started.md).

For contributing a new Workbench tool (HTTP API + CLI + SDK + container), read
[CONTRIBUTING.md](CONTRIBUTING.md) after this page.

## 1. Prerequisites

- macOS or Linux. Windows is not tested.
- Python 3.10 or newer (the `npa` package requires `>=3.10`).
- Git, `python3 -m venv`, and `pip`.
- Optional for running real workloads (not required for unit tests):
  - [`nebius` CLI](https://docs.nebius.com/cli/install)
  - `terraform` on `PATH`
  - `ffmpeg`
  - Docker, if you plan to build Workbench container images
  - SSH keypair (`~/.ssh/id_ed25519` by default)

Unit tests must not require any of the optional tools above. They mock SSH,
S3, Nebius APIs, and GPU calls at the call site.

## 2. Clone and install

The Python package lives in [npa/](npa/). Always work inside the in-repo
virtualenv at `npa/.venv` so the helper scripts and docs pick the right
interpreter.

```bash
git clone <REPO_URL> nebius-physical-ai
cd nebius-physical-ai

python3 -m venv npa/.venv
npa/.venv/bin/python -m pip install --upgrade pip
npa/.venv/bin/python -m pip install -e npa
```

Use the venv interpreter explicitly (`npa/.venv/bin/python`) or activate it:

```bash
source npa/.venv/bin/activate
```

Verify:

```bash
npa --version
npa --help
```

`npa --help` must print the command tree without requiring Nebius, Hugging
Face, NGC, Kubernetes, or S3 credentials.

### Optional extras

Install only what you need for the area you are working on:

```bash
npa/.venv/bin/python -m pip install -e "npa[server]"    # FastAPI policy server
npa/.venv/bin/python -m pip install -e "npa[adapter]"   # dataset conversion
npa/.venv/bin/python -m pip install -e "npa[genesis]"   # Genesis + distillation
npa/.venv/bin/python -m pip install -e "npa[groot]"     # GR00T SDK
```

Dev tools (ruff) come from the `dev` dependency group:

```bash
npa/.venv/bin/python -m pip install ruff
```

## 3. Repository layout

```text
.
|-- README.md                # Platform overview and solution index
|-- CONTRIBUTING.md          # How to add a new Workbench tool
|-- DEVELOPMENT.md           # This file
|-- docs/                    # Architecture, CLI reference, workbench, demos, testing
|-- npa/
|   |-- pyproject.toml       # Package metadata, extras, pytest markers
|   |-- src/npa/             # CLI, SDK, workbench tool implementations
|   |   |-- cli/             # Typer CLI entry points (npa <solution> <tool> <verb>)
|   |   |-- sdk/             # Compatibility SDK surface
|   |   |-- workbench/       # Per-tool service + wrapper modules
|   |   `-- solutions/       # Solution registry (solutions.toml)
|   |-- docker/workbench/    # Per-tool Dockerfiles and tag governance
|   |-- manifests/workbench/ # Kubernetes / runtime manifests
|   |-- workflows/workbench/ # SkyPilot workflow YAMLs and runners
|   |-- scripts/             # Pipeline runners (bdd100k, isaac lab rl, diagnostics)
|   |-- deploy/              # Service install + systemd unit
|   `-- tests/               # Pytest suite (unit, smoke, e2e)
|-- research/                # Research prototypes (not shipped)
`-- scripts/                 # Repo-level docs and drift helpers
```

The `npa` CLI is shaped as:

```text
npa <solution> <tool> <verb> [options]
```

Today `<solution>` is `workbench`. See
[docs/architecture/cli-namespaces.md](docs/architecture/cli-namespaces.md) for
the namespace contract and
[docs/architecture/solutions-model.md](docs/architecture/solutions-model.md)
before adding a new solution.

## 4. Running the test suite

The pytest config lives in [npa/pyproject.toml](npa/pyproject.toml). Run tests
from the repo root through the venv interpreter.

### Default (unit + smoke, no infrastructure)

```bash
npa/.venv/bin/python -m pytest npa/tests
```

E2E and live-target tests are skipped by default. Unit tests must not touch
real infrastructure; mock SSH, S3, Nebius APIs, and GPU calls at the call
site.

### Filter by area

```bash
# Single file
npa/.venv/bin/python -m pytest npa/tests/test_config.py

# By marker (see markers list in npa/pyproject.toml)
npa/.venv/bin/python -m pytest npa/tests -m smoke
```

Registered markers include `e2e`, `e2e_serverless`, `e2e_skypilot`,
`e2e_pipeline`, `smoke`, `multi_gpu`, `ngc_e2e`, and `byovm_live`.

### End-to-end tests (real infrastructure)

E2E tests create real Nebius resources and require valid credentials in
`~/.npa/credentials.yaml`. Gate them with `NPA_INTEGRATION_E2E=1`:

```bash
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest npa/tests/e2e -v
```

Full details: [docs/testing/e2e.md](docs/testing/e2e.md) and
[docs/testing/smoke-tests.md](docs/testing/smoke-tests.md).

### CLI tests

CLI tests use `typer.testing.CliRunner` against `npa.cli.main:app`. Do not
shell out to the `npa` binary from unit tests.

## 5. Lint and format

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and
formatting.

```bash
npa/.venv/bin/python -m ruff check npa
npa/.venv/bin/python -m ruff format npa
```

Run both before opening a PR.

## 6. Local config and credentials

`npa` reads two files in `~/.npa/`:

- `~/.npa/credentials.yaml` — user-authored secrets (Nebius, Hugging Face,
  NGC, etc.). See [docs/credentials.yaml.example](docs/credentials.yaml.example).
- `~/.npa/config.yaml` — machine-managed project, workbench, endpoint, SSH,
  storage, and Terraform state. Deploy commands write this; do not hand-edit
  it as part of dev setup.

Create and lock down the credentials file once:

```bash
mkdir -p ~/.npa
chmod 700 ~/.npa
touch ~/.npa/credentials.yaml
chmod 600 ~/.npa/credentials.yaml
```

Never hardcode project IDs, tenant IDs, registry IDs, bucket names, or
secrets in code, tests, or docs. Credentials always come from the credentials
file or environment variables.

## 7. Common developer tasks

### Add or change a CLI command

CLI entry points live under [npa/src/npa/cli/](npa/src/npa/cli/). The
top-level Typer app is `npa.cli.main:app`. Workbench tools are wired in
`npa/src/npa/cli/workbench/__init__.py`.

Cross-tool input/output handoff uses `--input-path` and `--output-path`
validated through `npa/src/npa/cli/path_contract.py`. Both must accept `s3://`
URIs.

### Add or change a Workbench tool

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the full HTTP + CLI + SDK +
container pattern. Reference implementations:

- LeRobot (full tool):
  [npa/src/npa/cli/workbench/lerobot.py](npa/src/npa/cli/workbench/lerobot.py)
- Detection training (cleanest HTTP + SDK shape):
  [npa/src/npa/workbench/detection_training/service.py](npa/src/npa/workbench/detection_training/service.py)

### Add or change a SkyPilot workflow

Workflow YAMLs and runner scripts live in
[npa/workflows/workbench/](npa/workflows/workbench/). SkyPilot is installed in
an isolated venv outside the `npa` package environment. See
[docs/orchestration/skypilot-setup.md](docs/orchestration/skypilot-setup.md).

### Build a Workbench container image

Per-tool Dockerfiles and tag governance live in
[npa/docker/workbench/](npa/docker/workbench/). Tag families and security
scanning are described in
[docs/security/image-reproducibility.md](docs/security/image-reproducibility.md).

### Regenerate CLI docs

CLI reference pages under [docs/cli/](docs/cli/) are generated from Typer help
output. Regenerate with:

```bash
scripts/build_docs.sh
```

## 8. Before opening a PR

1. `npa/.venv/bin/python -m ruff check npa`
2. `npa/.venv/bin/python -m ruff format --check npa`
3. `npa/.venv/bin/python -m pytest npa/tests`
4. Update or add docs under [docs/](docs/) when behavior, CLI flags, or
   credentials change.
5. Do not commit `~/.npa/` files, real project IDs, or secrets.

## 9. Where to get help

- Platform overview: [README.md](README.md)
- Operator quickstart: [docs/quickstart.md](docs/quickstart.md)
- Workbench setup: [docs/workbench/getting-started.md](docs/workbench/getting-started.md)
- CLI reference: [docs/cli/README.md](docs/cli/README.md)
- Architecture: [docs/architecture/](docs/architecture/)
- Testing: [docs/testing/](docs/testing/)
- Troubleshooting: [docs/workbench/troubleshooting/](docs/workbench/troubleshooting/)
