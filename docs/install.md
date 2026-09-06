# Installing npa on every platform

`npa` targets **macOS**, **Linux**, and **Windows via WSL2**, and works with
**Python 3.10+**. It is not published to PyPI, so you always install it editable
from a clone of this repository.

The [quickstart install](quickstart.md#3-install-npa) has the short,
copy-paste path. This page collects the full per-platform detail — Python setup,
the Nebius CLI, WSL2, and optional operator tools — for when the default needs a
little more.

## Platform support at a glance

| Platform | Supported |
| --- | --- |
| macOS (Intel / Apple Silicon) | native |
| Linux (Debian/Ubuntu, ...) | native |
| Windows | via **WSL2 Ubuntu** |

`npa` runs cloud workloads (S3, SkyPilot, Kubernetes) that assume a POSIX
environment, so on Windows do all `npa` work from **WSL2 Ubuntu**.

## 1. Get Python 3.10+

Check what you have:

```bash
python3 --version
```

If it is older than 3.10 (or missing):

- **macOS:** `brew install python@3.12`, or download from
  [python.org](https://www.python.org/downloads/), or use
  [pyenv](https://github.com/pyenv/pyenv).
- **Debian/Ubuntu:** the interpreter is usually current; also install the venv
  module (shipped separately): `sudo apt-get install -y python3 python3-venv`.
- **Windows:** use the Debian/Ubuntu commands inside WSL2 Ubuntu. A native
  Windows Python installation is not used by this setup.

## 2. Clone the repository and create a virtual environment

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, do this inside **WSL2 Ubuntu** (see [§6](#6-windows-via-wsl2)).

> Create the environment **after entering the clone** so activation and cleanup
> resolve the same directory. User-facing quickstarts use repo-root `.venv`.
> [Contributor tooling](../CONTRIBUTING.md#testing-requirements) and `AGENTS.md`
> use `npa/.venv/bin/python` for repository validation. Both layouts work; keep
> one environment per checkout and use its path consistently.

## 3. Install npa (editable, from the clone)

```bash
pip install -e npa
```

Verify:

```bash
npa --version
npa workbench --help
```

Prefer [`uv`](https://docs.astral.sh/uv/)? From the clone root, use these commands
instead of the venv creation and installation commands above:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -e npa
```

The base install is fully capable: a plain `pip install -e npa` already
includes every non-GPU workbench dependency (dataframe/reporting, LanceDB, the
Rerun viewer, and the local eval/agent server). There is no separate
`npa[full]` step. Only these extras are opt-in:

```bash
pip install -e "npa[genesis]"   # Genesis + distillation stages (GPU, local)
pip install -e "npa[groot]"     # GR00T SDK (GPU, local)
pip install -e "npa[sonic]"     # SONIC ONNX export/runtime (GPU, local)
pip install -e "npa[agent-eval]"  # guardrails-ai output validators (optional)
pip install -e "npa[agent-trace]" # Langfuse/OpenTelemetry tracing (optional)
pip install -e "npa[dev]"       # tests, lint (pytest, ruff)
```

The GPU/simulation wheels above are only needed when you run those engines
**locally**; cloud jobs execute them inside the Nebius images they launch. The
`full`, `data`, `lancedb`, `viz`, and `server` extras still resolve (as no-ops
now folded into the base install) so older `npa[full]` commands keep working.

From the clone root, activate the venv in every new shell
(`source .venv/bin/activate`), or call `.venv/bin/npa` directly without
activating.

### Safely uninstall the repository-local environment

Ordinary `npa cleanup` removes operational caches only; it never removes the
environment containing the running `npa` executable. From a supported checkout
layout (`<repo>/.venv` from this guide, or contributor `<repo>/npa/.venv`), first
preview the exact target:

```bash
npa uninstall
npa uninstall --json
```

Actual removal requires both explicit flags:

```bash
npa uninstall --remove-environment --yes
```

For removal safeguards, failure receipts, and retries, see
[environment removal and recovery](teardown.md#repository-local-environment-removal-and-recovery).

## 4. Nebius CLI (required)

`npa` runs on Nebius, so the Nebius AI Cloud CLI is part of the standard setup —
`npa configure` and every managed workbench deploy use it. Install it:

```bash
# macOS / Linux (and inside WSL2)
curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh \
  | NEBIUS_CLI_VERSION=0.12.254 bash
export PATH="${HOME}/.nebius/bin:${PATH}"   # add to ~/.zshrc or ~/.bashrc to persist
```

NPA's recommended/tested version is `0.12.254`; `0.12.227` is also tested and
continues with a compatibility warning. Untested versions are blocked with an
exact command to install the recommended version, so the install guide and the
runtime parser contract cannot silently drift apart.

`npa configure` creates or reuses a local profile for you (no manual
`nebius profile create` step). See <https://docs.nebius.com/cli/install> for
details.

<a id="5-optional-operator-tools"></a>

## 5. Tools for cloud workloads

Install the tools your workload uses:

- **Terraform 1.x** is required for managed VM/cluster provisioning and agent
  deployment. The Python package does not install it. Check `terraform version`;
  agent bootstrap installs the tested 1.13.3 baseline only if Terraform is absent.
- **kubectl** is required for Kubernetes operations.
- **socat** is required by SkyPilot Kubernetes on Debian/Ubuntu:
  `sudo apt-get install -y socat`.
- **Docker** is needed for local container runs and image builds. Supported NPA
  images pull anonymously from GHCR.
- **jq** is used by shell examples that parse JSON.

Platform installation:

- **macOS:** `brew install jq`, plus `kubectl` and `terraform` from their
  official installers (Terraform is no longer in Homebrew core:
  `brew install hashicorp/tap/terraform`).
- **Debian/Ubuntu:** `sudo apt-get install -y jq`. `kubectl` and `terraform` are
  **not** in the stock apt repositories — install `kubectl` from the
  [Kubernetes docs](https://kubernetes.io/docs/tasks/tools/) and `terraform`
  from [HashiCorp's apt repo](https://developer.hashicorp.com/terraform/install#linux).

## 6. Windows via WSL2

On Windows, run all `npa` work from WSL2 Ubuntu. In PowerShell (admin), install
WSL once:

```powershell
wsl --install -d Ubuntu
```

Restart if prompted, open **Ubuntu** from the Start menu, then follow the
Debian/Ubuntu steps above inside WSL. Keep the repo under your Linux home (for
example `~/nebius-physical-ai`), not under `/mnt/c/…`, for faster I/O and fewer
path issues.

## Out of scope (extra steps needed)

These work but are not covered by the copy-paste path:

- **Alpine / musl** distros: some wheels are glibc-only; expect to build from
  source or use a glibc base image.
- **Brand-new Python** (a release so new that dependency wheels do not exist
  yet): pin to the latest Python that has wheels, or build from source.
- **Air-gapped machines:** pre-mirror the repo and a wheelhouse; `pip install -e`
  still needs the transitive dependencies available offline.
