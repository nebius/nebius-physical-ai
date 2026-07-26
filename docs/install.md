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
- **Windows:** install from [python.org](https://www.python.org/downloads/)
  (check "Add python.exe to PATH"), or from the Microsoft Store.

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, do this inside **WSL2 Ubuntu** (see [§6](#6-windows-via-wsl2)).

## 3. Install npa (editable, from the clone)

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
pip install -e npa
```

Verify:

```bash
npa --version
```

Prefer [`uv`](https://docs.astral.sh/uv/)? It can install Python and create the
env in one go: `uv venv .venv && source .venv/bin/activate && uv pip install -e npa`.

The base install is the core CLI, with no GPU or database wheels. Add extras
when a workload requires them:

```bash
pip install -e "npa[full]"      # everything below except the GPU extras
pip install -e "npa[data]"      # pandas/scipy/matplotlib curation + reports
pip install -e "npa[lancedb]"   # LanceDB workbench tool
pip install -e "npa[viz]"       # Rerun viewer/recording integration
pip install -e "npa[server]"    # FastAPI policy/eval server
pip install -e "npa[adapter]"   # dataset conversion
pip install -e "npa[genesis]"   # Genesis + distillation stages (GPU)
pip install -e "npa[groot]"     # GR00T SDK (GPU)
pip install -e "npa[dev]"       # tests, lint (pytest, ruff)
```

Activate the venv in every new shell (`source .venv/bin/activate`), or call the
interpreter directly with `./.venv/bin/npa` without activating.

## 4. Nebius CLI (required)

`npa` runs on Nebius, so the Nebius AI Cloud CLI is part of the standard setup —
`npa configure` and every managed workbench deploy use it. Install it:

```bash
# macOS / Linux (and inside WSL2)
curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
export PATH="${HOME}/.nebius/bin:${PATH}"   # add to ~/.zshrc or ~/.bashrc to persist
```

`npa configure` creates or reuses a local profile for you (no manual
`nebius profile create` step). See <https://docs.nebius.com/cli/install> for
details.

## 5. Optional operator tools

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
