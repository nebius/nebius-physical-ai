# npa Quickstart

This is the platform entry point for Nebius Physical AI. Read this first to
install the `npa` CLI/SDK and configure the single user-authored credential
store. After this page is complete, continue with
[Workbench Getting Started](workbench/getting-started.md) for Kubernetes,
SkyPilot, registry, S3, and first workload setup.

## 1. Platform overview

`npa` is the Nebius Physical AI platform CLI/SDK. It provides a common command
surface for physical AI workflows such as simulation, training, inference,
visualization, dataset conversion, and storage handoff.

Workbench is the first solution namespace on the platform. Workbench tools are
containerized services that run on Nebius infrastructure and exchange data
through S3-compatible object storage. This quickstart stops before
workbench-specific setup so new engineers have one platform setup path and one
clear next document.

For a broader architecture map, see the repository [README](../README.md) and
the package overview in [npa/README.md](../npa/README.md).

## 2. Prerequisites

- Python 3.10 or newer. The package metadata requires `>=3.10`.
- Git, `python3 -m venv`, and `pip`.
- **macOS**, **Linux**, or **Windows via WSL2** (Ubuntu). Native Windows shells
  (PowerShell, cmd) are not supported for `npa` workflows — use WSL2.
- A Nebius AI Cloud account with billing enabled. Start with the Nebius signup
  guide: <https://docs.nebius.com/signup-billing/sign-up>.
- The Nebius AI Cloud CLI binary on `PATH`. Install it from
  <https://docs.nebius.com/cli/install>; `npa configure` creates or reuses a
  local profile for you (no manual `nebius profile create` step).
- Terraform on `PATH` for later managed `deploy` and `--destroy` commands.
- An SSH public key for later managed VM or BYOVM workbench commands. The
  bundled Terraform defaults to `~/.ssh/id_ed25519.pub`; pass
  `--tf-var ssh_public_key_path=<path>.pub` in deploy commands if you use a
  different key.
- Optional: a Hugging Face token for gated Cosmos and GR00T models, LeRobot
  datasets, and selected model weights:
  <https://huggingface.co/settings/tokens>. Use a read-only token unless a
  workflow explicitly needs write access.
- Optional: an NVIDIA NGC API key for GR00T NGC model paths:
  <https://ngc.nvidia.com/setup/api-key>.

Quick checks:

```bash
python3 --version
git --version
nebius version
nebius profile list
terraform version
```

### Fast install by platform

Copy-paste once per machine. All paths install the Nebius CLI, clone NPA, create
a venv, and verify `npa`. Finish with `nebius profile create` and
`npa configure` (Section 4).

| Platform | Shell | Nebius CLI |
| --- | --- | --- |
| macOS (Intel or Apple Silicon) | Terminal / zsh | Official install script (see below) |
| Linux (Debian/Ubuntu) | bash | `curl …/cli/install.sh \| bash` |
| Windows | **WSL2 Ubuntu** only | same curl one-liner inside WSL |

**macOS**

```bash
curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
export PATH="${HOME}/.nebius/bin:${PATH}"   # add to ~/.zshrc to persist
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e npa
npa --version
nebius profile create
npa configure
```

Optional operator tools: `brew install python@3.12 kubectl terraform jq`.

**Linux (Debian/Ubuntu)**

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip curl
curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
export PATH="${HOME}/.nebius/bin:${PATH}"   # add to ~/.bashrc to persist
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e npa
npa --version
nebius profile create
npa configure
```

Optional: `sudo apt-get install -y kubectl terraform jq` (or install kubectl
from the [Kubernetes docs](https://kubernetes.io/docs/tasks/tools/)).

**Windows (WSL2)**

In PowerShell (admin), install WSL once:

```powershell
wsl --install -d Ubuntu
```

Restart if prompted, open **Ubuntu** from the Start menu, then run the **Linux**
block above inside WSL. Keep the repo under your Linux home (for example
`~/nebius-physical-ai`), not under `/mnt/c/…`, for faster I/O and fewer path
issues.

**Windows (native)** — not supported. Use WSL2 Ubuntu; do not run `npa` from
PowerShell or Git Bash for cluster/S3 workflows.

After install, activate the venv in every new shell:

```bash
cd nebius-physical-ai
source .venv/bin/activate
```

## 3. Install npa

Clone the repository and install the Python package into a fresh virtual
environment. The venv can live anywhere (for example `.venv` in the repo, or
`~/.venvs/npa`); activating it puts `npa` on your `PATH`:

```bash
git clone <REPO_URL> nebius-physical-ai
cd nebius-physical-ai

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e npa
```

If you prefer not to activate the venv, call its interpreter directly
(`./.venv/bin/python -m pip install -e npa`) and use `./.venv/bin/npa` instead
of `npa`. The rest of this guide assumes the venv is activated.

Verify the install:

```bash
npa --version
npa --help
```

Gate: `npa --version` prints `npa <version>`, and `npa --help` prints the
command tree without requiring Nebius, Hugging Face, NGC, Kubernetes, or S3
credentials.

The base install is lightweight: it carries only what the offline paths need.
Optional extras are available when you need them:

```bash
pip install -e "npa[full]"      # everything below except the GPU extras
pip install -e "npa[data]"      # pandas/scipy/matplotlib curation + reports
pip install -e "npa[lancedb]"   # LanceDB workbench tool
pip install -e "npa[viz]"       # Rerun viewer/recording integration
pip install -e "npa[server]"    # FastAPI policy/eval server
pip install -e "npa[adapter]"   # dataset conversion
pip install -e "npa[genesis]"   # Genesis + distillation stages (GPU)
pip install -e "npa[groot]"     # GR00T SDK (GPU)
pip install -e "npa[dev]"       # tests, lint (pytest, ruff); see Section 6
```

Before running cloud workloads (Sections 5+), install `npa[full]` so every
workbench tool has its dependencies.

## 4. Configure credentials

For credential setup, `npa` has one user-authored file:

```text
~/.npa/credentials.yaml
```

Do not choose between multiple NPA credential files. Put user-level secrets in
`~/.npa/credentials.yaml` only. Deploy commands may create or update
`~/.npa/config.yaml` for machine-managed project, workbench, endpoint, SSH,
storage, and Terraform state metadata; do not manually populate
`~/.npa/config.yaml` as part of credential setup.

Environment variables can override file values for a single shell. They are
useful for temporary tests, but the canonical repeatable setup is
`~/.npa/credentials.yaml`. The current source does not read
`NPA_CREDENTIALS_PATH`; it resolves credentials from
`Path.home() / ".npa" / "credentials.yaml"`.

Create and secure the credentials file:

```bash
mkdir -p ~/.npa
chmod 700 ~/.npa
touch ~/.npa/credentials.yaml
chmod 600 ~/.npa/credentials.yaml
```

### 4a. Nebius account authentication

Nebius account authentication is handled by the `nebius` CLI profile, not by a
long-lived `NEBIUS_TOKEN` in `~/.npa/credentials.yaml`.

Before running `npa configure`, sign up for Nebius AI Cloud, note your tenant
id, and create a project in the target region. An Object Storage bucket is
optional — `npa configure` creates a default `npa-bucket` with **standard**
storage and a size cap when you press Enter at the bucket prompt (you can choose
`enhanced` storage or a custom size for new buckets). To reuse your own bucket,
create one first; see the README **Nebius AI Cloud account** section,
[Creating a tenant](https://docs.nebius.com/iam/create-tenants),
[Manage projects](https://docs.nebius.com/iam/manage-projects), and
[Manage buckets](https://docs.nebius.com/object-storage/buckets/manage).

Run interactive setup in a terminal. `npa configure` detects an existing
authenticated Nebius CLI profile (via `nebius iam get-access-token`), prompts
for your tenant id and project id (no defaults are shown), guides you to reuse
an existing bucket or create a default `npa-bucket` (standard storage, size
limit in GB), and only runs profile creation when none is authenticated. It
then writes `~/.npa/credentials.yaml` and `~/.npa/config.yaml`:

```bash
npa configure
```

You do not need to run `nebius profile create` manually; the Nebius CLI binary
must still be installed because `npa` invokes it internally.

In non-interactive environments (CI, pipes), run `npa configure --interactive`
in a real terminal, or `npa configure --show` for the file layout.

Gate: after interactive setup, `nebius iam get-access-token` exits successfully.

Keep these non-secret values handy for later workbench deploys:

- `<YOUR_PROJECT_ID>`: copy it from the Nebius console project selector or list
  projects with the Nebius CLI.
- `<YOUR_TENANT_ID>`: copy it from the Nebius console tenant selector. The
  Nebius docs also show CLI options:
  <https://docs.nebius.com/iam/get-tenants>.
- `<NEBIUS_REGION>`: the region where the project exists, for example
  `eu-north1`.
- `<PROJECT_ALIAS>`: a local alias you choose for `npa`, for example
  `quickstart`.

### 4b. Required credential key names

Use these canonical keys in `~/.npa/credentials.yaml`.

| Need | `credentials.yaml` key | Environment override | Required when |
|---|---|---|---|
| Hugging Face token | `tokens.HF_TOKEN` | `HF_TOKEN` | Downloading gated Hugging Face models, datasets, or weights |
| NGC API key | `ngc.api_key` | `NGC_API_KEY` | Using NGC-backed GR00T model references |
| NGC organization | `ngc.org` | `NGC_ORG` | Your NGC key is organization-scoped |
| NGC team | `ngc.team` | `NGC_TEAM` | Your NGC key is team-scoped |
| BYOVM SSH host | `ssh.host` | `NPA_BYOVM_HOST`, `NPA_SSH_HOST` | BYOVM commands need a default host |
| BYOVM SSH user | `ssh.user` | `NPA_BYOVM_SSH_USER`, `NPA_SSH_USER` | BYOVM commands need a default SSH user |
| BYOVM SSH private key | `ssh.key_path` | `NPA_BYOVM_SSH_KEY`, `NPA_SSH_KEY` | BYOVM commands need a default private key |
| Object-storage access key | `storage.aws_access_key_id` | `AWS_ACCESS_KEY_ID` | BYOVM, existing storage, or cross-project S3 workflows need explicit storage credentials |
| Object-storage secret key | `storage.aws_secret_access_key` | `AWS_SECRET_ACCESS_KEY` | BYOVM, existing storage, or cross-project S3 workflows need explicit storage credentials |
| Object-storage endpoint | `storage.endpoint_url` | `AWS_ENDPOINT_URL`, `NEBIUS_S3_ENDPOINT`, `NPA_STORAGE_ENDPOINT` | S3-compatible storage is not supplied by managed project config |
| Object-storage bucket | `storage.bucket` | `NPA_CHECKPOINT_BUCKET`, `NEBIUS_S3_BUCKET` | A workflow needs a default checkpoint or artifact bucket |

When `tokens.HF_TOKEN` is loaded, `npa` forwards it to remote services as both
`HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN`.

`NPA_STORAGE_ENDPOINT` is accepted as a convenience alias. For eu-north1
workbench clusters, use:

```bash
export NPA_STORAGE_ENDPOINT=storage.eu-north1.nebius.cloud
```

### 4c. Populate `~/.npa/credentials.yaml`

Use this complete template and delete keys you do not need yet:

```yaml
tokens:
  HF_TOKEN: <YOUR_HUGGING_FACE_TOKEN>

ngc:
  api_key: <YOUR_NGC_API_KEY>
  # org: <YOUR_NGC_ORG>
  # team: <YOUR_NGC_TEAM>

# Optional defaults for BYOVM commands only.
ssh:
  host: <BYOVM_HOST_OR_IP>
  user: ubuntu
  key_path: ~/.ssh/id_ed25519

# Optional shared object-storage credentials for BYOVM or existing storage.
storage:
  aws_access_key_id: <YOUR_S3_ACCESS_KEY_ID>
  aws_secret_access_key: <YOUR_S3_SECRET_ACCESS_KEY>
  endpoint_url: https://storage.<NEBIUS_REGION>.nebius.cloud
  bucket: s3://<YOUR_BUCKET>/<PREFIX>/
```

Omit keys you do not have yet. Do not leave placeholder token values in a file
you plan to use for model downloads.

Secure the file after editing:

```bash
chmod 600 ~/.npa/credentials.yaml
```

### 4d. Cross-project storage workflows

If you submit `npa` workloads from an orchestrator that reads resources from one
project and writes outputs to another, pass explicit project aliases for each
side of the S3 boundary:

```python
from npa import demo

demo.stage(
    source_project="project-a-where-source-artifacts-live",
    target_project="project-b-customer-bucket",
    target_bucket="s3://customer-bucket/demo-artifacts/",
)
```

Each project resolves credentials independently. If a scoped principal is
missing access on either side, `ScopedCredentialError` names the specific
project, operation, and bucket that failed.

For development workflows where host credentials are acceptable:

```bash
npa demo stage --source-project project-a --target-project project-b \
  --target-bucket s3://customer-bucket/demo-artifacts/ --allow-host-creds
```

## 5. First platform checks

These commands should not provision cloud resources:

```bash
npa --help
npa configure
```

Gate: both commands render local CLI output without requiring Kubernetes, S3,
NGC, or Hugging Face network access.

### 5a. Your first real result (offline)

You can produce a real eval result with no cloud, GPU, or credentials. The
`vlm-eval benchmark` command scores a shipped, labeled rollout set with the
offline `stub` backend:

```bash
npa workbench vlm-eval benchmark \
  --dataset npa/src/npa/workbench/vlm_eval/fixtures/sample_benchmark/benchmark.json \
  --output /tmp/vlm-eval-benchmark.json \
  --backend stub \
  --thresholds 0.5,0.8,0.9 \
  --rubrics default,strict \
  --models Qwen/Qwen2-VL-7B-Instruct \
  --format json
```

Gate: the report ranks configurations and reports `accuracy: 1.0` over four
labeled rollouts, and writes `/tmp/vlm-eval-benchmark.json`.

### 5b. The same eval, three coherent ways

Every Workbench capability is usable as a `npa` CLI command, a Python SDK call,
and a parameterizable SkyPilot YAML you can run with raw `sky`. The three stay
coherent; pick whichever fits your workflow.

**CLI** (shown above):

```bash
npa workbench vlm-eval benchmark --dataset <benchmark.json> --backend stub --format json
```

**Python SDK:**

```python
from npa.sdk.workbench import vlm_eval
from npa.workbench.vlm_eval import DEFAULT_MODEL, DEFAULT_SAMPLE_BENCHMARK_PATH

report = vlm_eval.benchmark(
    dataset=str(DEFAULT_SAMPLE_BENCHMARK_PATH),
    backend="stub",
    thresholds=[0.5, 0.8, 0.9],
    rubrics=["default", "strict"],
    models=[DEFAULT_MODEL],
)
print(report.best_config.metrics.accuracy)  # 1.0
```

**Standalone SkyPilot YAML (raw `sky`, BYO S3 endpoint + image).** Save this as
`vlm-eval-benchmark.sky.yaml` and run it with plain `sky launch` — no `npa` CLI
or SDK in the loop. Every value is a placeholder you override with `--env`:

```yaml
name: vlm-eval-benchmark
resources:
  cloud: kubernetes
  cpus: 4
  # Bring your own image. The default below is a generic CPU Python image;
  # point NPA_IMAGE at your registry, e.g. cr.<region>.nebius.cloud/<your-registry-id>/<image>:<tag>
  image_id: "docker:${NPA_IMAGE}"
envs:
  NPA_IMAGE: "python:3.11-slim"
  # Bring your own object storage. Leave these unset to read the in-repo fixture.
  BENCHMARK_URI: "s3://<your-bucket>/vlm-eval/benchmark.json"
  OUTPUT_URI: "s3://<your-bucket>/vlm-eval/benchmark-report.json"
  AWS_ENDPOINT_URL: "https://storage.<your-region>.nebius.cloud"
  VLM_BACKEND: "stub"
setup: |
  set -e
  pip install -e /opt/nebius-physical-ai/npa || pip install npa
run: |
  set -euo pipefail
  npa workbench vlm-eval benchmark \
    --dataset "${BENCHMARK_URI}" \
    --output "${OUTPUT_URI}" \
    --backend "${VLM_BACKEND}" \
    --format json
```

```bash
# Override any value at launch; nothing is hardcoded to a specific account.
sky launch -c vlm-eval vlm-eval-benchmark.sky.yaml \
  --env NPA_IMAGE=cr.<your-region>.nebius.cloud/<your-registry-id>/<image>:<tag> \
  --env BENCHMARK_URI=s3://<your-bucket>/vlm-eval/benchmark.json \
  --env AWS_ENDPOINT_URL=https://storage.<your-region>.nebius.cloud
```

For maintained, checked-in workflow YAMLs (including a self-hosted GPU VLM
variant), see `npa/workflows/workbench/skypilot/` and
[the workflows guide](workbench-yaml-guide.md).

## 6. Developing and testing npa

To work on `npa` itself, install the dev extra into your activated venv and use
the `make` targets from the repo root:

```bash
pip install -e "npa[dev]"   # pytest, pytest-mock, pytest-cov, pytest-timeout, ruff

make test         # fast default: full unit suite, no live/GPU/network
make test-smoke   # quickest: onboarding CLI smoke tests only
make lint         # ruff
make test-e2e     # opt-in: launches real Nebius infrastructure
```

The `make` targets call `python -m pytest`; pass `PYTHON=...` to target a
specific interpreter (for example `make test PYTHON=./.venv/bin/python`). Live,
GPU, and end-to-end tests are marked (`gpu`, `multi_gpu`, `e2e`, `byovm_live`,
`ngc_e2e`, ...) and are deselected from `make test`, so the default suite never
touches real infrastructure even if your shell has Nebius credentials exported.
See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full test layout and PR
conventions (branch → PR → squash, one approval, never self-approve).

## 7. Flagship GPU workload: NVIDIA Cosmos

Once the offline loop above works, the headline Workbench workload is **NVIDIA
Cosmos** — a world-foundation model for synthetic data and world generation.
Cosmos is the recommended first GPU workload because it runs across **multiple
NVIDIA GPU platforms** (for example `gpu-h100-sxm`, `gpu-h200-sxm`,
`gpu-b300-sxm`, `gpu-l40s`) selected with a single `--gpu-type` flag. It does
**not** require RT cores, so you are not locked to one GPU family the way
RT-core tools are.

This step needs Nebius credentials, a `HF_TOKEN` for the gated Cosmos weights
(Section 4), and GPU capacity. Set GPU routing with one flag and keep the same
command shape across platforms:

```bash
# Deploy a Cosmos serving endpoint on the GPU platform of your choice.
npa workbench cosmos -p <your-project-alias> -n cosmos deploy \
  --runtime serverless \
  --gpu-type <gpu-platform> \
  --gpu-preset <gpu-preset> \
  --wait

# Generate from a text prompt; output lands in your bucket.
npa workbench cosmos -p <your-project-alias> -n cosmos infer \
  --prompt "A robot arm stacks colored cubes on a table" \
  --output-path s3://<your-bucket>/cosmos/out/ \
  --output-format json

npa workbench cosmos -p <your-project-alias> -n cosmos teardown --yes
```

Artifact-bearing end-to-end validation (a real serverless GPU job that writes a
`checkpoint.json` to your bucket) is:

```bash
npa workbench cosmos train --runtime serverless --smoke --gpu-type <gpu-platform>
```

This same serverless job is available three coherent ways:

- **CLI:** the `npa workbench cosmos train --runtime serverless` command above.
- **SDK:** Cosmos serverless jobs are submitted programmatically with
  `npa.clients.serverless.ServerlessClient.create_job(...)` plus the
  `npa.serverless_common` env helpers (the `npa.sdk.workbench.cosmos` namespace
  itself currently exposes `check`/`fetch`). See the worked SDK example in
  [docs/sdk/cosmos-serverless.md](sdk/cosmos-serverless.md).
- **Raw `sky` (GPU-cluster alternative):** the checked-in, parameterizable
  SkyPilot YAMLs under `npa/workflows/workbench/skypilot/` (for example
  `cosmos3-text-to-image-inference.yaml`) run Cosmos on a GPU *cluster* with
  plain `sky launch`, using `--env`/`--gpu-type` overrides and a BYO `image_id`.
  This is a different runtime from Serverless AI Jobs (it provisions a cluster
  and needs network access to the Cosmos framework source + gated weights).

Because this launches a real, potentially long GPU job, run it from a durable
launcher (your job queue / SkyPilot-managed job) rather than an interactive
session you might close. See [the Cosmos guide](../.agents/skills/workbench/cosmos/SKILL.md)
and [the workflows guide](workbench-yaml-guide.md) for routing, backend
selection, and known limits. Isaac Lab is the simulation counterpart but is
RT-core-only (L40S / RTX Pro 6000); see its guide before choosing GPU type.

## 8. Where to next

- [Workbench Getting Started](workbench/getting-started.md): Kubernetes,
  SkyPilot, registry, S3, and first workload setup.
- [CLI and package overview](../npa/README.md): package-level command and
  development notes.
- [Repository overview](../README.md): project map and current workbench list.
- [Source sample config](../npa/src/npa/config/sample_config.yaml):
  machine-managed `~/.npa/config.yaml` shape for reference only.
- [Known onboarding and runtime gotchas](../FIXME.md): active follow-up list.

## 9. Troubleshooting

`npa: command not found`

Activate the virtualenv (or call its interpreter directly):

```bash
source .venv/bin/activate   # or: source ~/.venvs/npa/bin/activate
npa --help
```

`pytest` fails to collect with `ModuleNotFoundError: No module named 'fastapi'`

The test suite needs the dev tooling (which pulls in the server extra). Install
it into your venv:

```bash
pip install -e "npa[dev]"
make test
```

`aws s3 ls` fails with `Could not connect to the endpoint URL`

Nebius object storage is S3-compatible but is not AWS. Older `aws-cli` (v1)
ignores the `AWS_ENDPOINT_URL` environment variable, so it tries the AWS
endpoint and fails. Pass the endpoint explicitly, or use `aws-cli` v2:

```bash
aws s3 ls --endpoint-url https://storage.<your-region>.nebius.cloud
```

`npa` itself does not depend on this: it reads `storage.endpoint_url` from
`~/.npa/credentials.yaml` (or `AWS_ENDPOINT_URL`/`NPA_STORAGE_ENDPOINT`) and
passes it to the S3 client directly.

Jobs land on the wrong cluster, or `kubectl`/`sky` target the wrong place

Pin your Kubernetes context so submissions are unambiguous:

```bash
kubectl config get-contexts
kubectl config use-context <your-workbench-context>
```

`403`/`denied` when pushing or pulling a container image

Check that you are logged in to the registry. Nebius registries use a Docker
credential helper; confirm `~/.docker/config.json` references your registry
host and that `nebius iam get-access-token` succeeds.

Capacity, quota, or `Not enough resources` errors

These come from the cloud, not from `npa`. Retry in a few minutes, pick a
different GPU type/region, or request a quota increase in the Nebius console.
Run any command with `NPA_DEBUG=1` for a full traceback.

`source repo is not reachable` from `npa workbench cosmos check`

Cosmos checks the framework source repo with `git ls-remote`, injecting
`GITHUB_TOKEN` as an auth header when that variable is set. A stale or invalid
`GITHUB_TOKEN` makes even a public repo return `401`, so the check reports the
source as unreachable. Clear the bad token (the public clone works
anonymously) or export a valid one:

```bash
env -u GITHUB_TOKEN npa workbench cosmos check
```

SkyPilot pods do not inherit your shell's `GITHUB_TOKEN`, so the in-cluster
clone is unaffected.

`PermissionDenied` / `No permission` when submitting a serverless job

Serverless workloads (`cosmos train --runtime serverless`, `cosmos infer`, and
serverless `deploy`) create Nebius AI Jobs. If the principal behind your active
`nebius` profile is a service account that lacks the AI Jobs role, the submit is
rejected with `PermissionDenied: service iam ... appbox-...` even though
authentication, capacity lookup, subnet discovery, and S3 all succeed. This is
an authorization gap, not stale credentials — `nebius iam get-access-token`
still works. Grant that service account a role that permits creating and
managing AI Jobs on the project, or switch to a profile whose principal has it
(`nebius profile activate <profile>`), then retry the same command.

`credentials.yaml` is missing or tokens are not loading

Offline commands tolerate a missing credentials file, but token-dependent
commands will behave as if no token exists. Create the file under your home
directory and secure it:

```bash
mkdir -p ~/.npa
touch ~/.npa/credentials.yaml
chmod 600 ~/.npa/credentials.yaml
```

`credentials.yaml is readable by other users`

Workbench subcommands warn when group or world permissions are present:

```bash
chmod 600 ~/.npa/credentials.yaml
```

`Warning: HF_TOKEN not found in ~/.npa/credentials.yaml`

Add `tokens.HF_TOKEN` to `~/.npa/credentials.yaml` or export `HF_TOKEN` for the
current shell. Cosmos and GR00T deploy dry-runs fail fast without it unless you
pass `--skip-model-check`.

`Error: HF_TOKEN does not have access to <repo>` or `401/403 from Hugging Face`

Use a token from <https://huggingface.co/settings/tokens>, accept the gated
model's terms on Hugging Face, then retry. Environment variable `HF_TOKEN`
overrides the value in `credentials.yaml`.

`NGC API error` or `401 from NGC`

Use a current NGC key from <https://ngc.nvidia.com/setup/api-key>. Put it in
`ngc.api_key` in `credentials.yaml` or export `NGC_API_KEY` for the current
shell. If you use an organization or team-scoped key, also set `ngc.org` and
`ngc.team`, or export `NGC_ORG` and `NGC_TEAM`.

`nebius CLI not found on PATH`

Install the Nebius CLI and restart the shell:
<https://docs.nebius.com/cli/install>.

`Nebius auth failed`

Re-run `npa configure` in a terminal. If a profile exists but the wrong one is
active, switch with `nebius profile activate <profile>` and retry.

`terraform binary not found on PATH`

Install Terraform and verify `terraform version` before running managed deploys.
