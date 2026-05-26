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
- macOS or Linux. Windows is not currently tested.
- A Nebius AI Cloud account with billing enabled. Start with the Nebius signup
  guide: <https://docs.nebius.com/signup-billing/sign-up>.
- The Nebius AI Cloud CLI. Install and configure it:
  <https://docs.nebius.com/cli/install> and
  <https://docs.nebius.com/cli/configure>.
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

## 3. Install npa

Clone the repository and install the Python package from the `npa/` package
directory:

```bash
git clone <REPO_URL> nebius-physical-ai
cd nebius-physical-ai

python3 -m venv npa/.venv
npa/.venv/bin/python -m pip install --upgrade pip
npa/.venv/bin/python -m pip install -e npa
export PATH="$PWD/npa/.venv/bin:$PATH"
```

Verify the install:

```bash
npa --version
npa --help
```

Gate: `npa --version` prints `npa <version>`, and `npa --help` prints the
command tree without requiring Nebius, Hugging Face, NGC, Kubernetes, or S3
credentials.

Optional extras are available when you need them:

```bash
npa/.venv/bin/python -m pip install -e "npa[server]"
npa/.venv/bin/python -m pip install -e "npa[adapter]"
npa/.venv/bin/python -m pip install -e "npa[genesis]"
npa/.venv/bin/python -m pip install -e "npa[groot]"
```

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

Configure and verify the Nebius CLI:

```bash
nebius profile create
nebius profile list
nebius iam get-access-token >/dev/null
```

Gate: `nebius iam get-access-token` exits successfully.

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

## 6. Where to next

- [Workbench Getting Started](workbench/getting-started.md): Kubernetes,
  SkyPilot, registry, S3, and first workload setup.
- [CLI and package overview](../npa/README.md): package-level command and
  development notes.
- [Repository overview](../README.md): project map and current workbench list.
- [Source sample config](../npa/src/npa/config/sample_config.yaml):
  machine-managed `~/.npa/config.yaml` shape for reference only.
- [Known onboarding and runtime gotchas](../FIXME.md): active follow-up list.

## 7. Troubleshooting

`npa: command not found`

Activate the virtualenv or add it to `PATH`:

```bash
export PATH="$PWD/npa/.venv/bin:$PATH"
npa --help
```

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

Run `nebius profile create`, verify `nebius profile list`, and check that
`nebius iam get-access-token` returns successfully.

`terraform binary not found on PATH`

Install Terraform and verify `terraform version` before running managed deploys.
