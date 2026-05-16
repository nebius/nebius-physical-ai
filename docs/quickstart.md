# npa Quickstart

This guide takes a new developer from a fresh clone to a working `npa workbench`
command. It is written for macOS and Linux. Windows is not currently tested.

## 1. What is npa?

`npa` is the Nebius Physical AI Workbench CLI. It orchestrates physical AI
workloads across Nebius infrastructure, including Cosmos, Isaac Lab, GR00T,
FiftyOne, LeRobot, and Genesis. The CLI can provision Nebius workbench VMs,
install tool runtimes, pass shared credentials into remote services, and run
training, evaluation, serving, inference, visualization, and dataset commands.
For a broader architecture map, see the repository README and `npa/README.md`.

## 2. Prerequisites

- Python 3.10 or newer. The package metadata requires `>=3.10`.
- Git, `python -m venv`, and `pip`.
- macOS or Linux. Windows is not currently tested.
- A Nebius AI Cloud account with billing enabled. Start with the Nebius signup
  guide: <https://docs.nebius.com/signup-billing/sign-up>.
- The Nebius AI Cloud CLI for managed deploy commands. Install and configure it:
  <https://docs.nebius.com/cli/install> and
  <https://docs.nebius.com/cli/configure>.
- Terraform on `PATH` for managed `deploy` and `--destroy` commands.
- An SSH public key. The bundled Terraform defaults to
  `~/.ssh/id_ed25519.pub`; pass `--tf-var ssh_public_key_path=<path>.pub` if
  you use a different key.
- Optional: an NVIDIA NGC API key for GR00T NGC model paths:
  <https://ngc.nvidia.com/setup/api-key>.
- Optional: a Hugging Face token for gated Cosmos/GR00T models, LeRobot
  datasets, and select model weights:
  <https://huggingface.co/settings/tokens>. Use a read-only token unless a
  workflow explicitly needs write access.

Quick checks:

```bash
python3 --version
nebius version
nebius profile list
terraform version
```

## 3. Install npa

Clone the repository and install the Python package from the `npa/` package
directory:

```bash
git clone <REPO_URL> <your-clone-path>
cd <your-clone-path>

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e npa
```

Verify the install:

```bash
npa --help
```

`npa --help` should print the command tree without requiring Nebius, NGC, or
Hugging Face credentials.

Optional extras are available when you need them:

```bash
python -m pip install -e "npa[server]"
python -m pip install -e "npa[adapter]"
python -m pip install -e "npa[genesis]"
python -m pip install -e "npa[groot]"
```

## 4. Configure credentials

`npa` uses two files under `~/.npa/`:

- `~/.npa/credentials.yaml` is user-authored. Put user-level secrets here:
  Hugging Face tokens, NGC keys, optional BYOVM SSH defaults, and optional S3
  credentials for existing storage.
- `~/.npa/config.yaml` is machine-managed. `deploy` writes project,
  workbench, endpoint, SSH, storage, and Terraform state metadata here. Do not
  put user tokens in this file.

Environment variables override values from `credentials.yaml`. Current source
does not read an `NPA_CREDENTIALS_PATH`; it resolves credentials from
`Path.home() / ".npa" / "credentials.yaml"`.

Create the credentials directory and file:

```bash
mkdir -p ~/.npa
chmod 700 ~/.npa
touch ~/.npa/credentials.yaml
chmod 600 ~/.npa/credentials.yaml
```

### 4a. Nebius credentials

Nebius account authentication is handled by the `nebius` CLI profile, not by a
long-lived `NEBIUS_TOKEN` in `~/.npa/credentials.yaml`. During managed deploys,
`npa` calls `nebius iam get-access-token`, creates or reuses a service account,
creates S3-compatible access keys, creates or reuses an object-storage bucket,
and saves non-user-secret workbench metadata in `~/.npa/config.yaml`.

Configure and verify the Nebius CLI:

```bash
nebius profile create
nebius profile list
nebius iam get-access-token >/dev/null
```

You need these values for the first deploy:

- `<YOUR_PROJECT_ID>`: copy it from the Nebius console project selector or list
  projects with the Nebius CLI.
- `<YOUR_TENANT_ID>`: copy it from the Nebius console tenant selector. The
  Nebius docs also show CLI options:
  <https://docs.nebius.com/iam/get-tenants>.
- `<NEBIUS_REGION>`: the region where the project exists, for example
  `eu-north1`.
- `<PROJECT_ALIAS>`: a local alias you choose for `npa`, for example
  `quickstart`.

After a successful deploy, `~/.npa/config.yaml` will contain a structure like:

```yaml
default_project: <PROJECT_ALIAS>
default_workbench: quickstart-fiftyone
projects:
  <PROJECT_ALIAS>:
    project_id: <YOUR_PROJECT_ID>
    tenant_id: <YOUR_TENANT_ID>
    region: <NEBIUS_REGION>
    terraform_state:
      bucket: <STATE_BUCKET>
      endpoint: https://storage.<NEBIUS_REGION>.nebius.cloud
      access_key: <REDACTED>
      secret_key: <REDACTED>
    workbenches:
      quickstart-fiftyone:
        endpoint: http://<VM_IP>:5151
        ssh:
          host: <VM_IP>
          user: ubuntu
          key_path: ~/.ssh/id_ed25519
```

### 4b. NGC API key for GR00T

NGC credentials are used by GR00T when you download or serve an NGC-backed model
reference. `npa` accepts NGC credentials from either `~/.npa/credentials.yaml`
or environment variables. The credentials file is preferred for repeatable
workbench commands.

Credentials file:

```yaml
ngc:
  api_key: <YOUR_NGC_API_KEY>
  # org: <YOUR_NGC_ORG>
  # team: <YOUR_NGC_TEAM>
```

One-off environment variables:

```bash
export NGC_API_KEY=<YOUR_NGC_API_KEY>
export NGC_ORG=<YOUR_NGC_ORG>      # optional
export NGC_TEAM=<YOUR_NGC_TEAM>    # optional
```

Legacy layouts also work:

```yaml
tokens:
  NGC_API_KEY: <YOUR_NGC_API_KEY>
  NGC_ORG: <YOUR_NGC_ORG>
  NGC_TEAM: <YOUR_NGC_TEAM>
```

### 4c. Hugging Face token

Hugging Face credentials are used for gated Cosmos and GR00T models, LeRobot
datasets, and select model weights. Use a read-only token unless your workflow
explicitly pushes datasets or checkpoints.

Credentials file:

```yaml
tokens:
  HF_TOKEN: <YOUR_HUGGING_FACE_TOKEN>
```

One-off environment variable:

```bash
export HF_TOKEN=<YOUR_HUGGING_FACE_TOKEN>
```

When `HF_TOKEN` is loaded, `npa` forwards it to remote workbenches as both
`HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN`.

Combined `~/.npa/credentials.yaml` example:

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

### 4d. Cross-project workflows

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

## 5. First commands: offline, no infra

These commands should not provision cloud resources:

```bash
npa --help
npa configure
npa workbench --help
npa workbench fiftyone --help
npa workbench fiftyone list
```

On a fresh machine, `npa workbench fiftyone list` should print:

```text
No projects configured. Run 'npa workbench fiftyone deploy' to create one.
```

After you know your Nebius IDs, you can also dry-run the first deploy without
creating resources:

```bash
npa workbench fiftyone -p <PROJECT_ALIAS> -n quickstart-fiftyone deploy \
  --project-id <YOUR_PROJECT_ID> \
  --tenant-id <YOUR_TENANT_ID> \
  --region <NEBIUS_REGION> \
  --dry-run
```

## 6. First real command: deploy FiftyOne

FiftyOne is the smallest first workbench because the default deploy is CPU-only:
`cpu-d3` with `4vcpu-16gb`. Nebius currently lists CPU-only AMD EPYC Genoa
instances from `$0.10/hour`, plus storage charges. Confirm the exact current
price in the Nebius console before leaving resources running:
<https://nebius.com/prices>.

Deploy:

```bash
npa workbench fiftyone -p <PROJECT_ALIAS> -n quickstart-fiftyone deploy \
  --project-id <YOUR_PROJECT_ID> \
  --tenant-id <YOUR_TENANT_ID> \
  --region <NEBIUS_REGION>
```

Check the workbench:

```bash
npa workbench fiftyone -p <PROJECT_ALIAS> -n quickstart-fiftyone status
npa workbench fiftyone -p <PROJECT_ALIAS> -n quickstart-fiftyone launch
```

Tear it down when you are finished:

```bash
npa workbench fiftyone -p <PROJECT_ALIAS> -n quickstart-fiftyone deploy --destroy
```

Use explicit `-p` and `-n` on status, launch, serve, infer, and destroy
commands. Current known issues include confusing defaults when these flags are
omitted.

## 7. Viewing Rerun recordings in the browser

If you have generated `.rrd` files with `npa convert lerobot-to-rrd` and want
to share them without requiring teammates to install Rerun locally:

```bash
# Upload your .rrd to Nebius S3 and get a browser-viewable URL
npa rerun host /path/to/your-recording.rrd

# For team-shared review workflows with persistent links up to 7 days
npa rerun share /path/to/your-recording.rrd \
  --label "weekly-failure-review" \
  --workspace "team-perception"
```

The URL opens Rerun's hosted web viewer at `app.rerun.io` loaded with your
recording. Teammates can scrub the timeline, orbit the 3D view, and inspect
data without a local Rerun install.

## 8. Where to next

- [Repository overview](../README.md)
- [CLI and package overview](../npa/README.md)
- [Source sample config](../npa/src/npa/config/sample_config.yaml)
- [Known onboarding and runtime gotchas](../FIXME.md)
- [CLI source map](../npa/src/npa/cli/)

## 9. Troubleshooting

`npa: command not found`

Activate the virtualenv or reinstall the package:

```bash
source .venv/bin/activate
python -m pip install -e npa
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

Add `tokens.HF_TOKEN` to `~/.npa/credentials.yaml` or export `HF_TOKEN`. Cosmos
and GR00T deploy dry-runs fail fast without it unless you pass
`--skip-model-check`.

`Error: HF_TOKEN does not have access to <repo>` or `401/403 from Hugging Face`

Use a token from <https://huggingface.co/settings/tokens>, accept the gated
model's terms on Hugging Face, then retry. Environment variable `HF_TOKEN`
overrides the value in `credentials.yaml`.

`NGC API error` or `401 from NGC`

Use a current NGC key from <https://ngc.nvidia.com/setup/api-key>. Put it in
`ngc.api_key` in `credentials.yaml` or export `NGC_API_KEY`. If you use an
organization or team-scoped key, also set `NGC_ORG` and `NGC_TEAM`.

`Project '<alias>' not found`

The local project alias passed with `-p` is not in `~/.npa/config.yaml`. Use the
same `<PROJECT_ALIAS>` you used for deploy, or run the first deploy with
`--project-id`, `--tenant-id`, and `--region` so `npa` can write the project
entry.

`First deploy requires --project-id, --tenant-id, and --region`

The alias has not been bootstrapped yet. Pass all three values on the first
managed deploy. Later commands can reuse the saved values in `~/.npa/config.yaml`.

`nebius CLI not found on PATH`

Install the Nebius CLI and restart the shell:
<https://docs.nebius.com/cli/install>.

`Nebius auth failed`

Run `nebius profile create`, verify `nebius profile list`, and check that
`nebius iam get-access-token` returns successfully.

`terraform binary not found on PATH`

Install Terraform and verify `terraform version` before running managed deploys.

`Out of GPU quota` or capacity errors

Use a smaller preset, choose another available region or platform, or request a
quota increase through Nebius support. For quickstart, prefer the default
CPU-only FiftyOne deploy before trying GPU workbenches.
