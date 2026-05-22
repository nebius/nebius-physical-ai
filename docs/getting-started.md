# Getting Started

This guide takes a first-time partner from a fresh clone to a machine that can
run the Isaac Lab BYOF cookbook and write the first checkpoint to Nebius S3.

For a deeper CLI walkthrough, see [quickstart.md](quickstart.md). For the
BYOF training path, see
[cookbooks/byof-isaac-lab/README.md](cookbooks/byof-isaac-lab/README.md).
For the SkyPilot runtime details, see
[orchestration/skypilot-setup.md](orchestration/skypilot-setup.md).

## Day Zero Preconditions

Operator-required prerequisites:

- A Nebius account with access to the target project and tenant.
- A managed Kubernetes context for the target workbench cluster, normally
  `npa-workbench-eu-north1`.
- Provisioned RT-core GPU capacity for Isaac Lab, normally L40S in
  `eu-north1`. H100 and H200 do not satisfy Isaac Lab rendering requirements.
- A current registry pull secret in the Kubernetes namespace that SkyPilot will
  use, normally `default`.
- An S3 bucket in `eu-north1` with read and write access for your AWS profile.
- A Nebius container registry namespace that can push and pull Workbench images.

Partner-specific values to collect before starting:

```bash
<your-project-id>
<your-tenant-id>
<your-bucket>
<your-registry-id>
```

Constants for the primary workbench environment:

```bash
eu-north1
https://storage.eu-north1.nebius.cloud
cr.eu-north1.nebius.cloud
```

## Install Local Tools

Install these on the operator machine:

- Python 3.10 or newer.
- Git, `python -m venv`, and `pip`.
- Nebius CLI: <https://docs.nebius.com/cli/install>.
- AWS CLI v2.
- Docker with registry login access.
- `kubectl`.
- Terraform on `PATH` for managed VM deploys.

Verify the tools:

```bash
python3 --version
git --version
nebius version
aws --version
docker --version
kubectl version --client
terraform version
```

Gate: every command prints a version. `kubectl version --client` only checks the
local client; cluster authentication is verified later.

## Install NPA

Clone and install the package in the repo-local virtualenv:

```bash
git clone <REPO_URL> nebius-physical-ai
cd nebius-physical-ai

python3 -m venv npa/.venv
npa/.venv/bin/python -m pip install --upgrade pip
npa/.venv/bin/python -m pip install -e npa
export PATH="$PWD/npa/.venv/bin:$PATH"
```

Verify the CLI:

```bash
npa --help
npa configure
```

Gate: `npa --help` prints the command list and does not require Nebius, NGC,
Hugging Face, Kubernetes, or S3 credentials. The current CLI does not expose a
`npa --version` flag.

## Configure Nebius Credentials

`npa` reads user-authored secrets from `~/.npa/credentials.yaml`. Deploy
commands write machine-managed metadata to `~/.npa/config.yaml`.

Create the credentials file:

```bash
mkdir -p ~/.npa
chmod 700 ~/.npa
touch ~/.npa/credentials.yaml
chmod 600 ~/.npa/credentials.yaml
```

Use this template and substitute your values:

```yaml
tokens:
  HF_TOKEN: <your-hugging-face-token>

ngc:
  api_key: <your-ngc-api-key>
  # org: <your-ngc-org>
  # team: <your-ngc-team>

ssh:
  host: <byovm-host-or-ip>
  user: ubuntu
  key_path: ~/.ssh/id_ed25519

storage:
  aws_access_key_id: <your-s3-access-key-id>
  aws_secret_access_key: <your-s3-secret-access-key>
  endpoint_url: https://storage.eu-north1.nebius.cloud
  bucket: s3://<your-bucket>/
```

Authenticate the Nebius CLI:

```bash
nebius profile create
nebius profile list
nebius iam get-access-token >/dev/null
```

Gate: `nebius iam get-access-token` exits successfully.

## Configure S3 Access

Create an AWS profile for Nebius Object Storage:

```bash
aws configure --profile nebius-eu-north1
```

When prompted, enter:

```text
AWS Access Key ID: <your-s3-access-key-id>
AWS Secret Access Key: <your-s3-secret-access-key>
Default region name: eu-north1
Default output format: json
```

Export the profile and endpoint:

```bash
export AWS_PROFILE=nebius-eu-north1
export AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud
export NPA_STORAGE_ENDPOINT=storage.eu-north1.nebius.cloud
export NPA_S3_BUCKET=<your-bucket>
```

Verify bucket access:

```bash
aws s3 ls "s3://${NPA_S3_BUCKET}/" --endpoint-url "${AWS_ENDPOINT_URL}"
```

Gate: the command lists the bucket or exits successfully with an empty listing.
`NoSuchBucket` usually means the bucket name, AWS profile, or region is wrong.
`AccessDenied` means the profile lacks bucket access.

## Configure Workbench Identifiers

Export the non-secret resource identifiers for commands and examples:

```bash
export NEBIUS_PROJECT_ID=<your-project-id>
export NEBIUS_TENANT_ID=<your-tenant-id>
export NPA_REGISTRY_ID=<your-registry-id>
export NPA_REGISTRY=cr.eu-north1.nebius.cloud/${NPA_REGISTRY_ID}
```

`NPA_REGISTRY_ID` is the registry namespace only. `NPA_REGISTRY` is the full
registry prefix used by current image-resolution code.

Verify Docker registry access:

```bash
docker login cr.eu-north1.nebius.cloud
```

Gate: Docker stores a login for `cr.eu-north1.nebius.cloud`. If login fails,
refresh your registry credentials before building BYOF images.

## Verify Kubernetes Access

Select the Nebius managed Kubernetes context provided by your operator:

```bash
kubectl config get-contexts
kubectl config use-context <your-nebius-mk8s-context>
kubectl config current-context
```

Verify the account can create SkyPilot pods in `default`:

```bash
kubectl auth can-i create pods -n default
kubectl get nodes
kubectl get secret npa-nebius-registry -n default
```

Gate: `kubectl auth can-i` prints `yes`, `kubectl get nodes` lists the cluster
nodes, and the registry secret exists. If SkyPilot later reports HTTP 403 as an
anonymous user, the kube context is not authenticated for the cluster.

## Bootstrap SkyPilot

Install the pinned isolated SkyPilot runtime:

```bash
npa skypilot bootstrap
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"
npa skypilot status
"${NPA_SKYPILOT_BIN}" --version
```

Gate: the version is `0.12.2`, and `npa skypilot status` reports the isolated
venv path under `~/.npa/skypilot-venv` unless you passed `--path`.

Verify SkyPilot can see Kubernetes:

```bash
"${NPA_SKYPILOT_BIN}" check
```

Gate: the Kubernetes check succeeds. A 403 anonymous-user error means the local
kube context is missing or expired, not that the BYOF workflow is wrong.

## First Offline Commands

These should work before any GPU job is submitted:

```bash
npa workbench --help
npa workbench isaac-lab --help
npa workbench fiftyone list
```

Gate: command help renders, and a fresh machine may report that no workbench
projects are configured.

## First BYOF Run

After the gates above pass, continue with
[cookbooks/byof-isaac-lab/README.md](cookbooks/byof-isaac-lab/README.md) and
follow its sections in order.

The first live checkpoint path has this structure:

```bash
s3://${NPA_S3_BUCKET}/checkpoints/isaac-lab-byof/<run-id>/npa_isaac_lab_checkpoint.pt
```

Use the cookbook's verification commands to list the run prefix and fetch the
manifest from S3.

## Common Failures

| Symptom | Diagnosis | Fix |
|---|---|---|
| `sky check` reports HTTP 403 for an anonymous user | The active kube context is not authenticated to the Nebius MK8s cluster. | Run `kubectl config use-context <your-nebius-mk8s-context>` and refresh cluster credentials. |
| S3 upload logs contain literal `${AWS_ENDPOINT_URL}` | SkyPilot 0.12.2 does not interpolate variables inside YAML `envs` blocks at submission time. | Use `npa/scripts/run_isaac_lab_rl.py`, which materializes endpoint values before submission, or substitute `https://storage.eu-north1.nebius.cloud` in the YAML. |
| `NoSuchBucket` from AWS CLI or a workflow upload | Wrong `NPA_S3_BUCKET`, AWS profile, or region. | Re-export `AWS_PROFILE=nebius-eu-north1`, `AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud`, and the bucket name without `s3://`. |
| Image pull returns `401 Unauthorized` | The Kubernetes registry pull secret expired. | Ask the operator to refresh `npa-nebius-registry` in the SkyPilot namespace. |
| L40S scheduling backoff | The cluster has no available L40S capacity or the preset is too small for the CPU request. | Ask the operator to provision an L40S node group with sufficient CPU, or use a documented RT-core alternative. |

## Next Docs

- [cookbooks/byof-isaac-lab/README.md](cookbooks/byof-isaac-lab/README.md):
  first Isaac Lab BYOF checkpoint.
- [orchestration/skypilot-setup.md](orchestration/skypilot-setup.md): isolated
  SkyPilot runtime details.
- [quickstart.md](quickstart.md): broader `npa` CLI walkthrough.
- [workbench-yaml-guide.md](workbench-yaml-guide.md): pipeline YAML structure.
