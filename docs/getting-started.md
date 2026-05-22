# Getting Started

This guide takes a first-time user from a fresh clone to a validated BDD100K
pipeline invocation using only checked-in repo docs and CLI help.

For a deeper CLI walkthrough, see [quickstart.md](quickstart.md). For the
pipeline YAML contract, see [workbench-yaml-guide.md](workbench-yaml-guide.md).
For the demo narrative, see [demos/bdd100k-lancedb-demo.md](demos/bdd100k-lancedb-demo.md).

## Prerequisites

Install these on the operator machine:

- Python 3.10 or newer.
- Git, `python -m venv`, and `pip`.
- A Nebius AI Cloud account with billing enabled.
- Nebius CLI, configured with a profile that can create or access the target
  project: <https://docs.nebius.com/cli/install>.
- Terraform on `PATH` for managed VM deploys.
- `kubectl` for Kubernetes workbench services and cluster status.
- Docker for local container smoke runs, including local LanceDB validation.
- Optional but required for live SkyPilot submission: a SkyPilot 0.12.2 isolated
  venv. See [orchestration/skypilot-setup.md](orchestration/skypilot-setup.md).

Quick checks:

```bash
python3 --version
nebius version
nebius profile list
terraform version
kubectl version --client
docker version
```

## Install

Clone and install the package in an editable virtualenv:

```bash
git clone <REPO_URL> nebius-physical-ai
cd nebius-physical-ai

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e npa

npa --help
```

`npa --help` should not require Nebius, NGC, Hugging Face, or S3 credentials.

## Credentials

`npa` reads user-authored secrets from `~/.npa/credentials.yaml`. Deploy
commands write machine-managed metadata to `~/.npa/config.yaml`.

Create the directory and file:

```bash
mkdir -p ~/.npa
chmod 700 ~/.npa
touch ~/.npa/credentials.yaml
chmod 600 ~/.npa/credentials.yaml
```

Template:

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
  bucket: s3://<your-bucket>/<prefix>/
```

For eu-north1 clusters, use:

```bash
export NPA_STORAGE_ENDPOINT=storage.eu-north1.nebius.cloud
```

`NPA_STORAGE_ENDPOINT` is accepted as a convenience alias for the S3-compatible
endpoint and is forwarded to workbench services as `AWS_ENDPOINT_URL` and
`NEBIUS_S3_ENDPOINT`. If you still have the historical
`storage.uk-south1.nebius.cloud` endpoint in your shell or credentials, deploy
commands warn before continuing.

Nebius account authentication comes from the `nebius` CLI profile:

```bash
nebius profile create
nebius profile list
nebius iam get-access-token >/dev/null
```

For command examples and pipeline rendering, export the non-secret resource
identifiers for your project:

```bash
export NEBIUS_PROJECT_ID=<your-project-id>
export NEBIUS_TENANT_ID=<your-tenant-id>
export NPA_REGISTRY=cr.eu-north1.nebius.cloud/<your-registry-id>
export NPA_S3_BUCKET=<your-bucket>
```

`NPA_REGISTRY` is the full registry prefix used for workbench images.
`NPA_S3_BUCKET` is the bucket name only, without `s3://`.

## First Offline Commands

These should work before any cloud resources exist:

```bash
npa --help
npa configure
npa workbench --help
npa workbench fiftyone --help
npa workbench fiftyone list
```

On a fresh machine, `npa workbench fiftyone list` should say that no projects
are configured.

## First Cluster Profile

The Kubernetes workbench path expects a local NPA cluster profile. Create one
only when you are ready to provision Nebius resources:

```bash
npa cluster deploy \
  --name npa-workbench-eu-north1 \
  --project-id <YOUR_PROJECT_ID> \
  --region eu-north1 \
  --node-count 1 \
  --node-preset 8vcpu-32gb
```

For GPU pipeline stages, add an H100 node group or use an existing cluster that
already has an H100-capable node group. The exact quota, subnet, and node-group
policy depends on your Nebius project and should be confirmed by the operator.

## First Deploy

After the cluster profile exists, deploy the smallest browser-facing workbench
service:

```bash
npa workbench fiftyone deploy --public-ip
npa workbench fiftyone status
```

`--public-ip` exposes the FiftyOne app through a Kubernetes LoadBalancer
service. Omit it if your organization requires private-only access and use
`npa workbench fiftyone open` for local port-forwarding.

The managed VM path is also available and uses explicit project flags on first
deploy:

```bash
npa workbench fiftyone -p <PROJECT_ALIAS> -n quickstart-fiftyone deploy \
  --project-id <YOUR_PROJECT_ID> \
  --tenant-id <YOUR_TENANT_ID> \
  --region eu-north1
```

## First Pipeline Validation

Run the BDD100K pipeline wrapper against local mock endpoints first. This
validates YAML rendering and all HTTP request payloads without SkyPilot,
Kubernetes, GPUs, or object storage writes:

```bash
python npa/scripts/run_bdd100k_pipeline.py \
  --yaml npa/workflows/skypilot/bdd100k-pipeline.yaml \
  --synthetic 5000 \
  --mock-endpoints
```

For live submission, configure SkyPilot and verify it can see Nebius and
Kubernetes:

```bash
python -m venv /opt/npa/skypilot
/opt/npa/skypilot/bin/pip install 'skypilot[nebius,kubernetes]==0.12.2'
export NPA_SKYPILOT_BIN=/opt/npa/skypilot/bin/sky
"$NPA_SKYPILOT_BIN" check nebius kubernetes
```

Then submit the synthetic pipeline:

```bash
python npa/scripts/run_bdd100k_pipeline.py \
  --yaml npa/workflows/skypilot/bdd100k-pipeline.yaml \
  --synthetic 5000
```

Live submission requires reachable in-cluster LanceDB and detection-training
services, S3 credentials that can read and write the configured bucket, and a
cluster with enough CPU and H100 GPU capacity.

## Next Docs

- [quickstart.md](quickstart.md): full `npa` CLI setup walkthrough.
- [workbench-yaml-guide.md](workbench-yaml-guide.md): pipeline YAML structure.
- [cookbooks/bdd100k-pipeline.md](cookbooks/bdd100k-pipeline.md): BDD100K runbook.
- [demos/bdd100k-lancedb-demo.md](demos/bdd100k-lancedb-demo.md): demo reproduction notes.
- [orchestration/skypilot-setup.md](orchestration/skypilot-setup.md): SkyPilot venv setup.
