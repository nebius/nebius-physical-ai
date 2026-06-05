# Workbench Getting Started

> Prerequisites: complete docs/quickstart.md first
> ([open quickstart](../quickstart.md)).

This guide assumes `npa` is already installed, the Nebius CLI is authenticated,
and user-level credentials are already configured in
`~/.npa/credentials.yaml`. It takes a first-time partner from platform setup to a
workbench-capable machine that can run either the H100 sim-to-real quickstart or
the Isaac Lab BYOF cookbook and write the first checkpoint to S3.

For the canonical credential setup, see [quickstart.md](../quickstart.md). For
the H100 sim-to-real proof path, see
[sim-to-real-quickstart.md](sim-to-real-quickstart.md). For the BYOF training
path, see
[cookbooks/byof-isaac-lab/README.md](cookbooks/byof-isaac-lab/README.md).
For the SkyPilot runtime details, see
[orchestration/skypilot-setup.md](../orchestration/skypilot-setup.md).

## Day Zero Preconditions

Operator-required prerequisites:

- The platform quickstart is complete. Do not create another NPA credential
  file for workbench setup.
- A Nebius account with access to the target project and tenant.
- A managed Kubernetes context for the target workbench cluster, normally
  `npa-workbench-eu-north1`.
- H100 capacity for the headless sim-to-real quickstart.
- Provisioned RT-core GPU capacity for Isaac Lab, normally L40S in
  `eu-north1`. H100 and H200 do not satisfy Isaac Lab rendering requirements.
- A current registry pull secret in the Kubernetes namespace that SkyPilot will
  use, normally `default`.
- An S3 bucket in `eu-north1`. The access keys for that bucket should already be
  in `~/.npa/credentials.yaml` under `storage.aws_access_key_id`,
  `storage.aws_secret_access_key`, `storage.endpoint_url`, and `storage.bucket`
  if the workflow needs explicit storage credentials.
- A Nebius container registry namespace that can push and pull workbench images.

Partner-specific values to collect before starting:

```bash
<your-project-id>
<your-tenant-id>
<your-bucket>
<your-registry-id>
<your-nebius-mk8s-context>
```

Constants for the primary workbench environment:

```bash
eu-north1
https://storage.eu-north1.nebius.cloud
cr.eu-north1.nebius.cloud
```

## Install Workbench Tools

The quickstart already covers Python, Git, Terraform, `npa`, and the Nebius CLI.
Install these additional tools on the operator machine:

- AWS CLI v2 for direct S3 verification.
- Docker with registry login access.
- `kubectl`.

Verify the tools and platform setup:

```bash
npa --version
nebius iam get-access-token >/dev/null
aws --version
docker --version
kubectl version --client
terraform version
```

Gate: every command exits successfully. `kubectl version --client` only checks
the local client; cluster authentication is verified later.

## Confirm Platform Credentials

Do not recreate `~/.npa/credentials.yaml` here. The file and its permissions
come from the quickstart.

```bash
test -r ~/.npa/credentials.yaml
stat -f "%Sp %N" ~/.npa/credentials.yaml 2>/dev/null || \
  stat -c "%A %n" ~/.npa/credentials.yaml
```

Gate: the file exists and is not group- or world-readable. If it is too open,
run:

```bash
chmod 600 ~/.npa/credentials.yaml
```

For BYOF or other S3-backed workflows, confirm the quickstart credential file
contains these storage keys:

```yaml
storage:
  aws_access_key_id: <your-s3-access-key-id>
  aws_secret_access_key: <your-s3-secret-access-key>
  endpoint_url: https://storage.eu-north1.nebius.cloud
  bucket: s3://<your-bucket>/
```

## Configure Workbench Shell Values

Export the non-secret resource identifiers used by commands and examples:

```bash
export NEBIUS_PROJECT_ID=<your-project-id>
export NEBIUS_TENANT_ID=<your-tenant-id>
export NPA_S3_BUCKET=<your-bucket>
export NPA_REGISTRY_ID=<your-registry-id>
export NPA_REGISTRY=cr.eu-north1.nebius.cloud/${NPA_REGISTRY_ID}
export AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud
export NPA_STORAGE_ENDPOINT=storage.eu-north1.nebius.cloud
```

For local `aws s3` verification only, expose the same storage credentials that
are already stored in `~/.npa/credentials.yaml`:

```bash
export AWS_ACCESS_KEY_ID=<YOUR_S3_ACCESS_KEY_ID_FROM_CREDENTIALS_YAML>
export AWS_SECRET_ACCESS_KEY=<YOUR_S3_SECRET_ACCESS_KEY_FROM_CREDENTIALS_YAML>
```

These exports are not an alternate NPA credential store; they are shell values
for tools that do not read `~/.npa/credentials.yaml`.

Verify bucket access:

```bash
aws s3 ls "s3://${NPA_S3_BUCKET}/" --endpoint-url "${AWS_ENDPOINT_URL}"
```

Gate: the command lists the bucket or exits successfully with an empty listing.
`NoSuchBucket` usually means the bucket name, endpoint, or region is wrong.
`AccessDenied` means the access key lacks bucket access.

## Verify Docker Registry Access

`NPA_REGISTRY_ID` is the registry namespace only. The active image-resolution
paths use `NPA_REGISTRY` as the full registry prefix, or
`projects.<alias>.container_registry` in NPA config. Build and push commands
should therefore tag images as `${NPA_REGISTRY}/<image>:<tag>`, not as a value
derived from `NPA_REGISTRY_ID` alone.

Verify Docker registry access:

```bash
docker login cr.eu-north1.nebius.cloud
```

Gate: Docker stores a login for `cr.eu-north1.nebius.cloud`. If login fails,
refresh your registry credentials before building BYOF images.

## Verify Kubernetes Access

> Required only if using managed Kubernetes compute.
> Skip for serverless or VM-based workbench runs.

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

> Required only if using managed Kubernetes compute.
> Skip for serverless or VM-based workbench runs.

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

## First Offline Workbench Commands

These should work before any GPU job is submitted:

```bash
npa workbench --help
npa workbench isaac-lab --help
npa workbench fiftyone list
```

Gate: command help renders, and a fresh machine may report that no workbench
projects are configured.

## First Sim-To-Real Run

After the gates above pass, run the H100 quickstart:

```bash
npa/.venv/bin/python npa/scripts/run_sim_to_real_quickstart.py
```

The command prints the run ID, wall-clock time, task-success metric, checkpoint
URI, report URI, Rerun URI, and `cluster_absent=True` after teardown. The first
proof checkpoint path has this structure:

```bash
s3://${NPA_S3_BUCKET}/sim-to-real/<run-id>/checkpoints/policy/
```

Continue with [sim-to-real-quickstart.md](sim-to-real-quickstart.md) for the
exact output format and override options.

## First BYOF Run

For Isaac Lab, continue with
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
| `NoSuchBucket` from AWS CLI or a workflow upload | Wrong bucket name, endpoint, or region. | Confirm `storage.bucket` and `storage.endpoint_url` in `~/.npa/credentials.yaml`, then re-export `NPA_S3_BUCKET` without `s3://` and `AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud`. |
| `AccessDenied` from AWS CLI or a workflow upload | The access key does not have read/write access to the bucket. | Confirm `storage.aws_access_key_id` and `storage.aws_secret_access_key` in `~/.npa/credentials.yaml` are for the target bucket. |
| Image pull returns `401 Unauthorized` | The Kubernetes registry pull secret expired. | Ask the operator to refresh `npa-nebius-registry` in the SkyPilot namespace. |
| L40S scheduling backoff | The cluster has no available L40S capacity or the preset is too small for the CPU request. | Ask the operator to provision an L40S node group with sufficient CPU, or use a documented RT-core alternative. |

## Next Docs

- [cookbooks/byof-isaac-lab/README.md](cookbooks/byof-isaac-lab/README.md):
  first Isaac Lab BYOF checkpoint.
- [sim-to-real-quickstart.md](sim-to-real-quickstart.md): first H100
  sim-to-real checkpoint and eval metric.
- [orchestration/skypilot-setup.md](../orchestration/skypilot-setup.md):
  isolated SkyPilot runtime details.
- [quickstart.md](../quickstart.md): platform install and canonical credential
  setup.
- [workbench-yaml-guide.md](../workbench-yaml-guide.md): pipeline YAML
  structure.
