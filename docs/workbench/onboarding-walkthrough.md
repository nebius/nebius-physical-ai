# Onboarding Walkthrough

This walkthrough uses generic project, registry, and storage placeholders. Keep
real project IDs, tenant IDs, registry IDs, bucket names, and credentials in
local config only.

## 1. Configure Runtime Settings

Interactive:

```bash
npa configure
```

Non-interactive:

```bash
npa configure --non-interactive \
  --project default \
  --project-id <project-id> \
  --tenant-id <tenant-id> \
  --region eu-north1 \
  --registry-id <registry-id> \
  --s3-endpoint https://storage.eu-north1.nebius.cloud \
  --s3-bucket s3://<bucket>/checkpoints/ \
  --aws-access-key-id <aws-access-key-id> \
  --aws-secret-access-key <aws-secret-access-key>
```

`~/.npa/config.yaml` stores non-secret runtime settings. Project-scoped S3
access keys are written to `~/.npa/credentials.yaml`.

## 2. Provision If Absent

CLI:

```bash
npa provision-if-absent \
  --project default \
  --cluster-name npa-cluster \
  --terraform-dir deploy/cluster
```

SDK:

```python
from npa.sdk.provisioning import provision_if_absent

result = provision_if_absent(
    project="default",
    cluster_name="npa-cluster",
)
print(result.to_dict())
```

Standalone script:

```bash
export NPA_PROJECT_ID=<project-id>
export NPA_TENANT_ID=<tenant-id>
export NPA_REGION=eu-north1
export NPA_REGISTRY=cr.eu-north1.nebius.cloud/<registry-id>
export NPA_REGISTRY_ID=<registry-id>
export NPA_S3_BUCKET=s3://<bucket>/checkpoints/
export NPA_STORAGE_ENDPOINT=https://storage.eu-north1.nebius.cloud
export AWS_ACCESS_KEY_ID=<aws-access-key-id>
export AWS_SECRET_ACCESS_KEY=<aws-secret-access-key>
scripts/provision-if-absent.sh all
```

Standalone YAML:

```bash
scripts/provision-if-absent-from-yaml.py --config infra/bootstrap/provision-if-absent.yaml all
```

The hook is additive-only: it can ensure an S3 bucket, reuse a cached
kubeconfig, or invoke the existing Terraform cluster apply path. It does not
delete or tear down infrastructure.

## 3. Verify Access

Kubernetes:

```bash
export KUBECONFIG=~/.npa/clusters/npa-cluster/kubeconfig
kubectl config current-context
kubectl get nodes
kubectl auth can-i create pods -n default
```

S3:

```bash
aws s3 ls s3://<bucket>/ --endpoint-url https://storage.eu-north1.nebius.cloud
```

Registry:

```bash
docker manifest inspect cr.eu-north1.nebius.cloud/<registry-id>/npa-lancedb:0.30.2
```

## 4. Run A Demo Step

Use the BDD100K/LanceDB pipeline as the first Kubernetes workflow smoke:

```bash
npa workbench workflow submit npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml \
  --run-id bdd100k-demo \
  --var NPA_S3_BUCKET=<bucket>
```

For a dry local check, use the VLM eval stub path:

```bash
npa workbench vlm-eval run \
  --input-path ./rollout.json \
  --output-path ./eval.json \
  --backend stub \
  --score 0.9 \
  --dry-run \
  --output json
```
