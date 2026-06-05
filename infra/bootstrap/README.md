# Bootstrap

Generic setup assets live here. Demo-specific scripts belong under
`demos/<demo>/`.

## Configure

Interactive:

```bash
npa configure
```

Non-interactive template:

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

## Provision

CLI:

```bash
npa provision-if-absent --project default --cluster-name npa-cluster
```

SDK:

```python
from npa.sdk.provisioning import provision_if_absent

result = provision_if_absent(project="default", cluster_name="npa-cluster")
print(result.to_dict())
```

Standalone script path:

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

Standalone YAML path:

```bash
scripts/provision-if-absent-from-yaml.py --config infra/bootstrap/provision-if-absent.yaml all
```

The hook only creates resources when they are absent. It never runs Terraform
destroy and never deletes buckets, clusters, node groups, registries, or state.
