# MLflow and Postgres on a Linux workbench VM

[Workbench docs](../../docs/workbench/README.md)

Run a local MLflow tracking server with Postgres metadata and an S3 artifact
proxy. Docker Compose owns the two containers; Nebius holds the artifact bucket.
This stack is separate from NPA's Kubernetes workflow runtime.

## Before you start

Use a **Linux VM you own**, Docker Engine, Bash, `jq`, OpenSSL, an authenticated
Nebius CLI profile, and permission to manage the selected project's storage and
IAM. The deploy script installs the Compose plugin if needed. It also creates or
reuses a dedicated bucket, service account, IAM group, and access key.

Use a dedicated bucket: bootstrap updates its `mlflow/*` bucket policy.
Keep `.env`, `secrets/`, `evidence/`, and `postgres-data/` private and out of Git.
Set the region and endpoint explicitly; the legacy discovery fallback assumes
a particular operator checkout path.

## Deploy and verify

From the repository root:

```bash
cd workbench/mlflow
export NPA_PROJECT_ID='<your-project-id>'
export NPA_TENANT_ID='<your-tenant-id>'
export NPA_NEBIUS_PROFILE='<your-authenticated-profile>'
export NPA_REGION=us-central1
export NPA_STORAGE_ENDPOINT=https://storage.us-central1.nebius.cloud

./scripts/deploy.sh
./scripts/verify.sh
```

Use the region and HTTPS endpoint for your actual project. Deployment builds the
local image and waits for service health. Verification exercises the tracking,
model, and S3 artifact path and writes `evidence/verify-summary.json`; require
`status: passed`. Open `http://127.0.0.1:5000` on the VM, or access that loopback
port through an authenticated SSH tunnel.

| Location | Data and lifetime |
| --- | --- |
| `postgres-data/` | Database on the VM's block disk; survives container restart |
| Dedicated S3 bucket, `mlflow/*` | Artifacts through the server proxy where the installed MLflow supports it |
| `secrets/` | Runtime-mounted Postgres and S3 credentials |
| `evidence/resource-summary.json` | Exact cloud resources selected or created by bootstrap |

MLflow binds only to `127.0.0.1:5000`; Postgres has no host-published port. A
remote/public service needs an authenticating HTTPS proxy before its bind
address changes.

## Optional checks and image reuse

To test restart persistence, run:

```bash
./scripts/verify-twice-clean.sh
```

This stops and redeploys the stack twice, verifies each pass, and keeps the
existing database and S3 data. Use it only when those restarts are acceptable.

To publish the MLflow and pinned Postgres images to your configured operator
registry, follow the repository's [image packaging procedure](../../docs/workbench/container-packaging.md),
then use the existing scripts:

```bash
./scripts/push-image.sh
set -a
. evidence/pushed-images.env
set +a
MLFLOW_USE_PUBLISHED_IMAGE=1 ./scripts/deploy.sh
```

The environment file selects the pushed images for redeployment. Keep it private.

## Stop and remove resources

```bash
docker compose down --remove-orphans
```

This stops the containers and retains the database, secrets, and cloud resources.
Back up needed metadata with Postgres tools and preserve artifacts before
removing storage. Review `evidence/resource-summary.json`, then retire only the
bucket, access key, service account, and IAM group owned by this stack through
Nebius administration. Do not delete the VM disk until its database is saved.

For Managed PostgreSQL migration, create the destination database and role,
restore a `pg_dump`, and update `MLFLOW_PG_HOST` and credentials. The Compose
stack's containerized database is not automatically migrated by NPA.
