# MLflow + Postgres Workbench Stack

This VM-scoped stack uses Docker Compose because the dev VM already has Docker and the requested service is local workbench infrastructure, not a multi-node Workbench/Kubernetes workload. Postgres metadata stays on a host bind mount (`./postgres-data`) backed by the VM block disk; artifacts go to an isolated Nebius Object Storage bucket through the MLflow server artifact proxy when the installed MLflow version supports `--serve-artifacts`.

MLflow is published only on `127.0.0.1:5000`; Postgres is on an internal Docker network only and has no host-published port. If MLflow is exposed beyond localhost, place it behind TLS and authentication (for example an HTTPS reverse proxy with Basic/OIDC auth) before changing the bind address.

Postgres migration note: this Compose service is intentionally containerized for the dev workbench. To migrate to Nebius Managed PostgreSQL, create a managed instance/database/role, restore a `pg_dump` from `./postgres-data`, and replace `MLFLOW_PG_HOST`/credentials in `.env` and `secrets/postgres_password`; no MLflow schema changes are required.

## Commands

```bash
cd ~/nebius-physical-ai-mlflow/workbench/mlflow
./scripts/deploy.sh
./scripts/verify.sh
./scripts/verify-twice-clean.sh
```

`bootstrap-nebius.sh` discovers the project region and storage endpoint from `~/.npa/config.yaml` for `project-u00zhx4tpr00xh99b28n52`, creates/reuses a dedicated bucket, service account, bucket-scoped access permit, and runtime-mounted S3 key files under `secrets/`.
