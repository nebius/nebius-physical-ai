#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
umask 077
mkdir -p secrets evidence postgres-data
chmod 700 secrets postgres-data
if command -v sudo >/dev/null 2>&1; then
  sudo chown "$(id -u):$(id -g)" secrets secrets/* 2>/dev/null || true
  sudo chown 70:70 postgres-data
else
  chown "$(id -u):$(id -g)" secrets secrets/* 2>/dev/null || true
  chown 70:70 postgres-data
fi

PROJECT_ID="${NPA_PROJECT_ID:-project-u00zhx4tpr00xh99b28n52}"
TENANT_ID="${NPA_TENANT_ID:-tenant-e00qjewwehnmpeh4tf}"
PROFILE="${NPA_NEBIUS_PROFILE:-npa-mk8s}"
REGION="${NPA_REGION:-}"
ENDPOINT_URL="${NPA_STORAGE_ENDPOINT:-}"
BUCKET_NAME="${MLFLOW_BUCKET_NAME:-npa-mlflow-$(printf %s "$PROJECT_ID" | sha256sum | cut -c1-10)}"
SA_NAME="${MLFLOW_SERVICE_ACCOUNT_NAME:-npa-mlflow-artifacts}"
KEY_NAME="${MLFLOW_ACCESS_KEY_NAME:-npa-mlflow-artifacts}"

if [[ -z "$REGION" || -z "$ENDPOINT_URL" ]]; then
  eval "$(/home/ubuntu/nebius-physical-ai/npa/.venv/bin/python - <<PY
import pathlib, yaml
cfg=yaml.safe_load((pathlib.Path.home()/'.npa/config.yaml').read_text())
project='$PROJECT_ID'
entry=(cfg.get('projects') or {}).get(project) or {}
region=entry.get('region') or '$REGION' or 'us-central1'
endpoint=(entry.get('storage') or {}).get('endpoint_url') or (entry.get('terraform_state') or {}).get('endpoint') or '$ENDPOINT_URL' or f'https://storage.{region}.nebius.cloud'
print(f'REGION={region!r}')
print(f'ENDPOINT_URL={endpoint!r}')
PY
)"
fi

echo "Discovered project=${PROJECT_ID} tenant=${TENANT_ID} region=${REGION} endpoint=${ENDPOINT_URL} bucket=${BUCKET_NAME}"

if ! nebius storage bucket get-by-name --parent-id "$PROJECT_ID" --name "$BUCKET_NAME" --profile "$PROFILE" --format json > evidence/bucket.json 2>/dev/null; then
  nebius storage bucket create --parent-id "$PROJECT_ID" --name "$BUCKET_NAME" --versioning-policy enabled --object-audit-logging mutate_only --profile "$PROFILE" --format json > evidence/bucket.json
fi
BUCKET_ID="$(jq -r '.metadata.id // .id' evidence/bucket.json)"

if ! nebius iam service-account get-by-name --parent-id "$PROJECT_ID" --name "$SA_NAME" --profile "$PROFILE" --format json > evidence/service-account.json 2>/dev/null; then
  nebius iam service-account create --parent-id "$PROJECT_ID" --name "$SA_NAME" --description "MLflow artifact proxy for isolated workbench bucket ${BUCKET_NAME}" --profile "$PROFILE" --format json > evidence/service-account.json
fi
SA_ID="$(jq -r '.metadata.id // .id' evidence/service-account.json)"

if ! nebius iam access-permit list --parent-id "$SA_ID" --profile "$PROFILE" --format json > evidence/access-permits.json 2>/dev/null; then
  : > evidence/access-permits.json
fi
if ! jq -e --arg rid "$BUCKET_ID" '.. | objects | select((.spec.resource_id? // .resource_id? // "") == $rid)' evidence/access-permits.json >/dev/null; then
  nebius iam access-permit create --parent-id "$SA_ID" --resource-id "$BUCKET_ID" --role storage.objectAdmin --name npa-mlflow-bucket-objects --profile "$PROFILE" --format json > evidence/access-permit-created.json || \
  nebius iam access-permit create --parent-id "$SA_ID" --resource-id "$BUCKET_ID" --role storage.editor --name npa-mlflow-bucket-objects --profile "$PROFILE" --format json > evidence/access-permit-created.json
fi

if [[ ! -s secrets/aws_access_key_id || ! -s secrets/aws_secret_access_key ]]; then
  nebius iam v2 access-key create --parent-id "$PROJECT_ID" --account-service-account-id "$SA_ID" --name "$KEY_NAME-$(date -u +%Y%m%d%H%M%S)" --description "S3 key for ${BUCKET_NAME}/mlflow" --secret-delivery-mode inline --profile "$PROFILE" --format json > secrets/access-key-created.json
  jq -r '.status.aws_access_key_id // .aws_access_key_id // .spec.aws_access_key_id // .metadata.id' secrets/access-key-created.json > secrets/aws_access_key_id
  jq -r '.status.secret // .secret // .status.aws_secret_access_key // .aws_secret_access_key // .secret_access_key' secrets/access-key-created.json > secrets/aws_secret_access_key
  if [[ "$(cat secrets/aws_secret_access_key)" == "null" || ! -s secrets/aws_secret_access_key ]]; then
    key_id="$(jq -r '.metadata.id // .id' secrets/access-key-created.json)"
    nebius iam v2 access-key get-secret "$key_id" --profile "$PROFILE" --format json > secrets/access-key-secret.json
    jq -r '.secret // .aws_secret_access_key // .secret_access_key' secrets/access-key-secret.json > secrets/aws_secret_access_key
  fi
fi

if [[ ! -s secrets/postgres_password ]]; then
  openssl rand -base64 36 > secrets/postgres_password
fi
chmod 600 secrets/postgres_password secrets/aws_access_key_id secrets/aws_secret_access_key
if command -v sudo >/dev/null 2>&1; then
  sudo install -o 70 -g 70 -m 0400 secrets/postgres_password secrets/postgres_password.pg
  sudo install -o 10001 -g 10001 -m 0400 secrets/postgres_password secrets/postgres_password.mlflow
  sudo install -o 10001 -g 10001 -m 0400 secrets/aws_access_key_id secrets/aws_access_key_id.mlflow
  sudo install -o 10001 -g 10001 -m 0400 secrets/aws_secret_access_key secrets/aws_secret_access_key.mlflow
else
  cp secrets/postgres_password secrets/postgres_password.pg
  cp secrets/postgres_password secrets/postgres_password.mlflow
  cp secrets/aws_access_key_id secrets/aws_access_key_id.mlflow
  cp secrets/aws_secret_access_key secrets/aws_secret_access_key.mlflow
  chown 70:70 secrets/postgres_password.pg
  chown 10001:10001 secrets/postgres_password.mlflow secrets/aws_access_key_id.mlflow secrets/aws_secret_access_key.mlflow
  chmod 400 secrets/postgres_password.pg secrets/postgres_password.mlflow secrets/aws_access_key_id.mlflow secrets/aws_secret_access_key.mlflow
fi
cat > .env <<ENV
NPA_PROJECT_ID=${PROJECT_ID}
NPA_TENANT_ID=${TENANT_ID}
NPA_REGION=${REGION}
MLFLOW_BUCKET_NAME=${BUCKET_NAME}
MLFLOW_S3_ENDPOINT_URL=${ENDPOINT_URL}
AWS_ENDPOINT_URL_S3=${ENDPOINT_URL}
MLFLOW_ARTIFACT_ROOT=s3://${BUCKET_NAME}/mlflow
MLFLOW_PG_HOST=postgres
MLFLOW_PG_PORT=5432
MLFLOW_PG_DATABASE=mlflow
MLFLOW_PG_USER=mlflow
ENV
chmod 600 .env
cat > evidence/resource-summary.json <<JSON
{"project_id":"${PROJECT_ID}","tenant_id":"${TENANT_ID}","region":"${REGION}","endpoint_url":"${ENDPOINT_URL}","bucket":"${BUCKET_NAME}","bucket_id":"${BUCKET_ID}","service_account_id":"${SA_ID}","compose":"docker compose on dev VM","postgres_persistence":"./postgres-data bind mount on VM block device"}
JSON
