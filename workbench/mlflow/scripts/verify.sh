#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p evidence
source .env
export AWS_ACCESS_KEY_ID="$(cat secrets/aws_access_key_id)"
export AWS_SECRET_ACCESS_KEY="$(cat secrets/aws_secret_access_key)"
export AWS_EC2_METADATA_DISABLED=true
export MLFLOW_S3_ENDPOINT_URL AWS_ENDPOINT_URL_S3

docker compose ps > evidence/compose-ps.txt
curl -fsS http://127.0.0.1:5000/health > evidence/mlflow-health.txt
docker compose exec -T postgres pg_isready -U mlflow -d mlflow > evidence/pg-isready.txt

docker image history --no-trunc npa-mlflow-server:local > evidence/mlflow-image-history.txt
if grep -Ei "(AWS_SECRET|SECRET_ACCESS|PASSWORD=|PRIVATE_KEY|BEGIN .* KEY)" evidence/mlflow-image-history.txt; then
  echo "secret-looking value found in mlflow image history" >&2
  exit 1
fi
for image in npa-mlflow-server:local cgr.dev/chainguard/postgres@sha256:0edb7d98cf916a0f00f80c0f4b9257c8737c1ee1848d1e4e0f480b12a932d90b; do
  safe="$(echo "$image" | tr "/:@" "____")"
  trivy image --scanners vuln --parallel 1 --exit-code 1 --severity HIGH,CRITICAL --ignore-unfixed --no-progress "$image" > "evidence/trivy-vuln-${safe}.txt"
  trivy image --scanners secret --parallel 1 --exit-code 1 --no-progress "$image" > "evidence/trivy-secret-${safe}.txt"
done

docker compose exec -T mlflow mlflow --version > evidence/mlflow-version.txt
docker compose exec -T mlflow python -c "import psycopg, psycopg2; print(f'psycopg={psycopg.__version__} psycopg2={psycopg2.__version__}')" > evidence/psycopg-version.txt
docker compose exec -T mlflow mlflow server --help > evidence/mlflow-server-help.txt

docker compose exec -T mlflow python /opt/mlflow/src/verify_workflow.py | tee evidence/workflow-summary.json
run_id="$(jq -r .run_id evidence/workflow-summary.json)"
model_name="$(jq -r .model_name evidence/workflow-summary.json)"

aws --endpoint-url "$AWS_ENDPOINT_URL_S3" s3 ls "s3://${MLFLOW_BUCKET_NAME}/mlflow/" --recursive > evidence/s3-listing.txt
grep -E "checkpoint.npy|MLmodel" evidence/s3-listing.txt >/dev/null

docker compose exec -T postgres psql -U mlflow -d mlflow -v ON_ERROR_STOP=1 -c "select run_uuid, status from runs where run_uuid = '${run_id}';" > evidence/sql-run.txt
docker compose exec -T postgres psql -U mlflow -d mlflow -v ON_ERROR_STOP=1 -c "select name from registered_models where name = '${model_name}';" > evidence/sql-model.txt
docker compose exec -T postgres psql -U mlflow -d mlflow -v ON_ERROR_STOP=1 -c "select key, value, step from latest_metrics where run_uuid = '${run_id}' order by step desc limit 5;" > evidence/sql-metrics.txt

docker compose restart postgres mlflow
./scripts/wait-healthy.sh
docker compose exec -T -e RUN_ID="$run_id" -e MODEL_NAME="$model_name" mlflow python - <<'PY' | tee evidence/restart-client-check.json
import json, mlflow, os
from mlflow import MlflowClient
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client=MlflowClient()
run=client.get_run(os.environ["RUN_ID"])
model=client.get_registered_model(os.environ["MODEL_NAME"])
print(json.dumps({"run_id": run.info.run_id, "run_status": run.info.status, "model_name": model.name, "latest_versions": [v.version for v in model.latest_versions]}, indent=2))
PY
docker compose exec -T postgres psql -U mlflow -d mlflow -v ON_ERROR_STOP=1 -c "select count(*) as runs from runs; select count(*) as registered_models from registered_models;" > evidence/restart-sql-counts.txt
aws --endpoint-url "$AWS_ENDPOINT_URL_S3" s3 ls "s3://${MLFLOW_BUCKET_NAME}/mlflow/" --recursive > evidence/restart-s3-listing.txt

docker compose config > evidence/compose-rendered.yml
if docker compose port postgres 5432 >/tmp/pg-port.txt 2>/dev/null && [ -s /tmp/pg-port.txt ]; then
  echo "Postgres unexpectedly has a published port" >&2
  cat /tmp/pg-port.txt >&2
  exit 1
fi
if ! docker compose port mlflow 5000 | grep -q "127.0.0.1:5000"; then
  echo "MLflow is not bound to localhost" >&2
  docker compose port mlflow 5000 >&2 || true
  exit 1
fi
jq -n --arg run_id "$run_id" --arg model_name "$model_name" --arg bucket "$MLFLOW_BUCKET_NAME" \
  '{status:"passed",run_id:$run_id,model_name:$model_name,bucket:$bucket}' > evidence/verify-summary.json
cat evidence/verify-summary.json
