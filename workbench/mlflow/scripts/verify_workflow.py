#!/usr/bin/env python3
import json
import os
import pathlib
import tempfile

import boto3
import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

root = pathlib.Path(__file__).resolve().parents[1]
if (root / ".env").exists():
    for line in (root / ".env").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

def secret(name: str, fallback: pathlib.Path | None = None) -> str:
    file_name = os.getenv(f"{name}_FILE")
    if file_name and pathlib.Path(file_name).exists():
        return pathlib.Path(file_name).read_text().strip()
    if os.getenv(name):
        return os.environ[name]
    if fallback and fallback.exists():
        return fallback.read_text().strip()
    raise RuntimeError(f"missing secret {name}")

os.environ["AWS_ACCESS_KEY_ID"] = secret("AWS_ACCESS_KEY_ID", root / "secrets/aws_access_key_id")
os.environ["AWS_SECRET_ACCESS_KEY"] = secret("AWS_SECRET_ACCESS_KEY", root / "secrets/aws_secret_access_key")
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", os.environ["AWS_ENDPOINT_URL_S3"])

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient(tracking_uri=tracking_uri)
experiment_obj = client.get_experiment_by_name("npa-mlflow-postgres-e2e")
experiment_id = experiment_obj.experiment_id if experiment_obj else client.create_experiment("npa-mlflow-postgres-e2e")
model_name = "NpaMlflowPostgresRoundTrip"

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7, test_size=0.25)
model = Pipeline([("scale", StandardScaler()), ("sgd", SGDRegressor(max_iter=1, warm_start=True, learning_rate="constant", eta0=0.0005, random_state=7))])

with mlflow.start_run(experiment_id=experiment_id, run_name="postgres-s3-round-trip") as run:
    run_id = run.info.run_id
    mlflow.log_params({"model": "SGDRegressor", "epochs": 8, "eta0": 0.0005, "dataset": "sklearn_diabetes"})
    for epoch in range(8):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mlflow.log_metric("rmse", float(mean_squared_error(y_test, pred) ** 0.5), step=epoch)
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        config = {"epochs": 8, "features": int(X.shape[1]), "run_id": run_id}
        (td / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        np.save(td / "checkpoint.npy", model.predict(X_test[:8]))
        mlflow.log_artifacts(str(td), artifact_path="checkpoints")
    mlflow.sklearn.log_model(model, name="model", registered_model_name=model_name)

versions = client.search_model_versions(f"name='{model_name}'")
version = max(versions, key=lambda v: int(v.version))
for _ in range(60):
    version = client.get_model_version(model_name, version.version)
    if version.status == ModelVersionStatus.to_string(ModelVersionStatus.READY):
        break
else:
    raise RuntimeError(f"model version not ready: {version.status}")
try:
    client.transition_model_version_stage(model_name, version.version, "Staging", archive_existing_versions=False)
except Exception:
    client.set_registered_model_alias(model_name, "staging", version.version)

loaded = mlflow.pyfunc.load_model(f"models:/{model_name}/{version.version}")
preds = loaded.predict(X_test[:3])
if len(preds) != 3:
    raise RuntimeError("unexpected prediction shape")

s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"])
objects = s3.list_objects_v2(Bucket=os.environ["MLFLOW_BUCKET_NAME"], Prefix="mlflow/", MaxKeys=50)
keys = [obj["Key"] for obj in objects.get("Contents", [])]
if not any("checkpoint.npy" in k for k in keys) or not any("MLmodel" in k for k in keys):
    raise RuntimeError(f"expected artifacts missing from S3 listing: {keys}")
summary = {"run_id": run_id, "model_name": model_name, "model_version": version.version, "prediction_sample": [float(x) for x in preds], "s3_keys_sample": keys[:20]}
evidence_dir = os.getenv("EVIDENCE_DIR")
if evidence_dir:
    pathlib.Path(evidence_dir).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(evidence_dir) / "workflow-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
