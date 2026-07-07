#!/usr/bin/env python3
import os
import shlex
import subprocess
import sys


def read_secret(name: str, default: str = "") -> str:
    file_var = os.getenv(f"{name}_FILE")
    if file_var and os.path.exists(file_var):
        return open(file_var, "r", encoding="utf-8").read().strip()
    return os.getenv(name, default)


def main() -> None:
    db_password = read_secret("MLFLOW_DB_PASSWORD")
    pg_host = os.environ.get("MLFLOW_PG_HOST", "postgres")
    pg_port = os.environ.get("MLFLOW_PG_PORT", "5432")
    pg_db = os.environ.get("MLFLOW_PG_DATABASE", "mlflow")
    pg_user = os.environ.get("MLFLOW_PG_USER", "mlflow")
    artifact_root = os.environ["MLFLOW_ARTIFACT_ROOT"]

    backend_uri = f"postgresql+psycopg://{pg_user}:{db_password}@{pg_host}:{pg_port}/{pg_db}"
    help_text = subprocess.check_output(["mlflow", "server", "--help"], text=True)
    version = subprocess.check_output(["mlflow", "--version"], text=True).strip()
    psycopg_version = subprocess.check_output(
        [sys.executable, "-c", "import psycopg; print(psycopg.__version__)"], text=True
    ).strip()
    print(f"Starting {version} with psycopg {psycopg_version}", flush=True)

    cmd = [
        "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", "5000",
        "--backend-store-uri", backend_uri,
    ]
    if "--serve-artifacts" in help_text and "--artifacts-destination" in help_text:
        cmd.extend(["--serve-artifacts", "--artifacts-destination", artifact_root])
    else:
        cmd.extend(["--default-artifact-root", artifact_root])

    if "--gunicorn-opts" in help_text:
        cmd.extend(["--gunicorn-opts", "--workers 1 --access-logfile - --error-logfile - --timeout 120"])
    print("Exec:", " ".join(shlex.quote(part if not part.startswith("postgresql+") else "postgresql+psycopg://[REDACTED]") for part in cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
