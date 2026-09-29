"""Persist reviewed variants in Postgres and an authenticated MLflow service."""

from __future__ import annotations

import os
import time
from urllib.parse import urlsplit

import httpx

from npa.workflows.video_sweep.artifacts import digest, write_json
from npa.workflows.video_sweep.execution import reviewed


def _mlflow_client() -> httpx.Client:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    parsed = urlsplit(uri)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Set an unsigned HTTP(S) MLFLOW_TRACKING_URI; supply auth separately"
        )
    if parsed.scheme == "http" and parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError("Remote MLflow tracking requires HTTPS")
    token = os.environ.get("MLFLOW_TRACKING_TOKEN", "")
    headers = {"Authorization": "Bearer " + token} if token else {}
    return httpx.Client(
        base_url=uri.rstrip("/") + "/", headers=headers, follow_redirects=False
    )


def _post(client: httpx.Client, operation: str, payload: dict) -> dict:
    response = client.post("api/2.0/mlflow/" + operation, json=payload)
    response.raise_for_status()
    return response.json()


def _track(client: httpx.Client, row: dict, plan: dict, experiment: str) -> str:
    tags = {
        "npa.run_id": plan["run_id"],
        "npa.variant_id": row["id"],
        "npa.plan_sha256": digest(plan),
    }
    result = _post(
        client,
        "runs/create",
        {
            "experiment_id": experiment,
            "start_time": int(time.time() * 1000),
            "tags": [{"key": key, "value": value} for key, value in tags.items()],
        },
    )
    run = result["run"]["info"]["run_id"]
    metrics = [
        {"key": key, "value": value, "timestamp": int(time.time() * 1000), "step": 0}
        for key, value in {
            "review_score": row["score"],
            "accepted": int(row["accepted"]),
        }.items()
    ]
    _post(
        client,
        "runs/log-batch",
        {
            "run_id": run,
            "metrics": metrics,
            "params": [{"key": "video_sha256", "value": row["sha256"]}],
            "tags": [],
        },
    )
    _post(
        client,
        "runs/update",
        {"run_id": run, "status": "FINISHED", "end_time": int(time.time() * 1000)},
    )
    return run


def _record(
    connection, client, row: dict, plan: dict, report_hash: str, experiment: str
) -> str:
    from psycopg.types.json import Jsonb

    key = (plan["run_id"], row["id"])
    previous = connection.execute(
        "SELECT review_sha256, mlflow_run_id FROM npa_video_variants WHERE run_id=%s AND variant_id=%s",
        key,
    ).fetchone()
    if previous:
        if previous[0] != report_hash:
            raise ValueError("A different review is already tracked for this run")
        return previous[1]
    tracking_id = _track(client, row, plan, experiment)
    item = next(item for item in plan["items"] if item["id"] == row["id"])
    payload = {
        "source": item["source"],
        "variant": item["variant"],
        "prompt": item["prompt"],
        "merge_provenance": item["merge_provenance"],
        "review": row,
    }
    connection.execute(
        "INSERT INTO npa_video_variants (run_id,variant_id,review_sha256,mlflow_run_id,lineage) VALUES (%s,%s,%s,%s,%s)",
        (*key, report_hash, tracking_id, Jsonb(payload)),
    )
    return tracking_id


def lineage(args) -> None:
    """Commit Postgres lineage and MLflow metrics before allowing publication.

    Args:
        args: Parsed stage arguments.
    Returns:
        None.
    Raises:
        ValueError: Credentials, tracking configuration, or review binding fails.
    """
    import psycopg

    plan, report = reviewed(args)
    dsn = os.environ.get("NPA_LINEAGE_POSTGRES_DSN", "")
    experiment = os.environ.get("MLFLOW_EXPERIMENT_ID", "")
    if not dsn or not experiment:
        raise ValueError("Set NPA_LINEAGE_POSTGRES_DSN and MLFLOW_EXPERIMENT_ID")
    records = []
    with psycopg.connect(dsn) as connection, _mlflow_client() as client:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS npa_video_variants (run_id text NOT NULL, variant_id text NOT NULL, review_sha256 text NOT NULL, mlflow_run_id text NOT NULL, lineage jsonb NOT NULL, PRIMARY KEY (run_id, variant_id))"
        )
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (plan["run_id"],)
        )
        for row in report["items"]:
            tracking_id = _record(
                connection, client, row, plan, digest(report), experiment
            )
            records.append({"id": row["id"], "mlflow_run_id": tracking_id})
    write_json(
        args.root_uri + "/lineage.json",
        {"review_sha256": digest(report), "items": records},
    )
