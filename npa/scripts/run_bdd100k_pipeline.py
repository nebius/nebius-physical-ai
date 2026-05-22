#!/usr/bin/env python3
"""Submit or dry-validate the BDD100K SkyPilot pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

from npa.orchestration.skypilot import (
    WorkflowResult,  # noqa: F401 - kept for tests and downstream wrapper imports.
    cleanup_all_for_run,
    submit_workflow,
    workflow_status,
)
from npa.orchestration.skypilot._bin import (
    SkyPilotConfigError,
    SkyPilotNotInstalledError,
    SkyPilotVersionError,
    resolve_sky_bin,
)

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "workflows" / "skypilot" / "bdd100k-pipeline.yaml"
DEFAULT_BUCKET = os.environ.get("NPA_S3_BUCKET", "your-bucket-name")
DEFAULT_SOURCE = f"s3://{DEFAULT_BUCKET}/raw-bdd100k/subset-demo/"
DEFAULT_LANCEDB_ENDPOINT = "http://npa-lancedb.workbench.svc.cluster.local:8686"
DEFAULT_DETECTION_ENDPOINT = "http://npa-detection-training.workbench.svc.cluster.local:8790"
TERMINAL_STATUSES = {
    "SUCCEEDED",
    "CANCELLED",
    "FAILED",
    "FAILED_SETUP",
    "FAILED_PRECHECKS",
    "FAILED_NO_RESOURCE",
    "FAILED_CONTROLLER",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.mock_endpoints:
            return _run_mock_endpoint_validation(args)
        return _submit_and_wait(args)
    except (SkyPilotNotInstalledError, SkyPilotConfigError, SkyPilotVersionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "For a no-infrastructure validation, add --mock-endpoints. "
            "For live submission, configure SkyPilot with NPA_SKYPILOT_BIN.",
            file=sys.stderr,
        )
        return 2


def render_pipeline(
    yaml_path: Path,
    *,
    run_id: str,
    bucket: str = DEFAULT_BUCKET,
    source_uri: str = DEFAULT_SOURCE,
    bdd100k_limit: int = 10000,
    synthetic_rows: int = 0,
    lancedb_endpoint: str = DEFAULT_LANCEDB_ENDPOINT,
    detection_endpoint: str = DEFAULT_DETECTION_ENDPOINT,
    lancedb_token: str = "",
    detection_token: str = "",
) -> list[dict[str, Any]]:
    """Return SkyPilot YAML documents with run-scoped paths injected."""

    docs = _load_yaml_documents(yaml_path)
    prefix = f"bdd100k-pipeline/{run_id}"
    root_uri = f"s3://{bucket}/{prefix}"
    lance_uri = f"{root_uri}/lancedb/"
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        envs = doc.get("envs")
        if not isinstance(envs, dict):
            continue
        envs.update(
            {
                "NPA_PIPELINE_RUN_ID": run_id,
                "S3_BUCKET": bucket,
                "S3_PREFIX": prefix,
                "PIPELINE_ROOT_URI": root_uri,
                "LANCE_URI": lance_uri,
                "LANCEDB_ENDPOINT": lancedb_endpoint,
                "DETECTION_TRAINING_ENDPOINT": detection_endpoint,
            }
        )
        if "BDD100K_SOURCE_URI" in envs:
            envs["BDD100K_SOURCE_URI"] = source_uri
            envs["BDD100K_LIMIT"] = str(bdd100k_limit)
            envs["BDD100K_SYNTHETIC_ROWS"] = str(synthetic_rows)
        if "LANCEDB_TOKEN" in envs:
            envs["LANCEDB_TOKEN"] = lancedb_token
        if "DETECTION_TRAINING_TOKEN" in envs:
            envs["DETECTION_TRAINING_TOKEN"] = detection_token
        view_slug = envs.get("VIEW_SLUG")
        if view_slug and "TRAIN_OUTPUT_URI" in envs:
            envs["TRAIN_OUTPUT_URI"] = f"{root_uri}/training/{view_slug}"
        if view_slug and "EVAL_OUTPUT_URI" in envs:
            envs["EVAL_OUTPUT_URI"] = f"{root_uri}/eval/{view_slug}"
    return docs


def output_paths(run_id: str, *, bucket: str = DEFAULT_BUCKET) -> dict[str, Any]:
    root = f"s3://{bucket}/bdd100k-pipeline/{run_id}"
    views = [
        "bdd100k_rider_train",
        "bdd100k_nighttime_person_train",
        "bdd100k_distant_person_train",
    ]
    return {
        "root": f"{root}/",
        "lancedb": f"{root}/lancedb/",
        "training": {view: f"{root}/training/{view}/" for view in views},
        "eval": {view: f"{root}/eval/{view}/" for view in views},
    }


def _submit_and_wait(args: argparse.Namespace) -> int:
    run_id = args.run_id or _default_run_id()
    lancedb_token = args.lancedb_token or os.environ.get("LANCEDB_TOKEN", "")
    detection_token = args.detection_token or os.environ.get("DETECTION_TRAINING_TOKEN", "")
    docs = render_pipeline(
        args.yaml_path,
        run_id=run_id,
        bucket=args.bucket,
        source_uri=args.source_uri,
        bdd100k_limit=args.bdd100k_limit,
        synthetic_rows=args.synthetic_rows,
        lancedb_endpoint=args.lancedb_endpoint,
        detection_endpoint=args.detection_endpoint,
        lancedb_token=lancedb_token,
        detection_token=detection_token,
    )

    with tempfile.TemporaryDirectory(prefix=f"npa-bdd100k-pipeline-{run_id}-") as tmp:
        rendered_yaml = Path(tmp) / "bdd100k-pipeline.rendered.yaml"
        _write_yaml_documents(rendered_yaml, docs)
        if args.render_only:
            print(json.dumps({"run_id": run_id, "rendered_yaml": str(rendered_yaml), "outputs": output_paths(run_id, bucket=args.bucket)}, indent=2))
            return 0

        sky_bin = str(resolve_sky_bin(args.sky_bin or os.environ.get("NPA_SKYPILOT_BIN")))
        result = submit_workflow(
            rendered_yaml,
            run_id,
            isolated_config_dir=args.isolated_config_dir,
            sky_bin=sky_bin,
            timeout=args.submit_timeout,
        )
        summary: dict[str, Any] = {
            "run_id": run_id,
            "submit": result.__dict__,
            "outputs": output_paths(run_id, bucket=args.bucket),
        }
        if not result.ok or result.status != "SUBMITTED":
            print(json.dumps(summary, indent=2, sort_keys=True))
            return result.returncode or 1

        deadline = time.monotonic() + args.wait_timeout
        final = result
        while time.monotonic() < deadline:
            final = workflow_status(
                result.job_id,
                isolated_config_dir=args.isolated_config_dir,
                config_path=Path(result.log_paths["config"]) if result.log_paths.get("config") else None,
                sky_bin=sky_bin,
            )
            if final.status in TERMINAL_STATUSES:
                break
            time.sleep(args.poll_interval)
        summary["final"] = final.__dict__

        if args.cleanup:
            cleanup = cleanup_all_for_run(
                run_id,
                isolated_config_dir=args.isolated_config_dir,
                config_path=Path(result.log_paths["config"]) if result.log_paths.get("config") else None,
                sky_bin=sky_bin,
            )
            summary["cleanup"] = cleanup.__dict__

        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if final.status == "SUCCEEDED" else 1


def _run_mock_endpoint_validation(args: argparse.Namespace) -> int:
    run_id = args.run_id or _default_run_id()
    state = _MockState()
    lancedb_server = _start_mock_server("lancedb", state)
    detection_server = _start_mock_server("detection", state)
    lancedb_endpoint = f"http://127.0.0.1:{lancedb_server.server_port}"
    detection_endpoint = f"http://127.0.0.1:{detection_server.server_port}"
    docs = render_pipeline(
        args.yaml_path,
        run_id=run_id,
        bucket=args.bucket,
        source_uri=args.source_uri,
        bdd100k_limit=args.bdd100k_limit,
        synthetic_rows=args.synthetic_rows or 10,
        lancedb_endpoint=lancedb_endpoint,
        detection_endpoint=detection_endpoint,
    )

    failures: list[dict[str, str]] = []
    try:
        with tempfile.TemporaryDirectory(prefix=f"npa-bdd100k-mock-{run_id}-") as tmp:
            cwd = Path(tmp)
            for doc in docs[1:]:
                name = str(doc["name"])
                env = os.environ.copy()
                env.update({key: str(value) for key, value in doc.get("envs", {}).items()})
                env["TRAIN_POLL_SECONDS"] = "0"
                env["TRAIN_TIMEOUT_SECONDS"] = "60"
                env["WRITE_CANONICAL_EVAL_METRICS"] = "0"
                result = subprocess.run(
                    ["bash", "-lc", str(doc["run"])],
                    cwd=cwd,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=args.mock_task_timeout,
                    check=False,
                )
                state.task_results.append(
                    {
                        "name": name,
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                )
                if result.returncode != 0:
                    failures.append({"name": name, "stderr": result.stderr, "stdout": result.stdout})
                    break
    finally:
        lancedb_server.shutdown()
        detection_server.shutdown()

    summary = {
        "run_id": run_id,
        "lancedb_endpoint": lancedb_endpoint,
        "detection_endpoint": detection_endpoint,
        "lancedb_requests": state.lancedb_requests,
        "detection_requests": state.detection_requests,
        "task_results": state.task_results,
        "outputs": output_paths(run_id, bucket=args.bucket),
        "failures": failures,
    }
    if args.output_json:
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures and _mock_request_sequence_ok(summary) else 1


def _mock_request_sequence_ok(summary: dict[str, Any]) -> bool:
    lancedb_posts = [item["path"] for item in summary["lancedb_requests"] if item["method"] == "POST"]
    detection_posts = [item["path"] for item in summary["detection_requests"] if item["method"] == "POST"]
    return lancedb_posts == [
        "/import-bdd100k",
        "/backfill",
        "/backfill",
        "/backfill",
        "/backfill",
        "/backfill",
        "/backfill",
        "/create-mv",
        "/create-mv",
        "/create-mv",
    ] and detection_posts == [
        "/train",
        "/train",
        "/train",
        "/eval",
        "/eval",
        "/eval",
    ]


@dataclass
class _MockState:
    lancedb_requests: list[dict[str, Any]] = field(default_factory=list)
    detection_requests: list[dict[str, Any]] = field(default_factory=list)
    task_results: list[dict[str, Any]] = field(default_factory=list)
    runs: dict[str, dict[str, Any]] = field(default_factory=dict)


class _MockHandler(BaseHTTPRequestHandler):
    server_version = "NPABDD100KMock/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        state: _MockState = self.server.state  # type: ignore[attr-defined]
        kind: str = self.server.kind  # type: ignore[attr-defined]
        _record(state, kind, "GET", parsed.path, None)
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        if kind == "detection" and parsed.path == "/status":
            run_id = parse_qs(parsed.query).get("run_id", [""])[0]
            run = state.runs.get(run_id)
            if run is None:
                self._send_json({"detail": f"unknown run_id: {run_id}"}, status=404)
                return
            self._send_json(run)
            return
        if kind == "detection" and parsed.path == "/runs":
            self._send_json({"runs": list(state.runs.values())})
            return
        self._send_json({"detail": f"not found: {self.path}"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        payload = self._read_json()
        state: _MockState = self.server.state  # type: ignore[attr-defined]
        kind: str = self.server.kind  # type: ignore[attr-defined]
        _record(state, kind, "POST", parsed.path, payload)
        if kind == "lancedb":
            self._handle_lancedb_post(parsed.path, payload)
            return
        self._handle_detection_post(parsed.path, payload)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle_lancedb_post(self, path: str, payload: dict[str, Any]) -> None:
        if path == "/import-bdd100k":
            synthetic = payload.get("synthetic") or 10
            self._send_json(
                {
                    "table": payload.get("table", "bdd100k"),
                    "lance_uri": payload.get("lance_uri", ""),
                    "table_uri": f"{payload.get('lance_uri', '')}/{payload.get('table', 'bdd100k')}",
                    "rows_per_split": {"train": int(synthetic), "val": 1},
                    "total_rows": int(synthetic) + 1,
                    "table_version_before": None,
                    "table_version_after": 1,
                    "table_version": 1,
                    "manifest_sha256": "mock-import",
                    "row_checksum_sha256": "mock-rows",
                    "splits": ["train", "val"],
                    "synthetic": payload.get("synthetic"),
                    "synthetic_seed": None,
                    "source": payload.get("source", ""),
                }
            )
            return
        if path == "/backfill":
            udf = payload.get("udf", "")
            self._send_json(
                {
                    "table": payload.get("table", "bdd100k"),
                    "lance_uri": payload.get("lance_uri", ""),
                    "rows_updated": 11,
                    "rows_skipped": 0,
                    "table_version_before": 1,
                    "table_version_after": 2,
                    "udf": udf,
                    "output_column": udf,
                    "column_added": True,
                    "duration_ms": 1,
                    "manifest_sha256": f"mock-{udf}",
                    "gpu_used": udf == "clip_embedding",
                }
            )
            return
        if path == "/create-mv":
            self._send_json(
                {
                    "view_name": payload.get("name", ""),
                    "source_table": payload.get("source_table", "bdd100k"),
                    "filter_sql": payload.get("filter_sql", ""),
                    "row_count": 3,
                    "view_table_version": 1,
                    "manifest_sha256": "mock-mv",
                    "created_at": "2026-05-16T00:00:00Z",
                }
            )
            return
        self._send_json({"detail": f"not found: {path}"}, status=404)

    def _handle_detection_post(self, path: str, payload: dict[str, Any]) -> None:
        state: _MockState = self.server.state  # type: ignore[attr-defined]
        if path == "/train":
            view = str(payload.get("view", "view"))
            run_id = f"train-{view.replace('_', '-')}"
            output_uri = str(payload.get("output_uri", "s3://mock/out"))
            epochs = int(payload.get("epochs", 1))
            checkpoint_uri_pattern = f"{output_uri}/{run_id}/checkpoints/epoch_{{epoch}}.pt"
            metrics_uri = f"{output_uri}/{run_id}/metrics.json"
            state.runs[run_id] = {
                "run_id": run_id,
                "status": "completed",
                "epochs_completed": epochs,
                "total_epochs": epochs,
                "checkpoint_uri_pattern": checkpoint_uri_pattern,
                "metrics_uri": metrics_uri,
                "manifest_sha256": f"mock-{view}",
                "last_metrics": {"train_loss": 0.1},
                "error": None,
            }
            self._send_json(
                {
                    "run_id": run_id,
                    "status": "running",
                    "checkpoint_uri_pattern": checkpoint_uri_pattern,
                    "metrics_uri": metrics_uri,
                    "total_epochs": epochs,
                    "manifest_sha256": f"mock-{view}",
                }
            )
            return
        if path == "/eval":
            self._send_json(
                {
                    "mAP": 0.5,
                    "mAP_50": 0.6,
                    "mAP_75": 0.4,
                    "per_category_AP": {},
                    "eval_run_id": "eval-mock",
                    "manifest_sha256": "mock-eval",
                }
            )
            return
        self._send_json({"detail": f"not found: {path}"}, status=404)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length == 0:
            return {}
        data = self.rfile.read(length).decode("utf-8")
        return json.loads(data or "{}")

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _record(state: _MockState, kind: str, method: str, path: str, payload: dict[str, Any] | None) -> None:
    item = {"method": method, "path": path, "payload": payload}
    if kind == "lancedb":
        state.lancedb_requests.append(item)
    else:
        state.detection_requests.append(item)


def _start_mock_server(kind: str, state: _MockState) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
    server.kind = kind  # type: ignore[attr-defined]
    server.state = state  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _load_yaml_documents(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        docs = [doc for doc in yaml.safe_load_all(handle) if doc is not None]
    if not all(isinstance(doc, dict) for doc in docs):
        raise ValueError(f"SkyPilot YAML documents must be mappings: {path}")
    return docs


def _write_yaml_documents(path: Path, docs: list[dict[str, Any]]) -> None:
    path.write_text(yaml.safe_dump_all(docs, sort_keys=False), encoding="utf-8")


def _default_run_id() -> str:
    return "bdd100k-pipeline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml-path", "--yaml", dest="yaml_path", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--source-uri", default=DEFAULT_SOURCE)
    parser.add_argument("--bdd100k-limit", type=int, default=10000)
    parser.add_argument("--synthetic-rows", "--synthetic", dest="synthetic_rows", type=int, default=0)
    parser.add_argument("--lancedb-endpoint", default=DEFAULT_LANCEDB_ENDPOINT)
    parser.add_argument("--detection-endpoint", default=DEFAULT_DETECTION_ENDPOINT)
    parser.add_argument("--lancedb-token", default="")
    parser.add_argument("--detection-token", default="")
    parser.add_argument("--sky-bin", default="")
    parser.add_argument("--isolated-config-dir", type=Path, default=None)
    parser.add_argument("--submit-timeout", type=int, default=1800)
    parser.add_argument("--wait-timeout", type=int, default=43200)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--mock-endpoints", action="store_true")
    parser.add_argument("--mock-task-timeout", type=int, default=120)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
