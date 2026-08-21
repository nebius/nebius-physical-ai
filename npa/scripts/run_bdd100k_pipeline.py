#!/usr/bin/env python3
"""Submit or dry-validate the BDD100K SkyPilot pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from npa.orchestration.skypilot import (
    WorkflowResult,  # noqa: F401 - kept for tests and downstream wrapper imports.
    cleanup_all_for_run,
    submit_workflow,
    workflow_status,
)
from npa.orchestration.skypilot.signal_teardown import (
    SignalTeardown,
    install_teardown_signal_handlers,
    restore_signal_handlers,
)
from npa.orchestration.skypilot._bin import (
    SkyPilotConfigError,
    SkyPilotNotInstalledError,
    SkyPilotVersionError,
    resolve_sky_bin,
)
from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions
from npa.orchestration.npa_workflow.submit import prepare_npa_workflow_for_submit

#: The npa.workflow spec is the workflow surface; the raw SkyPilot template it replaced is
#: being retired. `--yaml` still accepts a customer's own SkyPilot YAML (see `--spec`).
DEFAULT_SPEC = (
    Path(__file__).resolve().parents[2]
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "bdd100k-pipeline.yaml"
)
DEFAULT_BUCKET = os.environ.get("NPA_S3_BUCKET", "your-bucket-name")
DEFAULT_SOURCE = f"s3://{DEFAULT_BUCKET}/raw-bdd100k/subset-demo/"
#: Spec config key -> the view its eval prefix belongs to. Used to redirect eval output to
#: a local directory in `--mock-endpoints` mode.
EVAL_URI_KEYS = {
    "rider_eval_uri": "bdd100k_rider_train",
    "nighttime_eval_uri": "bdd100k_nighttime_person_train",
    "distant_eval_uri": "bdd100k_distant_person_train",
}
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


def config_overrides(
    *,
    bucket: str = DEFAULT_BUCKET,
    source_uri: str = DEFAULT_SOURCE,
    bdd100k_limit: int = 10000,
    synthetic_rows: int = 0,
    lancedb_endpoint: str = DEFAULT_LANCEDB_ENDPOINT,
    detection_endpoint: str = DEFAULT_DETECTION_ENDPOINT,
    train_poll_seconds: int | None = None,
    local_eval_root: Path | None = None,
) -> dict[str, str]:
    """Map this runner's flags onto the spec's config keys.

    The mapping is 1:1 by design: the spec was authored with the same knobs the template
    exposed as `envs`, so the runner no longer injects environment variables into rendered
    documents — it overrides config and lets the engine resolve every derived URI
    (`prefix`, `lance_uri`, the per-view train/eval prefixes).
    """

    overrides = {
        "bucket": bucket,
        "source_uri": source_uri,
        "bdd100k_limit": str(bdd100k_limit),
        "synthetic_rows": str(synthetic_rows),
        "lancedb_endpoint": lancedb_endpoint,
        "detection_endpoint": detection_endpoint,
    }
    if train_poll_seconds is not None:
        overrides["train_poll_seconds"] = str(train_poll_seconds)
    if local_eval_root is not None:
        # `--mock-endpoints` has no object storage, and the eval stages publish a canonical
        # metrics.json under their output prefix. Pointing that prefix at a local directory
        # keeps the artifact real and checkable without credentials.
        for key, view in EVAL_URI_KEYS.items():
            overrides[key] = str(local_eval_root / "eval" / view) + "/"
    return overrides


def prepare_pipeline(
    spec_path: Path,
    *,
    run_id: str,
    bucket: str = DEFAULT_BUCKET,
    source_uri: str = DEFAULT_SOURCE,
    bdd100k_limit: int = 10000,
    synthetic_rows: int = 0,
    lancedb_endpoint: str = DEFAULT_LANCEDB_ENDPOINT,
    detection_endpoint: str = DEFAULT_DETECTION_ENDPOINT,
    train_poll_seconds: int | None = None,
    local_eval_root: Path | None = None,
    resolve_images: bool = True,
):
    """Render the npa.workflow spec into a SkyPilot YAML plus its execution plan.

    The caller owns `prepared.temp_dir.cleanup()`.

    ``resolve_images=False`` forces SkyPilot's default image instead of resolving the pinned
    workbench images, which would need registry credentials. `--mock-endpoints` uses it: that
    mode never launches a pod, and a no-infrastructure validation must not require a registry
    login to run.
    """

    render_options = (
        SkypilotRenderOptions() if resolve_images else SkypilotRenderOptions(image_overrides={"*": ""})
    )
    return prepare_npa_workflow_for_submit(
        spec_path,
        run_id=run_id,
        render_options=render_options,
        config_overrides=config_overrides(
            bucket=bucket,
            source_uri=source_uri,
            bdd100k_limit=bdd100k_limit,
            synthetic_rows=synthetic_rows,
            lancedb_endpoint=lancedb_endpoint,
            detection_endpoint=detection_endpoint,
            train_poll_seconds=train_poll_seconds,
            local_eval_root=local_eval_root,
        ),
    )


#: Secrets the pipeline's service calls need; the tokens are read from the environment by
#: the tools' `--token-env`, so they are forwarded rather than baked into a document.
PIPELINE_SECRET_ENVS = ("LANCEDB_TOKEN", "DETECTION_TRAINING_TOKEN")


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


def _resolve_secret_envs(args: argparse.Namespace) -> list[str]:
    """Forward the service tokens the tools read, when they are present."""

    envs = []
    if args.lancedb_token or os.environ.get("LANCEDB_TOKEN"):
        envs.append("LANCEDB_TOKEN")
    if args.detection_token or os.environ.get("DETECTION_TRAINING_TOKEN"):
        envs.append("DETECTION_TRAINING_TOKEN")
    return envs


def _submit_and_wait(args: argparse.Namespace) -> int:
    run_id = args.run_id or _default_run_id()
    # Explicit flags win, but the tools read the tokens from the environment via
    # `--token-env`, so a flag has to be exported rather than injected into a document.
    if args.lancedb_token:
        os.environ["LANCEDB_TOKEN"] = args.lancedb_token
    if args.detection_token:
        os.environ["DETECTION_TRAINING_TOKEN"] = args.detection_token
    prepared = prepare_pipeline(
        args.spec_path,
        run_id=run_id,
        bucket=args.bucket,
        source_uri=args.source_uri,
        bdd100k_limit=args.bdd100k_limit,
        synthetic_rows=args.synthetic_rows,
        lancedb_endpoint=args.lancedb_endpoint,
        detection_endpoint=args.detection_endpoint,
        # --render-only is a preview; --default-image keeps it usable without a registry login.
        resolve_images=not args.default_image,
    )

    try:
        rendered_yaml = prepared.skypilot_yaml_path
        if args.render_only:
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "rendered_yaml": str(rendered_yaml),
                        "rendered_skypilot": rendered_yaml.read_text(encoding="utf-8"),
                        "stages": [step.state for step in prepared.plan.steps],
                        "outputs": output_paths(run_id, bucket=args.bucket),
                    },
                    indent=2,
                )
            )
            return 0

        sky_bin = str(resolve_sky_bin(args.sky_bin or os.environ.get("NPA_SKYPILOT_BIN")))
        teardown_guard = SignalTeardown(
            run_id=run_id,
            isolated_config_dir=args.isolated_config_dir,
            sky_bin=sky_bin,
            poll_interval=max(float(args.poll_interval), 0.0),
        )
        # SIGTERM/SIGINT handlers call the same idempotent teardown path as normal exit.
        previous_handlers = install_teardown_signal_handlers(teardown_guard.teardown)
        summary: dict[str, Any] | None = None
        return_code = 1
        try:
            teardown_guard.mark_launched()
            result = submit_workflow(
                rendered_yaml,
                run_id,
                isolated_config_dir=args.isolated_config_dir,
                sky_bin=sky_bin,
                timeout=args.submit_timeout,
                secret_envs=_resolve_secret_envs(args),
            )
            config_path = Path(result.log_paths["config"]) if result.log_paths.get("config") else None
            teardown_guard.mark_launched(config_path=config_path)
            summary = {
                "run_id": run_id,
                "submit": result.__dict__,
                "outputs": output_paths(run_id, bucket=args.bucket),
            }
            if not result.ok or result.status != "SUBMITTED":
                return_code = result.returncode or 1
            else:
                deadline = time.monotonic() + args.wait_timeout
                final = result
                while time.monotonic() < deadline:
                    final = workflow_status(
                        result.job_id,
                        isolated_config_dir=args.isolated_config_dir,
                        config_path=config_path,
                        sky_bin=sky_bin,
                    )
                    if final.status in TERMINAL_STATUSES:
                        break
                    time.sleep(args.poll_interval)
                summary["final"] = final.__dict__
                return_code = 0 if final.status == "SUCCEEDED" else 1

            if args.cleanup:
                cleanup = cleanup_all_for_run(
                    run_id,
                    isolated_config_dir=args.isolated_config_dir,
                    config_path=config_path,
                    sky_bin=sky_bin,
                )
                summary["cleanup"] = cleanup.__dict__
        finally:
            teardown = teardown_guard.teardown()
            restore_signal_handlers(previous_handlers)

        if summary is not None:
            summary["teardown"] = teardown.__dict__
            print(json.dumps(summary, indent=2, sort_keys=True))
        return 1 if teardown.errors else return_code
    finally:
        prepared.temp_dir.cleanup()


def _mock_stage_env() -> dict[str, str]:
    """Environment for a locally executed plan step.

    A step's argv starts with the bare ``npa`` console script, which resolves in a
    task pod but not when this runner is driven by an interpreter whose ``bin/``
    directory is absent from ``PATH`` -- exactly what
    ``<venv>/bin/python -m pytest`` does. Put the running interpreter's scripts
    directory first so the stage runs the same ``npa`` that imported this module.

    Handing this to ``subprocess`` is sufficient: it resolves a bare program name
    against the ``PATH`` in the mapping it is given, not the parent's.

    ``sysconfig`` is the lookup that stays correct for a venv whose ``python`` is a
    symlink to the system interpreter: resolving ``sys.executable`` there would
    escape the venv and land in ``/usr/bin``.
    """

    env = os.environ.copy()
    scripts_dir = sysconfig.get_path("scripts") or str(Path(sys.executable).parent)
    path = env.get("PATH", "")
    if scripts_dir not in path.split(os.pathsep):
        env["PATH"] = f"{scripts_dir}{os.pathsep}{path}" if path else scripts_dir
    return env


def _run_mock_endpoint_validation(args: argparse.Namespace) -> int:
    run_id = args.run_id or _default_run_id()
    state = _MockState()
    lancedb_server = _start_mock_server("lancedb", state)
    detection_server = _start_mock_server("detection", state)
    lancedb_endpoint = f"http://127.0.0.1:{lancedb_server.server_port}"
    detection_endpoint = f"http://127.0.0.1:{detection_server.server_port}"
    failures: list[dict[str, str]] = []
    prepared = None
    try:
        with tempfile.TemporaryDirectory(prefix=f"npa-bdd100k-mock-{run_id}-") as tmp:
            cwd = Path(tmp)
            prepared = prepare_pipeline(
                args.spec_path,
                run_id=run_id,
                bucket=args.bucket,
                source_uri=args.source_uri,
                bdd100k_limit=args.bdd100k_limit,
                # Zero source rows would make the mock ingest meaningless.
                synthetic_rows=args.synthetic_rows or 10,
                lancedb_endpoint=lancedb_endpoint,
                detection_endpoint=detection_endpoint,
                # A fast poll keeps `--wait` honest without a 30 s sleep per training stage;
                # the mock reports `completed` on the first `/status`.
                train_poll_seconds=0,
                local_eval_root=cwd,
                resolve_images=False,
            )
            # Execute each PLAN STEP's argv. The retired template's mock mode ran each raw
            # document's `run:` bash with that document's `envs`; a spec has no such bash, so
            # the unit of execution is the stage's resolved command. That is also a stronger
            # check: it is exactly what the engine will run in a pod.
            stage_env = _mock_stage_env()
            for step in prepared.plan.steps:
                if not step.argv:
                    continue
                result = subprocess.run(
                    step.argv,
                    cwd=cwd,
                    env=stage_env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=args.mock_task_timeout,
                    check=False,
                )
                state.task_results.append(
                    {
                        "name": step.state,
                        "argv": list(step.argv),
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                )
                if result.returncode != 0:
                    failures.append(
                        {"name": step.state, "stderr": result.stderr, "stdout": result.stdout}
                    )
                    break
    finally:
        if prepared is not None:
            prepared.temp_dir.cleanup()
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


#: The exact LanceDB write sequence the pipeline must produce: one import, five CPU UDF
#: backfills plus the CLIP one, then the three failure-mode views.
EXPECTED_LANCEDB_POSTS = (
    ["/import-bdd100k"] + ["/backfill"] * 6 + ["/create-mv"] * 3
)
#: Three trainings then three evaluations.
EXPECTED_DETECTION_POSTS = ["/train"] * 3 + ["/eval"] * 3


def _mock_request_sequence_ok(summary: dict[str, Any]) -> bool:
    lancedb_posts = [item["path"] for item in summary["lancedb_requests"] if item["method"] == "POST"]
    detection = summary["detection_requests"]
    detection_posts = [item["path"] for item in detection if item["method"] == "POST"]
    if lancedb_posts != EXPECTED_LANCEDB_POSTS or detection_posts != EXPECTED_DETECTION_POSTS:
        return False
    return _detection_call_order_ok(detection)


def _detection_call_order_ok(detection: list[dict[str, Any]]) -> bool:
    """Every train must be awaited, and every eval must resolve its checkpoint first.

    These are the two behaviours the retired template implemented in bash and that
    `--wait` / `--discover-checkpoint` moved into the tool, so the mock validation checks the
    call *order*, not just the POST counts: `POST /train` is followed by `GET /status`, and
    each `POST /eval` is preceded by a `GET /runs`.
    """

    calls = [(item["method"], item["path"]) for item in detection]
    for index, call in enumerate(calls):
        if call == ("POST", "/train"):
            following = calls[index + 1 : index + 2]
            if following != [("GET", "/status")]:
                return False
        if call == ("POST", "/eval"):
            preceding = calls[index - 1 : index]
            if preceding != [("GET", "/runs")]:
                return False
    return True


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


def _default_run_id() -> str:
    return "bdd100k-pipeline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        dest="spec_path",
        type=Path,
        default=DEFAULT_SPEC,
        help="npa.workflow spec to render and submit (default: the shipped BDD100K spec).",
    )
    # Deprecated alias kept so existing invocations keep working; it now names a spec.
    parser.add_argument("--yaml-path", "--yaml", dest="spec_path", type=Path, default=DEFAULT_SPEC)
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
    parser.add_argument(
        "--default-image",
        action="store_true",
        help="Render with SkyPilot's default image instead of resolving pinned workbench images.",
    )
    parser.add_argument("--mock-endpoints", action="store_true")
    parser.add_argument("--mock-task-timeout", type=int, default=120)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
