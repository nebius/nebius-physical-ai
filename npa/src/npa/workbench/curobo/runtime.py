"""Shared cuRobo operations for CLI, SDK, service and SkyPilot stages."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from npa.cli.path_contract import validate_read_path, validate_write_path
from npa.workbench.dataset.storage import read_bytes_uri, uri_join, write_bytes_uri
from npa.workbench.storage_scope import authorize_uri

from .artifacts import (
    CuroboError,
    build_rrd,
    canonical,
    read_journal,
    summarize,
    validate_report,
)
from .schemas import BenchmarkManifest, PlanManifest, PrepareRequest, RunRequest


_LOGGER = logging.getLogger(__name__)


def _paths(request: RunRequest):
    validate_read_path(request.input_path, tool="curobo", allow_hf=False)
    validate_write_path(request.output_path, tool="curobo", required=True)
    authorize_uri(request.input_path, operation="read")
    authorize_uri(request.output_path, operation="write")


def _publish(uri: str, payload: bytes):
    write_bytes_uri(uri, payload)
    if hashlib.sha256(read_bytes_uri(uri)).digest() != hashlib.sha256(payload).digest():
        raise CuroboError("artifact S3 read-after-write digest mismatch")


def prepare(request: PrepareRequest):
    validate_write_path(request.output_path, tool="curobo", required=True)
    modes = ["kinematic", "dynamics"] if request.mode == "both" else [request.mode]
    manifest = BenchmarkManifest(modes=modes).model_dump(mode="json")
    _publish(request.output_path, canonical(manifest))
    return {
        "schema_version": "npa.curobo.prepared.v1",
        "output_path": request.output_path,
        "modes": modes,
    }


def _run(kind: str, request: RunRequest):
    _paths(request)
    manifest = json.loads(read_bytes_uri(request.input_path))
    model = BenchmarkManifest if kind == "benchmark" else PlanManifest
    manifest = model.model_validate(manifest).model_dump(mode="json")
    root = Path(
        tempfile.mkdtemp(
            prefix="npa-curobo-", dir=os.environ.get("NPA_CUROBO_WORK_DIR")
        )
    )
    root.chmod(0o700)
    # Retain local facts on solver/upload failure; telemetry retries never replay a GPU run.
    (root / "input.json").write_bytes(canonical(manifest))
    command = [
        os.environ.get("NPA_CUROBO_PYTHON", sys.executable),
        "-m",
        "npa.workbench.curobo.runner",
        "--kind",
        kind,
        "--input",
        str(root / "input.json"),
        "--output",
        str(root / "output"),
        "--run-id",
        request.run_id,
    ]
    started = time.perf_counter()
    with (root / "runtime.log").open("wb") as log:
        completed = subprocess.run(
            command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode:
        raise CuroboError(
            f"upstream cuRobo {kind} failed with exit code {completed.returncode}; retained local journal and log"
        )
    rows = read_journal(root / "output/problems.jsonl")
    report = json.loads((root / "output/result.json").read_text())
    validate_report(report, rows, run_id=request.run_id)
    if kind == "benchmark" and report.get("requested_modes") != manifest["modes"]:
        raise CuroboError("completed benchmark does not cover the requested modes")
    if kind == "plan":
        expected = [
            ("kinematic", "operator", problem["id"]) for problem in manifest["problems"]
        ]
        observed = [(row["mode"], row["dataset"], row["problem_id"]) for row in rows]
        if observed != expected:
            raise CuroboError(
                "completed plan does not exactly cover the requested problem identities"
            )
        if any(row["status"] == "invalid" for row in rows):
            raise CuroboError("operator plan cannot exclude a validated input problem")
    report["subprocess_wall_seconds"] = time.perf_counter() - started
    report["input_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    report["journal_sha256"] = hashlib.sha256(
        (root / "output/problems.jsonl").read_bytes()
    ).hexdigest()
    for filename in ("problems.jsonl", "result.json"):
        data = (
            canonical(report)
            if filename == "result.json"
            else (root / "output" / filename).read_bytes()
        )
        _publish(uri_join(request.output_path, filename), data)
    # Both artifacts are durable and verified. Only remove this call's directory;
    # cleanup failure must not turn successful GPU work into a replayable failure.
    try:
        shutil.rmtree(root)
    except Exception:
        _LOGGER.warning("cuRobo artifacts verified; local working-file cleanup failed")
    return report


def benchmark(request: RunRequest):
    return _run("benchmark", request)


def plan(request: RunRequest):
    return _run("plan", request)


def _download_artifacts(request: RunRequest, root: Path):
    _paths(request)
    journal = read_bytes_uri(uri_join(request.input_path, "problems.jsonl"))
    report = json.loads(read_bytes_uri(uri_join(request.input_path, "result.json")))
    (root / "problems.jsonl").write_bytes(journal)
    rows = read_journal(root / "problems.jsonl")
    validate_report(report, rows, run_id=request.run_id)
    if report["journal_sha256"] != hashlib.sha256(journal).hexdigest():
        raise CuroboError("artifact journal hash mismatch")
    if not any(row["status"] == "success" for row in rows):
        raise CuroboError("no successful trajectory exists for review")
    return rows


def validate(request: RunRequest):
    with tempfile.TemporaryDirectory(prefix="npa-curobo-validate-") as directory:
        rows = _download_artifacts(request, Path(directory))
        result = {
            "schema_version": "npa.curobo.validation.v1",
            "run_id": request.run_id,
            "problem_count": len(rows),
            "summary": summarize(rows),
            "valid": True,
        }
        _publish(request.output_path, canonical(result))
        return result


def visualize(request: RunRequest):
    with tempfile.TemporaryDirectory(prefix="npa-curobo-viz-") as directory:
        root = Path(directory)
        _download_artifacts(request, root)
        result = build_rrd(
            root / "problems.jsonl", root / "planning.rrd", run_id=request.run_id
        )
        # Decode/verify the recording, not just its extension or producer success.
        sibling = Path(sys.executable).with_name("rerun")
        rerun = str(sibling) if sibling.is_file() else shutil.which("rerun")
        if not rerun:
            raise CuroboError("Rerun CLI is unavailable")
        checked = subprocess.run(
            [rerun, "rrd", "verify", str(root / "planning.rrd")],
            capture_output=True,
            check=False,
        )
        if checked.returncode:
            raise CuroboError("Rerun rejected the generated recording")
        _publish(
            uri_join(request.output_path, "planning.rrd"),
            (root / "planning.rrd").read_bytes(),
        )
        _publish(uri_join(request.output_path, "rrd-manifest.json"), canonical(result))
        return result
