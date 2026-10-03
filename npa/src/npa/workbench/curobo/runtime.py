"""Shared cuRobo operations for CLI, SDK, service and SkyPilot stages."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from npa.cli.path_contract import validate_read_path, validate_write_path
from npa.workbench.dataset.storage import read_bytes_uri, uri_join, write_bytes_uri
from npa.workbench.dataset import storage as dataset_storage
from npa.workbench.storage_scope import authorize_uri

from .audit import audit_bytes
from .artifacts import (
    CuroboError,
    build_rrd,
    canonical,
    decode_rrd,
    read_journal,
    validate_report,
)
from .replay import ReplayError, replay_rows
from .schemas import BenchmarkManifest, PlanManifest, PrepareRequest, RunRequest


_LOGGER = logging.getLogger(__name__)
_FAILURE_NAMESPACE = "_failures"


def _paths(request: RunRequest):
    validate_read_path(request.input_path, tool="curobo", allow_hf=False)
    validate_write_path(request.output_path, tool="curobo", required=True)
    authorize_uri(request.input_path, operation="read")
    authorize_uri(request.output_path, operation="write")


def _publish(uri: str, payload: bytes):
    write_bytes_uri(uri, payload)
    if hashlib.sha256(read_bytes_uri(uri)).digest() != hashlib.sha256(payload).digest():
        raise CuroboError("artifact S3 read-after-write digest mismatch")


def _stream_digest(stream):
    digest, size = hashlib.sha256(), 0
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _publish_file(uri: str, path: Path):
    target = authorize_uri(uri, operation="write")
    authorize_uri(uri, operation="read")
    if target.kind != "s3":
        raise CuroboError("cuRobo evidence publication requires S3")
    client = dataset_storage._s3_client()
    with path.open("rb") as stream:
        expected = _stream_digest(stream)
        stream.seek(0)
        client.put_object(
            Bucket=target.bucket, Key=target.key, Body=stream, ContentLength=expected[1]
        )
    body = client.get_object(Bucket=target.bucket, Key=target.key)["Body"]
    try:
        observed = _stream_digest(body)
    finally:
        body.close()
    if expected != observed:
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


def _runtime_root(manifest: dict) -> Path:
    root = Path(
        tempfile.mkdtemp(
            prefix="npa-curobo-", dir=os.environ.get("NPA_CUROBO_WORK_DIR")
        )
    )
    root.chmod(0o700)
    (root / "input.json").write_bytes(canonical(manifest))
    return root


def _runner_command(kind: str, request: RunRequest, root: Path) -> list[str]:
    return [
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


def _invoke_runner(kind: str, request: RunRequest, root: Path):
    command = _runner_command(kind, request, root)
    with (root / "runtime.log").open("wb") as log:
        return subprocess.run(
            command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False
        )


def _evidence_record(role: str, path: str, payload: bytes, **facts) -> dict:
    return {
        "role": role,
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        **facts,
    }


def _partial_journal(root: Path):
    path = root / "output/problems.jsonl"
    if not path.is_file() or path.is_symlink():
        return None
    digest, size = hashlib.sha256(), 0
    physical_lines, complete_records = 0, 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            size += len(line)
            physical_lines += 1
            try:
                complete_records += isinstance(json.loads(line), dict)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    return path, {
        "role": "partial_journal",
        "path": "partial-problems.jsonl",
        "bytes": size,
        "sha256": digest.hexdigest(),
        "partial": True,
        "physical_line_count": physical_lines,
        "complete_record_count": complete_records,
    }


def _file_evidence_record(role: str, path: Path) -> dict:
    with path.open("rb") as stream:
        digest, size = _stream_digest(stream)
    return {"role": role, "path": path.name, "bytes": size, "sha256": digest}


def _publish_failure_evidence(
    kind: str, request: RunRequest, root: Path, exit_code: int
) -> str:
    namespace = uri_join(request.output_path, _FAILURE_NAMESPACE, request.run_id)
    runtime_log = root / "runtime.log"
    artifacts = [_file_evidence_record("runtime_log", runtime_log)]
    _publish_file(uri_join(namespace, "runtime.log"), runtime_log)
    partial = _partial_journal(root)
    if partial is not None:
        path, record = partial
        _publish_file(uri_join(namespace, record["path"]), path)
        artifacts.append(record)
    receipt = canonical(
        {
            "schema_version": "npa.curobo.failure.v1",
            "status": "failed",
            "failure_type": "subprocess_exit",
            "kind": kind,
            "run_id": request.run_id,
            "subprocess_exit_code": exit_code,
            "artifacts": artifacts,
        }
    )
    _publish(uri_join(namespace, "failure.json"), receipt)
    return f"{_FAILURE_NAMESPACE}/{request.run_id}/failure.json"


def _raise_runner_failure(
    kind: str, request: RunRequest, root: Path, exit_code: int
) -> None:
    primary = f"upstream cuRobo {kind} failed with exit code {exit_code}"
    try:
        receipt = _publish_failure_evidence(kind, request, root, exit_code)
    except Exception as evidence_error:
        error_type = type(evidence_error).__name__
        raise CuroboError(
            f"{primary}; failure evidence publication failed ({error_type})"
        ) from None
    raise CuroboError(f"{primary}; durable failure receipt: {receipt}")


def _completed_output(
    kind: str,
    request: RunRequest,
    root: Path,
    manifest: dict,
    started: float,
):
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
    return report


def _publish_completed(request: RunRequest, root: Path, report: dict) -> None:
    journal = (root / "output/problems.jsonl").read_bytes()
    for filename, data in (
        ("problems.jsonl", journal),
        ("result.json", canonical(report)),
    ):
        _publish(uri_join(request.output_path, filename), data)


def _cleanup_completed(root: Path) -> None:
    try:
        shutil.rmtree(root)
    except Exception:
        _LOGGER.warning("cuRobo artifacts verified; local working-file cleanup failed")


def _run(kind: str, request: RunRequest):
    _paths(request)
    model = BenchmarkManifest if kind == "benchmark" else PlanManifest
    manifest = model.model_validate(
        json.loads(read_bytes_uri(request.input_path))
    ).model_dump(mode="json")
    root = _runtime_root(manifest)
    # Telemetry retries must never replay GPU work, so every failure retains this root.
    started = time.perf_counter()
    completed = _invoke_runner(kind, request, root)
    if completed.returncode:
        _raise_runner_failure(kind, request, root, completed.returncode)
    report = _completed_output(kind, request, root, manifest, started)
    _publish_completed(request, root, report)
    _cleanup_completed(root)
    return report


def benchmark(request: RunRequest):
    return _run("benchmark", request)


def plan(request: RunRequest):
    return _run("plan", request)


def _download_artifacts(request: RunRequest, root: Path):
    _paths(request)
    if _FAILURE_NAMESPACE in request.input_path.rstrip("/").split("/"):
        raise CuroboError("failure evidence is not an accepted cuRobo result")
    journal = read_bytes_uri(uri_join(request.input_path, "problems.jsonl"))
    result_bytes = read_bytes_uri(uri_join(request.input_path, "result.json"))
    report = json.loads(result_bytes)
    (root / "problems.jsonl").write_bytes(journal)
    (root / "result.json").write_bytes(result_bytes)
    rows = read_journal(root / "problems.jsonl")
    validate_report(report, rows, run_id=request.run_id)
    if report["journal_sha256"] != hashlib.sha256(journal).hexdigest():
        raise CuroboError("artifact journal hash mismatch")
    if not any(row["status"] == "success" for row in rows):
        raise CuroboError("no successful trajectory exists for review")
    return result_bytes, journal, report, rows


def _require_replay_tolerance(replay: dict) -> None:
    values = (
        (replay.get("terminal_goal_distance_m", {}).get("max"), 0.005),
        (replay.get("terminal_goal_orientation_rad", {}).get("max"), 0.05),
    )
    for value, limit in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value > limit
        ):
            raise CuroboError("independent terminal goal replay exceeds tolerance")


def validate(request: RunRequest):
    with tempfile.TemporaryDirectory(
        prefix="npa-curobo-validate-", dir=os.environ.get("NPA_CUROBO_WORK_DIR")
    ) as directory:
        result_bytes, journal, report, rows = _download_artifacts(
            request, Path(directory)
        )
        result = audit_bytes(result_bytes, journal, run_id=request.run_id)
        try:
            replay = replay_rows(rows, report)
        except ReplayError as exc:
            raise CuroboError(
                "independent kinematics or dynamics replay failed"
            ) from exc
        _require_replay_tolerance(replay)
        result["independent_replay"] = replay
        _publish(request.output_path, canonical(result))
        return result


def visualize(request: RunRequest):
    with tempfile.TemporaryDirectory(
        prefix="npa-curobo-viz-", dir=os.environ.get("NPA_CUROBO_WORK_DIR")
    ) as directory:
        root = Path(directory)
        result_bytes, _journal, _report, rows = _download_artifacts(request, root)
        del _journal
        result = build_rrd(
            root / "problems.jsonl", root / "planning.rrd", run_id=request.run_id
        )
        result["result_sha256"] = hashlib.sha256(result_bytes).hexdigest()
        result["decode"] = decode_rrd(
            root / "planning.rrd",
            rows=rows,
            run_id=request.run_id,
        )
        _publish_file(
            uri_join(request.output_path, "planning.rrd"),
            root / "planning.rrd",
        )
        _publish(uri_join(request.output_path, "rrd-manifest.json"), canonical(result))
        return result
