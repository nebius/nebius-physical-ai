"""Standard Ray Jobs client for source revisions and exact-job cancellation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shlex
import tempfile
import time
from urllib.parse import urlparse
import uuid

import validation

SOURCE_FILES = ("application.py", "worker.py", "validation.py")
UDF_FILENAME = "npa_lancedb_bdd100k_udfs.py"


def prepare_source(source: Path, destination: Path, revision: str, *, udf_source: Path) -> dict:
    if revision not in {"baseline", "changed", "restored"}:
        raise ValueError("Unknown source revision")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    selected = [(name, source / name) for name in SOURCE_FILES]
    selected.append((UDF_FILENAME, udf_source))
    for name, path in selected:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Source files must be regular files, without symbolic links")
        content = path.read_bytes()
        if name == "worker.py":
            needle = b'CROP_POLICY = "left"'
            if content.count(needle) != 1:
                raise ValueError("The baseline source must have exactly one left crop policy")
            if revision == "changed":
                content = content.replace(needle, b'CROP_POLICY = "right"')
        (destination / name).write_bytes(content)
        manifest[name] = validation.file_hash(destination / name)
    return manifest


@contextmanager
def application_address(address: str):
    parsed = urlparse(address)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Use a loopback Jobs URL through authenticated port forwarding")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("Jobs URL must not contain credentials, a path, query, or fragment")
    if parsed.port == 8266:
        raise ValueError("SkyPilot management Dashboard is forbidden")
    previous = os.environ.get("RAY_ADDRESS")
    os.environ["RAY_ADDRESS"] = address
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("RAY_ADDRESS", None)
        else:
            os.environ["RAY_ADDRESS"] = previous


def report_from_logs(logs: str) -> dict:
    marker = "RAY_CLIP_REPORT "
    reports = [json.loads(line.split(marker, 1)[1]) for line in logs.splitlines() if marker in line]
    if len(reports) != 1:
        raise ValueError("Expected exactly one completed application report")
    return reports[0]


def run_job(client, args, source: Path, revision: str, *, cancel: bool = False, baseline_path: str | None = None) -> dict:
    identity = f"clip-{revision}-{uuid.uuid4().hex}"
    job_output = str(Path(args.output_path) / identity)
    command = [args.python, "application.py", "--output-path", job_output,
               "--model-path", args.model_path, "--model-revision", args.model_revision,
               "--records", str(args.records), "--actors", str(args.actors),
               "--batch-size", str(args.batch_size)]
    if args.recovery_check and not cancel:
        command.append("--recovery-check")
    if cancel:
        command.append("--cancellation-probe")
    if baseline_path:
        command.extend(["--compare-baseline-path", baseline_path])
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ray-clip-source-") as temporary:
        package = Path(temporary)
        manifest = prepare_source(source, package, revision, udf_source=Path(args.udf_source))
        start = time.perf_counter()
        submitted = client.submit_job(
            submission_id=identity,
            entrypoint=shlex.join(command),
            runtime_env={
                "working_dir": str(package),
                "env_vars": {"NPA_RAY_APP_ADDRESS": args.app_address,
                             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            },
            metadata={"application": "ray-clip-development", "revision": revision,
                      "source_sha256": validation.canonical_hash(manifest)},
        )
        submit_done = time.perf_counter()
        if submitted != identity:
            raise ValueError("Jobs server returned an unexpected submission ID")
        # The SDK has finished uploading its source package before returning.
        observations = []
        stop_requested = False
        while True:
            info = client.get_job_info(identity)
            status = str(info.status.value if hasattr(info.status, "value") else info.status)
            if not observations or observations[-1]["status"] != status:
                observations.append({"status": status, "since_submit_seconds": time.perf_counter() - start})
            if status in {"SUCCEEDED", "FAILED", "STOPPED"}:
                break
            if cancel and not stop_requested and "RAY_CLIP_FIRST_CHECKPOINT" in client.get_job_logs(identity):
                stop_requested = client.stop_job(identity)
                if not stop_requested:
                    raise ValueError("Cancellation request did not find the owned running job")
            time.sleep(0.25)
        end = time.perf_counter()
        logs = client.get_job_logs(identity)
        (evidence / f"{identity}.log").write_text(logs)
        measurement = {
            "submission_id": identity, "revision": revision, "status": status,
            "output_path": job_output, "source_manifest": manifest,
            "source_packaging_upload_and_submit_seconds": submit_done - start,
            "total_iteration_seconds": end - start,
            "status_observations": observations,
            "server_start_time_ms": info.start_time, "server_end_time_ms": info.end_time,
            "runtime_env": info.runtime_env,
            "stop_requested": stop_requested,
            "report": None if cancel or status != "SUCCEEDED" else report_from_logs(logs),
        }
        validation.atomic_json(evidence / f"{identity}.json", measurement)
        expected = "STOPPED" if cancel else "SUCCEEDED"
        if status != expected:
            raise RuntimeError(f"Ray application ended {status}; expected {expected}; see saved job logs")
        if cancel and "RAY_CLIP_FIRST_CHECKPOINT" not in logs:
            raise ValueError("Cancelled job never performed its real GPU checkpoint")
        if not cancel:
            validation.verify_submitted_sources(measurement["report"], manifest)
        return measurement


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True, help="Loopback Jobs URL through an authenticated tunnel")
    parser.add_argument("--app-address", required=True, help="Explicit application GCS address; never SkyPilot management Ray")
    parser.add_argument("--source-dir", default=str(Path(__file__).parent))
    parser.add_argument("--udf-source", required=True, help="Explicit canonical Workbench bdd100k_udfs.py source file")
    parser.add_argument("--python", default="python", help="Application interpreter in the existing image")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-path", required=True, help="Absolute run-owned output directory on the job driver node")
    parser.add_argument("--evidence-dir", required=True, help="Private local client receipt directory")
    parser.add_argument("--records", type=int, default=2048)
    parser.add_argument("--actors", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--recovery-check", action="store_true")
    parser.add_argument("--cancel-check", action="store_true")
    args = parser.parse_args(argv)
    os.umask(0o077)
    from ray.job_submission import JobSubmissionClient

    with application_address(args.address):
        client = JobSubmissionClient(args.address)
        jobs = []
        baseline_path = None
        for revision in ("baseline", "changed", "restored"):
            job = run_job(client, args, Path(args.source_dir), revision, baseline_path=baseline_path)
            jobs.append(job)
            if baseline_path is None:
                baseline_path = job["output_path"]
        comparison = validation.compare_reports(*(job["report"] for job in jobs))
        cancellation = run_job(client, args, Path(args.source_dir), "baseline", cancel=True) if args.cancel_check else None
    receipt = {"jobs": jobs, "comparison": comparison, "cancellation": cancellation,
               "image_builds": 0, "model_reuse_scope": "batches within one actor; source jobs reload models"}
    validation.atomic_json(Path(args.evidence_dir) / "sequence.json", receipt)
    print(json.dumps({"completed_revisions": len(jobs), **comparison, "cancellation_verified": cancellation is not None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
