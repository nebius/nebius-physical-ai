"""Persist a Ray development session's artifacts before signaling its finish."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile
import time
import uuid

import finish_worker
from submit import application_address
from validation import atomic_json


def parse_receipt(logs: str) -> dict:
    marker = "RAY_CLIP_ARTIFACTS "
    receipts = [json.loads(line.split(marker, 1)[1]) for line in logs.splitlines() if marker in line]
    if len(receipts) != 1 or not receipts[0].get("all_objects_read_after_write_verified"):
        raise ValueError("Upload job did not emit exactly one verified artifact receipt")
    return receipts[0]


def finish(client, storage, args) -> dict:
    # Validate routing before uploading anything. The stop marker never enters the worker job.
    artifact_bucket, artifact_prefix = finish_worker.parse_s3(args.artifact_uri)
    stop_bucket, stop_key = finish_worker.parse_s3(args.stop_uri)
    if artifact_bucket == stop_bucket and (stop_key == artifact_prefix or stop_key.startswith(artifact_prefix + "/")):
        raise ValueError("Keep the stop marker outside the artifact prefix")
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    submission_id = f"clip-finish-{uuid.uuid4().hex}"
    source = Path(__file__).with_name("finish_worker.py")
    if source.is_symlink() or not source.is_file():
        raise ValueError("The upload worker must be a reviewed regular source file")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ray-clip-finish-source-") as temporary:
        package = Path(temporary)
        content = source.read_bytes()
        (package / "finish_worker.py").write_bytes(content)
        worker_hash = hashlib.sha256(content).hexdigest()
        actual = client.submit_job(
            submission_id=submission_id,
            entrypoint=shlex.join([args.python, "finish_worker.py", "--output-path", args.output_path,
                                  "--artifact-uri", args.artifact_uri]),
            runtime_env={"working_dir": str(package)},
            metadata={"application": "ray-clip-finish", "source_sha256": worker_hash},
        )
        if actual != submission_id:
            raise ValueError("Unexpected upload submission identity")
        submitted_at = time.perf_counter()
        statuses = []
        while True:
            info = client.get_job_info(submission_id)
            status = str(info.status.value if hasattr(info.status, "value") else info.status)
            if not statuses or statuses[-1]["status"] != status:
                statuses.append({"status": status, "seconds": time.perf_counter() - started})
            if status in {"SUCCEEDED", "FAILED", "STOPPED"}:
                break
            time.sleep(0.25)
        logs = client.get_job_logs(submission_id)
        (evidence / f"{submission_id}.log").write_text(logs)
        result = {"submission_id": submission_id, "status": status, "statuses": statuses,
                  "worker_source_sha256": worker_hash, "stop_marker_written": False,
                  "source_packaging_upload_and_submit_seconds": submitted_at - started,
                  "upload_job_seconds": time.perf_counter() - submitted_at}
        atomic_json(evidence / "finish.json", result)
        if status != "SUCCEEDED":
            raise RuntimeError(f"Artifact upload job ended {status}; session finish was not signaled")
        receipt = parse_receipt(logs)
        expected_uri = f"s3://{artifact_bucket}/{artifact_prefix}/manifest.json"
        if receipt["manifest_uri"] != expected_uri:
            raise ValueError("Artifact receipt returned an unexpected manifest location")
        finish_worker.verify_object(storage, artifact_bucket, f"{artifact_prefix}/manifest.json",
                                    receipt["manifest_sha256"], receipt["manifest_bytes"])
        result["artifacts"] = receipt
        atomic_json(evidence / "finish.json", result)
        marker = {"finished": True, "manifest_sha256": receipt["manifest_sha256"]}
        payload = (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode()
        digest = hashlib.sha256(payload).hexdigest()
        # This runs on the operator client only, after the upload job is terminal.
        result["stop_marker_write_attempted"] = True
        atomic_json(evidence / "finish.json", result)
        try:
            finish_worker.put_immutable(storage, stop_bucket, stop_key, payload, digest, len(payload))
        except Exception as exc:
            # A failed response/readback cannot prove whether S3 accepted a PUT.
            result.update({"stop_marker_written": None, "stop_marker_read_after_write_verified": False,
                           "stop_marker_error_type": type(exc).__name__})
            atomic_json(evidence / "finish.json", result)
            raise
        result.update({"stop_marker_written": True, "stop_marker_sha256": digest,
                       "stop_marker_read_after_write_verified": True,
                       "finish_seconds": time.perf_counter() - started})
        atomic_json(evidence / "finish.json", result)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--python", required=True, help="Absolute application interpreter in the image")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--artifact-uri", required=True)
    parser.add_argument("--stop-uri", required=True)
    parser.add_argument("--evidence-dir", required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    from ray.job_submission import JobSubmissionClient

    with application_address(args.address):
        result = finish(JobSubmissionClient(args.address), finish_worker.s3_client(), args)
    print(json.dumps({"artifact_files": result["artifacts"]["file_count"],
                      "artifact_bytes": result["artifacts"]["total_bytes"],
                      "stop_marker_verified": result["stop_marker_read_after_write_verified"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
