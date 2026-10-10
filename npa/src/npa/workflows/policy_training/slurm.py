"""Reconcile durable Slurm submissions and monitor exact jobs without a held exec stream."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import PurePosixPath
import re
import signal
import time
import uuid

from botocore.exceptions import ClientError

from npa.workbench.dataset import storage
from npa.workbench.storage_scope import authorize_uri
from .contracts import digest
from .diagnostics import _cleanup, _record
from .slurm_transport import _execute

_ACTIVE = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "RESIZING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "SIGNALING",
    "STAGE_OUT",
}
_FAILED = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


def _reserve(uri, receipt):
    target = authorize_uri(uri, operation="write")
    payload = json.dumps(receipt).encode()
    if target.kind == "s3":
        try:
            storage._s3_client().put_object(
                Bucket=target.bucket, Key=target.key, Body=payload, IfNoneMatch="*"
            )
            return True
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"PreconditionFailed", "412"}:
                raise
            return False
    target.local_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with os.fdopen(
            os.open(target.local_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
            "wb",
        ) as stream:
            stream.write(payload)
        return True
    except FileExistsError:
        return False


def _submission(settings, stage, request_uri):
    request = storage.read_json_uri(request_uri)
    uri = request_uri.rsplit("/", 1)[0] + "/submission.json"
    receipt = {
        "request_sha256": digest(request),
        "settings_sha256": digest(settings),
        "job_name": "npa-policy-" + uuid.uuid4().hex,
        "accounting_start": (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
            "%Y-%m-%d"
        ),
    }
    if not _reserve(uri, receipt):
        receipt = storage.read_json_uri(uri)
        _validate_receipt(receipt)
        if receipt["request_sha256"] != digest(request) or receipt[
            "settings_sha256"
        ] != digest(settings):
            raise ValueError(
                "submission identity changed; refusing to adopt or resubmit"
            )
        if not receipt.get("job_id"):
            receipt["job_id"] = _reconcile(settings, receipt, uri)
    else:
        receipt["job_id"] = _sbatch(settings, stage, receipt, request_uri, uri)
    storage.write_json_uri(uri, receipt)
    return receipt, uri


def _validate_receipt(receipt):
    if not re.fullmatch(r"npa-policy-[a-f0-9]{32}", receipt.get("job_name", "")):
        raise ValueError("invalid saved Slurm job name")
    if "job_id" in receipt and not re.fullmatch(r"[1-9][0-9]*", str(receipt["job_id"])):
        raise ValueError("invalid saved Slurm job ID")
    datetime.strptime(receipt["accounting_start"], "%Y-%m-%d")


def _sbatch(settings, stage, receipt, request_uri, uri):
    output = _execute(
        settings,
        [
            "sbatch",
            "--parsable",
            f"--job-name={receipt['job_name']}",
            settings["scripts"][stage],
            "--request-uri",
            request_uri,
        ],
        uri,
    )
    if not re.fullmatch(r"[1-9][0-9]*", output):
        raise RuntimeError(
            "ambiguous Slurm submission; reconcile the saved receipt before retrying"
        )
    return output


def _reconcile(settings, receipt, uri):
    name = receipt["job_name"]
    queued = _execute(
        settings, ["squeue", "--noheader", f"--name={name}", "--format=%i|%j"], uri
    )
    accounted = _execute(
        settings,
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            f"--name={name}",
            f"--starttime={receipt['accounting_start']}",
            "--format=JobIDRaw,JobName%80",
        ],
        uri,
    )
    jobs = set()
    for line in (queued + "\n" + accounted).splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("|")
        if (
            len(fields) != 2
            or fields[1] != name
            or not re.fullmatch(r"[1-9][0-9]*", fields[0])
        ):
            raise RuntimeError("invalid Slurm reconciliation evidence")
        jobs.add(fields[0])
    if len(jobs) != 1:
        raise RuntimeError(
            "submission is unresolved; no job was resubmitted or cancelled"
        )
    return jobs.pop()


def _state(settings, receipt, uri):
    job = receipt["job_id"]
    queued = _execute(
        settings,
        ["squeue", "--noheader", f"--name={receipt['job_name']}", "--format=%i|%j|%T"],
        uri,
    )
    if queued:
        fields = queued.split("|")
        if len(fields) != 3 or fields[:2] != [job, receipt["job_name"]]:
            raise RuntimeError("Slurm queue returned a different job identity")
        if fields[2] in _ACTIVE:
            return False
        if fields[2] not in _FAILED | {"COMPLETED"}:
            raise RuntimeError(
                "unknown Slurm queue state; job retained for reconciliation"
            )
    return _accounted_state(settings, receipt, uri)


def _accounted_state(settings, receipt, uri):
    job = receipt["job_id"]
    output = _execute(
        settings,
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            f"--jobs={job}",
            f"--starttime={receipt['accounting_start']}",
            "--format=JobIDRaw,JobName%80,State%40,ExitCode",
        ],
        uri,
    )
    if not output:
        return False  # Accounting can lag the queue; absence is not completion.
    fields = output.split("|")
    if len(fields) != 4 or fields[:2] != [job, receipt["job_name"]]:
        raise RuntimeError("Slurm accounting returned a different job identity")
    _record(uri, {"accounting": output})
    state = fields[2].split()[0] if fields[2].strip() else "UNKNOWN"
    if state == "COMPLETED" and fields[3] == "0:0":
        return True
    if state in _FAILED or state == "COMPLETED":
        raise RuntimeError(
            "Slurm job failed; inspect private accounting and worker evidence"
        )
    if state not in _ACTIVE:
        raise RuntimeError("unknown Slurm state; job retained for reconciliation")
    return False


def _terminate(signum, frame):
    raise KeyboardInterrupt("Slurm stage interrupted")


def _submit_slurm(settings, stage, request_uri):
    script = settings["scripts"][stage]
    if not isinstance(script, str) or not PurePosixPath(script).is_absolute():
        raise ValueError("batch script must be an absolute worker-readable path")
    receipt, uri = _submission(settings, stage, request_uri)
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        while not _state(settings, receipt, uri):
            time.sleep(5)
    except KeyboardInterrupt:
        _cleanup(lambda: _execute(settings, ["scancel", receipt["job_id"]], uri), uri)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
