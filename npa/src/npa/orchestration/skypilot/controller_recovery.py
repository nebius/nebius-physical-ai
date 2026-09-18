"""Recover one controller-owned workflow without trusting an unrelated client queue."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import uuid

from npa.orchestration.skypilot.workflow_state import get_json, resolve_workflow_s3_config
from botocore.exceptions import ClientError


class ControllerRecoveryError(RuntimeError):
    """Original execution ownership could not be proven without mutation."""


def _require(condition, reason):
    if not condition:
        raise ControllerRecoveryError(reason)


def _private_record(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor) as stream:
        metadata = os.fstat(stream.fileno())
        _require(stat.S_ISREG(metadata.st_mode), "recovery record must be a regular file")
        _require(metadata.st_uid == os.getuid() and metadata.st_mode & 0o077 == 0,
                 "recovery record must be owner-only")
        record = json.load(stream)
    _require(record.get("schema") == "npa.workflow.controller-recovery.v1", "unsupported recovery record")
    for key in ("run_id", "project", "workflow_s3_uri", "context", "namespace", "controller",
                "controller_uid", "container", "user_hash", "workspace", "name", "attempt_id"):
        _require(isinstance(record.get(key), str) and record[key].strip(), "incomplete execution identity")
    _require(type(record.get("job_id")) is int and record["job_id"] > 0, "invalid job ID")
    _require(re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", record.get("image", "")), "image must be immutable")
    for key in ("dag_yaml_content_sha256", "config_file_content_sha256"):
        _require(re.fullmatch(r"[a-f0-9]{64}", record.get(key, "")), "missing original configuration digest")
    timestamp = record.get("submitted_at")
    _require(type(timestamp) in (int, float) and math.isfinite(timestamp) and timestamp > 0,
             "missing original submission timestamp")
    return record


def verify_ledger(record, manifest, runtime):
    """Bind recovery to exactly one durable NPA wave and immutable launch attempt.

    Args:
        record: Private original controller identity.
        manifest: Independently fetched durable run manifest.
        runtime: Independently fetched durable runtime ledger.
    Returns:
        None.
    Raises:
        ControllerRecoveryError: Missing, conflicting or ambiguous durable binding.
    """
    _require(manifest.get("run_id") == runtime.get("run_id") == record["run_id"], "durable run identity differs")
    waves = runtime.get("waves", [])
    matches = [wave for wave in waves if str(wave.get("job_id")) == str(record["job_id"])]
    _require(len(matches) == 1, "durable managed-job identity is ambiguous")
    wave = matches[0]
    _require(wave.get("job_name") == record["name"], "durable job name differs")
    _require(wave.get("logical_launch_id") == record["attempt_id"], "durable launch attempt differs")
    _require(wave.get("launch_sequence", 0) > 0, "durable ledger does not prove a launch")


def _read_ledger(record):
    state = resolve_workflow_s3_config(
        run_id=record["run_id"], project=record["project"],
        workflow_s3_uri=record["workflow_s3_uri"],
    )
    verify_ledger(record, get_json(state, "manifest.json"), get_json(state, "runtime.json"))
    return state


def _command(argv, *, source=None):
    completed = subprocess.run(argv, input=source, text=True, capture_output=True, check=False)
    _require(completed.returncode == 0, "controller probe failed; no absence may be inferred")
    try:
        return json.loads(completed.stdout)
    except ValueError:
        raise ControllerRecoveryError("controller probe did not return structured evidence") from None


def _inventory(record):
    return _command(["kubectl", "--context", record["context"], "get", "pods", "-A", "-o", "json"])["items"]


def verify_pods(record, pods):
    """Check immutable controller UID and every exact-run worker before acting.

    Args:
        record: Private original controller and job binding.
        pods: Fresh Kubernetes pod metadata from the exact context.
    Returns:
        Count of nonterminal exact-run workers.
    Raises:
        ControllerRecoveryError: Any controller or worker identity is uncertain.
    """
    controllers = [p for p in pods if p["metadata"]["name"] == record["controller"]
                   and p["metadata"]["namespace"] == record["namespace"]]
    _require(len(controllers) == 1, "original controller is absent or ambiguous")
    controller = controllers[0]
    _require(controller["metadata"]["uid"] == record["controller_uid"], "controller was replaced")
    _require(controller["status"]["phase"] == "Running", "controller is not running")
    _require(record["container"] in [c["name"] for c in controller["spec"]["containers"]], "controller container differs")
    workers = [p for p in pods if p["metadata"].get("annotations", {}).get("skypilot-managed-job-name") == record["name"]]
    for worker in workers:
        _require(worker["metadata"]["namespace"] == record["namespace"], "worker namespace differs")
        _require(worker["metadata"]["annotations"].get("skypilot-managed-job-id") == str(record["job_id"]), "worker job ID differs")
        _require(worker["spec"]["containers"][0]["image"] == record["image"], "worker immutable image differs")
        identities = worker["status"].get("containerStatuses", [])
        _require(bool(identities), "worker runtime image identity is missing")
        _require(identities[0].get("imageID", "").endswith(record["image"].split("@", 1)[1]),
                 "worker runtime image digest differs")
    return sum(p["status"]["phase"] not in {"Succeeded", "Failed"} for p in workers)


def _probe(record, *, cancel=False):
    source = Path(__file__).with_name("controller_recovery_probe.py").read_text()
    launcher = (
        "import os,pathlib,sys; "
        "p=str(pathlib.Path.home()/'skypilot-runtime/bin/python'); "
        "os.execv(p,[p,'-c',sys.argv[1]])"
    )
    argv = ["kubectl", "--context", record["context"], "exec", "-i", "-n", record["namespace"],
            record["controller"], "-c", record["container"], "--", "python3", "-c", launcher, source]
    return _command(argv, source=json.dumps({**record, "cancel": cancel}))


def _save_evidence(directory, record, observation):
    _require(not directory.is_symlink(), "diagnostic directory must not be linked")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = directory.stat()
    _require(metadata.st_uid == os.getuid() and metadata.st_mode & 0o077 == 0, "diagnostics must be owner-only")
    binding = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    path = directory / (uuid.uuid4().hex + ".json")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"binding_sha256": binding, **observation}, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    return binding


def _save_durable(state, binding, observation):
    body = json.dumps({"binding_sha256": binding, **observation}, sort_keys=True).encode()
    digest = hashlib.sha256(body).hexdigest()
    key = f"{state.prefix}/controller-recovery/{binding}/{digest}.json"
    client = state.client()
    try:
        client.put_object(Bucket=state.bucket, Key=key, Body=body,
                          ContentType="application/json", IfNoneMatch="*")
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {"PreconditionFailed", "412"}:
            raise
    actual = client.get_object(Bucket=state.bucket, Key=key)["Body"].read()
    _require(actual == body, "durable diagnostic read-back differs")


def reconcile_controller(record_path: Path, *, cancel: bool = False) -> dict:
    """Verify or cancel only an orphaned run on its exclusive original controller.

    Args:
        record_path: Owner-only original execution binding, retained after client loss.
        cancel: Request native exact-ID cancellation after independent verification.
    Returns:
        Sanitized observation; terminal verification also requires zero active workers.
    Raises:
        ControllerRecoveryError: Ambiguity, ownership mismatch or unavailable evidence.
        OSError: Private diagnostics could not be durably retained before mutation.
    """
    record = _private_record(record_path)
    state = _read_ledger(record)
    workers = verify_pods(record, _inventory(record))
    observation = {**_probe(record), "active_workers": workers}
    directory = record_path.parent / (record_path.stem + "-observations")
    binding = _save_evidence(directory, record, observation)
    terminal = observation["status"] in {"SUCCEEDED", "FAILED", "FAILED_SETUP", "CANCELLED"}
    if cancel:
        _save_durable(state, binding, {**observation, "cleanup_verified": terminal and workers == 0})
    if cancel and not terminal:
        # Recheck both external identity surfaces immediately before the native call.
        _read_ledger(record)
        verify_pods(record, _inventory(record))
        observation = {**_probe(record, cancel=True), "active_workers": workers}
        _save_evidence(directory, record, observation)
        _save_durable(state, binding, observation)
    return {**observation, "binding_sha256": binding,
            "cleanup_verified": terminal and workers == 0}
