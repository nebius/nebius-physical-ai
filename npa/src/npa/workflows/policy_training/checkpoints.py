"""Publish completed checkpoint events and submit idempotent Slurm evaluations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from .contracts import digest


_REQUIRED_STATE = {"model", "optimizer", "scheduler", "rng", "sampler"}


def publish_checkpoint(
    directory: Path, step: int, state_files: dict, events: Path
) -> Path:
    """Publish a trainer-completed checkpoint after hashing its recovery state.

    Args:
        directory: Immutable shared checkpoint directory after the trainer's save barrier.
        step: Completed optimizer step.
        state_files: Recovery role to relative file-list mapping.
        events: Run-scoped directory on the shared filesystem.
    Returns:
        Atomic ready-event path, safe to consume after this function returns.
    Raises:
        ValueError: The step, recovery roles or checkpoint paths are invalid.
        OSError: Checkpoint bytes cannot be read or the event cannot be persisted.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("checkpoint step must be a positive integer")
    if set(state_files) != _REQUIRED_STATE:
        raise ValueError(
            "checkpoint must include model, optimizer, scheduler, RNG and sampler state"
        )
    root = directory.resolve(strict=True)
    records = {role: _hash_files(root, names) for role, names in state_files.items()}
    payload = {
        "schema": "npa.policy.checkpoint-ready.v1",
        "step": step,
        "checkpoint_path": str(root),
        "state_files": records,
    }
    payload["checkpoint_sha256"] = digest(records)
    events.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = events / f"step-{step}-{payload['checkpoint_sha256']}.json"
    _atomic_json(path, payload)
    return path


def _hash_files(root, names):
    if not isinstance(names, list) or not names:
        raise ValueError("every recovery role needs at least one saved file")
    result = []
    for name in names:
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError("checkpoint filenames must be relative")
        path = (root / name).resolve(strict=True)
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.stat().st_size == 0
        ):
            raise ValueError(
                "checkpoint state must be a nonempty file inside its directory"
            )
        with path.open("rb") as stream:
            checksum = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        result.append(
            {"path": name, "sha256": checksum.hexdigest(), "bytes": path.stat().st_size}
        )
    return result


def _atomic_json(path, payload):
    temporary = path.with_suffix(".tmp-" + str(os.getpid()))
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def enqueue_evaluation(event_path: Path, script: Path, ledger: Path) -> dict:
    """Submit one evaluation per event and retain ambiguous submissions for reconciliation.

    Args:
        event_path: Completed checkpoint event on the Soperator shared filesystem.
        script: Absolute shared evaluation sbatch script accepting --checkpoint-event.
        ledger: Run-scoped submission receipt directory.
    Returns:
        Slurm job receipt, or the existing receipt for an already submitted event.
    Raises:
        ValueError: Event bytes, schema or script location are invalid.
        RuntimeError: A prior submission is ambiguous or Slurm rejects submission.
    """
    event = _verify_event(event_path)
    if not script.is_absolute() or not script.is_file():
        raise ValueError("evaluation script must be an existing absolute shared path")
    ledger.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipt = ledger / (digest(event) + ".json")
    pending = receipt.with_suffix(".pending")
    if receipt.exists():
        return json.loads(receipt.read_text())
    try:
        pending.touch(exist_ok=False, mode=0o600)
    except FileExistsError:
        raise RuntimeError(
            "evaluation submission is ambiguous; reconcile the named Slurm job"
        ) from None
    return _submit_event(event_path, script, receipt, pending, event)


def _verify_event(path):
    event = json.loads(path.read_text())
    if event.get("schema") != "npa.policy.checkpoint-ready.v1":
        raise ValueError("expected a completed checkpoint event")
    records = event["state_files"]
    if set(records) != _REQUIRED_STATE:
        raise ValueError("checkpoint recovery state is incomplete")
    root = Path(event["checkpoint_path"]).resolve(strict=True)
    actual = {
        role: _hash_files(root, [item["path"] for item in files])
        for role, files in records.items()
    }
    if actual != records or digest(records) != event["checkpoint_sha256"]:
        raise ValueError("checkpoint bytes changed after publication")
    return event


def _submit_event(event_path, script, receipt, pending, event):
    name = "npa-eval-" + digest(event)[:24]
    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            "--job-name=" + name,
            str(script),
            "--checkpoint-event",
            str(event_path.resolve()),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if result.returncode or not re.fullmatch(r"[0-9]+", job_id):
        raise RuntimeError(
            "Slurm evaluation submission failed; reconcile the pending receipt"
        )
    payload = {
        "job_id": job_id,
        "job_name": name,
        "checkpoint_sha256": event["checkpoint_sha256"],
    }
    _atomic_json(receipt, payload)
    pending.unlink()
    return payload
