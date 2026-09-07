"""Opt-in native Jobs/Train CUDA execution on an already preflighted owned runtime.

NPA_RAY_TRAIN_LIVE_CONFIG is an owner-only JSON file with address (loopback
Jobs tunnel), storage_uri (authorized S3 prefix), and evidence_dir. The hosting
services and test process must already have the verified workload AWS environment.
The test creates/cancels only its unique native Jobs; the operator owns hosting
task, cloud cleanup, provider placement evidence and post-teardown S3 readback.
"""

import json
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from urllib.parse import urlsplit
import uuid

import pytest


pytestmark = pytest.mark.e2e
EXAMPLE = Path(__file__).parents[2] / "workflows/workbench/ray-train-synthetic"


def _write_private(path, content):
    """Create evidence with owner-only access before writing its first byte."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(content)


def _stop_owned_job(client, job):
    """Attempt an exact-ID stop even when the initial status request fails."""
    try:
        status = client.get_job_status(job)
    except Exception:
        status = None
    if status is not None and status.is_terminal():
        return
    client.stop_job(job)
    while not client.get_job_status(job).is_terminal():
        time.sleep(1)


def _cancel_owned_jobs(client, owned, evidence):
    """Reconcile exact supplied IDs and attempt every cleanup even after API failures."""
    errors = []
    for job in owned:
        try:
            _stop_owned_job(client, job)
        except Exception as exc:
            errors.append({"job": job, "phase": "cancel", "error": type(exc).__name__})
        try:
            _write_private(evidence / f"{job}.log", client.get_job_logs(job))
        except Exception as exc:
            errors.append({"job": job, "phase": "logs", "error": type(exc).__name__})
    _write_private(evidence / "cleanup.json", json.dumps({"owned": owned, "errors": errors}))
    return errors


def _live_configuration():
    """Validate the explicit private endpoint before importing any native client."""
    if not os.environ.get("NPA_RAY_TRAIN_LIVE_CONFIG"):
        pytest.skip("Explicit preflighted Ray Train runtime required")
    path = Path(os.environ["NPA_RAY_TRAIN_LIVE_CONFIG"])
    assert path.stat().st_mode & 0o077 == 0, "Live configuration must be owner-only"
    config = json.loads(path.read_text())
    endpoint = urlsplit(config["address"])
    assert endpoint.hostname in {"127.0.0.1", "localhost"} and endpoint.scheme == "http"
    assert not endpoint.username and not endpoint.password and not endpoint.query
    if any(os.environ.get(name) for name in ("RAY_ADDRESS", "RAY_API_SERVER_ADDRESS")):
        raise ValueError("Unset RAY_ADDRESS and RAY_API_SERVER_ADDRESS before using the selected Jobs endpoint")
    evidence = Path(config["evidence_dir"])
    evidence.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert evidence.stat().st_mode & 0o077 == 0, "Evidence directory must be owner-only"
    return config, evidence


class _LiveRun:
    """Keep native job ownership and private evidence bound to one validated endpoint."""

    def __init__(self, config, evidence):
        from ray.job_submission import JobSubmissionClient

        self.config = config
        self.evidence = evidence
        self.client = JobSubmissionClient(config["address"])
        self.suffix = uuid.uuid4().hex[:12]
        self.owned = []

    def submit(self, kind, *extra):
        """Record the exact intent before a submission response can be lost."""
        name = f"train-{kind}-{self.suffix}"
        command = shlex.join([
            "/opt/npa-ray-train/env/bin/python", "train.py", "--storage-path", self.config["storage_uri"],
            "--run-name", name, "--output-dir", f"/opt/npa-ray-train/exports/{name}", *extra,
        ])
        self.owned.append(name)
        _write_private(self.evidence / "submission-intents.json", json.dumps(self.owned))
        job = self.client.submit_job(submission_id=name, entrypoint=command,
                                     runtime_env={"working_dir": str(EXAMPLE)})
        assert job == name
        return job

    def preserve(self, job):
        """Retain native diagnostics and reject detached cleanup-thread failures."""
        content = self.client.get_job_logs(job)
        _write_private(self.evidence / f"{job}.log", content)
        _write_private(self.evidence / f"{job}-native-status.json", self.client.get_job_info(job).json())
        assert "Exception in thread PlacementGroupCleanerMonitor" not in content


def _validate_rank_runtimes(journal):
    """Require observed ranks on distinct B200 hosts with the qualified runtime."""
    for row in journal:
        assert len({rank["node_fingerprint"] for rank in row["ranks"]}) == 2
        assert all("B200" in rank["device_name"] for rank in row["ranks"])
        assert all(rank["torch_version"] == "2.13.0+cu130" for rank in row["ranks"])
        assert all(rank["cuda_version"] == "13.0" for rank in row["ranks"])
        assert all(rank["ray_version"] == "2.58.0" for rank in row["ranks"])


def _validate_exported_state(destination, kind):
    """Recompute checkpoint and held-out evidence from safely reloaded native state."""
    import torch

    state = torch.load(destination / "state.pt", map_location="cpu", weights_only=True)
    assert state["step"] == 32 and len(state["journal"]) == 32
    assert state["optimizer"]["state"]
    report = json.loads((destination / "result.json").read_text())
    model = torch.nn.Linear(8, 1)
    model.load_state_dict(state["model"])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.8)
    optimizer.load_state_dict(state["optimizer"])
    assert set(optimizer.state) == set(model.parameters())
    for parameter in model.parameters():
        momentum = optimizer.state[parameter]["momentum_buffer"]
        assert momentum.shape == parameter.shape and torch.isfinite(momentum).all()
    assert optimizer.param_groups[0]["lr"] == 0.1 and optimizer.param_groups[0]["momentum"] == 0.8
    parameters = torch.cat([value.detach().flatten() for value in model.parameters()])
    assert hashlib.sha256(parameters.numpy().tobytes()).hexdigest() == report["parameter_sha256"]
    target = torch.arange(1, 9).reshape(8, 1) / 8 + 0.25
    assert torch.nn.functional.mse_loss(model(torch.eye(8)), target).item() == report["held_out_loss"]
    assert report["held_out_loss"] < report["baseline_held_out_loss"]
    assert report["final_loss"] < report["initial_loss"]
    _validate_rank_runtimes(state["journal"])
    if kind == "recovery":
        assert report["recovery_steps"] == [0, 8]
    return report


def _completed_export(run, kind, extra):
    """Require a successful native job and independently inspect its downloaded export."""
    from ray.job_submission import JobStatus

    job = run.submit(kind, *extra)
    while not run.client.get_job_status(job).is_terminal():
        time.sleep(1)
    run.preserve(job)
    assert run.client.get_job_status(job) == JobStatus.SUCCEEDED
    destination = run.evidence / job
    inspected = subprocess.run([
        sys.executable, str(EXAMPLE / "inspect_results.py"), str(destination),
        "--download", run.config["storage_uri"].rstrip("/") + f"/{job}/exports",
    ], capture_output=True, text=True)
    _write_private(run.evidence / f"{job}-inspection.txt", inspected.stdout + inspected.stderr)
    assert inspected.returncode == 0, "Private artifact inspection failed"
    return _validate_exported_state(destination, kind)


def _cancel_resource_identity(run, job):
    """Capture live placement and cleanup ownership before requesting cancellation."""
    from ray.util.state import list_actors, list_placement_groups

    info = run.client.get_job_info(job)
    assert info.submission_id == job
    driver_id = info.job_id
    assert driver_id, "Native driver identity required for cleanup evidence"
    groups = list_placement_groups(address=run.config["address"], detail=True,
                                   filters=[("creator_job_id", "=", driver_id)],
                                   raise_on_missing_output=True)
    cleaners = list_actors(address=run.config["address"], detail=True,
                           filters=[("job_id", "=", driver_id), ("class_name", "=", "PlacementGroupCleaner")],
                           raise_on_missing_output=True)
    assert len(groups) == 1 and groups[0].state == "CREATED"
    assert len(cleaners) == 1 and cleaners[0].state == "ALIVE"
    _write_private(run.evidence / "cancel-resources-before.json", json.dumps({
        "groups": [group.asdict() for group in groups],
        "cleaners": [actor.asdict() for actor in cleaners],
    }))
    return driver_id, groups, cleaners


def _wait_for_cancel_resources(run, driver_id, groups, cleaners):
    """Require terminal states for the exact captured native resources."""
    from ray.util.state import get_actor, get_placement_group

    for group in groups:
        while True:
            current = get_placement_group(group.placement_group_id, address=run.config["address"])
            assert current is not None, "Captured placement group state unavailable"
            assert current.placement_group_id == group.placement_group_id
            assert current.creator_job_id == driver_id
            if current.state == "REMOVED":
                break
            time.sleep(1)
    for actor in cleaners:
        while True:
            current = get_actor(actor.actor_id, address=run.config["address"])
            assert current is not None, "Captured cleanup actor state unavailable"
            assert current.actor_id == actor.actor_id and current.job_id == driver_id
            if current.state == "DEAD":
                break
            time.sleep(1)
    _write_private(run.evidence / "cancel-resources-after.json", json.dumps({
        "placement_groups_removed": len(groups), "cleanup_actors_dead": len(cleaners),
    }))


def _cancel_active_training(run):
    """Cancel only after the actual workload has uploaded its first checkpoint."""
    from ray.job_submission import JobStatus

    job = run.submit("cancel", "--steps", "1000000")
    while "optimizer_step=4 checkpoint_uploaded" not in run.client.get_job_logs(job):
        assert not run.client.get_job_status(job).is_terminal(), "Training ended before cancellation exercise"
        time.sleep(1)
    driver_id, groups, cleaners = _cancel_resource_identity(run, job)
    assert run.client.stop_job(job)
    while not run.client.get_job_status(job).is_terminal():
        time.sleep(1)
    run.preserve(job)
    assert run.client.get_job_status(job) == JobStatus.STOPPED
    _wait_for_cancel_resources(run, driver_id, groups, cleaners)


def test_native_train_cuda_recovery_artifacts_and_cancel():
    """Execute distributed CUDA recovery, factual exports, and active cancellation.

    Args:
        None; uses the explicitly supplied private live configuration.
    Returns:
        None.
    Raises:
        AssertionError: Native execution, artifacts, placement, or cleanup differs.
        ValueError: Inherited discovery conflicts with the selected endpoint.
    """
    config, evidence = _live_configuration()
    run = _LiveRun(config, evidence)
    results = {}
    try:
        for kind, extra in (("baseline", []), ("recovery", ["--fail-after-step", "8"])):
            results[kind] = _completed_export(run, kind, extra)
        assert results["baseline"]["parameter_sha256"] == results["recovery"]["parameter_sha256"]
        _cancel_active_training(run)
    finally:
        # Stop only this test's IDs, including when an assertion fails mid-training.
        errors = _cancel_owned_jobs(run.client, run.owned, evidence)
        assert not errors, "Owned native Jobs cleanup incomplete; private evidence retained"
    _write_private(evidence / "validation.json", json.dumps({
        "success": True, "jobs": len(run.owned), "optimizer_steps": 64, "cuda_ranks": 2,
        "native_worker_recovery": True, "cancel_status": "STOPPED",
        "cancel_placement_group_removed": True, "cancel_cleanup_actor_dead": True,
    }))
