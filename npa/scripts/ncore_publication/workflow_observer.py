"""Observe exact live NRE workflow jobs and retain final status evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from .process import ROOT, committed_source


_STAGES = ("reconstruct", "render")
_TERMINAL_FAILURES = {
    "CANCELLED",
    "FAILED",
    "FAILED_CONTROLLER",
    "FAILED_STARTUP",
    "TIMED_OUT",
}


def _write_private(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _status(
    *,
    run_id: str,
    workflow_s3_uri: str,
    project: str,
    sky_bin: str,
    runner: Callable[..., subprocess.CompletedProcess[bytes]],
) -> tuple[dict[str, Any], bytes]:
    command = [
        str(ROOT / "npa/.venv/bin/python"),
        "-m",
        "npa.cli.main",
        "workbench",
        "workflow",
        "status",
        run_id,
        "--workflow-s3-uri",
        workflow_s3_uri,
        "--sky-bin",
        sky_bin,
        "--output-format",
        "json",
    ]
    if project:
        command.extend(["--project", project])
    result = runner(
        command,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=None,
        cwd=ROOT,
    )
    if result.returncode != 0:
        raise ValueError("live workflow status command failed")
    try:
        payload = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("live workflow status is invalid") from exc
    if not isinstance(payload, dict) or payload.get("run_id") != run_id:
        raise ValueError("live workflow status run identity differs")
    return payload, result.stdout


def _stage(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    stages = payload.get("stages")
    if not isinstance(stages, dict):
        return None
    matches = [
        value
        for value in stages.values()
        if isinstance(value, dict) and value.get("workflow_state") == name
    ]
    if len(matches) > 1:
        raise ValueError("live workflow status has ambiguous stage identity")
    return matches[0] if matches else None


def observe_workflow(
    *,
    source_sha: str,
    run_id: str,
    workflow_s3_uri: str,
    project: str,
    sky_bin: str,
    context: str,
    namespace: str,
    expected_image: str,
    evidence_dir: Path,
    poll_seconds: float,
    status_runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    observer: Callable[..., dict[str, Any]] | None = None,
    bundler: Callable[..., dict[str, Any]] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Observe each GPU stage while live, then retain exact terminal status."""
    if observer is None or bundler is None:
        from npa.workbench.nurec.runtime_attestation import (
            bundle_runtime_attestations,
            observe_kubernetes_stage,
        )

        observer = observer or observe_kubernetes_stage
        bundler = bundler or bundle_runtime_attestations
    if poll_seconds <= 0:
        raise ValueError("workflow observation poll interval is invalid")
    if (
        evidence_dir.is_symlink()
        or not evidence_dir.is_dir()
        or evidence_dir.stat().st_uid != os.getuid()
        or evidence_dir.stat().st_mode & 0o077
    ):
        raise ValueError("workflow evidence directory is not private")
    observed: dict[str, dict[str, Any]] = {}
    while len(observed) != len(_STAGES):
        committed_source(source_sha)
        payload, raw = _status(
            run_id=run_id,
            workflow_s3_uri=workflow_s3_uri,
            project=project,
            sky_bin=sky_bin,
            runner=status_runner,
        )
        overall = str(payload.get("status") or "").upper()
        if overall in _TERMINAL_FAILURES:
            raise ValueError("workflow failed before runtime observation completed")
        for name in _STAGES:
            if name in observed:
                continue
            selected = _stage(payload, name)
            if not selected or selected.get("state") != "RUNNING":
                continue
            job_name = str(selected.get("job_name") or "")
            job_id = str(selected.get("managed_job_id") or "")
            snapshot = evidence_dir / f"workflow-status-{name}.json"
            _write_private(snapshot, raw)
            receipt = evidence_dir / f"runtime-{name}.json"
            observed[name] = observer(
                stage=name,
                pod_name="",
                namespace=namespace,
                container_name="ray-node",
                expected_image=expected_image,
                managed_job_name=job_name,
                managed_job_id=job_id,
                workflow_run_id=run_id,
                workflow_status_path=snapshot,
                output_path=receipt,
                context=context,
                max_wait_seconds=0,
                poll_seconds=poll_seconds,
                kubectl_bin="kubectl",
            )
        if len(observed) != len(_STAGES):
            sleeper(poll_seconds)
    runtime_path = evidence_dir / "nre-runtime.json"
    runtime = bundler(
        reconstruct_path=evidence_dir / "runtime-reconstruct.json",
        render_path=evidence_dir / "runtime-render.json",
        output_path=runtime_path,
    )
    while True:
        committed_source(source_sha)
        payload, raw = _status(
            run_id=run_id,
            workflow_s3_uri=workflow_s3_uri,
            project=project,
            sky_bin=sky_bin,
            runner=status_runner,
        )
        overall = str(payload.get("status") or "").upper()
        if overall == "SUCCEEDED":
            _write_private(evidence_dir / "workflow-status.json", raw)
            return {
                "status": "pass",
                "run_id": run_id,
                "runtime": runtime,
                "observed_stages": list(_STAGES),
            }
        if overall in _TERMINAL_FAILURES:
            raise ValueError("workflow terminal status is not successful")
        sleeper(poll_seconds)
