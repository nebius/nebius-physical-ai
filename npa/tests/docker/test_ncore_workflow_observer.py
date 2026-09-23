from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from ncore_publication import workflow_observer  # noqa: E402


def _payload(status: str, stages: dict) -> bytes:
    return json.dumps({"run_id": "run-1", "status": status, "stages": stages}).encode()


def _stage(name: str, state: str, job_id: str) -> dict:
    return {
        "workflow_state": name,
        "state": state,
        "job_name": f"run-1-{name}",
        "managed_job_id": job_id,
        "managed_job_attempts": [{"job_name": f"run-1-{name}", "job_id": job_id}],
    }


def test_observer_captures_each_running_stage_and_terminal_status(
    tmp_path, monkeypatch
):
    tmp_path.chmod(0o700)
    monkeypatch.setattr(workflow_observer, "committed_source", lambda _: "closure")
    results = iter(
        [
            _payload("RUNNING", {"reconstruct": _stage("reconstruct", "RUNNING", "1")}),
            _payload(
                "RUNNING",
                {
                    "reconstruct": _stage("reconstruct", "SUCCEEDED", "1"),
                    "render": _stage("render", "RUNNING", "2"),
                },
            ),
            _payload(
                "SUCCEEDED",
                {
                    "reconstruct": _stage("reconstruct", "SUCCEEDED", "1"),
                    "render": _stage("render", "SUCCEEDED", "2"),
                },
            ),
        ]
    )
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, next(results), b"")

    observed = []

    def observer(**kwargs):
        observed.append(kwargs)
        kwargs["output_path"].write_text(json.dumps({"status": "pass"}))
        kwargs["output_path"].chmod(0o600)
        return {"status": "pass", "stage": kwargs["stage"]}

    def bundler(**kwargs):
        kwargs["output_path"].write_text(json.dumps({"status": "pass"}))
        kwargs["output_path"].chmod(0o600)
        return {"status": "pass"}

    result = workflow_observer.observe_workflow(
        source_sha="a" * 40,
        run_id="run-1",
        workflow_s3_uri="s3://example/run-1",
        project="project-alias",
        sky_bin="/opt/sky/bin/sky",
        context="context",
        namespace="namespace",
        expected_image="example@nvidia@sha256:" + "a" * 64,
        evidence_dir=tmp_path,
        poll_seconds=1,
        status_runner=runner,
        observer=observer,
        bundler=bundler,
        sleeper=lambda _: None,
    )

    assert result["status"] == "pass"
    assert [call["stage"] for call in observed] == ["reconstruct", "render"]
    assert [call["managed_job_id"] for call in observed] == ["1", "2"]
    assert all(call["max_wait_seconds"] == 0 for call in observed)
    assert json.loads((tmp_path / "workflow-status.json").read_text())["status"] == (
        "SUCCEEDED"
    )
    assert all(
        command[:5]
        == [
            str(ROOT / "npa/.venv/bin/python"),
            "-m",
            "npa.cli.main",
            "workbench",
            "workflow",
        ]
        for command in commands
    )


def test_observer_fails_closed_before_missing_stage_can_be_attested(
    tmp_path, monkeypatch
):
    tmp_path.chmod(0o700)
    monkeypatch.setattr(workflow_observer, "committed_source", lambda _: "closure")

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            _payload("FAILED", {"reconstruct": _stage("reconstruct", "FAILED", "1")}),
            b"",
        )

    with pytest.raises(ValueError, match="failed before runtime observation"):
        workflow_observer.observe_workflow(
            source_sha="a" * 40,
            run_id="run-1",
            workflow_s3_uri="s3://example/run-1",
            project="",
            sky_bin="/opt/sky/bin/sky",
            context="context",
            namespace="namespace",
            expected_image="example@sha256:" + "a" * 64,
            evidence_dir=tmp_path,
            poll_seconds=1,
            status_runner=runner,
            observer=lambda **_: pytest.fail("observer should not run"),
            bundler=lambda **_: pytest.fail("bundler should not run"),
            sleeper=lambda _: None,
        )
