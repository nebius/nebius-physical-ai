"""Read-only validation of `job_status_payload` against one real, already-run Serverless Job.

This lane does not submit, cancel, or delete anything here. The job is
submitted and owned end to end by whoever runs this test (real training or
optimizer workload, real checkpoint/artifact publication, then a deliberate
post-checkpoint failure); this test only observes the job's already-terminal
state and its already-published artifacts, and asserts that
`npa.serverless_common.job_status_payload` (npa/src/npa/serverless_common/status.py)
reports the real failure reason instead of masking or dropping it.

Opt in with NPA_E2E_RUNTIME_DIAGNOSTICS_CONFIG pointing to an owner-only
(mode 0600) JSON file:
  project_id: Nebius project ID that owns the job.
  project: project alias to resolve S3 read credentials for the artifact
    count check (see npa.clients.project_credentials.s3_client_for_project).
  job_id: the exact provider job ID (not a name) to observe.
  expected_status: the terminal status this job must already be in, e.g.
    "failed".
  expected_marker: a distinguishing string that must appear in the real
    `log_tail` this test fetches, proving the diagnostics reflect this
    job's actual output rather than a cached or generic message.
  output_uri: s3://bucket/prefix/ the job's real artifacts were published
    under.
  min_artifact_count: minimum number of objects expected under output_uri.

Optional, for the cross-job recovery scenario (a second job loads this job's
checkpoint in a fresh process and continues training):
  recovery_job_id: provider job ID of the job that consumed this job's
    checkpoint. When present, this test also asserts that job reached
    `recovery_expected_status`.
  recovery_expected_status: terminal status the recovery job must already be
    in, e.g. "succeeded". Required if recovery_job_id is present.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest

_REQUIRED_KEYS = (
    "project_id", "project", "job_id", "expected_status",
    "expected_marker", "output_uri", "min_artifact_count",
)


def _config() -> dict:
    path = os.environ.get("NPA_E2E_RUNTIME_DIAGNOSTICS_CONFIG", "")
    if not path:
        pytest.skip("NPA_E2E_RUNTIME_DIAGNOSTICS_CONFIG not set")
    config_file = Path(path)
    assert config_file.stat().st_mode & 0o077 == 0, "live config must be owner-only"
    config = json.loads(config_file.read_text())
    missing = [key for key in _REQUIRED_KEYS if key not in config]
    assert not missing, f"config missing required keys: {missing}"
    return config


@pytest.mark.e2e_serverless
@pytest.mark.public_inputs
def test_status_reports_truthful_diagnostics_for_a_real_failed_job_with_real_checkpoint() -> None:
    """Read-only: never submits, cancels, or deletes the job it observes."""
    config = _config()

    from npa.clients.project_credentials import s3_client_for_project
    from npa.clients.serverless import ServerlessClient
    from npa.serverless_common import job_status_payload

    client = ServerlessClient()
    info = client.get_job(config["job_id"], config["project_id"])
    assert info.status == config["expected_status"], (
        f"expected terminal status {config['expected_status']!r}, observed "
        f"{info.status!r} — this job has not reached the state this test "
        "validates against yet"
    )

    payload = job_status_payload(client, info)
    assert payload["status"] == config["expected_status"]
    assert "log_fetch_error" not in payload, payload.get("log_fetch_error")
    assert config["expected_marker"] in payload["log_tail"]

    parsed = urlparse(config["output_uri"])
    s3 = s3_client_for_project(config["project"])
    paginator = s3.get_paginator("list_objects_v2")
    artifact_count = sum(
        len(page.get("Contents", []))
        for page in paginator.paginate(
            Bucket=parsed.netloc, Prefix=parsed.path.lstrip("/")
        )
    )
    assert artifact_count >= int(config["min_artifact_count"]), (
        f"expected at least {config['min_artifact_count']} real artifacts "
        f"under {config['output_uri']}, found {artifact_count}"
    )

    recovery_job_id = config.get("recovery_job_id")
    if recovery_job_id:
        recovery_info = client.get_job(recovery_job_id, config["project_id"])
        assert recovery_info.status == config["recovery_expected_status"], (
            f"cross-job recovery job {recovery_job_id!r} expected terminal "
            f"status {config['recovery_expected_status']!r}, observed "
            f"{recovery_info.status!r}"
        )


def test_config_rejects_group_or_world_readable_file(tmp_path) -> None:
    config_file = tmp_path / "diagnostics.json"
    config_file.write_text("{}")
    config_file.chmod(0o644)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("NPA_E2E_RUNTIME_DIAGNOSTICS_CONFIG", str(config_file))
        with pytest.raises(AssertionError, match="owner-only"):
            _config()


def test_config_rejects_missing_required_keys(tmp_path) -> None:
    config_file = tmp_path / "diagnostics.json"
    config_file.write_text(json.dumps({"project_id": "project-1"}))
    config_file.chmod(0o600)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("NPA_E2E_RUNTIME_DIAGNOSTICS_CONFIG", str(config_file))
        with pytest.raises(AssertionError, match="missing required keys"):
            _config()
