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

This module also invokes the real `cosmos train status` and `sonic status`
subprocess entrypoints directly against `job_id`/`project_id` (both take
them as flags, no saved alias needed) and compares their JSON output to a
fresh, independent `job_status_payload` call. One real terminal job's
provider response is reused across every CLI surface tested here, since all
of them read the same Nebius Serverless Job API response through the same
`job_status_payload` function.

Optional, to also exercise LeRobot's `status` command (which has no
job/project-ID flags of its own — it only reads a saved workbench alias):
  lerobot_project / lerobot_workbench: an already-configured project alias
    and workbench name in the operator's own `~/.npa/config.yaml` whose
    saved `serverless_job.job_id`/`project_id` already point at `job_id`
    (recorded automatically the first time `npa workbench lerobot train
    --runtime serverless` submits a job — no manual config authoring
    needed for a job that was actually submitted that way). Both keys must
    be present together or both absent; a half-filled pair fails config
    validation rather than silently skipping the check.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

_REQUIRED_KEYS = (
    "project_id",
    "project",
    "job_id",
    "expected_status",
    "expected_marker",
    "output_uri",
    "min_artifact_count",
)


def _validate_optional_lerobot_alias(config: dict) -> None:
    """The LeRobot CLI alias is opt-in as a pair: both keys or neither.

    A half-filled pair is a config mistake (typo, partial edit) — silently
    skipping it would hide a broken opt-in instead of surfacing it, so this
    fails config validation rather than letting `_lerobot_cli_target` treat
    it the same as "not configured".
    """
    has_project = "lerobot_project" in config
    has_workbench = "lerobot_workbench" in config
    assert has_project == has_workbench, (
        "lerobot_project and lerobot_workbench must both be set or both be absent"
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
    _validate_optional_lerobot_alias(config)
    return config


def _run_npa_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Invoke the real `npa` console-script entrypoint as a subprocess.

    No CliRunner, no monkeypatched app: this is the exact argv an operator
    would type, so a broken option/config contract fails the same way it
    would for them, in the same process-boundary shape as
    `test_cosmos_jobs_serverless_e2e.py`'s `_run_npa`.
    """
    env = os.environ.copy()
    repo_src = Path(__file__).resolve().parents[2] / "src"
    env["PYTHONPATH"] = str(repo_src) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", "from npa.cli.main import app; app()", *args],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )


def _independent_readback(config: dict) -> dict:
    """Fresh, separate `job_status_payload` call for the same job — what
    each CLI subprocess test's JSON output must match.
    """
    from npa.clients.serverless import ServerlessClient
    from npa.serverless_common import job_status_payload

    client = ServerlessClient()
    info = client.get_job(config["job_id"], config["project_id"])
    return job_status_payload(client, info)


_COMPARABLE_DIAGNOSTIC_KEYS = (
    "status",
    "raw_status",
    "log_tail",
    "pending_reason",
    "scheduling_state",
)


def _assert_cli_matches_readback(
    result: subprocess.CompletedProcess[str],
    expected: dict,
    marker: str,
    *,
    tool: str,
) -> None:
    """Assert a CLI status subprocess succeeded and matches an independent
    `job_status_payload` call.

    Never puts provider content, private IDs, or raw stderr into an
    assertion message: pytest's assertion rewriting reprs both sides of a
    bare `assert a == b`, so every comparison here is reduced to a plain
    bool first and asserted with a fixed, tool-named message instead.
    """
    ok = result.returncode == 0
    assert ok, f"{tool} CLI exited non-zero"
    payload = json.loads(result.stdout)
    no_fetch_error = "log_fetch_error" not in payload
    assert no_fetch_error, f"{tool} CLI reported a log_fetch_error"
    for key in _COMPARABLE_DIAGNOSTIC_KEYS:
        if key in expected:
            matches = payload.get(key) == expected[key]
            assert matches, (
                f"{tool} CLI payload key {key!r} did not match independent readback"
            )
    marker_present = marker in payload.get("log_tail", "")
    assert marker_present, f"{tool} CLI log_tail is missing the expected marker"


def _lerobot_cli_target(config: dict) -> tuple[str, str] | None:
    """Return (project, workbench) for the optional LeRobot CLI check, or
    None if the operator did not opt in. A half-filled pair is rejected
    earlier, by `_validate_optional_lerobot_alias` inside `_config()`.
    """
    project = str(config.get("lerobot_project", ""))
    workbench = str(config.get("lerobot_workbench", ""))
    if not project or not workbench:
        return None
    return project, workbench


@pytest.mark.e2e_serverless
@pytest.mark.public_inputs
def test_status_reports_truthful_diagnostics_for_a_real_failed_job_with_real_checkpoint() -> (
    None
):
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


@pytest.mark.e2e_serverless
@pytest.mark.public_inputs
def test_cosmos_train_status_cli_matches_provider_readback() -> None:
    """Read-only: `cosmos train status <job_id>` takes the job/project ID as
    flags, so it needs no saved workbench alias to observe this job.
    """
    config = _config()
    expected = _independent_readback(config)
    result = _run_npa_cli(
        [
            "workbench",
            "cosmos",
            "train",
            "--runtime",
            "serverless",
            "--project-id",
            config["project_id"],
            "status",
            config["job_id"],
            "--output-format",
            "json",
        ]
    )
    _assert_cli_matches_readback(
        result, expected, config["expected_marker"], tool="cosmos train status"
    )


@pytest.mark.e2e_serverless
@pytest.mark.public_inputs
def test_sonic_status_cli_matches_provider_readback() -> None:
    """Read-only: `sonic status --runtime serverless` takes the job/project
    ID as flags, so it needs no saved workbench alias to observe this job.
    """
    config = _config()
    expected = _independent_readback(config)
    result = _run_npa_cli(
        [
            "workbench",
            "sonic",
            "status",
            "--runtime",
            "serverless",
            "--job-id",
            config["job_id"],
            "--project-id",
            config["project_id"],
            "--output-format",
            "json",
        ]
    )
    _assert_cli_matches_readback(
        result, expected, config["expected_marker"], tool="sonic status"
    )


@pytest.mark.e2e_serverless
@pytest.mark.public_inputs
def test_lerobot_status_cli_matches_provider_readback() -> None:
    """Read-only: LeRobot's `status` has no job/project-ID flags — it only
    reads a saved workbench alias — so this needs `lerobot_project` /
    `lerobot_workbench` naming an alias already pointed at `job_id` (saved
    automatically by a real `lerobot train --runtime serverless` submission).
    """
    config = _config()
    target = _lerobot_cli_target(config)
    if target is None:
        pytest.skip("lerobot_project/lerobot_workbench not configured")
    project, workbench = target
    expected = _independent_readback(config)
    result = _run_npa_cli(
        [
            "workbench",
            "lerobot",
            "-p",
            project,
            "-n",
            workbench,
            "status",
            "--output",
            "json",
        ]
    )
    _assert_cli_matches_readback(
        result, expected, config["expected_marker"], tool="lerobot status"
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


def test_config_rejects_partial_lerobot_alias(tmp_path) -> None:
    """A half-filled lerobot_project/lerobot_workbench pair must fail
    config validation, not silently skip the CLI check it names.
    """
    complete = {key: "x" for key in _REQUIRED_KEYS}
    config_file = tmp_path / "diagnostics.json"

    config_file.write_text(json.dumps({**complete, "lerobot_project": "proj-a"}))
    config_file.chmod(0o600)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("NPA_E2E_RUNTIME_DIAGNOSTICS_CONFIG", str(config_file))
        with pytest.raises(
            AssertionError, match="lerobot_project and lerobot_workbench"
        ):
            _config()

    config_file.write_text(json.dumps(complete))
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("NPA_E2E_RUNTIME_DIAGNOSTICS_CONFIG", str(config_file))
        _config()  # neither key present is valid: no error
