"""Live SkyPilot submit coverage for npa.workflow/v0.0.1 twins.

Skip-by-default. Enable with:

  NPA_INTEGRATION_E2E=1
  NPA_E2E_NPA_WORKFLOW_SUBMIT=1

Optional filters:

  NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=cpu,gpu,multi   # default: all three
  NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=token-factory-caption.yaml,...
  NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS=3600
  NPA_E2E_NPA_WORKFLOW_SUBMIT_POLL_SECONDS=30
  NPA_E2E_NPA_WORKFLOW_SUBMIT_CANCEL_ON_TIMEOUT=1
  NPA_REGISTRY / --registry via NPA_E2E_REGISTRY
  NEBIUS_TOKEN_FACTORY_KEY for cpu-tier Token Factory twins

This exercises the full path: validate → plan → render → sky jobs launch →
poll until terminal. It does **not** delete SkyPilot originals; it submits the
npa.workflow twins through ``npa workbench workflow submit``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.skypilot.workflow import workflow_status
from .npa_workflow_live_helpers import (
    SUBMIT_LIVE_MATRIX,
    SubmitLiveCase,
    assert_no_credential_leakage,
    assume_decision_for,
    live_bucket,
    live_credential_markers,
    materialize_live_spec,
    parse_json_payload,
    selected_submit_cases,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.e2e_skypilot,
    pytest.mark.gpu,
]

REPO_ROOT = Path(__file__).resolve().parents[3]
SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
RUNNER = CliRunner()

TERMINAL_OK = frozenset({"SUCCEEDED", "SUCCESS", "COMPLETED", "DONE"})
TERMINAL_FAIL = frozenset(
    {
        "FAILED",
        "FAIL",
        "FAILED_PRECHECKS",
        "FAILED_SETUP",
        "FAILED_RUNTIME",
        "FAILED_CONTROLLER",
        "CANCELLED",
        "CANCELED",
        "STOPPED",
        "CANCELLING",
    }
)
NONTERMINAL = frozenset(
    {"PENDING", "STARTING", "RUNNING", "RECOVERING", "SUBMITTED", "INIT", "UNKNOWN"}
)


def _is_terminal_fail(status: str) -> bool:
    upper = status.upper()
    return upper in TERMINAL_FAIL or upper.startswith("FAILED")


@pytest.fixture(autouse=True)
def _require_live_submit() -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    if os.environ.get("NPA_E2E_NPA_WORKFLOW_SUBMIT") != "1":
        pytest.skip("NPA_E2E_NPA_WORKFLOW_SUBMIT not set")


@pytest.fixture(scope="module")
def forbidden_markers() -> list[str]:
    return live_credential_markers()


@pytest.fixture(scope="module")
def e2e_registry() -> str:
    registry = (
        os.environ.get("NPA_E2E_REGISTRY")
        or os.environ.get("NPA_REGISTRY")
        or ""
    ).strip()
    if not registry:
        pytest.skip("Set NPA_E2E_REGISTRY or NPA_REGISTRY for live npa.workflow submit")
    return registry


def _max_wait() -> int:
    return int(os.environ.get("NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS", "3600"))


def _poll_seconds() -> int:
    return int(os.environ.get("NPA_E2E_NPA_WORKFLOW_SUBMIT_POLL_SECONDS", "30"))


def _cancel_on_timeout() -> bool:
    return os.environ.get("NPA_E2E_NPA_WORKFLOW_SUBMIT_CANCEL_ON_TIMEOUT", "1") == "1"


def _secret_env_args(case: SubmitLiveCase) -> list[str]:
    args: list[str] = []
    for name in case.secret_envs:
        if os.environ.get(name):
            args.extend(["--secret-env", name])
        elif case.tier == "cpu" and name == "NEBIUS_TOKEN_FACTORY_KEY":
            pytest.skip(f"{name} required for cpu-tier twin {case.spec}")
    return args


def _run_id_for(case: SubmitLiveCase) -> str:
    stamp = uuid.uuid4().hex[:8]
    stem = case.spec.replace(".yaml", "").replace("_", "-")[:40]
    return f"npa-wf-{case.tier}-{stem}-{stamp}"


@pytest.mark.parametrize(
    "case",
    selected_submit_cases(),
    ids=lambda c: f"{c.tier}:{c.spec}",
)
def test_npa_workflow_submit_live_reaches_terminal(
    case: SubmitLiveCase,
    tmp_path: Path,
    e2e_project: str | None,
    e2e_registry: str,
    forbidden_markers: list[str],
) -> None:
    """Submit one npa.workflow twin and wait for a terminal SkyPilot status."""

    if case.requires_token_factory and not os.environ.get("NEBIUS_TOKEN_FACTORY_KEY"):
        pytest.skip("NEBIUS_TOKEN_FACTORY_KEY required for this twin")

    bucket = live_bucket(e2e_project)
    run_id = _run_id_for(case)
    path = materialize_live_spec(tmp_path, case.spec, bucket=bucket, run_id=run_id)

    # Preflight: render only (no cluster).
    plan_args = [
        "workbench",
        "workflow",
        "submit",
        str(path),
        "--run-id",
        f"{run_id}-plan",
        "--plan-only",
        "--registry",
        e2e_registry,
        "--output-format",
        "json",
    ]
    assume = assume_decision_for(case.spec)
    if assume:
        plan_args.extend(["--assume-decision", assume])
    planned = RUNNER.invoke(app, plan_args)
    plan_payload = parse_json_payload(planned, forbidden_markers)
    assert plan_payload["status"] == "PLANNED"
    assert plan_payload["steps"] >= 1
    assert "${" not in plan_payload.get("skypilot_yaml", "")

    if case.plan_only:
        return

    submit_args = [
        "workbench",
        "workflow",
        "submit",
        str(path),
        "--run-id",
        run_id,
        "--registry",
        e2e_registry,
        "--submit-timeout",
        "1800",
        "--output-format",
        "json",
    ]
    if assume:
        submit_args.extend(["--assume-decision", assume])
    submit_args.extend(_secret_env_args(case))

    submitted = RUNNER.invoke(app, submit_args)
    assert_no_credential_leakage(submitted.output, extra_forbidden=forbidden_markers)
    assert submitted.exit_code == 0, submitted.output
    submit_payload = json.loads(submitted.output)
    assert submit_payload.get("status") in {"SUBMITTED", "RUNNING", "PENDING", "STARTING"}
    job_id = str(submit_payload.get("job_id") or run_id)

    deadline = time.monotonic() + _max_wait()
    last_status = str(submit_payload.get("status") or "SUBMITTED")
    try:
        while time.monotonic() < deadline:
            current = workflow_status(job_id)
            last_status = (current.status or "UNKNOWN").upper()
            assert_no_credential_leakage(
                current.stdout + current.stderr,
                extra_forbidden=forbidden_markers,
            )
            if last_status in TERMINAL_OK:
                return
            if _is_terminal_fail(last_status):
                pytest.fail(
                    f"{case.spec} reached terminal failure status={last_status} "
                    f"job_id={job_id} stderr={current.stderr[-500:]}"
                )
            time.sleep(_poll_seconds())
        pytest.fail(
            f"{case.spec} did not reach terminal status within {_max_wait()}s; "
            f"last_status={last_status} job_id={job_id}"
        )
    finally:
        if _cancel_on_timeout() and last_status not in TERMINAL_OK and not _is_terminal_fail(
            last_status
        ):
            # Best-effort cancel via sky jobs cancel through workflow helper.
            try:
                from npa.orchestration.skypilot._bin import resolve_config
                from npa.orchestration.skypilot.workflow_state import cancel_workflow_job

                runtime = resolve_config()
                cancel_workflow_job(
                    sky_bin=str(runtime.sky_bin),
                    job_id=str(job_id),
                    run_id=run_id,
                    cluster=run_id,
                )
            except Exception:
                pass


@pytest.mark.parametrize("case", SUBMIT_LIVE_MATRIX, ids=lambda c: c.spec)
def test_npa_workflow_submit_plan_only_matrix_no_leak(
    case: SubmitLiveCase,
    tmp_path: Path,
    e2e_project: str | None,
    e2e_registry: str,
    forbidden_markers: list[str],
) -> None:
    """Always-safe preflight: every twin in the matrix must render cleanly."""

    if os.environ.get("NPA_E2E_NPA_WORKFLOW_SUBMIT") != "1":
        pytest.skip("NPA_E2E_NPA_WORKFLOW_SUBMIT not set")
    bucket = live_bucket(e2e_project)
    run_id = f"plan-{uuid.uuid4().hex[:8]}"
    path = materialize_live_spec(tmp_path, case.spec, bucket=bucket, run_id=run_id)
    args = [
        "workbench",
        "workflow",
        "submit",
        str(path),
        "--run-id",
        run_id,
        "--plan-only",
        "--registry",
        e2e_registry,
        "--output-format",
        "json",
    ]
    assume = assume_decision_for(case.spec)
    if assume:
        args.extend(["--assume-decision", assume])
    result = RUNNER.invoke(app, args)
    payload = parse_json_payload(result, forbidden_markers)
    assert payload["status"] == "PLANNED"
    assert payload["steps"] >= 1
    yaml_text = payload.get("skypilot_yaml", "")
    assert "execution: serial" in yaml_text
    assert "${" not in yaml_text
