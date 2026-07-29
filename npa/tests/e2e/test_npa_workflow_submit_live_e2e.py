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
    concurrency_overlaps,
    live_bucket,
    live_credential_markers,
    materialize_live_spec,
    parse_json_payload,
    parse_runtime_json,
    seed_live_workflow_inputs,
    selected_submit_cases,
    write_runtime_evidence,
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
        else:
            # Required secrets must be present; silent omission caused empty-stderr
            # terminal FAILED statuses in live runs.
            pytest.skip(f"{name} required for live submit of {case.spec}")
    # Nebius VM / burst path needs registry login before image pull.
    for name in (
        "SKYPILOT_DOCKER_SERVER",
        "SKYPILOT_DOCKER_USERNAME",
        "SKYPILOT_DOCKER_PASSWORD",
    ):
        if os.environ.get(name):
            args.extend(["--secret-env", name])
    return args


def _run_id_for(case: SubmitLiveCase) -> str:
    stamp = uuid.uuid4().hex[:8]
    stem = case.spec.replace(".yaml", "").replace("_", "-")[:40]
    return f"npa-wf-{case.tier}-{stem}-{stamp}"


def _one_shot_cases() -> list[SubmitLiveCase]:
    """Matrix cases for the classic one-shot submit path.

    Runtime cases (``parallel:`` fan-out, runtime gate loops) are covered by
    ``test_npa_workflow_runtime_live_reaches_terminal`` instead: submitting them
    through the one-shot path would render the flattened serial plan, which is
    valid but proves nothing about concurrency/early-exit and burns GPU hours
    running a sweep serially.
    """

    return [case for case in selected_submit_cases() if not case.runtime]


@pytest.mark.parametrize(
    "case",
    _one_shot_cases(),
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
    seed_live_workflow_inputs(
        spec_name=case.spec,
        bucket=bucket,
        run_id=run_id,
        e2e_project=e2e_project,
    )

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
    if os.environ.get("NPA_E2E_CLEAR_WORKBENCH_IMAGES", "").strip() in {"1", "true", "yes"}:
        plan_args.extend(["--image", "none"])
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
    # Workbench images often fail SkyPilot k8s apt-ssh setup; clear pins and
    # rely on NPA_SRC_S3_URI + default image (validated for Token Factory).
    if os.environ.get("NPA_E2E_CLEAR_WORKBENCH_IMAGES", "").strip() in {"1", "true", "yes"}:
        submit_args.extend(["--image", "none"])
    submit_args.extend(_secret_env_args(case))

    if (
        os.environ.get("NPA_E2E_CLEAR_WORKBENCH_IMAGES", "").strip() in {"1", "true", "yes"}
        and not os.environ.get("NPA_SRC_S3_URI", "").strip()
    ):
        pytest.skip(
            "NPA_E2E_CLEAR_WORKBENCH_IMAGES=1 requires NPA_SRC_S3_URI for live submit"
        )

    submitted = RUNNER.invoke(app, submit_args)
    submit_payload = parse_json_payload(submitted, forbidden_markers)
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
                detail = (
                    (current.stderr or "")[-500:]
                    or (current.stdout or "")[-500:]
                    or getattr(current, "error", "")
                    or "(no stderr/stdout; check: sky jobs logs "
                    f"{job_id})"
                )
                pytest.fail(
                    f"{case.spec} reached terminal failure status={last_status} "
                    f"job_id={job_id} detail={detail}"
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


def _runtime_cases() -> list[SubmitLiveCase]:
    return [case for case in selected_submit_cases() if case.runtime and not case.plan_only]


def _runtime_submit_args(
    path: Path,
    *,
    run_id: str,
    registry: str,
    case: SubmitLiveCase,
    extra_vars: dict[str, str] | None = None,
) -> list[str]:
    args = [
        "workbench",
        "workflow",
        "submit",
        str(path),
        "--run-id",
        run_id,
        "--runtime",
        "--registry",
        registry,
        "--poll-seconds",
        os.environ.get("NPA_E2E_NPA_WORKFLOW_SUBMIT_POLL_SECONDS", "30"),
        "--max-wait-seconds",
        str(_max_wait()),
        "--submit-timeout",
        "1800",
        "--output-format",
        "json",
    ]
    if not _cancel_on_timeout():
        args.append("--no-cancel-on-timeout")
    for key, value in [*case.config_vars, *sorted((extra_vars or {}).items())]:
        args.extend(["--var", f"{key}={value}"])
    if case.image_tool:
        # Stages that must run inside a baked workbench image (e.g. Isaac Lab's
        # training script). Branch code is layered on with NPA_SRC_OVERLAY=1.
        from npa.deploy.images import container_image_for_tool

        args.extend(["--image", container_image_for_tool(case.image_tool, registry=registry)])
    elif os.environ.get("NPA_E2E_CLEAR_WORKBENCH_IMAGES", "").strip() in {"1", "true", "yes"}:
        args.extend(["--image", "none"])
    args.extend(_secret_env_args(case))
    return args


def _prepare_runtime_run(
    case: SubmitLiveCase,
    tmp_path: Path,
    e2e_project: str | None,
    suffix: str = "",
) -> tuple[str, Path]:
    bucket = live_bucket(e2e_project)
    run_id = _run_id_for(case) + (f"-{suffix}" if suffix else "")
    path = materialize_live_spec(tmp_path, case.spec, bucket=bucket, run_id=run_id)
    seed_live_workflow_inputs(
        spec_name=case.spec,
        bucket=bucket,
        run_id=run_id,
        e2e_project=e2e_project,
    )
    return run_id, path


@pytest.mark.parametrize(
    "case",
    _runtime_cases(),
    ids=lambda c: f"{c.tier}:{c.spec}",
)
def test_npa_workflow_runtime_live_reaches_terminal(
    case: SubmitLiveCase,
    tmp_path: Path,
    e2e_project: str | None,
    e2e_registry: str,
    forbidden_markers: list[str],
) -> None:
    """Drive a spec through the runtime orchestrator against real infrastructure.

    Asserts the run reaches a terminal success, and — for specs with a
    ``parallel:`` group — that the group's SkyPilot tasks actually overlapped in
    time and that the barrier state started only after they finished.
    """

    if os.environ.get("NPA_E2E_NPA_WORKFLOW_RUNTIME") != "1":
        pytest.skip("NPA_E2E_NPA_WORKFLOW_RUNTIME not set")
    if case.requires_token_factory and not os.environ.get("NEBIUS_TOKEN_FACTORY_KEY"):
        pytest.skip("NEBIUS_TOKEN_FACTORY_KEY required for this twin")

    run_id, path = _prepare_runtime_run(case, tmp_path, e2e_project)
    result = RUNNER.invoke(
        app, _runtime_submit_args(path, run_id=run_id, registry=e2e_registry, case=case)
    )
    payload = parse_runtime_json(result, forbidden_markers)
    write_runtime_evidence(run_id, payload)

    assert payload["status"] == "succeeded", payload.get("error") or payload
    waves = payload["waves"]
    assert waves, payload

    if case.expected_parallel_tasks > 1:
        parallel_waves = [wave for wave in waves if wave["kind"] == "parallel"]
        assert parallel_waves, f"{case.spec} declared a parallel group but ran none: {waves}"
        launched = sum(len(wave["states"]) for wave in parallel_waves)
        assert launched == case.expected_parallel_tasks
        # Two independent concurrency signals: live RUNNING observations taken
        # while polling, and overlapping submitted/end intervals afterwards.
        observed = max(wave.get("max_concurrent_observed", 0) for wave in parallel_waves)
        overlaps = concurrency_overlaps(parallel_waves[0].get("tasks") or [])
        assert observed >= 2 or overlaps, (
            "parallel wave never showed concurrent tasks: "
            f"observed={observed} tasks={parallel_waves[0].get('tasks')}"
        )
        # Barrier: the first downstream (serial) wave was submitted only after
        # every member of the group had finished.
        group_end = max(
            float(task.get("end_at") or 0.0)
            for wave in parallel_waves
            for task in wave.get("tasks") or []
        )
        later_starts = [
            float(task.get("start_at") or task.get("submitted_at") or 0.0)
            for wave in waves
            if wave["kind"] == "serial"
            for task in wave.get("tasks") or []
            if float(task.get("start_at") or task.get("submitted_at") or 0.0) > 0
        ]
        assert later_starts, "no barrier task timings recorded"
        assert min(later_starts) >= group_end - 1.0, (
            f"barrier task started before the parallel group finished: "
            f"group_end={group_end} starts={later_starts}"
        )


def test_npa_workflow_runtime_gate_loop_early_exit_vs_full_budget(
    tmp_path: Path,
    e2e_project: str | None,
    e2e_registry: str,
    forbidden_markers: list[str],
) -> None:
    """The same bounded-loop spec must early-exit or run its full budget live.

    Run A passes the gate on iteration 1 (``grade_threshold=0.0``) and must NOT
    submit the remaining iterations. Run B can never pass the gate
    (``grade_threshold`` above any achievable score) and must run the whole
    budget and take the other branch. The decision each iteration comes from the
    real ``decision.json`` the gate stage wrote to S3.
    """

    if os.environ.get("NPA_E2E_NPA_WORKFLOW_RUNTIME") != "1":
        pytest.skip("NPA_E2E_NPA_WORKFLOW_RUNTIME not set")
    if not os.environ.get("NEBIUS_TOKEN_FACTORY_KEY"):
        pytest.skip("NEBIUS_TOKEN_FACTORY_KEY required for the gate-loop twin")

    case = next(
        c for c in SUBMIT_LIVE_MATRIX if c.spec == "token-factory-gate-loop.yaml"
    )
    payloads: dict[str, dict] = {}
    for label, threshold in (("early", "0.0"), ("full", "1.01")):
        run_id, path = _prepare_runtime_run(case, tmp_path, e2e_project, suffix=label)
        args = _runtime_submit_args(
            path,
            run_id=run_id,
            registry=e2e_registry,
            case=case,
            extra_vars={"grade_threshold": threshold},
        )
        result = RUNNER.invoke(app, args)
        payload = parse_runtime_json(result, forbidden_markers)
        write_runtime_evidence(run_id, payload)
        assert payload["status"] == "succeeded", payload.get("error") or payload
        payloads[label] = payload

    def _count(payload: dict, state: str) -> int:
        return sum(
            1 for wave in payload["waves"] for name in wave["states"] if name == state
        )

    early, full = payloads["early"], payloads["full"]
    assert _count(early, "quality-gate") == 1, "gate did not early-exit on iteration 1"
    assert _count(full, "quality-gate") >= 2, "gate loop did not run its budget"
    assert _count(full, "caption-batch") > _count(early, "caption-batch")
    assert early["decisions"], "no decision artifact was read at runtime"
    assert early["decisions"][-1]["decision"] == "promote_checkpoint"
    assert full["decisions"][-1]["decision"] == "loop_back_to_inner_loop"
    # Data-dependent branching: promote publishes, exhausted budget escalates.
    early_states = [name for wave in early["waves"] for name in wave["states"]]
    full_states = [name for wave in full["waves"] for name in wave["states"]]
    assert early_states[-1] == "publish"
    assert full_states[-1] == "escalate"


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
    # Plan-only must never mint/print live registry passwords.
    if "SKYPILOT_DOCKER_PASSWORD" in yaml_text:
        assert "<SKYPILOT_DOCKER_PASSWORD>" in yaml_text
        live_pw = (os.environ.get("SKYPILOT_DOCKER_PASSWORD") or "").strip()
        if live_pw and len(live_pw) >= 8:
            assert live_pw not in yaml_text
