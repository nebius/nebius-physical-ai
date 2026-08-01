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
from npa.orchestration.npa_workflow.submit_matrix import (
    one_shot_submit_cases,
    runtime_submit_cases,
)
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
    seed_trigger_inbox_later,
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
    tier = case.tier
    # NPA_E2E_FORCE_ACCELERATORS puts GPU-only resources on a cpu-tier spec, so the
    # run id would otherwise claim "cpu" for a run that consumed GPUs.
    if os.environ.get("NPA_E2E_FORCE_ACCELERATORS", "").strip() and tier == "cpu":
        tier = "cpu-forcedgpu"
    return f"npa-wf-{tier}-{stem}-{stamp}"


@pytest.mark.parametrize(
    "case",
    one_shot_submit_cases(),
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

    # A case may declare its own budget when it is much slower than the rest
    # (a big image pull, a self-hosted model's cold start); otherwise the tier's.
    max_wait = case.max_wait_seconds or _max_wait()
    deadline = time.monotonic() + max_wait
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
            f"{case.spec} did not reach terminal status within {max_wait}s; "
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


# Provisioning failures that say "this cluster/image cannot host the task", as
# opposed to "the workflow is wrong". Mirrors the capacity rotation in
# test_burst_live_e2e.py: an environment limitation skips with the reason, a real
# workflow failure still fails.
INFRA_UNAVAILABLE_MARKERS = (
    "errimagepull",
    "imagepullbackoff",
    "failed to authorize",
    "container not found",
    "failed to get ssh user",
    "resourcesunavailableerror",
    "failed to provision",
    "no resource available",
    "insufficient",
    "failed_no_resource",
    "failed_prechecks",
)


def _skip_or_fail_infra(case: SubmitLiveCase, payload: dict) -> None:
    """Skip when the cluster could not host the task; fail on real errors."""

    blob = json.dumps(payload).lower()
    hit = next((marker for marker in INFRA_UNAVAILABLE_MARKERS if marker in blob), "")
    if hit:
        pytest.skip(
            f"{case.spec}: cluster could not host the task ({hit}); "
            f"runtime status={payload.get('status')} error={str(payload.get('error'))[:200]}"
        )
    pytest.fail(
        f"{case.spec} runtime run failed: {payload.get('error') or payload.get('status')}"
    )


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
        str(case.max_wait_seconds or _max_wait()),
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

        image = container_image_for_tool(case.image_tool, registry=registry)
        # Operator hook: some workbench images cannot host a SkyPilot task as built
        # (the Isaac Lab image ships no system python3, which SkyPilot's Kubernetes
        # runtime requires). Point this at a SkyPilot-compatible variant of the same
        # image, e.g. NPA_E2E_IMAGE_OVERRIDE_ISAAC_LAB=<registry>/npa-isaac-lab:<tag>-sky
        override = os.environ.get(
            f"NPA_E2E_IMAGE_OVERRIDE_{case.image_tool.upper().replace('-', '_')}", ""
        ).strip()
        args.extend(["--image", override or image])
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
    runtime_submit_cases(),
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
    trigger_seeder = None
    if case.spec == "token-factory-trigger-watch.yaml":
        # Data lands only AFTER the driver starts polling, so the wait is real.
        trigger_seeder = seed_trigger_inbox_later(
            bucket=live_bucket(e2e_project),
            run_id=run_id,
            spec_name=case.spec,
            e2e_project=e2e_project,
            delay_seconds=float(os.environ.get("NPA_E2E_TRIGGER_SEED_DELAY", "45")),
        )
    try:
        result = RUNNER.invoke(
            app, _runtime_submit_args(path, run_id=run_id, registry=e2e_registry, case=case)
        )
    finally:
        if trigger_seeder is not None:
            trigger_seeder.cancel()
    payload = parse_runtime_json(result, forbidden_markers)
    write_runtime_evidence(run_id, payload)

    if payload["status"] != "succeeded":
        _skip_or_fail_infra(case, payload)
    waves = payload["waves"]
    assert waves, payload

    if case.spec == "token-factory-trigger-watch.yaml":
        # The driver must have polled an empty prefix before the data arrived.
        watermarks = payload.get("watermarks") or {}
        assert watermarks, f"no trigger watermark recorded: {payload}"
        observed = next(iter(watermarks.values()))
        assert observed["objects"] >= 1
        assert observed["polls"] >= 2, (
            f"trigger did not actually wait (polls={observed['polls']}); the inbox "
            "was seeded too early to prove the watch"
        )

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
        # Barrier: the waves *after* the group were submitted only once every
        # member of the group had finished. Indexing off the group's position keeps
        # this correct for specs that also have serial waves BEFORE the fan-out.
        last_parallel_index = max(
            index for index, wave in enumerate(waves) if wave["kind"] == "parallel"
        )
        group_end = max(
            float(task.get("end_at") or 0.0)
            for wave in waves[: last_parallel_index + 1]
            if wave["kind"] == "parallel"
            for task in wave.get("tasks") or []
        )
        downstream_starts = [
            float(task.get("start_at") or task.get("submitted_at") or 0.0)
            for wave in waves[last_parallel_index + 1 :]
            for task in wave.get("tasks") or []
            if float(task.get("start_at") or task.get("submitted_at") or 0.0) > 0
        ]
        assert downstream_starts, "no barrier task timings recorded"
        assert min(downstream_starts) >= group_end - 1.0, (
            f"barrier task started before the parallel group finished: "
            f"group_end={group_end} starts={downstream_starts}"
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
