"""Live validation of NPA workflow specs against real infrastructure (optional)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from .npa_workflow_live_helpers import (
    ALL_GOLDEN_SPECS,
    DYNAMIC_SPECS,
    assert_no_credential_leakage,
    assume_decision_for,
    live_bucket,
    live_credential_markers,
    materialize_live_spec,
    parse_json_payload,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("NPA_INTEGRATION_E2E") != "1",
    reason="Set NPA_INTEGRATION_E2E=1 to run live NPA workflow spec checks.",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
RUNNER = CliRunner()


@pytest.fixture(scope="module")
def forbidden_markers() -> list[str]:
    return live_credential_markers()


@pytest.mark.parametrize("name", ALL_GOLDEN_SPECS)
def test_live_npa_workflow_specs_plan(name: str, forbidden_markers: list[str]) -> None:
    """Ensure golden specs load and expand on the operator machine."""

    spec = load_spec(SPECS / name)
    validate_spec(spec)
    assume = assume_decision_for(name) or "promote_checkpoint"
    plan = build_plan(spec, run_id="live-spec-check", assume_decision=assume)
    assert plan.steps, name
    assert_no_credential_leakage(json.dumps(plan.to_dict()), extra_forbidden=forbidden_markers)


@pytest.mark.parametrize("name", ALL_GOLDEN_SPECS)
def test_live_npa_workflow_cli_validate_and_plan(
    name: str,
    forbidden_markers: list[str],
) -> None:
    """Exercise validate-spec / plan-spec / run-spec --plan-only on live creds."""

    path = SPECS / name
    validate = RUNNER.invoke(app, ["workbench", "workflow", "validate-spec", str(path), "--json"])
    payload = parse_json_payload(validate, forbidden_markers)
    assert payload["status"] == "valid"

    plan_args = [
        "workbench",
        "workflow",
        "plan-spec",
        str(path),
        "--run-id",
        "live-cli-check",
        "--json",
    ]
    assume = assume_decision_for(name)
    if assume:
        plan_args.extend(["--assume-decision", assume])
    plan = RUNNER.invoke(app, plan_args)
    plan_payload = parse_json_payload(plan, forbidden_markers)
    assert plan_payload["steps"], name

    run_args = [
        "workbench",
        "workflow",
        "run-spec",
        str(path),
        "--run-id",
        "live-cli-check",
        "--plan-only",
        "--json",
    ]
    if assume:
        run_args.extend(["--assume-decision", assume])
    run = RUNNER.invoke(app, run_args)
    run_payload = parse_json_payload(run, forbidden_markers)
    assert run_payload["steps"], name


@pytest.mark.parametrize("name", sorted(DYNAMIC_SPECS))
def test_live_dynamic_specs_loop_back_plan(
    name: str,
    forbidden_markers: list[str],
) -> None:
    path = SPECS / name
    plan = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(path),
            "--run-id",
            "live-loop-back",
            "--assume-decision",
            "loop_back",
            "--json",
        ],
    )
    payload = parse_json_payload(plan, forbidden_markers)
    assert payload["steps"], name


@pytest.mark.parametrize("name", ALL_GOLDEN_SPECS)
def test_live_golden_scheduler_json_no_leak(
    name: str,
    forbidden_markers: list[str],
) -> None:
    path = SPECS / name
    args = [
        "workbench",
        "workflow",
        "run-spec",
        str(path),
        "--run-id",
        "live-scheduler-check",
        "--plan-only",
        "--scheduler-plan",
        "--json",
    ]
    assume = assume_decision_for(name)
    if assume:
        args.extend(["--assume-decision", assume])
    result = RUNNER.invoke(app, args)
    payload = parse_json_payload(result, forbidden_markers)
    assert payload.get("scheduler", {}).get("tasks"), name


@pytest.mark.parametrize("name", ALL_GOLDEN_SPECS)
def test_live_golden_materialized_validate(
    name: str,
    tmp_path: Path,
    e2e_project: str | None,
    forbidden_markers: list[str],
) -> None:
    bucket = live_bucket(e2e_project)
    path = materialize_live_spec(tmp_path, name, bucket=bucket, run_id="live-materialized")
    validate = RUNNER.invoke(app, ["workbench", "workflow", "validate-spec", str(path), "--json"])
    payload = parse_json_payload(validate, forbidden_markers)
    assert payload["status"] == "valid"
