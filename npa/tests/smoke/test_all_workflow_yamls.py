"""Smoke validation for every checked-in workflow YAML."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec

REPO_ROOT = Path(__file__).resolve().parents[3]
NPA_SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
SKYPILOT_SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "skypilot"
RUNNER = CliRunner()


def _skypilot_yaml_paths() -> list[Path]:
    return sorted(SKYPILOT_SPECS.glob("*.yaml"))


def _npa_yaml_paths() -> list[Path]:
    return sorted(NPA_SPECS.glob("*.yaml"))


@pytest.mark.parametrize("path", _skypilot_yaml_paths(), ids=lambda p: p.name)
def test_skypilot_yaml_documents_parse(path: Path) -> None:
    docs = [
        doc
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if doc is not None
    ]
    assert docs, path.name
    assert isinstance(docs[0], dict)
    assert docs[0].get("name"), path.name
    assert docs[0].get("execution") in {"serial", None} or "execution" in docs[0]


@pytest.mark.parametrize("path", _npa_yaml_paths(), ids=lambda p: p.name)
def test_npa_workflow_yaml_validates(path: Path) -> None:
    spec = load_spec(path)
    validate_spec(spec)
    assert spec.api_version == "npa.workflow/v0.0.1"


@pytest.mark.parametrize("path", _npa_yaml_paths(), ids=lambda p: p.name)
def test_npa_workflow_cli_validate_and_plan(path: Path) -> None:
    validate = RUNNER.invoke(
        app,
        ["workbench", "workflow", "validate-spec", str(path), "--json"],
    )
    assert validate.exit_code == 0, validate.output
    payload = json.loads(validate.output)
    assert payload["status"] == "valid"

    assume = "loop_back" if path.name == "sim2real-vlm-rl.yaml" else "promote_checkpoint"
    plan_args = [
        "workbench",
        "workflow",
        "plan-spec",
        str(path),
        "--run-id",
        f"smoke-{path.stem}",
        "--json",
    ]
    if path.name == "sim2real-vlm-rl.yaml":
        plan_args.extend(["--assume-decision", assume])
    plan = RUNNER.invoke(app, plan_args)
    assert plan.exit_code == 0, plan.output
    plan_payload = json.loads(plan.output)
    assert plan_payload["steps"], path.name

    spec = load_spec(path)
    built = build_plan(spec, run_id=f"smoke-{path.stem}", assume_decision=assume)
    assert built.steps

    scheduler = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "run-spec",
            str(path),
            "--run-id",
            f"smoke-{path.stem}",
            "--plan-only",
            "--scheduler-plan",
            "--json",
            *(["--assume-decision", assume] if path.name == "sim2real-vlm-rl.yaml" else []),
        ],
    )
    assert scheduler.exit_code == 0, scheduler.output
    scheduler_payload = json.loads(scheduler.output)
    assert scheduler_payload.get("scheduler", {}).get("tasks"), path.name
