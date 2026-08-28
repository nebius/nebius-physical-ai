from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow.presets import (
    PUBLIC_FRANKA_LIFT_DATASET_ID,
    preset_overrides,
)


runner = CliRunner()
SPEC = (
    Path(__file__).resolve().parents[3]
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "sim2real.yaml"
)


def test_public_preset_validate_and_plan_resolve_the_same_contract() -> None:
    validate = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "validate-spec",
            str(SPEC),
            "--preset",
            "public-franka-lift",
            "--json",
        ],
    )
    assert validate.exit_code == 0, validate.output
    validated = json.loads(validate.stdout)
    assert validated["dataset_id"] == PUBLIC_FRANKA_LIFT_DATASET_ID
    assert validated["task_id"] == "Isaac-Lift-Cube-Franka-v0"
    assert validated["trigger_uri"].endswith("/public-franka-lift/")

    plan = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(SPEC),
            "--preset",
            "public-franka-lift",
            "--run-id",
            "preset-plan",
            "--assume-decision",
            "promote_checkpoint",
            "--json",
        ],
    )
    assert plan.exit_code == 0, plan.output
    planned = json.loads(plan.stdout)
    stage1 = next(step for step in planned["steps"] if step["state"] == "stage-01-trigger")
    assert PUBLIC_FRANKA_LIFT_DATASET_ID in stage1["argv"]
    assert "s3://example-bucket/sim2real-triggers/preset-plan/public-franka-lift/" in stage1["argv"]


def test_public_preset_refuses_identity_override_and_custom_mode_is_unchanged() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(SPEC),
            "--preset",
            "public-franka-lift",
            "--var",
            "dataset_id=private/custom",
        ],
    )
    assert result.exit_code == 1
    assert "owns config keys ['dataset_id']" in result.output
    assert preset_overrides(
        workflow_name="sim2real", preset="", explicit={"dataset_id": "private/custom"}
    ) == {"dataset_id": "private/custom"}


def test_submit_cli_forwards_the_selected_preset(monkeypatch) -> None:
    from npa.orchestration.npa_workflow import submit as submit_module

    captured = {}

    def stop_after_capture(path, *, config_overrides=None):
        captured.update(config_overrides or {})
        raise RuntimeError("captured preset")

    monkeypatch.setattr(submit_module, "load_spec_for_submit", stop_after_capture)
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(SPEC),
            "--preset",
            "public-franka-lift",
            "--plan-only",
        ],
    )
    assert result.exit_code == 1
    assert "captured preset" in result.output
    assert captured["dataset_id"] == PUBLIC_FRANKA_LIFT_DATASET_ID
    assert captured["workflow_preset"] == "public-franka-lift"


def test_trigger_preset_is_discoverable_and_stage_path_is_mockable(monkeypatch) -> None:
    listed = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "trigger",
            "list-presets",
            "--output-format",
            "json",
        ],
    )
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.stdout)["presets"] == ["public-franka-lift"]

    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_submit_credentials",
        lambda **_kwargs: type(
            "Credentials",
            (),
            {
                "missing": (),
                "endpoint_url": "https://s3.example.invalid",
                "secret_values": {
                    "AWS_ACCESS_KEY_ID": "unit",
                    "AWS_SECRET_ACCESS_KEY": "unit",
                },
            },
        )(),
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.public_seed.stage_public_franka_lift",
        lambda **_kwargs: {
            "status": "staged",
            "preset": "public-franka-lift",
            "trigger_uri": "s3://unit-bucket/sim2real-triggers/unit-run/public-franka-lift/",
        },
    )
    staged = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "trigger",
            "stage-preset",
            "--preset",
            "public-franka-lift",
            "--bucket",
            "unit-bucket",
            "--run-id",
            "unit-run",
            "--output-format",
            "json",
        ],
    )
    assert staged.exit_code == 0, staged.output
    assert json.loads(staged.stdout)["status"] == "staged"
