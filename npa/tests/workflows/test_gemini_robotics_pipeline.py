"""Unit tests for the Gemini Robotics BYOF pipeline (issue #503).

Uses a fake client — no HTTP, no key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from npa.cli.workbench.gemini_robotics import (
    EvalResult,
    PlanResult,
)
from npa.workflows.byof.gemini_robotics_pipeline import (
    GeminiRoboticsPipelineConfig,
    run_er_planning_stage,
    run_eval_stage,
    run_pipeline,
)


class FakeClient:
    def __init__(self) -> None:
        self.plans = 0

    def plan(self, **kwargs):
        self.plans += 1
        return PlanResult(
            text="1. Approach.\n2. Grasp.",
            safety_calls=[{"name": "check_safety", "args": {"action": "grasp"}}],
            model=kwargs.get("model", ""),
            finish_reason="STOP",
        )

    def eval_plan(self, **kwargs):
        return EvalResult(
            scores={"safety": 9},
            summary="Good.",
            raw_text="{}",
            model=kwargs.get("model", ""),
        )


def _hosted_workflow_spec(operation: str) -> dict:
    return {
        "apiVersion": "npa.workflow/v0.0.1",
        "kind": "Workflow",
        "metadata": {"name": "gemini-hosted-contract"},
        "config": {
            "api_base_url": "https://provider.example.invalid",
            "model_id": "operator-selected-model",
            "task": "inspect the scene",
            "output_dir": "/tmp/gemini-report",
            "plan_path": "/tmp/plan.json",
            "rubric_path": "/tmp/rubric.txt",
        },
        "resources": {"cpu": {"cloud": "kubernetes", "cpus": 2}},
        "initial": "audit",
        "states": {
            "audit": {
                "toolRef": f"workbench.gemini_robotics.{operation}",
                "resources": "cpu",
                "terminal": True,
            }
        },
    }


@pytest.mark.parametrize("operation", ["plan", "eval"])
def test_hosted_workflow_renders_cpu_and_declares_google_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from npa.orchestration.npa_workflow.submit import prepare_npa_workflow_for_submit

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source/npa")
    path = tmp_path / "gemini.yaml"
    path.write_text(yaml.safe_dump(_hosted_workflow_spec(operation)))
    prepared = prepare_npa_workflow_for_submit(path, run_id="gemini-contract")
    try:
        assert "GOOGLE_API_KEY" in prepared.secret_env_hints
        tasks = list(yaml.safe_load_all(prepared.skypilot_yaml_path.read_text()))
        assert "accelerators" not in tasks[1]["resources"]
        assert "image_id" not in tasks[1]["resources"]
        assert "gemini_robotics_pipeline" in tasks[1]["run"]
    finally:
        prepared.temp_dir.cleanup()


def _config(tmp_path: Path, **kwargs) -> GeminiRoboticsPipelineConfig:
    return GeminiRoboticsPipelineConfig(
        task="pick up the cup",
        output_dir=str(tmp_path / "out"),
        **kwargs,
    )


def test_planning_stage_writes_receipt(tmp_path: Path) -> None:
    receipt = run_er_planning_stage(_config(tmp_path), FakeClient())
    assert receipt["schema"] == "npa.gemini_robotics.plan.v1"
    assert receipt["plan_text"].startswith("1. Approach.")
    assert receipt["safety_calls"][0]["name"] == "check_safety"
    plan_path = Path(receipt["artifact_path"])
    assert plan_path.exists()
    stored = json.loads(plan_path.read_text())
    assert stored["plan_text"] == receipt["plan_text"]


def test_eval_stage_writes_receipt(tmp_path: Path) -> None:
    rubric = tmp_path / "rubric.txt"
    rubric.write_text("safety first")
    plan_receipt = run_er_planning_stage(_config(tmp_path), FakeClient())
    receipt = run_eval_stage(
        _config(tmp_path, rubric_path=str(rubric)), plan_receipt, FakeClient()
    )
    assert receipt["schema"] == "npa.gemini_robotics.eval.v1"
    assert receipt["scores"] == {"safety": 9}


def test_full_pipeline_plan_and_eval(tmp_path: Path) -> None:
    rubric = tmp_path / "rubric.txt"
    rubric.write_text("safety first")
    receipt = run_pipeline(_config(tmp_path, rubric_path=str(rubric)), FakeClient())
    assert receipt["schema"] == "npa.gemini_robotics.pipeline-receipt.v1"
    assert set(receipt["stages"]) == {"plan", "eval"}
    for stage in receipt["stages"].values():
        assert Path(stage["artifact_path"]).exists()
        assert len(stage["artifact_sha256"]) == 64


def _toolref_stage_argv(name: str) -> "list[str]":
    import re

    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    entry = TOOL_CATALOG[name]
    return [re.sub(r"{{(.*?)}}", r"DUMMY", a) for a in entry.argv_template]


def test_toolref_argv_parses_against_pipeline() -> None:
    from npa.workflows.byof import gemini_robotics_pipeline as pipe

    for tool_name, command in [
        ("workbench.gemini_robotics.plan", "plan"),
        ("workbench.gemini_robotics.eval", "eval"),
    ]:
        argv = _toolref_stage_argv(tool_name)
        assert argv[:3] == [
            "python3",
            "-m",
            "npa.workflows.byof.gemini_robotics_pipeline",
        ]
        args = pipe.build_parser().parse_args(argv[3:])
        assert args.command == command
        # Override-required: the template must carry an explicit base URL slot
        # and the stage must carry an explicit model slot.
        assert args.api_base_url == "DUMMY"


def test_toolref_descriptions_state_provisional() -> None:
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    for tool_name in (
        "workbench.gemini_robotics.plan",
        "workbench.gemini_robotics.eval",
    ):
        description = TOOL_CATALOG[tool_name].description
        assert "provisional" in description
        assert "no live access has been validated" in description


def test_pipeline_main_rejects_missing_overrides() -> None:
    import pytest

    from npa.workflows.byof import gemini_robotics_pipeline as pipe

    with pytest.raises(SystemExit):
        pipe.main(["plan", "--task", "t", "--model", "m", "--output-dir", "o"])
    with pytest.raises(SystemExit):
        pipe.main(
            [
                "--api-base-url",
                "https://example.test",
                "plan",
                "--task",
                "t",
                "--output-dir",
                "o",
            ]
        )
