"""Unit tests for the Gemini Robotics BYOF pipeline (issue #503).

Uses a fake client — no HTTP, no key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.cli.gemini_robotics import (
    AdaptationJob,
    EvalResult,
    PlanResult,
)
from npa.workflows.byof.gemini_robotics_pipeline import (
    GeminiRoboticsPipelineConfig,
    GeminiRoboticsPipelineError,
    run_adaptation_stage,
    run_er_planning_stage,
    run_eval_stage,
    run_pipeline,
)


class FakeClient:
    def __init__(self) -> None:
        self.plans = 0
        self.adaptations: list[str] = []

    def plan(self, **kwargs):
        self.plans += 1
        return PlanResult(
            text="1. Approach.\n2. Grasp.",
            safety_calls=[{"name": "check_safety", "args": {"action": "grasp"}}],
            model=kwargs.get("model", ""),
            finish_reason="STOP",
        )

    def submit_adaptation(self, **kwargs):
        self.adaptations.append(kwargs["display_name"])
        return "tunedModels/x/operations/op1"

    def wait_for_adaptation(self, operation_name, **kwargs):
        return AdaptationJob(
            name=operation_name,
            display_name="adapt",
            done=True,
            tuned_model="tunedModels/x",
        )

    def eval_plan(self, **kwargs):
        return EvalResult(
            scores={"safety": 9},
            summary="Good.",
            raw_text="{}",
            model=kwargs.get("model", ""),
        )


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


def test_adaptation_stage_requires_dataset(tmp_path: Path) -> None:
    with pytest.raises(GeminiRoboticsPipelineError, match="dataset_path"):
        run_adaptation_stage(_config(tmp_path), FakeClient())


def test_adaptation_stage_writes_receipt(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"input": "a", "output": "b"}\n')
    receipt = run_adaptation_stage(
        _config(tmp_path, dataset_path=str(dataset)), FakeClient()
    )
    assert receipt["schema"] == "npa.gemini_robotics.adaptation.v1"
    assert receipt["done"] is True
    assert receipt["tuned_model"] == "tunedModels/x"
    assert receipt["num_examples"] == 1


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


def test_full_pipeline_with_adaptation(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"input": "a", "output": "b"}\n')
    receipt = run_pipeline(
        _config(
            tmp_path,
            run_adaptation=True,
            dataset_path=str(dataset),
            adaptation_display_name="adapt-1",
        ),
        FakeClient(),
    )
    assert set(receipt["stages"]) == {"plan", "adaptation"}


def _toolref_stage_argv(name: str) -> "list[str]":
    import re

    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    entry = TOOL_CATALOG[name]
    return [re.sub(r"{{(.*?)}}", r"DUMMY", a) for a in entry.argv_template]


def test_toolref_argv_parses_against_pipeline() -> None:
    from npa.workflows.byof import gemini_robotics_pipeline as pipe

    for tool_name, command in [
        ("workbench.gemini_robotics.plan", "plan"),
        ("workbench.gemini_robotics.adapt", "adapt"),
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
        "workbench.gemini_robotics.adapt",
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
