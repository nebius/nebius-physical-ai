"""Unit tests for the hosted Gemini Robotics workflow stages.

The client and storage boundaries are both faked: these prove local contracts,
not provider acceptance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from npa.cli.workbench.gemini_robotics import EvalResult, PlanResult
from npa.workflows.byof.gemini_robotics_pipeline import (
    GeminiRoboticsPipelineConfig,
    GeminiRoboticsPipelineError,
    read_eval_input,
    run_er_planning_stage,
    run_eval_stage,
)


class FakeStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.writes: list[str] = []

    def put_bytes_conditional(
        self,
        payload: bytes,
        uri: str,
        *,
        if_none_match: bool = False,
        content_type: str = "application/octet-stream",
    ) -> str:
        assert if_none_match is True
        assert content_type == "application/json"
        if uri in self.objects:
            raise AssertionError(f"attempted to overwrite {uri}")
        self.objects[uri] = payload
        self.writes.append(uri)
        return f"etag-{len(self.writes)}"

    def read_bytes_with_etag(self, uri: str) -> tuple[bytes, str] | None:
        payload = self.objects.get(uri)
        return None if payload is None else (payload, "input-etag")


class FakeClient:
    def __init__(self) -> None:
        self.plans = 0
        self.evals = 0

    def plan(self, **kwargs):
        self.plans += 1
        return PlanResult(
            text="1. Approach.\n2. Grasp.",
            model_function_calls=[{"name": "unrequested_call", "args": {}}],
            model=kwargs["model"],
            finish_reason="STOP",
        )

    def eval_plan(self, **kwargs):
        self.evals += 1
        return EvalResult(
            scores={"safety": 9},
            summary="Good.",
            raw_text="{}",
            model=kwargs["model"],
        )


def _hosted_workflow_spec(operation: str, tmp_path: Path) -> dict:
    return {
        "apiVersion": "npa.workflow/v0.0.1",
        "kind": "Workflow",
        "metadata": {"name": "gemini-hosted-contract"},
        "config": {
            "api_base_url": "https://provider.example.invalid",
            "model_id": "operator-selected-model",
            "task": "inspect the scene",
            "input_uri": "s3://example-bucket/input/eval-request.json",
            "output_uri": "s3://example-bucket/output/receipt.json",
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
    path.write_text(yaml.safe_dump(_hosted_workflow_spec(operation, tmp_path)))
    prepared = prepare_npa_workflow_for_submit(path, run_id="gemini-contract")
    try:
        assert "GOOGLE_API_KEY" in prepared.secret_env_hints
        tasks = list(yaml.safe_load_all(prepared.skypilot_yaml_path.read_text()))
        assert "accelerators" not in tasks[1]["resources"]
        assert "image_id" not in tasks[1]["resources"]
        assert "gemini_robotics_pipeline" in tasks[1]["run"]
        assert "--output-path" in tasks[1]["run"]
    finally:
        prepared.temp_dir.cleanup()


def _config(**kwargs) -> GeminiRoboticsPipelineConfig:
    values = {
        "task": "pick up the cup",
        "output_path": "s3://example-bucket/receipts/plan.json",
        "model": "operator-selected-model",
    }
    values.update(kwargs)
    return GeminiRoboticsPipelineConfig(**values)


def test_planning_stage_writes_conditional_s3_receipt() -> None:
    storage = FakeStorage()
    receipt = run_er_planning_stage(_config(), FakeClient(), storage)
    assert receipt["schema"] == "npa.gemini_robotics.plan.v1"
    assert receipt["plan_text"].startswith("1. Approach.")
    assert receipt["model_function_calls"] == [{"name": "unrequested_call", "args": {}}]
    assert receipt["advisory_only"] is True
    assert receipt["artifact_path"] == "s3://example-bucket/receipts/plan.json"
    stored = storage.objects[receipt["artifact_path"]]
    assert hashlib.sha256(stored).hexdigest() == receipt["artifact_sha256"]
    assert json.loads(stored)["plan_text"] == receipt["plan_text"]


def test_eval_stage_writes_conditional_s3_receipt() -> None:
    storage = FakeStorage()
    plan_receipt = {"plan_text": "1. Approach.\n2. Grasp."}
    receipt = run_eval_stage(
        _config(output_path="s3://example-bucket/receipts/eval.json"),
        plan_receipt,
        "safety first",
        FakeClient(),
        storage,
        input_path="s3://example-bucket/inputs/eval.json",
        input_etag="input-etag",
        input_sha256="a" * 64,
    )
    assert receipt["schema"] == "npa.gemini_robotics.eval.v1"
    assert receipt["scores"] == {"safety": 9}
    assert receipt["artifact_etag"] == "etag-1"
    assert receipt["input_path"] == "s3://example-bucket/inputs/eval.json"
    assert receipt["input_etag"] == "input-etag"
    assert receipt["input_sha256"] == "a" * 64


def test_eval_input_requires_s3_json_plan_and_rubric() -> None:
    storage = FakeStorage()
    source_uri = "s3://example-bucket/inputs/eval.json"
    storage.objects[source_uri] = json.dumps(
        {"plan_text": "plan", "rubric": "safety first"}
    ).encode()
    payload, input_uri, etag, digest = read_eval_input(source_uri, storage)
    assert input_uri == source_uri
    assert etag == "input-etag"
    assert digest == hashlib.sha256(storage.objects[source_uri]).hexdigest()
    assert payload == {"plan_text": "plan", "rubric": "safety first"}


@pytest.mark.parametrize("output_path", ["", "/tmp/receipt.json", "s3://bucket/"])
def test_planning_rejects_invalid_output_before_client(output_path: str) -> None:
    client = FakeClient()
    with pytest.raises(GeminiRoboticsPipelineError):
        run_er_planning_stage(_config(output_path=output_path), client, FakeStorage())
    assert client.plans == 0


def test_planning_requires_explicit_model_before_client() -> None:
    client = FakeClient()
    with pytest.raises(GeminiRoboticsPipelineError, match="explicit model"):
        run_er_planning_stage(_config(model=""), client, FakeStorage())
    assert client.plans == 0


def _toolref_stage_argv(name: str) -> list[str]:
    import re

    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    entry = TOOL_CATALOG[name]
    return [
        re.sub(r"{{(.*?)}}", r"s3://example-bucket/DUMMY", arg)
        for arg in entry.argv_template
    ]


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
        assert args.api_base_url == "s3://example-bucket/DUMMY"
        assert args.output_path == "s3://example-bucket/DUMMY"


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
    from npa.workflows.byof import gemini_robotics_pipeline as pipe

    with pytest.raises(SystemExit):
        pipe.main(["plan", "--task", "t", "--model", "m", "--output-path", "s3://b/o"])
    with pytest.raises(SystemExit):
        pipe.main(
            [
                "--api-base-url",
                "https://example.test",
                "plan",
                "--task",
                "t",
                "--output-path",
                "s3://b/o",
            ]
        )
