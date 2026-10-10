"""Unit tests for the hosted Gemini Robotics workflow stages.

The client and storage boundaries are both faked: these prove local contracts,
not provider acceptance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest
import yaml

from npa.clients.gemini_robotics import EvalResult, PlanResult
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


def _use_missing_credentials_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Keep missing-key tests independent of the operator's saved credentials."""
    import npa.clients.credentials as creds_mod

    monkeypatch.setattr(creds_mod, "CREDENTIALS_PATH", tmp_path / "missing.yaml")


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


@pytest.mark.parametrize("output_path", ["", "file:///receipt.json", "s3://bucket/"])
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


@pytest.mark.parametrize("stage", ["plan", "eval"])
@pytest.mark.parametrize("missing", ["api_key", "api_base_url", "model"])
def test_pipeline_stages_reject_missing_config_before_storage_or_http(
    monkeypatch: pytest.MonkeyPatch, tmp_path, stage: str, missing: str
) -> None:
    """SDK stage defaults fail before selecting storage or provider transport."""
    from npa.workflows.byof import gemini_robotics_pipeline as pipe

    attempts = {
        "storage_construct": 0,
        "storage_download": 0,
        "storage_read": 0,
        "storage_write": 0,
        "http": 0,
    }

    class StorageProbe:
        def download_file(self, *_args: object) -> None:
            attempts["storage_download"] += 1

        def read_bytes_with_etag(self, *_args: object) -> None:
            attempts["storage_read"] += 1
            return None

        def put_bytes_conditional(self, *_args: object, **_kwargs: object) -> str:
            attempts["storage_write"] += 1
            return "unexpected-write"

    def storage_factory(_cls: object) -> StorageProbe:
        attempts["storage_construct"] += 1
        return StorageProbe()

    def http_request(*_args: object, **_kwargs: object) -> httpx.Response:
        attempts["http"] += 1
        raise AssertionError("incomplete config must not attempt HTTP")

    monkeypatch.setattr(
        pipe.StorageClient, "from_environment", classmethod(storage_factory)
    )
    monkeypatch.setattr(httpx.Client, "request", http_request)
    _use_missing_credentials_file(monkeypatch, tmp_path)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_ROBOTICS_BASE_URL", raising=False)
    if missing != "api_key":
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    if missing != "api_base_url":
        monkeypatch.setenv(
            "GEMINI_ROBOTICS_BASE_URL", "https://provider.example.invalid"
        )

    config = _config(model="" if missing == "model" else "operator-selected-model")
    with pytest.raises(GeminiRoboticsPipelineError):
        if stage == "plan":
            run_er_planning_stage(config, client=None, storage=None)
        else:
            run_eval_stage(
                config,
                {"plan_text": "plan"},
                "safety first",
                client=None,
                storage=None,
            )

    assert attempts == {
        "storage_construct": 0,
        "storage_download": 0,
        "storage_read": 0,
        "storage_write": 0,
        "http": 0,
    }


@pytest.mark.parametrize("command", ["plan", "eval"])
@pytest.mark.parametrize("missing", ["api_key", "api_base_url", "model"])
def test_pipeline_main_rejects_missing_config_before_storage_or_http(
    monkeypatch: pytest.MonkeyPatch, tmp_path, command: str, missing: str
) -> None:
    """Pipeline entrypoints reject incomplete config before storage side effects."""
    from npa.workflows.byof import gemini_robotics_pipeline as pipe

    attempts = {
        "storage_construct": 0,
        "storage_download": 0,
        "storage_read": 0,
        "storage_write": 0,
        "http": 0,
    }

    class StorageProbe:
        def download_file(self, *_args: object) -> None:
            attempts["storage_download"] += 1

        def read_bytes_with_etag(self, *_args: object) -> None:
            attempts["storage_read"] += 1
            return None

        def put_bytes_conditional(self, *_args: object, **_kwargs: object) -> str:
            attempts["storage_write"] += 1
            return "unexpected-write"

    def storage_factory(_storage: object) -> StorageProbe:
        attempts["storage_construct"] += 1
        return StorageProbe()

    def http_request(*_args: object, **_kwargs: object) -> httpx.Response:
        attempts["http"] += 1
        raise AssertionError("incomplete config must not attempt HTTP")

    monkeypatch.setattr(pipe, "_storage_or_default", storage_factory)
    monkeypatch.setattr(httpx.Client, "request", http_request)
    _use_missing_credentials_file(monkeypatch, tmp_path)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    if missing != "api_key":
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    args: list[str] = []
    if missing != "api_base_url":
        args.extend(["--api-base-url", "https://provider.example.invalid"])
    if command == "plan":
        args.extend(["plan", "--task", "inspect the scene"])
    else:
        args.extend(["eval", "--input-path", "s3://example-bucket/inputs/eval.json"])
    args.extend(
        [
            "--model",
            "" if missing == "model" else "operator-selected-model",
            "--output-path",
            "s3://example-bucket/outputs/receipt.json",
        ]
    )

    with pytest.raises((GeminiRoboticsPipelineError, SystemExit)):
        pipe.main(args)

    assert attempts == {
        "storage_construct": 0,
        "storage_download": 0,
        "storage_read": 0,
        "storage_write": 0,
        "http": 0,
    }
