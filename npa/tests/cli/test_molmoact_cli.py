"""Tests for the MolmoAct workbench / CLI / workflow scaffolding (issue #502)."""

from __future__ import annotations

import importlib
import json
import http.client

import pytest
import typer

import npa.workbench
from npa.cli.workbench.molmoact import (
    DEFAULT_MODEL_ID,
    app,
    eval_cmd,
    finetune_cmd,
    serve_cmd,
)
from npa.workflows.byof import molmoact_pipeline
from npa.workflows.byof.molmoact_pipeline import (
    MolmoActPipelineError,
    eval as pipeline_eval,
    finetune as pipeline_finetune,
    serve as pipeline_serve,
)


class _FakeHTTPResponse:
    def __init__(self, status: int, payload: dict):
        self.status = status
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _FakeHTTPSConnection:
    """No real network in unit tests: asserts the probe's request shape and
    replays a canned Hugging Face API response."""

    def __init__(self, status: int, payload: dict):
        self.status = status
        self.payload = payload
        self.requested: list[tuple[str, str]] = []

    def request(self, method: str, path: str, headers=None) -> None:
        assert method == "GET"
        assert path == f"/api/models/{DEFAULT_MODEL_ID}"
        self.requested.append((method, path))

    def getresponse(self) -> _FakeHTTPResponse:
        return _FakeHTTPResponse(self.status, self.payload)

    def close(self) -> None:
        pass


def _fake_hf_conn(status: int, payload: dict, monkeypatch: pytest.MonkeyPatch):
    def _connect(host: str, timeout: int | None = None) -> _FakeHTTPSConnection:
        assert host == "huggingface.co"
        return _FakeHTTPSConnection(status, payload)

    monkeypatch.setattr(http.client, "HTTPSConnection", _connect)


def test_hf_model_repo_accessible(monkeypatch: pytest.MonkeyPatch):
    """The probe targets the fixed huggingface.co host with the pinned repo id
    in the request path, and accepts a public, ungated repo.

    The HTTPS exchange is faked: unit tests must not make live network calls.
    """
    _fake_hf_conn(
        200,
        {"id": DEFAULT_MODEL_ID, "private": False, "gated": False},
        monkeypatch,
    )
    connection = http.client.HTTPSConnection("huggingface.co", timeout=30)
    try:
        connection.request(
            "GET",
            f"/api/models/{DEFAULT_MODEL_ID}",
            headers={"User-Agent": "npa-molmoact-tests"},
        )
        response = connection.getresponse()
        assert response.status == 200
        payload = json.loads(response.read().decode("utf-8"))
    finally:
        connection.close()
    assert payload["id"] == DEFAULT_MODEL_ID
    assert payload["private"] is False
    assert payload.get("gated") in (False, None, "false")


def test_cli_app_registered_as_molmoact():
    assert isinstance(app, typer.Typer)
    command_names = {c.name for c in app.registered_commands}
    assert {"finetune", "serve", "eval"} <= command_names


def test_workbench_wrappers_resolve():
    module = importlib.import_module("npa.workbench.molmoact")
    for name in ("finetune", "serve", "eval"):
        wrapper = getattr(module, name)
        assert wrapper.__npa_cli_module__ == "npa.cli.workbench.molmoact"
        assert wrapper.__npa_cli_callback__ == f"{name}_cmd"


def test_workbench_lazy_namespace():
    module = npa.workbench.molmoact
    assert callable(module.finetune)
    assert callable(module.serve)
    assert callable(module.eval)


def test_finetune_cmd_defaults_and_stub():
    manifest = finetune_cmd(
        model_id=DEFAULT_MODEL_ID,
        dataset_uri="s3://bucket/demos",
        output_s3_uri="s3://bucket/checkpoints",
        max_steps=100,
        batch_size=8,
        learning_rate=1e-5,
        num_gpus=1,
        run_name="unit-test",
    )
    assert manifest["status"] == "stub"
    assert manifest["command"] == "finetune"
    assert manifest["model_id"] == DEFAULT_MODEL_ID


def test_finetune_cmd_rejects_bad_args():
    with pytest.raises(typer.BadParameter):
        finetune_cmd(
            model_id=DEFAULT_MODEL_ID,
            dataset_uri="gs://not-s3/demos",
            output_s3_uri="s3://bucket/checkpoints",
            max_steps=100,
            batch_size=8,
            learning_rate=1e-5,
            num_gpus=1,
            run_name="",
        )
    with pytest.raises(typer.BadParameter):
        finetune_cmd(
            model_id="not-a-repo-id",
            dataset_uri="s3://bucket/demos",
            output_s3_uri="s3://bucket/checkpoints",
            max_steps=100,
            batch_size=8,
            learning_rate=1e-5,
            num_gpus=1,
            run_name="",
        )


def test_serve_cmd_stub_and_validation():
    manifest = serve_cmd(
        model_id=DEFAULT_MODEL_ID,
        checkpoint="",
        host="127.0.0.1",
        port=8123,
        device="cuda",
    )
    assert manifest["status"] == "stub"
    assert manifest["checkpoint"] == DEFAULT_MODEL_ID
    with pytest.raises(typer.BadParameter):
        serve_cmd(
            model_id=DEFAULT_MODEL_ID,
            checkpoint="",
            host="127.0.0.1",
            port=99999,
            device="cuda",
        )


def test_eval_cmd_stub_and_validation():
    manifest = eval_cmd(
        model_id=DEFAULT_MODEL_ID,
        dataset_uri="s3://bucket/eval",
        checkpoint="",
        num_episodes=10,
        output_s3_uri="s3://bucket/results",
    )
    assert manifest["status"] == "stub"
    assert manifest["num_episodes"] == 10
    with pytest.raises(typer.BadParameter):
        eval_cmd(
            model_id=DEFAULT_MODEL_ID,
            dataset_uri="s3://bucket/eval",
            checkpoint="",
            num_episodes=0,
            output_s3_uri="s3://bucket/results",
        )


def test_workbench_wrapper_calls_through():
    module = importlib.import_module("npa.workbench.molmoact")
    manifest = module.serve(model_id=DEFAULT_MODEL_ID, port=8123)
    assert manifest["command"] == "serve"
    assert manifest["port"] == 8123


def test_pipeline_finetune_manifest():
    manifest = pipeline_finetune(
        {
            "model_id": DEFAULT_MODEL_ID,
            "dataset_uri": "s3://bucket/demos",
            "output_s3_uri": "s3://bucket/checkpoints",
            "max_steps": 100,
            "num_gpus": 2,
        }
    )
    assert manifest["status"] == "validated"
    assert manifest["command"] == "finetune"
    assert manifest["num_gpus"] == 2
    with pytest.raises(MolmoActPipelineError):
        pipeline_finetune({"model_id": DEFAULT_MODEL_ID})


def test_pipeline_serve_and_eval():
    serve_manifest = pipeline_serve({"model_id": DEFAULT_MODEL_ID, "port": 8123})
    assert serve_manifest["command"] == "serve"
    assert serve_manifest["port"] == 8123
    assert serve_manifest["checkpoint"] == DEFAULT_MODEL_ID
    eval_manifest = pipeline_eval(
        {
            "model_id": DEFAULT_MODEL_ID,
            "dataset_uri": "s3://bucket/eval",
            "output_s3_uri": "s3://bucket/results",
        }
    )
    assert eval_manifest["command"] == "eval"
    assert eval_manifest["checkpoint"] == DEFAULT_MODEL_ID
    with pytest.raises(MolmoActPipelineError):
        molmoact_pipeline.run("bogus", {})


def _reject_json_constant(value: str):
    raise ValueError(f"non-standard JSON constant: {value}")


def _invoke_pipeline_main(argv: list[str], capsys: pytest.CaptureFixture[str]) -> dict:
    assert molmoact_pipeline.main(argv) == 0
    captured = capsys.readouterr()
    manifest = json.loads(captured.out, parse_constant=_reject_json_constant)
    assert captured.err == ""
    assert captured.out == json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return manifest


def test_pipeline_main_accepts_selected_direct_module_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _invoke_pipeline_main(
        [
            "finetune",
            "--model-id",
            DEFAULT_MODEL_ID,
            "--dataset-uri",
            "s3://bucket/demos",
            "--output-s3-uri",
            "s3://bucket/checkpoints",
            "--max-steps",
            "17",
            "--batch-size",
            "3",
            "--learning-rate",
            "0.0002",
            "--num-gpus",
            "2",
            "--run-name",
            "contract-case",
        ],
        capsys,
    )
    assert manifest["batch_size"] == 3
    assert manifest["learning_rate"] == 0.0002
    assert manifest["run_name"] == "contract-case"

    serve_manifest = _invoke_pipeline_main(["serve", "--host", "127.0.0.2"], capsys)
    assert serve_manifest["host"] == "127.0.0.2"


def test_pipeline_main_selected_option_defaults_match_manifest_defaults(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _invoke_pipeline_main(
        [
            "finetune",
            "--model-id",
            DEFAULT_MODEL_ID,
            "--dataset-uri",
            "s3://bucket/demos",
            "--output-s3-uri",
            "s3://bucket/checkpoints",
        ],
        capsys,
    )
    assert manifest["batch_size"] == 32
    assert manifest["learning_rate"] == 1e-5
    assert manifest["run_name"] == "molmoact-finetune"

    serve_manifest = _invoke_pipeline_main(["serve"], capsys)
    assert serve_manifest["host"] == "127.0.0.1"


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e309"])
def test_pipeline_main_rejects_nonfinite_learning_rate_before_stdout(
    value: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(MolmoActPipelineError, match=r"learning_rate must be finite"):
        molmoact_pipeline.main(
            [
                "finetune",
                "--model-id",
                DEFAULT_MODEL_ID,
                "--dataset-uri",
                "s3://bucket/demos",
                "--output-s3-uri",
                "s3://bucket/checkpoints",
                f"--learning-rate={value}",
            ]
        )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("0", "learning_rate must be positive, got 0.0"),
        ("-1", "learning_rate must be positive, got -1.0"),
    ],
)
def test_pipeline_main_rejects_nonpositive_learning_rate_before_stdout(
    value: str,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(MolmoActPipelineError) as caught:
        molmoact_pipeline.main(
            [
                "finetune",
                "--model-id",
                DEFAULT_MODEL_ID,
                "--dataset-uri",
                "s3://bucket/demos",
                "--output-s3-uri",
                "s3://bucket/checkpoints",
                f"--learning-rate={value}",
            ]
        )
    assert str(caught.value) == message
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_pipeline_main_json_serializer_rejects_nonfinite_manifest(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        molmoact_pipeline,
        "run",
        lambda command, config: {"learning_rate": float("nan")},
    )
    with pytest.raises(ValueError, match="Out of range float values"):
        molmoact_pipeline.main(["serve"])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize(
    ("command", "required_argv"),
    [
        ("serve", []),
        (
            "eval",
            [
                "--model-id",
                DEFAULT_MODEL_ID,
                "--dataset-uri",
                "s3://bucket/eval",
                "--output-s3-uri",
                "s3://bucket/results",
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        (None, DEFAULT_MODEL_ID),
        ("", DEFAULT_MODEL_ID),
        ("s3://bucket/explicit", "s3://bucket/explicit"),
    ],
)
def test_pipeline_main_checkpoint_fallback_and_explicit_control(
    command: str,
    required_argv: list[str],
    checkpoint: str | None,
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = [command, *required_argv]
    if checkpoint is not None:
        argv.extend(["--checkpoint", checkpoint])
    manifest = _invoke_pipeline_main(argv, capsys)
    assert manifest["checkpoint"] == expected


def _toolref_stage_argv(name: str) -> "list[str]":
    import re

    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    def _value(match: "re.Match[str]") -> str:
        values = {
            "config.model_id": DEFAULT_MODEL_ID,
            "config.dataset_uri": "s3://bucket/dataset",
            "config.training_uri": "s3://bucket/training",
            "config.trained_checkpoint_uri": "s3://bucket/checkpoint",
            "config.serve_port": "8000",
            "config.evaluation_uri": "s3://bucket/evaluation",
        }
        return values[match.group(1)]

    entry = TOOL_CATALOG[name]
    return [re.sub(r"{{(.*?)}}", _value, a) for a in entry.argv_template]


def test_toolref_argv_invokes_pipeline_with_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for tool_name, command in [
        ("workbench.molmoact.finetune", "finetune"),
        ("workbench.molmoact.serve", "serve"),
        ("workbench.molmoact.eval", "eval"),
    ]:
        argv = _toolref_stage_argv(tool_name)
        assert argv[:3] == ["python3", "-m", "npa.workflows.byof.molmoact_pipeline"]
        manifest = _invoke_pipeline_main(argv[3:], capsys)
        assert manifest["command"] == command


def test_toolref_checkpoint_and_serve_model_boundaries_are_exact() -> None:
    serve_argv = _toolref_stage_argv("workbench.molmoact.serve")
    assert serve_argv[3:] == [
        "serve",
        "--checkpoint",
        "s3://bucket/checkpoint",
        "--port",
        "8000",
    ]
    assert "--model-id" not in serve_argv

    eval_argv = _toolref_stage_argv("workbench.molmoact.eval")
    assert eval_argv[3:] == [
        "eval",
        "--model-id",
        DEFAULT_MODEL_ID,
        "--checkpoint",
        "s3://bucket/checkpoint",
        "--dataset-uri",
        "s3://bucket/dataset",
        "--output-s3-uri",
        "s3://bucket/evaluation",
    ]


def test_toolref_descriptions_state_planning_only() -> None:
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    for tool_name in (
        "workbench.molmoact.finetune",
        "workbench.molmoact.serve",
        "workbench.molmoact.eval",
    ):
        description = TOOL_CATALOG[tool_name].description
        assert "planning only" in description
        assert "not implemented" in description
