"""Tests for the MolmoAct workbench / CLI / workflow scaffolding (issue #502)."""

from __future__ import annotations

import importlib
import json
import urllib.request

import pytest
import typer

import npa.workbench
from npa.cli.molmoact import (
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


def test_hf_model_repo_accessible():
    """allenai/MolmoAct-7B-O-0812 must be a public, ungated HF repo."""
    url = f"https://huggingface.co/api/models/{DEFAULT_MODEL_ID}"
    with urllib.request.urlopen(url, timeout=30) as response:
        assert response.status == 200
        payload = json.load(response)
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
        assert wrapper.__npa_cli_module__ == "npa.cli.molmoact"
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
    eval_manifest = pipeline_eval(
        {
            "model_id": DEFAULT_MODEL_ID,
            "dataset_uri": "s3://bucket/eval",
            "output_s3_uri": "s3://bucket/results",
        }
    )
    assert eval_manifest["command"] == "eval"
    with pytest.raises(MolmoActPipelineError):
        molmoact_pipeline.run("bogus", {})


def _toolref_stage_argv(name: str) -> "list[str]":
    import re

    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    def _value(match: "re.Match[str]") -> str:
        # --port is type=int in the pipeline argparse; use a valid int.
        return "8000" if match.group(1) == "config.serve_port" else "DUMMY"

    entry = TOOL_CATALOG[name]
    return [re.sub(r"{{(.*?)}}", _value, a) for a in entry.argv_template]


def test_toolref_argv_parses_against_pipeline() -> None:
    from npa.workflows.byof import molmoact_pipeline as pipe

    for tool_name, command in [
        ("workbench.molmoact.finetune", "finetune"),
        ("workbench.molmoact.serve", "serve"),
        ("workbench.molmoact.eval", "eval"),
    ]:
        argv = _toolref_stage_argv(tool_name)
        assert argv[:3] == ["python3", "-m", "npa.workflows.byof.molmoact_pipeline"]
        args = pipe.build_parser().parse_args(argv[3:])
        assert args.command == command


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
