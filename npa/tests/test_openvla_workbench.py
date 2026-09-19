"""Tests for the OpenVLA workbench ToolRefs (issue #500).

Covers the honest surface of this PR: upstream argv planning for
``vla-scripts/finetune.py`` / ``vla-scripts/deploy.py``, the public
HuggingFace accessibility check (mocked transport, no network), the
plan-only eval contract, and the catalog ToolEntry argv templates.

No GPU, no network access, and no upstream OpenVLA checkout required.
"""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from npa.cli.openvla import app as openvla_app
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.workflows.byof import openvla_pipeline as pipe

MODEL_ID = "openvla/openvla-7b"


def _toolref_stage_argv(name: str) -> list[str]:
    """Substitute dummy config values into a catalog ToolEntry argv template."""
    entry = TOOL_CATALOG[name]
    return [re.sub(r"{{(.*?)}}", r"DUMMY", a) for a in entry.argv_template]


class _FakeResponse:
    def __init__(self, status: int, payload: dict):
        self.status = status
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(status: int, payload: dict, monkeypatch):
    def _urlopen(request, timeout=None):
        assert "huggingface.co/api/models/" in request.full_url
        return _FakeResponse(status, payload)

    monkeypatch.setattr(pipe.urllib.request, "urlopen", _urlopen)


def test_train_argv_targets_upstream_finetune_script() -> None:
    cfg = pipe.TrainConfig(
        model_id=MODEL_ID, dataset_uri="s3://bucket/dataset", output_dir="runs/test"
    )
    argv = cfg.to_upstream_argv()
    assert argv[1].endswith("vla-scripts/finetune.py")
    assert "--vla_path" in argv and MODEL_ID in argv
    assert "--data_root_dir" in argv and "s3://bucket/dataset" in argv
    assert "--run_root_dir" in argv and "runs/test" in argv
    assert "--lora_rank" in argv and "--learning_rate" in argv


def test_train_rejects_empty_dataset() -> None:
    cfg = pipe.TrainConfig(model_id=MODEL_ID, dataset_uri="")
    with pytest.raises(pipe.OpenVLAPipelineError):
        cfg.validate()


def test_serve_argv_targets_upstream_deploy_script() -> None:
    cfg = pipe.ServeConfig(checkpoint="runs/openvla-oft", host="127.0.0.1", port=8123)
    argv = cfg.to_upstream_argv()
    assert argv[1].endswith("vla-scripts/deploy.py")
    assert "--openvla_path" in argv and "runs/openvla-oft" in argv
    assert "--host" in argv and "127.0.0.1" in argv
    assert "--port" in argv and "8123" in argv


def test_serve_rejects_bad_port() -> None:
    with pytest.raises(pipe.OpenVLAPipelineError):
        pipe.ServeConfig(checkpoint="ckpt", port=0).validate()


def test_eval_plan_contract_is_json_serializable() -> None:
    cfg = pipe.EvalConfig(
        checkpoint="runs/openvla-oft", dataset_uri="s3://bucket/eval", num_episodes=4
    )
    plan = cfg.plan()
    assert plan["checkpoint"] == "runs/openvla-oft"
    assert plan["dataset_uri"] == "s3://bucket/eval"
    assert plan["num_episodes"] == 4
    json.dumps(plan)


def test_eval_rejects_empty_checkpoint() -> None:
    with pytest.raises(pipe.OpenVLAPipelineError):
        pipe.EvalConfig(checkpoint="", dataset_uri="s3://bucket/eval").plan()


def test_eval_is_plan_only_even_when_not_dry_run(monkeypatch, capsys) -> None:
    """Eval never executes rollouts: non-dry-run still returns 0 without a subprocess."""

    def _no_exec(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("eval must not execute a subprocess")

    monkeypatch.setattr(pipe.subprocess, "run", _no_exec)
    cfg = pipe.EvalConfig(
        checkpoint="runs/openvla-oft", dataset_uri="s3://bucket/eval", dry_run=False
    )
    assert pipe.evaluate(cfg) == 0
    out = capsys.readouterr().out
    assert "stub" in out and "rollout" in out


def test_check_model_accessible_public_repo(monkeypatch) -> None:
    _fake_urlopen(200, {"id": MODEL_ID, "private": False}, monkeypatch)
    assert pipe.check_model_accessible(MODEL_ID) is True


def test_check_model_accessible_private_repo(monkeypatch) -> None:
    _fake_urlopen(200, {"id": MODEL_ID, "private": True}, monkeypatch)
    assert pipe.check_model_accessible(MODEL_ID) is False


def test_check_model_accessible_http_error(monkeypatch) -> None:
    _fake_urlopen(404, {}, monkeypatch)
    assert pipe.check_model_accessible("no/such-model") is False


def test_check_model_accessible_network_failure(monkeypatch) -> None:
    def _boom(request, timeout=None):
        raise OSError("no network")

    monkeypatch.setattr(pipe.urllib.request, "urlopen", _boom)
    assert pipe.check_model_accessible(MODEL_ID) is False


@pytest.mark.parametrize(
    ("tool_name", "stage"),
    [
        ("workbench.openvla.train", "train"),
        ("workbench.openvla.serve", "serve"),
        ("workbench.openvla.eval", "eval"),
    ],
)
def test_toolref_argv_parses_against_pipeline(tool_name: str, stage: str) -> None:
    argv = _toolref_stage_argv(tool_name)
    assert argv[:3] == ["python3", "-m", "npa.workflows.byof.openvla_pipeline"]
    args = pipe.build_parser().parse_args(argv[3:])
    assert args.command == stage


def test_cli_eval_dry_run_prints_plan() -> None:
    result = CliRunner().invoke(
        openvla_app,
        [
            "eval",
            "--checkpoint",
            "runs/openvla-oft",
            "--dataset-uri",
            "s3://bucket/eval",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "runs/openvla-oft" in result.output
