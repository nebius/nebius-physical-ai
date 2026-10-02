"""Exercise flex-pi CLI stdout and domain-error boundaries without GPUs."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.flex_pi.runtime import (
    FlexPiError,
    REAL_INFERENCE_MARKER,
    run_inference,
)


def _successful_runtime(argv, **kwargs):
    artifact = {
        "regime": "action-only",
        "actions": [[0.0] * 14 for _ in range(32)],
        "metrics": {"inference_seconds": 1.0},
        "runtime": {"cuda": True, "gpu_name": "NVIDIA B200"},
    }
    Path(argv[argv.index("--output-json") + 1]).write_text(json.dumps(artifact))
    return subprocess.CompletedProcess(argv, 0, stdout=REAL_INFERENCE_MARKER)


@pytest.mark.parametrize("compile_flag", ["--torch-compile", "--no-torch-compile"])
def test_real_success_stdout_is_one_json_document(tmp_path, monkeypatch, compile_flag):
    manifest = (
        Path(__file__).resolve().parents[2]
        / "docker/workbench/flex-pi/public_robotwin_sample.json"
    )

    def execute(request):
        assert not request.dry_run
        assert request.torch_compile == (compile_flag == "--torch-compile")
        return run_inference(
            replace(request, input_path=str(manifest), output_path=str(tmp_path)),
            runner=_successful_runtime,
        )

    monkeypatch.setattr("npa.cli.workbench.flex_pi.run_inference", execute)
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "flex-pi",
            "infer",
            "--output-path",
            "s3://fixture/output/",
            compile_flag,
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert set(payload["artifacts"]) == {"actions.json", "input.json", "result.json"}
    assert result.stderr == REAL_INFERENCE_MARKER + "\n"


def test_domain_error_exits_one_and_keeps_diagnostic_on_stderr(monkeypatch):
    def fail(request):
        raise FlexPiError("checkpoint validation failed")

    monkeypatch.setattr("npa.cli.workbench.flex_pi.run_inference", fail)
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "flex-pi",
            "infer",
            "--output-path",
            "s3://fixture/output/",
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["result"] == "error"
    assert "flex-pi inference failed: checkpoint validation failed" in result.stderr
    assert "Traceback" not in result.output


def test_explicit_json_format_does_not_enter_runtime_request(monkeypatch):
    def execute(request):
        assert request.dry_run
        return {"status": "dry_run"}

    monkeypatch.setattr("npa.cli.workbench.flex_pi.run_inference", execute)
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "flex-pi",
            "infer",
            "--output-path",
            "s3://fixture/output/",
            "--dry-run",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"status": "dry_run"}
