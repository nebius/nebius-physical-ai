from __future__ import annotations

import json

from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.cosmos.cosmos3 import Cosmos3CheckResult, Cosmos3FetchResult


runner = CliRunner()


def test_cosmos3_check_cli_outputs_redacted_json(mocker) -> None:
    check = mocker.patch(
        "npa.cli.cosmos.check_cosmos3_access",
        return_value=Cosmos3CheckResult(
            ok=True,
            github_auth="configured",
            source_repo="reachable",
            hf_auth="configured",
            hf_model="reachable",
            ngc_auth="skipped",
            cache_dir="/tmp/npa-cosmos3-cache",
            reasoning_parser="qwen3",
            tool_call_parser="hermes",
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "check",
            "--model-id",
            "org/private-model",
            "--source-repo-url",
            "https://github.com/org/private-repo.git",
            "--output",
            "json",
        ],
        env={"GITHUB_TOKEN": "gh-secret", "HF_TOKEN": "hf-secret"},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["reasoning_parser"] == "qwen3"
    assert payload["tool_call_parser"] == "hermes"
    assert "org/private-model" not in result.output
    assert "https://github.com/org/private-repo.git" not in result.output
    cfg = check.call_args.args[0]
    assert cfg.model_id == "org/private-model"
    assert cfg.source_repo_url == "https://github.com/org/private-repo.git"


def test_cosmos3_fetch_cli_exits_nonzero_on_failed_result(mocker) -> None:
    mocker.patch(
        "npa.cli.cosmos.fetch_cosmos3_artifacts",
        return_value=Cosmos3FetchResult(
            ok=False,
            cache_dir="/tmp/npa-cosmos3-cache",
            source_checkout="",
            checkpoint_dir="",
            checkpoint="skipped",
            reasoning_parser="qwen3",
            tool_call_parser="hermes",
            errors=("HF model metadata is not reachable with current auth",),
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "fetch",
            "--model-id",
            "org/private-model",
            "--source-repo-url",
            "https://github.com/org/private-repo.git",
            "--skip-checkpoint",
            "--output",
            "json",
        ],
        env={"GITHUB_TOKEN": "gh-secret", "HF_TOKEN": "hf-secret"},
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["checkpoint"] == "skipped"


def test_cosmos3_generate_dry_run_plans_with_guardrails_on(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos3",
            "generate",
            "--prompt",
            "a robot arm sorting blocks",
            "--output-path",
            str(tmp_path / "out"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "planned"
    assert payload["mode"] == "text2image"
    assert payload["guardrails"] is True
    assert payload["weights_baked"] is False
    assert "--no-guardrails" not in payload["argv"]
    assert "cosmos_framework.scripts.inference" in payload["argv"]


def test_cosmos3_generate_dry_run_opts_out_of_guardrails_explicitly(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos3",
            "generate",
            "--prompt",
            "a robot arm sorting blocks",
            "--output-path",
            str(tmp_path / "out"),
            "--no-guardrails",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["guardrails"] is False
    assert "--no-guardrails" in payload["argv"]


def test_cosmos3_generate_fails_clearly_without_the_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("COSMOS3_REPO", str(tmp_path / "missing"))

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos3",
            "generate",
            "--prompt",
            "a robot arm sorting blocks",
            "--output-path",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert "runtime is not present" in result.output


def test_cosmos3_generate_rejects_a_conditioned_mode_without_input(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos3",
            "generate",
            "--mode",
            "video2video",
            "--prompt",
            "a robot arm sorting blocks",
            "--output-path",
            str(tmp_path / "out"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 1
    assert "--input-path" in result.output


def test_cosmos3_skill_commands_are_not_cli_surface() -> None:
    result = runner.invoke(
        app,
        ["workbench", "cosmos", "--help"],
    )

    assert result.exit_code == 0
    assert " skills " not in result.output
    assert " skill " not in result.output
