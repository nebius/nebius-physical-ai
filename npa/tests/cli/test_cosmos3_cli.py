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


def test_cosmos3_skills_cli_lists_integrated_nvidia_skills() -> None:
    result = runner.invoke(
        app,
        ["workbench", "cosmos", "skills", "--output", "json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert [skill["name"] for skill in payload["skills"]] == [
        "cosmos3-setup",
        "cosmos3-codebase-nav",
        "cosmos3-env-troubleshoot",
        "cosmos3-inference",
        "cosmos3-post-training",
    ]
    assert payload["integration_form"] == "npa-authored-by-reference"


def test_cosmos3_skill_cli_maps_no_guardrails_to_yaml_env() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "skill",
            "cosmos3-inference",
            "--prompt",
            "robot sorting blocks",
            "--no-guardrails",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["guardrails_default"] == "on"
    assert payload["env"]["NPA_COSMOS3_NO_GUARDRAILS"] == "1"
    assert payload["env"]["NPA_COSMOS3_INFER_PROMPT"] == "robot sorting blocks"
