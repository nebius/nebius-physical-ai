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


def test_cosmos3_skill_commands_are_not_cli_surface() -> None:
    result = runner.invoke(
        app,
        ["workbench", "cosmos", "--help"],
    )

    assert result.exit_code == 0
    assert " skills " not in result.output
    assert " skill " not in result.output


def test_cosmos_augment_cli_dry_run_maps_flags_to_sky_env() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "augment",
            "--source",
            "s3://example-bucket/input/sim.mp4",
            "--output",
            "s3://example-bucket/output/augment/",
            "--prompt",
            "preserve layout",
            "--control",
            "blur",
            "--variants",
            "2",
            "--replicas",
            "3",
            "--image",
            "registry.example/npa-cosmos:3.0.0",
            "--s3-endpoint",
            "https://storage.example.invalid",
            "--dry-run",
            "--format",
            "json",
        ],
        env={"HF_TOKEN": "hf-secret"},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "dry_run"
    assert payload["env"]["NPA_COSMOS_AUGMENT_CONTROL"] == "vis"
    assert payload["env"]["NPA_COSMOS_AUGMENT_VARIANTS"] == "2"
    assert payload["env"]["NPA_COSMOS_REPLICAS"] == "3"
    assert "--gpus" not in payload["command"]
    assert "--num-nodes" in payload["command"]


def test_cosmos_reason_cli_dry_run_maps_model_size_and_accelerator() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos",
            "reason",
            "--input",
            "s3://example-bucket/input/rollout.mp4",
            "--output",
            "s3://example-bucket/output/reason/",
            "--criteria-prompt",
            "did the robot complete the task?",
            "--model-size",
            "super",
            "--accelerator",
            "CUSTOMGPU:1",
            "--dry-run",
            "--format",
            "json",
        ],
        env={"HF_TOKEN": "hf-secret"},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "dry_run"
    assert payload["env"]["NPA_COSMOS_REASON_CHECKPOINT"] == "Cosmos3-Super"
    assert payload["env"]["NPA_COSMOS_REASON_MODEL_ID"] == "nvidia/Cosmos3-Super"
    assert "--gpus" in payload["command"]
    assert "CUSTOMGPU:1" in payload["command"]


def test_cosmos_new_workflows_do_not_expose_guardrail_disable_flags() -> None:
    for command in ("augment", "reason"):
        result = runner.invoke(app, ["workbench", "cosmos", command, "--help"])
        assert result.exit_code == 0
        assert "--no-guardrails" not in result.output
