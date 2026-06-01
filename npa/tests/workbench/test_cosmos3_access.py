from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx

from npa.workbench.cosmos.cosmos3 import (
    DEFAULT_REASONING_PARSER,
    DEFAULT_TOOL_CALL_PARSER,
    Cosmos3AccessConfig,
    check_cosmos3_access,
    fetch_cosmos3_artifacts,
)


def _runner(returncode: int = 0):
    calls: list[tuple[list[str], dict]] = []

    def run(args, **kwargs):
        command = list(args)
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, "ok", "")

    return run, calls


def test_cosmos3_from_env_resolves_runtime_knobs(tmp_path: Path) -> None:
    cfg = Cosmos3AccessConfig.from_env(
        environ={
            "NPA_COSMOS3_MODEL_ID": "org/private-model",
            "NPA_COSMOS3_SOURCE_REPO": "https://github.com/org/private-repo.git",
            "NPA_COSMOS3_CACHE": str(tmp_path),
            "NPA_COSMOS3_GITHUB_TOKEN_ENV": "CUSTOM_GH",
            "NPA_COSMOS3_HF_TOKEN_ENV": "CUSTOM_HF",
            "NPA_COSMOS3_NGC_API_KEY_ENV": "CUSTOM_NGC",
            "NPA_COSMOS3_REQUIRE_NGC": "1",
            "NPA_COSMOS3_REASONING_PARSER": "qwen3",
            "NPA_COSMOS3_TOOL_CALL_PARSER": "hermes",
        }
    )

    assert cfg.model_id == "org/private-model"
    assert cfg.source_repo_url == "https://github.com/org/private-repo.git"
    assert cfg.resolved_cache_dir == tmp_path
    assert cfg.github_token_env == "CUSTOM_GH"
    assert cfg.hf_token_env == "CUSTOM_HF"
    assert cfg.ngc_api_key_env == "CUSTOM_NGC"
    assert cfg.require_ngc is True
    assert cfg.serve.vllm_args() == [
        "--reasoning-parser",
        DEFAULT_REASONING_PARSER,
        "--tool-call-parser",
        DEFAULT_TOOL_CALL_PARSER,
    ]


def test_cosmos3_check_is_redacted_and_uses_env_auth(mocker, tmp_path: Path) -> None:
    mocker.patch("httpx.head", return_value=httpx.Response(200))
    run, calls = _runner()
    cfg = Cosmos3AccessConfig(
        model_id="org/private-model",
        source_repo_url="https://github.com/org/private-repo.git",
        cache_dir=tmp_path,
    )

    result = check_cosmos3_access(
        cfg,
        environ={"GITHUB_TOKEN": "gh-secret", "HF_TOKEN": "hf-secret"},
        runner=run,
    )

    assert result.ok is True
    assert result.github_auth == "configured"
    assert result.source_repo == "reachable"
    assert result.hf_model == "reachable"
    assert calls[0][0][:3] == ["git", "ls-remote", "--exit-code"]
    rendered = json.dumps(result.as_dict())
    assert "org/private-model" not in rendered
    assert "https://github.com/org/private-repo.git" not in rendered
    assert "gh-secret" not in rendered
    assert "hf-secret" not in rendered


def test_cosmos3_check_reports_missing_inputs(tmp_path: Path) -> None:
    run, calls = _runner()
    result = check_cosmos3_access(
        Cosmos3AccessConfig(cache_dir=tmp_path),
        environ={},
        runner=run,
    )

    assert result.ok is False
    assert "source repo URL is required" in result.errors
    assert "HF model ID is required" in result.errors
    assert "Hugging Face auth missing: set HF_TOKEN" in result.errors
    assert calls[0][0] == ["gh", "auth", "status"]


def test_cosmos3_fetch_clones_and_downloads_without_token_args(
    mocker,
    tmp_path: Path,
) -> None:
    mocker.patch("httpx.head", return_value=httpx.Response(200))
    run, calls = _runner()
    cfg = Cosmos3AccessConfig(
        model_id="org/private-model",
        source_repo_url="https://github.com/org/private-repo.git",
        cache_dir=tmp_path,
    )

    result = fetch_cosmos3_artifacts(
        cfg,
        environ={"GITHUB_TOKEN": "gh-secret", "HF_TOKEN": "hf-secret"},
        runner=run,
        hf_include_patterns=("config.json",),
    )

    assert result.ok is True
    assert result.source_checkout == str(tmp_path / "source")
    assert result.checkpoint_dir == str(tmp_path / "checkpoint")
    assert result.checkpoint == "downloaded"
    commands = [call[0] for call in calls]
    assert commands[0][:3] == ["git", "ls-remote", "--exit-code"]
    assert commands[1][:3] == ["git", "clone", "--depth"]
    assert commands[2][:2] == ["huggingface-cli", "download"]
    assert "--include" in commands[2]
    assert "gh-secret" not in " ".join(" ".join(command) for command in commands)
    assert "hf-secret" not in " ".join(" ".join(command) for command in commands)
    assert calls[1][1]["env"]["GIT_CONFIG_VALUE_0"] == "AUTHORIZATION: bearer gh-secret"
    assert calls[2][1]["env"]["HF_TOKEN"] == "hf-secret"


def test_cosmos3_fetch_can_clone_source_without_checkpoint(mocker, tmp_path: Path) -> None:
    mocker.patch("httpx.head", return_value=httpx.Response(200))
    run, calls = _runner()
    cfg = Cosmos3AccessConfig(
        model_id="org/private-model",
        source_repo_url="https://github.com/org/private-repo.git",
        cache_dir=tmp_path,
    )

    result = fetch_cosmos3_artifacts(
        cfg,
        environ={"GITHUB_TOKEN": "gh-secret", "HF_TOKEN": "hf-secret"},
        runner=run,
        download_checkpoint=False,
    )

    assert result.ok is True
    assert result.checkpoint == "skipped"
    assert [call[0][0] for call in calls] == ["git", "git"]
