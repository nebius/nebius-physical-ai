from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
import yaml

from npa.workbench.cosmos.cosmos3 import (
    DEFAULT_COSMOS3_MODEL_ID,
    DEFAULT_COSMOS3_SOURCE_REPO,
    DEFAULT_REASONING_PARSER,
    DEFAULT_TOOL_CALL_PARSER,
    Cosmos3AccessConfig,
    build_cosmos3_inference_args,
    check_cosmos3_access,
    fetch_cosmos3_artifacts,
)


ROOT = Path(__file__).resolve().parents[3]
SKYPILOT_ROOT = ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot"
SPEC_YAML = (
    Path(__file__).resolve().parents[3]
    / "npa/workflows/workbench/npa-workflows/cosmos3-text-to-image.yaml"
)
# The raw template is retired; its spec is the surface (EVIDENCE.md §R43).
SPEC_YAML = (
    Path(__file__).resolve().parents[3]
    / "npa/workflows/workbench/npa-workflows/cosmos3-text-to-image.yaml"
)
SKILL_ROOT = ROOT / "skills"
SKILL_INDEX = SKILL_ROOT / "index.yaml"


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


def test_cosmos3_check_uses_public_defaults_anonymously(
    mocker,
    tmp_path: Path,
) -> None:
    mocker.patch("httpx.head", return_value=httpx.Response(200))
    calls: list[tuple[list[str], dict]] = []

    def run(args, **kwargs):
        command = list(args)
        calls.append((command, kwargs))
        returncode = 1 if command == ["gh", "auth", "status"] else 0
        return subprocess.CompletedProcess(command, returncode, "ok", "")

    result = check_cosmos3_access(
        Cosmos3AccessConfig(cache_dir=tmp_path),
        environ={},
        runner=run,
    )

    assert result.ok is True
    assert result.github_auth == "missing"
    assert result.source_repo == "reachable"
    assert result.hf_auth == "anonymous"
    assert result.hf_model == "reachable"
    assert calls[0][0] == ["gh", "auth", "status"]
    assert calls[1][0] == [
        "git",
        "ls-remote",
        "--exit-code",
        DEFAULT_COSMOS3_SOURCE_REPO,
        "HEAD",
    ]


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
    assert Path(commands[2][0]).name == "huggingface-cli"
    assert commands[2][1] == "download"
    assert "--include" in commands[2]
    assert "gh-secret" not in " ".join(" ".join(command) for command in commands)
    assert "hf-secret" not in " ".join(" ".join(command) for command in commands)
    assert calls[1][1]["env"]["GIT_CONFIG_VALUE_0"] == "AUTHORIZATION: bearer gh-secret"
    assert calls[2][1]["env"]["HF_TOKEN"] == "hf-secret"


def test_cosmos3_fetch_can_clone_source_without_checkpoint(
    mocker, tmp_path: Path
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
        download_checkpoint=False,
    )

    assert result.ok is True
    assert result.checkpoint == "skipped"
    assert [call[0][0] for call in calls] == ["git", "git"]


def test_cosmos3_text_to_image_spec_defaults_to_the_public_framework_and_model() -> (
    None
):
    """The template's `envs:` were its contract; the spec's `config:` is.

    Live proof that the spec runs: job 320 generated a 960x960 image from the default prompt
    (EVIDENCE.md §R43). What the template expressed as a hundred lines of bash in an env var is
    `npa workbench cosmos3 text-to-image`, so this asserts the reachable knobs rather than the
    text of a script.
    """

    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    spec = load_spec(SPEC_YAML)
    config = spec.config
    assert config["cosmos_source_repo"] == DEFAULT_COSMOS3_SOURCE_REPO
    assert config["cosmos_model_id"] == DEFAULT_COSMOS3_MODEL_ID

    argv = next(step.argv for step in build_plan(spec, run_id="t2i").steps if step.argv)
    assert argv[:4] == ["npa", "workbench", "cosmos3", "text-to-image"]
    assert argv[argv.index("--checkpoint-name") + 1] == "Cosmos3-Nano"
    # Guardrails default off, as the template's NPA_COSMOS3_NO_GUARDRAILS did: they pull further
    # gated weights.
    assert "--no-guardrails" in argv


def test_cosmos3_agent_skills_are_discoverable_and_well_formed() -> None:
    expected = {
        "cosmos3-setup",
        "cosmos3-codebase-nav",
        "cosmos3-env-troubleshoot",
        "cosmos3-inference",
        "cosmos3-post-training",
    }
    index = yaml.safe_load(SKILL_INDEX.read_text())
    entries = {entry["name"]: entry for entry in index["skills"]}

    for name in expected:
        assert name in entries
        path = ROOT / entries[name]["path"]
        assert path.exists(), name
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        frontmatter = text.split("---\n", 2)[1]
        parsed = yaml.safe_load(frontmatter)
        assert parsed["name"] == name
        assert parsed["description"]
        assert "Source And Attribution" in text
        assert "NVIDIA CORPORATION & AFFILIATES" in text
        assert "LICENSE-NVIDIA-COSMOS3-OPENMDW-1.1" in text

    assert (SKILL_ROOT / "LICENSE-NVIDIA-COSMOS3-OPENMDW-1.1").exists()
    assert (SKILL_ROOT / "NOTICE-NVIDIA-COSMOS3").exists()


def test_cosmos3_inference_args_keep_guardrails_on_by_default() -> None:
    args = build_cosmos3_inference_args(input_json="input.json", output_dir="out")

    assert "--no-guardrails" not in args
    assert args == [
        "--parallelism-preset",
        "latency",
        "-i",
        "input.json",
        "-o",
        "out",
        "--checkpoint-path",
        "Cosmos3-Nano",
        "--seed=0",
    ]
    assert "--no-guardrails" in build_cosmos3_inference_args(
        input_json="input.json",
        output_dir="out",
        no_guardrails=True,
    )


def test_cosmos3_removed_skill_workflow_yaml_files_stay_removed() -> None:
    removed = {
        "cosmos3-setup.yaml",
        "cosmos3-codebase-nav.yaml",
        "cosmos3-env-troubleshoot.yaml",
        "cosmos3-post-training.yaml",
    }

    for filename in removed:
        assert not (SKYPILOT_ROOT / filename).exists()
