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
from npa.workbench.cosmos.workflows import (
    COSMOS_ATTRIBUTION,
    build_cosmos_augment_env,
    build_cosmos_reason_env,
)


ROOT = Path(__file__).resolve().parents[3]
SKYPILOT_ROOT = ROOT / "npa" / "workflows" / "workbench" / "skypilot"
INFERENCE_YAML = SKYPILOT_ROOT / "cosmos3-text-to-image-inference.yaml"
AUGMENT_YAML = SKYPILOT_ROOT / "cosmos3-augment.yaml"
REASON_YAML = SKYPILOT_ROOT / "cosmos3-reason.yaml"
SKILL_ROOT = ROOT / ".agents" / "skills"
THIRD_PARTY_COSMOS_ROOT = ROOT / "third_party" / "nvidia-cosmos"


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


def test_cosmos3_check_uses_public_defaults_and_reports_missing_hf_auth(
    tmp_path: Path,
) -> None:
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

    assert result.ok is False
    assert result.github_auth == "missing"
    assert result.source_repo == "reachable"
    assert "Hugging Face auth missing: set HF_TOKEN" in result.errors
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
    assert commands[2][:2] == ["huggingface-cli", "download"]
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


def test_cosmos3_inference_yaml_defaults_to_public_cosmos3_and_allows_s3() -> None:
    docs = [
        doc
        for doc in yaml.safe_load_all(INFERENCE_YAML.read_text(encoding="utf-8"))
        if doc
    ]

    assert len(docs) == 1
    doc = docs[0]
    envs = doc["envs"]
    rendered = INFERENCE_YAML.read_text(encoding="utf-8")
    assert doc["name"] == "cosmos3-text-to-image-inference"
    assert "image_id" not in doc["resources"]
    assert envs["NPA_COSMOS3_SOURCE_REPO"] == DEFAULT_COSMOS3_SOURCE_REPO
    assert envs["NPA_COSMOS3_MODEL_ID"] == DEFAULT_COSMOS3_MODEL_ID
    assert (
        "python -m cosmos_framework.scripts.inference"
        in envs["NPA_COSMOS3_INFER_COMMAND"]
    )
    assert "--checkpoint-path Cosmos3-Nano" in envs["NPA_COSMOS3_INFER_COMMAND"]
    assert "NPA_COSMOS3_NO_GUARDRAILS" not in envs
    assert "--no-guardrails" not in envs["NPA_COSMOS3_INFER_COMMAND"]
    assert "npa workbench cosmos fetch" not in doc["run"]
    assert "git clone --depth 1" in doc["run"]
    assert "huggingface-cli download" in doc["run"]
    assert envs["NPA_COSMOS3_CACHE"].startswith("/tmp/")
    assert envs["NPA_COSMOS3_OUTPUT_DIR"].startswith("/tmp/")
    assert envs["NPA_COSMOS3_OUTPUT_IMAGE"].startswith("/tmp/")
    assert "NPA_COSMOS3_OUTPUT_S3_URI" in rendered
    assert "NPA_COSMOS3_SOURCE_REPO" in rendered
    assert "NPA_COSMOS3_MODEL_ID" in rendered


def test_cosmos3_agent_skills_are_discoverable_and_well_formed() -> None:
    expected = {
        "cosmos3-setup",
        "codebase-nav",
        "env-troubleshoot",
        "inference",
        "cosmos3-post-training",
    }

    for name in expected:
        path = SKILL_ROOT / name / "SKILL.md"
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


def test_cosmos_workflow_env_builders_are_generic_and_attributed() -> None:
    augment = build_cosmos_augment_env(
        source="s3://example-bucket/input/sim.mp4",
        output_path="s3://example-bucket/output/augment/",
        prompt="preserve motion",
        control="blur",
        variants=2,
        replicas=3,
        image="registry.example/npa-cosmos:3.0.0",
        s3_endpoint="https://storage.example.invalid",
    )
    reason = build_cosmos_reason_env(
        input_path="s3://example-bucket/input/rollout.mp4",
        output_path="s3://example-bucket/output/reason/",
        criteria_prompt="did it succeed?",
        model_size="super",
        replicas=2,
        image="registry.example/npa-cosmos:3.0.0",
        s3_endpoint="https://storage.example.invalid",
    )

    assert augment["NPA_COSMOS_AUGMENT_CONTROL"] == "vis"
    assert augment["NPA_COSMOS_AUGMENT_VARIANTS"] == "2"
    assert augment["NPA_COSMOS_REPLICAS"] == "3"
    assert augment["NPA_COSMOS_ATTRIBUTION"] == COSMOS_ATTRIBUTION
    assert reason["NPA_COSMOS_REASON_CHECKPOINT"] == "Cosmos3-Super"
    assert reason["NPA_COSMOS_REPLICAS"] == "2"
    assert reason["NPA_COSMOS_ATTRIBUTION"] == COSMOS_ATTRIBUTION


def test_cosmos_augment_and_reason_raw_yamls_are_standalone_and_safe() -> None:
    for path in (AUGMENT_YAML, REASON_YAML):
        docs = [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]
        assert len(docs) == 1
        doc = docs[0]
        text = path.read_text(encoding="utf-8")
        assert doc["resources"]["cloud"] == "kubernetes"
        assert "accelerators" not in doc["resources"]
        assert doc["resources"]["image_id"] == "docker:${NPA_COSMOS_IMAGE}"
        assert "npa workbench" not in doc["run"]
        assert "--no-guardrails" not in text
        assert "--endpoint-url" in text
        assert "guardrails" in text.lower()
        assert doc["envs"]["NPA_COSMOS_ATTRIBUTION"] == COSMOS_ATTRIBUTION


def test_cosmos_license_and_notice_are_vendored() -> None:
    open_model_license = THIRD_PARTY_COSMOS_ROOT / "NVIDIA_OPEN_MODEL_LICENSE.txt"
    openmdw_license = THIRD_PARTY_COSMOS_ROOT / "OPENMDW-1.1.txt"
    notice = THIRD_PARTY_COSMOS_ROOT / "NOTICE.txt"

    assert open_model_license.exists()
    assert openmdw_license.exists()
    assert notice.exists()
    assert "NVIDIA Open Model License Agreement" in open_model_license.read_text(
        encoding="utf-8"
    )
    assert "OpenMDW License Agreement" in openmdw_license.read_text(encoding="utf-8")
    notice_text = notice.read_text(encoding="utf-8")
    assert COSMOS_ATTRIBUTION in notice_text
    assert "No customer-facing flag" in notice_text


def test_cosmos3_removed_skill_workflow_yaml_files_stay_removed() -> None:
    removed = {
        "cosmos3-setup.yaml",
        "cosmos3-codebase-nav.yaml",
        "cosmos3-env-troubleshoot.yaml",
        "cosmos3-post-training.yaml",
    }

    for filename in removed:
        assert not (SKYPILOT_ROOT / filename).exists()
