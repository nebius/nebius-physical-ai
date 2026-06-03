"""Credential and fetch helpers for public Cosmos3 model bundles."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

DEFAULT_CACHE_ENV = "NPA_COSMOS3_CACHE"
DEFAULT_CACHE_DIR = "/tmp/npa-cosmos3-cache"
DEFAULT_COSMOS3_MODEL_ID = "nvidia/Cosmos3-Nano"
DEFAULT_COSMOS3_SOURCE_REPO = "https://github.com/NVIDIA/cosmos-framework.git"
DEFAULT_GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
DEFAULT_HF_TOKEN_ENV = "HF_TOKEN"
DEFAULT_NGC_API_KEY_ENV = "NGC_API_KEY"
DEFAULT_REASONING_PARSER = "qwen3"
DEFAULT_TOOL_CALL_PARSER = "hermes"
COSMOS3_WORKFLOW_ROOT = Path("npa/workflows/workbench/skypilot")
COSMOS3_LICENSE = "OpenMDW-1.1"
COSMOS3_SKILL_SOURCE_ROOT = (
    "https://github.com/NVIDIA/cosmos-framework/tree/main/.agents/skills"
)

RunCallable = Callable[..., subprocess.CompletedProcess[str]]


class Cosmos3AccessError(RuntimeError):
    """Raised when Cosmos3 model access or fetch setup fails."""


@dataclass(frozen=True)
class Cosmos3SkillSpec:
    """NPA-authored integration metadata for a referenced NVIDIA Cosmos3 skill."""

    name: str
    nvidia_path: str
    purpose: str
    workflow: Path
    tier: str
    evidence: str
    capability: str
    generative: bool = False
    image: str = "source-based"
    integration_form: str = (
        "npa-authored-by-reference; NVIDIA skill files are not vendored"
    )

    @property
    def source_url(self) -> str:
        return f"{COSMOS3_SKILL_SOURCE_ROOT}/{self.name}/SKILL.md"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nvidia_path": self.nvidia_path,
            "source_url": self.source_url,
            "purpose": self.purpose,
            "workflow": str(self.workflow),
            "tier": self.tier,
            "evidence": self.evidence,
            "capability": self.capability,
            "generative": self.generative,
            "image": self.image,
            "integration_form": self.integration_form,
            "license": COSMOS3_LICENSE,
        }


@dataclass(frozen=True)
class Cosmos3SkillEnv:
    """Resolved environment overrides for a Cosmos3 skill SkyPilot workflow."""

    skill: str
    workflow: Path
    env: dict[str, str]
    no_guardrails: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "workflow": str(self.workflow),
            "env": dict(self.env),
            "no_guardrails": self.no_guardrails,
        }


@dataclass(frozen=True)
class Cosmos3ServeConfig:
    """vLLM serving knobs carried through fetch config for Phase 2."""

    reasoning_parser: str = DEFAULT_REASONING_PARSER
    tool_call_parser: str = DEFAULT_TOOL_CALL_PARSER

    def vllm_args(self) -> list[str]:
        """Return the parser flags expected by the vLLM serve path."""
        return [
            "--reasoning-parser",
            self.reasoning_parser,
            "--tool-call-parser",
            self.tool_call_parser,
        ]


@dataclass(frozen=True)
class Cosmos3AccessConfig:
    """Runtime configuration for a source checkout and HF checkpoint."""

    model_id: str = DEFAULT_COSMOS3_MODEL_ID
    source_repo_url: str = DEFAULT_COSMOS3_SOURCE_REPO
    cache_dir: Path | str | None = None
    github_token_env: str = DEFAULT_GITHUB_TOKEN_ENV
    hf_token_env: str = DEFAULT_HF_TOKEN_ENV
    ngc_api_key_env: str = DEFAULT_NGC_API_KEY_ENV
    require_ngc: bool = False
    serve: Cosmos3ServeConfig = field(default_factory=Cosmos3ServeConfig)

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        model_id: str = "",
        source_repo_url: str = "",
        cache_dir: Path | str | None = None,
        github_token_env: str = "",
        hf_token_env: str = "",
        ngc_api_key_env: str = "",
        require_ngc: bool | None = None,
        reasoning_parser: str = "",
        tool_call_parser: str = "",
    ) -> "Cosmos3AccessConfig":
        """Resolve config from explicit values first, then NPA_COSMOS3_* env."""
        env = environ if environ is not None else os.environ
        resolved_github_env = (
            github_token_env
            or env.get("NPA_COSMOS3_GITHUB_TOKEN_ENV", "")
            or DEFAULT_GITHUB_TOKEN_ENV
        )
        resolved_hf_env = (
            hf_token_env
            or env.get("NPA_COSMOS3_HF_TOKEN_ENV", "")
            or DEFAULT_HF_TOKEN_ENV
        )
        resolved_ngc_env = (
            ngc_api_key_env
            or env.get("NPA_COSMOS3_NGC_API_KEY_ENV", "")
            or DEFAULT_NGC_API_KEY_ENV
        )
        return cls(
            model_id=model_id
            or env.get("NPA_COSMOS3_MODEL_ID", "")
            or DEFAULT_COSMOS3_MODEL_ID,
            source_repo_url=source_repo_url
            or env.get("NPA_COSMOS3_SOURCE_REPO", "")
            or DEFAULT_COSMOS3_SOURCE_REPO,
            cache_dir=cache_dir or env.get(DEFAULT_CACHE_ENV, "") or DEFAULT_CACHE_DIR,
            github_token_env=resolved_github_env,
            hf_token_env=resolved_hf_env,
            ngc_api_key_env=resolved_ngc_env,
            require_ngc=_env_bool(env.get("NPA_COSMOS3_REQUIRE_NGC", "0"))
            if require_ngc is None
            else require_ngc,
            serve=Cosmos3ServeConfig(
                reasoning_parser=(
                    reasoning_parser
                    or env.get("NPA_COSMOS3_REASONING_PARSER", "")
                    or DEFAULT_REASONING_PARSER
                ),
                tool_call_parser=(
                    tool_call_parser
                    or env.get("NPA_COSMOS3_TOOL_CALL_PARSER", "")
                    or DEFAULT_TOOL_CALL_PARSER
                ),
            ),
        )

    @property
    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir or DEFAULT_CACHE_DIR).expanduser()


COSMOS3_SKILL_SPECS: tuple[Cosmos3SkillSpec, ...] = (
    Cosmos3SkillSpec(
        name="cosmos3-setup",
        nvidia_path=".agents/skills/cosmos3-setup/SKILL.md",
        purpose=(
            "Clone Cosmos3, install the selected CUDA extras group, and verify "
            "source/checkpoint access."
        ),
        workflow=COSMOS3_WORKFLOW_ROOT / "cosmos3-setup.yaml",
        tier="PARTIAL",
        evidence="YAML/SDK/CLI structure validated; live install requires GPU runtime.",
        capability="setup",
    ),
    Cosmos3SkillSpec(
        name="cosmos3-codebase-nav",
        nvidia_path=".agents/skills/cosmos3-codebase-nav/SKILL.md",
        purpose=(
            "Clone Cosmos3 and emit a machine-readable inventory of inference "
            "defaults, recipe TOMLs, scripts, and config locations."
        ),
        workflow=COSMOS3_WORKFLOW_ROOT / "cosmos3-codebase-nav.yaml",
        tier="PARTIAL",
        evidence="Static source navigation workflow validated without copying NVIDIA skill text.",
        capability="codebase-navigation",
    ),
    Cosmos3SkillSpec(
        name="cosmos3-env-troubleshoot",
        nvidia_path=".agents/skills/cosmos3-env-troubleshoot/SKILL.md",
        purpose=(
            "Collect Cosmos3 host, Python, CUDA, package, and checkpoint diagnostics "
            "for setup/inference failures."
        ),
        workflow=COSMOS3_WORKFLOW_ROOT / "cosmos3-env-troubleshoot.yaml",
        tier="PARTIAL",
        evidence="Diagnostic workflow is source-based and does not require a model backend.",
        capability="environment-troubleshooting",
    ),
    Cosmos3SkillSpec(
        name="cosmos3-inference",
        nvidia_path=".agents/skills/cosmos3-inference/SKILL.md",
        purpose=(
            "Run public Cosmos3 text-to-image inference through "
            "cosmos_framework.scripts.inference with guardrails on by default."
        ),
        workflow=COSMOS3_WORKFLOW_ROOT / "cosmos3-text-to-image-inference.yaml",
        tier="PARTIAL",
        evidence="Raw SkyPilot YAML, SDK env builder, and CLI inventory validate defaults.",
        capability="inference",
        generative=True,
    ),
    Cosmos3SkillSpec(
        name="cosmos3-post-training",
        nvidia_path=".agents/skills/cosmos3-post-training/SKILL.md",
        purpose=(
            "Stage the Cosmos3 SFT recipe flow with explicit dataset/checkpoint/Wan VAE "
            "inputs and an explicit plan/validate/train action."
        ),
        workflow=COSMOS3_WORKFLOW_ROOT / "cosmos3-post-training.yaml",
        tier="SEAM",
        evidence="Typed SFT extension point; full training is not faked without datasets/checkpoints.",
        capability="post-training",
    ),
)


def list_cosmos3_skills() -> tuple[Cosmos3SkillSpec, ...]:
    """Return the referenced NVIDIA Cosmos3 skills integrated into NPA."""

    return COSMOS3_SKILL_SPECS


def get_cosmos3_skill(name: str) -> Cosmos3SkillSpec:
    """Return a Cosmos3 skill spec by name, raising for unknown skills."""

    normalized = name.strip().lower()
    for spec in COSMOS3_SKILL_SPECS:
        if spec.name == normalized:
            return spec
    supported = ", ".join(spec.name for spec in COSMOS3_SKILL_SPECS)
    raise Cosmos3AccessError(f"Unknown Cosmos3 skill '{name}'. Supported: {supported}")


def build_cosmos3_inference_args(
    *,
    input_json: str,
    output_dir: str,
    checkpoint_path: str = "Cosmos3-Nano",
    seed: int = 0,
    no_guardrails: bool = False,
    parallelism_preset: str = "latency",
) -> list[str]:
    """Build Cosmos3 inference script arguments with guardrails on by default."""

    args = [
        "--parallelism-preset",
        parallelism_preset,
        "-i",
        input_json,
        "-o",
        output_dir,
        "--checkpoint-path",
        checkpoint_path,
    ]
    if no_guardrails:
        args.append("--no-guardrails")
    args.append(f"--seed={seed}")
    return args


def build_cosmos3_skill_env(
    skill: str,
    *,
    source_repo_url: str = "",
    model_id: str = "",
    cache_dir: str = "",
    github_token_env: str = "",
    hf_token_env: str = "",
    ngc_api_key_env: str = "",
    require_ngc: bool = False,
    output_s3_uri: str = "",
    prompt: str = "",
    uv_group: str = "",
    setup_json: str = "",
    nav_output: str = "",
    diagnostics_json: str = "",
    inference_output_dir: str = "",
    inference_output_image: str = "",
    inference_success_json: str = "",
    reasoning_parser: str = "",
    tool_call_parser: str = "",
    sft_recipe: str = "",
    sft_action: str = "",
    sft_validate_only: bool = False,
    sft_dataset_path: str = "",
    sft_base_checkpoint_path: str = "",
    sft_wan_vae_path: str = "",
    sft_output_root: str = "",
    sft_result_json: str = "",
    no_guardrails: bool = False,
) -> Cosmos3SkillEnv:
    """Resolve CLI/SDK parameters into SkyPilot env vars for a Cosmos3 skill."""

    spec = get_cosmos3_skill(skill)
    env: dict[str, str] = {}
    if source_repo_url:
        env["NPA_COSMOS3_SOURCE_REPO"] = source_repo_url
    if model_id:
        env["NPA_COSMOS3_MODEL_ID"] = model_id
    if cache_dir:
        env["NPA_COSMOS3_CACHE"] = cache_dir
    if github_token_env:
        env["NPA_COSMOS3_GITHUB_TOKEN_ENV"] = github_token_env
    if hf_token_env:
        env["NPA_COSMOS3_HF_TOKEN_ENV"] = hf_token_env
    if ngc_api_key_env:
        env["NPA_COSMOS3_NGC_API_KEY_ENV"] = ngc_api_key_env
    if require_ngc:
        env["NPA_COSMOS3_REQUIRE_NGC"] = "1"
    if output_s3_uri:
        env["NPA_COSMOS3_OUTPUT_S3_URI"] = output_s3_uri
    if uv_group:
        env["NPA_COSMOS3_UV_GROUP"] = uv_group
    if spec.generative:
        env["NPA_COSMOS3_NO_GUARDRAILS"] = "1" if no_guardrails else ""
    if spec.name == "cosmos3-setup" and setup_json:
        env["NPA_COSMOS3_SETUP_JSON"] = setup_json
    if spec.name == "cosmos3-codebase-nav" and nav_output:
        env["NPA_COSMOS3_NAV_OUTPUT"] = nav_output
    if spec.name == "cosmos3-env-troubleshoot" and diagnostics_json:
        env["NPA_COSMOS3_DIAGNOSTICS_JSON"] = diagnostics_json
    if spec.name == "cosmos3-inference":
        if prompt:
            env["NPA_COSMOS3_INFER_PROMPT"] = prompt
        if inference_output_dir:
            env["NPA_COSMOS3_OUTPUT_DIR"] = inference_output_dir
        if inference_output_image:
            env["NPA_COSMOS3_OUTPUT_IMAGE"] = inference_output_image
        if inference_success_json:
            env["NPA_COSMOS3_SUCCESS_JSON"] = inference_success_json
        if reasoning_parser:
            env["NPA_COSMOS3_REASONING_PARSER"] = reasoning_parser
        if tool_call_parser:
            env["NPA_COSMOS3_TOOL_CALL_PARSER"] = tool_call_parser
    if spec.name == "cosmos3-post-training":
        if sft_recipe:
            env["NPA_COSMOS3_SFT_RECIPE"] = sft_recipe
        if sft_action:
            env["NPA_COSMOS3_SFT_ACTION"] = sft_action
        if sft_validate_only:
            env["NPA_COSMOS3_SFT_VALIDATE_ONLY"] = "1"
        if sft_dataset_path:
            env["NPA_COSMOS3_SFT_DATASET_PATH"] = sft_dataset_path
        if sft_base_checkpoint_path:
            env["NPA_COSMOS3_SFT_BASE_CHECKPOINT_PATH"] = sft_base_checkpoint_path
        if sft_wan_vae_path:
            env["NPA_COSMOS3_SFT_WAN_VAE_PATH"] = sft_wan_vae_path
        if sft_output_root:
            env["IMAGINAIRE_OUTPUT_ROOT"] = sft_output_root
        if sft_result_json:
            env["NPA_COSMOS3_SFT_RESULT_JSON"] = sft_result_json
    return Cosmos3SkillEnv(
        skill=spec.name,
        workflow=spec.workflow,
        env=env,
        no_guardrails=no_guardrails,
    )


@dataclass(frozen=True)
class Cosmos3CheckResult:
    """Redacted access-check result suitable for CLI output and logs."""

    ok: bool
    github_auth: str
    source_repo: str
    hf_auth: str
    hf_model: str
    ngc_auth: str
    cache_dir: str
    reasoning_parser: str
    tool_call_parser: str
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ok else "failed",
            "github_auth": self.github_auth,
            "source_repo": self.source_repo,
            "hf_auth": self.hf_auth,
            "hf_model": self.hf_model,
            "ngc_auth": self.ngc_auth,
            "cache_dir": self.cache_dir,
            "reasoning_parser": self.reasoning_parser,
            "tool_call_parser": self.tool_call_parser,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class Cosmos3FetchResult:
    """Redacted fetch result with local ephemeral paths only."""

    ok: bool
    cache_dir: str
    source_checkout: str
    checkpoint_dir: str
    checkpoint: str
    reasoning_parser: str
    tool_call_parser: str
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ok else "failed",
            "cache_dir": self.cache_dir,
            "source_checkout": self.source_checkout,
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint": self.checkpoint,
            "reasoning_parser": self.reasoning_parser,
            "tool_call_parser": self.tool_call_parser,
            "errors": list(self.errors),
        }


def check_cosmos3_access(
    config: Cosmos3AccessConfig,
    *,
    environ: Mapping[str, str] | None = None,
    runner: RunCallable | None = None,
    timeout: float = 20.0,
) -> Cosmos3CheckResult:
    """Verify local auth and lightweight reachability for Cosmos3 artifacts."""
    env = dict(environ if environ is not None else os.environ)
    run = runner or subprocess.run
    errors: list[str] = []

    if not config.source_repo_url:
        errors.append("source repo URL is required")
    if not config.model_id:
        errors.append("HF model ID is required")

    github_auth = _check_github_auth(config, env, run, timeout)

    source_repo = "skipped"
    if config.source_repo_url:
        source_repo = _check_source_repo(config, env, run, timeout)
        if source_repo != "reachable":
            detail = "source repo is not reachable"
            if github_auth == "missing":
                detail += (
                    f"; set {config.github_token_env} or authenticate gh if using "
                    "a private source repo"
                )
            errors.append(detail)

    hf_auth = "configured" if env.get(config.hf_token_env, "") else "missing"
    if hf_auth == "missing":
        errors.append(f"Hugging Face auth missing: set {config.hf_token_env}")

    hf_model = "skipped"
    if config.model_id and hf_auth == "configured":
        hf_model = _check_hf_model(config, env, timeout)
        if hf_model != "reachable":
            errors.append("HF model metadata is not reachable with current auth")

    ngc_auth = "skipped"
    if config.require_ngc:
        ngc_auth = "configured" if env.get(config.ngc_api_key_env, "") else "missing"
        if ngc_auth == "missing":
            errors.append(f"NGC auth missing: set {config.ngc_api_key_env}")

    return Cosmos3CheckResult(
        ok=not errors,
        github_auth=github_auth,
        source_repo=source_repo,
        hf_auth=hf_auth,
        hf_model=hf_model,
        ngc_auth=ngc_auth,
        cache_dir=str(config.resolved_cache_dir),
        reasoning_parser=config.serve.reasoning_parser,
        tool_call_parser=config.serve.tool_call_parser,
        errors=tuple(errors),
    )


def fetch_cosmos3_artifacts(
    config: Cosmos3AccessConfig,
    *,
    environ: Mapping[str, str] | None = None,
    runner: RunCallable | None = None,
    download_checkpoint: bool = True,
    hf_include_patterns: Sequence[str] = (),
    hf_exclude_patterns: Sequence[str] = (),
    force: bool = False,
    timeout: float = 20.0,
) -> Cosmos3FetchResult:
    """Clone source and optionally download the HF checkpoint into runtime cache."""
    env = dict(environ if environ is not None else os.environ)
    run = runner or subprocess.run
    check = check_cosmos3_access(config, environ=env, runner=run, timeout=timeout)
    if not check.ok:
        return Cosmos3FetchResult(
            ok=False,
            cache_dir=check.cache_dir,
            source_checkout="",
            checkpoint_dir="",
            checkpoint="skipped",
            reasoning_parser=config.serve.reasoning_parser,
            tool_call_parser=config.serve.tool_call_parser,
            errors=check.errors,
        )

    cache_dir = config.resolved_cache_dir
    source_dir = cache_dir / "source"
    checkpoint_dir = cache_dir / "checkpoint"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if force and source_dir.exists():
        shutil.rmtree(source_dir)
    if not source_dir.exists():
        result = _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                config.source_repo_url,
                str(source_dir),
            ],
            env=_git_env(config, env),
            run=run,
            timeout=timeout,
        )
        if result.returncode != 0:
            return _fetch_error(
                config,
                cache_dir,
                source_dir,
                checkpoint_dir,
                "source clone failed",
                _sanitize_output(result, config, env),
            )

    checkpoint_state = "skipped"
    if download_checkpoint:
        if force and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        download_cmd = [
            _huggingface_cli(),
            "download",
            config.model_id,
            "--local-dir",
            str(checkpoint_dir),
        ]
        for pattern in hf_include_patterns:
            download_cmd.extend(["--include", pattern])
        for pattern in hf_exclude_patterns:
            download_cmd.extend(["--exclude", pattern])
        result = _run(
            download_cmd,
            env=_hf_env(config, env),
            run=run,
            timeout=None,
        )
        if result.returncode != 0:
            return _fetch_error(
                config,
                cache_dir,
                source_dir,
                checkpoint_dir,
                "checkpoint download failed",
                _sanitize_output(result, config, env),
            )
        checkpoint_state = "downloaded"

    return Cosmos3FetchResult(
        ok=True,
        cache_dir=str(cache_dir),
        source_checkout=str(source_dir),
        checkpoint_dir=str(checkpoint_dir),
        checkpoint=checkpoint_state,
        reasoning_parser=config.serve.reasoning_parser,
        tool_call_parser=config.serve.tool_call_parser,
    )


def _fetch_error(
    config: Cosmos3AccessConfig,
    cache_dir: Path,
    source_dir: Path,
    checkpoint_dir: Path,
    message: str,
    detail: str,
) -> Cosmos3FetchResult:
    error = message if not detail else f"{message}: {detail}"
    return Cosmos3FetchResult(
        ok=False,
        cache_dir=str(cache_dir),
        source_checkout=str(source_dir),
        checkpoint_dir=str(checkpoint_dir),
        checkpoint="failed",
        reasoning_parser=config.serve.reasoning_parser,
        tool_call_parser=config.serve.tool_call_parser,
        errors=(error,),
    )


def _run(
    args: Sequence[str],
    *,
    env: Mapping[str, str],
    run: RunCallable,
    timeout: float | None,
) -> subprocess.CompletedProcess[str]:
    command = list(args)
    try:
        return run(
            command,
            env=dict(env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            exc.stdout or "",
            exc.stderr or str(exc),
        )


def _check_github_auth(
    config: Cosmos3AccessConfig,
    env: Mapping[str, str],
    run: RunCallable,
    timeout: float,
) -> str:
    if env.get(config.github_token_env, ""):
        return "configured"
    result = _run(
        ["gh", "auth", "status"],
        env=env,
        run=run,
        timeout=timeout,
    )
    return "gh" if result.returncode == 0 else "missing"


def _check_source_repo(
    config: Cosmos3AccessConfig,
    env: Mapping[str, str],
    run: RunCallable,
    timeout: float,
) -> str:
    result = _run(
        ["git", "ls-remote", "--exit-code", config.source_repo_url, "HEAD"],
        env=_git_env(config, env),
        run=run,
        timeout=timeout,
    )
    return "reachable" if result.returncode == 0 else "unreachable"


def _check_hf_model(
    config: Cosmos3AccessConfig,
    env: Mapping[str, str],
    timeout: float,
) -> str:
    headers = {"Authorization": f"Bearer {env[config.hf_token_env]}"}
    url = f"https://huggingface.co/api/models/{config.model_id}"
    try:
        response = httpx.head(
            url, headers=headers, timeout=timeout, follow_redirects=True
        )
        if response.status_code == 405:
            response = httpx.get(
                url,
                headers=headers,
                timeout=timeout,
                follow_redirects=True,
            )
    except httpx.HTTPError:
        return "unreachable"
    return "reachable" if 200 <= response.status_code < 400 else "unreachable"


def _git_env(config: Cosmos3AccessConfig, env: Mapping[str, str]) -> dict[str, str]:
    child = dict(env)
    child["GIT_TERMINAL_PROMPT"] = "0"
    token = child.get(config.github_token_env, "")
    if token:
        host = urlparse(config.source_repo_url).hostname or "github.com"
        child["GIT_CONFIG_COUNT"] = "1"
        child["GIT_CONFIG_KEY_0"] = f"http.https://{host}/.extraheader"
        child["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: bearer {token}"
    return child


def _hf_env(config: Cosmos3AccessConfig, env: Mapping[str, str]) -> dict[str, str]:
    child = dict(env)
    token = child.get(config.hf_token_env, "")
    if token:
        child.setdefault("HF_TOKEN", token)
        child.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return child


def _huggingface_cli() -> str:
    return shutil.which("huggingface-cli") or "huggingface-cli"


def _sanitize_output(
    result: subprocess.CompletedProcess[str],
    config: Cosmos3AccessConfig,
    env: Mapping[str, str],
) -> str:
    text = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if not text:
        return ""
    redactions = [
        config.model_id,
        config.source_repo_url,
        env.get(config.github_token_env, ""),
        env.get(config.hf_token_env, ""),
        env.get(config.ngc_api_key_env, ""),
    ]
    sanitized = text[-2000:]
    for value in redactions:
        if value:
            sanitized = sanitized.replace(value, "<redacted>")
    return sanitized


def _env_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
