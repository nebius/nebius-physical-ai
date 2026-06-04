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

RunCallable = Callable[..., subprocess.CompletedProcess[str]]


class Cosmos3AccessError(RuntimeError):
    """Raised when Cosmos3 model access or fetch setup fails."""


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


def build_cosmos3_inference_args(
    *,
    input_json: str,
    output_dir: str,
    checkpoint_path: str = "Cosmos3-Nano",
    seed: int = 0,
    parallelism_preset: str = "latency",
) -> list[str]:
    """Build Cosmos3 inference script arguments with guardrails on."""

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
    args.append(f"--seed={seed}")
    return args


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
