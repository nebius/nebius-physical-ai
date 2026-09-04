"""Owner-only OpenAI-compatible provider configuration for NPA agents."""

from __future__ import annotations

from typing import Any, Callable

import typer

from npa.clients.ssh import SSHClient
from npa.cli.agent_env_files import _load_agent_llm_config_file, _stage_private_text


def llm_config_file_option():
    return typer.Option(
        "",
        "--llm-config-file",
        help=(
            "Owner-only JSON file for one OpenAI-compatible provider; contains "
            "a separate owner-only API-key file path, never the key itself."
        ),
    )


def resolve_agent_llm_runtime(
    record: dict[str, Any],
    *,
    llm_config_file: str,
    requested_model: str,
    requested_models: list[str],
    defaults: tuple[str, str, str, tuple[str, ...]],
    normalize_models: Callable[[list[str]], list[str]],
) -> dict[str, Any]:
    """Resolve custom owner configuration or the existing Token Factory defaults."""

    default_provider, default_api_key, default_model, default_models = defaults
    requested_file = llm_config_file if isinstance(llm_config_file, str) else ""
    saved_llm = record.get("llm") if isinstance(record.get("llm"), dict) else {}
    saved_file = str(saved_llm.get("config_file") or "").strip()
    source = str(requested_file or saved_file).strip()
    if source:
        custom = _load_agent_llm_config_file(source)
        persisted = {
            key: custom[key]
            for key in (
                "provider",
                "base_url",
                "api_key_file",
                "model",
                "models",
                "timeout_seconds",
                "max_concurrency",
                "config_file",
            )
        }
        return {**custom, "persisted": persisted}

    model = str(requested_model or "").strip() or default_model
    extras = requested_models if requested_models else list(default_models)
    models = normalize_models([model, *extras])
    if isinstance(saved_llm.get("models"), list):
        models = normalize_models(
            [*models, *[str(item) for item in saved_llm.get("models", [])]]
        )
    if (
        not str(requested_model or "").strip()
        and isinstance(saved_llm.get("model"), str)
        and saved_llm["model"].strip()
    ):
        model = saved_llm["model"].strip()
    if model not in models:
        models.insert(0, model)
    return {
        "provider": default_provider,
        "base_url": "",
        "api_key": default_api_key,
        "model": model,
        "models": models,
        "timeout_seconds": 120.0,
        "max_concurrency": 8,
        "config_file": "",
        "persisted": {
            "provider": default_provider,
            "model": model,
            "models": models,
        },
    }


def write_agent_llm_env(
    ssh: SSHClient,
    *,
    api_key: str,
    provider: str,
    model: str,
    providers: list[str] | tuple[str, ...],
    models: list[str] | tuple[str, ...],
    base_url: str = "",
    timeout_seconds: float = 120.0,
    max_concurrency: int = 8,
) -> None:
    """Stage provider credentials without placing them in shell argv."""

    if not api_key.strip():
        return
    normalized_provider = provider.strip().lower().replace("-", "_") or "token_factory"
    configured_models = list(
        dict.fromkeys(
            str(item).strip() for item in [model, *models] if str(item).strip()
        )
    )
    configured_providers = list(
        dict.fromkeys(
            value
            for item in [normalized_provider, *providers]
            if (value := str(item).strip().lower().replace("-", "_"))
        )
    )
    key_env = (
        "NEBIUS_TOKEN_FACTORY_KEY"
        if normalized_provider in {"token_factory", "tokenfactory"}
        else f"NPA_AGENT_{normalized_provider.upper()}_API_KEY"
    )
    env_lines = [
        f"{key_env}={api_key.strip()}",
        f"NPA_AGENT_LLM_PROVIDER={normalized_provider}",
        f"NPA_AGENT_LLM_PROVIDERS={','.join(configured_providers)}",
        f"NPA_AGENT_LLM_MODEL={model}",
        f"NPA_AGENT_LLM_MODELS={','.join(configured_models)}",
        f"NPA_AGENT_LLM_TIMEOUT_SECONDS={float(timeout_seconds):g}",
        f"NPA_AGENT_LLM_MAX_CONCURRENCY={int(max_concurrency)}",
    ]
    if base_url.strip():
        env_lines.append(
            f"NPA_AGENT_{normalized_provider.upper()}_BASE_URL={base_url.strip().rstrip('/')}"
        )
    _stage_private_text(
        ssh,
        content="\n".join([*env_lines, ""]),
        target="/opt/npa-agent/llm.env",
    )


def bootstrap_agent_llm_kwargs(runtime: dict[str, Any]) -> dict[str, Any]:
    """Translate resolved provider state to the remote bootstrap contract."""

    return {
        "llm_provider": str(runtime["provider"]),
        "llm_model": str(runtime["model"]),
        "llm_models": [str(item) for item in runtime["models"]],
        "llm_base_url": str(runtime["base_url"]),
        "llm_timeout_seconds": float(runtime["timeout_seconds"]),
        "llm_max_concurrency": int(runtime["max_concurrency"]),
        "llm_api_key": str(runtime["api_key"]),
    }


__all__ = [
    "bootstrap_agent_llm_kwargs",
    "llm_config_file_option",
    "resolve_agent_llm_runtime",
    "write_agent_llm_env",
]
