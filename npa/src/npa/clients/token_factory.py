"""Native Nebius Token Factory client.

Nebius Token Factory is an OpenAI-compatible inference API for hosted open
models (text and vision). This module is the single source of truth for
resolving the base URL and API key and for issuing chat-completion and
model-listing requests. Workbench tools and workflows call into here instead of
duplicating endpoint, auth, or request-shaping logic.

The default base URL is ``https://api.tokenfactory.nebius.com/v1/`` and the
default credential is the ``NEBIUS_API_KEY`` environment variable, matching the
upstream Token Factory documentation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_API_KEY_ENV = "NEBIUS_API_KEY"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_TEXT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_VISION_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
# NVIDIA Cosmos3 Super-Reasoner: hosted vision-language physical-AI reasoner.
# Confirm availability for your key with `npa workbench token-factory models`.
DEFAULT_REASONER_MODEL = "nvidia/Cosmos3-Super-Reasoner"

BASE_URL_ENV_KEYS = (
    "NEBIUS_TOKEN_FACTORY_BASE_URL",
    "NEBIUS_BASE_URL",
)
API_KEY_ENV_KEYS = (
    DEFAULT_API_KEY_ENV,
    "NEBIUS_TOKEN_FACTORY_API_KEY",
)


class TokenFactoryError(RuntimeError):
    """Raised when a Token Factory request is misconfigured or fails."""


@dataclass(frozen=True)
class TokenFactoryConfig:
    """Resolved connection settings for Nebius Token Factory."""

    base_url: str
    api_key: str
    timeout_s: float = DEFAULT_TIMEOUT_S

    @property
    def chat_completions_url(self) -> str:
        return _join_path(self.base_url, "chat/completions")

    @property
    def models_url(self) -> str:
        return _join_path(self.base_url, "models")


def resolve_config(
    *,
    base_url: str = "",
    api_key: str = "",
    api_key_env: str = DEFAULT_API_KEY_ENV,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    environ: dict[str, str] | None = None,
    require_api_key: bool = True,
) -> TokenFactoryConfig:
    """Resolve Token Factory connection settings.

    Precedence is explicit argument, then environment override, then the
    documented default. The API key is read from ``api_key`` if supplied,
    otherwise from ``api_key_env`` and the standard Token Factory env keys.
    """

    env = environ if environ is not None else os.environ
    resolved_base = base_url.strip() or _first_env(env, BASE_URL_ENV_KEYS) or DEFAULT_BASE_URL
    resolved_key = api_key.strip()
    if not resolved_key:
        key_candidates = (api_key_env, *API_KEY_ENV_KEYS) if api_key_env else API_KEY_ENV_KEYS
        resolved_key = _first_env(env, key_candidates)
    if require_api_key and not resolved_key:
        raise TokenFactoryError(
            "Nebius Token Factory API key not found. Set NEBIUS_API_KEY in your "
            "environment or ~/.npa/credentials.yaml (tokens.NEBIUS_API_KEY)."
        )
    if timeout_s <= 0:
        raise TokenFactoryError("timeout_s must be positive")
    return TokenFactoryConfig(base_url=resolved_base, api_key=resolved_key, timeout_s=timeout_s)


class TokenFactoryClient:
    """Thin OpenAI-compatible client for Nebius Token Factory."""

    def __init__(
        self,
        config: TokenFactoryConfig | None = None,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._config = config or resolve_config()
        self._http_client = http_client

    @property
    def config(self) -> TokenFactoryConfig:
        return self._config

    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue a chat-completion request and return the parsed JSON payload."""

        if not model:
            raise TokenFactoryError("model is required")
        if not messages:
            raise TokenFactoryError("messages must be a non-empty sequence")

        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if extra:
            payload.update(extra)

        data = self._post_json(self._config.chat_completions_url, payload)
        if not isinstance(data, dict):
            raise TokenFactoryError("Token Factory returned a non-object response")
        return data

    def chat_completion_text(self, **kwargs: Any) -> str:
        """Return the assistant message text from a chat completion."""

        data = self.chat_completion(**kwargs)
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise TokenFactoryError(
                "Token Factory response missing choices[0].message.content"
            ) from exc

    def list_models(self) -> list[str]:
        """Return the list of model IDs available to this API key."""

        data = self._get_json(self._config.models_url)
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise TokenFactoryError("Token Factory models response missing data list")
        return [str(item["id"]) for item in items if isinstance(item, dict) and item.get("id")]

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    def _post_json(self, url: str, payload: dict[str, Any]) -> Any:
        return self._request("POST", url, json_body=payload)

    def _get_json(self, url: str) -> Any:
        return self._request("GET", url)

    def _request(self, method: str, url: str, *, json_body: dict[str, Any] | None = None) -> Any:
        owns_client = self._http_client is None
        client = self._http_client or httpx.Client(timeout=self._config.timeout_s)
        try:
            response = client.request(method, url, headers=self._headers(), json=json_body)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise TokenFactoryError(
                f"Token Factory request failed ({exc.response.status_code}): "
                f"{_truncate(exc.response.text)}"
            ) from exc
        except httpx.HTTPError as exc:
            raise TokenFactoryError(f"Token Factory request failed: {exc}") from exc
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise TokenFactoryError("Token Factory returned non-JSON response") from exc
        finally:
            if owns_client:
                client.close()


def _first_env(env: dict[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        value = env.get(key)
        if value:
            return value.strip()
    return ""


def _join_path(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/")
    suffix = suffix.strip("/")
    if base.endswith(f"/{suffix}"):
        return base
    if base.endswith("/v1"):
        return f"{base}/{suffix}"
    return f"{base}/v1/{suffix}"


def _truncate(text: str, *, limit: int = 500) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "..."
