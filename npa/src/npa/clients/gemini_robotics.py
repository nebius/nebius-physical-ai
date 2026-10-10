"""Shared Gemini Robotics hosted-client configuration and response handling."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import httpx

# PROVISIONAL, UNVALIDATED GUESSES. The API base URL and model id below have
# never been validated against a live Gemini Robotics endpoint. They exist only
# to document the guess; every entry point requires explicit values
# (override-required) and fails closed otherwise. Do not treat these as
# operational.
PROVISIONAL_API_BASE_URL = "https://generativelanguage.googleapis.com"
PROVISIONAL_MODEL_ID = "gemini-robotics-er-2-preview"
API_KEY_ENV = "GOOGLE_API_KEY"
BASE_URL_ENV = "GEMINI_ROBOTICS_BASE_URL"

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_RETRY_ATTEMPTS = 4
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

ER_SYSTEM_INSTRUCTION = (
    "You are Gemini Robotics-ER, an embodied-reasoning planner for physical robots. "
    "Given a task and optional scene observations, produce a safe, step-by-step "
    "whole-body manipulation plan. Prefer conservative, reversible motions. When any "
    "step carries physical risk (contact with people, fragile objects, high forces), "
    "identify the risk and recommend human review. Do not claim that a safety review "
    "was performed, and do not present the response as authorization to execute."
)

PLAN_RECEIPT_SCHEMA = "npa.gemini_robotics.plan.v1"
EVAL_RECEIPT_SCHEMA = "npa.gemini_robotics.eval.v1"


class GeminiRoboticsError(RuntimeError):
    """Report invalid Gemini Robotics configuration or a failed API request."""


def _resolve_env(environ: Mapping[str, str] | None) -> dict[str, str]:
    """Build config environment with a non-exporting saved-key fallback.

    Args:
        environ: Explicit environment mapping, if supplied by a caller.

    Returns:
        A copied environment with a saved Gemini key when available.

    Raises:
        GeminiRoboticsError: If the saved credential store cannot be read safely.
    """

    if environ is not None:
        return dict(environ)
    from npa.clients.credentials import CredentialStoreError, load_credentials

    env = dict(os.environ)
    if env.get(API_KEY_ENV):
        return env
    try:
        file_key = load_credentials(environ=env).tokens.get(API_KEY_ENV, "")
    except CredentialStoreError as exc:
        raise GeminiRoboticsError(
            "Could not read saved Gemini credentials. Fix ~/.npa/credentials.yaml "
            f"or set {API_KEY_ENV} in the environment."
        ) from exc
    if file_key:
        env[API_KEY_ENV] = file_key
    return env


def _server_message(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return ""


@dataclass(frozen=True)
class GeminiRoboticsConfig:
    """Represent resolved Gemini API connection settings.

    Args:
        base_url: Explicit hosted API base URL.
        api_key: Operator-supplied Gemini API key.
        timeout_s: Per-request timeout in seconds.

    Returns:
        An immutable API connection configuration.

    Raises:
        None.
    """

    base_url: str
    api_key: str
    timeout_s: float = DEFAULT_TIMEOUT_S

    def generate_content_url(self, model: str) -> str:
        """Return the explicit model's generate-content endpoint.

        Args:
            model: Operator-selected Gemini model identifier.

        Returns:
            The generate-content URL for ``model``.

        Raises:
            None.
        """

        return f"{self.base_url}/v1beta/models/{model}:generateContent"


def resolve_config(
    api_key: str | None = None,
    base_url: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> GeminiRoboticsConfig:
    """Resolve explicit Gemini settings and reject missing key or endpoint first.

    The base URL is never guessed: pass ``base_url`` / ``--api-base-url`` or set
    the ``GEMINI_ROBOTICS_BASE_URL`` environment variable. When no explicit
    key is present, ``tokens.GOOGLE_API_KEY`` in the saved credential file is a
    non-exporting fallback. PROVISIONAL_API_BASE_URL is documentation only and
    is never used implicitly.

    Args:
        api_key: Explicit API key, if not read from the environment.
        base_url: Explicit API base URL, if not read from the environment.
        environ: Optional environment mapping for deterministic callers.

    Returns:
        The validated connection configuration.

    Raises:
        GeminiRoboticsError: If the API key or base URL is absent.
    """

    env = _resolve_env(environ)
    key = (api_key if api_key is not None else env.get(API_KEY_ENV, "")).strip()
    if not key:
        raise GeminiRoboticsError(
            f"Missing Gemini API key: set the {API_KEY_ENV} environment variable "
            "or add it under tokens: in ~/.npa/credentials.yaml "
            "before calling Gemini Robotics endpoints."
        )
    resolved_base = (
        (base_url if base_url is not None else env.get(BASE_URL_ENV, ""))
        .strip()
        .rstrip("/")
    )
    if not resolved_base:
        raise GeminiRoboticsError(
            "Missing Gemini API base URL: pass --api-base-url or set the "
            f"{BASE_URL_ENV} environment variable. The provisional default is an "
            "unvalidated guess and is never used implicitly."
        )
    return GeminiRoboticsConfig(base_url=resolved_base, api_key=key)


@dataclass
class PlanResult:
    """Represent a parsed embodied-reasoning planning response.

    Args:
        text: Provider response text.
        model_function_calls: Provider-returned function-call records.
        model: Exact model identifier used for the request.
        finish_reason: Provider completion reason.

    Returns:
        A normalized plan result.

    Raises:
        None.
    """

    text: str
    model_function_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    finish_reason: str = ""


@dataclass
class EvalResult:
    """Represent a parsed rubric-evaluation response.

    Args:
        scores: Provider rubric scores by criterion.
        summary: Provider-written evaluation summary.
        raw_text: Original provider text before JSON parsing.
        model: Exact model identifier used for the request.

    Returns:
        A normalized evaluation result.

    Raises:
        None.
    """

    scores: dict[str, Any]
    summary: str
    raw_text: str
    model: str = ""


class GeminiRoboticsClient:
    """Issue testable Gemini Robotics planning and evaluation requests.

    Args:
        config: Validated endpoint, credential, and timeout settings.
        http_client: Optional HTTP client for tests or custom transports.
        sleeper: Backoff callback used only for retryable responses.

    Returns:
        A client with no request made during construction.

    Raises:
        None.
    """

    def __init__(
        self,
        config: GeminiRoboticsConfig,
        http_client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._http = http_client or httpx.Client()
        self._sleeper = sleeper

    def _headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self._config.api_key,
            "Content-Type": "application/json",
        }

    def _checked(self, response: httpx.Response, what: str) -> dict[str, Any]:
        status = response.status_code
        if status == 200:
            try:
                payload = response.json()
            except ValueError as exc:
                raise GeminiRoboticsError(
                    f"Gemini API returned non-JSON for {what} (HTTP 200)."
                ) from exc
            return payload if isinstance(payload, dict) else {"value": payload}
        detail = _server_message(self._safe_json(response))
        if status in (401, 403):
            raise GeminiRoboticsError(
                f"Gemini API authentication failed for {what} (HTTP {status}). "
                f"Set a valid {API_KEY_ENV} environment variable."
                + (f" Server detail: {detail}" if detail else "")
            )
        if status == 404:
            raise GeminiRoboticsError(
                f"Gemini API endpoint not found for {what} (HTTP 404). "
                "Check the model id and base URL."
                + (f" Server detail: {detail}" if detail else "")
            )
        raise GeminiRoboticsError(
            f"Gemini API request failed for {what} (HTTP {status})."
            + (f" Server detail: {detail}" if detail else "")
        )

    @staticmethod
    def _safe_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return None

    def _request(
        self, method: str, url: str, payload: dict[str, Any] | None, what: str
    ) -> dict[str, Any]:
        attempts = 0
        while True:
            attempts += 1
            try:
                response = self._http.request(
                    method,
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self._config.timeout_s,
                )
            except httpx.HTTPError as exc:
                raise GeminiRoboticsError(
                    f"Gemini API transport error for {what}: {exc}"
                ) from exc
            if (
                response.status_code in RETRYABLE_STATUS_CODES
                and attempts < DEFAULT_RETRY_ATTEMPTS
            ):
                self._sleeper(2.0 ** (attempts - 1))
                continue
            return self._checked(response, what)

    def plan(
        self,
        *,
        task: str,
        images: Sequence[Path] = (),
        model: str,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> PlanResult:
        """Request an advisory embodied-reasoning plan.

        Args:
            task: Operator-supplied robot task.
            images: Local, already-authorized scene observations.
            model: Explicit Gemini model identifier.
            temperature: Sampling temperature.
            max_output_tokens: Provider output-token ceiling.

        Returns:
            The parsed advisory planning result.

        Raises:
            GeminiRoboticsError: If the transport or provider response is invalid.
        """

        parts: list[dict[str, Any]] = [{"text": task}]
        for image in images:
            parts.append(self._image_part(Path(image)))
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": ER_SYSTEM_INSTRUCTION}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        body = self._request(
            "POST", self._config.generate_content_url(model), payload, "ER planning"
        )
        return self._parse_plan(body, model=model)

    @staticmethod
    def _image_part(image: Path) -> dict[str, Any]:
        data = image.read_bytes()
        mime, _ = mimetypes.guess_type(str(image))
        return {
            "inlineData": {
                "mimeType": mime or "image/jpeg",
                "data": base64.b64encode(data).decode("ascii"),
            }
        }

    @staticmethod
    def _parse_plan(body: Mapping[str, Any], model: str) -> PlanResult:
        candidates = body.get("candidates") or []
        if not candidates:
            raise GeminiRoboticsError(
                "Gemini API returned no candidates for ER planning."
            )
        content = (candidates[0] or {}).get("content", {})
        texts: list[str] = []
        model_function_calls: list[dict[str, Any]] = []
        for part in content.get("parts", []) or []:
            if not isinstance(part, dict):
                continue
            if part.get("text"):
                texts.append(str(part["text"]))
            call = part.get("functionCall")
            if isinstance(call, dict):
                model_function_calls.append(
                    {"name": str(call.get("name", "")), "args": call.get("args", {})}
                )
        return PlanResult(
            text="\n".join(texts).strip(),
            model_function_calls=model_function_calls,
            model=model,
            finish_reason=str((candidates[0] or {}).get("finishReason", "")),
        )

    def eval_plan(
        self,
        *,
        plan_text: str,
        rubric: str,
        model: str,
        temperature: float = 0.0,
    ) -> EvalResult:
        """Request rubric-scored evaluation of a durable plan.

        Args:
            plan_text: The plan text selected for evaluation.
            rubric: Explicit evaluation rubric.
            model: Explicit Gemini model identifier.
            temperature: Sampling temperature for the structured response.

        Returns:
            The parsed rubric-evaluation result.

        Raises:
            GeminiRoboticsError: If the transport or provider response is invalid.
        """

        prompt = (
            "Evaluate the following robot manipulation plan against the rubric.\n\n"
            f"RUBRIC:\n{rubric}\n\nPLAN:\n{plan_text}\n\n"
            'Respond with a single JSON object: {"scores": {<criterion>: <0-10>}, '
            '"summary": "<one-paragraph justification>"}. No other text.'
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        body = self._request(
            "POST", self._config.generate_content_url(model), payload, "plan evaluation"
        )
        candidates = body.get("candidates") or []
        if not candidates:
            raise GeminiRoboticsError(
                "Gemini API returned no candidates for plan evaluation."
            )
        parts = ((candidates[0] or {}).get("content", {}) or {}).get("parts", []) or []
        raw_text = "\n".join(
            str(part.get("text", ""))
            for part in parts
            if isinstance(part, dict) and part.get("text")
        ).strip()
        parsed = self._parse_eval_json(raw_text)
        return EvalResult(
            scores=parsed.get("scores", {}),
            summary=str(parsed.get("summary", "")),
            raw_text=raw_text,
            model=model,
        )

    @staticmethod
    def _parse_eval_json(raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
        except ValueError as exc:
            raise GeminiRoboticsError(
                "Could not parse the evaluation response as JSON. "
                f"Raw response starts with: {raw_text[:200]!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise GeminiRoboticsError("Evaluation response was not a JSON object.")
        scores = parsed.get("scores")
        if not isinstance(scores, dict):
            raise GeminiRoboticsError("Evaluation response has no 'scores' object.")
        return parsed
