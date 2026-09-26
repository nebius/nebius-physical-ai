"""npa gemini-robotics — API-backed Gemini Robotics 2 / Robotics-ER toolRef.

Gemini Robotics 2 / Robotics-ER 2 (announced July 30, 2026) is closed-weight,
so no local containerization is possible.  This module follows the same
API-gateway posture as ``token_factory`` and ``vlm_eval``: it calls the hosted
Gemini API for

- ``plan`` — ER embodied-reasoning planning, with safety tool calls;
- ``eval`` — rubric-scored evaluation of a plan.

Credentials come from the ``GOOGLE_API_KEY`` environment variable (overrideable
with ``GEMINI_ROBOTICS_BASE_URL`` for the endpoint).  Auth failures raise a
clear, actionable error instead of leaking the key.  No live calls are made in
tests; unit tests drive the client through ``httpx.MockTransport``.
"""

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
import typer
from rich.console import Console

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
    "request a safety review by calling the check_safety tool with the proposed "
    "action before finalizing the plan."
)

SAFETY_TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "check_safety",
        "description": (
            "Request a safety review of a proposed physical action before execution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Natural-language description of the proposed action.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this step was flagged for review.",
                },
            },
            "required": ["action"],
        },
    },
]

PLAN_RECEIPT_SCHEMA = "npa.gemini_robotics.plan.v1"
EVAL_RECEIPT_SCHEMA = "npa.gemini_robotics.eval.v1"

app = typer.Typer(
    name="gemini-robotics",
    help=(
        "Gemini Robotics API-backed toolRef (plan, eval). "
        "Provisional adapter: API base URL and model id must be supplied "
        "explicitly; no live access has been validated."
    ),
    no_args_is_help=True,
)
console = Console(stderr=True)


class GeminiRoboticsError(RuntimeError):
    """Raised when a Gemini Robotics API request is misconfigured or fails."""


def _server_message(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return ""


@dataclass(frozen=True)
class GeminiRoboticsConfig:
    """Resolved connection settings for the Gemini API."""

    base_url: str
    api_key: str
    timeout_s: float = DEFAULT_TIMEOUT_S

    def generate_content_url(self, model: str) -> str:
        return f"{self.base_url}/v1beta/models/{model}:generateContent"


def resolve_config(
    api_key: str | None = None,
    base_url: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> GeminiRoboticsConfig:
    """Resolve the API key and base URL, failing closed when either is missing.

    The base URL is never guessed: pass ``base_url`` / ``--api-base-url`` or set
    the ``GEMINI_ROBOTICS_BASE_URL`` environment variable. PROVISIONAL_API_BASE_URL
    is documentation only and is never used implicitly.
    """
    env = environ if environ is not None else os.environ
    key = (api_key if api_key is not None else env.get(API_KEY_ENV, "")).strip()
    if not key:
        raise GeminiRoboticsError(
            f"Missing Gemini API key: set the {API_KEY_ENV} environment variable "
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
    """Parsed ER planning response."""

    text: str
    safety_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    finish_reason: str = ""


@dataclass
class EvalResult:
    """Parsed rubric evaluation."""

    scores: dict[str, Any]
    summary: str
    raw_text: str
    model: str = ""


class GeminiRoboticsClient:
    """Thin, testable client over the Gemini API for the Robotics toolRef."""

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
        safety_tools: Sequence[Mapping[str, Any]] = SAFETY_TOOL_DECLARATIONS,
    ) -> PlanResult:
        """Run ER embodied-reasoning planning for ``task``."""
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
        if safety_tools:
            payload["tools"] = [{"functionDeclarations": list(safety_tools)}]
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
        safety_calls: list[dict[str, Any]] = []
        for part in content.get("parts", []) or []:
            if not isinstance(part, dict):
                continue
            if part.get("text"):
                texts.append(str(part["text"]))
            call = part.get("functionCall")
            if isinstance(call, dict):
                safety_calls.append(
                    {"name": str(call.get("name", "")), "args": call.get("args", {})}
                )
        return PlanResult(
            text="\n".join(texts).strip(),
            safety_calls=safety_calls,
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
        """Score a plan against a rubric; expects a JSON object from the model."""
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


def _fail(message: str) -> None:
    console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(1)


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@app.command("plan")
def plan_cmd(
    task: str = typer.Argument(..., help="Natural-language task for the ER planner."),
    image: list[str] = typer.Option(
        [], "--image", help="Scene observation image (repeatable)."
    ),
    model: str = typer.Option(
        ..., "--model", help="Gemini API model id (required; no default)."
    ),
    api_base_url: str = typer.Option(
        "",
        "--api-base-url",
        help=f"Gemini API base URL (or set {BASE_URL_ENV}).",
    ),
    temperature: float = typer.Option(
        0.2, "--temperature", help="Sampling temperature."
    ),
    max_output_tokens: int = typer.Option(
        2048, "--max-output-tokens", help="Max output tokens."
    ),
    output_path: str = typer.Option(
        "", "--output-path", help="Write the plan receipt JSON here."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Do not write the receipt artifact."
    ),
) -> dict[str, Any]:
    """Run ER embodied-reasoning planning via the Gemini API."""
    try:
        config = resolve_config(base_url=api_base_url or None)
        client = GeminiRoboticsClient(config)
        result = client.plan(
            task=task,
            images=[Path(p) for p in image],
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    except GeminiRoboticsError as exc:
        _fail(str(exc))
        raise  # pragma: no cover - _fail raises
    receipt: dict[str, Any] = {
        "schema": PLAN_RECEIPT_SCHEMA,
        "task": task,
        "model": result.model,
        "plan_text": result.text,
        "safety_calls": result.safety_calls,
        "finish_reason": result.finish_reason,
    }
    if output_path and not dry_run:
        _write_receipt(Path(output_path), receipt)
        receipt["written_path"] = output_path
    console.print(json.dumps(receipt, indent=2))
    return receipt


@app.command("eval")
def eval_cmd(
    plan_path: str = typer.Argument(..., help="Path to a plan receipt or plan text."),
    rubric_path: str = typer.Option(
        ..., "--rubric-path", help="Path to the evaluation rubric text."
    ),
    model: str = typer.Option(
        ..., "--model", help="Gemini API model id (required; no default)."
    ),
    api_base_url: str = typer.Option(
        "",
        "--api-base-url",
        help=f"Gemini API base URL (or set {BASE_URL_ENV}).",
    ),
    output_path: str = typer.Option(
        "", "--output-path", help="Write the eval receipt JSON here."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Do not write the receipt artifact."
    ),
) -> dict[str, Any]:
    """Evaluate a plan against a rubric via the Gemini API."""
    try:
        raw_plan = Path(plan_path).read_text(encoding="utf-8")
        try:
            plan_text = str(json.loads(raw_plan).get("plan_text", raw_plan))
        except ValueError:
            plan_text = raw_plan
        rubric = Path(rubric_path).read_text(encoding="utf-8")
        config = resolve_config(base_url=api_base_url or None)
        client = GeminiRoboticsClient(config)
        result = client.eval_plan(plan_text=plan_text, rubric=rubric, model=model)
    except (GeminiRoboticsError, OSError) as exc:
        _fail(str(exc))
        raise  # pragma: no cover - _fail raises
    receipt: dict[str, Any] = {
        "schema": EVAL_RECEIPT_SCHEMA,
        "plan_path": plan_path,
        "model": result.model,
        "scores": result.scores,
        "summary": result.summary,
    }
    if output_path and not dry_run:
        _write_receipt(Path(output_path), receipt)
        receipt["written_path"] = output_path
    console.print(json.dumps(receipt, indent=2))
    return receipt
