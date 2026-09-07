"""Live provider contracts with allowlisted, secret-free observations.

These probes observe the provider, including improvements to structured output.
They never repair a malformed response or rewrite the configured baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from npa.clients.token_factory import (
    DEFAULT_REASONER_MODEL,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    TokenFactoryClient,
    split_reasoning,
)

LIGHTNING = "nvidia/Nemotron-3_5-Lightning"
MINIMAX = "MiniMaxAI/MiniMax-M3"
JSON_PROMPT = (
    'Return only a JSON object with exactly these fields: "score": 0.75, '
    '"success": true, "summary": "red square inside green outline". '
    "Do not use a markdown fence."
)
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "success": {"type": "boolean"},
        "summary": {"type": "string"},
    },
    "required": ["score", "success", "summary"],
    "additionalProperties": False,
}


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def model_reference(model: str) -> str:
    """Only canonical public names are publishable; hash custom deployment IDs."""
    if model in {LIGHTNING, MINIMAX}:
        return model
    return "configured-model-sha256:" + _hash(model)


def structured_behavior(text: str) -> str:
    """Classify the original bytes, without fence removal or score repair."""
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return "malformed_json"
    if not isinstance(value, dict) or set(value) != {"score", "success", "summary"}:
        return "schema_invalid"
    score = value["score"]
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or score != 0.75
        or value["success"] is not True
        or value["summary"] != "red square inside green outline"
    ):
        return "schema_invalid"
    return "healthy"


def response_evidence(response: dict[str, Any], requested_model: str) -> dict[str, Any]:
    """Keep model identity, accounting and hashes; omit arbitrary provider text."""
    choices = response.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    visible, reasoning = split_reasoning(choice.get("message") or {})
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    counts = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(name)
        counts[name] = value if isinstance(value, int) and not isinstance(value, bool) else None
    raw_reasoning = details.get("reasoning_tokens")
    reasoning_tokens = raw_reasoning if isinstance(raw_reasoning, int) else None
    reasons = []
    if response.get("model") != requested_model:
        reasons.append("served_model_mismatch")
    if not visible or choice.get("finish_reason") != "stop":
        reasons.append("incomplete_visible_output")
    if any(value is None or value <= 0 for value in counts.values()):
        reasons.append("missing_positive_usage")
    if not response.get("id"):
        reasons.append("missing_request_identity")
    return {
        "requested_model": model_reference(requested_model),
        # Only publish the requested, configured model ID; unexpected provider
        # identities remain hashed so dedicated deployment names cannot leak.
        "served_model_matches": response.get("model") == requested_model,
        "served_model_sha256": _hash(str(response.get("model") or "")),
        "request_id_sha256": _hash(str(response.get("id") or "")),
        "visible_sha256": _hash(visible),
        "visible_characters": len(visible),
        "reasoning_characters": len(reasoning or ""),
        "reasoning_tokens": reasoning_tokens,
        "finish_reason": choice.get("finish_reason") if choice.get("finish_reason") in {
            "stop", "length", "tool_calls", "content_filter"
        } else "unknown",
        "usage": counts,
        "errors": reasons,
    }


def run_contract(
    client: TokenFactoryClient,
    *,
    additional_models: tuple[str, ...] = (),
    expected_json_behavior: str = "malformed_json",
) -> dict[str, Any]:
    """Exercise all migration defaults and any explicitly configured models.

    A healthy constrained response is recorded as a baseline change when the
    configured expectation is malformed_json. Operators review the workaround
    and update the explicit baseline; no automatic acceptance masks drift.
    """
    if expected_json_behavior not in {"malformed_json", "schema_invalid", "healthy"}:
        raise ValueError("Invalid structured-output baseline")
    required = tuple(dict.fromkeys((
        DEFAULT_TEXT_MODEL, DEFAULT_REASONER_MODEL, DEFAULT_VISION_MODEL,
        *additional_models,
    )))
    report: dict[str, Any] = {
        "schema": "npa.token_factory.contract.v1",
        "provider": "nebius_token_factory",
        "required_models": [model_reference(model) for model in required],
        "expected_json_behavior": expected_json_behavior,
        "checks": [],
    }
    checks = report["checks"]
    try:
        available = client.list_models()
        missing = [model for model in required if model not in available]
        checks.append({"check": "required_model_catalog", "passed": not missing,
                       "required_count": len(required),
                       "missing_models": [model_reference(model) for model in missing]})
    except Exception as exc:
        checks.append({"check": "required_model_catalog", "passed": False,
                       "error_type": type(exc).__name__})

    def probe(name: str, model: str, *, thinking: bool | None = None,
              response_format: dict | None = None, json_expected: str | None = None) -> None:
        check: dict[str, Any] = {
            "check": name, "requested_model": model_reference(model), "passed": False,
        }
        checks.append(check)
        extra = None
        if thinking is not None:
            control = ({"enable_thinking": thinking} if model == LIGHTNING else
                       {"thinking_mode": "enabled" if thinking else "disabled"})
            extra = {"chat_template_kwargs": control}
            check["control"] = control
        prompt = JSON_PROMPT if json_expected is not None else (
            "A box has two red balls and three blue balls. Without replacement, "
            "what is the minimum number of draws that guarantees a red ball? "
            "Answer with the digit and a short explanation."
        )
        try:
            response = client.chat_completion(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0, extra=extra, response_format=response_format,
            )
            evidence = response_evidence(response, model)
            check.update(evidence)
            visible, _ = split_reasoning(response["choices"][0]["message"])
            if json_expected is None and not ("4" in visible or "four" in visible.lower()):
                check["errors"].append("incorrect_synthetic_answer")
            if thinking is False and (evidence["reasoning_characters"] or
                                      (evidence["reasoning_tokens"] or 0) > 0):
                check["errors"].append("thinking_not_disabled")
            if thinking is True and not (evidence["reasoning_characters"] or
                                         (evidence["reasoning_tokens"] or 0) > 0):
                check["errors"].append("thinking_not_enabled")
            if json_expected is not None:
                observed = structured_behavior(visible)
                check.update(observed_json_behavior=observed, expected_json_behavior=json_expected)
                if observed != json_expected:
                    check["errors"].append("structured_output_baseline_changed")
            check["passed"] = not check["errors"]
        except Exception as exc:
            # HTTP exception messages can carry URLs, provider bodies or headers.
            check["error_type"] = type(exc).__name__

    for model in required:
        probe("required_model_inference", model)
    for model in (LIGHTNING, MINIMAX):
        for enabled in (False, True):
            probe("thinking_enabled" if enabled else "thinking_disabled", model, thinking=enabled)
    for kind in ("json_object", "json_schema"):
        response_format = {"type": kind}
        if kind == "json_schema":
            response_format["json_schema"] = {
                "name": "evaluation", "strict": True, "schema": JSON_SCHEMA,
            }
        probe(kind, MINIMAX, response_format=response_format, json_expected=expected_json_behavior)
    probe("prompted_json_workaround", MINIMAX, json_expected="healthy")
    report["passed"] = all(check["passed"] for check in checks)
    return report
