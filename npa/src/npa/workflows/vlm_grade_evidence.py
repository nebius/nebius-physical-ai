"""Check retained VLM inference evidence before a data-factory promotion."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from npa.workbench import vlm_eval


def vlm_grade_block_reason(report: dict[str, Any]) -> str:
    """Return a bounded reason when a VLM grade lacks consistent inference evidence.

    Args:
        report: A serialized VLM evaluation result.
    Returns:
        An empty string for consistent evidence, otherwise a stable loop-back reason.
        Hashes establish internal integrity, not provider authentication.
    Raises:
        None.
    """

    if (
        report.get("backend") not in ("api", "self-hosted")
        or report.get("dry_run", False) is not False
    ):
        return "vlm_non_inference_backend"
    evidence = report.get("evidence")
    if not isinstance(evidence, dict):
        return "vlm_provider_evidence_missing"
    try:
        _validate_evidence(report, evidence)
    except (KeyError, TypeError, ValueError, AttributeError, RuntimeError):
        return "vlm_provider_evidence_invalid"
    return ""


def _validate_evidence(report: dict[str, Any], evidence: dict[str, Any]) -> None:
    if evidence["schema_version"] not in {
        "npa_vlm_eval_evidence_v1",
        vlm_eval.EVIDENCE_SCHEMA_VERSION,
    }:
        raise ValueError("unknown VLM evidence schema")
    request = evidence["request"]
    provider = evidence["provider"]
    if not isinstance(request, dict) or not isinstance(provider, dict):
        raise ValueError("missing VLM evidence objects")
    _validate_request(report, request, evidence["schema_version"])
    parsed = _retained_verdict(report, provider)
    _validate_result(report, parsed)


def _validate_request(report: dict[str, Any], request: dict, schema: str) -> None:
    manifest = request["request_manifest"]
    if not isinstance(manifest, dict):
        raise ValueError("invalid VLM request manifest")
    _require_hash(request["request_manifest_sha256"], vlm_eval._sha256_json(manifest))
    role = "hosted-api" if report["backend"] == "api" else "self-hosted"
    for key, expected in (
        ("schema_version", schema),
        ("endpoint_role", role),
        ("requested_model", report["model"]),
        ("frames", request["frames"]),
        ("prompt_sha256", request["prompt_sha256"]),
        ("rubric_sha256", request["rubric_sha256"]),
    ):
        if manifest.get(key) != expected:
            raise ValueError("VLM request binding differs")
    if (
        request["endpoint_role"] != role
        or not isinstance(request["requested_at"], str)
        or not request["requested_at"]
    ):
        raise ValueError("invalid VLM request context")
    if (
        not isinstance(report["model"], str)
        or not report["model"]
        or not isinstance(manifest.get("generation_parameters"), dict)
    ):
        raise ValueError("invalid VLM generation context")
    _validate_frames(report, request["frames"])
    prompt = vlm_eval._build_prompt(
        task=report["task"],
        rubric=report["rubric"],
        frame_selection=report["frame_selection"],
        frame_count=report["frame_count"],
    )
    _require_hash(request["prompt_sha256"], vlm_eval._sha256_text(prompt))
    _require_hash(request["rubric_sha256"], vlm_eval._sha256_text(report["rubric"]))


def _validate_frames(report: dict[str, Any], frames: Any) -> None:
    count = report["frame_count"]
    if type(count) is not int or count <= 0 or not isinstance(frames, list):
        raise ValueError("VLM inference requires submitted frames")
    if len(frames) != count:
        raise ValueError("VLM frame count differs")
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("invalid VLM frame evidence")
        if not isinstance(frame.get("label"), str) or not frame["label"]:
            raise ValueError("missing VLM frame label")
        if not isinstance(frame.get("media_type"), str) or not frame[
            "media_type"
        ].startswith("image/"):
            raise ValueError("invalid VLM frame media type")
        for field in ("byte_count", "width", "height"):
            if type(frame.get(field)) is not int or frame[field] <= 0:
                raise ValueError("invalid VLM frame dimensions")
        _require_hash(frame["sha256"], frame["sha256"])


def _require_hash(actual: Any, expected: str) -> None:
    if not isinstance(actual, str) or re.fullmatch(r"[0-9a-f]{64}", actual) is None:
        raise ValueError("invalid VLM evidence digest")
    if actual != expected:
        raise ValueError("VLM evidence digest differs")


def _retained_verdict(report: dict[str, Any], provider: dict):
    from npa.clients.token_factory import token_factory_chat_profile

    raw = provider["raw_response"]
    if not isinstance(raw, str) or not raw:
        raise ValueError("missing VLM provider response")
    _require_hash(provider["raw_response_sha256"], vlm_eval._sha256_text(raw))
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("invalid VLM provider response")
    _validate_transport(provider)
    choice, content = vlm_eval._response_choice_and_content(data)
    # Reuse the producer's backend contract locally; this performs no transport
    # and does not authenticate the retained provider response.
    parsed = vlm_eval._parse_backend_verdict(
        backend=report["backend"],
        requested_model=report["model"],
        profile=token_factory_chat_profile(report["model"]),
        data=data,
        message=content,
    )
    _validate_provider_metadata(provider, data, choice, parsed.parser_version)
    return parsed


def _validate_transport(provider: dict[str, Any]) -> None:
    status = provider["status_code"]
    if status is not None and (type(status) is not int or not 200 <= status < 300):
        raise ValueError("VLM provider transport did not succeed")
    latency = provider["latency_s"]
    if type(latency) not in {int, float} or not math.isfinite(latency) or latency < 0:
        raise ValueError("invalid VLM provider latency")


def _validate_provider_metadata(
    provider: dict, data: dict, choice: dict, parser: str
) -> None:
    response = vlm_eval._VlmBackendResponse(
        data=data,
        raw_body=provider["raw_response"],
        status_code=provider["status_code"],
        request_id_header=provider["provider_request_id"],
        latency_s=provider["latency_s"],
    )
    request_id, model, finish, usage = vlm_eval._provider_metadata(response, choice)
    expected = {
        "provider_request_id": request_id,
        "returned_model": model,
        "finish_reason": finish,
        "latency_s": round(response.latency_s, 6),
        "status_code": response.status_code,
        "usage": usage,
        "raw_response": response.raw_body,
        "raw_response_sha256": vlm_eval._sha256_text(response.raw_body),
        "parser_version": parser,
    }
    if expected != provider:
        raise ValueError("VLM provider metadata differs from retained response")


def _validate_result(
    report: dict[str, Any], parsed: vlm_eval.VlmStructuredResponse
) -> None:
    score = report["score"]
    threshold = report["success_threshold"]
    if any(
        type(value) not in {float, int} or not 0 <= value <= 1
        for value in (score, threshold)
    ):
        raise ValueError("invalid VLM score or threshold")
    expected_score = round(parsed.score, 4)
    passed = expected_score >= threshold
    expected = {
        "score": expected_score,
        "passed": passed,
        "status": "passed" if passed else "needs_iteration",
        "served_model": parsed.served_model,
        "rationale": parsed.rationale,
        "provider_success": parsed.provider_success,
        "provider_success_matches_score_gate": (
            parsed.provider_success == passed
            if parsed.provider_success is not None
            else None
        ),
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("VLM result differs from retained verdict")
    for field in ("passed", "provider_success", "provider_success_matches_score_gate"):
        if type(report.get(field)) is not type(expected[field]):
            raise ValueError("VLM boolean result fields have invalid types")
