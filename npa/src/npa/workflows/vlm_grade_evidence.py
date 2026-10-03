"""Check retained VLM inference evidence before a data-factory promotion."""

from __future__ import annotations

from dataclasses import replace
import json
import re
from typing import Any

from npa.literal_values import require_boolean, require_integer, require_number
from npa.workbench import vlm_eval


class _InvalidEvidence(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def vlm_grade_block_reason(report: dict[str, Any]) -> str:
    """Return a bounded reason when a VLM grade lacks consistent inference evidence.

    Args:
        report: A serialized VLM evaluation result.
    Returns:
        An empty string for promotion-eligible evidence, otherwise a stable reason.
        Hashes establish internal integrity, not provider authentication.
    Raises:
        None.
    """

    return vlm_grade_block_details(report).get("reason", "")


def vlm_grade_block_details(report: dict[str, Any]) -> dict[str, str]:
    """Return compatible gate reasons with a bounded evidence diagnostic.

    Args:
        report: A serialized VLM evaluation result.
    Returns:
        An empty mapping for promotion-eligible evidence, otherwise the stable ``reason``
        and, for invalid evidence, an additive ``evidence_reason`` classification.
    Raises:
        None.
    """

    if (
        report.get("backend") not in ("api", "self-hosted")
        or report.get("dry_run", False) is not False
    ):
        return {"reason": "vlm_non_inference_backend"}
    evidence = report.get("evidence")
    if not isinstance(evidence, dict):
        return {"reason": "vlm_provider_evidence_missing"}
    try:
        _validate_evidence(report, evidence)
    except _InvalidEvidence as exc:
        return {
            "reason": "vlm_provider_evidence_invalid",
            "evidence_reason": exc.reason,
        }
    except (KeyError, TypeError, ValueError, AttributeError, RuntimeError):
        return {
            "reason": "vlm_provider_evidence_invalid",
            "evidence_reason": "schema_invalid",
        }
    return {}


def _validate_evidence(report: dict[str, Any], evidence: dict[str, Any]) -> None:
    if evidence["schema_version"] not in {
        "npa_vlm_eval_evidence_v1",
        "npa_vlm_eval_evidence_v2",
    }:
        raise _InvalidEvidence("schema_invalid")
    request = evidence["request"]
    provider = evidence["provider"]
    if not isinstance(request, dict) or not isinstance(provider, dict):
        raise _InvalidEvidence("schema_invalid")
    _validate_request(report, request, evidence["schema_version"])
    parsed = _retained_verdict(report, provider)
    _validate_result(report, parsed)


def _validate_request(report: dict[str, Any], request: dict, schema: str) -> None:
    manifest = request["request_manifest"]
    if not isinstance(manifest, dict):
        raise _InvalidEvidence("request_invalid")
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
            raise _InvalidEvidence("request_binding_mismatch")
    if (
        request["endpoint_role"] != role
        or not isinstance(request["requested_at"], str)
        or not request["requested_at"]
    ):
        raise _InvalidEvidence("request_invalid")
    if (
        not isinstance(report["model"], str)
        or not report["model"]
        or not isinstance(manifest.get("generation_parameters"), dict)
    ):
        raise _InvalidEvidence("request_invalid")
    _validate_frames(report, request["frames"])
    if schema != "npa_vlm_eval_evidence_v1":
        _validate_sampling(report, manifest)
    _validate_prompt(report, request)


def _validate_prompt(report: dict[str, Any], request: dict) -> None:
    # Historical v1 writers omitted the rubric. Only the actual default can
    # reconstruct those hashes; custom-rubric results must retain their text.
    rubric = report.get("rubric", vlm_eval.DEFAULT_RUBRIC)
    prompt = vlm_eval._build_prompt(
        task=report["task"],
        rubric=rubric,
        frame_selection=report["frame_selection"],
        frame_count=report["frame_count"],
    )
    _require_hash(request["prompt_sha256"], vlm_eval._sha256_text(prompt))
    _require_hash(request["rubric_sha256"], vlm_eval._sha256_text(rubric))


def _validate_frames(report: dict[str, Any], frames: Any) -> None:
    count = report["frame_count"]
    reason = "frame_metadata_invalid"
    _require_literal(require_integer, count, "frame_count", reason, minimum=1)
    if not isinstance(frames, list) or len(frames) != count:
        raise _InvalidEvidence("frame_metadata_invalid")
    for frame in frames:
        if not isinstance(frame, dict):
            raise _InvalidEvidence("frame_metadata_invalid")
        if not isinstance(frame.get("label"), str) or not frame["label"]:
            raise _InvalidEvidence("frame_metadata_invalid")
        if not isinstance(frame.get("media_type"), str) or not frame[
            "media_type"
        ].startswith("image/"):
            raise _InvalidEvidence("frame_metadata_invalid")
        for field in ("byte_count", "width", "height"):
            _require_literal(
                require_integer, frame.get(field), f"frame.{field}", reason, minimum=1
            )
        # Result artifacts retain metadata, not the submitted image bytes.
        # Only digest format can be checked here; payload verification needs media.
        _require_digest_format(frame["sha256"])


def _validate_frame_source(frame: dict[str, Any]) -> None:
    kind = frame["source_kind"]
    index = frame["source_index"]
    count = frame["source_count"]
    timestamp = frame["source_timestamp_s"]
    if kind is not None and not isinstance(kind, str):
        raise _InvalidEvidence("frame_metadata_invalid")
    reason = "frame_metadata_invalid"
    if index is not None:
        _require_literal(require_integer, index, "source_index", reason, minimum=0)
    if count is not None:
        _require_literal(require_integer, count, "source_count", reason, minimum=1)
    if index is not None and count is not None and index >= count:
        raise _InvalidEvidence("frame_metadata_invalid")
    if timestamp is not None:
        if kind != "video":
            raise _InvalidEvidence("frame_metadata_invalid")
        _require_literal(require_number, timestamp, "source_timestamp_s", reason)


def _validate_sampling(report: dict[str, Any], manifest: dict) -> None:
    frames = manifest["frames"]
    for frame in frames:
        _validate_frame_source(frame)
    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict):
        raise _InvalidEvidence("sampling_invalid")
    max_frames = sampling.get("max_frames")
    _require_literal(
        require_integer,
        max_frames,
        "max_frames",
        "sampling_invalid",
        minimum=len(frames),
    )
    try:
        strategy = vlm_eval._normalize_frame_selection(report["frame_selection"])
    except vlm_eval.VlmEvalError as exc:
        raise _InvalidEvidence("sampling_invalid") from exc
    expected = _expected_sampling(frames, strategy, max_frames)
    # JSON comparison preserves the schema's boolean-versus-number distinction.
    if json.dumps(sampling, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise _InvalidEvidence("sampling_mismatch")


def _expected_sampling(frames: list[dict], strategy: str, max_frames: int) -> dict:
    kind = frames[0]["source_kind"] or None
    count = frames[0]["source_count"]
    if kind not in ("image-sequence", "numpy-episode", "video") or any(
        frame["source_kind"] != kind for frame in frames
    ):
        kind = None
    if any(frame["source_count"] != count for frame in frames):
        count = None
    indices = [frame["source_index"] for frame in frames]
    timestamps = [frame["source_timestamp_s"] for frame in frames]
    return {
        "strategy": strategy,
        "max_frames": max_frames,
        "selected_count": len(frames),
        "source_kind": kind,
        "source_count": count,
        "selected_indices": indices,
        "selected_timestamps_s": timestamps,
        "coverage_complete": (
            kind is not None
            and count is not None
            and all(i is not None for i in indices)
        ),
        "timestamps_complete": (
            all(timestamp is not None for timestamp in timestamps)
            if kind == "video"
            else None
        ),
    }


def _require_digest_format(value: Any) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise _InvalidEvidence("digest_invalid")


def _require_hash(actual: Any, expected: str) -> None:
    _require_digest_format(actual)
    if actual != expected:
        raise _InvalidEvidence("digest_mismatch")


def _retained_verdict(report: dict[str, Any], provider: dict):
    raw = provider["raw_response"]
    if not isinstance(raw, str) or not raw:
        raise _InvalidEvidence("provider_response_invalid")
    _require_hash(provider["raw_response_sha256"], vlm_eval._sha256_text(raw))
    try:
        data = json.loads(raw, object_pairs_hook=_unique_provider_fields)
    except json.JSONDecodeError as exc:
        raise _InvalidEvidence("provider_response_invalid") from exc
    if not isinstance(data, dict):
        raise _InvalidEvidence("provider_response_invalid")
    _validate_transport(provider)
    choice, content = _promotion_completion(data)
    if report["backend"] == "api" and data.get("model") != report["model"]:
        raise _InvalidEvidence("provider_model_mismatch")
    # Retained evidence uses content parsers, not request-profile dispatch.
    # Request tuning can evolve without changing historical parser provenance.
    try:
        parsed = _parse_retained_content(report["backend"], data, content)
    except vlm_eval.VlmEvalError as exc:
        raise _InvalidEvidence("provider_response_invalid") from exc
    _validate_provider_metadata(provider, data, choice, parsed.parser_version)
    return parsed


def _parse_retained_content(backend: str, data: dict, content: str):
    model = data.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise vlm_eval.VlmEvalError("Invalid retained model identity")
    if backend == "api":
        if model is None:
            raise vlm_eval.VlmEvalError("Missing retained hosted model identity")
        return vlm_eval._parse_api_structured_response(content, served_model=model)
    parsed = vlm_eval.parse_structured_response(content)
    return replace(parsed, served_model=model)


def _unique_provider_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key, value in pairs:
        if key in fields:
            raise _InvalidEvidence("provider_response_invalid")
        fields[key] = value
    return fields


def _promotion_completion(data: dict[str, Any]) -> tuple[dict, str]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise _InvalidEvidence("provider_response_invalid")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise _InvalidEvidence("provider_response_invalid")
    _validate_completion_status(choice, message)
    content = message["content"]
    try:
        # Validate content only; keep identity and parser provenance in the
        # original backend contract, not in this discarded strict parse result.
        vlm_eval._parse_api_structured_response(content, served_model="")
    except vlm_eval.VlmEvalError as exc:
        raise _InvalidEvidence("provider_verdict_invalid") from exc
    return choice, content


def _validate_completion_status(choice: dict, message: dict) -> None:
    refusal = message.get("refusal")
    if refusal is not None:
        if not isinstance(refusal, str):
            raise _InvalidEvidence("provider_refusal_invalid")
        if refusal.strip():
            raise _InvalidEvidence("provider_completion_refused")
    finish = choice.get("finish_reason")
    if finish == "content_filter":
        raise _InvalidEvidence("provider_completion_filtered")
    if finish != "stop":
        raise _InvalidEvidence("provider_completion_incomplete")


def _validate_transport(provider: dict[str, Any]) -> None:
    status = provider["status_code"]
    if status is not None:
        _require_literal(
            require_integer,
            status,
            "status_code",
            "provider_transport_invalid",
            minimum=200,
            maximum=299,
        )
    _require_literal(
        require_number,
        provider["latency_s"],
        "latency_s",
        "provider_metadata_invalid",
        minimum=0,
    )


def _require_literal(validator, value: Any, field: str, reason: str, **bounds):
    # Keep the shared scalar contract while exposing only bounded gate diagnostics.
    try:
        return validator(value, field=field, **bounds)
    except ValueError as exc:
        raise _InvalidEvidence(reason) from exc


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
        raise _InvalidEvidence("provider_metadata_mismatch")


def _validate_result(
    report: dict[str, Any], parsed: vlm_eval.VlmStructuredResponse
) -> None:
    score = report["score"]
    threshold = report["success_threshold"]
    for field, value in (("score", score), ("success_threshold", threshold)):
        _require_literal(
            require_number, value, field, "result_invalid", minimum=0, maximum=1
        )
    expected_score = round(parsed.score, 4)
    passed = expected_score >= threshold
    expected = {
        "score": expected_score,
        "passed": passed,
        "status": "passed" if passed else "needs_iteration",
        "served_model": parsed.served_model,
        "rationale": parsed.rationale,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise _InvalidEvidence("result_mismatch")
    _require_literal(require_boolean, report.get("passed"), "passed", "result_invalid")
    _validate_optional_provider_success(report, parsed.success, passed)


def _validate_optional_provider_success(
    report: dict, success: bool, passed: bool
) -> None:
    # Newer writers may expose these outcome/identity claims; old reports may
    # omit them. All current hosted profiles require exact served identity.
    # The strict retained verdict, not truthiness or field presence, is authoritative.
    for field, expected in (
        ("provider_success", success),
        ("provider_success_matches_score_gate", success == passed),
        ("served_model_match_enforced", report["backend"] == "api"),
    ):
        if field not in report:
            continue
        _require_literal(require_boolean, report[field], field, "result_invalid")
        if report[field] != expected:
            raise _InvalidEvidence("result_mismatch")
