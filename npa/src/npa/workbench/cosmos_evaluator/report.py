"""Read and safely summarize Cosmos Evaluator report artifacts."""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from npa.workbench.cosmos_evaluator.evaluate import REPORT_SCHEMA
from npa.workbench.cosmos_evaluator.upstream import CosmosEvaluatorError


class CosmosEvaluatorReportError(CosmosEvaluatorError):
    """Raised when an evaluator report cannot be safely inspected.

    Args:
        None.
    Returns:
        None.
    Raises:
        None.
    """


_DIAGNOSTIC_NAMES = (
    ("attributes", "attribute_verification"),
    ("hallucination", "hallucination"),
    ("temporal", "temporal_consistency"),
    ("appearance", "appearance_fidelity"),
)
_UNAVAILABLE_STATUSES = frozenset({"error", "failed", "skipped", "unavailable"})
_REPORT_STATUSES = frozenset({"completed", "degraded", "error", "failed"})
_CLIP_STATUSES = _REPORT_STATUSES | {"skipped"}
_MEASURED_STATUSES = frozenset({"", "completed", "evaluated"})


def inspect_evaluator_report(
    input_path: str, *, storage: Any | None = None
) -> dict[str, Any]:
    """Read one evaluator artifact and return a bounded diagnostic projection.

    Args:
        input_path: Exact local JSON path or exact ``s3://`` object URI.
        storage: Optional scoped storage client used only for an S3 object read.
    Returns:
        A JSON-safe projection containing scores, dispositions, and diagnostics.
    Raises:
        CosmosEvaluatorReportError: The artifact cannot be read or is invalid.
    """
    return summarize_evaluator_report(
        _read_report_document(input_path, storage=storage)
    )


def summarize_evaluator_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and project one already-loaded evaluator report.

    Args:
        report: Parsed evaluator report object with the v1 schema marker.
    Returns:
        A bounded report summary without source paths, prompts, or metadata.
    Raises:
        CosmosEvaluatorReportError: The report shape or gate semantics are invalid.
    """
    if not isinstance(report, Mapping):
        raise CosmosEvaluatorReportError("evaluator report must be a JSON object")
    document = dict(report)
    _validate_report(document)
    variants = [_summarize_clip(clip, document) for clip in document["clips"]]
    _validate_required_evidence(document, variants)
    return _report_projection(document, variants)


def _read_report_document(input_path: str, *, storage: Any | None) -> dict[str, Any]:
    if not isinstance(input_path, str):
        raise CosmosEvaluatorReportError("--input-path must be a local path or S3 URI")
    value = input_path.strip()
    if not value:
        raise CosmosEvaluatorReportError("--input-path is required")
    if value.startswith("s3://"):
        return _read_s3_document(value, storage=storage)
    return _read_local_document(Path(_local_path(value)))


def _read_s3_document(input_path: str, *, storage: Any | None) -> dict[str, Any]:
    _validate_exact_s3_uri(input_path)
    store = storage if storage is not None else _storage()
    with tempfile.TemporaryDirectory(prefix="npa-cosmos-evaluator-report-") as temp_dir:
        local_path = Path(temp_dir) / "report.json"
        try:
            store.download_file(input_path, str(local_path))
        except Exception as exc:  # noqa: BLE001 - storage errors are sanitized for output
            raise CosmosEvaluatorReportError(
                "could not read the evaluator report from object storage"
            ) from exc
        return _read_local_document(local_path)


def _read_local_document(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CosmosEvaluatorReportError("could not read the evaluator report") from exc
    try:
        document = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CosmosEvaluatorReportError("evaluator report must be valid JSON") from exc
    if not isinstance(document, dict):
        raise CosmosEvaluatorReportError("evaluator report must be a JSON object")
    return document


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _validate_exact_s3_uri(input_path: str) -> None:
    parsed = urlparse(input_path)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.lstrip("/")
        or parsed.path.endswith("/")
        or parsed.query
        or parsed.fragment
    ):
        raise CosmosEvaluatorReportError("--input-path must be an exact S3 object URI")


def _local_path(value: str) -> str:
    return value[len("file://") :] if value.startswith("file://") else value


def _storage() -> Any:
    from npa.clients.storage import LazyStorageClient

    return LazyStorageClient()


def _validate_report(document: dict[str, Any]) -> None:
    _validate_root_shape(document)
    _validate_clip_shapes(document["clips"])
    _validate_enforcement_consistency(document)
    _validate_scores(document)
    _validate_clip_identifiers(document["clips"])
    _validate_report_counts(document)
    _validate_dispositions(document)


def _validate_root_shape(document: dict[str, Any]) -> None:
    for field in ("schema", "score", "passed", "clips"):
        if field not in document:
            raise CosmosEvaluatorReportError(f"evaluator report is missing {field}")
    if document["schema"] != REPORT_SCHEMA:
        raise CosmosEvaluatorReportError("evaluator report has an unsupported schema")
    _require_bool(document["passed"], "passed")
    _validate_status(document, "status", _REPORT_STATUSES)
    _validate_optional_score(document, "threshold")
    _validate_count_fields(document, ("clip_count", "passed_clips"))
    _validate_modes(document)
    if not isinstance(document["clips"], list):
        raise CosmosEvaluatorReportError("clips must be a JSON array")


def _validate_clip_shapes(clips: list[Any]) -> None:
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict):
            raise CosmosEvaluatorReportError(f"clips.{index} must be a JSON object")
        _validate_clip_shape(clip, index)


def _validate_clip_shape(clip: dict[str, Any], index: int) -> None:
    location = f"clips.{index}"
    for field in ("clip_id", "score", "passed"):
        if field not in clip:
            raise CosmosEvaluatorReportError(f"{location} is missing {field}")
    _require_text(clip["clip_id"], f"{location}.clip_id")
    _require_bool(clip["passed"], f"{location}.passed")
    _validate_status(clip, "status", _CLIP_STATUSES)
    _validate_optional_booleans(
        clip, ("input_conditioned", "temporal_enforced", "appearance_enforced")
    )
    _validate_skipped(clip, location)
    for _name, field in _DIAGNOSTIC_NAMES:
        if field in clip:
            _validate_diagnostic_shape(clip[field], f"{location}.{field}")


def _validate_status(
    value: Mapping[str, Any], field: str, allowed: frozenset[str]
) -> None:
    if field not in value:
        return
    status = _require_text(value[field], field)
    if status not in allowed:
        raise CosmosEvaluatorReportError(f"{field} has an unsupported status")


def _validate_modes(document: Mapping[str, Any]) -> None:
    for field in ("temporal_mode", "appearance_mode"):
        if field in document and (
            not isinstance(document[field], str)
            or document[field] not in {"advisory", "required"}
        ):
            raise CosmosEvaluatorReportError(f"{field} must be advisory or required")


def _validate_optional_booleans(
    document: Mapping[str, Any], fields: tuple[str, ...]
) -> None:
    for field in fields:
        if field in document:
            _require_bool(document[field], field)


def _validate_skipped(clip: Mapping[str, Any], location: str) -> None:
    if "skipped" not in clip:
        return
    skipped = clip["skipped"]
    if not isinstance(skipped, list):
        raise CosmosEvaluatorReportError(f"{location}.skipped must be an array")
    for index, reason in enumerate(skipped):
        _require_text(reason, f"{location}.skipped.{index}")


def _validate_enforcement_consistency(document: Mapping[str, Any]) -> None:
    for clip in document["clips"]:
        _validate_companion_enforcement("temporal", clip, document)
        _validate_companion_enforcement("appearance", clip, document)


def _validate_companion_enforcement(
    name: str, clip: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    flag = f"{name}_enforced"
    mode = f"{name}_mode"
    conditioned = clip.get("input_conditioned")
    enforced = clip.get(flag)
    if conditioned is False and enforced is True:
        raise CosmosEvaluatorReportError(f"{flag} requires an input-conditioned clip")
    if conditioned is not True or mode not in report or flag not in clip:
        return
    if report[mode] == "required" and enforced is False:
        raise CosmosEvaluatorReportError(f"{flag} contradicts the required run mode")
    if report[mode] == "advisory" and enforced is True:
        raise CosmosEvaluatorReportError(f"{flag} contradicts the advisory run mode")


def _validate_scores(document: dict[str, Any]) -> None:
    _require_score(document["score"], "score")
    _validate_optional_score(document, "threshold")
    for index, clip in enumerate(document["clips"]):
        _require_score(clip["score"], f"clips.{index}.score")


def _require_score(value: Any, location: str) -> None:
    if type(value) not in {int, float} or (
        type(value) is float and not math.isfinite(value)
    ):
        raise CosmosEvaluatorReportError(f"{location} must be a finite numeric score")
    if not 0 <= value <= 1:
        raise CosmosEvaluatorReportError(f"{location} must be between 0 and 1")


def _validate_optional_score(document: Mapping[str, Any], field: str) -> None:
    if field in document:
        _require_score(document[field], field)


def _require_bool(value: Any, location: str) -> None:
    if type(value) is not bool:
        raise CosmosEvaluatorReportError(f"{location} must be a boolean disposition")


def _require_text(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise CosmosEvaluatorReportError(f"{location} must be safe non-empty text")
    return value


def _validate_count_fields(
    document: Mapping[str, Any], fields: tuple[str, ...]
) -> None:
    for field in fields:
        if field in document and (
            type(document[field]) is not int or document[field] < 0
        ):
            raise CosmosEvaluatorReportError(f"{field} must be a non-negative integer")


def _validate_diagnostic_shape(value: Any, location: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise CosmosEvaluatorReportError(f"{location} must be an object or null")
    status = _diagnostic_status(value, location=f"{location}.status")
    if status in _UNAVAILABLE_STATUSES:
        if "score" in value or "passed" in value:
            raise CosmosEvaluatorReportError(
                f"{location} cannot mix an unavailable status with a verdict"
            )
        return
    if status not in _MEASURED_STATUSES:
        raise CosmosEvaluatorReportError(f"{location} has an unsupported status")
    if "score" not in value or "passed" not in value:
        raise CosmosEvaluatorReportError(f"{location} must contain score and passed")
    _require_score(value["score"], f"{location}.score")
    _require_bool(value["passed"], f"{location}.passed")
    if "threshold" in value:
        _require_score(value["threshold"], f"{location}.threshold")
        if value["passed"] and value["score"] < value["threshold"]:
            raise CosmosEvaluatorReportError(
                f"{location} passes below its declared threshold"
            )
    _validate_count_fields(
        value, ("total_checks", "passed_checks", "failed_checks", "total_frames")
    )


def _validate_clip_identifiers(clips: list[dict[str, Any]]) -> None:
    identifiers = [clip["clip_id"] for clip in clips]
    if len(identifiers) != len(set(identifiers)):
        raise CosmosEvaluatorReportError("evaluator report contains duplicate clip IDs")


def _validate_report_counts(document: dict[str, Any]) -> None:
    clips = document["clips"]
    if document.get("clip_count", len(clips)) != len(clips):
        raise CosmosEvaluatorReportError("clip_count does not match the clip records")
    passed = sum(clip["passed"] for clip in clips)
    if document.get("passed_clips", passed) != passed:
        raise CosmosEvaluatorReportError("passed_clips does not match the clip records")


def _validate_dispositions(document: dict[str, Any]) -> None:
    if document["passed"] and document.get("status") != "completed":
        raise CosmosEvaluatorReportError("a non-completed evaluator report cannot pass")
    for clip in document["clips"]:
        if clip["passed"] and clip.get("status") != "completed":
            raise CosmosEvaluatorReportError("a non-completed clip cannot pass")
    if document["passed"] and not document["clips"]:
        raise CosmosEvaluatorReportError("an empty evaluator report cannot pass")
    if document["passed"] and any(not clip["passed"] for clip in document["clips"]):
        raise CosmosEvaluatorReportError("a passing report contains a failed clip")
    _validate_report_threshold(document)


def _validate_report_threshold(document: dict[str, Any]) -> None:
    threshold = document.get("threshold")
    if document["passed"] and threshold is not None and document["score"] < threshold:
        raise CosmosEvaluatorReportError(
            "a passing report score is below its threshold"
        )
    if threshold is not None and any(
        clip["passed"] and clip["score"] < threshold for clip in document["clips"]
    ):
        raise CosmosEvaluatorReportError(
            "a passing clip score is below the report threshold"
        )


def _summarize_clip(clip: dict[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = {
        name: _diagnostic_projection(name, field, clip, report)
        for name, field in _DIAGNOSTIC_NAMES
    }
    return {
        "id": clip["clip_id"],
        "status": clip.get("status", "not_evaluated"),
        "score": clip["score"],
        "passed": clip["passed"],
        "input_conditioned": clip.get("input_conditioned"),
        "diagnostics": diagnostics,
        "evidence_complete": _required_evidence_is_complete(diagnostics),
    }


def _diagnostic_projection(
    name: str, field: str, clip: dict[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    enforcement = _diagnostic_enforcement(name, clip, report)
    if field not in clip:
        return {"enforcement": enforcement, "status": "not_evaluated"}
    value = clip[field]
    if value is None:
        return {"enforcement": enforcement, "status": _skipped_status(name, clip)}
    status = _diagnostic_status(value)
    if status in _UNAVAILABLE_STATUSES:
        return {"enforcement": enforcement, "status": _normalized_status(status)}
    result = {
        "enforcement": enforcement,
        "status": "error" if _contains_error(value) else "evaluated",
        "score": value["score"],
        "passed": value["passed"],
    }
    _add_diagnostic_counts(result, value)
    return result


def _diagnostic_enforcement(
    name: str, clip: Mapping[str, Any], report: Mapping[str, Any]
) -> str:
    if name == "attributes":
        return "required"
    conditioned = clip.get("input_conditioned")
    if conditioned is None:
        return "unverified"
    if name == "hallucination":
        return "required" if conditioned else "advisory"
    return _companion_enforcement(name, conditioned, clip, report)


def _companion_enforcement(
    name: str, conditioned: bool, clip: Mapping[str, Any], report: Mapping[str, Any]
) -> str:
    if not conditioned:
        return "advisory"
    flag = f"{name}_enforced"
    mode = f"{name}_mode"
    if flag in clip:
        return "required" if clip[flag] else "advisory"
    if report.get(mode) == "required":
        return "required"
    return "unverified"


def _skipped_status(name: str, clip: Mapping[str, Any]) -> str:
    reasons = " ".join(str(item).lower() for item in clip.get("skipped", []))
    token = "attribute" if name == "attributes" else name
    if token not in reasons:
        return "unavailable"
    if "error" in reasons or "failed" in reasons:
        return "error"
    if "unavailable" in reasons or "not available" in reasons:
        return "unavailable"
    return "skipped"


def _diagnostic_status(
    value: Mapping[str, Any], *, location: str = "diagnostic.status"
) -> str:
    if "status" not in value:
        return ""
    return _require_text(value["status"], location)


def _normalized_status(status: str) -> str:
    return "error" if status in {"error", "failed"} else status


def _contains_error(value: Mapping[str, Any]) -> bool:
    if value.get("error"):
        return True
    checks = value.get("checks")
    return isinstance(checks, list) and any(
        isinstance(check, dict) and check.get("error") for check in checks
    )


def _add_diagnostic_counts(target: dict[str, Any], value: Mapping[str, Any]) -> None:
    keys = ("total_checks", "passed_checks", "failed_checks", "total_frames")
    counts = {key: value[key] for key in keys if key in value}
    if counts:
        target["counts"] = counts
    if "threshold" in value:
        target["threshold"] = value["threshold"]


def _required_evidence_is_complete(diagnostics: Mapping[str, Any]) -> bool:
    return not any(
        value["enforcement"] == "unverified"
        or (value["enforcement"] == "required" and value["status"] != "evaluated")
        for value in diagnostics.values()
    )


def _validate_required_evidence(
    document: Mapping[str, Any], variants: list[dict[str, Any]]
) -> None:
    if any(_missing_required_evidence(item) for item in variants if item["passed"]):
        raise CosmosEvaluatorReportError(
            "a passing clip has missing required diagnostic evidence"
        )
    if any(
        item["passed"] and not _required_diagnostics_pass(item) for item in variants
    ):
        raise CosmosEvaluatorReportError(
            "a passing clip has a failed required diagnostic"
        )


def _missing_required_evidence(variant: Mapping[str, Any]) -> bool:
    return any(
        diagnostic["enforcement"] == "required" and diagnostic["status"] != "evaluated"
        for diagnostic in variant["diagnostics"].values()
    )


def _required_diagnostics_pass(variant: Mapping[str, Any]) -> bool:
    return all(
        diagnostic["status"] == "evaluated" and diagnostic["passed"] is True
        for diagnostic in variant["diagnostics"].values()
        if diagnostic["enforcement"] == "required"
    )


def _report_projection(
    document: Mapping[str, Any], variants: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "report_status": document.get("status", "not_evaluated"),
        "reported_gate": _reported_gate(document),
        "score_is_calibrated_confidence": False,
        "evaluation_state": _evaluation_state(variants),
        "evidence_complete": bool(variants)
        and all(item["evidence_complete"] for item in variants),
        "diagnostic_counts": _diagnostic_counts(variants),
        "multiview_assessment": _multiview_projection(document),
        "limitations": list(_LIMITATIONS),
        "variants": variants,
    }


def _reported_gate(document: Mapping[str, Any]) -> dict[str, Any]:
    gate = {"score": document["score"], "passed": document["passed"]}
    if "threshold" in document:
        gate["threshold"] = document["threshold"]
    return gate


def _evaluation_state(variants: list[dict[str, Any]]) -> str:
    if not variants:
        return "ungraded"
    if all(item["evidence_complete"] for item in variants):
        return "graded"
    return "incomplete"


def _diagnostic_counts(variants: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for name, _field in _DIAGNOSTIC_NAMES:
        statuses = [item["diagnostics"][name]["status"] for item in variants]
        counts[name] = {
            status: statuses.count(status) for status in sorted(set(statuses))
        }
    return counts


def _multiview_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    keys = {"multiview", "multiview_assessment"}
    presented = bool(keys & document.keys()) or any(
        bool(keys & clip.keys()) for clip in document["clips"]
    )
    return {"presented": presented, "status": "not_evaluated", "supported": False}


_LIMITATIONS = [
    "The reported score is an evaluator gate score, not calibrated confidence.",
    "This report inspection does not establish semantic material validation.",
    "Multi-view assessment is not evaluated by this report inspection.",
    "Missing diagnostic evidence is never averaged into a passing score.",
]


__all__ = [
    "CosmosEvaluatorReportError",
    "inspect_evaluator_report",
    "summarize_evaluator_report",
]
