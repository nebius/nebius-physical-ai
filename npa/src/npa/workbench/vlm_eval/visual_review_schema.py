"""Describe the strict visual-review response types for a neutral judge prompt."""

from __future__ import annotations

from typing import Any


TASK_STATUSES = ("complete", "partial", "failure", "unclear", "no_evidence")
FIDELITY_STATUSES = ("no_visible_issue", "issues_visible", "unclear")
FIDELITY_CATEGORIES = (
    "unsupported_geometry",
    "misalignment",
    "hallucinated_detail",
    "label_inconsistency",
    "temporal_defect",
    "other",
)
ISSUE_SEVERITIES = ("critical", "major", "minor", "unclear")
REVIEWABILITY_STATUSES = ("reviewable", "limited", "unreviewable")
IMPRESSIVENESS_STATUSES = ("strong", "moderate", "limited", "none", "unresolved")
USEFULNESS_STATUSES = ("plausible", "unsupported", "unresolved")
COMPARISON_PREFERENCES = ("A", "B", "tie", "unresolved")
COMPARISON_DIFFERENCES = ("materially_better", "equivalent", "unresolved")
CONFIDENCES = ("high", "medium", "low")


def visual_review_response_schema(mode: str) -> dict[str, Any]:
    """Build the response schema embedded in a single or paired judge prompt.

    Args:
        mode: ``single`` or ``paired`` neutral review mode.
    Returns:
        A fresh JSON Schema describing response types and structural invariants.
    Raises:
        ValueError: The requested mode is unknown.
    """

    if mode not in {"single", "paired"}:
        raise ValueError("visual review mode must be single or paired")
    fields = {"arm": {"$ref": "#/$defs/arm"}}
    if mode == "paired":
        fields = {
            "A": {"$ref": "#/$defs/arm"},
            "B": {"$ref": "#/$defs/arm"},
            "comparison": _comparison_schema(),
        }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **_object_schema(fields),
        "$defs": {
            "assertion": _assertion_schema(),
            "arm": _arm_schema(),
        },
    }


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _text_schema(*, nullable: bool = False) -> dict[str, Any]:
    return {
        "type": ["string", "null"] if nullable else "string",
        "minLength": 1,
        "pattern": r"\S",
    }


def _array_schema(items: dict[str, Any], *, minimum: int = 0) -> dict[str, Any]:
    return {"type": "array", "items": items, "minItems": minimum}


def _enum_schema(values: tuple[str, ...]) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def _citation_schema() -> dict[str, Any]:
    # The parser also checks exact submitted membership and the cited arm.
    return {
        **_array_schema({"type": "string", "pattern": "^[AB][0-9]{4}$"}, minimum=1),
        "uniqueItems": True,
    }


def _assertion_schema() -> dict[str, Any]:
    return _object_schema({"text": _text_schema(), "frame_ids": _citation_schema()})


def _assertions_schema(*, minimum: int = 0) -> dict[str, Any]:
    return _array_schema({"$ref": "#/$defs/assertion"}, minimum=minimum)


def _arm_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "task_evidence": _object_schema(
                {
                    "visible_status": _enum_schema(TASK_STATUSES),
                    "observations": _assertions_schema(minimum=1),
                    "hidden_state_limits": _array_schema(_text_schema(), minimum=1),
                }
            ),
            "artifact_fidelity": _fidelity_schema(),
            "reviewability": _reviewability_schema(),
            "impressiveness": _object_schema(
                {
                    "status": _enum_schema(IMPRESSIVENESS_STATUSES),
                    "visible_basis": _assertions_schema(minimum=1),
                    "cosmetic_only": {"type": "boolean"},
                }
            ),
            "physical_ai_usefulness": _usefulness_schema(),
        }
    )


def _fidelity_schema() -> dict[str, Any]:
    issue = _object_schema(
        {
            "category": _enum_schema(FIDELITY_CATEGORIES),
            "severity": _enum_schema(ISSUE_SEVERITIES),
            "assertion": {"$ref": "#/$defs/assertion"},
        }
    )
    return {
        **_object_schema(
            {
                "status": _enum_schema(FIDELITY_STATUSES),
                "issues": _array_schema(issue),
                "uncertainty": _text_schema(),
            }
        ),
        "if": {"properties": {"status": {"const": "issues_visible"}}},
        "then": {"properties": {"issues": {"minItems": 1}}},
        "else": {"properties": {"issues": {"maxItems": 0}}},
    }


def _reviewability_schema() -> dict[str, Any]:
    requirements = {
        "reviewable": {"strengths": {"minItems": 1}},
        "limited": {"limitations": {"minItems": 1}},
        "unreviewable": {
            "strengths": {"maxItems": 0},
            "limitations": {"minItems": 1},
        },
    }
    return {
        **_object_schema(
            {
                "status": _enum_schema(REVIEWABILITY_STATUSES),
                "strengths": _assertions_schema(),
                "limitations": _assertions_schema(),
            }
        ),
        "allOf": [
            {
                "if": {"properties": {"status": {"const": status}}},
                "then": {"properties": properties},
            }
            for status, properties in requirements.items()
        ],
    }


def _usefulness_schema() -> dict[str, Any]:
    prose = ("downstream_operation", "hypothesis", "measured_consumer_test_needed")
    fields = {
        "status": _enum_schema(USEFULNESS_STATUSES),
        "visible_basis": _assertions_schema(minimum=1),
        "required_properties": _array_schema(_text_schema()),
        **{name: _text_schema(nullable=True) for name in prose},
    }
    return {
        **_object_schema(fields),
        "if": {"properties": {"status": {"const": "unsupported"}}},
        "then": {
            "properties": {
                **{name: {"type": "null"} for name in prose},
                "required_properties": {"maxItems": 0},
            }
        },
        "else": {
            "properties": {
                **{name: _text_schema() for name in prose},
                "required_properties": {"minItems": 1},
            }
        },
    }


def _comparison_schema() -> dict[str, Any]:
    observation = _object_schema(
        {
            "text": _text_schema(),
            "A_frame_ids": _citation_schema(),
            "B_frame_ids": _citation_schema(),
        }
    )
    differences = {"A": "materially_better", "B": "materially_better"}
    differences.update(tie="equivalent", unresolved="unresolved")
    return {
        **_object_schema(
            {
                "preferred_set": _enum_schema(COMPARISON_PREFERENCES),
                "visible_evidence_difference": _enum_schema(COMPARISON_DIFFERENCES),
                "confidence": _enum_schema(CONFIDENCES),
                "observations": _array_schema(observation, minimum=1),
            }
        ),
        "oneOf": [
            {
                "properties": {
                    "preferred_set": {"const": preferred},
                    "visible_evidence_difference": {"const": difference},
                }
            }
            for preferred, difference in differences.items()
        ],
    }
