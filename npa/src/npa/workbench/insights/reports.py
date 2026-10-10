"""Inspect supported report schemas through one shared, read-only interface."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from npa.workbench.report_document import ReportDocumentError, read_report_document


class InsightsReportError(ValueError):
    """Raised when a report cannot be inspected by a supported schema adapter.

    Args:
        None.
    Returns:
        None.
    Raises:
        None.
    """


@dataclass(frozen=True)
class _ReportAdapter:
    tool: str
    summarize: Callable[[dict[str, Any]], dict[str, Any]]


def _cosmos_summary(document: dict[str, Any]) -> dict[str, Any]:
    from npa.workbench.cosmos_evaluator.report import (
        CosmosEvaluatorReportError,
        summarize_evaluator_report,
    )

    try:
        return summarize_evaluator_report(document)
    except CosmosEvaluatorReportError as exc:
        raise InsightsReportError(str(exc)) from exc


def _dataset_summary(document: dict[str, Any]) -> dict[str, Any]:
    from npa.workbench.dataset.report import (
        DatasetReportError,
        summarize_validation_report,
    )

    try:
        return summarize_validation_report(document)
    except DatasetReportError as exc:
        raise InsightsReportError(str(exc)) from exc


# Each version owns its validation and semantics; unsupported JSON is never guessed.
_ADAPTERS = MappingProxyType(
    {
        "npa.cosmos_evaluator.report.v1": _ReportAdapter(
            "cosmos_evaluator", _cosmos_summary
        ),
        "npa.dataset.validation_report.v1": _ReportAdapter("dataset", _dataset_summary),
    }
)


def inspect_report(input_path: str, *, storage: Any | None = None) -> dict[str, Any]:
    """Inspect one supported report and preserve its producer's gate decision.

    Args:
        input_path: Exact local report path or S3 object URI.
        storage: Optional scoped storage client used only for the selected object.
    Returns:
        A common envelope containing a validated, source-specific summary.
    Raises:
        InsightsReportError: The report is unreadable, unsupported or contradictory.
        StorageAuthorizationError: An active service scope denies the read.
    """
    try:
        document = read_report_document(input_path, storage=storage)
    except ReportDocumentError as exc:
        raise InsightsReportError(str(exc)) from exc
    schema = document.get("schema")
    adapter = _ADAPTERS.get(schema) if isinstance(schema, str) else None
    if adapter is None:
        raise InsightsReportError(
            "unsupported report schema; see insights report documentation"
        )
    summary = adapter.summarize(document)
    return {
        "schema": "npa.insights.report.v1",
        "source_schema": schema,
        "tool": adapter.tool,
        "reported_gate": summary["reported_gate"],
        "evaluation_state": summary["evaluation_state"],
        "evidence_complete": summary["evidence_complete"],
        "summary": summary,
    }


def format_report(payload: dict[str, Any]) -> str:
    """Render the shared envelope and its validated diagnostic details as text.

    Args:
        payload: A projection returned by ``inspect_report``.
    Returns:
        A concise text view; source values keep their original units and meaning.
    Raises:
        KeyError: The payload is not a report projection.
    """
    summary = payload["summary"]
    lines = [
        f"report: {payload['tool']} ({payload['source_schema']})",
        f"reported gate: {json.dumps(payload['reported_gate'], sort_keys=True)}",
        f"evaluation state: {payload['evaluation_state']}  evidence complete: {payload['evidence_complete']}",
    ]
    if "variants" in summary:
        lines.append("variant\tscore\tpassed\tdiagnostics")
        lines.extend(_variant_row(variant) for variant in summary["variants"])
    if "metrics" in summary:
        lines.append("metric\tvalue")
        lines.extend(f"{name}\t{value}" for name, value in summary["metrics"].items())
    if "failed_check_count" in summary:
        lines.append(f"failed checks: {summary['failed_check_count']}")
    lines.extend(summary.get("limitations", []))
    return "\n".join(lines)


def _variant_row(variant: dict[str, Any]) -> str:
    cells = []
    for name, diagnostic in variant["diagnostics"].items():
        facts = [diagnostic["enforcement"]]
        if "score" in diagnostic:
            facts.extend(
                [f"score={diagnostic['score']}", f"passed={diagnostic['passed']}"]
            )
        cells.append(f"{name}={diagnostic['status']} ({'; '.join(facts)})")
    return f"{variant['id']}\t{variant['score']}\t{variant['passed']}\t" + "; ".join(
        cells
    )
