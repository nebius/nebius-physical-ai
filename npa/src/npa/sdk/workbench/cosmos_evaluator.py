"""SDK access to read-only Cosmos Evaluator report inspection."""

from __future__ import annotations

from typing import Any

from npa.workbench.cosmos_evaluator.report import inspect_evaluator_report


def report(*, input_path: str, storage: Any | None = None) -> dict[str, Any]:
    """Inspect one existing evaluator report without executing evaluation.

    Args:
        input_path: Exact local JSON path or exact ``s3://`` report object URI.
        storage: Optional scoped storage client used only for an S3 object read.
    Returns:
        A bounded diagnostic projection that preserves reported gate outcomes.
    Raises:
        CosmosEvaluatorReportError: The selected report is unreadable or invalid.
    """
    return inspect_evaluator_report(input_path, storage=storage)


__all__ = ["report"]
