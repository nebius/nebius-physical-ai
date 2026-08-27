"""Terminal PAIDF candidate review command registration."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

import typer


class ReviewOutputFormat(str, Enum):
    """Output formats supported by the terminal review command."""

    text = "text"
    json = "json"


def register_review_augmented(
    app: typer.Typer,
    *,
    output_format: type,
    fail: Callable[[str], Any],
    emit: Callable[[object, Any], None],
) -> None:
    """Register the real accepted/rejected FiftyOne review exporter."""

    @app.command("review-augmented")
    def review_augmented_cmd(
        run_root_uri: str = typer.Option(
            ..., "--run-root-uri", help="Canonical PAIDF run root in S3."
        ),
        quality_disposition_uri: str = typer.Option(
            ...,
            "--quality-disposition-uri",
            help="Accepted/rejected PAIDF disposition JSON.",
        ),
        dataset_uri: str = typer.Option(
            ...,
            "--dataset-uri",
            help="Append-only S3 prefix for the portable FiftyOneDataset archive.",
        ),
        report_uri: str = typer.Option(
            ..., "--report-uri", help="S3 URI for the terminal review report."
        ),
        dataset_name: str = typer.Option(
            ..., "--dataset-name", help="Stable dataset name used by the review viewer."
        ),
        output: ReviewOutputFormat = typer.Option(
            ReviewOutputFormat.text, "--output", help="Output format."
        ),
    ) -> None:
        """Export all accepted or rejected PAIDF candidates for real FiftyOne review."""

        values = {
            "--run-root-uri": run_root_uri,
            "--quality-disposition-uri": quality_disposition_uri,
            "--dataset-uri": dataset_uri,
            "--report-uri": report_uri,
        }
        for option, value in values.items():
            if not value.strip().startswith("s3://"):
                fail(f"{option} must be an s3:// URI.")
        if not dataset_name.strip():
            fail("--dataset-name must not be empty.")
        normalized_output = output_format(output.value)

        from npa.workflows.data_factory_stages import review_terminal_candidates

        try:
            report = review_terminal_candidates(
                run_root_uri.strip(),
                quality_disposition_uri.strip(),
                dataset_uri.strip(),
                report_uri.strip(),
                dataset_name.strip(),
            )
        except Exception as exc:  # noqa: BLE001 - normalize workflow failure for CLI
            fail(f"terminal review failed: {exc}")
            return
        emit(
            {
                "status": report.get("status"),
                "engine": report.get("engine"),
                "dataset_name": report.get("dataset_name"),
                "candidate_count": report.get("candidate_count"),
                "quality_disposition": report.get("quality_disposition"),
                "review_only": report.get("review_only"),
                "promotion_eligible_count": report.get("promotion_eligible_count"),
                "report_uri": report.get("written_uri"),
            },
            normalized_output,
        )

    # The catalog argv guard resolves Typer callbacks by module + function name.
    # Registration keeps this small command out of the CLI monolith, while the
    # module-level alias preserves that introspection contract.
    globals()[review_augmented_cmd.__name__] = review_augmented_cmd
