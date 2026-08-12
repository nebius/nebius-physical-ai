"""Tests for `npa workbench fiftyone curate-augmented` (in-container paidf curation)."""

from __future__ import annotations

import json

from click.utils import strip_ansi
from typer.testing import CliRunner

from npa.cli.main import app

runner = CliRunner()


def test_curate_augmented_help_documents_flags() -> None:
    result = runner.invoke(app, ["workbench", "fiftyone", "curate-augmented", "--help"])
    output = strip_ansi(result.output)
    assert result.exit_code == 0
    for flag in ("--augment-uri", "--report-uri", "--curator-report-uri", "--dedup-threshold"):
        assert flag in output


def test_curate_augmented_rejects_non_s3_augment_uri() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "curate-augmented",
            "--augment-uri",
            "/tmp/aug",
            "--report-uri",
            "s3://b/p/curation/report.json",
        ],
    )
    assert result.exit_code == 1
    assert "--augment-uri must be an s3:// URI" in result.output


def test_curate_augmented_rejects_non_s3_report_uri() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "curate-augmented",
            "--augment-uri",
            "s3://b/p/cosmos_augmented/",
            "--report-uri",
            "/tmp/report.json",
        ],
    )
    assert result.exit_code == 1
    assert "--report-uri must be an s3:// URI" in result.output


def test_curate_augmented_invokes_curate_and_emits_summary(mocker) -> None:
    fake_report = {
        "schema": "npa.fiftyone.curation.v1",
        "clip_ids": ["c1", "c2"],
        "multiply": {"mode": "multi-variant"},
        "curation_engine": "fiftyone-brain",
        "curated_kept": 2,
        "curated_dropped": 0,
        "written_uri": "s3://b/p/curation/report.json",
        "fiftyone": {"brain": {"uniqueness": {"count": 2, "mean": 0.5}}},
    }
    curate = mocker.patch("npa.workflows.data_factory_stages.curate", return_value=fake_report)

    result = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "curate-augmented",
            "--augment-uri",
            "s3://b/p/cosmos_augmented/",
            "--report-uri",
            "s3://b/p/curation/report.json",
            "--dedup-threshold",
            "0.2",
            "--curator-report-uri",
            "s3://b/p/curation/cosmos_curator.json",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    curate.assert_called_once_with(
        "s3://b/p/cosmos_augmented/",
        "s3://b/p/curation/report.json",
        dedup_threshold=0.2,
        curator_report_uri="s3://b/p/curation/cosmos_curator.json",
    )
    payload = json.loads(result.output)
    assert payload["engine"] == "fiftyone-brain"
    assert payload["kept"] == 2
    assert payload["uniqueness"] == {"count": 2, "mean": 0.5}
