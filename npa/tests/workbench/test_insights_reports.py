"""Behavioral contracts for shared report inspection and its service boundaries."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from npa.cli.main import app
from npa.sdk.workbench import cosmos_evaluator, insights
from npa.workbench.dataset.schemas import ValidateRequest
from npa.workbench.dataset.validation import validate_manifest
from npa.workbench.insights.reports import InsightsReportError, inspect_report
from npa.workbench.insights.service import create_app
from npa.workbench.storage_scope import (
    StorageAuthorizationError,
    StorageScope,
    use_storage_scope,
)


def _cosmos_report() -> dict:
    return {
        "schema": "npa.cosmos_evaluator.report.v1",
        "status": "completed",
        "score": 0.9,
        "passed": True,
        "threshold": 0.8,
        "clips": [
            {
                "clip_id": "variant-001",
                "status": "completed",
                "score": 0.9,
                "passed": True,
                "input_conditioned": False,
                "attribute_verification": {"score": 1.0, "passed": True},
            }
        ],
        "metadata": {"prompt": "private-source-marker"},
    }


def _write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "selected-report.json"
    path.write_text(json.dumps(document))
    return path


def _dataset_report(tmp_path: Path, *, corrupt: bool = False) -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "record-001",
                        "modality": "camera",
                        "uri": "private-source-marker",
                    }
                ],
                "quality_stats": {
                    "record_count": 1,
                    "mean_completeness": 1.0,
                    "corrupt_count": int(corrupt),
                },
            }
        )
    )
    response = validate_manifest(
        ValidateRequest(
            input_uri=str(manifest),
            output_uri=str(tmp_path / "validation"),
        )
    )
    return Path(response.report_uri)


def test_shared_cosmos_summary_preserves_existing_sdk_contract(tmp_path: Path) -> None:
    path = _write(tmp_path, _cosmos_report())
    common = insights.report(input_path=str(path))
    assert common["source_schema"] == "npa.cosmos_evaluator.report.v1"
    assert common["summary"] == cosmos_evaluator.report(input_path=str(path))
    assert common["reported_gate"] == common["summary"]["reported_gate"]
    assert "private-source-marker" not in json.dumps(common)


@pytest.mark.parametrize("corrupt", [False, True])
def test_second_adapter_reads_real_dataset_producer_output(
    tmp_path: Path, corrupt: bool
) -> None:
    path = _dataset_report(tmp_path, corrupt=corrupt)
    common = insights.report(input_path=str(path))
    assert common["tool"] == "dataset"
    assert common["reported_gate"]["passed"] is (not corrupt)
    assert "score" not in common["reported_gate"]
    assert common["summary"]["metrics"]["corruption_rate"] == int(corrupt)
    assert common["summary"]["failed_check_count"] == int(corrupt)
    assert common["evidence_complete"] is True
    assert str(tmp_path) not in json.dumps(common)
    assert "private-source-marker" not in json.dumps(common)


def test_dataset_adapter_excludes_free_form_check_text(tmp_path: Path) -> None:
    path = _dataset_report(tmp_path, corrupt=True)
    report = json.loads(path.read_text())
    report["failed_checks"] = ["private-source-marker"]
    path.write_text(json.dumps(report))
    assert "private-source-marker" not in json.dumps(inspect_report(str(path)))


@pytest.mark.parametrize(
    "mutation",
    [
        "gate",
        "counts",
        "rate",
        "thresholds",
        "bool-count",
        "huge-rate",
        "bool-stats-count",
    ],
)
def test_dataset_adapter_rejects_contradictory_or_missing_evidence(
    tmp_path: Path, mutation: str
) -> None:
    path = _dataset_report(tmp_path, corrupt=True)
    report = json.loads(path.read_text())
    if mutation == "gate":
        report["passed"] = True
    elif mutation == "counts":
        report["quality_stats"]["corrupt_count"] = 2
    elif mutation == "rate":
        report["corruption_rate"] = 0.0
    elif mutation == "thresholds":
        del report["thresholds"]
    elif mutation == "huge-rate":
        report["corruption_rate"] = 10**1000
    elif mutation == "bool-stats-count":
        report["quality_stats"]["record_count"] = True
    else:
        report["record_count"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(InsightsReportError):
        inspect_report(str(path))


@pytest.mark.parametrize(
    "text",
    [
        '{"schema":"private-source-marker","score":1,"passed":true}',
        '{"schema":[]}',
        "[1,2]",
        '{"schema":"first","schema":"second"}',
        '{"score":NaN}',
    ],
)
def test_unknown_or_invalid_json_is_not_guessed_or_echoed(
    tmp_path: Path, text: str
) -> None:
    path = tmp_path / "unknown.json"
    path.write_text(text)
    with pytest.raises(InsightsReportError) as error:
        inspect_report(str(path))
    assert "private-source-marker" not in str(error.value)


def test_shared_reader_reuses_cosmos_gate_validation(tmp_path: Path) -> None:
    document = _cosmos_report()
    document["clips"][0]["attribute_verification"]["passed"] = False
    with pytest.raises(InsightsReportError, match="failed required"):
        inspect_report(str(_write(tmp_path, document)))


class _ExactStorage:
    def __init__(self, source: Path, *, fail: bool = False):
        self.source = source
        self.fail = fail
        self.calls: list[tuple[str, Path]] = []

    def download_file(self, uri: str, destination: str) -> str:
        path = Path(destination)
        self.calls.append((uri, path))
        assert path.parent.stat().st_mode & 0o777 == 0o700
        if self.fail:
            raise RuntimeError("private-source-marker")
        shutil.copyfile(self.source, path)
        return destination


@pytest.mark.parametrize("fail", [False, True])
def test_exact_s3_read_cleans_private_staging_on_success_or_failure(
    tmp_path: Path, fail: bool
) -> None:
    storage = _ExactStorage(_dataset_report(tmp_path), fail=fail)
    uri = "s3://example-bucket/reports/selected.json"
    if fail:
        with pytest.raises(InsightsReportError) as error:
            insights.report(input_path=uri, storage=storage)
        assert "private-source-marker" not in str(error.value)
    else:
        assert insights.report(input_path=uri, storage=storage)["tool"] == "dataset"
    assert len(storage.calls) == 1 and storage.calls[0][0] == uri
    assert not storage.calls[0][1].parent.exists()


def test_scoped_s3_denial_happens_before_storage_io(tmp_path: Path) -> None:
    storage = _ExactStorage(_dataset_report(tmp_path))
    scope = StorageScope.from_config(s3_roots=["s3://example-bucket/allowed"])
    with use_storage_scope(scope), pytest.raises(StorageAuthorizationError):
        insights.report(
            input_path="s3://example-bucket/elsewhere/report.json", storage=storage
        )
    assert storage.calls == []


@pytest.mark.parametrize(
    "uri",
    [
        "s3://example-bucket/",
        "s3://example-bucket/reports/",
        "s3://example-bucket/report.json?private-source-marker",
        "https://example.invalid/report.json",
        "s3://[",
    ],
)
def test_report_rejects_prefixes_and_remote_urls_without_io(
    tmp_path: Path, uri: str
) -> None:
    storage = _ExactStorage(_dataset_report(tmp_path))
    with pytest.raises(InsightsReportError):
        insights.report(input_path=uri, storage=storage)
    assert storage.calls == []


def test_report_service_requires_auth_and_contains_local_reads(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    path = _dataset_report(allowed)
    client = TestClient(
        create_app(token="synthetic-token", allowed_local_roots=[allowed])
    )
    assert client.post("/report", json={"input_path": str(path)}).status_code == 401
    headers = {"Authorization": "Bearer synthetic-token"}
    response = client.post("/report", headers=headers, json={"input_path": str(path)})
    assert response.status_code == 200 and response.json()["tool"] == "dataset"
    denied = client.post(
        "/report", headers=headers, json={"input_path": str(tmp_path / "outside.json")}
    )
    assert denied.status_code == 403
    symlink = allowed / "escape.json"
    symlink.symlink_to(tmp_path / "outside.json")
    assert (
        client.post(
            "/report", headers=headers, json={"input_path": str(symlink)}
        ).status_code
        == 403
    )


def test_service_reports_invalid_schema_as_bad_request(tmp_path: Path) -> None:
    path = _write(tmp_path, {"schema": "private-source-marker"})
    client = TestClient(create_app(auth_mode="none", allowed_local_roots=[tmp_path]))
    response = client.post("/report", json={"input_path": str(path)})
    assert response.status_code == 400
    assert "private-source-marker" not in response.text


def test_cli_and_sdk_service_use_the_same_projection(
    tmp_path: Path, monkeypatch
) -> None:
    path = _dataset_report(tmp_path, corrupt=True)
    client = TestClient(create_app(auth_mode="none", allowed_local_roots=[tmp_path]))

    def transport(method, url, **kwargs):
        import httpx

        response = client.request(method, "/report", json=kwargs["json"])
        return httpx.Response(
            response.status_code,
            json=response.json(),
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(insights.httpx, "request", transport)
    expected = insights.report(input_path=str(path))
    assert (
        insights.report(
            input_path=str(path), service=True, endpoint="https://example.invalid"
        )
        == expected
    )
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "insights",
            "report",
            "--input-path",
            str(path),
            "--service",
            "--endpoint",
            "https://example.invalid",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == expected
    assert expected["reported_gate"]["passed"] is False


def test_cli_text_shows_source_metrics_and_limits(tmp_path: Path) -> None:
    path = _dataset_report(tmp_path)
    result = CliRunner().invoke(
        app, ["workbench", "insights", "report", "--input-path", str(path)]
    )
    assert result.exit_code == 0, result.output
    assert "dataset" in result.output and "corruption_rate" in result.output
    assert "not evaluator scores" in result.output


def test_service_mode_rejects_an_embedded_storage_client(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="embedded mode"):
        insights.report(
            input_path=str(tmp_path / "report.json"), storage=object(), service=True
        )


def test_cli_rejects_blank_input_without_echoing_validation_metadata() -> None:
    result = CliRunner().invoke(
        app, ["workbench", "insights", "report", "--input-path", " "]
    )
    assert result.exit_code == 1
    assert "input_path must be a nonempty string" in result.output
