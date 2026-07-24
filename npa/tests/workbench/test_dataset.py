from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from npa.cli.workbench.dataset import app as dataset_app
from npa.workbench.dataset.curation import DatasetCurateError, curate_dataset, query_dataset
from npa.workbench.dataset.ingestion import DatasetIngestError, ingest_dataset
from npa.workbench.dataset.schemas import (
    MANIFEST_SCHEMA,
    CurateRequest,
    IngestRequest,
    QueryRequest,
    SensorSchema,
    ValidateRequest,
)
from npa.workbench.dataset.validation import DatasetValidationError, validate_manifest

runner = CliRunner()


def _raw(tmp_path: Path, records: list[dict[str, Any]] | None = None) -> str:
    default = [
        {"record_id": "r1", "modality": "camera", "uri": "s3://b/r1.png", "event": "cut_in", "location": "sf", "timestamp": "t", "quality": {"corruption": 0.0}, "embedding": [0.1]},
        {"record_id": "r2", "modality": "lidar", "uri": "s3://b/r2.bin", "event": "cut_in", "location": "la", "timestamp": "t", "quality": {"corruption": 0.0}},
        {"record_id": "r3", "modality": "camera", "uri": "s3://b/r3.png", "event": "jaywalk", "location": "sf", "quality": {"corruption": 0.0}},
    ]
    path = tmp_path / "raw.json"
    path.write_text(json.dumps({"records": records if records is not None else default}))
    return str(path)


def _ingest(tmp_path: Path, **overrides: Any):
    payload: dict[str, Any] = {
        "input_uri": _raw(tmp_path),
        "output_uri": str(tmp_path / "ds"),
        "dataset_id": "fleet",
        "version": "v1",
        "workflow_run": "run-1",
    }
    payload.update(overrides)
    return ingest_dataset(IngestRequest(**payload))


def test_ingest_normalizes_and_registers_versioned_manifest(tmp_path: Path) -> None:
    response = _ingest(tmp_path)
    assert response.record_count == 3
    assert response.status == "completed"
    assert response.quality_stats.corrupt_count == 0
    assert set(response.modalities) == {"camera", "lidar"}
    assert response.lineage.workflow_run == "run-1"

    manifest = json.loads(Path(response.manifest_uri).read_text())
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["dataset_id"] == "fleet"
    assert manifest["version"] == "v1"
    assert manifest["quality_stats"]["events"] == ["cut_in", "jaywalk"]
    assert manifest["index"] == {"indexed": False, "backend": "manifest", "table": "fleet"}


def test_ingest_missing_required_field_raises(tmp_path: Path) -> None:
    raw = _raw(tmp_path, [{"record_id": "r1", "modality": "camera"}])
    with pytest.raises(DatasetIngestError):
        ingest_dataset(IngestRequest(input_uri=raw, output_uri=str(tmp_path / "ds"), dataset_id="d"))


def test_ingest_rejects_undeclared_modality(tmp_path: Path) -> None:
    with pytest.raises(DatasetIngestError):
        ingest_dataset(
            IngestRequest(
                input_uri=_raw(tmp_path),
                output_uri=str(tmp_path / "ds"),
                dataset_id="d",
                sensor_schema=SensorSchema(modalities=["camera"]),
            )
        )


def test_ingest_rejects_duplicate_record_id(tmp_path: Path) -> None:
    raw = _raw(
        tmp_path,
        [
            {"record_id": "r1", "modality": "camera", "uri": "s3://b/a"},
            {"record_id": "r1", "modality": "camera", "uri": "s3://b/b"},
        ],
    )
    with pytest.raises(DatasetIngestError):
        ingest_dataset(IngestRequest(input_uri=raw, output_uri=str(tmp_path / "ds"), dataset_id="d"))


def test_ingest_calls_lancedb_and_fiftyone_seams(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import npa.workbench.dataset.ingestion as ing

    calls: dict[str, Any] = {}
    monkeypatch.setattr(ing, "index_in_lancedb", lambda records, **kw: calls.setdefault("lancedb", kw) or {"indexed": True, "backend": "lancedb", "table": kw["table"]})
    monkeypatch.setattr(ing, "fiftyone_handoff", lambda **kw: calls.setdefault("fiftyone", kw) or {"handoff": True})

    _ingest(tmp_path, output_uri=str(tmp_path / "ds"))
    ingest_dataset(
        IngestRequest(input_uri=_raw(tmp_path), output_uri=str(tmp_path / "ds2"), dataset_id="fleet"),
    )
    assert "lancedb" in calls
    assert "fiftyone" in calls


def test_validate_passes_clean_dataset(tmp_path: Path) -> None:
    ingested = _ingest(tmp_path)
    report = validate_manifest(
        ValidateRequest(input_uri=ingested.manifest_uri, output_uri=str(tmp_path / "val"))
    )
    assert report.passed is True
    assert report.failed_checks == []
    assert Path(report.report_uri).exists()


def test_validate_rejects_on_corruption(tmp_path: Path) -> None:
    raw = _raw(
        tmp_path,
        [
            {"record_id": "r1", "modality": "camera", "uri": "s3://b/r1", "quality": {"corruption": 0.9}},
            {"record_id": "r2", "modality": "camera", "uri": "s3://b/r2", "quality": {"corruption": 0.0}},
        ],
    )
    ingested = ingest_dataset(IngestRequest(input_uri=raw, output_uri=str(tmp_path / "ds"), dataset_id="d"))
    report = validate_manifest(
        ValidateRequest(input_uri=ingested.manifest_uri, output_uri=str(tmp_path / "val"), max_corruption_rate=0.1)
    )
    assert report.passed is False
    assert any("corruption" in check for check in report.failed_checks)


def test_validate_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError):
        validate_manifest(ValidateRequest(input_uri=str(tmp_path / "nope.json"), output_uri=str(tmp_path / "val")))


def test_curate_slices_and_threads_lineage(tmp_path: Path) -> None:
    ingested = _ingest(tmp_path)
    curated = curate_dataset(
        CurateRequest(
            input_uri=ingested.manifest_uri,
            output_uri=str(tmp_path / "cur"),
            event="cut_in",
            location="sf",
        )
    )
    assert curated.record_count == 1
    assert curated.parent_version == "v1"
    assert curated.version.startswith("v1.curated-")
    assert curated.lineage.parent_dataset_id == "fleet"
    manifest = json.loads(Path(curated.manifest_uri).read_text())
    assert manifest["parent_version"] == "v1"
    assert manifest["filter_predicate"]["event"] == "cut_in"


def test_curate_zero_match_raises(tmp_path: Path) -> None:
    ingested = _ingest(tmp_path)
    with pytest.raises(DatasetCurateError):
        curate_dataset(
            CurateRequest(input_uri=ingested.manifest_uri, output_uri=str(tmp_path / "cur"), event="does-not-exist")
        )


def test_query_manifest_backend_filters(tmp_path: Path) -> None:
    ingested = _ingest(tmp_path)
    result = query_dataset(QueryRequest(input_uri=ingested.manifest_uri, event="cut_in"))
    assert result.backend == "manifest"
    assert result.count == 2


def test_query_lancedb_backend_when_endpoint_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import npa.workbench.dataset.curation as cur

    monkeypatch.setattr(cur, "query_lancedb", lambda **kw: [{"record_id": "x"}])
    result = query_dataset(
        QueryRequest(input_uri=str(tmp_path / "unused.json"), event="cut_in", lancedb_endpoint="http://lancedb.example")
    )
    assert result.backend == "lancedb"
    assert result.count == 1


def test_ingest_endpoint_success_and_status_list(tmp_path: Path) -> None:
    from npa.workbench.dataset.service import create_app

    client = TestClient(create_app(auth_mode="none"))
    response = client.post(
        "/ingest",
        json={"input_uri": _raw(tmp_path), "output_uri": str(tmp_path / "ds"), "dataset_id": "fleet", "version": "v1"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["record_count"] == 3

    status = client.get("/status", params={"dataset_id": "fleet", "version": "v1"})
    assert status.status_code == 200
    listing = client.get("/list")
    assert any(d["dataset_id"] == "fleet" for d in listing.json()["datasets"])


def test_ingest_endpoint_failure_returns_400(tmp_path: Path) -> None:
    from npa.workbench.dataset.service import create_app

    client = TestClient(create_app(auth_mode="none"))
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"records": []}))
    response = client.post(
        "/ingest",
        json={"input_uri": str(empty), "output_uri": str(tmp_path / "ds"), "dataset_id": "fleet"},
    )
    assert response.status_code == 400


def test_validate_and_curate_and_query_endpoints(tmp_path: Path) -> None:
    from npa.workbench.dataset.service import create_app

    client = TestClient(create_app(auth_mode="none"))
    ingested = client.post(
        "/ingest",
        json={"input_uri": _raw(tmp_path), "output_uri": str(tmp_path / "ds"), "dataset_id": "fleet"},
    ).json()
    manifest_uri = ingested["manifest_uri"]

    validated = client.post("/validate", json={"input_uri": manifest_uri, "output_uri": str(tmp_path / "val")})
    assert validated.status_code == 200
    assert validated.json()["passed"] is True

    curated = client.post("/curate", json={"input_uri": manifest_uri, "output_uri": str(tmp_path / "cur"), "event": "cut_in"})
    assert curated.status_code == 200

    queried = client.get("/query", params={"input_uri": manifest_uri, "event": "cut_in"})
    assert queried.status_code == 200
    assert queried.json()["count"] == 2

    bad = client.post("/curate", json={"input_uri": manifest_uri, "output_uri": str(tmp_path / "cur2"), "event": "nope"})
    assert bad.status_code == 400


def test_health_system_info_and_token_auth() -> None:
    from npa.workbench.dataset.service import create_app

    open_client = TestClient(create_app(auth_mode="none"))
    assert open_client.get("/health").json()["status"] == "ok"
    assert open_client.get("/system-info").json()["tool"] == "dataset"

    secure = TestClient(create_app(auth_mode="token", token="s3cr3t"))
    assert secure.get("/health").status_code == 401
    assert secure.get("/health", headers={"Authorization": "Bearer s3cr3t"}).status_code == 200


def test_cli_ingest_validate_curate_query(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    ingest = runner.invoke(
        dataset_app,
        ["ingest", "--input-path", raw, "--output-path", str(tmp_path / "ds"), "--dataset-id", "fleet", "--output", "json"],
    )
    assert ingest.exit_code == 0, ingest.output
    manifest_uri = json.loads(ingest.output)["manifest_uri"]

    validate = runner.invoke(
        dataset_app,
        ["validate", "--input-path", manifest_uri, "--output-path", str(tmp_path / "val"), "--output", "json"],
    )
    assert validate.exit_code == 0, validate.output

    query = runner.invoke(
        dataset_app,
        ["query", "--input-path", manifest_uri, "--event", "cut_in", "--output", "json"],
    )
    assert json.loads(query.output)["count"] == 2


def test_cli_and_sdk_do_not_import_heavy_ml_dependencies_at_module_level() -> None:
    npa_root = Path(__file__).resolve().parents[2]
    cli_source = (npa_root / "src/npa/cli/workbench/dataset.py").read_text()
    sdk_source = (npa_root / "src/npa/sdk/workbench/dataset.py").read_text()
    for source in (cli_source, sdk_source):
        assert "import torch" not in source
        assert "import lancedb" not in source
        assert "import fiftyone" not in source


def test_sdk_workbench_namespace_exports_dataset() -> None:
    from npa.sdk import workbench

    assert workbench.dataset.__name__ == "npa.sdk.workbench.dataset"
    for attr in ("ingest", "validate", "curate", "query"):
        assert hasattr(workbench.dataset, attr)
