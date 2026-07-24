from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from npa.workbench.insights.analytics import (
    InsightsQueryError,
    build_dashboard,
    compare_runs,
    query_metrics,
    traverse_lineage,
)
from npa.workbench.insights.schemas import (
    COMPARISON_SCHEMA,
    DASHBOARD_SCHEMA,
    METRIC_RECORD_SCHEMA,
    CompareRequest,
    DashboardRequest,
    IngestRunRequest,
    LineageRequest,
    QueryRequest,
    RecordRequest,
)
from npa.workbench.insights.store import (
    InsightsStoreError,
    ingest_run,
    read_edges,
    read_records,
    record_metrics,
)


def _metric(run_id: str, name: str, value: float, **kw: Any) -> dict[str, Any]:
    return {"run_id": run_id, "metric_name": name, "value": value, **kw}


def _seed_two_runs(store: str) -> None:
    record_metrics(
        RecordRequest(
            output_uri=store,
            records=[
                _metric("r1", "accuracy", 0.80, tool="rl", stage="eval"),
                _metric("r1", "corruption_rate", 0.20, tool="dataset", stage="validate"),
                _metric("r1", "latency", 1.00, tool="rl", stage="eval"),
                _metric("r2", "accuracy", 0.90, tool="rl", stage="eval"),
                _metric("r2", "corruption_rate", 0.10, tool="dataset", stage="validate"),
                _metric("r2", "latency", 1.20, tool="rl", stage="eval"),
            ],
        )
    )


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------
def test_record_appends_records_and_edges(tmp_path: Path) -> None:
    store = str(tmp_path / "store")
    response = record_metrics(
        RecordRequest(
            output_uri=store,
            records=[_metric("r1", "accuracy", 0.9, tool="rl")],
            edges=[{"from_uri": "s3://a", "to_uri": "s3://b", "relation": "derived_from"}],
        )
    )
    assert response.recorded_count == 1
    assert response.edge_count == 1
    assert response.total_records == 1
    rows = read_records(store)
    assert rows[0]["schema"] == METRIC_RECORD_SCHEMA
    assert rows[0]["timestamp"]  # auto-filled

    again = record_metrics(RecordRequest(output_uri=store, records=[_metric("r1", "loss", 0.1)]))
    assert again.total_records == 2  # append-only


def test_record_reads_input_uri_document(tmp_path: Path) -> None:
    doc = tmp_path / "metrics.json"
    doc.write_text(json.dumps({"records": [_metric("r1", "accuracy", 0.7)], "edges": []}))
    response = record_metrics(RecordRequest(output_uri=str(tmp_path / "store"), input_uri=str(doc)))
    assert response.recorded_count == 1


def test_record_without_payload_raises(tmp_path: Path) -> None:
    with pytest.raises(InsightsStoreError):
        record_metrics(RecordRequest(output_uri=str(tmp_path / "store")))


def test_record_indexes_lancedb_seam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import npa.workbench.insights.store as store_module

    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        store_module,
        "index_metrics_in_lancedb",
        lambda records, **kw: calls.setdefault("lancedb", kw) or {"indexed": True},
    )
    record_metrics(
        RecordRequest(
            output_uri=str(tmp_path / "store"),
            records=[_metric("r1", "accuracy", 0.9)],
            lancedb_endpoint="http://lancedb.example",
        )
    )
    assert calls["lancedb"]["lancedb_endpoint"] == "http://lancedb.example"


# ---------------------------------------------------------------------------
# ingest-run (non-invasive extraction)
# ---------------------------------------------------------------------------
def _write_run_prefix(run_dir: Path) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_uri = str(run_dir / "dataset" / "manifest.json")
    (run_dir / "dataset").mkdir(parents=True, exist_ok=True)
    Path(manifest_uri).write_text(
        json.dumps(
            {
                "schema": "npa.dataset.manifest.v1",
                "dataset_id": "fleet",
                "version": "v1",
                "record_count": 4,
                "modalities": ["camera", "lidar"],
                "lineage": {"workflow_run": "run-A", "input_uris": ["s3://raw/records.json"]},
                "quality_stats": {"record_count": 4, "mean_completeness": 0.8, "corrupt_count": 1, "modalities": ["camera", "lidar"]},
                "records": [],
            }
        )
    )
    (run_dir / "validation").mkdir(parents=True, exist_ok=True)
    Path(run_dir / "validation" / "validation_report.json").write_text(
        json.dumps(
            {
                "schema": "npa.dataset.validation_report.v1",
                "source_manifest_uri": manifest_uri,
                "passed": True,
                "record_count": 4,
                "corruption_rate": 0.25,
                "failed_checks": [],
                "quality_stats": {"record_count": 4, "mean_completeness": 0.8},
                "lineage": {"workflow_run": "run-A"},
            }
        )
    )
    (run_dir / "adversarial").mkdir(parents=True, exist_ok=True)
    Path(run_dir / "adversarial" / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "npa.scenario_gen.adversarial_set.v1",
                "run_id": "run-A",
                "scenario_count": 2,
                "lineage": {"workflow_run": "run-A", "policy_uri": "s3://p/ckpt.pt", "base_config_uri": "s3://c/task.json"},
                "scenarios": [
                    {"scenario_id": "adv-0", "severity": 0.9, "diversity": 0.5},
                    {"scenario_id": "adv-1", "severity": 0.6, "diversity": 0.4},
                ],
            }
        )
    )
    (run_dir / "gate").mkdir(parents=True, exist_ok=True)
    Path(run_dir / "gate" / "decision.json").write_text(json.dumps({"decision": "promote_checkpoint"}))
    return manifest_uri


def test_ingest_run_extracts_metrics_and_lineage(tmp_path: Path) -> None:
    manifest_uri = _write_run_prefix(tmp_path / "run")
    store = str(tmp_path / "store")
    response = ingest_run(IngestRunRequest(input_uri=str(tmp_path / "run"), output_uri=store, workflow="wf"))

    assert response.scanned == 4
    # 5 (manifest) + 4 (validation) + 4 (adversarial) + 1 (decision) = 14
    assert response.recorded_count == 14
    # manifest input(1) + validation->manifest(1) + adversarial inputs(2) = 4
    assert response.edge_count == 4
    schemas = {a.schema_id for a in response.ingested}
    assert schemas == {
        "npa.dataset.manifest.v1",
        "npa.dataset.validation_report.v1",
        "npa.scenario_gen.adversarial_set.v1",
        "decision",
    }

    records = read_records(store)
    by_name = {r["metric_name"]: r for r in records if r["tool"] == "scenario_gen"}
    assert by_name["top_severity"]["value"] == 0.9
    assert by_name["scenario_count"]["value"] == 2
    gate = next(r for r in records if r["metric_name"] == "gate_promote")
    assert gate["value"] == 1.0
    assert gate["labels"]["decision"] == "promote_checkpoint"

    edges = read_edges(store)
    relations = {(e["from_uri"], e["to_uri"], e["relation"]) for e in edges}
    assert (manifest_uri, str(tmp_path / "run" / "validation" / "validation_report.json"), "evaluated_on") in relations


def test_ingest_run_empty_prefix_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(InsightsStoreError):
        ingest_run(IngestRunRequest(input_uri=str(tmp_path / "empty"), output_uri=str(tmp_path / "store")))


def test_ingest_run_skips_unknown_schema(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "unknown.json").write_text(json.dumps({"schema": "npa.other.v1", "foo": 1}))
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "npa.dataset.manifest.v1",
                "dataset_id": "d",
                "version": "v1",
                "record_count": 1,
                "modalities": ["camera"],
                "lineage": {"workflow_run": "r", "input_uris": []},
                "quality_stats": {"record_count": 1, "mean_completeness": 1.0, "corrupt_count": 0, "modalities": ["camera"]},
                "records": [],
            }
        )
    )
    response = ingest_run(IngestRunRequest(input_uri=str(run), output_uri=str(tmp_path / "store")))
    assert response.scanned == 2
    assert len(response.ingested) == 1


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------
def test_query_filters_by_facet(tmp_path: Path) -> None:
    store = str(tmp_path / "store")
    _seed_two_runs(store)
    result = query_metrics(QueryRequest(input_uri=store, run_id="r1", tool="rl"))
    assert result.backend == "jsonl"
    names = sorted(r["metric_name"] for r in result.records)
    assert names == ["accuracy", "latency"]


def test_query_threshold_predicate(tmp_path: Path) -> None:
    store = str(tmp_path / "store")
    _seed_two_runs(store)
    result = query_metrics(
        QueryRequest(input_uri=store, metric_name="accuracy", threshold_metric="accuracy", threshold_op="ge", threshold_value=0.85)
    )
    assert result.count == 1
    assert result.records[0]["run_id"] == "r2"


def test_query_lancedb_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import npa.workbench.insights.analytics as analytics

    monkeypatch.setattr(analytics, "query_metrics_in_lancedb", lambda **kw: [{"metric_name": "x"}])
    result = query_metrics(QueryRequest(input_uri=str(tmp_path / "unused"), lancedb_endpoint="http://lancedb.example"))
    assert result.backend == "lancedb"
    assert result.count == 1


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def test_compare_flags_improved_and_regressed(tmp_path: Path) -> None:
    store = str(tmp_path / "store")
    _seed_two_runs(store)
    result = compare_runs(CompareRequest(input_uri=store, base_run="r1", candidate_run="r2"))
    assert result.comparison_schema == COMPARISON_SCHEMA
    status = {m.metric_name: m.status for m in result.metrics}
    # accuracy up (higher better) -> improved; corruption_rate down (lower better)
    # -> improved; latency up (lower better) -> regressed.
    assert status["accuracy"] == "improved"
    assert status["corruption_rate"] == "improved"
    assert status["latency"] == "regressed"
    assert set(result.improved) == {"accuracy", "corruption_rate"}
    assert result.regressed == ["latency"]
    accuracy = next(m for m in result.metrics if m.metric_name == "accuracy")
    assert accuracy.delta == pytest.approx(0.1)


def test_compare_missing_run_raises(tmp_path: Path) -> None:
    store = str(tmp_path / "store")
    _seed_two_runs(store)
    with pytest.raises(InsightsQueryError):
        compare_runs(CompareRequest(input_uri=store, base_run="r1", candidate_run="does-not-exist"))


# ---------------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------------
def test_lineage_traverses_ancestors_and_descendants(tmp_path: Path) -> None:
    store = str(tmp_path / "store")
    record_metrics(
        RecordRequest(
            output_uri=store,
            records=[_metric("r1", "n", 1.0)],
            edges=[
                {"from_uri": "s3://raw", "to_uri": "s3://manifest", "relation": "produced_from"},
                {"from_uri": "s3://manifest", "to_uri": "s3://curated", "relation": "derived_from"},
                {"from_uri": "s3://manifest", "to_uri": "s3://report", "relation": "evaluated_on"},
            ],
        )
    )
    result = traverse_lineage(LineageRequest(input_uri=store, uri="s3://manifest"))
    ancestors = {(e["from_uri"], e["to_uri"]) for e in result.ancestors}
    descendants = {(e["from_uri"], e["to_uri"]) for e in result.descendants}
    assert ("s3://raw", "s3://manifest") in ancestors
    assert ("s3://manifest", "s3://curated") in descendants
    assert ("s3://manifest", "s3://report") in descendants
    assert set(result.nodes) == {"s3://raw", "s3://manifest", "s3://curated", "s3://report"}


def test_lineage_descendants_only(tmp_path: Path) -> None:
    store = str(tmp_path / "store")
    record_metrics(
        RecordRequest(
            output_uri=store,
            records=[_metric("r1", "n", 1.0)],
            edges=[{"from_uri": "s3://a", "to_uri": "s3://b", "relation": "produced_from"}],
        )
    )
    result = traverse_lineage(LineageRequest(input_uri=store, uri="s3://a", direction="descendants"))
    assert result.ancestors == []
    assert len(result.descendants) == 1


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------
def test_dashboard_groups_and_writes_html(tmp_path: Path) -> None:
    store = str(tmp_path / "store")
    _seed_two_runs(store)
    result = build_dashboard(DashboardRequest(input_uri=store, output_path=str(tmp_path / "dash"), group_by="tool"))
    assert result.dashboard_schema == DASHBOARD_SCHEMA
    assert result.total_records == 6
    assert set(result.runs) == {"r1", "r2"}
    keys = {g.key for g in result.groups}
    assert keys == {"rl", "dataset"}
    assert result.html_uri.endswith("dashboard.html")
    html = Path(result.html_uri).read_text()
    assert "NPA Insights Dashboard" in html


# ---------------------------------------------------------------------------
# service
# ---------------------------------------------------------------------------
def test_service_record_query_lineage_compare_dashboard(tmp_path: Path) -> None:
    from npa.workbench.insights.service import create_app

    client = TestClient(create_app(auth_mode="none"))
    store = str(tmp_path / "store")
    record = client.post(
        "/record",
        json={
            "output_uri": store,
            "records": [
                _metric("r1", "accuracy", 0.8),
                _metric("r2", "accuracy", 0.9),
            ],
            "edges": [{"from_uri": "s3://a", "to_uri": "s3://b", "relation": "produced_from"}],
        },
    )
    assert record.status_code == 200, record.text
    assert record.json()["total_records"] == 2

    query = client.get("/query", params={"input_uri": store, "metric_name": "accuracy"})
    assert query.status_code == 200
    assert query.json()["count"] == 2

    lineage = client.get("/lineage", params={"input_uri": store, "uri": "s3://a"})
    assert lineage.status_code == 200
    assert lineage.json()["descendants"]

    compare = client.get("/compare", params={"input_uri": store, "base_run": "r1", "candidate_run": "r2"})
    assert compare.status_code == 200
    assert compare.json()["improved"] == ["accuracy"]

    dashboard = client.get("/dashboard", params={"input_uri": store})
    assert dashboard.status_code == 200
    assert dashboard.json()["total_records"] == 2

    status = client.get("/status", params={"input_uri": store, "run_id": "r1"})
    assert status.status_code == 200
    assert status.json()["run_record_count"] == 1
    listing = client.get("/list")
    assert any(s["store_uri"] == store for s in listing.json()["stores"])


def test_service_ingest_run_endpoint_and_failure(tmp_path: Path) -> None:
    from npa.workbench.insights.service import create_app

    client = TestClient(create_app(auth_mode="none"))
    _write_run_prefix(tmp_path / "run")
    ok = client.post(
        "/ingest-run",
        json={"input_uri": str(tmp_path / "run"), "output_uri": str(tmp_path / "store")},
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["recorded_count"] == 14

    (tmp_path / "empty").mkdir()
    bad = client.post("/ingest-run", json={"input_uri": str(tmp_path / "empty"), "output_uri": str(tmp_path / "s2")})
    assert bad.status_code == 400


def test_service_compare_failure_returns_400(tmp_path: Path) -> None:
    from npa.workbench.insights.service import create_app

    client = TestClient(create_app(auth_mode="none"))
    store = str(tmp_path / "store")
    _seed_two_runs(store)
    bad = client.get("/compare", params={"input_uri": store, "base_run": "r1", "candidate_run": "nope"})
    assert bad.status_code == 400


def test_service_health_system_info_and_token_auth() -> None:
    from npa.workbench.insights.service import create_app

    open_client = TestClient(create_app(auth_mode="none"))
    assert open_client.get("/health").json()["status"] == "ok"
    assert open_client.get("/system-info").json()["tool"] == "insights"

    secure = TestClient(create_app(auth_mode="token", token="s3cr3t"))
    assert secure.get("/health").status_code == 401
    assert secure.get("/health", headers={"Authorization": "Bearer s3cr3t"}).status_code == 200


def test_sdk_workbench_namespace_exports_insights() -> None:
    from npa.sdk import workbench

    assert workbench.insights.__name__ == "npa.sdk.workbench.insights"
    for attr in ("record", "ingest_run", "query", "lineage", "compare", "dashboard"):
        assert hasattr(workbench.insights, attr)


def test_cli_and_sdk_do_not_import_heavy_ml_dependencies_at_module_level() -> None:
    npa_root = Path(__file__).resolve().parents[2]
    cli_source = (npa_root / "src/npa/cli/workbench/insights.py").read_text()
    sdk_source = (npa_root / "src/npa/sdk/workbench/insights.py").read_text()
    for source in (cli_source, sdk_source):
        assert "import torch" not in source
        assert "import lancedb" not in source
        assert "import fiftyone" not in source
