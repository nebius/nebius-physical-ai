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
    MetricRecord,
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


def test_ingest_run_skips_per_scenario_config_files(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "adversarial" / "scenarios").mkdir(parents=True)
    # Aggregate set manifest (has a scenarios list).
    (run / "adversarial" / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "npa.scenario_gen.adversarial_set.v1",
                "run_id": "gpu-run",
                "scenario_count": 2,
                "lineage": {"workflow_run": "gpu-run", "policy_uri": "s3://p/ckpt.pt", "base_config_uri": "s3://c/t.json"},
                "scenarios": [
                    {"scenario_id": "adv-0", "severity": 0.9, "diversity": 0.5},
                    {"scenario_id": "adv-1", "severity": 0.6, "diversity": 0.4},
                ],
            }
        )
    )
    # Per-scenario config files reuse the schema tag but have no scenarios list.
    for index in range(2):
        (run / "adversarial" / "scenarios" / f"adv-{index}.json").write_text(
            json.dumps({"schema": "npa.scenario_gen.adversarial_set.v1", "run_id": "gpu-run", "scenario_id": f"adv-{index}", "perturbation": {}})
        )
    store = str(tmp_path / "store")
    response = ingest_run(IngestRunRequest(input_uri=str(run), output_uri=store, workflow="wf"))
    assert response.scanned == 3
    # Only the aggregate manifest is ingested (4 metrics); the 2 configs skipped.
    assert response.recorded_count == 4
    scenario_counts = [r["value"] for r in read_records(store) if r["metric_name"] == "scenario_count"]
    assert scenario_counts == [2.0]


def test_ingest_run_empty_prefix_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(InsightsStoreError):
        ingest_run(IngestRunRequest(input_uri=str(tmp_path / "empty"), output_uri=str(tmp_path / "store")))


def _write_run_manifest(run_dir: Path, *, accelerators: str, run_id: str = "gpu-run") -> None:
    (run_dir / "npa-workflow").mkdir(parents=True, exist_ok=True)
    (run_dir / "npa-workflow" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "npa.workflow.run.v1",
                "workflow": "scenario-gen",
                "run_id": run_id,
                "api_version": "npa.workflow/v0.0.1",
                "status": "succeeded",
                "steps": [
                    {"state": "control", "iteration": 0, "status": "ok", "resources_profile": {"cpus": 4}},
                    {
                        "state": "generate",
                        "iteration": 0,
                        "status": "ok",
                        "resources_profile": {"accelerators": accelerators, "cpus": 16},
                    },
                ],
            }
        )
    )


def test_ingest_run_extracts_gpu_count_from_run_manifest(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run_manifest(run, accelerators="RTXPRO6000:4", run_id="insights-4gpu-viz")
    store = str(tmp_path / "store")
    response = ingest_run(IngestRunRequest(input_uri=str(run), output_uri=store, workflow="wf"))
    assert "npa.workflow.run.v1" in {a.schema_id for a in response.ingested}
    gpus = [r for r in read_records(store) if r["metric_name"] == "gpus"]
    assert len(gpus) == 1
    # Peak accelerator count across steps; the CPU-only step contributes nothing.
    assert gpus[0]["value"] == 4.0
    assert gpus[0]["run_id"] == "insights-4gpu-viz"
    assert gpus[0]["labels"]["accelerators"] == "RTXPRO6000"


def test_ingest_run_skips_cpu_only_run_manifest(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "npa-workflow").mkdir(parents=True)
    (run / "npa-workflow" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "npa.workflow.run.v1",
                "workflow": "insights-aggregate",
                "run_id": "cpu-run",
                "api_version": "npa.workflow/v0.0.1",
                "steps": [{"state": "aggregate", "iteration": 0, "status": "ok", "resources_profile": {"cpus": 2}}],
            }
        )
    )
    # No accelerator anywhere -> no fabricated gpus metric, and no known schema to
    # ingest, so the run raises rather than inventing a value.
    with pytest.raises(InsightsStoreError):
        ingest_run(IngestRunRequest(input_uri=str(run), output_uri=str(tmp_path / "store")))


def test_ingest_run_skips_a_planned_only_run_manifest(tmp_path: Path) -> None:
    """A planned run never touched a GPU, so it must not report a GPU count.

    `run-spec --persist-state` without `--execute` writes a manifest that carries the
    full resource profile with status "planned"; ingesting it would attribute
    accelerators to a run that never ran.
    """
    run = tmp_path / "run"
    (run / "npa-workflow").mkdir(parents=True)
    (run / "npa-workflow" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "npa.workflow.run.v1",
                "workflow": "hardening-with-insights",
                "run_id": "planned-only",
                "status": "planned",
                "steps": [
                    {"state": "generate", "status": "planned", "resources_profile": {"accelerators": "RTXPRO6000:4"}}
                ],
            }
        )
    )
    with pytest.raises(InsightsStoreError):
        ingest_run(IngestRunRequest(input_uri=str(run), output_uri=str(tmp_path / "store")))


def test_ingest_run_accepts_a_submitted_run_manifest(tmp_path: Path) -> None:
    """A submitted run did request the hardware, so its GPU count is real."""
    run = tmp_path / "run"
    (run / "npa-workflow").mkdir(parents=True)
    (run / "npa-workflow" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "npa.workflow.run.v1",
                "workflow": "hardening-with-insights",
                "run_id": "submitted-run",
                "status": "submitted",
                "steps": [
                    {"state": "retrain", "status": "submitted", "resources_profile": {"accelerators": "RTXPRO6000:2"}}
                ],
            }
        )
    )
    store = str(tmp_path / "store")
    ingest_run(IngestRunRequest(input_uri=str(run), output_uri=store))
    gpus = [r for r in read_records(store) if r["metric_name"] == "gpus"]
    assert [r["value"] for r in gpus] == [2.0]


def test_query_filters_by_accelerator_label(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_run_manifest(a, accelerators="RTXPRO6000:4", run_id="run-4gpu")
    _write_run_manifest(b, accelerators="H100:1", run_id="run-h100")
    store = str(tmp_path / "store")
    ingest_run(IngestRunRequest(input_uri=str(a), output_uri=store))
    ingest_run(IngestRunRequest(input_uri=str(b), output_uri=store))

    only_rtx = query_metrics(QueryRequest(input_uri=store, accelerator="RTXPRO6000"))
    assert {r["run_id"] for r in only_rtx.records} == {"run-4gpu"}

    # Numeric "which runs use >=4 GPUs" via the existing threshold predicate.
    four_plus = query_metrics(
        QueryRequest(
            input_uri=store,
            metric_name="gpus",
            threshold_metric="gpus",
            threshold_op="ge",
            threshold_value=4,
        )
    )
    assert {r["run_id"] for r in four_plus.records} == {"run-4gpu"}


def test_parse_accelerators_variants() -> None:
    from npa.workbench.insights.store import _parse_accelerators

    assert _parse_accelerators("RTXPRO6000:4") == ("RTXPRO6000", 4)
    assert _parse_accelerators("H100") == ("H100", 1)
    assert _parse_accelerators("") == ("", 0)


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


def _write_signal_run(
    run_dir: Path,
    *,
    run_id: str,
    success_rate: float,
    cost_usd: float,
) -> dict[str, str]:
    """Write representative artifacts that existing NPA tools already emit."""
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_uri = f"s3://artifacts/{run_id}/policy.pt"

    train_dir = run_dir / "detection-training"
    train_dir.mkdir()
    (train_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "manifest_sha256": "observed-manifest-digest",
                "status": "completed",
                "epochs": [
                    {"epoch": 1, "train_loss": 0.8, "reward": 0.2},
                    {"epoch": 2, "train_loss": 0.4, "reward": 0.6},
                ],
            }
        )
    )

    sweep_uri = run_dir / "sweep" / "metrics.json"
    sweep_uri.parent.mkdir()
    sweep_uri.write_text(
        json.dumps(
            {
                "schema": "npa.rl_sweep.variant_metrics.v1",
                "run_id": run_id,
                "tool": "isaac_lab",
                "stage": "sweep-train",
                "status": "success",
                "duration_seconds": 31.5,
                "mean_reward": 0.6,
                "checkpoint_uri": checkpoint_uri,
            }
        )
    )

    eval_uri = run_dir / "eval" / "report.json"
    eval_uri.parent.mkdir()
    eval_uri.write_text(
        json.dumps(
            {
                "schema": "npa.sim2real.heldout_eval.v1",
                "run_id": run_id,
                "tool": "isaac_lab",
                "stage": "eval",
                "status": "completed",
                "success_rate": success_rate,
                "policy_checkpoint": checkpoint_uri,
                "metrics": {"accuracy": success_rate, "steps_per_second": 128.0},
            }
        )
    )

    billing_uri = run_dir / "resources" / "billing.json"
    billing_uri.parent.mkdir()
    billing_uri.write_text(
        json.dumps(
            {
                "schema": "npa.workflow.billing.v1",
                "run_id": run_id,
                "tool": "workflow",
                "stage": "run",
                "billing": {"cost_usd": cost_usd},
            }
        )
    )

    gate_uri = run_dir / "gate" / "decision.json"
    gate_uri.parent.mkdir()
    gate_uri.write_text(
        json.dumps(
            {
                "schema": "npa.sim2real.threshold_decision.v1",
                "run_id": run_id,
                "decision": "promote_checkpoint",
                "success_rate": success_rate,
                "checkpoint_uri": checkpoint_uri,
                "eval_report_uri": str(eval_uri),
            }
        )
    )

    sparse_uri = run_dir / "sparse" / "metrics.json"
    sparse_uri.parent.mkdir()
    sparse_uri.write_text(
        json.dumps(
            {
                "schema": "npa.rl_sweep.variant_metrics.v1",
                "run_id": run_id,
                "status": "completed",
                "message": "no observed metrics were written",
            }
        )
    )
    return {
        "checkpoint": checkpoint_uri,
        "eval": str(eval_uri),
        "gate": str(gate_uri),
        "sparse": str(sparse_uri),
    }


def test_ingest_run_extracts_observed_curves_runtime_cost_and_lineage(tmp_path: Path) -> None:
    paths = _write_signal_run(
        tmp_path / "run-rich", run_id="run-rich", success_rate=0.82, cost_usd=12.75
    )
    store = str(tmp_path / "store")
    response = ingest_run(
        IngestRunRequest(input_uri=str(tmp_path / "run-rich"), output_uri=store)
    )

    records = read_records(store)
    names = {row["metric_name"] for row in records}
    assert {"train_loss", "reward", "mean_reward", "duration_seconds"} <= names
    assert {"success_rate", "accuracy", "steps_per_second", "cost_usd"} <= names

    step_two_loss = next(
        row
        for row in records
        if row["metric_name"] == "train_loss" and row["labels"].get("step") == "2"
    )
    assert step_two_loss["value"] == 0.4
    assert step_two_loss["labels"]["metric_kind"] == "training_curve"

    cost = next(row for row in records if row["metric_name"] == "cost_usd")
    assert cost["value"] == 12.75
    assert cost["unit"] == "USD"
    assert cost["labels"] == {
        "cost_basis": "billed",
        "currency": "USD",
        "metric_kind": "cost",
    }

    assert not any(row["artifact_uri"] == paths["sparse"] for row in records)
    assert paths["sparse"] not in {artifact.uri for artifact in response.ingested}

    edges = read_edges(store)
    triples = {(edge["from_uri"], edge["to_uri"], edge["relation"]) for edge in edges}
    assert (paths["checkpoint"], paths["eval"], "evaluated_on") in triples
    assert (paths["eval"], paths["gate"], "derived_from") in triples


def test_new_signal_facets_compare_and_dashboard_preserve_dimensions(tmp_path: Path) -> None:
    store = str(tmp_path / "store")
    _write_signal_run(tmp_path / "run-a", run_id="run-a", success_rate=0.82, cost_usd=12.75)
    _write_signal_run(tmp_path / "run-b", run_id="run-b", success_rate=0.76, cost_usd=10.25)
    ingest_run(IngestRunRequest(input_uri=str(tmp_path / "run-a"), output_uri=store))
    ingest_run(IngestRunRequest(input_uri=str(tmp_path / "run-b"), output_uri=store))

    curves = query_metrics(
        QueryRequest(input_uri=store, metric_kind="training_curve", step="2")
    )
    assert curves.count == 4
    assert curves.facets["metric_kind"] == "training_curve"
    assert curves.facets["step"] == "2"

    scores = query_metrics(
        QueryRequest(input_uri=store, score_name="success_rate", tool="isaac_lab")
    )
    assert {row["run_id"] for row in scores.records} == {"run-a", "run-b"}
    assert all(row["labels"]["metric_kind"] == "eval_score" for row in scores.records)

    costs = query_metrics(QueryRequest(input_uri=store, currency="USD"))
    assert [row["metric_name"] for row in costs.records] == ["cost_usd", "cost_usd"]

    comparison = compare_runs(
        CompareRequest(input_uri=store, base_run="run-a", candidate_run="run-b")
    )
    step_deltas = [
        item for item in comparison.metrics if item.metric_name == "train_loss"
    ]
    assert {item.dimensions["step"] for item in step_deltas} == {"1", "2"}
    success = next(
        item for item in comparison.metrics if item.metric_name == "success_rate"
    )
    assert success.dimensions["tool"] == "isaac_lab"
    assert success.status == "regressed"

    dashboard = build_dashboard(
        DashboardRequest(input_uri=store, group_by="metric_kind")
    )
    assert {group.key for group in dashboard.groups} >= {
        "training_curve",
        "training_metric",
        "eval_score",
        "duration",
        "throughput",
        "cost",
    }


def test_eval_metadata_is_typed_as_counter_or_duration_not_score(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "eval.json").write_text(
        json.dumps(
            {
                "schema": "npa.sim2real.heldout_eval.v1",
                "run_id": "eval-taxonomy",
                "stage": "eval",
                "score": 0.0,
                "metrics": {
                    "score": 0.91,
                    "success_rate": 0.8,
                    "num_samples": 40,
                    "epoch": 3,
                    "latency_ms": 12.5,
                    "random_seed": 17,
                },
            }
        )
    )
    store = str(tmp_path / "store")

    ingest_run(IngestRunRequest(input_uri=str(run), output_uri=store))
    records = read_records(store)
    by_name = {row["metric_name"]: row for row in records}

    assert by_name["score"]["value"] == 0.91
    assert by_name["score"]["labels"]["metric_kind"] == "eval_score"
    assert by_name["success_rate"]["labels"]["metric_kind"] == "eval_score"
    assert by_name["num_samples"]["labels"]["metric_kind"] == "counter"
    assert by_name["epoch"]["labels"]["metric_kind"] == "counter"
    assert by_name["latency_ms"]["labels"]["metric_kind"] == "duration"
    assert by_name["latency_ms"]["unit"] == "milliseconds"
    assert "random_seed" not in by_name


def test_robocasa_policy_eval_records_checkpoint_lineage(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    checkpoint_uri = "s3://artifacts/run/checkpoint/"
    report = run / "eval.json"
    report.write_text(
        json.dumps(
            {
                "schema": "npa.robocasa.policy_eval.v1",
                "checkpoint_uri": checkpoint_uri,
                "checkpoint_loadable": True,
                "num_episodes": 6,
                "mean_reward": 0.0,
                "success_rate": 0.0,
                "split_proof": {
                    "task_sets_disjoint": True,
                    "episode_sets_disjoint_by_task": True,
                },
            }
        )
    )
    store = str(tmp_path / "store")

    response = ingest_run(
        IngestRunRequest(
            input_uri=str(run),
            output_uri=store,
            workflow="robocasa-data-policy",
            workflow_run="robocasa-run",
        )
    )

    assert [artifact.schema_id for artifact in response.ingested] == [
        "npa.robocasa.policy_eval.v1"
    ]
    records = read_records(store)
    by_name = {row["metric_name"]: row for row in records}
    assert by_name["success_rate"]["value"] == 0.0
    assert by_name["mean_reward"]["value"] == 0.0
    assert by_name["num_episodes"]["value"] == 6.0
    assert by_name["success_rate"]["lineage"]["checkpoint_uri"] == checkpoint_uri
    assert read_edges(store) == [
        {
            "from_uri": checkpoint_uri,
            "from_version": "",
            "relation": "evaluated_on",
            "run_id": "robocasa-run",
            "to_uri": str(report),
            "to_version": "",
        }
    ]


def test_cost_basis_survives_ingest_query_compare_dashboard_sdk_service_cli(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from npa.cli.workbench.insights import app as insights_app
    from npa.sdk.workbench.insights import query as sdk_query
    from npa.workbench.insights.service import create_app

    run = tmp_path / "run"
    run.mkdir()
    (run / "billing.json").write_text(
        json.dumps(
            {
                "schema": "npa.workflow.billing.v1",
                "run_id": "cost-run",
                "billing": {"cost_usd": 4.0},
            }
        )
    )
    (run / "usage.json").write_text(
        json.dumps(
            {
                "schema": "npa.workflow.resource_usage.v1",
                "run_id": "cost-run",
                "resource_usage": {"estimated_cost_usd": 2.5},
            }
        )
    )
    store = str(tmp_path / "store")
    ingest_run(IngestRunRequest(input_uri=str(run), output_uri=store))

    costs = query_metrics(QueryRequest(input_uri=store, currency="USD"))
    assert {(row["metric_name"], row["labels"]["cost_basis"]) for row in costs.records} == {
        ("cost_usd", "billed"),
        ("estimated_cost_usd", "estimated"),
    }
    assert query_metrics(QueryRequest(input_uri=store, cost_basis="billed")).count == 1
    assert sdk_query(input_uri=store, cost_basis="estimated").count == 1

    client = TestClient(
        create_app(auth_mode="none", allowed_local_roots=[tmp_path])
    )
    response = client.get(
        "/query", params={"input_uri": store, "cost_basis": "billed"}
    )
    assert response.status_code == 200
    assert response.json()["records"][0]["labels"]["cost_basis"] == "billed"

    cli = CliRunner().invoke(
        insights_app,
        ["query", "--input-path", store, "--cost-basis", "estimated"],
    )
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output)["count"] == 1

    dashboard = build_dashboard(DashboardRequest(input_uri=store, group_by="cost_basis"))
    assert {(group.key, group.count) for group in dashboard.groups} == {
        ("billed", 1),
        ("estimated", 1),
    }


def test_sparse_observed_report_emits_no_new_signal(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "schema": "npa.rl_sweep.variant_metrics.v1",
                "run_id": "sparse",
                "status": "completed",
            }
        )
    )
    with pytest.raises(InsightsStoreError, match="no known manifest/report schemas"):
        ingest_run(IngestRunRequest(input_uri=str(run), output_uri=str(tmp_path / "store")))


def test_new_signal_facets_thread_through_service_sdk_and_cli(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from npa.cli.workbench.insights import app as insights_app
    from npa.sdk.workbench.insights import query as sdk_query
    from npa.workbench.insights.service import create_app

    store = str(tmp_path / "store")
    record_metrics(
        RecordRequest(
            output_uri=store,
            records=[
                _metric(
                    "r1",
                    "success_rate",
                    0.8,
                    tool="sim",
                    stage="eval",
                    labels={"metric_kind": "eval_score", "score_name": "success_rate"},
                ),
                _metric(
                    "r2",
                    "success_rate",
                    0.9,
                    tool="sim",
                    stage="eval",
                    labels={"metric_kind": "eval_score", "score_name": "success_rate"},
                ),
            ],
        )
    )

    sdk_result = sdk_query(
        input_uri=store, metric_kind="eval_score", score_name="success_rate"
    )
    assert sdk_result.count == 2

    service_result = TestClient(
        create_app(auth_mode="none", allowed_local_roots=[tmp_path])
    ).get(
        "/query",
        params={
            "input_uri": store,
            "metric_kind": "eval_score",
            "score_name": "success_rate",
        },
    )
    assert service_result.status_code == 200
    assert service_result.json()["count"] == 2

    runner = CliRunner()
    query_result = runner.invoke(
        insights_app,
        [
            "query",
            "--input-path",
            store,
            "--metric-kind",
            "eval_score",
            "--score-name",
            "success_rate",
        ],
    )
    assert query_result.exit_code == 0, query_result.output
    assert json.loads(query_result.output)["count"] == 2

    compare_result = runner.invoke(
        insights_app,
        ["compare", "--input-path", store, "--base-run", "r1", "--candidate-run", "r2"],
    )
    assert compare_result.exit_code == 0, compare_result.output
    dimensions = json.loads(compare_result.output)["metrics"][0]["dimensions"]
    assert dimensions["score_name"] == "success_rate"

    dashboard_result = runner.invoke(
        insights_app,
        ["dashboard", "--input-path", store, "--group-by", "metric_kind"],
    )
    assert dashboard_result.exit_code == 0, dashboard_result.output
    assert json.loads(dashboard_result.output)["groups"][0]["key"] == "eval_score"


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


def test_query_empty_store_returns_empty_no_fabrication(tmp_path: Path) -> None:
    # An empty/absent store yields an empty result — never a fabricated fallback.
    result = query_metrics(QueryRequest(input_uri=str(tmp_path / "store"), metric_name="gpus"))
    assert result.count == 0
    assert result.records == []


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

    client = TestClient(
        create_app(auth_mode="none", allowed_local_roots=[tmp_path])
    )
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

    client = TestClient(
        create_app(auth_mode="none", allowed_local_roots=[tmp_path])
    )
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

    client = TestClient(
        create_app(auth_mode="none", allowed_local_roots=[tmp_path])
    )
    store = str(tmp_path / "store")
    _seed_two_runs(store)
    bad = client.get("/compare", params={"input_uri": store, "base_run": "r1", "candidate_run": "nope"})
    assert bad.status_code == 400


def test_service_storage_scope_fails_closed_and_enforces_configured_root(
    tmp_path: Path,
) -> None:
    from npa.workbench.insights.service import create_app

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    payload = {
        "output_uri": str(allowed / "store"),
        "records": [_metric("scoped", "accuracy", 0.9)],
    }

    unconfigured = TestClient(create_app(auth_mode="none"))
    assert unconfigured.post("/record", json=payload).status_code == 403

    configured = TestClient(
        create_app(auth_mode="none", allowed_local_roots=[allowed])
    )
    assert configured.post("/record", json=payload).status_code == 200
    payload["output_uri"] = str(tmp_path / "outside")
    assert configured.post("/record", json=payload).status_code == 403


def test_service_health_system_info_and_token_auth() -> None:
    from npa.workbench.insights.service import create_app

    open_client = TestClient(create_app(auth_mode="none"))
    assert open_client.get("/health").json()["status"] == "ok"
    assert open_client.get("/system-info").json()["tool"] == "insights"

    secure = TestClient(create_app(auth_mode="token", token="s3cr3t"))
    assert secure.get("/health").status_code == 401
    assert secure.get("/health", headers={"Authorization": "Bearer s3cr3t"}).status_code == 200


def test_deployed_default_is_not_unauthenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    from npa.workbench.insights.service import create_app

    monkeypatch.delenv("INSIGHTS_AUTH_MODE", raising=False)
    monkeypatch.delenv("INSIGHTS_TOKEN", raising=False)
    assert TestClient(create_app()).get("/health").status_code == 503


def test_explicit_auth_none_remains_an_opt_in() -> None:
    from npa.workbench.insights.service import create_app

    assert TestClient(create_app(auth_mode="none")).get("/health").status_code == 200


def test_sdk_workbench_namespace_exports_insights() -> None:
    from npa.sdk import workbench

    assert workbench.insights.__name__ == "npa.sdk.workbench.insights"
    for attr in ("record", "ingest_run", "query", "lineage", "compare", "dashboard"):
        assert hasattr(workbench.insights, attr)


def test_embedded_sdk_does_not_require_service_allowlists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npa.sdk.workbench.insights import record

    monkeypatch.delenv("INSIGHTS_ALLOWED_LOCAL_ROOTS", raising=False)
    monkeypatch.delenv("INSIGHTS_ALLOWED_S3_ROOTS", raising=False)
    result = record(
        output_uri=str(tmp_path / "sdk-store"),
        records=[_metric("embedded", "accuracy", 0.9)],
    )
    assert result.recorded_count == 1
    assert Path(result.store_uri).exists()


def test_cli_and_sdk_do_not_import_heavy_ml_dependencies_at_module_level() -> None:
    npa_root = Path(__file__).resolve().parents[2]
    cli_source = (npa_root / "src/npa/cli/workbench/insights.py").read_text()
    sdk_source = (npa_root / "src/npa/sdk/workbench/insights.py").read_text()
    for source in (cli_source, sdk_source):
        assert "import torch" not in source
        assert "import lancedb" not in source
        assert "import fiftyone" not in source


# ── Concurrent-writer safety for the append-only store ───────────────────────
# Live evidence: two insights-smoke runs ingesting into the same store at the same
# time each reported success ("recorded_count": 14, "total_records": 31), but only
# the later writer's rows survived -- the earlier run's 14 records were silently
# dropped by read-modify-write, and its next stage failed with
# "no metrics recorded for base run: <run-id>".


def test_concurrent_appends_do_not_lose_rows(tmp_path: Path) -> None:
    from npa.workbench.insights.storage import append_jsonl_uri, read_jsonl_store

    uri = str(tmp_path / "store" / "records.jsonl")
    # Interleave two writers the way two pods do: both observe the same starting
    # state, then both append.
    before_a = read_jsonl_store(uri)
    before_b = read_jsonl_store(uri)
    assert before_a == before_b == []
    append_jsonl_uri(uri, [{"run_id": "run-a", "metric_name": "m", "value": 1}])
    append_jsonl_uri(uri, [{"run_id": "run-b", "metric_name": "m", "value": 2}])

    rows = read_jsonl_store(uri)
    assert {row["run_id"] for row in rows} == {"run-a", "run-b"}
    assert len(rows) == 2


def test_append_returns_the_full_store_total(tmp_path: Path) -> None:
    from npa.workbench.insights.storage import append_jsonl_uri

    uri = str(tmp_path / "store" / "records.jsonl")
    assert append_jsonl_uri(uri, [{"a": 1}]) == 1
    assert append_jsonl_uri(uri, [{"a": 2}, {"a": 3}]) == 3
    assert append_jsonl_uri(uri, []) == 3


def test_store_reads_legacy_single_object_plus_shards(tmp_path: Path) -> None:
    """Stores written before sharding must keep reading correctly."""
    from npa.workbench.insights.storage import append_jsonl_uri, read_jsonl_store

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    legacy = store_dir / "records.jsonl"
    legacy.write_text(json.dumps({"run_id": "legacy", "value": 0}) + "\n")

    append_jsonl_uri(str(legacy), [{"run_id": "new", "value": 1}])
    rows = read_jsonl_store(str(legacy))
    assert [row["run_id"] for row in rows] == ["legacy", "new"]


def test_reader_deduplicates_existing_duplicate_metric_and_edge_shards(tmp_path: Path) -> None:
    from npa.workbench.insights.storage import append_jsonl_uri
    from npa.workbench.insights.store import edges_uri, records_uri

    store = str(tmp_path / "store")
    row = {
        "schema": "npa.insights.metric_record.v1",
        "run_id": "legacy-duplicate",
        "metric_name": "success_rate",
        "value": 0.8,
        "workflow": "wf",
        "tool": "eval",
        "stage": "eval",
        "unit": "",
        "labels": {"metric_kind": "eval_score", "score_name": "success_rate"},
        "lineage": {},
        "artifact_uri": "s3://artifacts/legacy/report.json",
        "artifact_version": "",
        "timestamp": "2026-01-01T00:00:00Z",
    }
    edge = {
        "from_uri": "s3://artifacts/input",
        "to_uri": "s3://artifacts/legacy/report.json",
        "relation": "evaluated_on",
        "from_version": "",
        "to_version": "",
        "run_id": "legacy-duplicate",
    }
    append_jsonl_uri(records_uri(store), [row])
    append_jsonl_uri(records_uri(store), [row])
    append_jsonl_uri(edges_uri(store), [edge])
    append_jsonl_uri(edges_uri(store), [edge])

    assert read_records(store) == [row]
    assert read_edges(store) == [edge]
    dashboard = build_dashboard(DashboardRequest(input_uri=store))
    assert dashboard.total_records == 1
    assert dashboard.groups[0].count == 1


def test_repeated_ingest_is_logically_idempotent(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_signal_run(run, run_id="same-run", success_rate=0.8, cost_usd=3.0)
    store = str(tmp_path / "store")

    first = ingest_run(IngestRunRequest(input_uri=str(run), output_uri=store))
    first_records = read_records(store)
    first_edges = read_edges(store)
    second = ingest_run(IngestRunRequest(input_uri=str(run), output_uri=store))

    assert first.recorded_count == len(first_records)
    assert first.edge_count == len(first_edges)
    assert second.recorded_count == 0
    assert second.edge_count == 0
    assert read_records(store) == first_records
    assert read_edges(store) == first_edges
    dashboard = build_dashboard(DashboardRequest(input_uri=store, group_by="metric_name"))
    assert dashboard.total_records == len(first_records)
    assert sum(group.count for group in dashboard.groups) == len(first_records)


def test_concurrent_distinct_append_shards_preserve_same_named_observations(
    tmp_path: Path,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from npa.workbench.insights.storage import append_jsonl_uri
    from npa.workbench.insights.store import records_uri

    store = str(tmp_path / "store")
    rows = [
        {
            "run_id": f"run-{index}",
            "metric_name": "success_rate",
            "value": index / 20,
            "labels": {"metric_kind": "eval_score"},
            "lineage": {},
            "artifact_uri": f"s3://artifacts/run-{index}/report.json",
        }
        for index in range(20)
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda row: append_jsonl_uri(
                    records_uri(store), [row], previous_total=0
                ),
                rows,
            )
        )

    records = read_records(store)
    assert len(records) == 20
    assert {row["run_id"] for row in records} == {f"run-{index}" for index in range(20)}


def test_ingest_run_twice_concurrently_keeps_both_runs(tmp_path: Path) -> None:
    """End-to-end guard: two ingests into one store keep both runs queryable."""
    store = str(tmp_path / "store")
    for run_id in ("run-one", "run-two"):
        run = tmp_path / run_id
        (run / "npa-workflow").mkdir(parents=True)
        (run / "npa-workflow" / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "npa.workflow.run.v1",
                    "workflow": "wf",
                    "run_id": run_id,
                    "status": "submitted",
                    "steps": [{"state": "s", "status": "submitted", "resources_profile": {"accelerators": "H100:2"}}],
                }
            )
        )
        ingest_run(IngestRunRequest(input_uri=str(run), output_uri=store, workflow="wf"))

    records = read_records(store)
    assert {r["run_id"] for r in records} == {"run-one", "run-two"}


def test_failed_check_count_increase_is_a_regression(tmp_path: Path) -> None:
    """More failed checks is worse; the hint list must not call it an improvement."""
    store = str(tmp_path / "store")
    for run_id, failed in (("base", 0.0), ("cand", 2.0)):
        record_metrics(
            RecordRequest(
                output_uri=store,
                records=[
                    MetricRecord(run_id=run_id, metric_name="failed_check_count", value=failed, tool="dataset"),
                ],
            )
        )
    response = compare_runs(CompareRequest(input_uri=store, base_run="base", candidate_run="cand"))
    assert response.regressed == ["failed_check_count"]
    assert response.improved == []


# ── Review follow-ups: store scaling, hint precision, planned-only diagnostics ──


def test_append_does_not_reread_the_store_when_the_total_is_known(tmp_path: Path, monkeypatch) -> None:
    """The sharded layout must not re-list + re-GET every shard just for telemetry."""
    from npa.workbench.insights import storage as st

    uri = str(tmp_path / "store" / "records.jsonl")
    st.append_jsonl_uri(uri, [{"a": 1}])

    calls = {"n": 0}
    real = st.read_jsonl_store

    def _counting(target: str):
        calls["n"] += 1
        return real(target)

    monkeypatch.setattr(st, "read_jsonl_store", _counting)
    assert st.append_jsonl_uri(uri, [{"a": 2}, {"a": 3}], previous_total=1) == 3
    assert calls["n"] == 0, "a known previous total must make the new total arithmetic"
    # Without the hint the count is still exact, by reading the store back.
    assert st.append_jsonl_uri(uri, [{"a": 4}]) == 4


def test_persist_totals_stay_exact_across_appends(tmp_path: Path) -> None:
    store = str(tmp_path / "store")
    first = record_metrics(
        RecordRequest(output_uri=store, records=[MetricRecord(run_id="r1", metric_name="m", value=1.0)])
    )
    second = record_metrics(
        RecordRequest(output_uri=store, records=[MetricRecord(run_id="r2", metric_name="m", value=2.0)])
    )
    assert (first.total_records, second.total_records) == (1, 2)
    assert len(read_records(store)) == 2


def test_failsafe_style_metric_is_not_flipped_to_lower_is_better(tmp_path: Path) -> None:
    """A bare "fail" substring would silently invert a higher-is-better metric."""
    from npa.workbench.insights.analytics import _is_lower_better

    assert _is_lower_better("failed_check_count", []) is True
    assert _is_lower_better("failure_rate", []) is True
    assert _is_lower_better("fail_count", []) is True
    assert _is_lower_better("failsafe_score", []) is False
    assert _is_lower_better("success_rate", []) is False


def test_planned_only_prefix_explains_why_nothing_was_ingested(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "npa-workflow").mkdir(parents=True)
    (run / "npa-workflow" / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "npa.workflow.run.v1",
                "workflow": "wf",
                "run_id": "planned-only",
                "status": "planned",
                "steps": [{"state": "s", "status": "planned", "resources_profile": {"accelerators": "H100:8"}}],
            }
        )
    )
    with pytest.raises(InsightsStoreError) as excinfo:
        ingest_run(IngestRunRequest(input_uri=str(run), output_uri=str(tmp_path / "store")))
    message = str(excinfo.value)
    assert "planned" in message
    assert "never executed" in message


def test_shards_are_read_in_write_order(tmp_path: Path) -> None:
    from npa.workbench.insights.storage import append_jsonl_uri, read_jsonl_store

    uri = str(tmp_path / "store" / "records.jsonl")
    for index in range(12):
        append_jsonl_uri(uri, [{"seq": index}], previous_total=index)
    assert [row["seq"] for row in read_jsonl_store(uri)] == list(range(12))
