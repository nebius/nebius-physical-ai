"""Append-only metric + lineage store and non-invasive run ingestion.

The store is an append-only index on S3 (JSONL under a configurable prefix).
``record_metrics`` writes explicit emissions; ``ingest_run`` scans an existing
run prefix for manifest/report schemas already produced by other tools and
extracts their metrics + provenance without modifying the emitting tools.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from npa.workbench.storage_scope import StorageAuthorizationError

from .integrations import index_metrics_in_lancedb
from .schemas import (
    ACCELERATORS_LABEL,
    COST_BASIS_LABEL,
    CURRENCY_LABEL,
    EDGES_OBJECT,
    GPU_METRIC_NAME,
    METRIC_RECORD_SCHEMA,
    METRIC_KIND_LABEL,
    RECORDS_OBJECT,
    SCORE_NAME_LABEL,
    STEP_LABEL,
    IngestedArtifact,
    IngestRunRequest,
    IngestRunResponse,
    LineageEdge,
    LineageRef,
    MetricRecord,
    RecordRequest,
    RecordResponse,
)
from .storage import (
    append_jsonl_uri,
    list_json_uris,
    read_json_uri,
    read_jsonl_store,
    uri_join,
)

DATASET_MANIFEST_SCHEMA = "npa.dataset.manifest.v1"
DATASET_VALIDATION_SCHEMA = "npa.dataset.validation_report.v1"
SCENARIO_ADVERSARIAL_SCHEMA = "npa.scenario_gen.adversarial_set.v1"
# The durable npa.workflow run manifest tags its schema under ``schema_version``
# (not ``schema``); it carries per-step resource profiles we mine for GPU counts.
WORKFLOW_RUN_SCHEMA = "npa.workflow.run.v1"

# Known report shapes already emitted by NPA tools. Structural profiles below
# cover the older schema-less detection/VLM artifacts without treating arbitrary
# JSON as metrics.
REPORT_PROFILES: dict[str, tuple[str, str]] = {
    "npa.rl_sweep.variant_metrics.v1": ("isaac_lab", "sweep-train"),
    "npa.sim2real.heldout_eval.v1": ("sim2real", "eval"),
    "npa.sim2real.vlm_eval.v1": ("vlm_eval", "eval"),  # archived artifacts
    "npa.sim2real.vlm_eval.v2": ("vlm_eval", "eval"),
    "npa.rl.eval_report.v1": ("rl", "eval"),
    "npa.workbench.vlm_eval.report.v1": ("vlm_eval", "eval"),
    "npa.workbench.vlm_eval.benchmark.v1": ("vlm_eval", "benchmark"),
}
FORMAT_PROFILES: dict[str, tuple[str, str, str]] = {
    "npa_sonic_eval_result_v1": ("npa_sonic_eval_result_v1", "sonic", "eval"),
}


class InsightsStoreError(RuntimeError):
    """Raised when recording or ingesting into the store fails."""


def records_uri(store_uri: str) -> str:
    return uri_join(store_uri, RECORDS_OBJECT)


def edges_uri(store_uri: str) -> str:
    return uri_join(store_uri, EDGES_OBJECT)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_records(store_uri: str) -> list[dict[str, Any]]:
    return _dedupe_rows(read_jsonl_store(records_uri(store_uri)), _record_identity)


def read_edges(store_uri: str) -> list[dict[str, Any]]:
    return _dedupe_rows(read_jsonl_store(edges_uri(store_uri)), _edge_identity)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _record_identity(row: dict[str, Any]) -> tuple[Any, ...] | None:
    """Stable logical identity for one metric observation.

    Ingested rows use their real source artifact URI plus run/metric/dimensions;
    regenerated timestamps and values do not create a second logical observation.
    Explicit emissions without source provenance retain timestamp+value so two
    measurements sharing a metric name are never collapsed.
    """
    artifact_uri = str(row.get("artifact_uri") or "")
    base = (
        str(row.get("run_id") or ""),
        artifact_uri,
        str(row.get("artifact_version") or ""),
        str(row.get("metric_name") or ""),
        str(row.get("workflow") or ""),
        str(row.get("tool") or ""),
        str(row.get("stage") or ""),
        str(row.get("unit") or ""),
        _canonical_json(row.get("labels") or {}),
        _canonical_json(row.get("lineage") or {}),
    )
    if artifact_uri:
        return ("provenance", *base)
    timestamp = str(row.get("timestamp") or "")
    if timestamp:
        return ("emission", *base, timestamp, row.get("value"))
    return None


def _edge_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("from_uri") or ""),
        str(row.get("from_version") or ""),
        str(row.get("to_uri") or ""),
        str(row.get("to_version") or ""),
        str(row.get("relation") or ""),
        str(row.get("run_id") or ""),
    )


def _dedupe_rows(rows: list[dict[str, Any]], identity_fn: Any) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        identity = identity_fn(row)
        if identity is not None and identity in seen:
            continue
        if identity is not None:
            seen.add(identity)
        deduped.append(row)
    return deduped


def _persist(
    store_uri: str,
    records: list[MetricRecord],
    edges: list[LineageEdge],
    *,
    lancedb_endpoint: str = "",
) -> tuple[int, int, int, int]:
    """Append unseen records/edges and return added counts plus new totals.

    The totals are read **once** before writing and then advanced arithmetically.
    Reading the whole store back after each write would re-list and re-GET every
    shard purely to produce a telemetry number, which is the one cost the sharded
    layout would otherwise add to every ingest.
    """
    rec_rows: list[dict[str, Any]] = []
    for record in records:
        row = record.model_dump(mode="json")
        row["schema"] = METRIC_RECORD_SCHEMA
        if not row.get("timestamp"):
            row["timestamp"] = _now()
        rec_rows.append(row)
    edge_rows = [edge.model_dump(mode="json") for edge in edges]

    existing_records = read_records(store_uri)
    existing_edges = read_edges(store_uri)
    existing_record_ids = {
        identity
        for row in existing_records
        if (identity := _record_identity(row)) is not None
    }
    existing_edge_ids = {_edge_identity(row) for row in existing_edges}
    filtered_records: list[dict[str, Any]] = []
    for row in rec_rows:
        identity = _record_identity(row)
        if identity is not None and identity in existing_record_ids:
            continue
        if identity is not None:
            existing_record_ids.add(identity)
        filtered_records.append(row)
    filtered_edges: list[dict[str, Any]] = []
    for row in edge_rows:
        identity = _edge_identity(row)
        if identity in existing_edge_ids:
            continue
        existing_edge_ids.add(identity)
        filtered_edges.append(row)

    previous_records = len(existing_records)
    previous_edges = len(existing_edges)
    total_records = append_jsonl_uri(
        records_uri(store_uri), filtered_records, previous_total=previous_records
    )
    total_edges = append_jsonl_uri(
        edges_uri(store_uri), filtered_edges, previous_total=previous_edges
    )

    if filtered_records:
        index_metrics_in_lancedb(
            filtered_records,
            lancedb_endpoint=lancedb_endpoint,
            table="insights_metrics",
        )
    return len(filtered_records), len(filtered_edges), total_records, total_edges


def _load_record_payload(payload: Any) -> tuple[list[MetricRecord], list[LineageEdge]]:
    """Parse a records/edges JSON document into validated models."""
    if isinstance(payload, list):
        rows, edge_rows = payload, []
    elif isinstance(payload, dict):
        rows = payload.get("records", [])
        edge_rows = payload.get("edges", [])
    else:
        raise InsightsStoreError("record input must be a JSON object or list of records")
    if not isinstance(rows, list) or not isinstance(edge_rows, list):
        raise InsightsStoreError("record input 'records'/'edges' must be lists")
    records = [MetricRecord.model_validate(row) for row in rows]
    edges = [LineageEdge.model_validate(row) for row in edge_rows]
    return records, edges


def record_metrics(request: RecordRequest) -> RecordResponse:
    """Record explicit metric emissions + lineage edges into the store."""
    records = list(request.records)
    edges = list(request.edges)
    if request.input_uri.strip():
        try:
            payload = read_json_uri(request.input_uri)
        except StorageAuthorizationError:
            raise
        except FileNotFoundError as exc:
            raise InsightsStoreError(f"record input not found: {request.input_uri}") from exc
        except Exception as exc:  # noqa: BLE001
            raise InsightsStoreError(f"cannot read record input {request.input_uri}: {exc}") from exc
        loaded_records, loaded_edges = _load_record_payload(payload)
        records.extend(loaded_records)
        edges.extend(loaded_edges)

    if not records and not edges:
        raise InsightsStoreError("record requires at least one metric record or lineage edge")

    if request.workflow_run:
        for record in records:
            if not record.run_id:
                record.run_id = request.workflow_run

    recorded_count, edge_count, total_records, total_edges = _persist(
        request.output_uri,
        records,
        edges,
        lancedb_endpoint=request.lancedb_endpoint,
    )
    return RecordResponse(
        store_uri=request.output_uri,
        records_uri=records_uri(request.output_uri),
        edges_uri=edges_uri(request.output_uri),
        recorded_count=recorded_count,
        edge_count=edge_count,
        total_records=total_records,
        total_edges=total_edges,
    )


def ingest_run(request: IngestRunRequest) -> IngestRunResponse:
    """Scan an S3 run prefix for known manifests and extract metrics + lineage."""
    uris = list_json_uris(request.input_uri)
    if not uris:
        raise InsightsStoreError(f"no JSON artifacts found under run prefix: {request.input_uri}")

    all_records: list[MetricRecord] = []
    all_edges: list[LineageEdge] = []
    ingested: list[IngestedArtifact] = []
    scanned = 0

    skipped_planned = 0

    for uri in uris:
        scanned += 1
        try:
            payload = read_json_uri(uri)
        except StorageAuthorizationError:
            raise
        except Exception:  # noqa: BLE001 - skip unreadable/non-object artifacts.
            continue
        if not isinstance(payload, dict):
            continue
        if (
            str(payload.get("schema_version", "")) == WORKFLOW_RUN_SCHEMA
            and str(payload.get("status", "")).strip().lower() == "planned"
        ):
            skipped_planned += 1
        records, edges, schema_id = _extract(
            payload,
            source_uri=uri,
            workflow=request.workflow,
            workflow_run=request.workflow_run,
        )
        if schema_id is None:
            continue
        all_records.extend(records)
        all_edges.extend(edges)
        ingested.append(
            IngestedArtifact(uri=uri, schema_id=schema_id, records=len(records), edges=len(edges))
        )

    if not ingested:
        # Say *why* nothing was ingested. A planned-only run prefix is a legitimate
        # thing to point at (`run-spec --persist-state` without `--execute` writes
        # one), and "no known schemas found" would send the operator looking for a
        # missing artifact instead of explaining that the run never executed.
        if skipped_planned:
            raise InsightsStoreError(
                f"nothing to ingest under run prefix: {request.input_uri} — found "
                f"{skipped_planned} npa.workflow.run.v1 manifest(s) with status 'planned'. "
                "A planned run never executed, so it reports no metrics (including no GPU "
                "count). Execute or submit the run first."
            )
        raise InsightsStoreError(
            f"no known manifest/report schemas found under run prefix: {request.input_uri}"
        )

    recorded_count, edge_count, total_records, total_edges = _persist(
        request.output_uri,
        all_records,
        all_edges,
        lancedb_endpoint=request.lancedb_endpoint,
    )
    return IngestRunResponse(
        store_uri=request.output_uri,
        records_uri=records_uri(request.output_uri),
        edges_uri=edges_uri(request.output_uri),
        scanned=scanned,
        ingested=ingested,
        recorded_count=recorded_count,
        edge_count=edge_count,
        total_records=total_records,
        total_edges=total_edges,
    )


def _metric(
    *,
    run_id: str,
    workflow: str,
    tool: str,
    stage: str,
    name: str,
    value: float,
    unit: str = "",
    labels: dict[str, str] | None = None,
    lineage: LineageRef | None = None,
    artifact_uri: str = "",
    artifact_version: str = "",
) -> MetricRecord:
    return MetricRecord(
        run_id=run_id or "unknown",
        metric_name=name,
        value=float(value),
        workflow=workflow,
        tool=tool,
        stage=stage,
        unit=unit,
        labels=labels or {},
        lineage=lineage or LineageRef(),
        artifact_uri=artifact_uri,
        artifact_version=artifact_version,
    )


def _extract(
    payload: dict[str, Any],
    *,
    source_uri: str,
    workflow: str,
    workflow_run: str,
) -> tuple[list[MetricRecord], list[LineageEdge], str | None]:
    """Route a discovered artifact to the matching extractor."""
    schema_id = str(payload.get("schema", ""))
    schema_version = str(payload.get("schema_version", ""))
    if schema_version == WORKFLOW_RUN_SCHEMA:
        records, edges = _extract_run_manifest(payload, source_uri, workflow, workflow_run)
        if not records and not edges:
            return [], [], None
        return records, edges, WORKFLOW_RUN_SCHEMA
    if schema_id == DATASET_MANIFEST_SCHEMA:
        return (*_extract_dataset_manifest(payload, source_uri, workflow, workflow_run), schema_id)
    if schema_id == DATASET_VALIDATION_SCHEMA:
        return (*_extract_validation_report(payload, source_uri, workflow, workflow_run), schema_id)
    if schema_id == SCENARIO_ADVERSARIAL_SCHEMA:
        # The set manifest carries the ``scenarios`` list; per-scenario config
        # files reuse the same schema tag but describe a single scenario. Only
        # ingest the aggregate manifest so per-scenario configs are not counted
        # as zero-metric records.
        if not isinstance(payload.get("scenarios"), list):
            return [], [], None
        return (*_extract_adversarial_set(payload, source_uri, workflow, workflow_run), schema_id)
    if "decision" in payload and (not schema_id or "decision" in schema_id):
        decision_schema = schema_id or "decision"
        return (*_extract_decision(payload, source_uri, workflow, workflow_run), decision_schema)
    profile = _report_profile(payload, source_uri)
    if profile is not None:
        report_schema, tool, stage = profile
        records, edges = _extract_observed_report(
            payload,
            source_uri,
            workflow,
            workflow_run,
            tool=tool,
            stage=stage,
            schema_id=report_schema,
        )
        if records or edges:
            return records, edges, report_schema
    return [], [], None


def _run_id(payload: dict[str, Any], workflow_run: str) -> str:
    lineage = payload.get("lineage") or {}
    return (
        workflow_run
        or str(payload.get("run_id") or "")
        or str(payload.get("eval_run_id") or "")
        or str(lineage.get("workflow_run") or "")
        or "unknown"
    )


def _report_profile(
    payload: dict[str, Any], source_uri: str
) -> tuple[str, str, str] | None:
    schema_id = str(payload.get("schema") or "")
    if schema_id in REPORT_PROFILES:
        tool, stage = REPORT_PROFILES[schema_id]
        return schema_id, tool, stage
    format_id = str(payload.get("format") or "")
    if format_id in FORMAT_PROFILES:
        return FORMAT_PROFILES[format_id]
    if (
        isinstance(payload.get("epochs"), list)
        and payload.get("run_id")
        and payload.get("manifest_sha256")
    ):
        return "detection_training.metrics", "detection_training", "train"
    if all(key in payload for key in ("mAP", "mAP_50", "mAP_75", "eval_run_id")):
        return "detection_training.eval", "detection_training", "eval"
    if all(key in payload for key in ("score", "success_threshold", "passed")):
        return "vlm_eval.result", "vlm_eval", "eval"
    if schema_id and ("billing" in schema_id or "resource_usage" in schema_id):
        return schema_id, str(payload.get("tool") or "workflow"), str(
            payload.get("stage") or "run"
        )
    # Older VLM aggregate reports have no schema but do carry this exact output
    # contract. The filename alone is not enough to classify arbitrary JSON.
    if all(key in payload for key in ("total_rollouts", "success_rate", "mean_score")):
        return "vlm_eval.loop_report", "vlm_eval", "eval"
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_curve_metric(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in {"loss", "reward", "success_rate", "train_loss", "eval_loss"}
        or lowered.endswith(("_loss", "_reward", "_success_rate"))
    )


def _is_duration(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith("_ms") or lowered in {
        "duration",
        "duration_s",
        "duration_seconds",
        "elapsed_s",
        "elapsed_seconds",
        "wall_clock_s",
        "wall_clock_seconds",
        "latency_s",
    }


def _is_throughput(name: str) -> bool:
    lowered = name.lower()
    return "throughput" in lowered or lowered.endswith(
        ("_per_second", "_per_sec", "_per_s")
    )


def _is_cost(name: str) -> bool:
    lowered = name.lower()
    return lowered in {
        "billed_cost_usd",
        "cost_usd",
        "estimated_cost_usd",
        "total_cost_usd",
    }


def _cost_basis(name: str, *, container: str, schema_id: str) -> str:
    """Classify cost provenance without promoting estimates to billed values."""
    lowered = name.lower()
    if "estimated" in lowered:
        return "estimated"
    if container == "billing" or "billing" in schema_id.lower():
        return "billed"
    return "estimated"


_EVAL_SCORE_NAMES = frozenset(
    {
        "accuracy",
        "f1",
        "f1_score",
        "map",
        "map_50",
        "map_75",
        "mean_reward",
        "mean_score",
        "precision",
        "recall",
        "reward",
        "score",
        "success_rate",
    }
)
_COUNTER_NAMES = frozenset(
    {
        "epoch",
        "epochs",
        "global_step",
        "iteration",
        "num_samples",
        "sample_count",
        "samples",
        "step",
        "steps",
        "token_count",
        "tokens",
        "train_steps",
    }
)


def _is_counter(name: str) -> bool:
    lowered = name.lower()
    return lowered in _COUNTER_NAMES or lowered.endswith(
        ("_count", "_samples", "_steps", "_tokens")
    )


def _is_eval_score(name: str, *, container: str) -> bool:
    lowered = name.lower()
    return (
        lowered in _EVAL_SCORE_NAMES
        or lowered.endswith(
            ("_accuracy", "_f1", "_precision", "_recall", "_return", "_reward", "_score", "_success_rate")
        )
    )


def _metric_unit(name: str, kind: str) -> str:
    if kind == "duration":
        return "milliseconds" if name.lower().endswith("_ms") else "seconds"
    if kind == "cost":
        return "USD"
    if kind == "throughput":
        lowered = name.lower()
        prefix = lowered.split("_per_", 1)[0]
        return f"{prefix}/s" if prefix != lowered else "units/s"
    return ""


def _first_uri(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _observed_lineage(payload: dict[str, Any]) -> tuple[list[str], str]:
    lineage = payload.get("lineage") if isinstance(payload.get("lineage"), dict) else {}
    input_uris = [str(uri) for uri in lineage.get("input_uris", []) if uri]
    for key in ("input_uri", "input_path", "dataset_uri", "data_path", "lance_uri"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip() and value.strip() not in input_uris:
            input_uris.append(value.strip())
    checkpoint_uri = _first_uri(
        payload,
        ("checkpoint_uri", "policy_checkpoint", "checkpoint_path", "model_uri"),
    ) or str(lineage.get("checkpoint_uri") or "").strip()
    policy = payload.get("policy")
    if not checkpoint_uri and isinstance(policy, dict):
        checkpoint_uri = _first_uri(policy, ("checkpoint_uri", "model_uri", "onnx_uri"))
    return input_uris, checkpoint_uri


def _extract_observed_report(
    payload: dict[str, Any],
    source_uri: str,
    workflow: str,
    workflow_run: str,
    *,
    tool: str,
    stage: str,
    schema_id: str,
) -> tuple[list[MetricRecord], list[LineageEdge]]:
    """Extract only numeric signals and URIs physically present in a known report."""
    run_id = _run_id(payload, workflow_run)
    resolved_tool = str(payload.get("tool") or tool)
    resolved_stage = str(payload.get("stage") or stage)
    input_uris, checkpoint_uri = _observed_lineage(payload)
    ref = LineageRef(input_uris=input_uris, checkpoint_uri=checkpoint_uri)
    records: list[MetricRecord] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

    def emit(name: str, value: Any, kind: str, labels: dict[str, str] | None = None) -> None:
        if not _is_number(value):
            return
        resolved_labels = {METRIC_KIND_LABEL: kind, **(labels or {})}
        identity = (name, tuple(sorted(resolved_labels.items())))
        if identity in seen:
            return
        seen.add(identity)
        records.append(
            _metric(
                run_id=run_id,
                workflow=workflow,
                tool=resolved_tool,
                stage=resolved_stage,
                name=name,
                value=float(value),
                unit=_metric_unit(name, kind),
                labels=resolved_labels,
                lineage=ref,
                artifact_uri=source_uri,
            )
        )

    for curve_name in ("epochs", "history", "training_curve", "eval_curve"):
        curve = payload.get(curve_name)
        if not isinstance(curve, list):
            continue
        curve_kind = "eval_curve" if "eval" in curve_name else "training_curve"
        for index, point in enumerate(curve, start=1):
            if not isinstance(point, dict):
                continue
            step = point.get("step", point.get("epoch", point.get("iteration", index)))
            for name, value in point.items():
                if not _is_number(value):
                    continue
                if _is_duration(name):
                    emit(name, value, "duration", {STEP_LABEL: str(step)})
                elif _is_throughput(name):
                    emit(name, value, "throughput", {STEP_LABEL: str(step)})
                elif _is_curve_metric(name):
                    emit(name, value, curve_kind, {STEP_LABEL: str(step)})

    billing_context = (
        "billing" in schema_id
        or "resource_usage" in schema_id
        or isinstance(payload.get("billing"), dict)
        or isinstance(payload.get("resource_usage"), dict)
    )
    # Authoritative nested containers win over root placeholders. ``emit`` keeps
    # the first metric/dimension identity, so this order is the precedence rule.
    containers: list[tuple[str, dict[str, Any]]] = []
    for key in ("metrics", "success_summary", "billing", "resource_usage"):
        value = payload.get(key)
        if isinstance(value, dict):
            containers.append((key, value))
    containers.append(("root", payload))

    stage_kind = f"{stage} {resolved_stage}".lower()
    eval_stage = "eval" in stage_kind or "benchmark" in stage_kind
    train_stage = "train" in stage_kind
    for container_name, values in containers:
        for name, value in values.items():
            if not _is_number(value):
                continue
            if _is_duration(name):
                emit(name, value, "duration")
            elif _is_throughput(name):
                emit(name, value, "throughput")
            elif _is_cost(name) and billing_context:
                emit(
                    name,
                    value,
                    "cost",
                    {
                        CURRENCY_LABEL: "USD",
                        COST_BASIS_LABEL: _cost_basis(
                            name, container=container_name, schema_id=schema_id
                        ),
                    },
                )
            elif eval_stage and _is_eval_score(name, container=container_name):
                emit(name, value, "eval_score", {SCORE_NAME_LABEL: name})
            elif eval_stage and _is_counter(name):
                emit(name, value, "counter")
            elif train_stage and _is_curve_metric(name):
                emit(name, value, "training_metric")

    edges: list[LineageEdge] = []
    if eval_stage:
        if checkpoint_uri:
            edges.append(
                LineageEdge(
                    from_uri=checkpoint_uri,
                    to_uri=source_uri,
                    relation="evaluated_on",
                    run_id=run_id,
                )
            )
        edges.extend(
            LineageEdge(
                from_uri=input_uri,
                to_uri=source_uri,
                relation="evaluated_on",
                run_id=run_id,
            )
            for input_uri in input_uris
            if input_uri != checkpoint_uri
        )
    elif train_stage and input_uris:
        target_uri = checkpoint_uri or source_uri
        edges.extend(
            LineageEdge(
                from_uri=input_uri,
                to_uri=target_uri,
                relation="produced_from",
                run_id=run_id,
            )
            for input_uri in input_uris
        )
    return records, edges


def _extract_dataset_manifest(
    payload: dict[str, Any], source_uri: str, workflow: str, workflow_run: str
) -> tuple[list[MetricRecord], list[LineageEdge]]:
    stats = payload.get("quality_stats") or {}
    lineage_meta = payload.get("lineage") or {}
    dataset_id = str(payload.get("dataset_id", ""))
    version = str(payload.get("version", ""))
    dataset_version = f"{dataset_id}@{version}" if dataset_id else version
    record_count = int(payload.get("record_count", stats.get("record_count", 0)) or 0)
    corrupt = int(stats.get("corrupt_count", 0) or 0)
    stage = "curate" if payload.get("parent_version") else "ingest"
    run_id = _run_id(payload, workflow_run)
    input_uris = [str(u) for u in lineage_meta.get("input_uris", []) if u]
    ref = LineageRef(
        input_uris=input_uris,
        dataset_version=dataset_version,
        parent_uri=str(payload.get("parent_dataset_id", "") or lineage_meta.get("parent_dataset_id", "")),
        parent_version=str(payload.get("parent_version", "") or lineage_meta.get("parent_version", "")),
    )
    metrics = [
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage=stage, name="record_count", value=record_count, unit="records", lineage=ref, artifact_uri=source_uri, artifact_version=dataset_version),
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage=stage, name="mean_completeness", value=float(stats.get("mean_completeness", 0.0) or 0.0), lineage=ref, artifact_uri=source_uri, artifact_version=dataset_version),
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage=stage, name="corrupt_count", value=corrupt, unit="records", lineage=ref, artifact_uri=source_uri, artifact_version=dataset_version),
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage=stage, name="corruption_rate", value=round(corrupt / record_count, 4) if record_count else 0.0, lineage=ref, artifact_uri=source_uri, artifact_version=dataset_version),
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage=stage, name="modality_count", value=len(stats.get("modalities", []) or []), unit="modalities", lineage=ref, artifact_uri=source_uri, artifact_version=dataset_version),
    ]
    edges = [
        LineageEdge(from_uri=input_uri, to_uri=source_uri, to_version=dataset_version, relation="produced_from", run_id=run_id)
        for input_uri in input_uris
    ]
    return metrics, edges


def _extract_validation_report(
    payload: dict[str, Any], source_uri: str, workflow: str, workflow_run: str
) -> tuple[list[MetricRecord], list[LineageEdge]]:
    stats = payload.get("quality_stats") or {}
    source_manifest = str(payload.get("source_manifest_uri", ""))
    run_id = _run_id(payload, workflow_run)
    ref = LineageRef(input_uris=[source_manifest] if source_manifest else [])
    metrics = [
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage="validate", name="validation_passed", value=1.0 if payload.get("passed") else 0.0, lineage=ref, artifact_uri=source_uri),
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage="validate", name="corruption_rate", value=float(payload.get("corruption_rate", 0.0) or 0.0), lineage=ref, artifact_uri=source_uri),
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage="validate", name="record_count", value=int(payload.get("record_count", stats.get("record_count", 0)) or 0), unit="records", lineage=ref, artifact_uri=source_uri),
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage="validate", name="failed_check_count", value=len(payload.get("failed_checks", []) or []), lineage=ref, artifact_uri=source_uri),
    ]
    edges: list[LineageEdge] = []
    if source_manifest:
        edges.append(LineageEdge(from_uri=source_manifest, to_uri=source_uri, relation="evaluated_on", run_id=run_id))
    return metrics, edges


def _extract_adversarial_set(
    payload: dict[str, Any], source_uri: str, workflow: str, workflow_run: str
) -> tuple[list[MetricRecord], list[LineageEdge]]:
    lineage_meta = payload.get("lineage") or {}
    scenarios = payload.get("scenarios") or []
    severities = [float(s.get("severity", 0.0) or 0.0) for s in scenarios]
    diversities = [float(s.get("diversity", 0.0) or 0.0) for s in scenarios]
    run_id = _run_id(payload, workflow_run)
    policy_uri = str(lineage_meta.get("policy_uri", ""))
    base_config_uri = str(lineage_meta.get("base_config_uri", ""))
    input_uris = [u for u in (policy_uri, base_config_uri) if u]
    ref = LineageRef(input_uris=input_uris, checkpoint_uri=policy_uri)
    metrics = [
        _metric(run_id=run_id, workflow=workflow, tool="scenario_gen", stage="generate", name="scenario_count", value=int(payload.get("scenario_count", len(scenarios)) or 0), unit="scenarios", lineage=ref, artifact_uri=source_uri, artifact_version=run_id),
        _metric(run_id=run_id, workflow=workflow, tool="scenario_gen", stage="generate", name="top_severity", value=max(severities) if severities else 0.0, lineage=ref, artifact_uri=source_uri, artifact_version=run_id),
        _metric(run_id=run_id, workflow=workflow, tool="scenario_gen", stage="generate", name="mean_severity", value=round(sum(severities) / len(severities), 4) if severities else 0.0, lineage=ref, artifact_uri=source_uri, artifact_version=run_id),
        _metric(run_id=run_id, workflow=workflow, tool="scenario_gen", stage="generate", name="mean_diversity", value=round(sum(diversities) / len(diversities), 4) if diversities else 0.0, lineage=ref, artifact_uri=source_uri, artifact_version=run_id),
    ]
    edges = [
        LineageEdge(from_uri=input_uri, to_uri=source_uri, to_version=run_id, relation="produced_from", run_id=run_id)
        for input_uri in input_uris
    ]
    return metrics, edges


def _parse_accelerators(spec: str) -> tuple[str, int]:
    """Parse a SkyPilot-style accelerator spec (``TYPE:COUNT``) into (type, count).

    ``"RTXPRO6000:4"`` -> ``("RTXPRO6000", 4)``; a bare ``"H100"`` implies one
    device; empty/unparseable specs yield ``("", 0)`` so nothing is emitted.
    """
    text = str(spec or "").strip()
    if not text:
        return "", 0
    if ":" in text:
        acc_type, _, count_raw = text.partition(":")
        try:
            count = int(float(count_raw.strip()))
        except (TypeError, ValueError):
            count = 1 if acc_type.strip() else 0
        return acc_type.strip(), max(count, 0)
    return text, 1


def _extract_run_manifest(
    payload: dict[str, Any], source_uri: str, workflow: str, workflow_run: str
) -> tuple[list[MetricRecord], list[LineageEdge]]:
    """Extract observed resource/runtime signals from a workflow run manifest.

    Reads each step's ``resources_profile.accelerators`` (already produced by the
    workflow interpreter/planner) and records the peak accelerator count for the
    run. Duration, throughput, curve, and billing signals follow the same rule:
    only values physically present in the manifest or step records are emitted.
    """
    # A run that was only *planned* never touched an accelerator, so reporting a GPU
    # count for it would be a fabrication: `run-spec --persist-state` without
    # `--execute` writes a manifest with the full resource profile and status
    # "planned". Submitted/running/completed/failed runs did request the hardware.
    if str(payload.get("status") or "").strip().lower() == "planned":
        return [], []
    run_id = str(payload.get("run_id") or workflow_run or "unknown")
    resolved_workflow = workflow or str(payload.get("workflow") or "")
    steps = payload.get("steps") or []
    max_gpus = 0
    accel_types: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        profile = step.get("resources_profile")
        accel = str(profile.get("accelerators", "")) if isinstance(profile, dict) else ""
        acc_type, count = _parse_accelerators(accel)
        if count <= 0:
            continue
        max_gpus = max(max_gpus, count)
        if acc_type and acc_type not in accel_types:
            accel_types.append(acc_type)
    records: list[MetricRecord] = []
    edges: list[LineageEdge] = []
    if max_gpus > 0:
        labels = {ACCELERATORS_LABEL: ",".join(accel_types)} if accel_types else {}
        records.append(
            _metric(
                run_id=run_id,
                workflow=resolved_workflow,
                tool="workflow",
                stage="run",
                name=GPU_METRIC_NAME,
                value=float(max_gpus),
                unit="gpus",
                labels=labels,
                artifact_uri=source_uri,
            )
        )

    observed, observed_edges = _extract_observed_report(
        payload,
        source_uri,
        resolved_workflow,
        workflow_run,
        tool="workflow",
        stage="run",
        schema_id=WORKFLOW_RUN_SCHEMA,
    )
    records.extend(observed)
    edges.extend(observed_edges)
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_payload = {**step, "run_id": run_id}
        step_tool = str(step.get("tool_ref") or "workflow")
        step_stage = str(step.get("state") or step.get("stage") or "run")
        step_records, step_edges = _extract_observed_report(
            step_payload,
            source_uri,
            resolved_workflow,
            workflow_run,
            tool=step_tool,
            stage=step_stage,
            schema_id=WORKFLOW_RUN_SCHEMA,
        )
        records.extend(step_records)
        edges.extend(step_edges)
    return records, edges


def _extract_decision(
    payload: dict[str, Any], source_uri: str, workflow: str, workflow_run: str
) -> tuple[list[MetricRecord], list[LineageEdge]]:
    from npa.orchestration.npa_workflow.decisions import normalize_decision

    raw = str(payload.get("decision", ""))
    decision = normalize_decision(raw)
    promoted = 1.0 if decision == "promote_checkpoint" else 0.0
    run_id = _run_id(payload, workflow_run)
    metric = _metric(
        run_id=run_id,
        workflow=workflow,
        tool="workflow",
        stage="gate",
        name="gate_promote",
        value=promoted,
        labels={"decision": decision},
        artifact_uri=source_uri,
    )
    checkpoint_uri = _first_uri(
        payload, ("checkpoint_uri", "policy_checkpoint", "checkpoint_path")
    )
    eval_uri = _first_uri(
        payload,
        ("eval_report_uri", "heldout_report_uri", "evaluation_uri", "report_uri"),
    )
    edges: list[LineageEdge] = []
    if checkpoint_uri and eval_uri:
        edges.append(
            LineageEdge(
                from_uri=checkpoint_uri,
                to_uri=eval_uri,
                relation="evaluated_on",
                run_id=run_id,
            )
        )
    if eval_uri:
        edges.append(
            LineageEdge(
                from_uri=eval_uri,
                to_uri=source_uri,
                relation="derived_from",
                run_id=run_id,
            )
        )
    return [metric], edges
