"""Query, comparison, lineage traversal, and dashboard rollups over the store."""

from __future__ import annotations

import html
import time
from typing import Any, Callable

from .integrations import query_metrics_in_lancedb
from .schemas import (
    LOWER_IS_BETTER_HINTS,
    CompareRequest,
    CompareResponse,
    DashboardGroup,
    DashboardRequest,
    DashboardResponse,
    LineageNode,
    LineageRequest,
    LineageResponse,
    MetricDelta,
    QueryRequest,
    QueryResponse,
)
from .store import read_edges, read_records
from .storage import uri_join, write_text_uri

_OPS: dict[str, Callable[[float, float], bool]] = {
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
}


class InsightsQueryError(RuntimeError):
    """Raised when a query, comparison, or lineage traversal fails."""


def _facets(request: QueryRequest) -> dict[str, Any]:
    return {
        "workflow": request.workflow,
        "run_id": request.run_id,
        "tool": request.tool,
        "stage": request.stage,
        "dataset_version": request.dataset_version,
        "model_version": request.model_version,
        "metric_name": request.metric_name,
        "time_start": request.time_start,
        "time_end": request.time_end,
        "threshold_metric": request.threshold_metric,
        "threshold_op": request.threshold_op,
        "threshold_value": request.threshold_value,
    }


def _matches(record: dict[str, Any], request: QueryRequest) -> bool:
    lineage = record.get("lineage") or {}
    labels = record.get("labels") or {}
    if request.workflow and record.get("workflow") != request.workflow:
        return False
    if request.run_id and record.get("run_id") != request.run_id:
        return False
    if request.tool and record.get("tool") != request.tool:
        return False
    if request.stage and record.get("stage") != request.stage:
        return False
    if request.metric_name and record.get("metric_name") != request.metric_name:
        return False
    if request.dataset_version and request.dataset_version not in (
        lineage.get("dataset_version", ""),
        record.get("artifact_version", ""),
    ):
        return False
    if request.model_version and request.model_version not in (
        lineage.get("checkpoint_uri", ""),
        labels.get("model_version", ""),
        record.get("artifact_version", ""),
    ):
        return False
    timestamp = str(record.get("timestamp", ""))
    if request.time_start and timestamp < request.time_start:
        return False
    if request.time_end and timestamp > request.time_end:
        return False
    if request.threshold_op and request.threshold_value is not None:
        if request.threshold_metric and record.get("metric_name") != request.threshold_metric:
            return False
        if not _OPS[request.threshold_op](float(record.get("value", 0.0)), request.threshold_value):
            return False
    return True


def query_metrics(request: QueryRequest) -> QueryResponse:
    """Query metric records by facet (LanceDB index or JSONL fallback)."""
    facets = _facets(request)
    if request.lancedb_endpoint.strip():
        records = query_metrics_in_lancedb(
            lancedb_endpoint=request.lancedb_endpoint,
            filter_predicate=facets,
            limit=request.limit,
        )
        return QueryResponse(backend="lancedb", count=len(records), records=records, facets=facets)

    matched = [record for record in read_records(request.input_uri) if _matches(record, request)]
    limited = matched[: request.limit]
    return QueryResponse(backend="jsonl", count=len(limited), records=limited, facets=facets)


def _is_lower_better(metric_name: str, overrides: list[str]) -> bool:
    if metric_name in overrides:
        return True
    lowered = metric_name.lower()
    return any(hint in lowered for hint in LOWER_IS_BETTER_HINTS)


def _run_metric_map(records: list[dict[str, Any]], run_id: str) -> dict[str, float]:
    """Latest value per metric for a run (by timestamp, then order)."""
    latest: dict[str, tuple[str, float]] = {}
    for index, record in enumerate(records):
        if record.get("run_id") != run_id:
            continue
        name = str(record.get("metric_name", ""))
        stamp = f"{record.get('timestamp', '')}:{index:08d}"
        value = float(record.get("value", 0.0))
        if name not in latest or stamp >= latest[name][0]:
            latest[name] = (stamp, value)
    return {name: value for name, (_, value) in latest.items()}


def compare_runs(request: CompareRequest) -> CompareResponse:
    """Compare a metric set between two run ids; flag regressed/improved."""
    records = read_records(request.input_uri)
    base = _run_metric_map(records, request.base_run)
    candidate = _run_metric_map(records, request.candidate_run)
    if not base:
        raise InsightsQueryError(f"no metrics recorded for base run: {request.base_run}")
    if not candidate:
        raise InsightsQueryError(f"no metrics recorded for candidate run: {request.candidate_run}")

    names = request.metric_names or sorted(set(base) & set(candidate))
    if not names:
        raise InsightsQueryError("no shared metrics between the two runs to compare")

    deltas: list[MetricDelta] = []
    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []
    for name in names:
        if name not in base or name not in candidate:
            continue
        base_value = base[name]
        candidate_value = candidate[name]
        delta = round(candidate_value - base_value, 6)
        pct = round((delta / base_value) * 100, 4) if base_value else None
        lower_better = _is_lower_better(name, request.lower_is_better)
        if delta == 0:
            status = "unchanged"
            unchanged.append(name)
        elif (delta < 0) == lower_better:
            status = "improved"
            improved.append(name)
        else:
            status = "regressed"
            regressed.append(name)
        deltas.append(
            MetricDelta(
                metric_name=name,
                base_value=base_value,
                candidate_value=candidate_value,
                delta=delta,
                pct_change=pct,
                lower_is_better=lower_better,
                status=status,
            )
        )
    return CompareResponse(
        base_run=request.base_run,
        candidate_run=request.candidate_run,
        metrics=deltas,
        improved=improved,
        regressed=regressed,
        unchanged=unchanged,
        summary={"improved": len(improved), "regressed": len(regressed), "unchanged": len(unchanged)},
    )


def traverse_lineage(request: LineageRequest) -> LineageResponse:
    """Reconstruct ancestors + descendants of an artifact from lineage edges."""
    edges = read_edges(request.input_uri)
    incoming: dict[str, list[dict[str, Any]]] = {}
    outgoing: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        outgoing.setdefault(str(edge.get("from_uri")), []).append(edge)
        incoming.setdefault(str(edge.get("to_uri")), []).append(edge)

    nodes: set[str] = {request.uri}
    ancestors: list[dict[str, Any]] = []
    descendants: list[dict[str, Any]] = []

    if request.direction in ("both", "ancestors"):
        ancestors = _walk(request.uri, incoming, "from_uri", request.depth, nodes)
    if request.direction in ("both", "descendants"):
        descendants = _walk(request.uri, outgoing, "to_uri", request.depth, nodes)

    return LineageResponse(
        root=LineageNode(uri=request.uri, version=request.version),
        ancestors=ancestors,
        descendants=descendants,
        nodes=sorted(nodes),
    )


def _walk(
    start: str,
    adjacency: dict[str, list[dict[str, Any]]],
    next_key: str,
    depth: int,
    nodes: set[str],
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    frontier = [(start, 0)]
    while frontier:
        current, level = frontier.pop(0)
        if depth != -1 and level >= depth:
            continue
        for edge in adjacency.get(current, []):
            key = (str(edge.get("from_uri")), str(edge.get("to_uri")), str(edge.get("relation")))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            collected.append(edge)
            nxt = str(edge.get(next_key))
            nodes.add(nxt)
            frontier.append((nxt, level + 1))
    return collected


def build_dashboard(request: DashboardRequest) -> DashboardResponse:
    """Build a grouped metric rollup + optional static HTML report."""
    records = [
        record
        for record in read_records(request.input_uri)
        if not request.workflow or record.get("workflow") == request.workflow
    ]
    runs = sorted({str(record.get("run_id", "")) for record in records if record.get("run_id")})
    latest_run = request.latest_run or _latest_run(records)

    groups = _group(records, request.group_by)
    latest_rollup = {
        name: value for name, value in _run_metric_map(records, latest_run).items()
    } if latest_run else {}

    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    response = DashboardResponse(
        store_uri=request.input_uri,
        generated_at=generated_at,
        total_records=len(records),
        runs=runs,
        latest_run=latest_run,
        group_by=request.group_by,
        groups=groups,
        latest_rollup=latest_rollup,
    )
    if request.output_path.strip():
        html_uri = uri_join(request.output_path, "dashboard.html")
        write_text_uri(html_uri, _render_html(response))
        response.html_uri = html_uri
    return response


def _latest_run(records: list[dict[str, Any]]) -> str:
    latest_stamp = ""
    latest_run = ""
    for index, record in enumerate(records):
        stamp = f"{record.get('timestamp', '')}:{index:08d}"
        if stamp >= latest_stamp:
            latest_stamp = stamp
            latest_run = str(record.get("run_id", ""))
    return latest_run


def _group(records: list[dict[str, Any]], group_by: str) -> list[DashboardGroup]:
    buckets: dict[str, list[tuple[str, float]]] = {}
    for index, record in enumerate(records):
        key = str(record.get(group_by, "")) or "(none)"
        stamp = f"{record.get('timestamp', '')}:{index:08d}"
        buckets.setdefault(key, []).append((stamp, float(record.get("value", 0.0))))
    groups: list[DashboardGroup] = []
    for key in sorted(buckets):
        entries = buckets[key]
        values = [value for _, value in entries]
        latest_value = max(entries, key=lambda item: item[0])[1]
        groups.append(
            DashboardGroup(
                key=key,
                count=len(values),
                latest_value=round(latest_value, 6),
                min=round(min(values), 6),
                max=round(max(values), 6),
                mean=round(sum(values) / len(values), 6),
            )
        )
    return groups


def _render_html(response: DashboardResponse) -> str:
    """Render a self-contained single-file HTML report (no external deps)."""
    rows = "\n".join(
        "<tr><td>{key}</td><td>{count}</td><td>{latest}</td><td>{mn}</td>"
        "<td>{mx}</td><td>{mean}</td></tr>".format(
            key=html.escape(group.key),
            count=group.count,
            latest=group.latest_value,
            mn=group.min,
            mx=group.max,
            mean=group.mean,
        )
        for group in response.groups
    )
    rollup_rows = "\n".join(
        f"<tr><td>{html.escape(name)}</td><td>{value}</td></tr>"
        for name, value in sorted(response.latest_rollup.items())
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NPA Insights Dashboard</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
h1 {{ font-size: 1.4rem; }}
table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.7rem; text-align: left; }}
th {{ background: #f2f2f2; }}
.meta {{ color: #555; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>NPA Insights Dashboard</h1>
<p class="meta">store: {html.escape(response.store_uri)}<br>
generated: {html.escape(response.generated_at)} &middot;
records: {response.total_records} &middot;
runs: {len(response.runs)} &middot;
latest run: {html.escape(response.latest_run)}</p>
<h2>Metrics grouped by {html.escape(response.group_by)}</h2>
<table>
<thead><tr><th>{html.escape(response.group_by)}</th><th>count</th><th>latest</th>
<th>min</th><th>max</th><th>mean</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
<h2>Latest-run rollup</h2>
<table>
<thead><tr><th>metric</th><th>value</th></tr></thead>
<tbody>
{rollup_rows}
</tbody>
</table>
</body>
</html>
"""
