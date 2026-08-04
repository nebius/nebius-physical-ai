---
name: insights
description: Use when turning the reports/manifests other workbench tools emit into a queryable lineage graph + common metrics store, or when querying/comparing metrics, tracing artifact lineage, or building a run dashboard.
---

# Insights (Lineage & Metrics backbone)

Insights is the connective tissue that makes workflow runs legible. It turns the
structured reports/manifests other tools already emit into a queryable **lineage
graph + common metrics store** — the foundation for dashboards and an
orchestrating agent. It does not replace any tool; it aggregates them.

## Three-access pattern

Source of truth is the FastAPI service
(`npa/src/npa/workbench/insights/service.py`). The CLI
(`npa/src/npa/cli/workbench/insights.py`) and SDK
(`npa/src/npa/sdk/workbench/insights.py`) are thin clients. Do not duplicate
logic across layers.

## Store layout

The store is an **append-only index on S3** under a configurable prefix
(`--output-path`), with a JSONL fallback so it works without any database:

- `records.jsonl` — metric records (`npa.insights.metric_record.v1`).
- `edges.jsonl` — lineage edges (`npa.insights.lineage_edge.v1`).
- `records.d/` and `edges.d/` — immutable append shards, one object per write.

Every append writes a **new shard object**; readers concatenate the base object
(legacy stores) plus all shards. Never rewrite a whole JSONL object to append:
object storage has no native append, so read-modify-write silently drops rows
when two writers overlap (both read N, both write N + their own).

Readers expose a logically idempotent view: metric rows are deduplicated by run,
source artifact URI, metric, stage/tool, canonical labels (including curve step),
and lineage; lineage edges use their endpoint/version/relation/run identity. This
also repairs legacy stores that already contain duplicate shards. Explicit
emissions without an artifact URI retain their timestamp/value identity so
distinct observations with the same metric name are not collapsed.

Do NOT introduce a database service or hardcode a metadata backend. Reuse the
**LanceDB** tool as the optional query index (HTTP seam in `integrations.py`),
exactly as `dataset` does; absence degrades to the JSONL scan.

## Interfaces

CLI:

```bash
npa workbench insights record --input-path <records.json> --output-path <store>
npa workbench insights ingest-run --input-path <run-prefix> --output-path <store>
npa workbench insights query --input-path <store> --tool <t> --metric-name <m>
npa workbench insights lineage --input-path <store> --uri <artifact>
npa workbench insights compare --input-path <store> --base-run <a> --candidate-run <b>
npa workbench insights dashboard --input-path <store> --output-path <s3>
npa workbench insights status --input-path <store>
npa workbench insights system-info
npa workbench insights list
```

Endpoints: `/health`, `/status`, `/system-info`, `/list`, `POST /record`,
`POST /ingest-run`, `GET /lineage`, `GET /query`, `GET /compare`,
`GET /dashboard`.

## API contract

- `POST /record`: append one or more metric emissions (+ lineage edges) keyed by
  run id + lineage refs. Rows validate against `npa.insights.metric_record.v1`.
- `POST /ingest-run`: **non-invasive ingestion** — scan an S3 run prefix for
  known schemas (`npa.dataset.manifest.v1`, `npa.dataset.validation_report.v1`,
  `npa.scenario_gen.adversarial_set.v1`, and gate/threshold decision JSONs),
  extract their metrics + provenance, and write them into the store. This does
  NOT require modifying the emitting tools.
- `GET /lineage`: traverse the provenance graph (ancestors + descendants) for an
  artifact/version, reconstructed from recorded `lineage_edge` records.
- `GET /query`: query metric records by facet (workflow, run id, tool, stage,
  dataset/model version, metric name, metric kind, cost basis, time range,
  threshold predicate). Cost records always label `cost_basis` as `estimated` or
  `billed`; only billing artifacts are authoritative for billed dollars.
- `GET /compare`: cross-run/cross-stack comparison; emits
  `npa.insights.comparison.v1` (per-metric delta + regressed/improved flags).
  Metrics whose name looks failure-like (corruption/latency/loss/…) are treated
  as lower-is-better; override with `--lower-is-better`.
- `GET /dashboard`: return `npa.insights.dashboard.v1` (grouped metrics +
  latest-run rollup) and optionally write a self-contained static HTML report to
  `--output-path`. Keep viz thin — JSON + a single-file HTML, no web UI.

Known evaluation reports use an explicit numeric taxonomy: score/quality fields
become `eval_score`, step/sample/epoch values become `counter`, and latency or
`*_ms` values become `duration`. Arbitrary numeric metadata is not promoted to a
score. When root placeholders conflict with nested `metrics` or
`success_summary`, the authoritative nested value wins deterministically.

## Lineage

Thread and preserve lineage that already exists in upstream manifests (input
URIs, dataset/checkpoint versions, parent versions, produced_from/derived_from/
evaluated_on relations). The whole point is cross-tool traceability — do not
drop it.

## GPU routing

CPU-only. Aggregation, query, comparison, lineage, and dashboard need no GPU and
no rendering path (headless).

## Workflows

SkyPilot stays the execution engine, but the shown catalog is npa.workflow-only
(no raw SkyPilot task YAMLs). All insights pipelines are declarative
`npa.workflow/v0.0.1` specs, CPU-only and `cloud: kubernetes`:

- Aggregate a run (ingest-run -> dashboard):
  `npa/workflows/workbench/npa-workflows/insights-aggregate.yaml`
- Hardening with insights (hardening stages -> ingest-run -> dashboard):
  `npa/workflows/workbench/npa-workflows/hardening-with-insights.yaml`
- CPU-only smoke (ingest fixture -> compare -> dashboard):
  `npa/workflows/workbench/npa-workflows/insights-smoke.yaml`

toolRefs: `workbench.insights.record`, `workbench.insights.ingest_run`,
`workbench.insights.compare`, `workbench.insights.dashboard`.

## Known issues

- Object storage has no native append, so each write lands in its own immutable
  shard under `records.d/` / `edges.d/` and reads concatenate base + shards. This
  is what makes concurrent ingests safe; a store is never rewritten in place.
- Re-ingesting a source prefix may append nothing (`recorded_count: 0`) when all
  logical observations already exist. Concurrent writers may still create
  duplicate immutable shards, but readers deduplicate them without losing
  distinct source observations or skewing dashboard means.
- **Reader version skew:** a reader older than sharding sees only the base object
  and silently reports a truncated store (e.g. an agent VM answering "no runs
  found" for runs that did ingest). Re-bootstrap deployed agents
  (`npa agent bootstrap --project <alias> --name agent`) after upgrading the
  store writers.
- `compare` needs both run ids present in the store; comparing a run to itself
  reports every metric as unchanged (useful as a smoke self-check). A `compare`
  that fails with `no metrics recorded for base run` right after a successful
  ingest means the base run's rows are missing from the store, not that the run
  never ingested — check for rows dropped by a writer that rewrote the object.
- A `gpus` metric only exists when an ingested `npa.workflow.run.v1` manifest has
  a step whose `resources_profile.accelerators` parses to >= 1, and manifests with
  status `planned` are skipped. CPU-only and never-executed runs therefore report
  no GPU count at all rather than a fabricated zero.
