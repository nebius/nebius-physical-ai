---
name: dataset
description: Use when ingesting, validating, curating, or querying production sensor data as a versioned dataset-of-record, or wiring the dataset-ingest-curate workflow.
---

# Dataset (Dataset-of-Record)

A unified ingestion / validation / curation layer that turns raw production
sensor data into a queryable, versioned **dataset-of-record**, filterable by
event, location, and quality. It composes existing primitives (FiftyOne for
curation/visualization, LanceDB for the vector/metadata query index, S3 as the
bus) behind one tool instead of leaving them disconnected.

## Three-access pattern

Source of truth is the FastAPI service (`npa/src/npa/workbench/dataset/service.py`).
The CLI (`npa/src/npa/cli/workbench/dataset.py`) and SDK
(`npa/src/npa/sdk/workbench/dataset.py`) are thin clients. Do not duplicate logic
across layers.

## Interfaces

CLI:

```bash
npa workbench dataset ingest --input-path <s3> --output-path <s3> --dataset-id <id>
npa workbench dataset validate --input-path <s3-manifest> --output-path <s3>
npa workbench dataset curate --input-path <s3-manifest> --output-path <s3> --event <e> --location <l>
npa workbench dataset query --input-path <s3-manifest> --event <e> --location <l>
npa workbench dataset status --dataset-id <id> --version <v>
npa workbench dataset system-info
npa workbench dataset list
```

Endpoints: `/health`, `/status`, `/system-info`, `/list`, `POST /ingest`,
`POST /validate`, `POST /curate`, `GET /query`.

## API contract

- `POST /ingest`: pull raw sensor data from `--input-path`, validate against the
  declared sensor schema, normalize to canonical records, and register a
  versioned manifest at `--output-path` (schema `npa.dataset.manifest.v1`:
  dataset id + version, record count, sensor modalities, source lineage,
  per-record S3 pointers, quality stats).
- `POST /validate`: schema + quality-metric validation (completeness, corruption,
  per-sensor sanity); emits `npa.dataset.validation_report.v1`.
- `POST /curate`: filter/slice by **event of interest, location, and quality
  metric**; writes a derived version whose manifest records lineage back to the
  parent (parent dataset id/version + filter predicate).
- `GET /query`: query records by event/location/quality facets. Backed by the
  LanceDB index when `--lancedb-endpoint` is set; falls back to the manifest so
  the tool works without a running LanceDB.

Reuse the FiftyOne tool for curation/visualization handoff and the LanceDB tool
for the query index (see `integrations.py`) rather than re-implementing either —
these are HTTP seams mocked in tests.

## Lineage

Every manifest threads provenance (workflow run, input URIs, dataset version,
parent dataset id/version, filter predicate) so a later lineage/metadata service
can consume it. Do not hardcode a metadata backend; keep lineage in the S3
manifests.

## GPU routing

Ingest / validate / curate / query are CPU-only. The optional embedding backfill
that populates the LanceDB query index runs on H100 (general training class).

## SkyPilot + workflow

- SkyPilot (CPU + optional GPU embedding backfill, `cloud: kubernetes`):
  `npa/workflows/workbench/skypilot/dataset-ingest-curate.yaml`
- Declarative pipeline (ingest -> validate quality gate -> curate -> register
  queryable version): `npa/workflows/workbench/npa-workflows/dataset-ingest-curate.yaml`

toolRefs: `workbench.dataset.ingest`, `workbench.dataset.validate`,
`workbench.dataset.curate`, `workbench.dataset.query`,
`workbench.dataset.write_quality_decision`, `workbench.dataset.report_rejection`.

## Known issues

- The quality gate rejects the version when mean completeness is below
  `config.completeness_min` or the corruption rate exceeds
  `config.max_corruption_rate`.
- Curated child versions are content-addressed (`<parent>.curated-<hash>`); a
  workflow that queries a curated version wires the concrete manifest URI at
  runtime.
