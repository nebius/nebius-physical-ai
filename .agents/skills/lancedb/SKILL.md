---
name: lancedb
description: Use when working on LanceDB vector storage, BDD100K imports, UDFs, materialized views, CLIP embeddings, or AV/perception data flows.
---

# LanceDB

LanceDB is the vector database and multimodal data store for AV/perception workloads.

It is best suited for frame-aligned data, such as images plus annotations plus vectors. It is not the right store for multi-rate sensor streams.

## Interfaces

API:

- `POST /import-bdd100k`
- `POST /backfill`
- `POST /create-mv`
- `GET /tables`
- `GET /status`

CLI:

```bash
npa workbench lancedb deploy
npa workbench lancedb launch
npa workbench lancedb load-dataset
npa workbench lancedb status
npa workbench lancedb system-info
npa workbench lancedb list
```

## BDD100K Contract

BDD100K UDFs:

- `has_person`
- `has_rider`
- `person_bbox_area_pct`
- `dhash`
- `is_duplicate`
- `clip_embedding`

`PERSON_CATEGORIES = {"person", "pedestrian"}`. Real BDD100K uses `pedestrian`; synthetic data uses `person`. Both must be accepted.

Materialized views are SQL-defined failure-mode slices: `rider_train`, `nighttime_person_train`, and `distant_person_train`.

CLIP embeddings are 512-dimensional `float32`, use a GPU UDF, and route to H100 rather than L40S.

Inject the detection-training label map through the pipeline YAML env var `BDD100K_LABEL_MAP`. Do not hardcode the label map in tool source.

Lance URI pattern:

```text
s3://${NPA_S3_BUCKET}/bdd100k-pipeline/<run-id>/lancedb/
```

Always use storage endpoint `storage.eu-north1.nebius.cloud`; the uk-south1 default is wrong.
