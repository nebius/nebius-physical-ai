---
name: lancedb
description: Use when working on LanceDB vector storage, table creation/querying, LeRobot/BDD100K imports, UDF backfills, materialized views, CLIP embeddings, or AV/perception data flows.
---

# LanceDB

## When To Use

Use this skill for vector-search workbench changes, perception dataset imports,
BDD100K failure-mode slices, materialized views, CLIP embedding backfills, and
LanceDB CLI/API/SDK parity reviews.

## Procedure

1. Pick the data shape first. LanceDB is best for frame-aligned records such as
   image paths, annotations, metadata, and vectors. It is not the right store for
   raw multi-rate sensor streams.
2. Create or inspect tables before ingestion:

   ```bash
   npa workbench lancedb create-table --help
   npa workbench lancedb query --help
   ```

3. Import supported datasets through current commands:

   ```bash
   npa workbench lancedb import-lerobot --help
   npa workbench lancedb import-bdd100k --help
   ```

4. Add derived fields through `backfill`, then materialize reusable SQL slices
   with `create-mv`, `refresh-mv`, and `query-table`.

## Three-Tier Contract

- CLI: `deploy`, `status`, `list`, `create-table`, `query`,
  `import-lerobot`, `import-bdd100k`, `backfill`, `create-mv`, `refresh-mv`,
  and `query-table`.
- SDK/API: keep table import, backfill, and query behavior in shared
  implementation paths so CLI, SDK, and service endpoints produce equivalent
  manifests and row counts.
- YAML: workflow tasks should pass S3-backed LanceDB URIs and table names through
  environment variables, not hardcoded project paths.

## BDD100K Contract

BDD100K UDFs:

- `has_person`
- `has_rider`
- `person_bbox_area_pct`
- `dhash`
- `is_duplicate`
- `clip_embedding`

`PERSON_CATEGORIES = {"person", "pedestrian"}`. Real BDD100K uses
`pedestrian`; synthetic data may use `person`. Both must be accepted.

Materialized views are SQL-defined failure-mode slices such as `rider_train`,
`nighttime_person_train`, and `distant_person_train`. CLIP embeddings are
512-dimensional `float32`, use a GPU UDF, and route to H100.

The current GPU-capable image is
`npa-lancedb:cuda13-b300-0.30.3-sm80-sm90-sm100-sm103-sm120-20260803T031514Z`
(`sha256:a303b53d0769e612101d468ba957656997838f4b8e6a03430f2b5a2c89e3f8b5`
in both registries). It was measured on physical B200, B300, H100, and RTX PRO
6000 by embedding three images with CLIP, checking normalized distinct vectors,
writing them to Lance, and checking top-1 self-search. The historical `0.30.3`
image predates the current transformers return-type fix and must not be used for
the CLIP path.

## Gotchas

- Do not document stale `launch` or `load-dataset` commands for LanceDB.
- Inject detection-training label maps through workflow env vars such as
  `BDD100K_LABEL_MAP`; do not hardcode them in tool source.
- Use `https://storage.eu-north1.nebius.cloud` for primary-region object
  storage.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

The smoke test invokes current LanceDB command help and fails if stale commands
return to the manifest.
