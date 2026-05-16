# BDD100K SkyPilot Pipeline

This cookbook describes the SkyPilot workflow at
`npa/workflows/skypilot/bdd100k-pipeline.yaml`.

The workflow composes the six BDD100K reproduction stages:

1. Import the BDD100K subset into LanceDB with `POST /import-bdd100k`.
2. Backfill CPU UDF columns with five sequential `POST /backfill` calls.
3. Backfill `clip_embedding` with `POST /backfill` on an H100 task.
4. Create the three failure-mode materialized views with `POST /create-mv`.
5. Train one detector per failure-mode view with `POST /train`.
6. Evaluate each trained detector with `POST /eval`.

SkyPilot 0.12.2 supports serial pipelines and all-parallel job groups, but not
mixed dependency graphs in one YAML. This pipeline therefore serializes the
three training tasks and three evaluation tasks. The logical DAG is still:

```text
ingest -> CPU backfill -> CLIP backfill -> materialized views -> training x3 -> eval x3
```

## Dry Validation

Use the wrapper's mock-endpoint path to validate the curl requests without
submitting GPU work:

```bash
PYTHONPATH=npa/src npa/.venv/bin/python npa/scripts/run_bdd100k_pipeline.py \
  --mock-endpoints \
  --run-id bdd100k-pipeline-dryrun
```

## Full Submission

Full submission requires a working SkyPilot 0.12.2 binary:

```bash
export NPA_SKYPILOT_BIN=/opt/npa/skypilot/bin/sky
PYTHONPATH=npa/src npa/.venv/bin/python npa/scripts/run_bdd100k_pipeline.py \
  --run-id bdd100k-pipeline-$(date -u +%Y%m%dT%H%M%SZ) \
  --cleanup
```

The wrapper renders run-specific S3 paths before calling
`npa.orchestration.skypilot.submit_workflow`.

## Required Services

The task pods call existing workbench services by HTTP. Override these if the
service names differ:

```bash
--lancedb-endpoint http://npa-lancedb.workbench.svc.cluster.local:8686
--detection-endpoint http://npa-detection-training.workbench.svc.cluster.local:8790
```

The default input source is
`s3://YOUR_S3_BUCKET/raw-bdd100k/subset-demo/`. Full submission requires the
configured S3 credentials to list and read this prefix.

## Images

The YAML pins the validated images:

- `cr.eu-north1.nebius.cloud/YOUR_REGISTRY_ID/npa-lancedb:bdd100k-clip-w9bdd100k-clip-embedding-20260516T174407Z`
- `cr.eu-north1.nebius.cloud/YOUR_REGISTRY_ID/npa-detection-training:bdd100k-w9-detection-training-tool-20260516T173720Z`

## Output Layout

For `run_id=<run-id>`, outputs are rooted at:

```text
s3://YOUR_S3_BUCKET/bdd100k-pipeline/<run-id>/
```

The LanceDB URI is `<root>/lancedb/`. Training outputs are under
`<root>/training/<view-name>/`, and evaluation outputs are under
`<root>/eval/<view-name>/`.
