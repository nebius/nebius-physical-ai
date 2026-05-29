# BDD100K SkyPilot Pipeline

This cookbook describes the SkyPilot workflow at
`npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml`.

The workflow composes the six BDD100K reproduction stages:

1. Import the BDD100K subset into LanceDB with `POST /import-bdd100k`.
2. Backfill CPU UDF columns with five sequential `POST /backfill` calls.
3. Backfill `clip_embedding` with `POST /backfill` on an H100 task.
4. Create the three failure-mode materialized views with `POST /create-mv`.
5. Train one detector per failure-mode view with `POST /train`.
6. Evaluate each trained detector with `POST /eval`.
7. Launch a FiftyOne App on `--address 0.0.0.0 --port 5151` with the SkyPilot port exposed for public review.

SkyPilot 0.12.2 supports serial pipelines and all-parallel job groups, but not
mixed dependency graphs in one YAML. This pipeline therefore serializes the
three training tasks and three evaluation tasks. The logical DAG is still:

```text
ingest -> CPU backfill -> CLIP backfill -> materialized views -> training x3 -> eval x3 -> FiftyOne app
```

## Dry Validation

Use the wrapper's mock-endpoint path to validate the curl requests without
submitting GPU work:

```bash
python npa/scripts/run_bdd100k_pipeline.py \
  --yaml npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml \
  --synthetic 5000 \
  --mock-endpoints \
  --run-id <your-run-id>
```

## Full Submission

Full submission requires a working SkyPilot 0.12.2 binary:

```bash
export NPA_SKYPILOT_BIN=/opt/npa/skypilot/bin/sky
python npa/scripts/run_bdd100k_pipeline.py \
  --yaml npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml \
  --synthetic 5000 \
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
`s3://${NPA_S3_BUCKET}/raw-bdd100k/subset-demo/`. Full submission requires the
configured S3 credentials to list and read this prefix.

## Images

The YAML uses placeholder image tags because SkyPilot 0.12.2 does not expand
same-block environment variables inside `image_id`. Replace them before live
submission:

- `cr.eu-north1.nebius.cloud/<your-registry-id>/npa-lancedb:<lancedb-image-tag>`
- `cr.eu-north1.nebius.cloud/<your-registry-id>/npa-detection-training:<detection-training-image-tag>`
- `cr.eu-north1.nebius.cloud/<your-registry-id>/npa-fiftyone:<fiftyone-image-tag>`

The final FiftyOne task exposes port `5151` through SkyPilot. The app does not add authentication; restrict the run inputs to datasets that are safe to show publicly and use `sky status --endpoint 5151 <cluster>` to resolve the public URL.

## Output Layout

For `run_id=<run-id>`, outputs are rooted at:

```text
s3://${NPA_S3_BUCKET}/bdd100k-pipeline/<run-id>/
```

The LanceDB URI is `<root>/lancedb/`. Training outputs are under
`<root>/training/<view-name>/`, and evaluation outputs are under
`<root>/eval/<view-name>/`.
