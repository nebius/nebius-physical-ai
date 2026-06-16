# LanceDB Vector Search

LanceDB gives the Workbench a CPU-only vector-search and data-lake layer for
robotics datasets. The v1 integration wraps the OSS Python package in a small
NPA service that stores Lance data in a local path or an S3-compatible object
store prefix.

## Quick Start

Build the local image:

```bash
npa/docker/workbench/lancedb/build.sh
```

The first-party registry default is
`cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-lancedb:0.30.3`.

Start a local container-backed service:

```bash
npa workbench lancedb deploy \
  --runtime container \
  --storage-path /tmp/npa-lancedb \
  --port 8686 \
  --auth-mode none \
  --image cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-lancedb:0.30.3
```

Create a table from local JSON, JSONL, parquet, or a directory of parquet
files:

```bash
npa workbench lancedb create-table \
  --endpoint http://localhost:8686 \
  --table robot_embeddings \
  --input-path /tmp/robot_embeddings.json \
  --mode create
```

Query nearest neighbors:

```bash
npa workbench lancedb query \
  --endpoint http://localhost:8686 \
  --table robot_embeddings \
  --vector '[0.1, 0.2, 0.3, 0.4]' \
  --top-k 5
```

## Storage Model

For production OSS deployments, use an S3-compatible prefix:

```bash
npa workbench lancedb deploy \
  --runtime vm \
  --storage-path s3://my-bucket/lancedb/ \
  --port 8686 \
  --auth-mode token \
  --token-env LANCEDB_TOKEN
```

The container receives standard AWS-compatible variables:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_ENDPOINT_URL`
- `AWS_REGION`

The Workbench loader resolves these from the process environment and
`~/.npa/credentials.yaml` storage fields.

## Table Inputs

`create-table` accepts:

- Local `.parquet` files
- Local `.json` files containing a row list or `{ "rows": [...] }`
- Local `.jsonl` files
- Local directories containing parquet files
- `s3://` source paths for future server-side imports

Rows should include a vector column, defaulting to `vector`.

## Filtering And Projection

Scalar filtering is passed through to LanceDB:

```bash
npa workbench lancedb query \
  --endpoint http://localhost:8686 \
  --table robot_embeddings \
  --vector '[0.1, 0.2, 0.3, 0.4]' \
  --filter "split = 'val'" \
  --select id \
  --select episode_id
```

Use LanceDB scalar columns for dataset metadata such as `episode_id`, `task`,
`split`, `camera`, `failure_mode`, or artifact URIs.

## LeRobot Import

LeRobot datasets can be imported from their parquet layout:

```bash
npa workbench lancedb import-lerobot \
  --endpoint http://localhost:8686 \
  --dataset-path /datasets/pick-place \
  --table pick_place_embeddings \
  --vector-column vector \
  --id-column id
```

If a row does not already contain the vector column, the importer creates a
small numeric fallback vector from scalar values so the row remains queryable.
Production embedding extraction should write explicit vectors before import.

## Cloud Mode

Cloud mode is connection-only. It does not provision LanceDB Cloud or BYOC
resources:

```bash
export LANCEDB_API_KEY=...
npa workbench lancedb deploy \
  --runtime cloud \
  --endpoint https://your-lancedb-cloud-endpoint \
  --database robot-data \
  --cloud-region us-east-1
```

Enterprise provisioning, BYOC networking, and LanceDB-side account setup are
deferred to the partnership path.
